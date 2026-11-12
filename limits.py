"""Rate limits for a public demo running on one free API key.

The exposure here is not abuse, it is arithmetic. Groq's free tier allows 200,000
tokens a day; a question costs ~3,300 to answer and ~3,500 more to grade. Thirty
curious minutes exhausts the day, and the next person to open the link sees a dead
app. So the app stops before Groq does and says why.

State is in memory on purpose. There is one worker, and Render's free tier spins
the service down after 15 minutes idle, so counters reset. That is the right
failure direction for a demo: limits forgive rather than accumulate, and no
database exists to go wrong.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

# Per visitor. Generous enough to explore properly, tight enough that one person
# cannot spend everyone's day.
PER_HOUR = 12
PER_DAY = 40

# Below Groq's real 200,000 so the app refuses politely before the API refuses
# rudely, leaving headroom for the eval to still run.
DAILY_TOKENS = 140_000

# The box is 512 MB with one worker; two generations in flight is the ceiling
# before memory and latency both suffer.
MAX_CONCURRENT = 2

MAX_QUESTION_CHARS = 600


class Limiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._tokens: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()
        self.slots = threading.Semaphore(MAX_CONCURRENT)

    # ------------------------------------------------------------------ util
    @staticmethod
    def _trim(q: deque, window: float, now: float) -> None:
        while q and (q[0][0] if isinstance(q[0], tuple) else q[0]) <= now - window:
            q.popleft()

    def tokens_used(self) -> int:
        with self._lock:
            self._trim(self._tokens, 86_400, time.time())
            return sum(n for _, n in self._tokens)

    def spend(self, tokens: int) -> None:
        with self._lock:
            self._tokens.append((time.time(), tokens))

    # ------------------------------------------------------------------ check
    def check(self, who: str, estimated: int) -> str | None:
        """None if the request may proceed, else the reason to show the visitor."""
        now = time.time()
        with self._lock:
            self._trim(self._tokens, 86_400, now)
            if sum(n for _, n in self._tokens) + estimated > DAILY_TOKENS:
                return ("This demo shares one free API key and today's budget is "
                        "spent. It resets on a rolling 24-hour window — try again "
                        "in a few hours.")

            seen = self._hits[who]
            self._trim(seen, 86_400, now)
            if len(seen) >= PER_DAY:
                return ("You've reached the daily limit for one visitor (%d "
                        "questions). Retrieval still works; generation resumes "
                        "tomorrow." % PER_DAY)

            recent = sum(1 for t in seen if t > now - 3_600)
            if recent >= PER_HOUR:
                return ("You've reached the hourly limit (%d questions). This is a "
                        "demo on a free tier — try again shortly." % PER_HOUR)

            seen.append(now)
            return None


def client_key(request) -> str:
    """Render terminates TLS upstream, so the socket peer is the proxy. The first
    entry of X-Forwarded-For is the visitor."""
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def estimate_tokens(hits: list[dict], reserve: int = 1200) -> int:
    """Rough prompt cost. Legal English runs ~3 chars a token; `reserve` covers
    the system prompt and the reserved completion, which Groq charges whether or
    not it is used."""
    return sum(len(h["text"]) for h in hits) // 3 + reserve
