"""HTTP layer for the hybrid legal RAG.

Everything in rag/ is untouched: this only replaces the UI. The browser holds
the page state, which is what makes "grade this answer" possible at all --
under Streamlit every widget click re-ran the script and discarded the answer.

    uvicorn server:app --port 8000
"""
from __future__ import annotations

import json
import os
import queue
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"

# Render supplies the key as a real env var; locally it sits in .env, which is
# three lines of parsing rather than a dependency.
_env = ROOT / ".env"
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        key, _, value = line.strip().partition("=")
        if key and not key.startswith("#"):
            os.environ.setdefault(key, value)

from limits import (DAILY_TOKENS, MAX_QUESTION_CHARS, Limiter,
                    client_key, estimate_tokens)

STATE: dict = {}
LIMITER = Limiter()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Loading the index costs ~300 MB and several seconds; do it once at boot
    # rather than per request.
    from rag.retrieve import Retriever
    STATE["retriever"] = Retriever()
    yield
    STATE.clear()


app = FastAPI(title="EU & Irish law RAG", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    mode: str = "agentic"                      # agentic | oneshot
    fusion: str = "hybrid"                     # hybrid | bm25 | dense
    source_types: list[str] = Field(default_factory=list)
    k: int = 8


class JudgeRequest(BaseModel):
    answer: str
    hit_ids: list[int]


def _client():
    from rag.generate import client
    return client()


def _shape(hits: list[dict]) -> list[dict]:
    """Only what the page renders, in citation order."""
    return [{
        "n": i,
        "id": h["id"],
        "citation": h["citation"],
        "instrument": h["instrument"],
        "source_type": h["source_type"],
        "text": h["text"],
        "url": h["url"],
        "note": h.get("note") or "",
        "ireland_bound": h.get("ireland_bound"),
        "bm25_rank": h.get("bm25_rank"),
        "dense_rank": h.get("dense_rank"),
        "found_by": h.get("found_by", "agent"),
    } for i, h in enumerate(hits, 1)]


@app.get("/api/meta")
def meta() -> dict:
    from rag.generate import MODEL
    from rag.index import MODEL as EMBED_MODEL
    from rag.retrieve import LAW_TYPES
    r = STATE["retriever"]
    rows = r.con.execute(
        "SELECT source_type, count(*) FROM chunks GROUP BY source_type").fetchall()
    return {
        "chunks": r.con.execute("SELECT count(*) FROM chunks").fetchone()[0],
        "instruments": r.con.execute(
            "SELECT count(DISTINCT instrument) FROM chunks").fetchone()[0],
        "by_type": {row[0]: row[1] for row in rows},
        "law_types": list(LAW_TYPES),
        "model": MODEL,
        "embed_model": EMBED_MODEL,
        "has_key": bool(os.environ.get("GROQ_API_KEY")),
        "tokens_used_today": LIMITER.tokens_used(),
        "daily_token_budget": DAILY_TOKENS,
    }


@app.post("/api/ask")
def ask(req: AskRequest, request: Request):
    from rag.generate import abstained, answer
    from rag.retrieve import LAW_TYPES
    from rag.score import live_scores

    question = req.question.strip()
    if not question:
        return JSONResponse({"detail": "Ask a question first."}, status_code=400)
    if len(question) > MAX_QUESTION_CHARS:
        return JSONResponse(
            {"detail": "That question is %d characters; the limit is %d."
                       % (len(question), MAX_QUESTION_CHARS)}, status_code=400)

    r = STATE["retriever"]
    types = tuple(req.source_types) or LAW_TYPES
    trace: list = []

    # Retrieval is free and local, so it runs before the budget check: even a
    # rate-limited visitor gets to see what the corpus holds.
    preview = r.search(question, k=req.k, mode=req.fusion, source_types=types)
    # agentic makes several calls, so reserve for several
    reserve = estimate_tokens(preview) * (3 if req.mode == "agentic" else 1)

    denied = LIMITER.check(client_key(request), reserve)
    if denied:
        return JSONResponse(
            {"detail": denied, "hits": _shape(preview)}, status_code=429)

    if not LIMITER.slots.acquire(blocking=False):
        return JSONResponse(
            {"detail": "Two answers are already being generated. This runs on one "
                       "free instance — try again in a few seconds."},
            status_code=503)

    # Answering takes 10-25s in agentic mode. Returning one response at the end
    # of that is a blank wait with nothing to read, so the steps are streamed as
    # newline-delimited JSON while they happen.
    return StreamingResponse(_stream(req, question, types, reserve),
                             media_type="application/x-ndjson")


class _Emitter(list):
    """Behaves as the list rag.agent already appends its trace to, and publishes
    each step as it lands. Keeps the streaming concern out of rag/."""

    def __init__(self, sink: queue.Queue):
        super().__init__()
        self._sink = sink

    def append(self, item) -> None:
        super().append(item)
        self._sink.put({"step": "search", **item})


def _stream(req, question, types, reserve):
    sink: queue.Queue = queue.Queue()
    trace = _Emitter(sink)

    def work():
        try:
            sink.put({"step": "retrieving"})
            sink.put({"done": _answer(req, question, types, trace, reserve)})
        except Exception as exc:
            sink.put({"error": "%s: %s" % (type(exc).__name__, str(exc)[:200])})
        finally:
            LIMITER.slots.release()
            sink.put(None)

    threading.Thread(target=work, daemon=True).start()
    while True:
        item = sink.get()
        if item is None:
            return
        yield json.dumps(item, ensure_ascii=False) + "\n"


def _emit(trace, event) -> None:
    sink = getattr(trace, "_sink", None)
    if sink is not None:
        sink.put(event)


def _answer(req, question, types, trace, reserve) -> dict:
    from rag.generate import abstained, answer
    from rag.score import live_scores
    r = STATE["retriever"]

    if req.mode == "agentic":
        from rag.agent import run as agent_run
        text, hits = agent_run(question, r, _client(), k=req.k,
                               trace=trace, source_types=types)
    else:
        hits = r.search(question, k=req.k, mode=req.fusion, source_types=types)
        _emit(trace, {"step": "writing", "n": len(hits)})
        text = answer(question, hits, client=_client())

    LIMITER.spend(reserve)
    _emit(trace, {"step": "scoring"})
    scores = live_scores(text, hits, r.model)
    return {
        "answer": text,
        "abstained": abstained(text),
        "hits": _shape(hits),
        "scores": scores,
        "trace": trace,
    }


@app.post("/api/judge")
def judge_answer(req: JudgeRequest, request: Request):
    from rag.score import judge
    r = STATE["retriever"]
    if not req.hit_ids:
        return {"faithfulness": None, "claims": []}
    # Refetch by id rather than trusting the client's copy of the source text.
    sql = "SELECT * FROM chunks WHERE id IN (%s)" % ",".join("?" * len(req.hit_ids))
    rows = {row["id"]: dict(row) for row in r.con.execute(sql, req.hit_ids)}
    hits = [rows[i] for i in req.hit_ids if i in rows]

    reserve = estimate_tokens(hits, 1200 + len(req.answer) // 3)
    denied = LIMITER.check(client_key(request), reserve)
    if denied:
        return JSONResponse({"detail": denied}, status_code=429)
    if not LIMITER.slots.acquire(blocking=False):
        return JSONResponse({"detail": "Busy — try again in a few seconds."},
                            status_code=503)
    try:
        verdict = judge(req.answer, hits, _client())
        LIMITER.spend(reserve)
        return verdict
    finally:
        LIMITER.slots.release()


app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": "retriever" in STATE}
