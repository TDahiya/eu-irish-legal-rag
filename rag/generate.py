"""Grounded answer generation via Groq.

The model is reachable only as an id that Groq rotates often - llama-3.3-70b-versatile
was withdrawn from the free tier in August 2026 and fourteen other ids are already
dead - so it is an env var with a current default rather than a constant.
openai/gpt-oss-120b is Apache-2.0 open-weight, which is the point: the generator is an
open-source model, just not one that would fit in Render's 512 MB.
"""
from __future__ import annotations

import os
import re

from groq import Groq

MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# gpt-oss emits reasoning tokens before its answer, and they are charged against
# max_completion_tokens. At the default effort a JSON-mode call can exhaust the budget
# before producing any JSON at all ("max completion tokens reached before generating a
# valid document"), so the budget is generous and the effort is low. Measured on the
# claim judge: "low" reached the same verdicts as "medium" for a third of the tokens,
# which also keeps 68 eval questions inside the free tier's per-minute token limit.
REASONING_EFFORT = os.environ.get("GROQ_REASONING_EFFORT", "low")
# Reserved completion tokens are charged against the quota whether or not they are
# used, and the free tier's binding limit is 200,000 tokens per DAY. Grounded answers
# here run to a few sentences, so 2000 was buying nothing and costing a fifth of a
# question's budget. 900 leaves room for low-effort reasoning plus the answer.
MAX_TOKENS = int(os.environ.get("GROQ_MAX_TOKENS", "900"))

# A fixed sentinel, so eval/run_eval.py can detect a refusal without an LLM call.
ABSTAIN = "The corpus does not contain an answer to this."

SYSTEM = """You answer questions about EU and Irish law using ONLY the numbered \
sources provided. You are a research aid, not a solicitor, and you never give legal \
advice.

Rules, in order of importance:

1. Ground everything. Every factual sentence ends with the source it came from, as
   [1] or [2][5]. A sentence you cannot cite is a sentence you must not write.
2. Abstain rather than guess. If the sources do not answer the question, reply with
   exactly this sentence first: "%s" Then, in one sentence, say what the sources do
   cover that is closest to the question.
3. Respect territorial scope. Some EU instruments carry a note saying Ireland is not
   bound by them (Ireland has a Title V opt-out, and is outside Schengen). Never
   present the rules of such an instrument as law in Ireland. Say that it does not
   apply here and, if the sources show it, what applies instead.
4. Be exact. Quote figures, deadlines, stamp numbers and article numbers verbatim
   from the sources. Do not round, convert or infer them.
5. Be brief. Answer the question asked in a few sentences. No preamble, no summary of
   your own reasoning, no restating the question.""" % ABSTAIN


def client() -> Groq:
    """A client that waits out rate limits instead of failing the caller.

    The free tier caps tokens per minute (8,000), not just requests, and one
    question here costs ~5,600 tokens because the retrieved context is sent twice:
    once to generate and once to grade. The SDK honours the server's Retry-After,
    so the fix is simply to let it retry more than twice.
    """
    return Groq(max_retries=8)


def format_context(hits: list[dict]) -> str:
    """Numbered source blocks. The note field is what defends against trap questions:
    it tells the model, inline, that an instrument does not bind Ireland."""
    blocks = []
    for i, h in enumerate(hits, 1):
        head = "[%d] %s" % (i, h["citation"])
        if h.get("note"):
            head += "  (NOTE: %s)" % h["note"]
        elif h.get("ireland_bound") == 0:
            head += "  (NOTE: Ireland is not bound by this instrument.)"
        blocks.append("%s\n%s" % (head, h["text"]))
    return "\n\n".join(blocks)


def answer(question: str, hits: list[dict], model: str = MODEL,
           client: Groq | None = None, stream: bool = False):
    """Generate a grounded answer. Returns the text, or a token iterator if stream."""
    if not hits:
        return ABSTAIN + " No sources matched the question."

    client = client or Groq()
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Sources:\n\n%s\n\nQuestion: %s"
                                    % (format_context(hits), question)},
    ]
    kwargs = dict(model=model, messages=messages, temperature=0,
                  max_completion_tokens=MAX_TOKENS,
                  reasoning_effort=REASONING_EFFORT)

    if stream:
        def tokens():
            for chunk in client.chat.completions.create(stream=True, **kwargs):
                piece = chunk.choices[0].delta.content
                if piece:
                    yield piece
        return tokens()

    return client.chat.completions.create(**kwargs).choices[0].message.content


def cited_indexes(text: str) -> set[int]:
    """The [n] markers the answer actually used."""
    return {int(n) for n in re.findall(r"\[(\d+)\]", text)}


def abstained(text: str) -> bool:
    return text.strip().startswith(ABSTAIN[:40])
