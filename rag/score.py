"""Answer quality scores.

Two tiers, deliberately:

  Embedding-based (grounding, citation validity, citation aptness) cost one embedding
  call on text already in memory. They are free and instant, so the UI can show a
  number on every answer. They measure *similarity* to the sources, which is a proxy
  for support, not proof of it.

  LLM-judged (faithfulness, hallucination rate) decompose the answer into claims and
  check each against the context. This is the real measurement, and it costs an API
  call, so it runs offline in eval/ and on demand in the UI.

eval/run_eval.py reports the correlation between the two, which is what justifies
showing the cheap one live.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

from rag.generate import ABSTAIN, MAX_TOKENS, REASONING_EFFORT

JUDGE_MODEL = os.environ.get("GROQ_JUDGE_MODEL", "openai/gpt-oss-120b")

JUDGE_SYSTEM = """You are a strict grader checking whether an answer is supported by \
its sources.

Split the answer into atomic factual claims. Ignore hedges, restatements of the
question, and sentences that only say what the sources do not cover. For each claim,
decide against the numbered sources ONLY:

  "supported"    - a source states it, or it follows directly from one
  "unsupported"  - no source states it; it may still be true in the world
  "contradicted" - a source states something incompatible with it

Judge only against the sources given. Your own knowledge of the law is not evidence.

Reply with JSON exactly: {"claims": [{"claim": "...", "verdict": "...", "source": n}]}
where n is the supporting source number, or null."""


def sentences(text: str) -> list[str]:
    """Answer sentences with the [n] markers stripped, so citations don't inflate
    similarity against a chunk that happens to contain digits."""
    clean = re.sub(r"\[\d+\]", "", text)
    parts = re.split(r"(?<=[.!?])\s+", clean)
    return [p.strip() for p in parts if len(p.strip()) > 25]


def _cos_matrix(model, answer_sents: list[str], hits: list[dict]) -> np.ndarray:
    """(n_sentences, n_hits) cosine similarities."""
    sent_vecs = np.array(list(model.embed(answer_sents)), dtype="float32")
    hit_vecs = np.array(list(model.embed([h["text"] for h in hits])), dtype="float32")
    sent_vecs /= np.linalg.norm(sent_vecs, axis=1, keepdims=True)
    hit_vecs /= np.linalg.norm(hit_vecs, axis=1, keepdims=True)
    return sent_vecs @ hit_vecs.T


def live_scores(answer_text: str, hits: list[dict], model) -> dict:
    """The free scores, shown on every answer in the UI.

    grounding        mean over sentences of the best similarity to any source
    citation_valid   fraction of [n] markers that point at a source that exists
    citation_apt     how often the source cited is the one that best supports the
                     sentence (1.0 = always)
    """
    if answer_text.strip().startswith(ABSTAIN[:40]):
        return {"grounding": None, "citation_valid": None, "citation_apt": None,
                "abstained": True, "per_sentence": []}

    sents = sentences(answer_text)
    if not sents or not hits:
        return {"grounding": None, "citation_valid": None, "citation_apt": None,
                "abstained": False, "per_sentence": []}

    sims = _cos_matrix(model, sents, hits)
    best = sims.max(axis=1)

    # citations are per-sentence, so re-find them on the uncleaned sentences
    raw_sents = [s for s in re.split(r"(?<=[.!?])\s+", answer_text)
                 if len(re.sub(r"\[\d+\]", "", s).strip()) > 25]
    valid, total, apt = 0, 0, []
    per_sentence = []
    for i, raw in enumerate(raw_sents[:len(sents)]):
        marks = [int(n) for n in re.findall(r"\[(\d+)\]", raw)]
        in_range = [n for n in marks if 1 <= n <= len(hits)]
        valid += len(in_range)
        total += len(marks)
        if in_range and best[i] > 0:
            apt.append(max(sims[i][n - 1] for n in in_range) / best[i])
        per_sentence.append({
            "sentence": sents[i][:160],
            "grounding": round(float(best[i]), 3),
            "cites": marks,
            "best_source": int(sims[i].argmax()) + 1,
        })

    return {
        "grounding": round(float(best.mean()), 3),
        "citation_valid": round(valid / total, 3) if total else None,
        "citation_apt": round(float(np.mean(apt)), 3) if apt else None,
        "abstained": False,
        "per_sentence": per_sentence,
    }


def judge(answer_text: str, hits: list[dict], client, model: str = JUDGE_MODEL) -> dict:
    """LLM-judged faithfulness and hallucination rate. Costs one API call."""
    if answer_text.strip().startswith(ABSTAIN[:40]):
        return {"faithfulness": None, "hallucination_rate": None,
                "claims": [], "abstained": True}

    from rag.generate import format_context
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        # Without a generous budget and low reasoning effort, JSON mode fails with
        # "max completion tokens reached before generating a valid document".
        max_completion_tokens=MAX_TOKENS,
        reasoning_effort=REASONING_EFFORT,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": "Sources:\n\n%s\n\nAnswer to grade:\n%s"
                                        % (format_context(hits), answer_text)},
        ],
    )
    try:
        claims = json.loads(resp.choices[0].message.content).get("claims", [])
    except (json.JSONDecodeError, AttributeError):
        return {"faithfulness": None, "hallucination_rate": None,
                "claims": [], "error": "judge returned unparseable JSON"}

    claims = [c for c in claims if c.get("verdict") in
              ("supported", "unsupported", "contradicted")]
    if not claims:
        return {"faithfulness": None, "hallucination_rate": None, "claims": []}

    supported = sum(c["verdict"] == "supported" for c in claims)
    return {
        "faithfulness": round(supported / len(claims), 3),
        "hallucination_rate": round(1 - supported / len(claims), 3),
        "contradicted": sum(c["verdict"] == "contradicted" for c in claims),
        "n_claims": len(claims),
        "claims": claims,
        "abstained": False,
    }
