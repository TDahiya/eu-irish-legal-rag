"""Automated evaluation. Writes eval/report.md.

    python -m eval.run_eval --retrieval-only   # no API key needed
    python -m eval.run_eval                    # adds answer quality (needs GROQ_API_KEY)
    python -m eval.run_eval --mode agentic     # grade the agent instead of one-shot

Retrieval relevance is judged at instrument level ("did it find the AI Act?") rather
than exact chunk id. Chunk ids move whenever the chunker changes; the instrument that
answers a question does not. It is the weaker claim, and it is the one that stays true.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import yaml

from rag.generate import ABSTAIN, abstained, answer
from rag.retrieve import LAW_TYPES, Retriever
from rag.score import judge, live_scores

ROOT = Path(__file__).resolve().parent.parent
GOLDSET = Path(__file__).with_name("goldset.yaml")
REPORT = Path(__file__).with_name("report.md")
RAW_RESULTS = Path(__file__).with_name("results.json")

MODES = ("bm25", "dense", "hybrid")
K = 10


# ------------------------------------------------------------------ retrieval metrics
def _relevant_ranks(hits: list[dict], expected: list[str]) -> list[int]:
    """1-based ranks of hits whose instrument is one the question expects."""
    want = {e.lower() for e in expected}
    return [i for i, h in enumerate(hits, 1)
            if any(w in h["instrument"].lower() for w in want)]


def recall_at(ranks: list[int], k: int) -> float:
    return 1.0 if any(r <= k for r in ranks) else 0.0


def mrr(ranks: list[int]) -> float:
    return 1.0 / min(ranks) if ranks else 0.0


def ndcg_at(ranks: list[int], k: int) -> float:
    """Binary-relevance nDCG@k. The ideal ranking puts every relevant hit first."""
    dcg = sum(1.0 / math.log2(r + 1) for r in ranks if r <= k)
    n_rel = min(len([r for r in ranks if r <= k]), k)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    return dcg / ideal if ideal else 0.0


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _fmt(x, pct=True):
    if x is None:
        return "n/a"
    return "%.1f%%" % (100 * x) if pct else "%.3f" % x


# ------------------------------------------------------------------------- the run
def validate_goldset(gold: list[dict], retriever: Retriever) -> list[str]:
    """Every expected instrument must exist in the corpus, or the metric is a lie
    about the retriever when it is really a typo in the goldset."""
    known = {r[0].lower() for r in
             retriever.con.execute("SELECT DISTINCT instrument FROM chunks")}
    problems = []
    for item in gold:
        for exp in item.get("expect_instrument", []):
            if not any(exp.lower() in k for k in known):
                problems.append("%r expects unknown instrument %r"
                                % (item["q"][:50], exp))
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--limit", type=int, help="first N questions only")
    ap.add_argument("--mode", default="oneshot", choices=("oneshot", "agentic"))
    ap.add_argument("--sleep", type=float, default=2.0,
                    help="seconds between API calls (Groq free tier is ~30/min)")
    args = ap.parse_args()

    gold = yaml.safe_load(GOLDSET.read_text(encoding="utf-8"))
    if args.limit:
        gold = gold[:args.limit]
    retriever = Retriever()

    problems = validate_goldset(gold, retriever)
    if problems:
        print("goldset does not match the corpus:")
        for p in problems:
            print("  ", p)
        return 1

    graded = [g for g in gold if not g.get("expect_abstain")]
    traps = [g for g in gold if g.get("expect_abstain")]
    print("%d questions (%d answerable, %d traps)" % (len(gold), len(graded), len(traps)))

    # ---------------- retrieval: every mode, every answerable question
    retrieval = {m: [] for m in MODES}
    for item in graded:
        for mode in MODES:
            hits = retriever.search(item["q"], k=K, mode=mode)
            ranks = _relevant_ranks(hits, item["expect_instrument"])
            retrieval[mode].append({
                "q": item["q"], "kind": item.get("kind", "other"),
                "recall@5": recall_at(ranks, 5), "recall@10": recall_at(ranks, 10),
                "mrr": mrr(ranks), "ndcg@10": ndcg_at(ranks, K),
                "found_by": [h["found_by"] for h in hits[:5]],
            })
    print("retrieval done")

    # ---------------- answers: generation, live scores, LLM judge
    answers = []
    if not args.retrieval_only:
        from groq import Groq
        client = Groq()
        search = _agentic_search(retriever, client) if args.mode == "agentic" else None

        for i, item in enumerate(gold, 1):
            if search:
                text, hits = search(item["q"])
            else:
                hits = retriever.search(item["q"], k=8)
                text = answer(item["q"], hits, client=client)
            live = live_scores(text, hits, retriever.model)
            time.sleep(args.sleep)
            verdict = judge(text, hits, client)
            time.sleep(args.sleep)
            answers.append({
                "q": item["q"], "kind": item.get("kind", "other"),
                "is_trap": bool(item.get("expect_abstain")),
                "abstained": abstained(text), "answer": text,
                "citations": [h["citation"] for h in hits],
                **{k: v for k, v in live.items() if k != "per_sentence"},
                "faithfulness": verdict.get("faithfulness"),
                "hallucination_rate": verdict.get("hallucination_rate"),
                "n_claims": verdict.get("n_claims"),
            })
            print("  answered %d/%d" % (i, len(gold)), end="\r", flush=True)
        print()

    results = {"retrieval": retrieval, "answers": answers, "mode": args.mode}
    RAW_RESULTS.write_text(json.dumps(results, indent=1, ensure_ascii=False),
                           encoding="utf-8")
    REPORT.write_text(render(results, gold), encoding="utf-8")
    print("wrote %s and %s" % (REPORT.name, RAW_RESULTS.name))

    # the gate from the plan: hybrid must beat both single retrievers
    h = _mean([r["recall@10"] for r in retrieval["hybrid"]])
    b = _mean([r["recall@10"] for r in retrieval["bm25"]])
    d = _mean([r["recall@10"] for r in retrieval["dense"]])
    print("recall@10  bm25 %s  dense %s  hybrid %s" % (_fmt(b), _fmt(d), _fmt(h)))
    if h < max(b, d):
        print("GATE FAILED: hybrid does not beat the better single retriever")
        return 1
    return 0


def _agentic_search(retriever, client):
    from rag.agent import run as agent_run
    return lambda q: agent_run(q, retriever, client)


# ------------------------------------------------------------------------- report
def render(results: dict, gold: list[dict]) -> str:
    retrieval, answers = results["retrieval"], results["answers"]
    L = ["# Evaluation report",
         "",
         "Generated by `python -m eval.run_eval`. Relevance is judged at instrument",
         "level: a hit counts if it comes from an instrument the question expects.",
         ""]

    # --- retrieval, three ways
    L += ["## Retrieval: hybrid vs each retriever alone", "",
          "| retriever | recall@5 | recall@10 | MRR | nDCG@10 |",
          "|---|---|---|---|---|"]
    for mode in MODES:
        rows = retrieval[mode]
        L.append("| %s | %s | %s | %s | %s |" % (
            mode, _fmt(_mean([r["recall@5"] for r in rows])),
            _fmt(_mean([r["recall@10"] for r in rows])),
            _fmt(_mean([r["mrr"] for r in rows]), pct=False),
            _fmt(_mean([r["ndcg@10"] for r in rows]), pct=False)))

    # --- by query kind: the point of the whole exercise
    kinds = sorted({r["kind"] for r in retrieval["hybrid"]})
    L += ["", "### recall@10 by question type", "",
          "| question type | n | " + " | ".join(MODES) + " |",
          "|---|---|" + "---|" * len(MODES)]
    for kind in kinds:
        n = len([r for r in retrieval["hybrid"] if r["kind"] == kind])
        cells = [_fmt(_mean([r["recall@10"] for r in retrieval[m]
                            if r["kind"] == kind])) for m in MODES]
        L.append("| %s | %d | %s |" % (kind, n, " | ".join(cells)))

    # --- which retriever actually contributed
    contrib: dict[str, int] = {}
    for r in retrieval["hybrid"]:
        for f in r["found_by"]:
            contrib[f] = contrib.get(f, 0) + 1
    total = sum(contrib.values()) or 1
    L += ["", "### Where the top-5 hybrid results came from", ""]
    for key in ("bm25", "dense", "both"):
        L.append("- **%s**: %.0f%%" % (key, 100 * contrib.get(key, 0) / total))

    if not answers:
        L += ["", "_Answer quality not measured in this run "
              "(`--retrieval-only`)._", ""]
        return "\n".join(L) + "\n"

    # --- answer quality
    real = [a for a in answers if not a["is_trap"]]
    traps = [a for a in answers if a["is_trap"]]
    L += ["", "## Answer quality (%s)" % results["mode"], "",
          "| metric | value |", "|---|---|",
          "| faithfulness (LLM-judged) | %s |"
          % _fmt(_mean([a["faithfulness"] for a in real])),
          "| hallucination rate | %s |"
          % _fmt(_mean([a["hallucination_rate"] for a in real])),
          "| grounding score (free proxy) | %s |"
          % _fmt(_mean([a["grounding"] for a in real]), pct=False),
          "| citations pointing at a real source | %s |"
          % _fmt(_mean([a["citation_valid"] for a in real])),
          "| cited the best-supporting source | %s |"
          % _fmt(_mean([a["citation_apt"] for a in real])),
          "| abstained on a trap question | %s (%d/%d) |" % (
              _fmt(_mean([float(a["abstained"]) for a in traps])),
              sum(a["abstained"] for a in traps), len(traps)),
          "| abstained on an answerable question | %s |"
          % _fmt(_mean([float(a["abstained"]) for a in real]))]

    # --- does the free proxy track the expensive judge?
    pairs = [(a["grounding"], a["faithfulness"]) for a in real
             if a["grounding"] is not None and a["faithfulness"] is not None]
    if len(pairs) > 2:
        gs = [p[0] for p in pairs]
        fs = [p[1] for p in pairs]
        mg, mf = sum(gs) / len(gs), sum(fs) / len(fs)
        cov = sum((g - mg) * (f - mf) for g, f in pairs)
        sg = math.sqrt(sum((g - mg) ** 2 for g in gs))
        sf = math.sqrt(sum((f - mf) ** 2 for f in fs))
        r = cov / (sg * sf) if sg and sf else 0.0
        L += ["", "### Is the free grounding score a usable proxy?", "",
              "Pearson r between the embedding grounding score and LLM-judged",
              "faithfulness over %d answers: **%.2f**." % (len(pairs), r),
              "That is what justifies showing the cheap score on every answer in the",
              "UI and spending a judge call only on demand."]

    # --- the trap questions, individually, because this is the headline
    L += ["", "### Trap questions", "",
          "Questions with no answer in the corpus. Abstaining is the correct behaviour.",
          "", "| question | abstained | why it is a trap |", "|---|---|---|"]
    whys = {g["q"]: g.get("why", "") for g in gold}
    for a in traps:
        L.append("| %s | %s | %s |" % (a["q"][:70], "yes" if a["abstained"] else "**NO**",
                                       whys.get(a["q"], "")[:80]))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
