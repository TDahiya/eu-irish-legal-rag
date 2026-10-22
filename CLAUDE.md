# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

# Project: hybrid legal RAG (EU + Ireland)

Hybrid retrieval (SQLite FTS5 bm25 + FAISS dense, fused with RRF) over EU and Irish
law — tech/AI regulation and immigration/work-visa — with an automated evaluation
layer. Streamlit UI, deployed on Render free tier.

## Hard constraints

- **512 MB RAM on Render free.** No torch, no sentence-transformers, no cross-encoder
  reranker, no local LLM. Embeddings via `fastembed` (ONNX). Measure RSS before
  deploying; budget is < 450 MB peak.
- **No persistent disk, no external vector DB.** The FAISS index and SQLite DB are built
  locally and fetched from a GitHub Release at Render build time.
- **Lexical search is stdlib.** `sqlite3` ships FTS5 with a built-in `bm25()` ranking
  function. Do not add a BM25 library.
- `data/` is generated and gitignored. Everything in it must be reproducible by
  `python -m rag.fetch && python -m rag.index`.

## Corpus rules

- Sources are declared in `rag/sources.yaml` only — never hardcode a URL in a module.
- EU law comes from the **Cellar** endpoint, not `eur-lex.europa.eu` (which bot-gates
  with HTTP 202 and an empty body):
  `GET https://publications.europa.eu/resource/celex/{CELEX}` with
  `Accept: application/xhtml+xml` and `Accept-Language: eng`.
- Respect `Crawl-delay` per host. irishimmigration.ie is 10s.
- Only openly licensed sources. EU: Decision 2011/833/EU. Irish Statute Book: CC-BY 4.0.
  Books: public domain only. No copyrighted legal textbooks.
- `source_type` tags every chunk. `philosophy` must never be retrieved for a question
  about current law.

## Answer rules

- Answers are grounded only in retrieved context, cite `[n]`, and abstain when the
  corpus does not support an answer. Abstention is a measured metric, not a failure.
- This is not legal advice and the UI says so.
