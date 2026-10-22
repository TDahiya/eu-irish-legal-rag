# Hybrid RAG over EU & Irish law, with automated evaluation

A retrieval-augmented generation pipeline over **EU technology regulation** and
**Irish/EU immigration law**, with an evaluation layer that measures retrieval
accuracy, answer faithfulness and hallucination rate — and feeds that measurement
back into generation as a control signal.

**Live demo:** _(URL after deploy)_ · Render's free tier sleeps after 15 minutes
idle, so the first request takes about a minute to wake.

> A research aid, not legal advice.

---

## What makes it different from a tutorial RAG

**1. Hybrid retrieval, because legal questions come in two incompatible shapes.**

| question | what answers it | what fails |
|---|---|---|
| "AI Act Article 6(2)", "Stamp 1G", "S.I. 548/2015" | BM25 — exact tokens | embeddings blur article numbers into near neighbours |
| "can my spouse work on a Critical Skills permit" | dense — meaning | BM25 shares no vocabulary with the statute that answers it |

BM25 comes from **SQLite FTS5**, which ships in the Python standard library with a
built-in `bm25()` ranking function — so lexical search costs zero dependencies and
zero RAM. Dense comes from **FAISS**. They are fused with **Reciprocal Rank Fusion**,
which uses ranks only, so the two incompatible score scales (negative BM25, cosine)
never have to be normalised against each other.

`eval/report.md` reports recall for BM25 alone, dense alone, and hybrid, **broken down
by question type** — the evidence that hybrid was worth it, rather than the claim.

**2. Abstention is a measured metric.**

Ten of the 68 evaluation questions have no answer in the corpus, and are built from
real traps:

> "What is the minimum salary threshold for an Irish EU Blue Card?"

There isn't one. Ireland has a Title V opt-out and is not bound by the Blue Card
Directive. A system that answers this confidently is hallucinating, and the only way
to know is to ask it. Every chunk carries an `ireland_bound` flag, and instruments
that do not bind Ireland are labelled inline in the prompt context.

**3. The evaluation layer is wired into generation, not bolted beside it.**

After generating, the agent scores its own answer's grounding; if it is poor, it
retries once with the accumulated context and keeps the retry only if it scores
better. A faithfulness measurement that changes the output is worth more than one
that decorates a dashboard.

---

## Architecture

```
                     question
                        │
      ┌─────────────────┴─────────────────┐
      │                                   │
SQLite FTS5  bm25()               FAISS  bge-small-384
  top 50 lexical                    top 50 dense
      │                                   │
      └────────► Reciprocal Rank Fusion ◄──┘
                   (k=60) top 8
                        │
                 openai/gpt-oss-120b  (Groq, Apache-2.0 open weights)
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
  grounded answer              grounding score
  with [n] citations                   │
         ▲                             │
         └──── retry once if poorly ───┘
                   grounded
```

In agentic mode the model holds `search(query, source_type)` as a tool and calls it
once per sub-question, up to five times — so a question spanning immigration and
technology law gets a separate retrieval for each, instead of one query embedding
landing between the two and retrieving neither.

### Why FAISS in production and ChromaDB only in the benchmark

FAISS is a *library*: an index file loaded into the process, no server, lowest
memory. ChromaDB is a *database*: it stores vectors, documents and metadata together,
with filtering built in — friendlier, and heavier at runtime. On Render's free 512 MB,
that difference decides it. `eval/bench_stores.py` indexes the identical vectors into
both and reports build time, query latency, disk and recall, so the choice is
measured rather than asserted. Results in `eval/report.md`.

### Why no hosted vector database, and why the index is a release asset

Qdrant Cloud's free cluster is the best of the free tiers, but it is **suspended after
one week idle and deleted after four**. A portfolio demo opened six weeks later would
be dead. So there is no vector service: the index is two files, built locally.

Where they live changed once, on measurement. At an estimated 10k chunks and ~25 MB
they went in the repository, which made the Render build a plain `git clone`. The real
corpus is 36k chunks — ~56 MB of FAISS and ~75 MB of SQLite — and *every rebuild would
add that again to git history permanently*. A portfolio repository nobody can clone
cheaply is worse than one extra build step, so they are published as a **GitHub
Release asset** instead, which doesn't count against clone size and allows 2 GB.

Because the repository is public the asset URL is public too, so this costs no second
secret: `GROQ_API_KEY` is still the only one. Rebuilding on Render is not an
alternative — embedding runs at ~10 chunks/s on a normal CPU, so an hour locally would
be many hours on 0.1 CPU.

---

## Corpus

Every source is declared in [`rag/sources.yaml`](rag/sources.yaml); no module
hardcodes a URL. All of it is openly licensed.

| source | what | licence |
|---|---|---|
| **EUR-Lex / Cellar** | AI Act, GDPR, DSA, DMA, Data Act, NIS2, Cyber Resilience Act, ePrivacy, DSM Copyright; Free Movement, Dublin III, Blue Card, Single Permit, Family Reunification, Schengen Borders Code, Qualification and Asylum Procedures Directives; Charter, TEU, TFEU | Decision 2011/833/EU |
| **CJEU** | Metock, Ruiz Zambrano, Chavez-Vilchez, Coman, Schrems I & II, Google Spain | Decision 2011/833/EU |
| **Irish Statute Book** | Immigration Acts 1999/2003/2004, International Protection Act 2015, Nationality & Citizenship Acts, Employment Permits Acts, Data Protection Act 2018, ECHR Act 2003, Equal Status and Employment Equality Acts | CC-BY 4.0 (Oireachtas PSI) |
| **Immigration Service Delivery** | visa routes, border entry, Common Travel Area, study, work, family reunification, naturalisation | Irish PSI, crawled at the 10s delay their `robots.txt` requests |
| **Project Gutenberg** | Mill, Locke, Hobbes, Plato, Aristotle, Kant, Montesquieu | public domain |

The philosophy books are tagged `source_type: philosophy` and **excluded by default
from retrieval**, so a question about a visa can never be answered out of Hobbes.
They are in the corpus to stress the chunker at book length and to answer questions
about justice as an idea.

### Two gotchas worth knowing

`gpt-oss` emits reasoning tokens before its answer, and they are charged against
`max_completion_tokens`. A JSON-mode call at the default effort can spend the whole
budget reasoning and return `max completion tokens reached before generating a valid
document` — the claim judge failed exactly this way. Fix: `reasoning_effort="low"`
and a budget of 2000. Measured: `low` reached the same verdicts as `medium` for a
third of the tokens, which also keeps 68 eval questions inside the free tier's
per-minute token cap. The reasoning text itself arrives in a separate `.reasoning`
field, so it never pollutes the answer.

### And one about EUR-Lex

`eur-lex.europa.eu` bot-gates every route with `HTTP 202` and an empty body. The
working endpoint is Cellar:

```
GET https://publications.europa.eu/resource/celex/32024R1689
Accept: application/xhtml+xml
Accept-Language: eng
```

Pre-2010 directives and most CJEU judgments have no XHTML manifestation and need
`Accept: text/html`; treaty CELEX ids contain a slash (`12016E/TXT`) that must be
percent-encoded. `Accept: */*` on the same URI returns RDF metadata.

### Chunking

On real structure, not fixed windows: EU acts split on `Article N`, Irish Acts on
their ELI-XML `<sect>` elements, books on chapter. A unit longer than the embedder's
window is split on sentence boundaries with overlap. Every chunk carries its citation
(`AI Act, Article 6`, `Immigration Act 2004, s. 4`), so an answer points at something
a human can check.

---

## Results

Corpus: **868 documents, 36,165 chunks**. Goldset: **70 questions**, 10 of them
unanswerable traps. Full output in [`eval/report.md`](eval/report.md).

### Retrieval

| | recall@5 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| BM25 only | 88.3% | 91.7% | **0.850** | **0.853** |
| Dense only | 85.0% | 86.7% | 0.711 | 0.729 |
| **Hybrid (RRF)** | **90.0%** | **93.3%** | 0.843 | 0.838 |

recall@10 by question type:

| type | n | BM25 | dense | hybrid |
|---|---|---|---|---|
| citation lookup | 16 | 100% | 81.2% | 100% |
| conceptual | 31 | 87.1% | 87.1% | 87.1% |
| multi-hop | 5 | 80% | 80% | **100%** |
| numeric | 8 | 100% | 100% | 100% |

**Hybrid's win is recall, and it is concentrated in multi-hop** — 80% for either
retriever alone, 100% fused. Two results cut against the usual story: dense adds
nothing on conceptual questions here, and dense alone is the weakest retriever on
every measure. On legal text the lexical side carries more than "embeddings beat
keywords" suggests.

**And fusion costs ranking precision.** BM25 alone has the better MRR (0.850 vs
0.843). It hurts 5 questions and helps 4, with the harm on citation lookups: *"What
right does Article 17 of the GDPR create?"* falls from MRR 1.00 to 0.20, because with
equal weights a document both retrievers rank mid-pack outscores one BM25 ranks first
(`1/76 + 1/61 > 1/61`). Weighting the lexical list higher is the obvious fix and
[the sweep](eval/rrf_sweep.md) says it does not work: 3:1 lifts MRR to 0.861 but drops
recall@10 to 91.7%. All eight top chunks are given to the generator, so a relevant
chunk being in the window beats its position inside it. Equal weights stay.

### FAISS vs ChromaDB

Same 10,000 vectors, embedded once, indexed into both:

| store | build | query p50 | query p95 | disk | recall@10 |
|---|---|---|---|---|---|
| **FAISS** | **0.36 s** | **0.80 ms** | **10.2 ms** | **15.4 MB** | 1.000 |
| ChromaDB | 9.38 s | 1.88 ms | 50.8 ms | 121.7 MB | 1.000 |

Recall is identical — Chroma's HNSW returned the same top-10 as exact search at this
scale — so the difference is pure overhead: **7.9x the disk** for the same vectors,
because Chroma also stores documents, metadata and its own SQLite alongside the graph.
On a 512 MB box that decides it. Reproduce with `python -m eval.bench_stores`.

### Answer quality

_Running; see [`eval/report.md`](eval/report.md)._

---

## Run it

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements-dev.txt

python -m rag.fetch --dry-run     # verify all 56 sources resolve
python -m rag.fetch               # download (~25 min; polite crawl delays)
python -m rag.chunk               # structure-aware chunking
python -m rag.index               # FAISS + SQLite FTS5

python -m eval.run_eval --retrieval-only    # no API key needed
export GROQ_API_KEY=...                     # free key: console.groq.com
python -m eval.run_eval                     # adds answer quality
python -m eval.bench_stores                 # FAISS vs Chroma

streamlit run app.py
```

Locally the key can live in `.streamlit/secrets.toml` (gitignored) instead of the
environment:

```toml
GROQ_API_KEY = "gsk_..."
```

### Tests

```bash
python -m pytest tests/
```

### Deploy

[`render.yaml`](render.yaml) describes the service. Set `GROQ_API_KEY` in the Render
dashboard; that is the only secret. The build installs dependencies, downloads the
index from the latest GitHub Release, and warms the ONNX model so the first request
isn't slow.

To publish a rebuilt index:

```bash
tar -czf index.tar.gz -C data/index chunks.db dense.faiss
gh release create index-v1 index.tar.gz --title "Prebuilt index"
```

---

## Constraints this was built under

Render's free tier is **512 MB RAM, no persistent disk, sleeps after 15 minutes**.
That ruled out torch, sentence-transformers, a cross-encoder reranker and any local
LLM, and it is why:

- embeddings run through **fastembed** (ONNX, 67 MB, no torch)
- lexical search is **SQLite FTS5** rather than an in-memory BM25 library
- generation is an **open-weight model over an API** rather than in-process
- the index is **built offline and downloaded at build time** rather than written at runtime

Measured: **376 MB peak RSS** through a full hybrid search plus answer scoring in a
bare process, and the Streamlit process idles at 315 MB against 298 MB bare, so the
real peak is roughly **390 MB against the 512 MB limit**. Scoring an answer adds only
~74 MB, because a query embeds one text and scoring embeds a handful — the
`batch * heads * seq_len^2` cost that makes bulk indexing expensive barely applies at
query time.

## Deliberately not built

A cross-encoder reranker (won't fit; RRF captures most of the gain), RAGAS (heavy,
LangChain-coupled, and the metrics are more defensible hand-rolled), multi-agent
orchestration (one corpus and one tool do not need a state machine), and conversational
memory. Each would be the next thing, in that order.
