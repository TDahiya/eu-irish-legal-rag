# Hybrid RAG over EU & Irish law, with automated evaluation

A retrieval-augmented generation pipeline over **EU technology regulation** and
**Irish/EU immigration law**, with an evaluation layer that measures retrieval
accuracy, answer faithfulness and hallucination rate — and that has already been
used to delete one of this project's own features for not working.

**Live demo:** [eu-irish-law-rag.onrender.com](https://eu-irish-law-rag.onrender.com) · Render's free tier sleeps after 15 minutes
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

**3. A feature was measured, found inert, and deleted.**

The agent used to retry any answer whose grounding score fell below 0.45 — the idea
being that a quality measurement should change the output rather than just be
displayed. Measuring it killed it:

- over 36 evaluated answers, grounding ran **0.68 to 0.96**, so the retry **never
  fired once**;
- splitting those answers at the median, the **lower**-grounding half scored
  **98.2%** faithfulness against **91.7%** for the upper half.

The signal is *inversely* related to the thing it was gating on. An answer that
paraphrases a statute in plain English scores lower on similarity than one that
copies its wording, while being just as accurate — so "low grounding" often means
"explained it well". No threshold fixes that, so the retry is gone rather than
retuned, and nothing replaced it, because there is no evidence yet that a
replacement would help.

What survives is the part that *is* measured: the multi-search loop, which takes
multi-hop recall from 80% to 100%.

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
  grounded answer              similarity score + on-demand
  with [n] citations           claim-by-claim LLM judge
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

All 70 questions, 57 of them judged claim-by-claim (an abstention has no claims to
grade).

| metric | value |
|---|---|
| faithfulness (LLM-judged) | **95.2%** |
| hallucination rate | **4.8%** |
| citations pointing at a real source | **100%** |
| cited the best-supporting source | 99.5% |
| refused a trap question outright | 8 / 10 |
| refused an answerable question | 8.3% (5 / 60) |

**On the two traps it did not refuse — it was right, and the metric was wrong.**

> *"What are Ireland's Schengen visa requirements?"*
> → *"Ireland does not participate in the Schengen acquis and therefore is not bound
> by Schengen visa rules; it issues its own visas independently."* — **100% faithful**

Rebutting the false premise from the sources is a *better* answer than refusing, and
the `abstained` column counts only the literal refusal sentinel, so it scored that as
a failure. The report now prints faithfulness beside every trap so the two cases can
be told apart. Nothing invented an Irish Blue Card salary or a Schengen visa rule.

The 8.3% figure is the cost of that caution: five answerable questions were refused
when the corpus did support an answer. Tuning abstention down would raise answer
coverage and raise hallucination with it; this is where that dial currently sits.

**The embedding grounding score is still not a proxy for faithfulness** (Pearson
r = **0.15** over 55 answers). It measures whether an answer *resembles* the sources,
not whether they support it — an answer paraphrasing a statute in plain English
scores lower than one copying its wording while being equally true. Faithfulness also
barely varies here, so the sample cannot settle it either way. The UI labels the
score as the similarity signal it is, and the judge is the measurement of record.
This is the finding that got a feature deleted — see above.

### A measurement bug worth recording

The first full run reported citation validity on only 39 of 57 answers. `gpt-oss`
emits full-width markers — `【6】`, sometimes `【7†L13-L16】` — in about 30% of
answers regardless of what the prompt asks for, and those were invisible to the
parser, the scorer and the UI alike. Answers are now normalised to `[n]` on the way
out, which lifted coverage to 56/57 without changing the underlying quality. Because
retrieval is deterministic and embedding is local, `--rescore` recomputed every local
metric over the stored answers for no API calls at all.

---

## Interface

The UI is a hand-built page served by FastAPI. It replaced a Streamlit app, for one
reason that matters and one that doesn't.

The one that matters: **Streamlit re-runs the whole script on every widget click**,
so the "grade this answer" button discarded the answer it was meant to grade — the
result lived in local variables inside an `if submitted:` block, and on the re-run
that block simply didn't execute. The browser holding its own state makes the bug
impossible rather than fixed.

The one that doesn't: design control. Every colour has exactly one job, and two of
them encode real data —

| colour | job |
|---|---|
| `#004777` ink | structure, headings, "both retrievers agreed" |
| `#FF7700` lexical | BM25 found it — the highlighter on a photocopied statute |
| `#00AFB5` semantic | embeddings found it |
| `#A30000` refusal | abstention only, never decorative |
| `#EFD28D` paper | surface warmth in light mode, hairlines in dark |

The signature is the **provenance spine**: a 4px bar down each source, orange above
and teal below, split by which retriever actually found it. One glance says whether a
question was answered by exact tokens or by meaning — the hybrid thesis, visible
without reading a number.

Type is Bricolage Grotesque on a variable optical-size axis, so one family covers the
44px display and the 14px metadata and earns its place; JetBrains Mono appears only
where numerals must align. Spacing is 8px multiples throughout, depth is borders and
tonal shift with no shadows anywhere (a document has edges, not elevation), and the
background grid is the Official Journal's column rule at 4% opacity, radially masked
so it fades instead of tiling.

## Run it

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements-dev.txt

python -m rag.fetch --dry-run     # verify every source resolves
python -m rag.fetch               # download (polite crawl delays; ~40 min)
python -m rag.chunk               # structure-aware chunking
python -m rag.index               # FAISS + SQLite FTS5 (~1 hour to embed)

python -m eval.run_eval --retrieval-only    # no API key needed
export GROQ_API_KEY=...                     # free key: console.groq.com
python -m eval.run_eval                     # adds answer quality; resumable
python -m eval.sweep_rrf                    # fusion weight sweep
python -m eval.bench_stores                 # FAISS vs Chroma

uvicorn server:app --port 8000
```

The key can live in a gitignored `.env` (`GROQ_API_KEY=gsk_...`) instead of the
environment; `server.py` reads it if present.

### Tests

```bash
python -m pytest tests/
```

### Deploy

[`render.yaml`](render.yaml) describes the service. `GROQ_API_KEY` in the Render
dashboard is the only secret — the index is a public release asset. The build
installs dependencies, downloads the index and warms the ONNX model so the first
request isn't slow. One uvicorn worker on purpose: the index and model are ~300 MB
resident and a second worker would load its own copy straight past the 512 MB limit.

To publish a rebuilt index:

```bash
tar -czf index.tar.gz -C data/index chunks.db dense.faiss
gh release create index-v2 index.tar.gz --title "Prebuilt index"
```

## Constraints this was built under

Render's free tier is **512 MB RAM, no persistent disk, sleeps after 15 minutes**.
That ruled out torch, sentence-transformers, a cross-encoder reranker and any local
LLM, and it is why:

- embeddings run through **fastembed** (ONNX, 67 MB, no torch)
- lexical search is **SQLite FTS5** rather than an in-memory BM25 library
- generation is an **open-weight model over an API** rather than in-process
- the index is **built offline and downloaded at build time**

Measured, both stacks, through a full hybrid search plus generation and scoring:
**380 MB on FastAPI against ~390 MB on Streamlit.** Essentially unchanged, and that
was not the expectation — dropping a ~90 MB framework looked like it should buy real
headroom. It doesn't, because the FAISS index and the ONNX model account for ~300 MB
between them and the web framework is noise beside them. The reason to leave
Streamlit was the state bug, not the memory.

## Deliberately not built

A cross-encoder reranker (won't fit; RRF captures most of the gain), RAGAS (heavy,
LangChain-coupled, and the metrics are more defensible hand-rolled), multi-agent
orchestration (one corpus and one tool do not need a state machine), conversational
memory, and — after measuring it — the self-correction retry described above.
