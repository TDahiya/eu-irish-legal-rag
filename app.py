"""Streamlit UI: ask a question, see the answer and how well grounded it is.

The point of the side panel is to make the hybrid retrieval visible rather than
claimed: every source shows its BM25 rank, its dense rank, and which retriever
actually surfaced it.
"""
from __future__ import annotations

import os

import streamlit as st

from rag.generate import MODEL, abstained, answer
from rag.index import DB, FAISS_PATH
from rag.retrieve import LAW_TYPES, Retriever
from rag.score import judge, live_scores

TYPE_LABELS = {
    "eu_law": "EU regulations & directives",
    "ie_legislation": "Acts of the Oireachtas",
    "ie_guidance": "Irish immigration guidance",
    "case_law": "CJEU case law",
    "philosophy": "Legal philosophy (public domain books)",
}

EXAMPLES = [
    "Can the non-EEA spouse of an EU citizen working in Ireland be refused residence "
    "for never having lawfully lived in the EU before?",
    "What does Article 6 of the AI Act classify as a high-risk AI system?",
    "Can the spouse of a Critical Skills Employment Permit holder work in Ireland?",
    "What is the minimum salary for an Irish EU Blue Card?",
    "I'm a non-EEA machine learning engineer moving to Ireland. What permit do I need, "
    "and would the AI Act apply to the system I build?",
]

st.set_page_config(page_title="EU & Irish Law RAG", page_icon="scales", layout="wide")

# Locally the key lives in .streamlit/secrets.toml, which costs no extra dependency;
# on Render it arrives as a real env var. The Groq client only reads the env var.
try:
    if "GROQ_API_KEY" in st.secrets and not os.environ.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass


@st.cache_resource(show_spinner="Loading indexes...")
def get_retriever() -> Retriever:
    return Retriever()


@st.cache_resource
def get_client():
    from rag.generate import client
    return client()


def grounding_badge(score: float | None) -> str:
    if score is None:
        return "n/a"
    mark = "ok" if score >= 0.6 else "weak" if score >= 0.45 else "poor"
    return "%.2f (%s)" % (score, mark)


# ----------------------------------------------------------------------- sidebar
st.sidebar.title("Settings")

if not (DB.exists() and FAISS_PATH.exists()):
    st.error(
        "No index found at `data/index/`. Build it with:\n\n"
        "```\npython -m rag.fetch\npython -m rag.chunk\npython -m rag.index\n```"
    )
    st.stop()

mode = st.sidebar.radio(
    "Retrieval", ["Agentic (multi-search)", "One-shot"], index=0,
    help="Agentic lets the model search repeatedly and re-check its own grounding. "
         "It costs 3-6 model calls instead of 1.",
)
fusion = st.sidebar.radio(
    "Fuse", ["hybrid", "bm25", "dense"], index=0,
    help="hybrid = BM25 and dense fused with Reciprocal Rank Fusion. The other two "
         "are here so you can see what each retriever misses on its own.",
)
chosen = st.sidebar.multiselect(
    "Sources", list(TYPE_LABELS), default=list(LAW_TYPES),
    format_func=lambda k: TYPE_LABELS[k],
)
top_k = st.sidebar.slider("Sources per answer", 3, 15, 8)

st.sidebar.caption("Generator: `%s` (open-weight, via Groq)" % MODEL)
st.sidebar.caption("Embeddings: `BAAI/bge-small-en-v1.5` (384-d, ONNX)")
if not os.environ.get("GROQ_API_KEY"):
    st.sidebar.warning("GROQ_API_KEY is not set — retrieval works, generation will not.")

retriever = get_retriever()
n_chunks = retriever.con.execute("SELECT count(*) FROM chunks").fetchone()[0]
st.sidebar.caption("Corpus: %s chunks" % format(n_chunks, ","))

# -------------------------------------------------------------------------- main
st.title("EU & Irish law — hybrid RAG")
st.caption(
    "Hybrid retrieval over EU regulation, Acts of the Oireachtas, Irish immigration "
    "guidance and CJEU case law, with faithfulness scored on every answer. "
    "**Research aid, not legal advice.**"
)

with st.expander("Try one of these"):
    for i, ex in enumerate(EXAMPLES):
        if st.button(ex, key="ex%d" % i, use_container_width=True):
            st.session_state.question = ex

question = st.text_area(
    "Question", key="question", height=80,
    placeholder="e.g. What obligations does a provider of a high-risk AI system have?",
)
go = st.button("Ask", type="primary")

if go and question.strip():
    if not chosen:
        st.warning("Pick at least one source type.")
        st.stop()
    types = tuple(chosen)

    trace: list = []
    with st.spinner("Searching and answering..."):
        try:
            if mode.startswith("Agentic"):
                from rag.agent import run as agent_run
                text, hits = agent_run(question, retriever, get_client(),
                                       k=top_k, trace=trace, source_types=types)
            else:
                hits = retriever.search(question, k=top_k, mode=fusion,
                                        source_types=types)
                text = answer(question, hits, client=get_client())
        except Exception as exc:
            st.error("Generation failed: %s: %s" % (type(exc).__name__, exc))
            st.stop()

    scores = live_scores(text, hits, retriever.model)

    left, right = st.columns([3, 2])
    with left:
        st.markdown("### Answer")
        if abstained(text):
            st.info(text)
            st.caption("Abstaining is the correct behaviour when the corpus does not "
                       "support an answer — it is a measured metric, not a failure.")
        else:
            st.markdown(text)

    with right:
        st.markdown("### Quality")
        c1, c2 = st.columns(2)
        c1.metric("Grounding", grounding_badge(scores["grounding"]))
        c2.metric("Citations valid",
                  "n/a" if scores["citation_valid"] is None
                  else "%.0f%%" % (100 * scores["citation_valid"]))
        st.caption("Grounding = mean best cosine similarity between each answer "
                   "sentence and the retrieved sources. Free and instant; the LLM "
                   "judge below is the real measurement.")

        if st.button("Run LLM judge on this answer"):
            with st.spinner("Grading claims..."):
                v = judge(text, hits, get_client())
            if v.get("faithfulness") is None:
                st.write("Nothing to grade (abstained or unparseable).")
            else:
                j1, j2 = st.columns(2)
                j1.metric("Faithfulness", "%.0f%%" % (100 * v["faithfulness"]))
                j2.metric("Hallucination", "%.0f%%" % (100 * v["hallucination_rate"]))
                for c in v["claims"]:
                    icon = {"supported": "+", "unsupported": "?",
                            "contradicted": "!"}.get(c["verdict"], "?")
                    st.write("`%s` %s" % (icon, c["claim"]))

        if trace:
            st.markdown("### Agent trace")
            for step in trace:
                if "retry" in step:
                    st.write("retry: %s" % step["retry"])
                else:
                    st.write("search `%s` in **%s** -> %d new sources"
                             % (step["query"], step["source_type"], step["n_new"]))

    st.markdown("### Sources")
    st.caption("`found_by` is the whole argument for hybrid retrieval: bm25 catches "
               "exact article numbers and figures, dense catches paraphrase.")
    for i, h in enumerate(hits, 1):
        bm = h.get("bm25_rank") or "-"
        dn = h.get("dense_rank") or "-"
        head = "**[%d] %s** — %s  ·  bm25 #%s · dense #%s" % (
            i, h["citation"], h.get("found_by", "agent"), bm, dn)
        with st.expander(head):
            if h.get("note"):
                st.warning(h["note"])
            st.write(h["text"])
            if h.get("url"):
                st.markdown("[Source](%s)" % h["url"])

    with st.expander("Per-sentence grounding"):
        for row in scores["per_sentence"]:
            st.write("`%.2f` cites %s (best: [%d]) — %s"
                     % (row["grounding"], row["cites"] or "nothing",
                        row["best_source"], row["sentence"]))
