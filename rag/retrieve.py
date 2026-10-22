"""Hybrid retrieval: SQLite FTS5 (BM25) + FAISS (dense), fused with RRF.

Why both. Legal questions split into two kinds and no single retriever is good at
both:

  "AI Act Article 6(2)", "Stamp 1G", "S.I. 548/2015", "EUR 38,000"
        exact tokens -- embeddings blur these into near neighbours; BM25 nails them

  "can my spouse work if I'm on a Critical Skills permit"
        no shared vocabulary with the statute that answers it -- BM25 misses, dense wins

Reciprocal Rank Fusion combines the two ranked lists using ranks only, so the two
incomparable score scales (negative BM25, cosine) never have to be normalised.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import faiss
import numpy as np
from fastembed import TextEmbedding

from rag.index import DB, FAISS_PATH, MODEL

# Everything except the public-domain philosophy books. Those are in the corpus for
# chunking scale and for questions about justice as an idea; letting them near a
# question about current law is how you get a visa answer sourced from Hobbes.
LAW_TYPES = ("eu_law", "ie_legislation", "ie_guidance", "case_law")

RRF_K = 60          # standard constant; damps the top-rank dominance
POOL = 50           # candidates per retriever before fusion


def _fts_query(question: str) -> str:
    """Make an FTS5 MATCH expression that cannot be a syntax error.

    Everything that is not a word character is dropped (FTS5 treats ", *, :, -, (, )
    and NEAR as operators), each surviving token is quoted, and tokens are OR-ed.
    OR rather than FTS5's default AND: a natural-language question AND-ed against a
    corpus returns nothing, and BM25 already rewards documents that match more and
    rarer terms.
    """
    tokens = re.findall(r"\w+", question.lower())
    tokens = [t for t in tokens if len(t) > 1 or t.isdigit()]
    return " OR ".join('"%s"' % t for t in tokens)


class Retriever:
    def __init__(self, db: Path = DB, faiss_path: Path = FAISS_PATH):
        self.con = sqlite3.connect(str(db), check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.index = faiss.read_index(str(faiss_path))
        self.model = TextEmbedding(MODEL)

    # ---------------------------------------------------------------- retrievers
    def bm25(self, question: str, pool: int = POOL,
             source_types: tuple[str, ...] = LAW_TYPES) -> list[int]:
        match = _fts_query(question)
        if not match:
            return []
        placeholders = ",".join("?" * len(source_types))
        # FTS5's bm25() returns a NEGATIVE score where more negative is a better
        # match, so this orders ASC, not DESC.
        sql = (
            "SELECT f.rowid FROM chunks_fts f JOIN chunks c ON c.id = f.rowid "
            "WHERE chunks_fts MATCH ? AND c.source_type IN (%s) "
            "ORDER BY bm25(chunks_fts) ASC LIMIT ?" % placeholders
        )
        rows = self.con.execute(sql, (match, *source_types, pool)).fetchall()
        return [r["rowid"] for r in rows]

    def dense(self, question: str, pool: int = POOL,
              source_types: tuple[str, ...] = LAW_TYPES) -> list[int]:
        # query_embed applies bge's "represent this sentence for searching..."
        # instruction prefix; embed() does not. Passages were indexed with embed(),
        # queries must use query_embed() or the asymmetry is lost.
        vec = np.array(list(self.model.query_embed([question])), dtype="float32")
        faiss.normalize_L2(vec)
        # Over-fetch, then drop what the filter excludes, so filtering cannot leave
        # the pool short.
        _, ids = self.index.search(vec, pool * 5)
        allowed = self._allowed(ids[0], source_types)
        return [int(i) for i in ids[0] if i >= 0 and int(i) in allowed][:pool]

    def _allowed(self, ids, source_types: tuple[str, ...]) -> set[int]:
        cand = [int(i) for i in ids if i >= 0]
        if not cand:
            return set()
        sql = "SELECT id FROM chunks WHERE id IN (%s) AND source_type IN (%s)" % (
            ",".join("?" * len(cand)), ",".join("?" * len(source_types)))
        return {r["id"] for r in self.con.execute(sql, (*cand, *source_types))}

    # ---------------------------------------------------------------------- fuse
    def search(self, question: str, k: int = 8, mode: str = "hybrid",
               source_types: tuple[str, ...] = LAW_TYPES) -> list[dict]:
        lex = self.bm25(question, POOL, source_types) if mode in ("hybrid", "bm25") else []
        den = self.dense(question, POOL, source_types) if mode in ("hybrid", "dense") else []

        fused = rrf({"bm25": lex, "dense": den})[:k]
        if not fused:
            return []

        ids = [cid for cid, _ in fused]
        rows = {r["id"]: dict(r) for r in self.con.execute(
            "SELECT * FROM chunks WHERE id IN (%s)" % ",".join("?" * len(ids)), ids)}

        hits = []
        for rank, (cid, score) in enumerate(fused, 1):
            hit = rows[cid]
            hit["rank"] = rank
            hit["score"] = round(score, 5)
            hit["bm25_rank"] = lex.index(cid) + 1 if cid in lex else None
            hit["dense_rank"] = den.index(cid) + 1 if cid in den else None
            hit["found_by"] = ("both" if hit["bm25_rank"] and hit["dense_rank"]
                               else "bm25" if hit["bm25_rank"] else "dense")
            hits.append(hit)
        return hits


def rrf(ranked_lists: dict[str, list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank(d)).

    Uses ranks only, so the retrievers' incompatible score scales never meet.
    """
    scores: dict[int, float] = {}
    for ids in ranked_lists.values():
        for rank, cid in enumerate(ids, 1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
