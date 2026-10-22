"""Build the two search indexes from data/chunks.jsonl.

    python -m rag.index

Produces, in data/index/:
  chunks.db    SQLite: chunk rows + an FTS5 index over the text (lexical / BM25)
  dense.faiss  FAISS inner-product index over L2-normalised embeddings (cosine)

FAISS position i is chunk id i. That invariant is what lets the two retrievers fuse
by id without a lookup table, so it is asserted rather than assumed.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import faiss
import numpy as np
from fastembed import TextEmbedding

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "chunks.jsonl"
INDEX_DIR = ROOT / "data" / "index"
DB = INDEX_DIR / "chunks.db"
FAISS_PATH = INDEX_DIR / "dense.faiss"

MODEL = "BAAI/bge-small-en-v1.5"      # 384-dim, 67 MB ONNX, no torch

# Small on purpose. Transformer attention is O(seq_len^2) per item and a batch is
# allocated in one go, so with chunks up to MAX_CHARS the cost is
# batch * heads * seq_len^2 * 4 bytes. At batch=256 that measured 4.8 GB of RSS and
# 33 s for the first 256 chunks, purely from thrashing; at 32 it is a fraction of
# that. Short test strings hide this completely -- it only appears on real
# full-length chunks.
BATCH = 32

SCHEMA = """
CREATE TABLE chunks (
    id            INTEGER PRIMARY KEY,
    text          TEXT NOT NULL,
    citation      TEXT,
    instrument    TEXT,
    source_type   TEXT,
    domain        TEXT,
    ireland_bound INTEGER,
    note          TEXT,
    url           TEXT
);
-- content='chunks' makes this a contentless FTS index: it stores the inverted index
-- only and reads the text back from `chunks`, instead of keeping a second copy.
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text, content='chunks', content_rowid='id', tokenize='porter unicode61'
);
CREATE INDEX idx_source_type ON chunks(source_type);
CREATE INDEX idx_domain ON chunks(domain);
"""


def load_chunks() -> list[dict]:
    rows = [json.loads(line) for line in CHUNKS.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    assert [r["id"] for r in rows] == list(range(len(rows))), \
        "chunk ids must be contiguous from 0 — FAISS positions depend on it"
    return rows


def build_sqlite(rows: list[dict]) -> None:
    DB.unlink(missing_ok=True)
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    con.executemany(
        "INSERT INTO chunks (id, text, citation, instrument, source_type, domain,"
        " ireland_bound, note, url) VALUES (?,?,?,?,?,?,?,?,?)",
        [(r["id"], r["text"], r["citation"], r["instrument"], r["source_type"],
          r["domain"],
          None if r["ireland_bound"] is None else int(r["ireland_bound"]),
          r["note"], r["url"]) for r in rows],
    )
    con.execute("INSERT INTO chunks_fts (rowid, text) SELECT id, text FROM chunks")
    con.commit()
    con.execute("VACUUM")
    con.close()


def build_faiss(rows: list[dict]) -> faiss.Index:
    model = TextEmbedding(MODEL)
    dim = 384
    index = faiss.IndexFlatIP(dim)
    texts = [r["text"] for r in rows]

    done = 0
    for start in range(0, len(texts), BATCH):
        batch = texts[start:start + BATCH]
        vecs = np.array(list(model.embed(batch)), dtype="float32")
        faiss.normalize_L2(vecs)          # inner product on unit vectors == cosine
        index.add(vecs)
        done += len(batch)
        print("  embedded {:,}/{:,}".format(done, len(texts)), end="\r", flush=True)
    print()

    assert index.ntotal == len(rows), (index.ntotal, len(rows))
    faiss.write_index(index, str(FAISS_PATH))
    return index


def main() -> int:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_chunks()
    print("{:,} chunks".format(len(rows)))

    build_sqlite(rows)
    print("  sqlite  {:>8.1f} MB  {}".format(DB.stat().st_size / 1e6, DB.name))

    build_faiss(rows)
    size = FAISS_PATH.stat().st_size / 1e6
    print("  faiss   {:>8.1f} MB  {}".format(size, FAISS_PATH.name))

    # Render free gives 512 MB total. A flat index is fully resident during search,
    # so if it is large, switch to an 8-bit scalar quantiser (one-line change in
    # build_faiss) which costs ~4x less memory for ~1% recall.
    if size > 120:
        print("  WARNING: flat index is {:.0f} MB — quantise before deploying".format(size))
    return 0


if __name__ == "__main__":
    sys.exit(main())
