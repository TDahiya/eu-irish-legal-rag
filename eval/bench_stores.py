"""FAISS vs ChromaDB on this corpus: build time, query latency, recall, footprint.

    python -m eval.bench_stores --sample 20000

Both stores get the same vectors, embedded once, so the comparison is of the stores
and not of two embedding runs. FAISS IndexFlatIP is an exact search, so it is used as
ground truth for recall; Chroma's default index is approximate (HNSW), so its recall
against that ground truth is the interesting number.

This is why the deployed app uses FAISS: see eval/report.md for the result.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import faiss
import numpy as np
from fastembed import TextEmbedding

from rag.index import CHUNKS, MODEL

QUERIES = [
    "high-risk AI system classification",
    "right of residence for more than three months",
    "critical skills employment permit spouse",
    "personal data breach notification 72 hours",
    "deportation order procedure Ireland",
    "naturalisation reckonable residence",
    "gatekeeper core platform service",
    "subsidiary protection application",
    "text and data mining exception",
    "transparency obligations for chatbots",
    "right to erasure search engine",
    "incident reporting obligations essential entities",
]


def _dir_size(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=20000,
                    help="chunks to index in each store")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    rows = [json.loads(l) for l in CHUNKS.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = rows[:args.sample]
    texts = [r["text"] for r in rows]
    print("%d chunks, embedding once with %s" % (len(rows), MODEL))

    model = TextEmbedding(MODEL)
    t0 = time.perf_counter()
    vecs = np.array(list(model.embed(texts)), dtype="float32")
    embed_s = time.perf_counter() - t0
    faiss.normalize_L2(vecs)
    qvecs = np.array(list(model.query_embed(QUERIES)), dtype="float32")
    faiss.normalize_L2(qvecs)
    print("  embedding took %.1fs (%.0f chunks/s)" % (embed_s, len(rows) / embed_s))

    tmp = Path(tempfile.mkdtemp(prefix="bench-"))
    result = {"n_chunks": len(rows), "embed_seconds": round(embed_s, 1), "stores": {}}

    try:
        # ------------------------------------------------------------- FAISS
        t0 = time.perf_counter()
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        faiss_build = time.perf_counter() - t0
        fpath = tmp / "dense.faiss"
        faiss.write_index(index, str(fpath))

        lat, truth = [], []
        for q in qvecs:
            t0 = time.perf_counter()
            _, ids = index.search(q.reshape(1, -1), args.k)
            lat.append((time.perf_counter() - t0) * 1000)
            truth.append(set(int(i) for i in ids[0] if i >= 0))
        result["stores"]["faiss"] = {
            "build_s": round(faiss_build, 2),
            "p50_ms": round(statistics.median(lat), 2),
            "p95_ms": round(max(lat), 2) if len(lat) < 20
                      else round(statistics.quantiles(lat, n=20)[18], 2),
            "disk_mb": round(_dir_size(fpath), 1),
            "recall_at_k": 1.0,          # exact search is its own ground truth
        }

        # ------------------------------------------------------------ Chroma
        import chromadb
        cpath = tmp / "chroma"
        t0 = time.perf_counter()
        client = chromadb.PersistentClient(path=str(cpath))
        coll = client.create_collection("bench", metadata={"hnsw:space": "cosine"})
        for start in range(0, len(rows), 2000):
            end = start + 2000
            coll.add(ids=[str(r["id"]) for r in rows[start:end]],
                     embeddings=vecs[start:end].tolist(),
                     documents=texts[start:end])
        chroma_build = time.perf_counter() - t0

        lat, recalls = [], []
        for q, want in zip(qvecs, truth):
            t0 = time.perf_counter()
            res = coll.query(query_embeddings=[q.tolist()], n_results=args.k)
            lat.append((time.perf_counter() - t0) * 1000)
            got = {int(i) for i in res["ids"][0]}
            recalls.append(len(got & want) / max(len(want), 1))
        result["stores"]["chroma"] = {
            "build_s": round(chroma_build, 2),
            "p50_ms": round(statistics.median(lat), 2),
            "p95_ms": round(max(lat), 2) if len(lat) < 20
                      else round(statistics.quantiles(lat, n=20)[18], 2),
            "disk_mb": round(_dir_size(cpath), 1),
            "recall_at_k": round(statistics.mean(recalls), 3),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    out = Path(__file__).with_name("bench_stores.json")
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    print("\n| store | build (s) | query p50 | query p95 | disk | recall@%d |" % args.k)
    print("|---|---|---|---|---|---|")
    for name, s in result["stores"].items():
        print("| %s | %.2f | %.2f ms | %.2f ms | %.1f MB | %.3f |" % (
            name, s["build_s"], s["p50_ms"], s["p95_ms"], s["disk_mb"],
            s["recall_at_k"]))
    print("\nwrote %s" % out.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
