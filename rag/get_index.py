"""Fetch the prebuilt index at deploy time.

The index is built locally and published as a GitHub Release asset rather than
committed. At 36k chunks it is ~56 MB of FAISS plus ~75 MB of SQLite, and every
rebuild would add that again to git history forever -- a portfolio repository
nobody can clone cheaply is worse than one extra build step. Release assets don't
count against clone size and allow 2 GB.

Rebuilding on Render instead is not an option: embedding runs at ~10 chunks/s on a
normal CPU, so an hour here would be many hours on 0.1 CPU.

The repository is public, so the asset URL is public too and needs no secret.
GROQ_API_KEY stays the only one.

    python -m rag.get_index

A no-op if the index is already present, so it is safe to leave in the build
command.
"""
from __future__ import annotations

import os
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from rag.index import DB, FAISS_PATH, INDEX_DIR

URL = os.environ.get(
    "INDEX_URL",
    "https://github.com/TDahiya/eu-irish-legal-rag/releases/latest/download/index.tar.gz",
)


def main() -> int:
    if DB.exists() and FAISS_PATH.exists():
        print("index already present, nothing to do")
        return 0

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    print("downloading %s" % URL)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        with urllib.request.urlopen(URL) as resp:      # noqa: S310 - fixed https URL
            tmp.write(resp.read())
        archive = Path(tmp.name)

    try:
        with tarfile.open(archive) as tar:
            # filter="data" refuses absolute paths, '..' and special files
            tar.extractall(INDEX_DIR, filter="data")
    finally:
        archive.unlink(missing_ok=True)

    if not (DB.exists() and FAISS_PATH.exists()):
        print("archive did not contain chunks.db and dense.faiss")
        return 1
    print("index ready: %.1f MB db, %.1f MB faiss"
          % (DB.stat().st_size / 1e6, FAISS_PATH.stat().st_size / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
