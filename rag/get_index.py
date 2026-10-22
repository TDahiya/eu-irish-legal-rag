"""Fetch the prebuilt index at deploy time.

Render's free tier has no persistent disk and 0.1 CPU, so the index is built locally
and published as a GitHub Release asset rather than rebuilt on the box (embedding
~100k chunks there would take hours) or committed to git (GitHub's per-file limit is
100 MB, and LFS bandwidth is spent on every deploy).

    python -m rag.get_index

Set INDEX_URL to the release asset. A no-op if the index is already present, so it is
safe to leave in the build command.
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
    "https://github.com/OWNER/REPO/releases/latest/download/index.tar.gz",
)


def main() -> int:
    if DB.exists() and FAISS_PATH.exists():
        print("index already present, nothing to do")
        return 0
    if "OWNER/REPO" in URL:
        print("INDEX_URL is not set and the default is still a placeholder.\n"
              "Either set INDEX_URL, or build locally:\n"
              "  python -m rag.fetch && python -m rag.chunk && python -m rag.index")
        return 1

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    print("downloading %s" % URL)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        with urllib.request.urlopen(URL) as resp:          # noqa: S310 - fixed https URL
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
