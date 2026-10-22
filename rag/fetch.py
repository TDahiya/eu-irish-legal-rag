"""Fetch the corpus declared in sources.yaml into data/raw/.

    python -m rag.fetch --dry-run    # check every URL resolves, download nothing
    python -m rag.fetch              # download (skips anything already cached)

Writes data/raw/index.json: {relative_path: metadata} for chunk.py to consume, so
nothing downstream needs to re-read sources.yaml.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from xml.etree import ElementTree

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MANIFEST = Path(__file__).with_name("sources.yaml")

UA = "legal-rag/0.1 (student portfolio project; +https://github.com/)"

# eur-lex.europa.eu bot-gates with HTTP 202 and an empty body. Cellar is the way in.
CELLAR = "https://publications.europa.eu/resource/celex/{celex}"
# Pre-2010 directives and most CJEU judgments have no XHTML manifestation in Cellar
# (404), only text/html. Try the better format first, fall back.
CELLAR_ACCEPTS = ("application/xhtml+xml", "text/html")
ELI = "https://www.irishstatutebook.ie/eli/{eli}/{stage}"
GUTENBERG = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

_last_hit: dict[str, float] = {}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


def _host(url: str) -> str:
    return url.split("/", 3)[2]


def _get(session: requests.Session, url: str, delays: dict, headers=None, stream=False):
    """GET with a per-host crawl delay."""
    host = _host(url)
    wait = delays.get(host, 1.0) - (time.monotonic() - _last_hit.get(host, 0.0))
    if wait > 0:
        time.sleep(wait)
    _last_hit[host] = time.monotonic()
    return session.get(url, headers=headers, timeout=60, stream=stream,
                       allow_redirects=True)


def _jobs(manifest: dict) -> list[dict]:
    """Flatten the manifest into {url, path, accepts, meta} download jobs.

    Sitemap groups are expanded later (they need a network call), so they come back
    as a single job with expand='sitemap'.
    """
    out = []
    for source_type, group in manifest.items():
        if source_type == "crawl_delay":
            continue
        fetcher = group["fetcher"]
        for item in group["items"]:
            meta = {"source_type": source_type, **{k: v for k, v in item.items()}}
            if fetcher == "cellar":
                celex = item["celex"]
                out.append({
                    # Treaty CELEX ids contain a slash (12016E/TXT) — it must be encoded
                    # or Cellar reads it as a path segment and 404s.
                    "url": CELLAR.format(celex=celex.replace("/", "%2F")),
                    "path": f"{source_type}/{_slug(celex)}.html",
                    "accepts": CELLAR_ACCEPTS,
                    "meta": {**meta, "fmt": "html"},
                })
            elif fetcher == "eli":
                eli = item["eli"]
                # Acts are published as ELI XML; statutory instruments are HTML only.
                stage = "made/en/html" if "/si/" in eli else "enacted/en/xml"
                out.append({
                    "url": ELI.format(eli=eli, stage=stage),
                    "path": f"{source_type}/{_slug(eli)}.{stage.rsplit('/', 1)[-1]}",
                    "meta": {**meta, "fmt": stage.rsplit("/", 1)[-1]},
                })
            elif fetcher == "gutenberg":
                out.append({
                    "url": GUTENBERG.format(id=item["id"]),
                    "path": f"{source_type}/{item['id']}-{_slug(item['short'])[:40]}.txt",
                    "meta": {**meta, "fmt": "txt"},
                })
            elif fetcher == "sitemap":
                out.append({"expand": "sitemap", "url": item["sitemap"], "meta": meta,
                            "include": item.get("include", []),
                            "exclude": item.get("exclude", [])})
            else:
                raise ValueError(f"unknown fetcher {fetcher!r}")
    return out


def _expand_sitemap(job: dict, session, delays) -> list[dict]:
    """Turn a sitemap job into one job per matching page."""
    r = _get(session, job["url"], delays)
    r.raise_for_status()
    locs = [el.text for el in ElementTree.fromstring(r.content).iter()
            if el.tag.endswith("loc") and el.text]
    inc, exc = job["include"], job["exclude"]
    keep = [u for u in locs
            if (not inc or any(w in u.lower() for w in inc))
            and not any(w in u.lower() for w in exc)]
    st = job["meta"]["source_type"]
    print(f"  sitemap: {len(locs)} urls -> {len(keep)} kept")
    return [{
        "url": u,
        "path": f"{st}/{_slug(u.split('//')[-1].split('/', 1)[-1])[:90] or 'index'}.html",
        "meta": {**job["meta"], "url": u, "fmt": "html"},
    } for u in keep]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="check URLs resolve; download nothing")
    ap.add_argument("--only", help="only this source_type")
    args = ap.parse_args()

    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    delays = manifest["crawl_delay"]
    session = requests.Session()
    session.headers["User-Agent"] = UA

    jobs = _jobs(manifest)
    if args.only:
        jobs = [j for j in jobs if j["meta"]["source_type"] == args.only]

    # On a dry run, don't expand sitemaps: those URLs came out of the site's own sitemap,
    # so checking each one proves nothing and costs 10s apiece. Check the sitemap itself.
    expanded = []
    for j in jobs:
        if j.get("expand") != "sitemap":
            expanded.append(j)
        elif args.dry_run:
            expanded.append({"url": j["url"], "path": "-", "meta": j["meta"]})
        else:
            expanded.extend(_expand_sitemap(j, session, delays))
    jobs = expanded

    index, bad, total_bytes = {}, [], 0
    for i, job in enumerate(jobs, 1):
        dest = RAW / job["path"]
        label = job["meta"].get("short") or job["path"]

        if not args.dry_run and dest.exists() and dest.stat().st_size > 0:
            index[job["path"]] = job["meta"]
            total_bytes += dest.stat().st_size
            print(f"[{i:3}/{len(jobs)}] cached {label}")
            continue

        try:
            for accept in job.get("accepts") or (None,):
                hdrs = {"Accept": accept, "Accept-Language": "eng"} if accept else None
                r = _get(session, job["url"], delays, hdrs, stream=args.dry_run)
                if r.status_code == 200:
                    break
                r.close()
            ok = r.status_code == 200 and (args.dry_run or len(r.content) > 500)
            if not ok:
                bad.append((label, job["url"], r.status_code, len(r.content or b"")))
                print(f"[{i:3}/{len(jobs)}] FAIL {r.status_code} {label}")
                r.close()
                continue
            if args.dry_run:
                r.close()
                print(f"[{i:3}/{len(jobs)}] ok   {label}")
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(r.content)
                index[job["path"]] = job["meta"]
                total_bytes += len(r.content)
                print(f"[{i:3}/{len(jobs)}] got  {len(r.content):>9,}B {label}")
        except Exception as exc:                      # network/DNS/timeout
            bad.append((label, job["url"], type(exc).__name__, 0))
            print(f"[{i:3}/{len(jobs)}] ERR  {type(exc).__name__} {label}")

    if not args.dry_run:
        (RAW / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False),
                                        encoding="utf-8")

    print(f"\n{len(jobs) - len(bad)}/{len(jobs)} ok, {total_bytes:,} bytes")
    if bad:
        print(f"\n{len(bad)} FAILED — fix these in sources.yaml:")
        for label, url, status, size in bad:
            print(f"  {status:>12}  {label}\n                {url}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
