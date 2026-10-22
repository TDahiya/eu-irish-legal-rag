"""Turn data/raw/ into data/chunks.jsonl, splitting on real document structure.

Legal text has authoritative units - an EU article, an Irish section, a judgment
paragraph - and a citation is only useful if it points at one. So we split on those
markers rather than on a fixed character window, and only fall back to windowing
when a single unit is too long to embed.

    python -m rag.chunk
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from lxml import etree, html

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "chunks.jsonl"

# bge-small-en-v1.5 truncates at 512 tokens; ~1400 chars of English is ~350 tokens,
# comfortably inside it with room for the citation prefix.
MAX_CHARS = 1400
OVERLAP = 200
MIN_CHARS = 120          # drop navigation scraps and stray headings

# A structural marker alone in a block begins a new unit.
ART = re.compile(r"^Article\s+(\d+[a-z]?)\b", re.I)
DIV = re.compile(r"^(CHAPTER|SECTION|TITLE|ANNEX|PART)\s+([IVXLC\d]+[A-Za-z]?)\b")
BOOK_CH = re.compile(r"^(CHAPTER|BOOK|PART|Chapter)\s+([IVXLC\d]+[A-Za-z]?)\.?$")

EURLEX_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}"
ISB_URL = "https://www.irishstatutebook.ie/eli/{eli}/{stage}/en/html"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _window(text: str) -> list[str]:
    """Split an over-long unit on sentence boundaries, with overlap."""
    if len(text) <= MAX_CHARS:
        return [text]
    parts, start = [], 0
    while start < len(text):
        end = start + MAX_CHARS
        if end < len(text):
            # prefer a sentence break, else any space, in the last quarter
            floor = start + MAX_CHARS * 3 // 4
            cut = text.rfind(". ", floor, end)
            if cut == -1:
                cut = text.rfind(" ", floor, end)
            if cut != -1:
                end = cut + 1
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - OVERLAP, start + 1)
    return [p for p in parts if p]


def units_html(path: Path) -> list[tuple[str, str]]:
    """(label, text) per structural unit of an EU act, judgment or guidance page.

    One flat pass over block elements in document order works for both the modern OJ
    XHTML (class="oj-ti-art") and the pre-2010 flat HTML, because both render an
    article heading as its own short paragraph.
    """
    root = html.parse(str(path)).getroot()
    for bad in root.xpath("//script|//style|//nav|//header|//footer"):
        bad.getparent().remove(bad)

    blocks = [_clean(" ".join(p.itertext()))
              for p in root.xpath("//p|//li|//h1|//h2|//h3")]

    units: list[tuple[str, str]] = []
    label, buf = "", []
    for b in blocks:
        if not b:
            continue
        m = ART.match(b)
        if m and len(b) < 30:
            units.append((label, " ".join(buf)))
            label, buf = "Article " + m.group(1), []
            continue
        m = DIV.match(b)
        if m and len(b) < 60:
            units.append((label, " ".join(buf)))
            label, buf = m.group(1).title() + " " + m.group(2), []
            continue
        buf.append(b)
    units.append((label, " ".join(buf)))
    return [(lab, _clean(t)) for lab, t in units if len(_clean(t)) >= MIN_CHARS]


def units_xml(path: Path) -> list[tuple[str, str]]:
    """(label, text) per <sect> of an Irish Act's ELI XML."""
    root = etree.parse(str(path)).getroot()
    out = []
    for sect in root.findall(".//sect"):
        # ponytail: <number> is the citable part and is reliable; the section heading
        # sits in running text and its position differs between acts, so we don't
        # try to pull it out separately.
        num = (sect.findtext("number") or "").strip().rstrip(".")
        text = _clean(" ".join(sect.itertext()))
        if len(text) >= MIN_CHARS:
            out.append(("s. " + num if num else "", text))
    if not out:   # statutory instruments arrive as HTML, not ELI XML
        return units_html(path)
    return out


def units_txt(path: Path) -> list[tuple[str, str]]:
    """(label, text) per chapter of a Project Gutenberg plain-text book."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    i = raw.find("*** START OF")
    if i != -1:
        raw = raw[raw.find("\n", i) + 1:]
    i = raw.find("*** END OF")
    if i != -1:
        raw = raw[:i]

    units: list[tuple[str, str]] = []
    label, buf = "", []
    for para in re.split(r"\n\s*\n", raw):
        p = _clean(para)
        if not p:
            continue
        m = BOOK_CH.match(p)
        if m and len(p) < 40:
            units.append((label, " ".join(buf)))
            label, buf = m.group(1).title() + " " + m.group(2), []
            continue
        buf.append(p)
    units.append((label, " ".join(buf)))
    return [(lab, _clean(t)) for lab, t in units if len(_clean(t)) >= MIN_CHARS]


def _citation(meta: dict, label: str) -> str:
    base = meta.get("short", "?")
    return base + ", " + label if label else base


def _url(meta: dict) -> str:
    if "celex" in meta:
        return EURLEX_URL.format(celex=meta["celex"])
    if "eli" in meta:
        stage = "made" if "/si/" in meta["eli"] else "enacted"
        return ISB_URL.format(eli=meta["eli"], stage=stage)
    if "id" in meta:
        return "https://www.gutenberg.org/ebooks/" + str(meta["id"])
    return meta.get("url", "")


PARSERS = {"html": units_html, "xml": units_xml, "txt": units_txt}


def main() -> int:
    index = json.loads((RAW / "index.json").read_text(encoding="utf-8"))

    rows, skipped = [], []
    for rel, meta in sorted(index.items()):
        path = RAW / rel
        if not path.exists():
            continue
        try:
            units = PARSERS[meta.get("fmt", "html")](path)
        except Exception as exc:
            skipped.append(rel + ": " + type(exc).__name__ + ": " + str(exc)[:80])
            continue
        url = _url(meta)
        for label, text in units:
            for part in _window(text):
                rows.append({
                    "id": len(rows),
                    "text": part,
                    "citation": _citation(meta, label),
                    "instrument": meta.get("short", ""),
                    "source_type": meta["source_type"],
                    "domain": meta.get("domain", "general"),
                    "ireland_bound": meta.get("ireland_bound"),
                    "note": meta.get("note", ""),
                    "url": url,
                })

    with OUT.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_type: dict[str, int] = {}
    chars = 0
    for r in rows:
        by_type[r["source_type"]] = by_type.get(r["source_type"], 0) + 1
        chars += len(r["text"])
    print("{:,} chunks from {} documents -> {}".format(len(rows), len(index), OUT))
    for k, v in sorted(by_type.items(), key=lambda kv: -kv[1]):
        print("  {:16} {:>7,}".format(k, v))
    print("  mean chunk {} chars, {:,} total".format(chars // max(len(rows), 1), chars))
    for s in skipped:
        print("  SKIPPED", s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
