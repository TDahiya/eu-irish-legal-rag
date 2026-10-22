"""The smallest checks that fail if the chunking or fusion logic breaks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chunk import MAX_CHARS, OVERLAP, _window, units_txt


def test_window_leaves_short_text_alone():
    assert _window("short") == ["short"]


def test_window_respects_the_embedder_limit_and_loses_nothing():
    text = ". ".join("sentence number %d with some filler" % i for i in range(400)) + "."
    parts = _window(text)
    assert len(parts) > 1
    assert all(len(p) <= MAX_CHARS for p in parts), [len(p) for p in parts]
    assert parts[0][:40] in text and parts[-1][-40:] in text


def test_window_overlaps_so_a_fact_on_a_seam_survives():
    text = ". ".join("sentence number %d with some filler" % i for i in range(400)) + "."
    parts = _window(text)
    tail_words = parts[0].split()[-6:]
    assert any(w in parts[1] for w in tail_words)


def test_window_terminates_on_text_with_no_spaces():
    parts = _window("x" * (MAX_CHARS * 3))
    assert len(parts) >= 3
    assert all(len(p) <= MAX_CHARS for p in parts)


def test_gutenberg_licence_boilerplate_is_stripped(tmp_path):
    body = "Liberty is the only unfailing source of progress. " * 8
    book = tmp_path / "b.txt"
    book.write_text(
        "The Project Gutenberg eBook of Something\nLICENCE NOISE HERE\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\n\n"
        "Chapter I\n\n" + body + "\n\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\n"
        "MORE LICENCE NOISE\n",
        encoding="utf-8",
    )
    units = units_txt(book)
    assert units, "chapter body should survive"
    joined = " ".join(label + " " + text for label, text in units)
    assert "LICENCE NOISE" not in joined
    assert "Chapter I" in [label for label, _ in units]


# ----------------------------------------------------------------- fusion / query
from rag.retrieve import RRF_K, _fts_query, rrf


def test_rrf_rewards_agreement_between_retrievers():
    # id 9 is top-ranked by both retrievers; id 1 is top-ranked by only one.
    fused = dict(rrf({"bm25": [9, 1], "dense": [9, 2]}))
    assert fused[9] == 2 / (RRF_K + 1)
    assert fused[9] > fused[1]


def test_rrf_scores_are_rank_sums_not_normalised_scores():
    # id 2 sits at rank 2 in both lists
    fused = dict(rrf({"bm25": [1, 2, 3], "dense": [3, 2, 1]}))
    assert fused[2] == 2 / (RRF_K + 2)
    # ...and 1/61 + 1/63 > 2/62, so being first somewhere beats being second
    # everywhere. That is RRF working as intended, not a bug.
    assert fused[1] > fused[2]


def test_rrf_is_a_passthrough_for_a_single_list():
    assert [cid for cid, _ in rrf({"dense": [7, 4, 9]})] == [7, 4, 9]


def test_rrf_is_deterministic_on_ties():
    assert rrf({"a": [5, 3], "b": [3, 5]}) == rrf({"b": [3, 5], "a": [5, 3]})


def test_fts_query_strips_operators_that_would_be_syntax_errors():
    hostile = 'AI Act Article 6(2) -- "high risk" NEAR/3 * : AND'
    out = _fts_query(hostile)
    assert "(" not in out and ")" not in out and "*" not in out and ":" not in out
    assert '"article"' in out and '"6"' in out and '"2"' in out
    assert out.count(" OR ") == len(out.split(" OR ")) - 1


def test_fts_query_keeps_single_digits_but_drops_stray_letters():
    out = _fts_query("stamp 1 g permit")
    assert '"1"' in out and '"stamp"' in out and '"permit"' in out
    assert '"g"' not in out


def test_fts_query_of_pure_punctuation_is_empty_not_invalid():
    assert _fts_query("?!  ... ()") == ""


# ------------------------------------------------- agent citation numbering
from rag.agent import _render


def _hit(i, text):
    return {"id": i, "citation": "Instrument s. %d" % i, "text": text}


def test_agent_numbers_tool_results_globally_across_searches():
    """[n] in the final answer must index the single growing hit list, or a
    multi-search answer cites the wrong source."""
    first = [_hit(10, "alpha"), _hit(11, "beta")]
    second = [_hit(12, "gamma")]

    hits = []
    out1 = _render(first, len(hits))
    hits.extend(first)
    out2 = _render(second, len(hits))
    hits.extend(second)

    assert "[1] Instrument s. 10" in out1
    assert "[2] Instrument s. 11" in out1
    # the second search continues the numbering rather than restarting at [1]
    assert "[3] Instrument s. 12" in out2
    assert "[1]" not in out2
    # and the numbers resolve back to the right hits
    for n, expected in [(1, "alpha"), (2, "beta"), (3, "gamma")]:
        assert hits[n - 1]["text"] == expected


def test_agent_flags_instruments_that_do_not_bind_ireland():
    h = _hit(1, "Blue Card rules")
    h["note"] = "Ireland not bound - Title V opt-out."
    assert "Title V opt-out" in _render([h], 0)

    bare = _hit(2, "Schengen rules")
    bare["ireland_bound"] = 0
    assert "Ireland is not bound" in _render([bare], 0)


def test_agent_tells_the_model_when_a_search_found_nothing():
    assert "No results" in _render([], 0)
