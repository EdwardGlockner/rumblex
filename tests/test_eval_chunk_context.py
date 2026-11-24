import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "eval_chunk_context.py"


@pytest.fixture(scope="module")
def cc():
    spec = importlib.util.spec_from_file_location("eval_chunk_context", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_context_comes_from_the_right_chunk(cc):
    texts = {"none": [["a0", "a1", "a2"], ["b0", "b1"]], "prev": [["A0", "A1", ""], ["B0", ""]]}
    assert cc.context_for("none", texts, 0, 1) is None
    assert cc.context_for("prev", texts, 0, 0) is None
    assert cc.context_for("prev", texts, 0, 2) == "A1"
    assert cc.context_for("twopass", texts, 0, 2) == "a1"
    assert cc.context_for("twopass", texts, 1, 0) is None
    # another talk's chunk at the same position, wrapping when that talk is shorter
    assert cc.context_for("other", texts, 0, 2) == "b0"
    assert cc.context_for("other", texts, 1, 1) == "a1"


def test_empty_previous_text_gives_no_context(cc):
    texts = {"none": [["", "x"]], "prev": [["", ""]]}
    assert cc.context_for("twopass", texts, 0, 1) is None
    assert cc.context_for("prev", texts, 0, 1) is None


def test_prev_decodes_a_talk_in_order(cc):
    assert cc.decode_order("prev", [3, 1]) == [[(0, 0), (1, 0)], [(0, 1)], [(0, 2)]]
    assert cc.decode_order("twopass", [2, 1]) == [[(0, 0), (0, 1), (1, 0)]]


def test_edge_distance(cc):
    assert cc.edge_distance([3, 2]) == [2, 1, 0, 0, 1]
    assert cc.edge_distance([4]) == [4, 4, 4, 4]  # one chunk has no boundary
    assert cc.edge_distance([1, 1, 1]) == [0, 0, 0]


def test_zone_errors_places_each_kind_of_error(cc):
    ref = ["a", "b", "c", "d", "e", "f", "g", "h"]
    # substitution at the boundary (d -> x), one far from it (a -> y)
    z = cc.zone_errors(ref, [["y", "b", "c", "x"], ["e", "f", "g", "h"]], zone=1)
    assert z["near"] == {"errors": 1, "ref_words": 2}
    assert z["far"] == {"errors": 1, "ref_words": 6}
    # a word dropped at the boundary counts as near it
    z = cc.zone_errors(ref, [["a", "b", "c"], ["e", "f", "g", "h"]], zone=1)
    assert z["near"] == {"errors": 1, "ref_words": 3}
    assert z["far"] == {"errors": 0, "ref_words": 5}
    # a word repeated at the boundary is an insertion there
    z = cc.zone_errors(ref, [["a", "b", "c", "d"], ["d", "e", "f", "g", "h"]], zone=1)
    assert z["near"]["errors"] == 1 and z["far"]["errors"] == 0
    assert z["near"]["ref_words"] + z["far"]["ref_words"] == 8


def test_zone_errors_with_nothing_recognised(cc):
    z = cc.zone_errors(["a", "b"], [[], []])
    assert z["far"] == {"errors": 2, "ref_words": 2}


def test_boundary_shape_skips_empty_chunks(cc):
    s = cc.boundary_shape(["It ended.", "", "and went on", "Then"])
    assert s == {"boundaries": 2, "ends_sentence": 1, "starts_lowercase": 1}


def test_score_variant_adds_up(cc):
    talks = [{"id": "t", "ref": "a b c d e f"}]
    r = cc.score_variant(talks, [["a b c", "x e f"]], lambda s: s.lower())
    assert r["errors"] == 1 and r["ref_words"] == 6
    assert r["near"]["ref_words"] + r["far"]["ref_words"] == 6
    assert r["wer_per_talk"] == {"t": round(1 / 6, 4)}
