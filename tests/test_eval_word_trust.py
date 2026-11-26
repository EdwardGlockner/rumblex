import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).parent.parent / "scripts"


@pytest.fixture(scope="module")
def wt():
    pytest.importorskip("whisper_normalizer")
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location("eval_word_trust", SCRIPTS / "eval_word_trust.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(SCRIPTS))
    return mod


def test_align_ops_places_a_deletion_in_its_gap(wt):
    matched, deleted = wt.align_ops(["we", "will", "not", "go", "there"], ["we", "will", "go", "their"])
    assert matched == [0, 1, 3, None]
    assert deleted == [(2, 2)]


def test_align_ops_with_words_missing_at_both_ends(wt):
    matched, deleted = wt.align_ops(["so", "it", "works", "now"], ["it", "works"])
    assert matched == [1, 2]
    assert deleted == [(0, 0), (3, 2)]
    assert wt.align_ops([], ["a"]) == ([None], [])


def test_utterance_row_labels_words_and_gaps(wt):
    words = [{"text": t, "dec_mean": 0.9} for t in ("We", "will", "go", "their.")]
    ctc = [{"text": t, "conf": c} for t, c in (("we", 0.9), ("will", 0.8), ("not", 0.6), ("go", 0.9))]
    row = wt.utterance_row(words, ctc, "we will not go there")
    assert [w["wrong"] for w in row["words"]] == [False, False, False, True]
    assert [w["ctc_word_match"] for w in row["words"]] == [1.0, 1.0, 1.0, 0.0]
    assert row["deleted"] == [["not", 2]]
    assert row["ctc_extra"] == [["not", 2, 0.6]]
    stats = wt.deletion_stats([row], 0.5)
    assert stats["precision"] == 1.0 and stats["recall"] == 1.0
    assert wt.deletion_stats([row], 0.7)["gaps_flagged"] == 0


def test_auc_counts_ties_as_half(wt):
    wrong = np.array([True, False, True, False])
    assert wt.auc(np.array([0.1, 0.9, 0.2, 0.8]), wrong) == 1.0
    assert wt.auc(np.array([0.5, 0.5, 0.5, 0.5]), wrong) == 0.5
