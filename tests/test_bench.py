from pathlib import Path

import pytest

from rumblex.bench import (
    ALIGNMENT_REPO,
    TimedWord,
    iter_librispeech,
    match_words,
    normalize_word,
    score_timings,
)


def T(w, s, e):
    return TimedWord(w, s, e)


@pytest.mark.parametrize(
    ("w", "out"), [("Hello,", "hello"), ("DON'T", "don't"), ("'quoted'", "quoted"), ("...", "")]
)
def test_normalize_word(w, out):
    assert normalize_word(w) == out


def test_match_words_with_errors():
    ref = ["the", "cat", "sat", "on", "the", "mat"]
    hyp = ["the", "cat", "sit", "on", "mat", "today"]
    assert match_words(ref, hyp) == [(0, 0), (1, 1), (3, 3), (5, 4)]
    assert match_words([], hyp) == [] and match_words(ref, []) == []


def test_score_timings_signed_errors():
    ref = [T("the", 0.5, 0.66), T("dull", 0.66, 0.95), T("light", 0.95, 1.41)]
    hyp = [T("The", 0.45, 0.70), T("doll", 0.70, 0.90), T("light.", 1.00, 1.30)]
    s = score_timings(ref, hyp)
    assert (s.n_ref, s.n_hyp, s.n_matched) == (3, 3, 2)  # "dull"/"doll" not timed
    assert s.start_err_ms == pytest.approx([-50, 50])
    assert s.end_err_ms == pytest.approx([40, -110])
    summ = s.summary()
    assert summ["start_mae_ms"] == 50 and summ["end_mae_ms"] == 75
    assert summ["start_within_50ms"] == 1.0 and summ["end_within_50ms"] == 0.5
    assert summ["start_mean_signed_ms"] == 0


def test_score_ignores_punctuation_only_tokens_and_accumulates():
    s = score_timings([T("a", 0, 1)], [T("—", 0, 0.1), T("a", 0.02, 1)])
    assert (s.n_hyp, s.n_matched) == (1, 1)
    total = s.add(score_timings([T("b", 1, 2)], [T("b", 1, 2)]))
    assert total.n_matched == 2 and total.summary()["start_mae_ms"] == 10


def _cached_parquet() -> str | None:
    from huggingface_hub import try_to_load_from_cache

    from rumblex.bench import ALIGNMENT_REVISION

    p = try_to_load_from_cache(
        ALIGNMENT_REPO,
        "data/test_clean-00000-of-00001.parquet",
        revision=ALIGNMENT_REVISION,
        repo_type="dataset",
    )
    return p if isinstance(p, str) and Path(p).exists() else None


@pytest.mark.skipif(_cached_parquet() is None, reason="LibriSpeech alignment parquet not downloaded")
def test_librispeech_rows_are_sane():
    utts = list(iter_librispeech(limit=3, path=_cached_parquet()))
    assert [u.id for u in utts] == sorted(u.id for u in utts)
    for u in utts:
        dur = len(u.audio) / 16000
        assert u.audio.dtype.name == "float32" and 1 < dur < 40
        assert [normalize_word(w.word) for w in u.words] == u.transcript.lower().split()
        assert all(0 <= w.start < w.end <= dur + 0.05 for w in u.words)
        assert all(a.start <= b.start for a, b in zip(u.words, u.words[1:]))
