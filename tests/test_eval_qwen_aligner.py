import importlib.util
from pathlib import Path

import pytest

from rumblex.bench import normalize_word

SCRIPTS = Path(__file__).parent.parent / "scripts"


@pytest.fixture(scope="module")
def qa():
    spec = importlib.util.spec_from_file_location("eval_qwen_aligner", SCRIPTS / "eval_qwen_aligner.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def job(uid="a", transcript="HELLO BIG WORLD", ref=None, duration=2.0):
    ref = ref or [["HELLO", 0.1, 0.4], ["BIG", 0.5, 0.8], ["WORLD", 0.9, 1.5]]
    return {"id": uid, "transcript": transcript, "duration": duration, "ref_words": ref}


def shifted(j, by, seconds=0.5):
    return {"words": [[w.lower(), s + by, e + by] for w, s, e in j["ref_words"]], "seconds": seconds}


def test_errors_are_paired_by_index_and_skip_unk(qa):
    j = job(ref=[["HELLO", 0.1, 0.4], ["<unk>", 0.5, 0.8], ["WORLD", 0.9, 1.5]])
    res = {"words": [["hello", 0.12, 0.4], ["big", 0.0, 0.0], ["world,", 0.9, 1.45]], "seconds": 0.1}
    e, why = qa.utterance_errors(j, res, normalize_word)
    assert why is None
    assert e["start"] == pytest.approx([20, 0])
    assert e["end"] == pytest.approx([0, -50])
    assert e["zero_duration"] == 0


def test_utterances_that_cannot_be_paired_give_a_reason(qa):
    j = job()
    assert qa.utterance_errors(j, None, normalize_word)[1] == "no output"
    assert qa.utterance_errors(j, {"error": "boom"}, normalize_word)[1] == "failed: boom"
    short = {"words": [["hello", 0.1, 0.4], ["world", 0.9, 1.5]]}
    assert "2 words returned for 3" in qa.utterance_errors(j, short, normalize_word)[1]
    other = {"words": [["hello", 0.1, 0.4], ["bag", 0.5, 0.8], ["world", 0.9, 1.5]]}
    assert qa.utterance_errors(j, other, normalize_word)[1] == "returned words differ from the transcript"
    split = job(ref=[["HELLO", 0.1, 0.4], ["WORLD", 0.9, 1.5]])
    assert (
        qa.utterance_errors(split, shifted(j, 0), normalize_word)[1] == "MFA words differ from the transcript"
    )


def test_times_in_milliseconds_are_refused(qa):
    j = job()
    ms = {"words": [[w, 1000 * s, 1000 * e] for w, s, e in j["ref_words"]]}
    with pytest.raises(ValueError, match="not seconds"):
        qa.utterance_errors(j, ms, normalize_word)


def test_score_uses_only_utterances_both_systems_returned(qa):
    jobs = [job("a"), job("b"), job("c")]
    rumblex = {"meta": {}, "utterances": {j["id"]: shifted(j, 0.02) for j in jobs}}
    qwen = {"meta": {}, "utterances": {j["id"]: shifted(j, -0.05, seconds=1.0) for j in jobs[:2]}}
    qwen["utterances"]["b"]["words"][0][2] = qwen["utterances"]["b"]["words"][0][1]  # a zero-length word
    report = qa.score(jobs, {"rumblex": rumblex, "qwen": qwen}, normalize_word)
    assert report["utterances_scored"] == 2
    assert report["not_scored"] == [{"id": "c", "qwen": "no output"}]
    assert report["rumblex"]["words"] == report["qwen"]["words"] == 6
    assert report["rumblex"]["start_mae_ms"] == 20.0
    assert report["rumblex"]["start_mean_signed_ms"] == 20.0
    assert report["qwen"]["start_mae_ms"] == 50.0
    assert report["qwen"]["starts_within_20ms"] == 0.0
    assert report["qwen"]["zero_duration_words"] == 1
    assert report["rumblex_minus_qwen"]["start_mae"] == {"difference_ms": -30.0, "ci95_ms": [-30.0, -30.0]}
    assert report["rumblex"]["speed"] == {
        "utterances_timed": 3,
        "audio_seconds": 6.0,
        "seconds": 1.5,
        "times_real_time": 4.0,
    }
    assert report["qwen"]["speed"]["utterances_timed"] == 2
