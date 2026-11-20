"""Benchmark plumbing with a fake system (the real systems need a GPU, and WhisperX isn't installed here)."""

import numpy as np
import pytest

from rumblex.bench import (
    System,
    SystemOutput,
    TimedWord,
    Utterance,
    english_normalizer,
    long_audio_from_librispeech,
    markdown_table,
    normalize_text,
    run_accuracy,
    run_speed,
    wer,
)


def utt(i, words, seconds=1.0):
    return Utterance(
        f"u{i}", np.zeros(int(seconds * 16000), np.float32), " ".join(w.word for w in words), words
    )


class Echo(System):
    """Returns the reference words, shifted by `shift` seconds, with one word wrong per utterance."""

    name = "echo"

    def __init__(self, refs, shift=0.05):
        self.refs, self.shift, self.calls = refs, shift, 0

    def run(self, audio, language):
        ws = self.refs[self.calls % len(self.refs)]
        self.calls += 1
        out = [TimedWord(w.word, w.start + self.shift, w.end + self.shift) for w in ws]
        out[0] = TimedWord("WRONG", out[0].start, out[0].end)
        return SystemOutput(" ".join(w.word for w in out), out, {"transcribe": 0.1, "align": 0.02})


def test_run_accuracy_and_table():
    refs = [
        [TimedWord("the", 0.1, 0.3), TimedWord("cat", 0.3, 0.6), TimedWord("sat", 0.6, 0.9)],
        [TimedWord("a", 0.0, 0.2), TimedWord("dog", 0.2, 0.5)],
    ]
    utts = [utt(i, w) for i, w in enumerate(refs)]
    r = run_accuracy(Echo(refs), utts, "English")
    assert r["utterances"] == 2 and r["ref_words"] == 5 and r["sub"] == 2 and r["wer"] == pytest.approx(0.4)
    assert r["timing"]["n_matched"] == 3 and r["timing"]["start_mae_ms"] == pytest.approx(50.0)
    assert r["rtfx_per_utterance"] == pytest.approx(2 / 0.24, rel=0.01)
    s = run_speed(Echo(refs), np.zeros(32000, np.float32), "English")
    assert s["audio_seconds"] == 2.0 and s["stages"] == {"transcribe": 0.1, "align": 0.02}
    table = markdown_table({"echo": {"accuracy": r, "speed": s}})
    assert "| echo | 40.0% | 50.0 ms |" in table


def test_wer_skips_empty_refs_and_normalizes():
    assert wer(["Hello, World!", ""], ["hello world", "noise"])["wer"] == 0.0
    assert normalize_text("Don't STOP—now.") == "don't stop now"


def test_english_normalizer_falls_back():
    name, fn = english_normalizer()
    assert name in ("basic", "whisper-english") and fn("Hello.").strip() == "hello"


def test_long_audio_cycles_to_length():
    utts = [utt(0, [], 1.0), utt(1, [], 0.5)]
    a = long_audio_from_librispeech(utts, 5.0, gap=0.5)
    assert len(a) == 5 * 16000
    assert len(long_audio_from_librispeech([], 3.0)) == 0


def test_normalizer_for_languages():
    from rumblex.bench import normalizer_for

    en_name, _ = normalizer_for("English")
    nl_name, nl = normalizer_for("Dutch")
    assert en_name in ("basic", "whisper-english") and nl_name in ("basic", "whisper-basic")
    assert nl("Het Japanse agentschap, voor kernenergie.").split() == [
        "het",
        "japanse",
        "agentschap",
        "voor",
        "kernenergie",
    ]


def test_markdown_table_missing_values():
    table = markdown_table({"x": {"accuracy": {"wer": 0.1, "timing": {"start_mae_ms": None}},
                                  "speed": {"rtfx": None, "peak_gpu_mb": None}}})  # fmt: skip
    assert "None" not in table and "| n/a | " in table and "n/a MB" not in table
