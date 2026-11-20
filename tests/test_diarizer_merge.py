from rumblex.diarizer import Turn, assign_speakers, assign_word_speakers, split_by_speaker
from rumblex.types import Segment, TranscriptResult, WordSegment

A, B = "SPEAKER_00", "SPEAKER_01"


def W(word, s, e):
    return WordSegment(word, s, e, 0.9)


def test_word_takes_max_overlap_summed_per_speaker():
    turns = [Turn(0.0, 1.0, A), Turn(1.0, 1.3, B), Turn(1.3, 1.45, A), Turn(1.45, 3.0, B)]
    words = [W("a", 0.2, 0.8), W("b", 0.9, 1.5), W("c", 2.0, 2.5)]
    assign_word_speakers(words, turns)
    # "b": A overlaps 0.1 + 0.15 = 0.25, B overlaps 0.3 + 0.05 = 0.35
    assert [w.speaker for w in words] == [A, B, B]


def test_no_overlap_nearest_within_gap():
    turns = [Turn(0.0, 1.0, A), Turn(5.0, 6.0, B)]
    words = [W("x", 1.3, 1.5), W("y", 3.0, 3.2), W("z", 4.5, 4.8)]
    assign_word_speakers(words, turns, fill_nearest=True, max_gap=1.0)
    assert [w.speaker for w in words] == [A, None, B]
    assign_word_speakers(words, turns, fill_nearest=False)
    assert [w.speaker for w in words] == [None, None, None]
    assign_word_speakers(words, [])
    assert words[0].speaker is None


def test_split_by_speaker_with_gaps():
    words = [W("hi", 0, 0.5), W("there", 0.6, 1), W("yes", 1.2, 1.5), W("ok", 1.6, 2)]
    for w, s in zip(words, [None, A, B, None]):
        w.speaker = s
    segs = split_by_speaker(Segment(0, 2, "hi there yes ok", words))
    assert [(s.speaker, s.text, s.start, s.end) for s in segs] == [
        (A, "hi there", 0, 1),
        (B, "yes ok", 1.2, 2),
    ]
    assert all(w.speaker == s.speaker for s in segs for w in s.words)


def test_single_speaker_keeps_original_text():
    words = [W("Hello,", 0, 0.5), W("world.", 0.6, 1)]
    for w in words:
        w.speaker = A
    (seg,) = split_by_speaker(Segment(0, 1, "Hello, world.", words))
    assert seg.text == "Hello, world." and seg.speaker == A


def test_assign_speakers_on_result():
    res = TranscriptResult(
        [
            Segment(0, 2, "a b", [W("a", 0, 0.9), W("b", 1.1, 2)]),
            Segment(3, 4, "no words here", []),
        ],
        "English",
    )
    out = assign_speakers(res, [Turn(1.0, 2.5, B), Turn(0.0, 1.0, A), Turn(2.9, 4.2, B)])
    assert [(s.speaker, s.text) for s in out.segments] == [(A, "a"), (B, "b"), (B, "no words here")]
    assert out.metadata["speakers"] == [A, B]


def test_many_words_blocks():
    turns = [Turn(float(i), i + 1.0, A if i % 2 == 0 else B) for i in range(5000)]
    words = [W(str(i), i + 0.1, i + 0.6) for i in range(5000)]
    assign_word_speakers(words, turns)
    assert all(w.speaker == (A if i % 2 == 0 else B) for i, w in enumerate(words))


def test_diarizer_wrapper_with_pyannote_output():
    import numpy as np
    import pytest

    core = pytest.importorskip("pyannote.core")
    from rumblex.diarizer import Diarizer

    class Out:
        def __init__(self):
            self.speaker_diarization = core.Annotation()
            self.speaker_diarization[core.Segment(0.0, 2.0)] = "A"
            self.speaker_diarization[core.Segment(1.5, 3.0)] = "B"  # overlap
            self.exclusive_speaker_diarization = core.Annotation()
            self.exclusive_speaker_diarization[core.Segment(0.0, 1.5)] = "A"
            self.exclusive_speaker_diarization[core.Segment(1.5, 3.0)] = "B"

    seen = {}

    def pipeline(file, **kw):
        seen.update(kw, shape=tuple(file["waveform"].shape), sr=file["sample_rate"])
        return Out()

    dz = Diarizer(pipeline=pipeline)
    turns = dz(np.zeros(48000, np.float32), max_speakers=3)
    assert [(t.start, t.end, t.speaker) for t in turns] == [(0.0, 1.5, "A"), (1.5, 3.0, "B")]
    assert seen == {
        "num_speakers": None,
        "min_speakers": None,
        "max_speakers": 3,
        "shape": (1, 48000),
        "sr": 16000,
    }
    assert len(dz(np.zeros(16000, np.float32), exclusive=False)) == 2


def test_telemetry_default_off(monkeypatch):
    from rumblex.diarizer import _disable_pyannote_telemetry

    monkeypatch.delenv("PYANNOTE_METRICS_ENABLED", raising=False)
    _disable_pyannote_telemetry()
    import os

    assert os.environ["PYANNOTE_METRICS_ENABLED"] == "false"
    monkeypatch.setenv("PYANNOTE_METRICS_ENABLED", "true")  # an explicit opt-in is respected
    _disable_pyannote_telemetry()
    assert os.environ["PYANNOTE_METRICS_ENABLED"] == "true"


def test_speakers_lists_only_those_in_the_output():
    res = TranscriptResult([Segment(0, 2, "a b", [W("a", 0, 0.9), W("b", 1.1, 2)])], "English")
    out = assign_speakers(res, [Turn(0.0, 2.0, B), Turn(5.0, 6.0, A)])  # A speaks where nothing was said
    assert out.metadata["speakers"] == [B]


def test_diarizer_hides_pyannote_std_warning():
    import warnings

    import numpy as np

    from rumblex.diarizer import Diarizer

    class Empty:
        def itertracks(self, yield_label=False):
            return iter(())

    def pipeline(file, **kw):
        warnings.warn("std(): degrees of freedom is <= 0. Correction should be strictly less", UserWarning)
        warnings.warn("something else", UserWarning)
        return Empty()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert Diarizer(pipeline=pipeline)(np.zeros(1600, np.float32)) == []
    assert [str(w.message) for w in caught] == ["something else"]
