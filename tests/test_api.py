"""The top-level API (rumblex.load_model / transcribe / write_result) with a fake model."""

import numpy as np

import rumblex
from rumblex.types import Segment, TranscriptResult


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, keep_encoder_states=False, **kw):
        self.calls.append((len(audio), keep_encoder_states, kw))
        return TranscriptResult([Segment(0, 1, "hi")], "English", {"chunk_transcripts": ["states"]})


def test_transcribe_reuses_one_aligner(monkeypatch):
    built = []

    class FakeAligner:
        def align(self, result, audio):
            assert audio is not None
            return result

    import rumblex.aligner as A

    monkeypatch.setattr(
        A.Aligner, "from_transcriber", classmethod(lambda cls, m: built.append(m) or FakeAligner())
    )
    m = FakeModel()
    for _ in range(3):
        res = rumblex.transcribe(np.zeros(16000, np.float32), m, language="English")
        assert "chunk_transcripts" not in res.metadata
    assert len(built) == 1  # the CTC head is loaded once, not per call
    assert m.calls[0] == (16000, True, {"language": "English"})


def test_transcribe_without_alignment_and_resampling():
    m = FakeModel()
    rumblex.transcribe(np.zeros(8000, np.float32), m, align=False, sr=8000)
    assert m.calls[0][:2] == (16000, False)  # resampled to 16 kHz, no encoder states kept
