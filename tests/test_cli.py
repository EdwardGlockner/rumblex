import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from rumblex.cli import _load_pipeline, build_parser, main
from rumblex.types import Segment, TranscriptResult, WordSegment


def test_parser_defaults_and_flags():
    a = build_parser().parse_args(["x.wav"])
    assert a.output_format == ["all"] and a.batch_size == 8 and not a.no_align and a.language is None
    a = build_parser().parse_args(["x.wav", "y.mp4", "-f", "srt", "json", "--no_digits", "--language", "nl"])
    assert a.audio == ["x.wav", "y.mp4"] and a.output_format == ["srt", "json"] and a.no_digits
    with pytest.raises(SystemExit):
        build_parser().parse_args(["x.wav", "-f", "docx"])
    a = build_parser().parse_args(["x.wav", "--vllm_url", "http://localhost:8000/v1"])
    assert a.vllm_url == "http://localhost:8000/v1"


class FakeTr:
    device, dtype = "cpu", "float32"

    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kw):
        self.calls.append(kw)
        kw["progress"](1, 1)
        assert isinstance(audio, np.ndarray)
        seg = Segment(0.0, 1.0, "Hello world.", language="English")
        return TranscriptResult([seg], "English", {"duration": 2.0, "chunk_transcripts": ["big"]})


def _wav(path, seconds=2.0):
    sf.write(path, np.zeros(int(16000 * seconds), np.float32), 16000)
    return path


class FakeDiarizer:
    def __init__(self):
        self.kw = None

    def __call__(self, audio, **kw):
        from rumblex.diarizer import Turn

        self.kw = kw
        return [Turn(0.0, 0.45, "SPEAKER_00"), Turn(0.45, 2.0, "SPEAKER_01")]


class FakeAligner:
    def align(self, result, audio=None):
        assert audio is not None  # the CLI passes the decoded audio for word-end refinement
        result.segments[0].words = [WordSegment("Hello", 0.0, 0.4, 0.9), WordSegment("world.", 0.5, 1.0, 0.8)]
        return result


def test_main_end_to_end(tmp_path):
    audio = _wav(tmp_path / "talk.wav")
    tr = FakeTr()
    rc = main([str(audio), "-o", str(tmp_path / "out"), "-f", "srt", "json", "--no_digits", "--prompt", "Reson8", "-q"],
              load_pipeline=lambda args: (tr, FakeAligner(), None))  # fmt: skip
    assert rc == 0
    kw = tr.calls[0]
    assert kw["digits"] is False and kw["prompt"] == "Reson8" and kw["keep_encoder_states"] is True
    d = json.loads((tmp_path / "out" / "talk.json").read_text(encoding="utf-8"))
    assert d["segments"][0]["words"][1]["word"] == "world."
    assert "chunk_transcripts" not in d["metadata"] and "transcribe_s" in d["metadata"]["timings"]
    assert "00:00:00,000 --> 00:00:01,000" in (tmp_path / "out" / "talk.srt").read_text(encoding="utf-8")


def test_main_summary_line(tmp_path, capsys):
    audio = _wav(tmp_path / "talk.wav")
    rc = main(
        [str(audio), "-o", str(tmp_path), "-f", "txt"], load_pipeline=lambda args: (FakeTr(), None, None)
    )
    captured = capsys.readouterr()
    out = " ".join((captured.out + captured.err).split())  # rich may wrap the line
    assert rc == 0 and "2.0s audio, English, 1 segment in" in out


def test_main_no_align_and_failures(tmp_path):
    good, bad = _wav(tmp_path / "a.wav"), tmp_path / "bad.wav"
    bad.write_bytes(b"not audio at all")  # decoding fails; the other file must still be written
    tr = FakeTr()
    rc = main([str(bad), str(good), "-o", str(tmp_path), "-f", "txt", "--no_align", "-q"],
              load_pipeline=lambda args: (tr, None, None))  # fmt: skip
    assert rc == 1  # one file failed, the other was still written
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "Hello world.\n"
    assert tr.calls[0]["keep_encoder_states"] is False


def test_main_missing_file(tmp_path, capsys):
    assert main([str(tmp_path / "nope.wav"), "-q"], load_pipeline=lambda a: SimpleNamespace()) == 2
    out = _err(capsys)  # --quiet hides progress, not errors
    assert "not found" in out and "nope.wav" in out


def test_main_diarize(tmp_path):
    audio = _wav(tmp_path / "talk.wav")
    dz = FakeDiarizer()
    rc = main([str(audio), "-o", str(tmp_path), "-f", "json", "srt", "--diarize", "--num_speakers", "2", "-q"],
              load_pipeline=lambda args: (FakeTr(), FakeAligner(), dz))  # fmt: skip
    assert rc == 0 and dz.kw == {"num_speakers": 2, "min_speakers": None, "max_speakers": None}
    d = json.loads((tmp_path / "talk.json").read_text(encoding="utf-8"))
    assert [(s["speaker"], s["text"]) for s in d["segments"]] == [
        ("SPEAKER_00", "Hello"),
        ("SPEAKER_01", "world."),
    ]
    assert "diarize_s" in d["metadata"]["timings"] and d["metadata"]["speakers"] == [
        "SPEAKER_00",
        "SPEAKER_01",
    ]
    assert "[SPEAKER_01] world." in (tmp_path / "talk.srt").read_text(encoding="utf-8")


def _err(capsys):
    captured = capsys.readouterr()
    return " ".join((captured.out + captured.err).split())


def test_main_directory_input(tmp_path, capsys):
    assert main([str(tmp_path), "-q"], load_pipeline=lambda a: SimpleNamespace()) == 2
    assert "not a file" in _err(capsys)


def test_main_same_stem_inputs_clash(tmp_path, capsys):
    (tmp_path / "sub").mkdir()
    a, b = _wav(tmp_path / "a.wav"), _wav(tmp_path / "sub" / "a.flac")

    def never(args):
        raise AssertionError("must stop before loading")

    assert main([str(a), str(b), "-o", str(tmp_path / "out")], load_pipeline=never) == 2
    out = _err(capsys)
    assert "a.wav" in out and "a.flac" in out and not (tmp_path / "out").exists()


def test_main_rejects_unknown_language(tmp_path, capsys):
    audio = _wav(tmp_path / "talk.wav")

    def never(args):
        raise AssertionError("must stop before loading")

    assert main([str(audio), "--language", "klingon", "-q"], load_pipeline=never) == 2
    assert "unsupported language 'klingon'" in _err(capsys)
    tr = FakeTr()
    rc = main([str(audio), "-o", str(tmp_path), "-f", "txt", "--language", " nl ", "-q"],
              load_pipeline=lambda args: (tr, None, None))  # fmt: skip
    assert rc == 0 and tr.calls[0]["language"] == "nl"


def test_main_load_failure_has_hint_and_no_traceback(tmp_path, capsys):
    audio = _wav(tmp_path / "talk.wav")

    def missing_extra(args):
        raise ImportError("No module named 'pyannote'", name="pyannote")

    assert main([str(audio), "--diarize", "-q"], load_pipeline=missing_extra) == 2
    out = _err(capsys)
    assert 'pip install "rumblex[diarize]"' in out and "Traceback" not in out

    from huggingface_hub.errors import GatedRepoError

    def gated(args):
        e = GatedRepoError.__new__(GatedRepoError)
        e.args = ("Access to model pyannote/x is restricted.",)
        raise e

    assert main([str(audio), "--diarize"], load_pipeline=gated) == 2
    out = _err(capsys)
    assert "hf auth login" in out and "HF_TOKEN" in out


def test_load_pipeline_builds_diarizer_first(monkeypatch):
    import rumblex.diarizer
    import rumblex.transcriber

    def no_pyannote(*a, **kw):
        raise ImportError("No module named 'pyannote'", name="pyannote")

    def too_late(*a, **kw):
        raise AssertionError("the model was loaded before the diarizer")

    monkeypatch.setattr(rumblex.diarizer, "Diarizer", no_pyannote)
    monkeypatch.setattr(rumblex.transcriber, "RumbleTranscriber", too_late)
    args = build_parser().parse_args(["x.wav", "--diarize"])
    with pytest.raises(ImportError):
        _load_pipeline(args)


def test_main_warnings(tmp_path, capsys):
    audio = _wav(tmp_path / "talk.wav")
    rc = main([str(audio), "-o", str(tmp_path), "-f", "txt", "--num_speakers", "2"],
              load_pipeline=lambda args: (FakeTr(), None, None))  # fmt: skip
    assert rc == 0 and "ignored without --diarize" in _err(capsys)
    rc = main([str(audio), "-o", str(tmp_path), "-f", "txt", "--diarize", "--no_align"],
              load_pipeline=lambda args: (FakeTr(), None, FakeDiarizer()))  # fmt: skip
    assert rc == 0 and "per chunk" in _err(capsys)


def test_parser_rejects_non_positive_sizes():
    for flag in ("--batch_size", "--chunk_size", "--max_line_count"):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["x.wav", flag, "0"])
    assert build_parser().parse_args(["x.wav", "--chunk_size", "12.5"]).chunk_size == 12.5


def test_python_dash_m():
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "rumblex", "--version"], check=False, capture_output=True, text=True
    )
    assert out.returncode == 0 and out.stdout.startswith("rumblex ")


def test_cli_import_does_not_load_torch():
    import subprocess
    import sys

    code = "import sys, rumblex.cli; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)
    assert out.stdout.strip() == "False", out.stderr
