import importlib.util
import json
from pathlib import Path

import pytest

import rumblex
from rumblex.types import Segment, TranscriptResult, WordSegment

SCRIPT = Path(__file__).parent.parent / "scripts" / "fabench_worker.py"


@pytest.fixture
def worker(monkeypatch):
    spec = importlib.util.spec_from_file_location("fabench_worker", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    calls = []

    def result(words):
        ws = [WordSegment(w, i * 0.5, i * 0.5 + 0.4, 0.9) for i, w in enumerate(words)]
        return TranscriptResult([Segment(0.0, 1.0, " ".join(words), ws)], "English")

    def align(audio, text, model):
        calls.append(("align", audio, text))
        if "bad" in audio:
            raise FileNotFoundError(audio)
        return result(text.split())

    def transcribe(audio, model, **kwargs):
        calls.append(("transcribe", audio, kwargs))
        return result(["heard", "this"])

    monkeypatch.setattr(rumblex, "load_model", lambda model_id, device=None: (model_id, device))
    monkeypatch.setattr(rumblex, "align", align)
    monkeypatch.setattr(rumblex, "transcribe", transcribe)
    return mod, calls


def write_jobs(tmp_path, jobs):
    p = tmp_path / "jobs.jsonl"
    p.write_text("".join(json.dumps(j) + "\n" for j in jobs), encoding="utf-8")
    return str(p)


def test_align_gives_one_record_per_job_and_keeps_going_after_an_error(worker, tmp_path, capsys):
    mod, calls = worker
    jobs = [
        {"item_id": "a", "audio_path": "a.wav", "transcript": "so yknow"},
        {"item_id": "b", "audio_path": "bad.wav", "transcript": "x"},
        {"item_id": "c", "audio_path": "c.wav", "transcript": "okay"},
    ]
    assert mod.main([write_jobs(tmp_path, jobs), "some/model", "cpu", "align"]) == 0
    recs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [r["item_id"] for r in recs] == ["a", "b", "c"]
    assert recs[0]["words"] == [["so", 0.0, 0.4, 0.9], ["yknow", 0.5, 0.9, 0.9]]
    assert "FileNotFoundError" in recs[1]["error"] and "words" not in recs[1]
    assert [c[2] for c in calls] == ["so yknow", "x", "okay"]


def test_transcribe_ignores_the_transcript_and_asks_for_raw_words(worker, tmp_path, capsys):
    mod, calls = worker
    jobs = [{"item_id": "a", "audio_path": "a.wav"}]
    assert mod.main([write_jobs(tmp_path, jobs), "some/model", "cpu", "transcribe"]) == 0
    rec = json.loads(capsys.readouterr().out.splitlines()[0])
    assert [w[0] for w in rec["words"]] == ["heard", "this"]
    options = {"language": "en", "digits": False, "filler": True, "verbatim": True}
    assert calls == [("transcribe", "a.wav", options)]


def test_unknown_task_is_refused(worker, tmp_path):
    mod, _ = worker
    with pytest.raises(SystemExit):
        mod.main([write_jobs(tmp_path, []), "some/model", "cpu", "score"])
