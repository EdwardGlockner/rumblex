"""rumblex serve, driven by the official openai client over FastAPI's in-process test client (fake models)."""

import io

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("fastapi")
openai = pytest.importorskip("openai")
from fastapi.testclient import TestClient

from rumblex.cli import serve_main
from rumblex.diarizer import Turn
from rumblex.server import Pipeline, compression_ratio, create_app
from rumblex.types import Segment, TranscriptResult, WordSegment


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [len(w) for w in text.split()]


class FakeTr:
    tokenizer = FakeTokenizer()

    def __init__(self):
        self.calls = []

    def transcribe(self, wav, **kw):
        self.calls.append(kw)
        if kw.get("language") == "xx":
            raise ValueError("unknown language 'xx'")
        words = [
            WordSegment("Hello", 0.1, 0.5, 0.9, probability=0.99),
            WordSegment("world.", 0.6, 1.0, 0.2, probability=0.5, ctc_match=False),
        ]
        seg = Segment(0.1, 1.0, "Hello world.", words, language="English")
        return TranscriptResult([seg], "English", {"duration": len(wav) / 16000, "chunk_transcripts": []})


class FakeDiarizer:
    def __call__(self, wav, **kw):
        return [Turn(0.0, 0.55, "SPEAKER_00"), Turn(0.55, 2.0, "SPEAKER_01")]


def wav_bytes(seconds=2.0):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(16000 * seconds), np.float32), 16000, format="WAV")
    return buf.getvalue()


def client_for(pipeline, api_key=None, key="unused"):
    http = TestClient(create_app(pipeline, model_name="Reson8/Rumble-2-Flash", api_key=api_key))
    return openai.OpenAI(base_url="http://testserver/v1", api_key=key, http_client=http, max_retries=0)


def transcribe(client, **kw):
    return client.audio.transcriptions.create(model="rumble-2-flash", file=("a.wav", wav_bytes()), **kw)


def test_formats_through_the_openai_client():
    tr = FakeTr()
    c = client_for(Pipeline(tr))
    r = transcribe(c, language="en", prompt="Reson8")
    assert r.text == "Hello world."
    assert tr.calls[0]["language"] == "en" and tr.calls[0]["prompt"] == "Reson8"
    assert tr.calls[0]["keep_encoder_states"] is False  # no aligner in this pipeline
    assert transcribe(c, response_format="text").strip() == "Hello world."
    srt = transcribe(c, response_format="srt")
    assert srt.startswith("1\n00:00:00,100 --> 00:00:01,000\nHello world.")
    assert transcribe(c, response_format="vtt").startswith("WEBVTT")


def test_verbose_json_words_and_segments():
    c = client_for(Pipeline(FakeTr()))
    r = transcribe(c, response_format="verbose_json", timestamp_granularities=["word", "segment"])
    assert r.language == "english" and r.duration == 2.0 and r.text == "Hello world."
    assert [(w.word, w.start, w.end) for w in r.words] == [("Hello", 0.1, 0.5), ("world.", 0.6, 1.0)]
    assert r.words[1].probability == 0.5 and r.words[1].score == 0.2  # rumblex's extra fields
    assert r.words[1].ctc_match is False and not hasattr(r.words[0], "ctc_match")
    seg = r.segments[0]
    assert seg.text == " Hello world." and seg.tokens == [5, 6] and seg.temperature == 0.0
    assert seg.avg_logprob == pytest.approx(round((np.log(0.99) + np.log(0.5)) / 2, 4))
    assert seg.compression_ratio == pytest.approx(round(compression_ratio("Hello world."), 4))
    only_words = transcribe(c, response_format="verbose_json", timestamp_granularities=["word"])
    assert only_words.segments is None and len(only_words.words) == 2
    default = transcribe(c, response_format="verbose_json")  # OpenAI's default granularity: segments
    assert default.words is None and len(default.segments) == 1


def test_model_options_as_extra_fields():
    tr = FakeTr()
    c = client_for(Pipeline(tr))
    transcribe(c, extra_body={"digits": "false", "filler": "true", "verbatim": "false"})
    assert (
        tr.calls[0]["digits"] is False and tr.calls[0]["filler"] is True and tr.calls[0]["verbatim"] is False
    )


def test_diarization():
    c = client_for(Pipeline(FakeTr(), diarizer=FakeDiarizer()))
    r = transcribe(c, response_format="diarized_json")
    assert [(s.speaker, s.text) for s in r.segments] == [("SPEAKER_00", "Hello"), ("SPEAKER_01", "world.")]
    v = transcribe(c, response_format="verbose_json", timestamp_granularities=["word"])
    assert [w.speaker for w in v.words] == ["SPEAKER_00", "SPEAKER_01"]
    with pytest.raises(openai.BadRequestError, match="--diarize"):
        transcribe(client_for(Pipeline(FakeTr())), response_format="diarized_json")


def test_bad_requests_get_openai_errors():
    c = client_for(Pipeline(FakeTr()))
    with pytest.raises(openai.BadRequestError, match="response_format") as e:
        transcribe(c, response_format="mp3")
    assert e.value.body["param"] == "response_format"
    with pytest.raises(openai.BadRequestError, match="unknown language"):
        transcribe(c, language="xx")
    with pytest.raises(openai.BadRequestError, match="decode"):
        c.audio.transcriptions.create(model="m", file=("a.wav", b"not audio"))
    with pytest.raises(openai.BadRequestError, match="digits"):
        transcribe(c, extra_body={"digits": "maybe"})
    http = TestClient(create_app(Pipeline(FakeTr()), model_name="m"))
    assert http.post("/v1/audio/transcriptions", data={"model": "m"}).json()["error"]["param"] == "file"


def test_api_key():
    with pytest.raises(openai.AuthenticationError):
        transcribe(client_for(Pipeline(FakeTr()), api_key="s3cret", key="wrong"))
    assert transcribe(client_for(Pipeline(FakeTr()), api_key="s3cret", key="s3cret")).text == "Hello world."
    http = TestClient(create_app(Pipeline(FakeTr()), model_name="m", api_key="s3cret"))
    assert http.get("/health").json()["status"] == "ok"  # health stays open for load balancers


def test_models_endpoint():
    c = client_for(Pipeline(FakeTr()))
    assert [m.id for m in c.models.list()] == ["Reson8/Rumble-2-Flash"]


def test_serve_main_builds_the_pipeline():
    seen = {}

    def run(pipeline, args):
        seen.update(pipeline=pipeline, args=args)

    tr = FakeTr()
    argv = ["--port", "9001", "--no_align", "--chunk_size", "15", "--api_key", "k"]
    assert serve_main(argv, load_pipeline=lambda args: (tr, None, None), run=run) == 0
    assert seen["pipeline"].transcriber is tr and seen["pipeline"].chunk_size == 15.0
    assert seen["args"].port == 9001 and seen["args"].api_key == "k" and seen["args"].no_align
    same_port = ["--vllm_url", "http://localhost:8000/v1", "--port", "8000"]
    assert serve_main(same_port, load_pipeline=None) == 2
