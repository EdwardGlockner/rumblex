"""OpenAI-compatible transcription server: `rumblex serve` (needs the [server] extra).

`POST /v1/audio/transcriptions` takes the multipart form of OpenAI's audio API (file, model, language,
prompt, response_format, temperature, timestamp_granularities[]) and answers in its formats: json, text,
srt, vtt, verbose_json, and diarized_json when the server runs with diarization. The official `openai`
client, or anything else written against that API, works by pointing its base URL here.

`model` is accepted but the server uses the model it was started with; `temperature` is ignored (decoding
is greedy) and `stream` isn't supported. In verbose_json, words also carry rumblex's `score` and
`probability`, and words and segments carry `speaker` with diarization. A segment's `avg_logprob` is the
mean of its words' mean token log-probs, `no_speech_prob` is 0 (the VAD drops non-speech before decoding)
and `seek` is 0. The form fields `digits`, `filler` and `verbatim` set the model's options as on the
command line.

Requests run on the model one at a time, in a worker thread, so the server keeps answering meanwhile.
"""

# No `from __future__ import annotations`: FastAPI reads the handler's annotations at runtime.

import io
import math
import secrets
import tempfile
import threading
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rumblex import __version__
from rumblex.types import Segment, TranscriptResult, WordSegment
from rumblex.writers import WriterOptions, write_srt, write_vtt

RESPONSE_FORMATS = ("json", "text", "srt", "vtt", "verbose_json", "diarized_json")


class RequestError(ValueError):
    """A bad request: answered with HTTP 400 in OpenAI's error format."""

    def __init__(self, message: str, param: str | None = None):
        super().__init__(message)
        self.param = param


@dataclass
class Pipeline:
    """The loaded models, and one lock so requests don't run on them at the same time."""

    transcriber: Any
    aligner: Any = None
    diarizer: Any = None
    chunk_size: float = 20.0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def run(self, wav, *, language=None, prompt=None, digits=True, filler=False, verbatim=True):
        with self.lock:
            result = self.transcriber.transcribe(
                wav,
                max_chunk_s=self.chunk_size,
                language=language,
                prompt=prompt,
                digits=digits,
                filler=filler,
                verbatim=verbatim,
                keep_encoder_states=self.aligner is not None,
            )
            if self.aligner is not None:
                self.aligner.align(result, wav)
            if self.diarizer is not None:
                from rumblex.diarizer import assign_speakers

                assign_speakers(result, self.diarizer(wav))
        result.metadata.pop("chunk_transcripts", None)  # encoder states, not for the response
        return result


def compression_ratio(text: str) -> float:
    """Whisper's repetition check: text bytes over zlib-compressed bytes."""
    b = text.encode("utf-8")
    return len(b) / len(zlib.compress(b)) if b else 0.0


def _avg_logprob(seg: Segment) -> float:
    lps = [math.log(max(w.probability, 1e-10)) for w in seg.words if w.probability is not None]
    return sum(lps) / len(lps) if lps else 0.0


def _word(w: WordSegment) -> dict[str, Any]:
    d: dict[str, Any] = {"word": w.word, "start": w.start, "end": w.end, "score": w.score}
    if w.probability is not None:
        d["probability"] = w.probability
    if w.speaker is not None:
        d["speaker"] = w.speaker
    return d


def _duration(result: TranscriptResult) -> float:
    return round(float(result.metadata.get("duration", 0.0)), 3)


def verbose_json(
    result: TranscriptResult, granularities: set[str], encode: Callable[[str], list[int]] | None = None
) -> dict[str, Any]:
    """OpenAI's verbose_json. Segments are listed unless only word timestamps were asked for."""
    duration = _duration(result)
    out: dict[str, Any] = {
        "task": "transcribe",
        "language": (result.language or "").lower(),
        "duration": duration,
        "text": result.text,
    }
    if not granularities or "segment" in granularities:
        out["segments"] = [
            {
                "id": i,
                "seek": 0,
                "start": s.start,
                "end": s.end,
                "text": " " + s.text.strip(),
                "tokens": encode(s.text.strip()) if encode else [],
                "temperature": 0.0,
                "avg_logprob": round(_avg_logprob(s), 4),
                "compression_ratio": round(compression_ratio(s.text.strip()), 4),
                "no_speech_prob": 0.0,
                **({"speaker": s.speaker} if s.speaker is not None else {}),
            }
            for i, s in enumerate(result.segments)
        ]
    if "word" in granularities:
        out["words"] = [_word(w) for s in result.segments for w in s.words]
    out["usage"] = {"type": "duration", "seconds": duration}
    return out


def diarized_json(result: TranscriptResult) -> dict[str, Any]:
    return {
        "task": "transcribe",
        "duration": _duration(result),
        "text": result.text,
        "segments": [
            {
                "type": "transcript.text.segment",
                "id": f"seg_{i}",
                "start": s.start,
                "end": s.end,
                "text": s.text.strip(),
                "speaker": s.speaker or "unknown",
            }
            for i, s in enumerate(result.segments)
        ],
        "usage": {"type": "duration", "seconds": _duration(result)},
    }


def subtitles(result: TranscriptResult, fmt: str) -> str:
    buf = io.StringIO()
    (write_srt if fmt == "srt" else write_vtt)(result, buf, WriterOptions())
    return buf.getvalue()


def _flag(form, name: str, default: bool) -> bool:
    v = form.get(name)
    if v is None or v == "":
        return default
    s = str(v).strip().lower()
    if s in ("true", "1"):
        return True
    if s in ("false", "0"):
        return False
    raise RequestError(f"{name} must be true or false, got {v!r}", name)


def parse_request(form, has_diarizer: bool) -> dict[str, Any]:
    """Options from the form fields; raises RequestError."""
    fmt = str(form.get("response_format") or "json")
    if fmt not in RESPONSE_FORMATS:
        raise RequestError(f"response_format must be one of {', '.join(RESPONSE_FORMATS)}", "response_format")
    if fmt == "diarized_json" and not has_diarizer:
        raise RequestError("diarized_json needs the server to run with --diarize", "response_format")
    if _flag(form, "stream", False):
        raise RequestError("streaming is not supported", "stream")
    grans = {
        str(g) for key in ("timestamp_granularities[]", "timestamp_granularities") for g in form.getlist(key)
    }
    if grans - {"word", "segment"}:
        raise RequestError("timestamp_granularities must be word and/or segment", "timestamp_granularities")
    temperature = form.get("temperature")
    if temperature not in (None, ""):
        try:
            float(temperature)  # accepted for compatibility; decoding is greedy
        except ValueError:
            raise RequestError("temperature must be a number", "temperature") from None
    return {
        "response_format": fmt,
        "granularities": grans,
        "language": (str(form.get("language") or "").strip() or None),
        "prompt": (str(form.get("prompt") or "") or None),
        "digits": _flag(form, "digits", True),
        "filler": _flag(form, "filler", False),
        "verbatim": _flag(form, "verbatim", True),
    }


def create_app(pipeline: Pipeline, *, model_name: str, api_key: str | None = None):
    """The FastAPI app. `api_key`, when set, is required as `Authorization: Bearer <key>` on /v1/."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, PlainTextResponse
    from starlette.concurrency import run_in_threadpool
    from starlette.datastructures import UploadFile

    from rumblex.audio import AudioLoadError, load_audio

    app = FastAPI(title="rumblex", version=__version__, docs_url=None, redoc_url=None)
    tok = getattr(pipeline.transcriber, "tokenizer", None)
    encode = (lambda t: tok.encode(t, add_special_tokens=False)) if tok is not None else None

    def error(status: int, message: str, param: str | None = None, kind: str = "invalid_request_error"):
        body = {"error": {"message": message, "type": kind, "param": param, "code": None}}
        return JSONResponse(body, status_code=status)

    @app.middleware("http")
    async def check_key(request: Request, call_next):
        if api_key and request.url.path.startswith("/v1/"):
            sent = request.headers.get("authorization", "")
            if not secrets.compare_digest(sent.encode(), f"Bearer {api_key}".encode()):
                return error(401, "missing or wrong API key", kind="authentication_error")
        return await call_next(request)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_name, "version": __version__}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "created": 0,
                                            "owned_by": "rumblex"}]}  # fmt: skip

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(request: Request):
        try:
            form = await request.form()
        except Exception:  # noqa: BLE001 - not a form body
            return error(400, "expected a multipart/form-data body")
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            return error(400, "missing file", "file")
        try:
            opts = parse_request(form, pipeline.diarizer is not None)
        except RequestError as e:
            return error(400, str(e), e.param)
        data = await upload.read()
        try:
            wav = await run_in_threadpool(decode_upload, data, upload.filename, load_audio)
        except AudioLoadError as e:
            return error(400, f"could not decode the audio: {e}", "file")
        run = {k: opts[k] for k in ("language", "prompt", "digits", "filler", "verbatim")}
        try:
            result = await run_in_threadpool(lambda: pipeline.run(wav, **run))
        except ValueError as e:  # e.g. an unknown language
            return error(400, str(e))
        except Exception as e:  # noqa: BLE001 - reported to the client, the server keeps running
            return error(500, f"{type(e).__name__}: {e}", kind="server_error")
        fmt = opts["response_format"]
        if fmt == "text":
            return PlainTextResponse(result.text + "\n")
        if fmt in ("srt", "vtt"):
            return PlainTextResponse(subtitles(result, fmt))
        if fmt == "verbose_json":
            return verbose_json(result, opts["granularities"], encode)
        if fmt == "diarized_json":
            return diarized_json(result)
        return {"text": result.text, "usage": {"type": "duration", "seconds": _duration(result)}}

    return app


def decode_upload(data: bytes, filename: str | None, load_audio):
    """Decode uploaded bytes through a temporary file (ffmpeg reads from a path)."""
    suffix = "".join(c for c in Path(filename or "").suffix[:16] if c.isalnum() or c == ".")
    fd, name = tempfile.mkstemp(prefix="rumblex-", suffix=suffix)
    try:
        with open(fd, "wb") as f:
            f.write(data)
        return load_audio(name)
    finally:
        Path(name).unlink(missing_ok=True)


def serve(pipeline: Pipeline, *, host: str, port: int, model_name: str, api_key: str | None = None) -> None:
    import uvicorn

    uvicorn.run(create_app(pipeline, model_name=model_name, api_key=api_key), host=host, port=port)
