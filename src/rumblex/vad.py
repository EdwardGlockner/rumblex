"""Silero VAD speech detection and chunking for long recordings.

Pipeline: `detect_speech` finds speech regions (pauses shorter than `min_silence_s` never split a
region, so sentences stay together), then `merge_regions` packs neighbouring regions into chunks of at
most `max_chunk_s`. Chunks keep exact sample offsets so chunk-local times map back to the whole file.

The Silero model ships inside the `silero-vad` package: no download, no token.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import pairwise

import numpy as np
import torch

from rumblex.audio import SAMPLE_RATE


@dataclass(frozen=True)
class SpeechRegion:
    """Half-open sample range [start, end) on the global timeline."""

    start: int
    end: int

    @property
    def duration(self) -> float:
        return (self.end - self.start) / SAMPLE_RATE


@dataclass(frozen=True)
class SpeechChunk:
    """A chunk sent to the model as one input. `start`/`end` are sample offsets in the full file;
    `regions` are the speech regions it contains (global offsets), useful for word end times later."""

    start: int
    end: int
    regions: tuple[SpeechRegion, ...] = field(default_factory=tuple)

    @property
    def start_s(self) -> float:
        return self.start / SAMPLE_RATE

    @property
    def end_s(self) -> float:
        return self.end / SAMPLE_RATE

    @property
    def duration(self) -> float:
        return (self.end - self.start) / SAMPLE_RATE

    def slice(self, audio: np.ndarray) -> np.ndarray:
        return audio[self.start : self.end]


@lru_cache(maxsize=4)
def load_vad_model(device: str = "cpu"):
    """Silero VAD (TorchScript) on `device`. Cached per device; not safe to share across threads (it keeps
    RNN state)."""
    import warnings

    from silero_vad import load_silero_vad

    with warnings.catch_warnings():  # newer torch deprecates torch.jit.load, which silero-vad uses
        warnings.filterwarnings("ignore", message=r".*torch\.jit\.load.*", category=FutureWarning)
        model = load_silero_vad(onnx=False)
    return model if device == "cpu" else model.to(device)


VAD_WINDOW = 512  # samples per Silero window at 16 kHz (what get_speech_timestamps uses)
BATCH_MIN_SECONDS = 60  # shorter audio: the plain sequential pass is fast enough
BATCH_STREAMS = 64  # CPU
GPU_BATCH_STREAMS = 256  # GPU: wider steps are cheap there (0.6 s for 20 min of audio on an RTX 4090)
BATCH_WARMUP_WINDOWS = 256  # 8 s before each stream's own part, to settle the recurrent state (see below)


@torch.inference_mode()
def batched_speech_probs(
    audio: np.ndarray,
    model,
    streams: int = BATCH_STREAMS,
    warmup: int = BATCH_WARMUP_WINDOWS,
    device: str | torch.device = "cpu",
) -> list[float]:
    """Silero speech probability for every 512-sample window, computed `streams` windows per model call.

    Silero is a small recurrent model run one 32 ms window at a time, so a long file costs tens of
    thousands of tiny steps. Here the windows are cut into `streams` contiguous parts that run side by
    side as one batch. Each part starts `warmup` windows early so the recurrent state has settled before
    its own windows; those warm-up outputs are discarded. Same windows (and zero-padded last window) as
    `get_speech_timestamps`.

    Silero's state has a long memory, so this is close to, not identical with, the sequential pass:
    - 20 min of LibriSpeech, 64 streams on 4 CPU threads: same 232 regions, 99.96% of 10 ms frames agree
      on speech/non-speech, 4.8x faster. 256 streams on an RTX 4090: 99.92% agreement, 0.6 s against
      13.5 s for the sequential pass.
    - 2 min of noisy TV speech: 99.87% agreement, but where a long stretch must be cut at 20 s, the two
      passes can pick different pauses (both valid, both within the limit).
    Fully identical output needed ~66 s of warm-up, which leaves little speed-up; `batched=False` gives it.

    Each step's windows are gathered as needed, so memory stays at the size of the audio itself.
    """
    x = np.ascontiguousarray(audio, dtype=np.float32)
    n_win = -(-len(x) // VAD_WINDOW)
    x = np.pad(x, (0, (n_win + 1) * VAD_WINDOW - len(x)))  # plus one all-zero window for padding
    windows = torch.from_numpy(x).view(n_win + 1, VAD_WINDOW)
    streams = max(1, min(streams, n_win // max(1, warmup)))
    own = -(-n_win // streams)
    starts = torch.tensor([max(0, s * own - warmup) for s in range(streams)])
    steps = own + warmup
    model.reset_states()
    probs = torch.empty(steps, streams)
    for t in range(steps):
        idx = (starts + t).clamp(max=n_win)  # past the end: the zero window (output discarded below)
        probs[t] = model(windows[idx].to(device), SAMPLE_RATE).reshape(-1).cpu()
    out = np.empty(n_win, dtype=np.float64)
    for s, a in enumerate(starts.tolist()):
        lo, hi = s * own, min(n_win, (s + 1) * own)
        if lo < hi:
            out[lo:hi] = probs[lo - a : hi - a, s].numpy()
    model.reset_states()
    return out.tolist()


class _ReplayModel:
    """Stands in for the Silero model inside get_speech_timestamps, returning precomputed probabilities
    in order, so Silero's own post-processing (thresholds, padding, max-length splits) is reused as is."""

    def __init__(self, probs: list[float]):
        self.probs, self.i = probs, 0

    def reset_states(self) -> None:
        self.i = 0

    def __call__(self, chunk, sampling_rate):
        p = self.probs[self.i]
        self.i += 1
        return torch.tensor(p)


def detect_speech(
    audio: np.ndarray,
    *,
    threshold: float = 0.5,
    min_speech_s: float = 0.25,
    min_silence_s: float = 0.5,
    speech_pad_s: float = 0.1,
    max_region_s: float = 20.0,
    batched: bool = True,
    device: str | torch.device | None = None,
) -> list[SpeechRegion]:
    """Speech regions in 16 kHz mono `audio`.

    Regions longer than `max_region_s` are split by Silero at the best pause it can find,
    falling back to a hard cut. With `batched`, long audio has its per-window speech probabilities
    computed many streams at a time (see `batched_speech_probs`); Silero's own timestamp logic then runs
    on them unchanged. `device` (e.g. "cuda") runs that batched pass on a GPU.
    """
    from silero_vad import get_speech_timestamps

    if audio.size == 0:
        return []
    model = load_vad_model()
    if batched and len(audio) > BATCH_MIN_SECONDS * SAMPLE_RATE:
        dev = torch.device(device) if device is not None else torch.device("cpu")
        if dev.type == "cpu":
            probs = batched_speech_probs(audio, model)
        else:
            probs = batched_speech_probs(audio, load_vad_model(str(dev)), GPU_BATCH_STREAMS, device=dev)
        model = _ReplayModel(probs)
    stamps = get_speech_timestamps(
        torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)),
        model,
        threshold=threshold,
        sampling_rate=SAMPLE_RATE,
        min_speech_duration_ms=int(min_speech_s * 1000),
        min_silence_duration_ms=int(min_silence_s * 1000),
        speech_pad_ms=int(speech_pad_s * 1000),
        max_speech_duration_s=max_region_s,
    )
    return [SpeechRegion(int(s["start"]), int(s["end"])) for s in stamps]


def merge_regions(
    regions: list[SpeechRegion],
    *,
    max_chunk_s: float = 20.0,
    max_gap_s: float = 2.0,
    total_samples: int | None = None,
) -> list[SpeechChunk]:
    """Greedily pack consecutive regions into chunks no longer than `max_chunk_s`.

    A region joins the current chunk only if the pause before it is at most `max_gap_s`
    (so long silences aren't fed to the model). A single region longer than `max_chunk_s`
    is cut into equal pieces. Chunks are clamped to `[0, total_samples)` when given.
    """
    if max_chunk_s <= 0:
        raise ValueError("max_chunk_s must be positive")
    max_len = round(max_chunk_s * SAMPLE_RATE)
    max_gap = round(max_gap_s * SAMPLE_RATE)

    pieces: list[SpeechRegion] = []
    for r in sorted(regions, key=lambda r: r.start):
        s, e = max(r.start, 0), r.end if total_samples is None else min(r.end, total_samples)
        if e <= s:
            continue
        if pieces and s < pieces[-1].end:  # overlapping (e.g. after padding): fuse
            prev = pieces.pop()
            s, e = prev.start, max(prev.end, e)
        pieces.append(SpeechRegion(s, e))
    pieces = [p for r in pieces for p in _split(r, max_len)]

    chunks: list[SpeechChunk] = []
    cur: list[SpeechRegion] = []
    for r in pieces:
        if cur and r.end - cur[0].start <= max_len and r.start - cur[-1].end <= max_gap:
            cur.append(r)
            continue
        if cur:
            chunks.append(SpeechChunk(cur[0].start, cur[-1].end, tuple(cur)))
        cur = [r]
    if cur:
        chunks.append(SpeechChunk(cur[0].start, cur[-1].end, tuple(cur)))
    return chunks


def _split(r: SpeechRegion, max_len: int) -> list[SpeechRegion]:
    n = r.end - r.start
    if n <= max_len:
        return [r]
    k = math.ceil(n / max_len)
    bounds = [r.start + (n * i) // k for i in range(k + 1)]
    return [SpeechRegion(a, b) for a, b in pairwise(bounds)]


def vad_chunks(
    audio: np.ndarray,
    *,
    max_chunk_s: float = 20.0,
    max_gap_s: float = 2.0,
    threshold: float = 0.5,
    min_speech_s: float = 0.25,
    min_silence_s: float = 0.5,
    speech_pad_s: float = 0.1,
    device: str | torch.device | None = None,
) -> list[SpeechChunk]:
    """Detect speech and pack it into model-sized chunks. See `detect_speech` and `merge_regions`."""
    regions = detect_speech(
        audio,
        threshold=threshold,
        min_speech_s=min_speech_s,
        min_silence_s=min_silence_s,
        speech_pad_s=speech_pad_s,
        max_region_s=max_chunk_s,
        device=device,
    )
    return merge_regions(regions, max_chunk_s=max_chunk_s, max_gap_s=max_gap_s, total_samples=len(audio))
