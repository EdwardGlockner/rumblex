"""Audio loading: any file ffmpeg can read (wav, mp3, m4a, mp4, mkv, ...) or an in-memory array,
returned as 16 kHz mono float32 numpy."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

SAMPLE_RATE = 16_000

AudioInput = str | os.PathLike | np.ndarray | torch.Tensor


class AudioLoadError(RuntimeError):
    pass


def find_ffmpeg() -> str | None:
    """`RUMBLEX_FFMPEG` if set, else whatever `ffmpeg` is on PATH."""
    env = os.environ.get("RUMBLEX_FFMPEG")
    if env:
        return env
    return shutil.which("ffmpeg")


def load_audio(source: AudioInput, *, sr: int | None = None) -> np.ndarray:
    """Return `source` as a 1-D float32 array at 16 kHz.

    `source` is a path, or an array/tensor of samples. For arrays, `sr` is their sample rate
    (default 16 kHz); 2-D arrays are downmixed, with the shorter axis taken as channels.
    """
    if isinstance(source, (np.ndarray, torch.Tensor)):
        return _from_array(source, sr if sr is not None else SAMPLE_RATE)
    if sr is not None:
        raise ValueError("sr is only meaningful for array input; files are resampled by ffmpeg")
    return _from_file(Path(source))


def _from_array(x: np.ndarray | torch.Tensor, sr: int) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.dtype in (torch.bfloat16, torch.float16):  # numpy has no bf16
            x = x.float()
        x = x.numpy()
    x = np.asarray(x)
    if x.dtype == np.uint8:  # 8-bit PCM is unsigned, centred on 128
        x = (x.astype(np.float32) - 128) / 128
    elif np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    x = x.astype(np.float32, copy=False)
    if x.ndim == 2:
        x = x.mean(axis=0 if x.shape[0] <= x.shape[1] else 1)
    elif x.ndim != 1:
        raise ValueError(f"expected 1-D or 2-D audio, got shape {x.shape}")
    if sr != SAMPLE_RATE:
        x = _resample(x, sr)
    return np.ascontiguousarray(x)


def _resample(x: np.ndarray, sr: int) -> np.ndarray:
    import torchaudio.functional as F

    return F.resample(torch.from_numpy(x), sr, SAMPLE_RATE).numpy()


def _from_file(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    ffmpeg = find_ffmpeg()
    if ffmpeg is not None:
        return _decode_ffmpeg(ffmpeg, path)
    # No ffmpeg: libsndfile covers wav/flac/ogg/mp3, but not video containers or m4a.
    try:
        import soundfile as sf

        data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as e:
        raise AudioLoadError(
            f"ffmpeg not found and soundfile can't read {path.name} ({e}). "
            "Install ffmpeg or set RUMBLEX_FFMPEG to its path."
        ) from e
    return _from_array(data.T, file_sr)


def _decode_ffmpeg(ffmpeg: str, path: Path) -> np.ndarray:
    cmd = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "0",
        "-i", str(path),
        "-vn", "-map", "0:a:0",  # first audio stream only, skip video
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-acodec", "pcm_f32le", "-",
    ]  # fmt: skip
    proc = subprocess.run(cmd, check=False, capture_output=True)
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace").strip().splitlines()
        raise AudioLoadError(f"ffmpeg failed on {path.name}: {' | '.join(err[-3:])}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise AudioLoadError(f"no audio decoded from {path.name}")
    return audio.copy()  # frombuffer is read-only
