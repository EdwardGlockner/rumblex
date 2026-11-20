import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from rumblex import audio as A
from rumblex.audio import SAMPLE_RATE, AudioLoadError, load_audio

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
needs_ffmpeg = pytest.mark.skipif(A.find_ffmpeg() is None, reason="ffmpeg not on PATH")


def _tone(sr: int, seconds: float, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_array_passthrough():
    x = _tone(SAMPLE_RATE, 1.0)
    y = load_audio(x)
    assert y.dtype == np.float32 and y.shape == x.shape
    np.testing.assert_array_equal(x, y)


def test_array_resample_and_downmix():
    left = _tone(44_100, 2.0)
    stereo = np.stack([left, left])  # (channels, samples)
    y = load_audio(stereo, sr=44_100)
    assert y.ndim == 1 and abs(len(y) - 2 * SAMPLE_RATE) <= 1
    y2 = load_audio(stereo.T, sr=44_100)  # (samples, channels)
    np.testing.assert_allclose(y, y2)


def test_int16_and_tensor_input():
    x = (_tone(SAMPLE_RATE, 0.5) * 32767).astype(np.int16)
    y = load_audio(x)
    assert y.dtype == np.float32 and np.abs(y).max() <= 1.0
    z = load_audio(torch.from_numpy(_tone(SAMPLE_RATE, 0.5)))
    assert z.shape == (SAMPLE_RATE // 2,)


def test_sr_rejected_for_files(tmp_path):
    p = tmp_path / "a.wav"
    sf.write(p, _tone(SAMPLE_RATE, 0.1), SAMPLE_RATE)
    with pytest.raises(ValueError):
        load_audio(p, sr=8000)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_audio("does/not/exist.wav")


def test_soundfile_fallback_without_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "find_ffmpeg", lambda: None)
    p = tmp_path / "stereo_44k.wav"
    sf.write(p, np.stack([_tone(44_100, 1.0)] * 2, axis=1), 44_100)
    y = load_audio(p)
    assert abs(len(y) - SAMPLE_RATE) <= 1


@needs_ffmpeg
def test_wav_via_ffmpeg_matches_soundfile(tmp_path):
    p = tmp_path / "mono16k.wav"
    x = _tone(SAMPLE_RATE, 1.0)
    sf.write(p, x, SAMPLE_RATE, subtype="FLOAT")
    y = load_audio(p)
    np.testing.assert_allclose(y, x, atol=1e-6)


@needs_ffmpeg
@pytest.mark.parametrize("name", ["bcn_weather.mp3", "librispeech_mr_quilter.wav", "two_speakers.wav"])
def test_sample_files(name):
    if not (SAMPLES / name).exists():
        pytest.skip("samples/ not present")
    y = load_audio(SAMPLES / name)
    assert y.dtype == np.float32 and y.ndim == 1 and len(y) > SAMPLE_RATE
    assert 0.01 < np.abs(y).max() <= 1.0


@needs_ffmpeg
@pytest.mark.parametrize("ext", ["mp4", "mkv", "m4a"])
def test_containers(tmp_path, ext):
    """Video/AAC containers: build one from a 44.1 kHz stereo wav (plus a video track for mp4/mkv)."""
    src = tmp_path / "src.wav"
    sf.write(src, np.stack([_tone(44_100, 2.0)] * 2, axis=1), 44_100)
    out = tmp_path / f"clip.{ext}"
    video = [] if ext == "m4a" else ["-f", "lavfi", "-i", "color=c=black:s=64x64:d=2"]
    cmd = [A.find_ffmpeg(), "-y", "-loglevel", "error", *video, "-i", str(src), "-c:a", "aac", "-shortest"]
    subprocess.run([*cmd, str(out)], check=True)
    y = load_audio(out)
    assert abs(len(y) / SAMPLE_RATE - 2.0) < 0.1  # AAC adds priming/padding samples


@needs_ffmpeg
def test_video_without_audio(tmp_path):
    out = tmp_path / "silent.mp4"
    subprocess.run(
        [
            A.find_ffmpeg(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:d=1",
            str(out),
        ],
        check=True,
    )
    with pytest.raises(AudioLoadError):
        load_audio(out)


def test_half_precision_tensors_and_uint8():
    x = _tone(SAMPLE_RATE, 0.25)
    for dt in (torch.bfloat16, torch.float16):
        y = load_audio(torch.from_numpy(x).to(dt))
        assert y.dtype == np.float32 and np.abs(y - x).max() < 0.01
    u8 = np.array([0, 128, 255], np.uint8)
    np.testing.assert_allclose(load_audio(u8), [-1.0, 0.0, 127 / 128])
