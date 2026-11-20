from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from rumblex.audio import SAMPLE_RATE, find_ffmpeg, load_audio
from rumblex.vad import SpeechChunk, SpeechRegion, detect_speech, merge_regions, vad_chunks

SR = SAMPLE_RATE
SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def R(a: float, b: float) -> SpeechRegion:
    return SpeechRegion(int(a * SR), int(b * SR))


# --- merge_regions (pure) ---------------------------------------------------------------------------


def test_merge_empty():
    assert merge_regions([]) == []


def test_merge_packs_up_to_max():
    regions = [R(0, 5), R(6, 11), R(12, 17), R(18, 23)]
    chunks = merge_regions(regions, max_chunk_s=20, max_gap_s=2)
    assert [(c.start_s, c.end_s) for c in chunks] == [(0, 17), (18, 23)]
    assert chunks[0].regions == tuple(regions[:3])
    assert all(c.duration <= 20 for c in chunks)


def test_merge_does_not_bridge_long_silence():
    chunks = merge_regions([R(0, 3), R(10, 12)], max_chunk_s=20, max_gap_s=2)
    assert len(chunks) == 2


def test_merge_splits_overlong_region():
    chunks = merge_regions([R(0, 50)], max_chunk_s=20)
    assert len(chunks) == 3
    assert chunks[0].start == 0 and chunks[-1].end == 50 * SR
    assert all(c.duration <= 20 for c in chunks)
    # contiguous, no samples lost
    assert all(a.end == b.start for a, b in pairwise(chunks))


def test_merge_fuses_overlap_and_clamps():
    chunks = merge_regions([R(1, 4), R(3, 6), R(9, 12)], max_chunk_s=20, max_gap_s=5, total_samples=10 * SR)
    assert len(chunks) == 1
    assert chunks[0].regions == (R(1, 6), R(9, 10))
    assert chunks[0].end == 10 * SR


def test_merge_unsorted_input():
    chunks = merge_regions([R(6, 8), R(0, 2)], max_chunk_s=20, max_gap_s=5)
    assert chunks[0].start == 0 and chunks[0].end == 8 * SR


def test_chunk_offsets():
    c = SpeechChunk(int(12.5 * SR), int(20 * SR))
    audio = np.arange(30 * SR, dtype=np.float32)
    piece = c.slice(audio)
    assert len(piece) == int(7.5 * SR) and piece[0] == 12.5 * SR
    assert (c.start_s, c.end_s, c.duration) == (12.5, 20.0, 7.5)


# --- Silero on real audio ---------------------------------------------------------------------------


def test_silence_and_noise_have_no_speech():
    rng = np.random.default_rng(0)
    assert detect_speech(np.zeros(5 * SR, dtype=np.float32)) == []
    assert detect_speech((0.01 * rng.standard_normal(5 * SR)).astype(np.float32)) == []
    assert detect_speech(np.zeros(0, dtype=np.float32)) == []


@pytest.mark.skipif(find_ffmpeg() is None, reason="ffmpeg not on PATH")
@pytest.mark.skipif(not (SAMPLES / "librispeech_mr_quilter.wav").exists(), reason="samples/ not present")
def test_speech_found_at_right_offset():
    """Prepending 3 s of silence must shift the detected speech by 3 s (the clip itself starts with
    ~0.5 s of silence, so compare against the unpadded detection rather than against 3.0)."""
    speech = load_audio(SAMPLES / "librispeech_mr_quilter.wav")
    ref = vad_chunks(speech)
    audio = np.concatenate([np.zeros(3 * SR, np.float32), speech, np.zeros(4 * SR, np.float32)])
    chunks = vad_chunks(audio)
    assert len(chunks) == len(ref) >= 1
    assert chunks[0].start_s == pytest.approx(ref[0].start_s + 3.0, abs=0.05)
    # The unpadded run ends at the file end, so check the end against where the energy stops instead.
    frames = speech[: len(speech) // 160 * 160].reshape(-1, 160)
    db = 10 * np.log10((frames**2).mean(axis=1) + 1e-12)
    speech_end = (len(db) - np.argmax(db[::-1] > db.max() - 30)) / 100
    assert 3.0 + speech_end - 0.1 <= chunks[-1].end_s <= 3.0 + speech_end + 0.5
    assert all(c.duration <= 20 for c in chunks)


@pytest.mark.skipif(find_ffmpeg() is None, reason="ffmpeg not on PATH")
@pytest.mark.skipif(not (SAMPLES / "obama_first_45_secs.mp3").exists(), reason="samples/ not present")
def test_long_audio_chunks_bounded_and_ordered():
    speech = load_audio(SAMPLES / "obama_first_45_secs.mp3")
    gap = np.zeros(5 * SR, np.float32)
    audio = np.concatenate([speech, gap, speech])
    chunks = vad_chunks(audio, max_chunk_s=15)
    assert len(chunks) >= 6  # ~90 s of speech in <=15 s chunks
    assert all(c.duration <= 15 + 1e-6 for c in chunks)
    assert all(a.end <= b.start for a, b in pairwise(chunks))
    # nothing straddles the 5 s silence
    g0, g1 = len(speech), len(speech) + len(gap)
    assert not any(c.start < g0 + SR and c.end > g1 - SR for c in chunks)


@pytest.mark.skipif(not (SAMPLES / "two_speakers.wav").exists(), reason="samples/ not present")
def test_batched_vad_agrees_with_sequential():
    """Batched probabilities feed Silero's own timestamp logic. Speech/non-speech must agree on >= 99.5% of
    10 ms frames and regions must respect the length limit. Where a long stretch of speech has to be split,
    the two may pick different (equally valid) pauses, so split points are not compared."""
    a = load_audio(SAMPLES / "two_speakers.wav")
    b = load_audio(SAMPLES / "librispeech_mr_quilter.wav")
    audio = np.concatenate([a, np.zeros(SR, np.float32), b] * 2)  # ~2 min, above the batching threshold
    seq = detect_speech(audio, batched=False)
    bat = detect_speech(audio, batched=True)

    def mask(regions):
        m = np.zeros(len(audio) // 160 + 1, bool)
        for r in regions:
            m[r.start // 160 : r.end // 160] = True
        return m

    assert (mask(seq) == mask(bat)).mean() >= 0.995
    assert all(r.duration <= 20.0 + 1e-6 for r in bat)
