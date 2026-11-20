"""Check that the vLLM backend's local encoder pass gives the same ln_post states as the transformers
backend captures during generate() (so alignment behaves the same with either backend).

    python scripts/check_encoder_runner.py --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rumblex.audio import SAMPLE_RATE, load_audio
from rumblex.transcriber import RumbleTranscriber, prepare_clip
from rumblex.vad import vad_chunks
from rumblex.vllm_backend import EncoderRunner

SAMPLES = ["librispeech_mr_quilter.wav", "bcn_weather.mp3", "obama_first_45_secs.mp3", "two_speakers.wav"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1] / "samples"
    gap = np.zeros(SAMPLE_RATE, np.float32)
    wav = np.concatenate([x for n in SAMPLES for x in (load_audio(root / n), gap)])
    chunks = vad_chunks(wav)
    tr = RumbleTranscriber(device=args.device, dtype=args.dtype, batch_size=8)
    captured = tr.transcribe_chunks(wav, chunks, keep_encoder_states=True)
    del tr
    enc = EncoderRunner(device=args.device, dtype=args.dtype)
    clips = [prepare_clip(c.slice(wav)) for c in chunks]
    local = [s for b in range(0, len(clips), 8) for s in enc.states(clips[b : b + 8])]
    shapes = [(tuple(a.encoder_states.shape), tuple(b.shape)) for a, b in zip(captured, local)]
    same = [a.encoder_states.shape == b.shape for a, b in zip(captured, local)]
    rep = {
        "chunks": len(chunks),
        "shapes_equal": all(same),
        "shape_mismatches": [s for s, ok in zip(shapes, same) if not ok],
        "max_abs_diff": max(
            (float((a.encoder_states - b).abs().max()) for a, b, ok in zip(captured, local, same) if ok),
            default=None,
        ),
    }
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
