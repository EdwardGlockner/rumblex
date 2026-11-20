"""Batched Silero VAD on GPU vs CPU: time and agreement on --seconds of LibriSpeech test_clean speech.

python scripts/check_vad_gpu.py --device cuda:0 --seconds 1200 --out vad_gpu.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from rumblex.bench import iter_librispeech
from rumblex.vad import detect_speech


def mask(regions, n):
    m = np.zeros(n // 160 + 1, bool)
    for r in regions:
        m[r.start // 160 : r.end // 160] = True
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seconds", type=float, default=1200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    pieces, total = [], 0
    for u in iter_librispeech("test_clean"):
        pieces += [u.audio, np.zeros(8000, np.float32)]
        total += len(u.audio) + 8000
        if total >= args.seconds * 16000:
            break
    audio = np.concatenate(pieces)
    detect_speech(audio[: 16000 * 90], device=args.device)  # warm-up (CUDA init, model to GPU)
    rep = {"audio_seconds": round(len(audio) / 16000, 1), "torch_threads": torch.get_num_threads(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}  # fmt: skip
    regs = {}
    for name, kw in (
        ("cpu_sequential", {"batched": False}),
        ("cpu_batched", {}),
        ("gpu_batched", {"device": args.device}),
    ):
        t0 = time.perf_counter()
        regs[name] = detect_speech(audio, **kw)
        rep[name] = {"seconds": round(time.perf_counter() - t0, 2), "regions": len(regs[name]),
                     "longest_s": round(max(r.duration for r in regs[name]), 2)}  # fmt: skip
        print(name, rep[name], flush=True)
    ref = mask(regs["cpu_sequential"], len(audio))
    for name in ("cpu_batched", "gpu_batched"):
        rep[name]["agreement_vs_sequential"] = round(float((mask(regs[name], len(audio)) == ref).mean()), 5)
    print(json.dumps(rep, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
