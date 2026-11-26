"""Runs Qwen3-ForcedAligner-0.6B over the jobs written by `eval_qwen_aligner.py prepare`.

This file imports nothing from rumblex: `qwen-asr` pins an older transformers than rumblex uses, so it
runs under its own interpreter (a venv with torch and `qwen-asr`), and the scoring is done afterwards by
`eval_qwen_aligner.py score`.

    <qwen venv>/python scripts/qwen_aligner_worker.py data/qwen_fa/jobs.jsonl data/qwen_fa/qwen.json \\
        --device cuda:0 --dtype float32

The output has the same shape as `eval_qwen_aligner.py rumblex` writes: `meta`, and per utterance the
words as [text, start, end] and the seconds the call took. The first utterance is aligned once before
timing starts.
"""

from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def parameter_count(obj) -> int | None:
    """Parameters of the torch module behind the aligner object, if it can be found."""
    import torch

    seen = [obj] + [getattr(obj, name, None) for name in ("model", "aligner", "_model")]
    for m in seen:
        if isinstance(m, torch.nn.Module):
            return sum(p.numel() for p in m.parameters())
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("jobs", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--language", default="English")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    args = ap.parse_args()

    import torch
    from qwen_asr import Qwen3ForcedAligner

    jobs = [json.loads(line) for line in args.jobs.read_text(encoding="utf-8").splitlines() if line.strip()]
    cuda = args.device.startswith("cuda")
    t0 = time.perf_counter()
    # device_map has to be given: without it the weights stay on the CPU
    model = Qwen3ForcedAligner.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), device_map=args.device
    )
    load_s = time.perf_counter() - t0

    def run(job: dict) -> list[list]:
        res = model.align(audio=job["audio_path"], text=job["transcript"], language=args.language)
        items = res[0].items if res else []
        return [[it.text, float(it.start_time), float(it.end_time)] for it in items]

    run(jobs[0])  # warm-up
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    utts: dict = {}
    for k, job in enumerate(jobs):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            words = run(job)
            if cuda:
                torch.cuda.synchronize()
            utts[job["id"]] = {"words": words, "seconds": round(time.perf_counter() - t0, 4)}
        except Exception as e:  # noqa: BLE001 - one bad utterance shouldn't lose the run
            utts[job["id"]] = {"error": str(e)[:300]}
        if (k + 1) % 20 == 0:
            print(f"{k + 1}/{len(jobs)}", flush=True)

    meta = {
        "system": "Qwen3-ForcedAligner-0.6B",
        "model": args.model,
        "language": args.language,
        "device": args.device,
        "dtype": args.dtype,
        "gpu": torch.cuda.get_device_name(args.device) if cuda else None,
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "qwen_asr": package_version("qwen-asr"),
        "parameters": parameter_count(model),
        "load_seconds": round(load_s, 1),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20) if cuda else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"meta": meta, "utterances": utts}), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
