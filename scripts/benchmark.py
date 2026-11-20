"""Compare rumblex with WhisperX on the same audio: WER, word timestamp error, long-file speed, GPU memory.

Run one system per process (no shared GPU memory or warm caches), then merge:

    python scripts/benchmark.py run --system rumblex  --out results/rumblex.json
    python scripts/benchmark.py run --system whisperx --out results/whisperx.json
    python scripts/benchmark.py run --system rumblex-vllm --out results/rumblex_vllm.json   # vllm serve running
    python scripts/benchmark.py report results/*.json > results/table.md

Data: a fixed sample of LibriSpeech test_clean (seed 0) with MFA word alignments for accuracy, and real
LibriSpeech speech concatenated to --long-seconds for speed. One warm-up utterance runs before timing.
WER uses Whisper's English normalizer when `whisper-normalizer` is installed (recorded in the output).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from rumblex.bench import (
    RumbleXSystem,
    RumbleXVLLMSystem,
    WhisperXSystem,
    iter_librispeech,
    load_eval_set,
    long_audio_from_librispeech,
    markdown_table,
    normalizer_for,
    run_accuracy,
    run_speed,
    sample_ids,
)


def environment() -> dict:
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "date": dt.date.today().isoformat(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        here = Path(__file__).resolve().parent
        env["rumblex_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=here, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return env


def make_system(args):
    if args.system == "rumblex":
        return RumbleXSystem(
            device=args.device,
            compute_type=args.compute_type,
            batch_size=args.batch_size or 8,
            max_chunk_s=args.max_chunk,
            digits=args.digits,
        )
    if args.system == "rumblex-vllm":
        return RumbleXVLLMSystem(
            base_url=args.vllm_url,
            concurrency=args.vllm_concurrency,
            device=args.device,
            compute_type=args.compute_type,
            batch_size=args.batch_size or 8,
            max_chunk_s=args.max_chunk,
            digits=args.digits,
        )
    if args.system == "whisperx":
        return WhisperXSystem(
            model=args.whisperx_model,
            device=args.device or "cuda",
            compute_type=args.compute_type or "float16",
            batch_size=args.batch_size or 16,
        )
    raise SystemExit(f"unknown system {args.system}")


def cmd_run(args) -> None:
    if args.dataset == "librispeech":
        args.language = args.language or "English"
        ids = sample_ids(args.split, args.n, args.seed)
        utts = list(iter_librispeech(args.split, ids=ids))
        long_src = list(iter_librispeech(args.split, limit=args.long_utts))
    else:
        utts, set_language = load_eval_set(args.dataset, args.n, args.seed)
        args.language = args.language or set_language
        long_src = utts  # the speed file is made of the set's own audio
    norm_name, norm = normalizer_for(args.language)
    long_audio = long_audio_from_librispeech(long_src, args.long_seconds)

    system = make_system(args)
    t0 = time.perf_counter()
    system.load()
    load_s = time.perf_counter() - t0
    system.run(utts[0].audio, args.language)  # warm-up (CUDA init, lazy model parts)

    result = {
        "system": system.name,
        "settings": {**system.info(), "language": args.language},
        "environment": environment(),
        "data": {
            "dataset": args.dataset,
            "split": args.split if args.dataset == "librispeech" else "test",
            "n": len(utts),
            "seed": args.seed,
            "wer_normalizer": norm_name,
        },
        "load_seconds": round(load_s, 2),
    }
    print(f"[{system.name}] accuracy on {len(utts)} utterances", file=sys.stderr, flush=True)
    if not args.speed_only:
        result["accuracy"] = run_accuracy(system, utts, args.language, norm)
    print(f"[{system.name}] speed on {len(long_audio) / 16000:.0f} s", file=sys.stderr, flush=True)
    result["speed"] = run_speed(system, long_audio, args.language)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: result.get(k) for k in ("system", "load_seconds", "accuracy", "speed")}, indent=1))


def cmd_report(args) -> None:
    results = {}
    for p in args.files:
        r = json.loads(Path(p).read_text(encoding="utf-8"))
        results[Path(p).stem] = r  # file name, so several configs of one system can sit side by side
    print(markdown_table(results))
    print()
    for name, r in results.items():
        env, data = r["environment"], r["data"]
        print(f"- {name}: {r['settings']} | {env.get('gpu', 'cpu')}, torch {env.get('torch')}, "
              f"commit {env.get('rumblex_commit')}, {env.get('date')} | {data}")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--system", required=True, choices=["rumblex", "rumblex-vllm", "whisperx"])
    r.add_argument("--out", required=True)
    r.add_argument("--device", default=None)
    r.add_argument("--compute-type", default=None)
    r.add_argument("--batch-size", type=int, default=None)
    r.add_argument("--whisperx-model", default="large-v3")
    r.add_argument("--vllm-url", default="http://localhost:8000/v1")
    r.add_argument("--vllm-concurrency", type=int, default=32)
    r.add_argument("--language", default=None, help="default: the dataset's language")
    r.add_argument(
        "--dataset", default="librispeech", choices=["librispeech", "ami", "earnings22", "fleurs_nl"]
    )
    r.add_argument("--split", default="test_clean")
    r.add_argument("--n", type=int, default=200)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--long-seconds", type=float, default=3600.0)
    r.add_argument("--long-utts", type=int, default=None, help="limit utterances read for the long file")
    r.add_argument("--max-chunk", type=float, default=20.0, help="rumblex: max seconds per model input")
    r.add_argument("--speed-only", action="store_true", help="skip the per-utterance accuracy pass")
    r.add_argument(
        "--digits", action=argparse.BooleanOptionalAction, default=False,
        help="rumblex: numbers as digits (model default). Off by default here because LibriSpeech spells numbers "
        "out; turn it on for sets whose references use digits and whose normalizer doesn't convert (e.g. Dutch)",
    )  # fmt: skip
    rep = sub.add_parser("report")
    rep.add_argument("files", nargs="+")
    args = ap.parse_args()
    {"run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
