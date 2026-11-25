"""Transcribe a sample of LibriSpeech's training splits and save the encoder states, as training data for
the boundary head (scripts/boundary_head.py). Same pipeline and file layout as
collect_alignment_states.py, one file per parquet shard, so a stopped run picks up at the next shard.

    python scripts/collect_boundary_train.py --device cuda:0 --out data/boundary_train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from collect_alignment_states import run
from huggingface_hub import hf_hub_download

from rumblex.bench import ALIGNMENT_REPO, ALIGNMENT_REVISION, iter_librispeech, sample_ids
from rumblex.transcriber import DEFAULT_MODEL, RumbleTranscriber

N_SHARDS = {"train_clean_100": 14, "train_clean_360": 49, "train_other_500": 66}
DEFAULT = ["train_clean_360:0", "train_clean_360:16", "train_clean_360:32", "train_other_500:0",
           "train_other_500:33"]  # fmt: skip
BATCH = 100  # utterances transcribed together


def shard_path(split: str, k: int) -> str:
    name = f"data/{split}-{k:05d}-of-{N_SHARDS[split]:05d}.parquet"
    return hf_hub_download(ALIGNMENT_REPO, name, repo_type="dataset", revision=ALIGNMENT_REVISION)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--shards", nargs="+", default=DEFAULT, help="split:shard")
    ap.add_argument("--per-shard", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tr = None
    summary_path = args.out / "collect_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    for spec in args.shards:
        split, k = spec.split(":")
        dest = args.out / f"{split}_{int(k):02d}.pt"
        if dest.exists():
            print("have", dest, flush=True)
            continue
        path = shard_path(split, int(k))
        ids = sample_ids(split, args.per_shard, args.seed, path)
        utts = [
            {"id": u.id, "audio": u.audio, "ref_text": u.transcript,
             "ref_words": [(w.word, w.start, w.end) for w in u.words]}
            for u in iter_librispeech(path=path, ids=ids)
        ]  # fmt: skip
        if tr is None:
            tr = RumbleTranscriber(DEFAULT_MODEL, device=args.device, dtype=args.dtype,
                                   batch_size=args.batch_size)  # fmt: skip
        res, seconds, speech = [], 0.0, 0.0
        for i in range(0, len(utts), BATCH):
            r, dt, s = run(tr, utts[i : i + BATCH], language="English", digits=False)
            res += r
            seconds += dt
            speech += s
            print(f"{spec}: {len(res)}/{len(utts)} utterances", flush=True)
        tmp = dest.with_suffix(".tmp")
        torch.save(res, tmp)
        tmp.replace(dest)
        summary[spec] = {"utterances": len(res), "audio_seconds": round(speech, 1),
                         "transcribe_seconds": round(seconds, 1), "device": args.device,
                         "dtype": str(tr.dtype), "seed": args.seed}  # fmt: skip
        summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(spec, summary[spec], flush=True)


if __name__ == "__main__":
    main()
