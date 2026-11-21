"""Rescore saved transcripts with the Open ASR Leaderboard's normalisation and alignment.

Reads the `diag_*.json` files written by `diagnose_wer.py`, or `benchmark.py run --save-transcripts` output, and
scores each variant with the leaderboard's own code: its multilingual normaliser (diacritics kept, numbers
written out with num2words for the given language) and its word alignment, which counts a compound written
split on one side and joined on the other as correct. No GPU or model is needed.

Needs a checkout of https://github.com/huggingface/open_asr_leaderboard and `regex kaldialign num2words`:

    git clone https://github.com/huggingface/open_asr_leaderboard
    python scripts/rescore_leaderboard.py --leaderboard open_asr_leaderboard --language nl \
        --out docs/bench_2026-11-20_rtx4090_diagnose/leaderboard_rescore.json \
        docs/bench_2026-11-20_rtx4090_diagnose/diag_*.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import types
from pathlib import Path


def load_scorer(repo: Path):
    sys.path.insert(0, str(repo))
    # data_utils imports these at module level for loading datasets; scoring never uses them.
    for name in ("datasets", "huggingface_hub"):
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.Audio = stub.IterableDataset = stub.load_dataset = stub.snapshot_download = None
            sys.modules[name] = stub
    from normalizer import eval_utils

    return eval_utils


def score(eval_utils, refs: list[str], hyps: list[str], language: str) -> dict:
    tot = {"sub": 0, "del": 0, "ins": 0, "total": 0, "ref_len": 0}
    for ref, hyp in zip(refs, hyps, strict=True):
        cur = eval_utils._align_pair((ref, hyp, language, True))  # the leaderboard's per-pair scoring
        for k in tot:
            tot[k] += cur[k]
    tot["wer"] = round(tot["total"] / tot["ref_len"], 4)
    return tot


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+", type=Path, help="diag_*.json files from diagnose_wer.py")
    ap.add_argument("--leaderboard", type=Path, required=True, help="open_asr_leaderboard checkout")
    ap.add_argument("--language", required=True, help="language code, e.g. nl")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    eval_utils = load_scorer(args.leaderboard)
    commit = subprocess.run(
        ["git", "-C", str(args.leaderboard), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    result = {"leaderboard_commit": commit, "language": args.language, "files": {}}
    for path in args.files:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "variants" not in data:  # benchmark.py output, run with --save-transcripts
            acc = data["accuracy"]
            data = {
                "normalizer": data["data"]["wer_normalizer"],
                "refs": acc["refs"],
                "variants": {data["system"]: {"wer": acc["wer"], "hyps": acc["hyps"]}},
            }
        rows = {}
        for name, variant in data["variants"].items():
            rows[name] = {"saved_wer": variant["wer"], "leaderboard": score(
                eval_utils, data["refs"], variant["hyps"], args.language
            )}  # fmt: skip
            lb = rows[name]["leaderboard"]
            print(f"{path.name:28s} {name:13s} saved {variant['wer']:.2%}  leaderboard {lb['wer']:.2%}")
        result["files"][path.name] = {"normalizer": data.get("normalizer"), "variants": rows}
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
