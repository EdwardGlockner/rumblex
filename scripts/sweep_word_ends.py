"""Two refinements of word times on the audio, scored on LibriSpeech: starts moved to a loudness dip near the
CTC start (`dip_window`, `dip_smooth`), and the last word of a chunk allowed to end inside its last CTC frame
(`last_inside`). Both on top of the shipped aligner (blank penalty 5, next_start ends, silence cut).

For each setting the start shift is recalibrated on dev_clean (median signed start error), and the setting
with the lowest start MAE + end MAE on dev is reported once on test_clean next to the shipped one.

    python scripts/sweep_word_ends.py --dev data/align_states/dev_clean.pt \\
        --test data/align_states_cpu/test_clean.pt --out docs/alignment_word_ends_sweep.json
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
from sweep_smoothing import align, load

from rumblex.aligner import CTCHead, LoudnessRefinement, word_times
from rumblex.audio import SAMPLE_RATE
from rumblex.bench import TimedWord, TimingScore, score_timings

PENALTY = 5.0
SHIPPED = {"dip_window": 0.0, "dip_smooth": 1, "last_inside": False}
GRID = [SHIPPED] + [
    {"dip_window": w, "dip_smooth": s, "last_inside": last}
    for w, s, last in product((0.005, 0.01, 0.02, 0.03, 0.04, 0.05), (1, 3, 5, 7, 9, 11, 15), (False, True))
] + [{"dip_window": 0.0, "dip_smooth": 1, "last_inside": True}]  # fmt: skip


def score(utts, audio, cfg: dict, shift: float) -> TimingScore:
    total = TimingScore()
    for u in utts:
        hyp = []
        for c in u["chunks"]:
            if not c["aligned"]:
                continue
            words, n = c["aligned"]
            a0, a1 = round(c["start"] * SAMPLE_RATE), round(c["end"] * SAMPLE_RATE)
            lr = LoudnessRefinement(audio[u["id"]][a0:a1], a0 / SAMPLE_RATE, **cfg)
            s, e = word_times(words, n, c["start"], c["regions"], "block", "next_start", shift, lr)
            hyp += [TimedWord(w.text, float(x), float(y)) for w, x, y in zip(words, s, e)]
        total.add(score_timings([TimedWord(*w) for w in u["ref_words"]], hyp))
    return total


def run(utts, audio, cfg: dict, shift: float | None = None) -> dict:
    if shift is None:  # centre the starts: minus the median signed start error with no shift
        shift = -float(np.median(score(utts, audio, cfg, 0.0).start_err_ms)) / 1000
    return {**cfg, "start_shift_ms": round(1000 * shift, 1), **score(utts, audio, cfg, shift).summary()}


def line(tag: str, r: dict) -> str:
    return (
        f"{tag:5s} dip {1000 * r['dip_window']:3.0f} ms smooth {r['dip_smooth']} last_inside "
        f"{r['last_inside']!s:5s} shift {r['start_shift_ms']:+6.1f} | start MAE {r['start_mae_ms']:5.1f} "
        f"med {r['start_median_ae_ms']:4.1f} | end MAE {r['end_mae_ms']:5.1f} med {r['end_median_ae_ms']:4.1f} "
        f"signed {r['end_mean_signed_ms']:+5.1f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--test", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoProcessor

    tok = AutoProcessor.from_pretrained("Reson8/Rumble-2-Flash").tokenizer
    head = CTCHead.from_pretrained(device="cpu")

    dev, dev_audio = load(args.dev, "dev_clean")
    align(dev, head, tok, 1.0, PENALTY)
    out: dict = {"blank_penalty": PENALTY, "dev": []}
    for cfg in GRID:
        r = run(dev, dev_audio, cfg, -0.010 if cfg is SHIPPED else None)  # shipped: its own -10 ms shift
        out["dev"].append(r)
        print(line("dev", r), flush=True)
    best = min(out["dev"][1:], key=lambda r: r["start_mae_ms"] + r["end_mae_ms"])
    out["chosen"] = {k: best[k] for k in (*SHIPPED, "start_shift_ms")}
    print("chosen on dev:", out["chosen"])

    if args.test:
        test, test_audio = load(args.test, "test_clean")
        align(test, head, tok, 1.0, PENALTY)
        out["test"] = []
        for r in (out["dev"][0], best):
            cfg = {k: r[k] for k in SHIPPED}
            res = run(test, test_audio, cfg, r["start_shift_ms"] / 1000)
            out["test"].append(res)
            print(line("test", res), flush=True)
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
