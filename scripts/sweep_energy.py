"""Sweep the loudness refinement of word times on LibriSpeech dev_clean; report the setting chosen on dev
(lowest start MAE + end MAE) on test_clean, next to the unrefined baseline.

Base settings are the shipped ones: block frame grid, next_start ends, start shift -22 ms. Needs the
states from scripts/collect_alignment_states.py and the LibriSpeech alignment parquet (rumblex.bench).
The committed output is docs/alignment_energy_sweep.json.

    python scripts/sweep_energy.py --states data/align_states --out docs/alignment_energy_sweep.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from eval_alignment import align_chunks  # same folder: run from the repo root with scripts/ importable

from rumblex.aligner import CTCHead, LoudnessRefinement, word_times
from rumblex.audio import SAMPLE_RATE
from rumblex.bench import TimedWord, TimingScore, iter_librispeech, score_timings

GRID, RULE, SHIFT = "block", "next_start", -0.022


def load(states: Path, split: str, head: CTCHead, tok):
    utts = torch.load(states / f"{split}.pt", weights_only=False)
    audio = {u.id: u.audio for u in iter_librispeech(split, ids=[u["id"] for u in utts])}
    align_chunks(utts, head, tok)
    return utts, audio


def score(utts, audio, cfg: dict | None) -> dict:
    total = TimingScore()
    for u in utts:
        hyp = []
        for c in u["chunks"]:
            if not c["aligned"]:
                continue
            words, n_frames = c["aligned"]
            loudness = None
            if cfg is not None:
                a0, a1 = round(c["start"] * SAMPLE_RATE), round(c["end"] * SAMPLE_RATE)
                loudness = LoudnessRefinement(audio[u["id"]][a0:a1], a0 / SAMPLE_RATE, **cfg)
            s, e = word_times(words, n_frames, c["start"], c["regions"], GRID, RULE, SHIFT, loudness)
            hyp += [TimedWord(w.text, float(x), float(y)) for w, x, y in zip(words, s, e)]
        total.add(score_timings([TimedWord(*w) for w in u["ref_words"]], hyp))
    return total.summary()


def row(name: str, s: dict) -> str:
    return (
        f"{name:34s} start MAE {s['start_mae_ms']:5.1f} med {s['start_median_ae_ms']:4.1f} "
        f"<=100 {100 * s['start_within_100ms']:4.1f}% | end MAE {s['end_mae_ms']:5.1f} "
        f"med {s['end_median_ae_ms']:4.1f} signed {s['end_mean_signed_ms']:+5.1f} "
        f"<=50 {100 * s['end_within_50ms']:4.1f}% <=100 {100 * s['end_within_100ms']:4.1f}%"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--states", type=Path, required=True, help="folder with dev_clean.pt and test_clean.pt")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoProcessor

    tok = AutoProcessor.from_pretrained("Reson8/Rumble-2-Flash").tokenizer
    head = CTCHead.from_pretrained(device="cpu")

    dev, dev_audio = load(args.states, "dev_clean", head, tok)
    out: dict = {"dev": {"baseline": score(dev, dev_audio, None)}}
    print(row("dev baseline", out["dev"]["baseline"]))
    cands = []
    for starts in (False, True):
        for db in (20, 25, 30, 35):
            for ms in (0.03, 0.05, 0.08):
                cfg = {"silence_db": db, "min_silence": ms, "starts": starts, "ends": True}
                s = score(dev, dev_audio, cfg)
                name = f"{'starts+ends' if starts else 'ends'} db={db} min={int(ms * 1000)}ms"
                print(row("dev " + name, s))
                out["dev"][name] = s
                cands.append((s["start_mae_ms"] + s["end_mae_ms"], name, cfg))
    best = min(cands, key=lambda c: c[0])
    best_ends = min((c for c in cands if not c[2]["starts"]), key=lambda c: c[0])
    print("chosen on dev:", best[1], "| best ends-only:", best_ends[1])

    test, test_audio = load(args.states, "test_clean", head, tok)
    out["test"] = {
        "baseline": score(test, test_audio, None),
        best_ends[1]: score(test, test_audio, best_ends[2]),
        best[1]: score(test, test_audio, best[2]),
    }
    for k, v in out["test"].items():
        print(row("test " + k, v))
    out["chosen"] = {"name": best[1], "cfg": best[2]}
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
