"""Score word-timestamp variants on states saved by scripts/collect_alignment_states.py (CPU only).

For every chunk the CTC head + Viterbi run once; the variants only differ in how frames become times:
  grid       block (c + 0.08 k, the conv geometry) | uniform (i / 13)
  end rule   spike | next_start | midpoint | region   (see rumblex.aligner.END_RULES)
  shift      0, or a constant start shift calibrated on dev_clean (median signed start error).
Calibration uses dev_clean only; test_clean is reported with the dev shift. Also reports WER and
writes the Dutch clips with their aligned words for reading.

    python scripts/eval_alignment.py --states data/align_states
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from rumblex.aligner import (
    BLANK,
    END_RULES,
    CTCHead,
    ctc_viterbi,
    group_words,
    token_spans,
    word_times,
)
from rumblex.bench import TimedWord, TimingScore, score_timings, wer

GRIDS = ("block", "uniform")


def align_chunks(utts: list[dict], head: CTCHead, tok) -> tuple[int, int, float]:
    """Run head + Viterbi once per chunk; stores per-chunk word frames in chunk['aligned']."""
    n_chunks = n_fail = 0
    t0 = time.perf_counter()
    for u in utts:
        for c in u["chunks"]:
            n_chunks += 1
            c["aligned"] = None
            ids = c["token_ids"]
            if not ids:
                continue
            em = head.emissions(c["states"].float(), [BLANK, *ids])
            same = [False] + [a == b for a, b in zip(ids, ids[1:])]  # noqa: RUF007
            path = ctc_viterbi(em, len(ids), same)
            if path is None:
                n_fail += 1
                continue
            spans = token_spans(path, len(ids))
            probs = [float(np.exp(em[a : b + 1, k + 1]).mean()) for k, (a, b) in enumerate(spans)]
            words = group_words(ids, spans, probs, tok.convert_ids_to_tokens(ids),
                                lambda x: tok.decode(x, clean_up_tokenization_spaces=False))  # fmt: skip
            c["aligned"] = (words, len(em))
    return n_chunks, n_fail, time.perf_counter() - t0


def hyp_words(u: dict, grid: str, rule: str, shift: float) -> list[TimedWord]:
    out = []
    for c in u["chunks"]:
        if not c["aligned"]:
            continue
        words, T = c["aligned"]
        starts, ends = word_times(words, T, c["start"], c["regions"], grid, rule, shift)
        out += [TimedWord(w.text, float(s), float(e)) for w, s, e in zip(words, starts, ends)]
    return out


def score_split(utts: list[dict], grid: str, rule: str, shift: float) -> TimingScore:
    total = TimingScore()
    for u in utts:
        ref = [TimedWord(*w) for w in u["ref_words"]]
        total.add(score_timings(ref, hyp_words(u, grid, rule, shift)))
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    args = ap.parse_args()
    root = Path(args.states)

    from transformers import AutoProcessor

    tok = AutoProcessor.from_pretrained("Reson8/Rumble-2-Flash").tokenizer
    head = CTCHead.from_pretrained(device="cpu")
    report: dict = {"collect": json.loads((root / "collect_summary.json").read_text(encoding="utf-8"))}

    splits = {s: torch.load(root / f"{s}.pt", weights_only=False) for s in ("dev_clean", "test_clean")
              if (root / f"{s}.pt").exists()}  # fmt: skip
    for name, utts in splits.items():
        n, fail, secs = align_chunks(utts, head, tok)
        audio = sum(u["duration"] for u in utts)
        report[name] = {
            "utterances": len(utts),
            "chunks": n,
            "viterbi_failures": fail,
            "align_cpu_seconds": round(secs, 2),
            "align_rtfx_cpu": round(audio / secs, 1),
            **wer([u["ref_text"] for u in utts], [" ".join(c["text"] for c in u["chunks"]) for u in utts]),
        }

    # calibrate a start shift per grid on dev only
    shifts = {g: 0.0 for g in GRIDS}
    if "dev_clean" in splits:
        for g in GRIDS:
            errs = score_split(splits["dev_clean"], g, "spike", 0.0).start_err_ms
            shifts[g] = -float(np.median(errs)) / 1000 if errs else 0.0
    report["dev_start_shift_s"] = {g: round(s, 4) for g, s in shifts.items()}

    rows = []
    for name, utts in splits.items():
        for g in GRIDS:
            for rule in END_RULES:
                for shift_name, shift in (("none", 0.0), ("dev", shifts[g])):
                    s = score_split(utts, g, rule, shift).summary()
                    rows.append({"split": name, "grid": g, "end_rule": rule, "shift": shift_name, **s})
    report["variants"] = rows

    if (root / "fleurs_nl.pt").exists():
        nl = torch.load(root / "fleurs_nl.pt", weights_only=False)
        align_chunks(nl, head, tok)
        report["fleurs_nl"] = wer(
            [u["ref_text"] for u in nl], [" ".join(c["text"] for c in u["chunks"]) for u in nl]
        )
        report["fleurs_nl"]["languages"] = sorted({c["language"] or "?" for u in nl for c in u["chunks"]})
        lines = []
        for u in nl:
            lines += [
                f"## {u['id']}",
                f"REF: {u['ref_text']}",
                f"HYP: {' '.join(c['text'] for c in u['chunks'])}",
            ]
            lines += [
                f"  {w.start:7.2f} {w.end:7.2f}  {w.word}"
                for w in hyp_words(u, "block", "region", shifts["block"])
            ]
            lines.append("")
        (root / "fleurs_nl_aligned.txt").write_text("\n".join(lines), encoding="utf-8")

    (root / "eval_alignment.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "variants"}, indent=1))
    keys = ["split", "grid", "end_rule", "shift", "n_matched", "start_mae_ms", "start_median_ae_ms",
            "start_mean_signed_ms", "start_within_50ms", "end_mae_ms", "end_median_ae_ms", "end_within_50ms",
            "end_within_100ms"]  # fmt: skip
    print("| " + " | ".join(keys) + " |\n|" + "---|" * len(keys))
    for r in rows:
        print("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")


if __name__ == "__main__":
    main()
