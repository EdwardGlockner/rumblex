"""Try temperature-smoothing the CTC logits before Viterbi (log_softmax(logits / T)) and score word times on
LibriSpeech. T and the start shift are chosen on dev_clean; the chosen T is reported once on test_clean next to
T = 1 (the shipped aligner).

For each T the start shift is recalibrated on dev (median signed start error), since smoothing can move spikes.
Each T is scored with three end rules: the shipped one (next_start + loudness refinement), next_start alone and
spike (end of the last CTC frame), because the spike rule is the one smoothing could help most.

    python scripts/sweep_smoothing.py --dev data/align_states/dev_clean.pt \
        --test data/align_states_cpu/test_clean.pt --out docs/alignment_smoothing_sweep.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rumblex.aligner import (
    BLANK,
    CTCHead,
    LoudnessRefinement,
    ctc_viterbi,
    group_words,
    token_spans,
    word_times,
)
from rumblex.audio import SAMPLE_RATE
from rumblex.bench import TimedWord, TimingScore, iter_librispeech, score_timings

TEMPS = (1.0, 1.5, 2.0, 3.0, 5.0, 8.0)
PENALTIES = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0)  # subtracted from the blank log-prob, at T = 1
RULES = {"shipped": ("next_start", True), "next_start": ("next_start", False), "spike": ("spike", False)}


@torch.inference_mode()
def emissions(
    head: CTCHead, states: torch.Tensor, classes: list[int], temp: float, penalty: float
) -> np.ndarray:
    logits = torch.nn.functional.linear(states.float(), head.weight, head.bias) / temp
    idx = torch.as_tensor(classes)
    em = (logits[:, idx] - torch.logsumexp(logits, -1, keepdim=True)).numpy()
    em[:, 0] -= penalty  # column 0 is the blank
    return em


def align(utts: list[dict], head: CTCHead, tok, temp: float, penalty: float = 0.0) -> dict:
    """Viterbi per chunk at temperature `temp`; stores words in chunk['aligned']. Returns path stats."""
    frames = token_frames = n_tok = 0
    for u in utts:
        for c in u["chunks"]:
            c["aligned"] = None
            ids = c["token_ids"]
            if not ids:
                continue
            em = emissions(head, c["states"], [BLANK, *ids], temp, penalty)
            path = ctc_viterbi(em, len(ids), [False] + [a == b for a, b in zip(ids, ids[1:])])  # noqa: RUF007
            if path is None:
                continue
            spans = token_spans(path, len(ids))
            probs = [float(np.exp(em[a : b + 1, k + 1]).mean()) for k, (a, b) in enumerate(spans)]
            words = group_words(ids, spans, probs, tok.convert_ids_to_tokens(ids),
                                lambda x: tok.decode(x, clean_up_tokenization_spaces=False))  # fmt: skip
            c["aligned"] = (words, len(em))
            frames += len(em)
            token_frames += int((path % 2 == 1).sum())
            n_tok += len(ids)
    return {
        "blank_frac": round(1 - token_frames / frames, 4),
        "frames_per_token": round(token_frames / n_tok, 3),
    }


def score(utts, audio, rule: str, refine: bool, shift: float) -> dict:
    total = TimingScore()
    for u in utts:
        hyp = []
        for c in u["chunks"]:
            if not c["aligned"]:
                continue
            words, n = c["aligned"]
            lr = None
            if refine:
                a0, a1 = round(c["start"] * SAMPLE_RATE), round(c["end"] * SAMPLE_RATE)
                lr = LoudnessRefinement(audio[u["id"]][a0:a1], a0 / SAMPLE_RATE)
            s, e = word_times(words, n, c["start"], c["regions"], "block", rule, shift, lr)
            hyp += [TimedWord(w.text, float(x), float(y)) for w, x, y in zip(words, s, e)]
        total.add(score_timings([TimedWord(*w) for w in u["ref_words"]], hyp))
    return total.summary()


def calibrate_shift(utts) -> float:
    """The start shift (seconds) that centres starts: minus the median signed start error with no shift."""
    errs: list[float] = []
    for u in utts:
        hyp = []
        for c in u["chunks"]:
            if c["aligned"]:
                words, n = c["aligned"]
                s, e = word_times(words, n, c["start"], c["regions"], "block", "next_start", 0.0)
                hyp += [TimedWord(w.text, float(x), float(y)) for w, x, y in zip(words, s, e)]
        errs += score_timings([TimedWord(*w) for w in u["ref_words"]], hyp).start_err_ms
    return -float(np.median(errs)) / 1000


def run_split(utts, audio, head, tok, temp, shift=None, penalty=0.0) -> dict:
    stats = align(utts, head, tok, temp, penalty)
    if shift is None:
        shift = calibrate_shift(utts)
    out = {"temp": temp, "blank_penalty": penalty, "start_shift_ms": round(1000 * shift, 1), **stats}
    for name, (rule, refine) in RULES.items():
        out[name] = score(utts, audio, rule, refine, shift)
    return out


def line(tag: str, r: dict) -> str:
    head = (
        f"{tag:10s} T={r['temp']:<4} pen {r['blank_penalty']:<4} shift {r['start_shift_ms']:+6.1f} "
        f"blank {r['blank_frac']:.3f} fr/tok {r['frames_per_token']:.2f} | start MAE {r['shipped']['start_mae_ms']:5.1f}"
    )
    parts = [head]
    parts += [f"{k} end MAE {r[k]['end_mae_ms']:5.1f} med {r[k]['end_median_ae_ms']:4.1f}" for k in RULES]
    return " | ".join(parts)


def load(path: Path, split: str):
    utts = torch.load(path, weights_only=False)
    audio = {u.id: u.audio for u in iter_librispeech(split, ids=[u["id"] for u in utts])}
    return utts, audio


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
    out: dict = {"dev": []}
    for t in TEMPS:
        r = run_split(dev, dev_audio, head, tok, t)
        out["dev"].append(r)
        print(line("dev", r), flush=True)
    for pen in PENALTIES:
        r = run_split(dev, dev_audio, head, tok, 1.0, penalty=pen)
        out["dev"].append(r)
        print(line("dev", r), flush=True)
    best = min(out["dev"], key=lambda r: r["shipped"]["start_mae_ms"] + r["shipped"]["end_mae_ms"])
    best_spike = min(out["dev"], key=lambda r: r["spike"]["end_mae_ms"])
    out["chosen"] = {
        "shipped_rule": [best["temp"], best["blank_penalty"]],
        "spike_rule": [best_spike["temp"], best_spike["blank_penalty"]],
    }
    print("chosen on dev (T, penalty):", out["chosen"])

    if args.test:
        test, test_audio = load(args.test, "test_clean")
        base = out["dev"][0]
        out["test"] = []
        picks = {(r["temp"], r["blank_penalty"]): r for r in (base, best, best_spike)}
        for r in picks.values():
            res = run_split(
                test, test_audio, head, tok, r["temp"], r["start_shift_ms"] / 1000, r["blank_penalty"]
            )
            out["test"].append(res)
            print(line("test", res), flush=True)
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
