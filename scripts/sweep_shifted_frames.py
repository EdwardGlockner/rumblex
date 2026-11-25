"""Finer word times from encoding each chunk more than once, each time starting a little earlier.

The encoder gives one frame per 80 ms, so a word start read off the CTC path sits on an 80 ms grid. If the
chunk is encoded again from 20, 40 and 60 ms earlier, the grid lands in three other places, and the word's
start is read four times. This script averages those readings and scores the result on LibriSpeech with
the rest of the shipped aligner unchanged (blank penalty 5, loudness dips, next_start ends, silence cut).

Two steps. `encode` runs the encoder (no decoding) on the chunks saved by collect_alignment_states.py:

    python scripts/sweep_shifted_frames.py encode --states data/align_states/dev_clean.pt \\
        --split dev_clean --out data/align_shift

`score` picks the shift set on dev_clean (start MAE + end MAE, start shift recalibrated per setting) and
reports it once on test_clean:

    python scripts/sweep_shifted_frames.py score --states data/align_states --shifted data/align_shift \\
        --out docs/alignment_shift_sweep.json
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
    FRAME_STEP,
    CTCHead,
    LoudnessRefinement,
    ctc_viterbi,
    frame_times,
    group_words,
    spike_times,
    token_spans,
)
from rumblex.audio import SAMPLE_RATE
from rumblex.bench import TimedWord, TimingScore, iter_librispeech, score_timings

PENALTY = 5.0
SHIFTS_MS = (10, 20, 30, 40, 50, 60, 70)  # encoded; 0 is the saved run
SETS = {
    "1 pass (shipped)": (0,),
    "2 passes, 40 ms apart": (0, 40),
    "4 passes, 20 ms apart": (0, 20, 40, 60),
    "8 passes, 10 ms apart": (0, 10, 20, 30, 40, 50, 60, 70),
}
DIPS = [(0.0, 1), (0.01, 7), (0.02, 7), (0.03, 7)]  # (dip_window, dip_smooth); shipped is (0.02, 7)


def load(path: Path, split: str):
    utts = torch.load(path, weights_only=False)
    audio = {u.id: u.audio for u in iter_librispeech(split, ids=[u["id"] for u in utts])}
    return utts, audio


def encode(args) -> None:
    from rumblex.transcriber import DEFAULT_MODEL, RumbleTranscriber
    from rumblex.vad import SpeechChunk, SpeechRegion

    utts, audio = load(args.states, args.split)
    tr = RumbleTranscriber(DEFAULT_MODEL, device=args.device, dtype="float32", batch_size=args.batch_size)
    args.out.mkdir(parents=True, exist_ok=True)
    for ms in args.shifts:
        dest = args.out / f"{args.split}_shift{ms:02d}.pt"
        if dest.exists():
            print("have", dest, flush=True)
            continue
        d = round(ms * SAMPLE_RATE / 1000)
        pieces, chunks, base = [], [], 0
        for u in utts:
            wav = np.concatenate([np.zeros(d, np.float32), audio[u["id"]]])  # so a chunk at 0 can start early
            for c in u["chunks"]:
                a, b = round(c["start"] * SAMPLE_RATE), round(c["end"] * SAMPLE_RATE)
                chunks.append(SpeechChunk(base + a, base + b + d, (SpeechRegion(base + a, base + b + d),)))
            pieces.append(wav)
            base += len(wav)
        t0 = time.perf_counter()
        states = tr.encode(np.concatenate(pieces), chunks)
        dt = time.perf_counter() - t0
        it = iter(states)
        out = {u["id"]: [next(it).half() for _ in u["chunks"]] for u in utts}
        torch.save(out, dest)
        print(f"shift {ms} ms: {len(chunks)} chunks in {dt:.0f} s -> {dest}", flush=True)


def spikes(states: torch.Tensor, ids: list[int], head: CTCHead, tok, start: float):
    """(words, centre of each word's first frame, end of its last frame) in global seconds, or None."""
    em = head.emissions(states.float(), [BLANK, *ids])
    em[:, 0] -= PENALTY
    path = ctc_viterbi(em, len(ids), [False] + [a == b for a, b in zip(ids, ids[1:])])  # noqa: RUF007
    if path is None:
        return None
    spans = token_spans(path, len(ids))
    words = group_words(ids, spans, [0.0] * len(ids), tok.convert_ids_to_tokens(ids),
                        lambda x: tok.decode(x, clean_up_tokenization_spaces=False))  # fmt: skip
    t = frame_times(len(em)) + start
    first = np.array([t[w.first] for w in words])
    last = np.array([t[w.last] for w in words]) + FRAME_STEP / 2
    return words, first, last


def align(utts, shifted: dict[int, dict], head: CTCHead, tok) -> None:
    """Per chunk: c['spikes'][ms] = (words, first, last) for the saved states (0) and every shifted set."""
    for u in utts:
        for k, c in enumerate(u["chunks"]):
            c["spikes"] = {}
            if not c["token_ids"]:
                continue
            for ms in (0, *shifted):
                st = c["states"] if ms == 0 else shifted[ms][u["id"]][k]
                r = spikes(st, c["token_ids"], head, tok, c["start"] - ms / 1000)
                if r is not None:
                    c["spikes"][ms] = r


def score(utts, audio, shifts: tuple[int, ...], dip: tuple[float, int], shift: float) -> TimingScore:
    total = TimingScore()
    for u in utts:
        hyp = []
        for c in u["chunks"]:
            got = [c["spikes"][ms] for ms in shifts if ms in c["spikes"]]
            if not got or 0 not in c["spikes"]:
                continue
            words = got[0][0]
            first = np.mean([g[1] for g in got], axis=0)
            last = np.mean([g[2] for g in got], axis=0)
            a0, a1 = round(c["start"] * SAMPLE_RATE), round(c["end"] * SAMPLE_RATE)
            lr = LoudnessRefinement(audio[u["id"]][a0:a1], a0 / SAMPLE_RATE, dip_window=dip[0],
                                    dip_smooth=dip[1], last_inside=True)  # fmt: skip
            s, e = spike_times(first, last, c["start"], c["regions"], "next_start", shift, lr)
            hyp += [TimedWord(w.text, float(x), float(y)) for w, x, y in zip(words, s, e)]
        total.add(score_timings([TimedWord(*w) for w in u["ref_words"]], hyp))
    return total


def run(utts, audio, name: str, dip, shift: float | None = None) -> dict:
    if shift is None:  # centre the starts: minus the median signed start error with no shift
        shift = -float(np.median(score(utts, audio, SETS[name], dip, 0.0).start_err_ms)) / 1000
    s = score(utts, audio, SETS[name], dip, shift)
    return {"passes": name, "dip_window": dip[0], "dip_smooth": dip[1],
            "start_shift_ms": round(1000 * shift, 1), **s.summary()}  # fmt: skip


def line(tag: str, r: dict) -> str:
    return (
        f"{tag:4s} {r['passes']:22s} dip {1000 * r['dip_window']:2.0f} ms shift {r['start_shift_ms']:+6.1f} | "
        f"start MAE {r['start_mae_ms']:5.1f} med {r['start_median_ae_ms']:4.1f} | "
        f"end MAE {r['end_mae_ms']:5.1f} med {r['end_median_ae_ms']:4.1f}"
    )


def load_shifted(folder: Path, split: str) -> dict[int, dict]:
    return {ms: torch.load(p, weights_only=False) for ms in SHIFTS_MS
            if (p := folder / f"{split}_shift{ms:02d}.pt").exists()}  # fmt: skip


def do_score(args) -> None:
    from transformers import AutoProcessor

    tok = AutoProcessor.from_pretrained("Reson8/Rumble-2-Flash").tokenizer
    head = CTCHead.from_pretrained(device="cpu")
    dev, dev_audio = load(args.states / "dev_clean.pt", "dev_clean")
    shifted = load_shifted(args.shifted, "dev_clean")
    align(dev, shifted, head, tok)
    out: dict = {"blank_penalty": PENALTY, "dev": []}
    for name, ms in SETS.items():
        if any(m and m not in shifted for m in ms):
            continue
        for dip in DIPS:
            r = run(dev, dev_audio, name, dip)
            out["dev"].append(r)
            print(line("dev", r), flush=True)
    best = min(out["dev"], key=lambda r: r["start_mae_ms"] + r["end_mae_ms"])
    out["chosen"] = {k: best[k] for k in ("passes", "dip_window", "dip_smooth", "start_shift_ms")}
    print("chosen on dev:", out["chosen"])
    if args.test:
        test, test_audio = load(args.states / "test_clean.pt", "test_clean")
        align(test, load_shifted(args.shifted, "test_clean"), head, tok)
        shipped = next(r for r in out["dev"] if r["passes"].startswith("1 pass") and r["dip_window"] == 0.02)
        out["test"] = []
        for r in (shipped, best):
            res = run(test, test_audio, r["passes"], (r["dip_window"], r["dip_smooth"]),
                      r["start_shift_ms"] / 1000)  # fmt: skip
            out["test"].append(res)
            print(line("test", res), flush=True)
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode")
    e.add_argument("--states", type=Path, required=True)
    e.add_argument("--split", required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--shifts", type=int, nargs="+", default=list(SHIFTS_MS))
    e.add_argument("--device", default="cpu")
    e.add_argument("--batch-size", type=int, default=8)
    e.set_defaults(fn=encode)
    s = sub.add_parser("score")
    s.add_argument("--states", type=Path, required=True)
    s.add_argument("--shifted", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--test", action="store_true")
    s.set_defaults(fn=do_score)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
