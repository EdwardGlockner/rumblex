"""A small learned correction for word starts and ends, read off the encoder states.

The CTC path puts a word's start on an 80 ms frame, and the model's spike for a word sits at a fairly
steady but word-dependent distance from where the word really starts. A linear map from the encoder states
around the word's first frame predicts that distance; a second one does the same for the word's end from
the states around its last frame. Both are fitted on LibriSpeech training utterances against the MFA word
times, chosen on dev_clean and reported once on test_clean.

Two steps. `features` runs the CTC head and Viterbi on saved states and keeps the states around every
word (needs the head, so run it where the head fits; the rest is light):

    python scripts/boundary_head.py features --states data/boundary_train/*.pt --split train \\
        --out data/boundary/train.pt
    python scripts/boundary_head.py features --states data/align_states/dev_clean.pt --split dev_clean \\
        --out data/boundary/dev_clean.pt

`fit` fits the ridge regressions, scores every setting on dev_clean through the rest of the shipped
aligner, and writes the chosen head:

    python scripts/boundary_head.py fit --data data/boundary --out docs/boundary_head_sweep.json \\
        --save data/boundary/boundary_head.safetensors [--test]
"""

from __future__ import annotations

import argparse
import glob
import json
from itertools import product
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
from rumblex.bench import TimedWord, TimingScore, iter_librispeech, match_words, normalize_word, score_timings

PENALTY = 5.0
HALF = 3  # frames kept on each side of a word's first and last frame
CLIP_MS = 150.0  # training targets are clipped here, so misplaced words don't drive the fit
WIDTHS = (0, 1, 2, 3)
LAMBDAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0)
DIPS = ((0.0, 1), (0.01, 7), (0.02, 7))


def around(states: torch.Tensor, idx: list[int]) -> torch.Tensor:
    """[len(idx), 2 * HALF + 1, d]: the states from HALF frames before to HALF after each index, repeating
    the first or last frame past the chunk's edges."""
    n = len(states)
    rows = torch.as_tensor(idx)[:, None] + torch.arange(-HALF, HALF + 1)[None, :]
    return states[rows.clamp(0, n - 1)]


def features(args) -> None:
    from transformers import AutoProcessor

    tok = AutoProcessor.from_pretrained("Reson8/Rumble-2-Flash").tokenizer
    head = CTCHead.from_pretrained(device=args.device)
    utts_out, fs, fe, ys, ye = [], [], [], [], []
    row = 0
    # cmd.exe and the job queue pass patterns through unexpanded
    paths = [p for pat in args.states for p in (sorted(glob.glob(str(pat))) or [pat])]
    for path in paths:
        for u in torch.load(path, weights_only=False):
            hyp, chunks = [], []
            for c in u["chunks"]:
                ids = c["token_ids"]
                if not ids:
                    continue
                st = c["states"].float()
                em = head.emissions(st, [BLANK, *ids])
                em[:, 0] -= PENALTY
                path_ = ctc_viterbi(em, len(ids), [False] + [a == b for a, b in zip(ids, ids[1:])])  # noqa: RUF007
                if path_ is None:
                    continue
                spans = token_spans(path_, len(ids))
                words = group_words(ids, spans, [0.0] * len(ids), tok.convert_ids_to_tokens(ids),
                                    lambda x: tok.decode(x, clean_up_tokenization_spaces=False))  # fmt: skip
                t = frame_times(len(em)) + c["start"]
                first, last = [w.first for w in words], [w.last for w in words]
                fs.append(around(st, first).half())
                fe.append(around(st, last).half())
                chunks.append({
                    "start": c["start"], "end": c["end"], "regions": c["regions"], "n_frames": len(em),
                    "words": [w.text for w in words], "first": first, "last": last, "row": row,
                })  # fmt: skip
                hyp += [(w.text, t[w.first] - FRAME_STEP / 2, t[w.last] + FRAME_STEP / 2) for w in words]
                row += len(words)
            # targets: how far the raw start and end are from the reference, for the words both agree on
            s_t, e_t = np.full(len(hyp), np.nan), np.full(len(hyp), np.nan)
            ref = [w for w in u["ref_words"] if normalize_word(w[0])]
            keep = [j for j, h in enumerate(hyp) if normalize_word(h[0])]
            pairs = match_words(
                [normalize_word(w[0]) for w in ref], [normalize_word(hyp[j][0]) for j in keep]
            )
            for i, j in pairs:
                s_t[keep[j]] = (ref[i][1] - hyp[keep[j]][1]) * 1000
                e_t[keep[j]] = (ref[i][2] - hyp[keep[j]][2]) * 1000
            ys.append(s_t)
            ye.append(e_t)
            utts_out.append({"id": u["id"], "ref_words": u["ref_words"], "chunks": chunks})
        print(path, "->", row, "words", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"split": args.split, "utts": utts_out, "start_states": torch.cat(fs), "end_states": torch.cat(fe),
                "start_ms": np.concatenate(ys), "end_ms": np.concatenate(ye)}, args.out)  # fmt: skip
    print("wrote", args.out, flush=True)


def design(states: torch.Tensor, width: int) -> np.ndarray:
    """[n, (2 * width + 1) * d] float32 from the saved [n, 2 * HALF + 1, d]."""
    return states[:, HALF - width : HALF + width + 1].float().flatten(1).numpy()


def gram(states: torch.Tensor, width: int, y: np.ndarray) -> dict:
    """Sums for a ridge regression of `y` on the states, built in pieces so the inputs stay float16."""
    ok = np.flatnonzero(np.isfinite(y))
    centre = float(np.median(y[ok]))
    t = np.clip(y[ok] - centre, -CLIP_MS, CLIP_MS)
    d = (2 * width + 1) * states.shape[-1]
    xx, xt, sx = np.zeros((d, d)), np.zeros(d), np.zeros(d)
    for i in range(0, len(ok), 4096):
        rows = ok[i : i + 4096]
        x = design(states[rows], width).astype(np.float64)
        xx += x.T @ x
        xt += x.T @ t[i : i + 4096]
        sx += x.sum(0)
    return {"xx": xx, "xt": xt, "sx": sx, "st": float(t.sum()), "n": len(ok), "centre": centre}


def ridge(g: dict, lam: float) -> tuple[np.ndarray, float]:
    """Weights and bias of a ridge regression on standardised inputs, folded back to raw inputs.
    `lam` is per 1,000 training rows, so a setting means the same with more data."""
    n, mu = g["n"], g["sx"] / g["n"]
    cov = g["xx"] - n * np.outer(mu, mu)
    sd = np.sqrt(np.diag(cov) / n) + 1e-6
    a = cov / np.outer(sd, sd)
    rhs = (g["xt"] - mu * g["st"]) / sd
    w = np.linalg.solve(a + lam * n / 1000 * np.eye(len(mu)), rhs) / sd
    return w.astype(np.float32), float(g["centre"] - mu @ w)


def score(
    data: dict, audio: dict, ds: np.ndarray | None, de: np.ndarray | None, dip, shift: float
) -> TimingScore:
    """Timing through the shipped rules, with starts moved by `ds` ms and (with `de`) ends set to the
    corrected end, never past the next word's start."""
    total = TimingScore()
    for u in data["utts"]:
        hyp = []
        for c in u["chunks"]:
            n, r = len(c["words"]), c["row"]
            t = frame_times(c["n_frames"]) + c["start"]
            first = t[c["first"]] + (0 if ds is None else ds[r : r + n] / 1000)
            last = t[c["last"]] + FRAME_STEP / 2
            a0, a1 = round(c["start"] * SAMPLE_RATE), round(c["end"] * SAMPLE_RATE)
            lr = LoudnessRefinement(audio[u["id"]][a0:a1], a0 / SAMPLE_RATE, dip_window=dip[0],
                                    dip_smooth=dip[1], last_inside=True)  # fmt: skip
            s, e = spike_times(first, last, c["start"], c["regions"], "next_start", shift, lr)
            if de is not None:
                e2 = last + de[r : r + n] / 1000
                e2[:-1] = np.minimum(e2[:-1], s[1:])
                e = np.maximum(e2, s + 0.01)
            hyp += [TimedWord(w, float(x), float(y)) for w, x, y in zip(c["words"], s, e)]
        total.add(score_timings([TimedWord(*w) for w in u["ref_words"]], hyp))
    return total


def run(data, audio, ds, de, dip, shift: float | None = None) -> dict:
    if shift is None:  # centre the starts: minus the median signed start error with no shift
        shift = -float(np.median(score(data, audio, ds, de, dip, 0.0).start_err_ms)) / 1000
    return {"dip_window": dip[0], "dip_smooth": dip[1], "start_shift_ms": round(1000 * shift, 1),
            **score(data, audio, ds, de, dip, shift).summary()}  # fmt: skip


def line(tag: str, r: dict) -> str:
    return (
        f"{tag:4s} {r['name']:34s} dip {1000 * r['dip_window']:2.0f} shift {r['start_shift_ms']:+5.1f} | start "
        f"{r['start_mae_ms']:5.1f} med {r['start_median_ae_ms']:4.1f} <=20 {r['start_within_20ms']:.3f} | end "
        f"{r['end_mae_ms']:5.1f} med {r['end_median_ae_ms']:4.1f} <=20 {r['end_within_20ms']:.3f}"
    )


def load_audio(data: dict) -> dict:
    return {u.id: u.audio for u in iter_librispeech(data["split"], ids=[u["id"] for u in data["utts"]])}


def fit(args) -> None:
    train = torch.load(args.data / "train.pt", weights_only=False)
    dev = torch.load(args.data / "dev_clean.pt", weights_only=False)
    dev_audio = load_audio(dev)
    out: dict = {"train_words": int(np.isfinite(train["start_ms"]).sum()), "dev": []}
    print("training words:", out["train_words"], flush=True)

    def add(name: str, r: dict, **kw) -> dict:
        r = {"name": name, **kw, **r}
        out["dev"].append(r)
        print(line("dev", r), flush=True)
        return r

    shipped = add("shipped", run(dev, dev_audio, None, None, (0.02, 7), -0.004))
    # starts: every width and penalty, then with a loudness dip on top
    heads: dict = {}
    grams = {(k, width): gram(train[f"{k}_states"], width, train[f"{k}_ms"])
             for k, width in product(("start", "end"), WIDTHS)}  # fmt: skip
    for width, lam in product(WIDTHS, LAMBDAS):
        w, b = ridge(grams["start", width], lam)
        ds = design(dev["start_states"], width) @ w + b
        heads["start", width, lam] = (w, b, ds)
        add(f"starts, +-{width} frames, lambda {lam:g}", run(dev, dev_audio, ds, None, (0.0, 1)),
            kind="start", width=width, lam=lam)  # fmt: skip
    key = lambda r: r["start_mae_ms"] + r["end_mae_ms"]
    best_s = min((r for r in out["dev"] if r.get("kind") == "start"), key=key)
    ds = heads["start", best_s["width"], best_s["lam"]][2]
    for dip in DIPS[1:]:
        add(f"starts as chosen, dip {1000 * dip[0]:.0f} ms", run(dev, dev_audio, ds, None, dip),
            kind="start", width=best_s["width"], lam=best_s["lam"])  # fmt: skip
    best_s = min((r for r in out["dev"] if r.get("kind") == "start"), key=key)
    dip = (best_s["dip_window"], best_s["dip_smooth"])
    # ends on top of the chosen starts
    for width, lam in product(WIDTHS, LAMBDAS):
        w, b = ridge(grams["end", width], lam)
        de = design(dev["end_states"], width) @ w + b
        heads["end", width, lam] = (w, b, de)
        add(
            f"+ ends, +-{width} frames, lambda {lam:g}",
            run(dev, dev_audio, ds, de, dip, best_s["start_shift_ms"] / 1000),
            kind="end",
            width=width,
            lam=lam,
        )
    best_e = min((r for r in out["dev"] if r.get("kind") == "end"), key=key)
    use_end = key(best_e) < key(best_s)
    out["chosen"] = {
        "start": {k: best_s[k] for k in ("width", "lam", "dip_window", "dip_smooth", "start_shift_ms")},
        "end": {k: best_e[k] for k in ("width", "lam")} if use_end else None,
    }
    print("chosen on dev:", out["chosen"], flush=True)

    ws, bs, _ = heads["start", best_s["width"], best_s["lam"]]
    we, be, _ = heads["end", best_e["width"], best_e["lam"]]
    if args.save:
        from safetensors.torch import save_file

        tensors = {"start.weight": torch.from_numpy(ws), "start.bias": torch.tensor([bs])}
        if use_end:
            tensors |= {"end.weight": torch.from_numpy(we), "end.bias": torch.tensor([be])}
        save_file(tensors, str(args.save), metadata={"chosen": json.dumps(out["chosen"])})
        print("saved", args.save, flush=True)

    if args.test:
        test = torch.load(args.data / "test_clean.pt", weights_only=False)
        test_audio = load_audio(test)
        ds = design(test["start_states"], best_s["width"]) @ ws + bs
        de = design(test["end_states"], best_e["width"]) @ we + be if use_end else None
        out["test"] = [
            {"name": "shipped", **run(test, test_audio, None, None, (0.02, 7), -0.004)},
            {"name": "boundary head", **run(test, test_audio, ds, de, dip, best_s["start_shift_ms"] / 1000)},
        ]
        for r in out["test"]:
            print(line("test", r), flush=True)
    args.out.write_text(json.dumps({"shipped_dev": shipped, **out}, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features")
    f.add_argument("--states", type=Path, nargs="+", required=True)
    f.add_argument("--split", required=True, help="train, dev_clean or test_clean")
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--device", default="cpu")
    f.set_defaults(fn=features)
    t = sub.add_parser("fit")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--save", type=Path)
    t.add_argument("--test", action="store_true")
    t.set_defaults(fn=fit)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
