"""Does disagreement between the CTC head and the decoder say which words are wrong, beyond the decoder's
own `probability`? And does the CTC head hear words the decoder left out? Runs on the states saved by
scripts/collect_confidence_states.py, CPU only.

`features` goes through every chunk once and writes one JSON per set. Per transcribed word, besides the two
scores rumblex already writes (`dec_mean` = `probability`, `ctc_peak_min` = `score`):
  ctc_margin_min  per token, the best frame of (log-prob of the token minus log-prob of the best other
                  non-blank class); lowest token of the word. Above 0 when the head prefers the decoder's
                  token to every other token somewhere in its span
  ctc_top1        share of the word's tokens that are the head's first choice (blank included) on at
                  least one of their frames
  ctc_gap         minus the mean over the word's frames of (best log-prob on the frame minus log-prob of
                  the forced path's class). 0 when the forced path is the greedy path
  ctc_word_match  1 when the word is matched on an alignment of the decoder's words to the words of the
                  greedy CTC text (both normalised), else 0
It also keeps the reference words, the decoder's words and the greedy CTC words of every utterance.

`score` then reports, with everything fitted or chosen on dev and applied to test:
  - AUC of each score for telling wrong words from right ones, and of logistic regressions that add the
    disagreement features to `dec_mean`, with a bootstrap interval (over utterances) on the gain
  - the same among words the decoder is sure of (`dec_mean` >= 0.95), where `probability` has little left
  - deleted words: places where the greedy CTC text has a word the decoder's text lacks, against places
    where the reference has one

Words are labelled as in scripts/eval_confidence.py.

    python scripts/eval_word_trust.py features --states data/confidence_states --out data/word_trust
    python scripts/eval_word_trust.py score --features data/word_trust --out docs/word_trust_eval.json
"""

from __future__ import annotations

import argparse
import json
import re
from itertools import pairwise
from pathlib import Path

import numpy as np
from eval_confidence import BLANK_PENALTY, fit_logistic, normalize

SINGLES = ("dec_mean", "ctc_peak_min", "ctc_margin_min", "ctc_top1", "ctc_gap", "ctc_word_match")
COMBOS = {
    "dec+peak": ("dec_mean", "ctc_peak_min"),
    "dec+margin": ("dec_mean", "ctc_margin_min"),
    "dec+gap": ("dec_mean", "ctc_gap"),
    "dec+match": ("dec_mean", "ctc_word_match"),
    "dec+all": ("dec_mean", "ctc_peak_min", "ctc_margin_min", "ctc_top1", "ctc_gap", "ctc_word_match"),
    "ctc_all": ("ctc_peak_min", "ctc_margin_min", "ctc_top1", "ctc_gap", "ctc_word_match"),
}
_PROBS = ("dec_mean", "ctc_peak_min")
_ALNUM = re.compile(r"\w")
SURE = 0.95
EXTRA_THRESHOLDS = (0.0, 0.3, 0.5, 0.7, 0.9)


def align_ops(ref: list[str], hyp: list[str]) -> tuple[list[int | None], list[tuple[int, int]]]:
    """Minimum-edit-distance alignment, most matches among equal costs.
    Returns (for each hyp word the index of the equal ref word or None, deletions as (ref index, gap)),
    where gap g means "before hyp word g" (len(hyp) is after the last one)."""
    n, m = len(ref), len(hyp)
    big = n + m + 1
    d = np.zeros((n + 1, m + 1), dtype=np.int64)
    d[:, 0] = np.arange(n + 1) * big
    d[0, :] = np.arange(m + 1) * big
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = d[i - 1, j - 1] + (-1 if ref[i - 1] == hyp[j - 1] else big)
            d[i, j] = min(diag, d[i - 1, j] + big, d[i, j - 1] + big)
    matched: list[int | None] = [None] * m
    deleted = []
    i, j = n, m
    while i > 0 or j > 0:
        eq = i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (-1 if eq else big):
            if eq:
                matched[j - 1] = i - 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + big:
            deleted.append((i - 1, j))
            i -= 1
        else:
            j -= 1
    return matched, deleted[::-1]


def chunk_features(c: dict, head, tok) -> tuple[list[dict], list[dict]]:
    """(the decoder's words with their scores, the words of the greedy CTC text) for one chunk."""
    import torch

    from rumblex.aligner import BLANK, ctc_viterbi, token_spans

    ids = list(c["token_ids"])
    lp = head.log_probs(c["states"].float())
    if lp.shape[0] == 0:
        return [], []
    bestv_t, best_t = lp.max(-1)
    best, bestv = best_t.numpy(), bestv_t.numpy()

    def decode(x):
        return tok.decode(x, clean_up_tokenization_spaces=False)

    def group(tokens):
        groups: list[list[int]] = []
        for k, t in enumerate(tokens):
            if not groups or t.startswith("Ġ"):
                groups.append([k])
            else:
                groups[-1].append(k)
        return groups

    # Greedy CTC text: runs of the same best class, blanks dropped; a token's confidence is its best frame.
    g_ids, g_conf = [], []
    for t, k in enumerate(best):
        if k == BLANK:
            continue
        if t and best[t - 1] == k:
            g_conf[-1] = max(g_conf[-1], float(np.exp(bestv[t])))
        else:
            g_ids.append(int(k))
            g_conf.append(float(np.exp(bestv[t])))
    ctc_words = []
    for g in group(tok.convert_ids_to_tokens(g_ids)):
        text = decode([g_ids[k] for k in g]).strip()
        if text and _ALNUM.search(text):
            ctc_words.append({"text": text, "conf": min(g_conf[k] for k in g)})
    if not ids:
        return [], ctc_words

    em = lp[:, [BLANK, *ids]].numpy()
    vit = em.copy()
    vit[:, 0] -= BLANK_PENALTY
    path = ctc_viterbi(vit, len(ids), [False] + [a == b for a, b in pairwise(ids)])
    if path is None:
        return [], ctc_words
    spans = token_spans(path, len(ids))
    top2v, top2i = (x.numpy() for x in torch.topk(lp[:, :BLANK], 2, dim=-1))
    column = np.zeros(len(em), dtype=np.int64)  # the forced path's column in em per frame, 0 is blank
    peak, margin, top1 = [], [], []
    for k, (a, b) in enumerate(spans):
        column[a : b + 1] = k + 1
        own = em[a : b + 1, k + 1]
        other = np.where(top2i[a : b + 1, 0] == ids[k], top2v[a : b + 1, 1], top2v[a : b + 1, 0])
        peak.append(float(np.exp(own.max())))
        margin.append(float((own - other).max()))
        top1.append(bool((best[a : b + 1] == ids[k]).any()))
    gap = bestv - em[np.arange(len(em)), column]
    dec = c["decoder_logprobs"]
    words: list[dict] = []
    for g in group(tok.convert_ids_to_tokens(ids)):
        text = decode([ids[k] for k in g]).strip()
        if not text:
            continue
        if not _ALNUM.search(text):
            if words:
                words[-1]["text"] += text
            continue
        words.append({
            "text": text,
            "dec_mean": float(np.exp(np.mean([dec[k] for k in g]))),
            "ctc_peak_min": min(peak[k] for k in g),
            "ctc_margin_min": min(margin[k] for k in g),
            "ctc_top1": float(np.mean([top1[k] for k in g])),
            "ctc_gap": -float(gap[spans[g[0]][0] : spans[g[-1]][1] + 1].mean()),
        })  # fmt: skip
    return words, ctc_words


def utterance_row(words: list[dict], ctc_words: list[dict], ref_text: str) -> dict:
    """Labels the decoder's words against the reference and against the greedy CTC text."""
    ref = normalize(ref_text)
    hyp, owner = [], []
    for i, w in enumerate(words):
        for p in normalize(w["text"]):
            hyp.append(p)
            owner.append(i)
    ctc, ctc_conf = [], []
    for w in ctc_words:
        for p in normalize(w["text"]):
            ctc.append(p)
            ctc_conf.append(w["conf"])
    in_ref, deleted = align_ops(ref, hyp)
    in_ctc, extra = align_ops(ctc, hyp)
    scored = []
    for i, w in enumerate(words):
        parts = [j for j, o in enumerate(owner) if o == i]
        if not parts:
            continue
        scored.append({
            **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in w.items()},
            "wrong": any(in_ref[j] is None for j in parts),
            "ctc_word_match": float(all(in_ctc[j] is not None for j in parts)),
        })  # fmt: skip
    return {
        "words": scored,
        "ref": ref,
        "hyp": hyp,
        "deleted": [[ref[i], g] for i, g in deleted],
        "ctc_extra": [[ctc[i], g, round(ctc_conf[i], 4)] for i, g in extra],
    }


def cmd_features(args) -> None:
    import torch
    from transformers import AutoTokenizer

    from rumblex.aligner import CTCHead
    from rumblex.transcriber import DEFAULT_MODEL

    head = CTCHead.from_pretrained(DEFAULT_MODEL, device="cpu")
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.sets:
        utts = torch.load(Path(args.states) / f"{name}.pt", weights_only=False)[: args.limit]
        rows, skipped = [], 0
        for u in utts:
            if re.search(r"\d", u["ref_text"]):
                skipped += 1
                continue
            words, ctc_words = [], []
            for c in u["chunks"]:
                w, cw = chunk_features(c, head, tok)
                words += w
                ctc_words += cw
            rows.append(
                {"id": u["id"], "split": u["split"], **utterance_row(words, ctc_words, u["ref_text"])}
            )
        (out / f"{name}.json").write_text(
            json.dumps({"utterances_with_digits_skipped": skipped, "utterances": rows}), encoding="utf-8"
        )
        print(name, len(rows), "utterances,", sum(len(r["words"]) for r in rows), "words", flush=True)


def auc(score: np.ndarray, wrong: np.ndarray) -> float:
    """P(score of a random wrong word < score of a random right word), ties counted half."""
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2)[inv]
    n_w, n_r = int(wrong.sum()), int((~wrong).sum())
    if not n_w or not n_r:
        return float("nan")
    return float((ranks[~wrong].sum() - n_r * (n_r + 1) / 2) / (n_w * n_r))


def design(table: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray:
    cols = []
    for n in names:
        x = table[n]
        cols.append(np.log(np.clip(x, 1e-6, 1)) if n in _PROBS else np.clip(x, -30, 30))
    return np.stack(cols, axis=1)


def flag_stats(score: np.ndarray, wrong: np.ndarray, t: float) -> dict:
    flag = score < t
    return {
        "flagged": round(float(flag.mean()), 4),
        "errors_caught": round(float(flag[wrong].mean()), 4),
        "flagged_wrong": round(float(wrong[flag].mean()), 4) if flag.any() else None,
    }


def bootstrap_gain(
    a: np.ndarray, b: np.ndarray, wrong: np.ndarray, utt: np.ndarray, reps: int, seed: int = 0
):
    """95% interval of AUC(a) - AUC(b), resampling utterances."""
    rng = np.random.default_rng(seed)
    members = [np.flatnonzero(utt == u) for u in np.unique(utt)]
    out = []
    for _ in range(reps):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        out.append(auc(a[idx], wrong[idx]) - auc(b[idx], wrong[idx]))
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def deletion_stats(utts: list[dict], threshold: float) -> dict:
    """Places (gaps between the decoder's words) where the reference has a word the decoder lacks, against
    places where the greedy CTC text has one with confidence >= threshold."""
    gaps = truth = flagged = both = near = 0
    words = found = 0
    for u in utts:
        m = len(u["hyp"])
        gaps += m + 1
        t = {g for _, g in u["deleted"]}
        f = {g for _, g, c in u["ctc_extra"] if c >= threshold}
        truth += len(t)
        flagged += len(f)
        both += len(t & f)
        near += sum(any(abs(g - x) <= 1 for x in t) for g in f)
        words += len(u["deleted"])
        for w, g in u["deleted"]:
            found += any(w == x and abs(g - y) <= 1 and c >= threshold for x, y, c in u["ctc_extra"])
    return {
        "threshold": threshold,
        "gaps": gaps,
        "gaps_with_deletion": truth,
        "gaps_flagged": flagged,
        "precision": round(both / flagged, 4) if flagged else None,
        "precision_within_one_gap": round(near / flagged, 4) if flagged else None,
        "recall": round(both / truth, 4) if truth else None,
        "deleted_words": words,
        "deleted_words_found_with_same_text": round(found / words, 4) if words else None,
    }


def cmd_score(args) -> None:
    data = {n: json.loads((Path(args.features) / f"{n}.json").read_text(encoding="utf-8")) for n in args.sets}
    tables: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for name, d in data.items():
        for split in ("dev", "test"):
            us = [u for u in d["utterances"] if u["split"] == split]
            t = {k: np.array([w[k] for u in us for w in u["words"]], float) for k in SINGLES}
            t["wrong"] = np.array([w["wrong"] for u in us for w in u["words"]], bool)
            t["utt"] = np.array([i for i, u in enumerate(us) for _ in u["words"]])
            tables[name, split] = t

    dev = {k: np.concatenate([tables[n, "dev"][k] for n in data]) for k in (*SINGLES, "wrong")}
    report: dict = {"blank_penalty": BLANK_PENALTY, "sure_threshold": SURE, "weights": {}, "sets": {}}
    sure_names = COMBOS["ctc_all"]
    sure_dev = dev["dec_mean"] >= SURE
    fits = {}
    for combo, names in COMBOS.items():
        fits[combo] = fit_logistic(design(dev, names), (~dev["wrong"]).astype(float))
        report["weights"][combo] = dict(zip(("bias", *names), (round(float(x), 4) for x in fits[combo])))
    sure_fit = fit_logistic(
        design({k: v[sure_dev] for k, v in dev.items()}, sure_names), (~dev["wrong"][sure_dev]).astype(float)
    )
    for t in tables.values():
        for combo, names in COMBOS.items():
            x = design(t, names)
            t[combo] = np.hstack([np.ones((len(x), 1)), x]) @ fits[combo]
        x = design(t, sure_names)
        t["ctc_all_fitted_on_sure"] = np.hstack([np.ones((len(x), 1)), x]) @ sure_fit

    variants = (*SINGLES, *COMBOS)
    dev_scores = {v: np.concatenate([tables[n, "dev"][v] for n in data]) for v in variants}
    thresholds = {v: float(np.quantile(dev_scores[v], 0.10)) for v in variants}
    for name, d in data.items():
        entry: dict = {"utterances_with_digits_skipped": d["utterances_with_digits_skipped"]}
        for split in ("dev", "test"):
            t = tables[name, split]
            wrong = t["wrong"]
            e: dict = {"utterances": int(t["utt"].max()) + 1, "words": len(wrong), "wrong": int(wrong.sum()),
                       "error_rate": round(float(wrong.mean()), 4), "auc": {}, "flag_10pct": {}}  # fmt: skip
            for v in variants:
                e["auc"][v] = round(auc(t[v], wrong), 4)
                e["flag_10pct"][v] = flag_stats(t[v], wrong, thresholds[v])
            if split == "test":
                e["auc_gain_over_dec_mean_95ci"] = {
                    c: bootstrap_gain(t[c], t["dec_mean"], wrong, t["utt"], args.bootstrap) for c in COMBOS
                }
            sure = t["dec_mean"] >= SURE
            e["sure"] = {
                "words": int(sure.sum()),
                "share_of_words": round(float(sure.mean()), 4),
                "wrong": int(wrong[sure].sum()),
                "share_of_all_errors": round(float(wrong[sure].sum() / wrong.sum()), 4),
                "auc": {v: round(auc(t[v][sure], wrong[sure]), 4)
                        for v in (*SINGLES, "ctc_all", "ctc_all_fitted_on_sure")},
            }  # fmt: skip
            us = [u for u in d["utterances"] if u["split"] == split]
            e["deletions"] = [deletion_stats(us, th) for th in EXTRA_THRESHOLDS]
            entry[split] = e
            print(name, split, e["words"], "words", e["wrong"], "wrong", e["auc"], flush=True)
            print("   sure:", e["sure"], flush=True)
            for row in e["deletions"]:
                print("   del:", row, flush=True)
            if split == "test":
                print("   gain:", e["auc_gain_over_dec_mean_95ci"], flush=True)
        report["sets"][name] = entry
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features")
    f.add_argument("--states", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--sets", nargs="+", default=["ami", "earnings22"])
    f.add_argument("--limit", type=int, default=None, help="first N utterances per set, for a quick check")
    f.set_defaults(func=cmd_features)
    s = sub.add_parser("score")
    s.add_argument("--features", required=True)
    s.add_argument("--out", default=None)
    s.add_argument("--sets", nargs="+", default=["ami", "earnings22"])
    s.add_argument("--bootstrap", type=int, default=1000)
    s.set_defaults(func=cmd_score)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
