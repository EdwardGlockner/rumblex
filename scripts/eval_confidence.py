"""How well word scores tell right words from wrong ones, on states saved by
scripts/collect_confidence_states.py (CPU only).

A transcribed word is "wrong" when it is not matched to the reference on a minimum-edit-distance
alignment (a substitution or an insertion). Both sides go through Whisper's English normaliser without
its number step (spellings, contractions, fillers; numbers would merge across words, and are skipped
anyway), with "ok" taken as "okay"; dotted letters ("D. V. D.", "U.S.") are joined first. Utterances whose
reference contains a digit are left out, since the model was run with digits off and would be counted
wrong on every number. Words left empty by the normalisation (a lone dash) are not scored.

Scores per word, each from the tokens of that word (punctuation attached to a word doesn't count):
  ctc_mean       mean over tokens of the token's mean CTC probability over its frames (rumblex 0.2.0's `score`)
  ctc_min        the same, lowest token instead of the mean
  ctc_peak_min   lowest token of each token's highest CTC probability over its frames
  dec_mean       exp of the mean decoder log-prob of the word's tokens
  dec_min        lowest decoder probability of the word's tokens
  combined       logistic regression on log ctc_mean and log dec_min, fitted on dev
The CTC path is found as rumblex does it (blank penalty 5 before Viterbi, unpenalised log-probs scored).

Reported per set and split: error rate, AUC (chance a random wrong word scores lower than a random
right one), and how many errors a threshold catches. Thresholds and the combination are chosen on dev
and applied to test. `--audit` writes wrong words with their reference context for reading.

    python scripts/eval_confidence.py --states data/confidence_states --out docs/confidence_eval.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from rumblex.aligner import BLANK, CTCHead, ctc_viterbi, group_words, token_spans
from rumblex.bench import match_words
from rumblex.transcriber import DEFAULT_MODEL

BLANK_PENALTY = 5.0
SINGLES = ("ctc_mean", "ctc_min", "ctc_peak_min", "dec_mean", "dec_min")
_ACRONYM = re.compile(r"\b(?:[A-Za-z]\. ?){2,}")
_ALNUM = re.compile(r"\w")


def join_acronyms(s: str) -> str:
    return _ACRONYM.sub(lambda m: m.group(0).replace(".", "").replace(" ", "") + " ", s)


def _english_normalizer():
    from whisper_normalizer.english import EnglishTextNormalizer

    n = EnglishTextNormalizer()
    n.standardize_numbers = lambda s: s
    return n


_NORMALIZER = _english_normalizer()
_SAME = {"ok": "okay"}


def normalize(s: str) -> list[str]:
    return [_SAME.get(w, w) for w in _NORMALIZER(join_acronyms(s)).split()]


def chunk_words(c: dict, head: CTCHead, tok) -> list[dict]:
    """Words of one chunk with their scores, grouped exactly as rumblex.aligner.group_words does."""
    ids = c["token_ids"]
    if not ids:
        return []
    em = head.emissions(c["states"].float(), [BLANK, *ids])
    vit = em.copy()
    vit[:, 0] -= BLANK_PENALTY
    path = ctc_viterbi(vit, len(ids), [False] + [a == b for a, b in pairwise(ids)])
    if path is None:
        return []
    spans = token_spans(path, len(ids))
    p = np.exp(em)
    mean = [float(p[a : b + 1, k + 1].mean()) for k, (a, b) in enumerate(spans)]
    peak = [float(p[a : b + 1, k + 1].max()) for k, (a, b) in enumerate(spans)]
    dec = c["decoder_logprobs"]
    tokens = tok.convert_ids_to_tokens(ids)

    def decode(x):
        return tok.decode(x, clean_up_tokenization_spaces=False)

    groups: list[list[int]] = []
    for k, t in enumerate(tokens):
        if not groups or t.startswith("Ġ"):
            groups.append([k])
        else:
            groups[-1].append(k)
    words: list[dict] = []
    for g in groups:
        text = decode([ids[k] for k in g]).strip()
        if not text:
            continue
        if not _ALNUM.search(text):
            if words:
                words[-1]["text"] += text
            continue
        words.append({
            "text": text,
            "ctc_mean": float(np.mean([mean[k] for k in g])),
            "ctc_min": min(mean[k] for k in g),
            "ctc_peak_min": min(peak[k] for k in g),
            "dec_mean": float(np.exp(np.mean([dec[k] for k in g]))),
            "dec_min": float(np.exp(min(dec[k] for k in g))),
        })  # fmt: skip
    ref = [w.text for w in group_words(ids, spans, mean, tokens, decode)]
    got = [w["text"] for w in words]
    if ref[-len(got) :] != got and ref != got:  # leading punctuation is merged into the first word there
        raise RuntimeError(f"word grouping differs from rumblex's: {ref} vs {got}")
    return words


def label(words: list[dict], ref_text: str) -> list[dict]:
    """Sets w['wrong'] on words that normalise to something; returns those."""
    ref = normalize(ref_text)
    parts, owner = [], []
    for i, w in enumerate(words):
        for p in normalize(w["text"]):
            parts.append(p)
            owner.append(i)
    matched = {j for _, j in match_words(ref, parts)}
    ok = [True] * len(words)
    seen = [False] * len(words)
    for j, i in enumerate(owner):
        seen[i] = True
        ok[i] = ok[i] and j in matched
    out = []
    for i, w in enumerate(words):
        if seen[i]:
            w["wrong"] = not ok[i]
            out.append(w)
    return out


def auc(score: np.ndarray, wrong: np.ndarray) -> float:
    """P(score of a random wrong word < score of a random right word), ties counted half."""
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(s))
    r[order] = ranks
    n_w, n_r = int(wrong.sum()), int((~wrong).sum())
    return float((r[~wrong].sum() - n_r * (n_r + 1) / 2) / (n_w * n_r))


def fit_logistic(x: np.ndarray, y: np.ndarray, iters: int = 50) -> np.ndarray:
    """Weights (bias first) of P(right) = sigmoid(w . [1, x]), by Newton's method."""
    X = np.hstack([np.ones((len(x), 1)), x])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(len(w))
        w -= np.linalg.solve(H, X.T @ (p - y))
    return w


def features(rows: list[dict]) -> np.ndarray:
    return np.log(np.clip([[r["ctc_mean"], r["dec_min"]] for r in rows], 1e-6, 1))


def flag_stats(score: np.ndarray, wrong: np.ndarray, t: float) -> dict:
    flag = score < t
    return {
        "threshold": round(float(t), 4),
        "flagged": round(float(flag.mean()), 4),
        "errors_caught": round(float(flag[wrong].mean()), 4),
        "flagged_wrong": round(float(wrong[flag].mean()), 4) if flag.any() else None,
    }


def calibration(score: np.ndarray, wrong: np.ndarray) -> list[dict]:
    edges = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0001]
    out = []
    for a, b in pairwise(edges):
        m = (score >= a) & (score < b)
        if m.any():
            out.append(
                {
                    "bin": f"{a:g}-{min(b, 1):g}",
                    "words": int(m.sum()),
                    "right": round(float(1 - wrong[m].mean()), 4),
                }
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--sets", nargs="+", default=["ami", "earnings22"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--audit", default=None, help="write 60 random wrong dev words with context here")
    args = ap.parse_args()
    root = Path(args.states)
    head = CTCHead.from_pretrained(DEFAULT_MODEL, device=args.device)
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)

    rows: dict[str, dict[str, list[dict]]] = {}
    report: dict = {"blank_penalty": BLANK_PENALTY, "sets": {}}
    audit = []
    for name in args.sets:
        utts = torch.load(root / f"{name}.pt", weights_only=False)
        rows[name] = {"dev": [], "test": []}
        skipped = 0
        for u in utts:
            if re.search(r"\d", u["ref_text"]):
                skipped += 1
                continue
            words = [w for c in u["chunks"] for w in chunk_words(c, head, tok)]
            scored = label(words, u["ref_text"])
            rows[name][u["split"]] += scored
            if u["split"] == "dev":
                hyp = " ".join(w["text"] for w in words)
                audit += [f"[{name}] {w['text']!r}  ctc {w['ctc_mean']:.2f} dec {w['dec_min']:.2f}\n  ref: {u['ref_text']}\n  hyp: {hyp}\n"
                          for w in scored if w["wrong"]]  # fmt: skip
        report["sets"][name] = {"utterances_with_digits_skipped": skipped,
                                "utterances": len(utts) - skipped}  # fmt: skip

    dev_all = [r for name in rows for r in rows[name]["dev"]]
    w = fit_logistic(features(dev_all), np.array([not r["wrong"] for r in dev_all], float))
    report["combined_weights"] = [round(float(x), 4) for x in w]
    for name in rows:
        for split in ("dev", "test"):
            rs = rows[name][split]
            z = np.hstack([np.ones((len(rs), 1)), features(rs)]) @ w
            for r, zi in zip(rs, z):
                r["combined"] = float(1 / (1 + np.exp(-zi)))

    variants = (*SINGLES, "combined")
    # Thresholds chosen on dev (all sets pooled) to flag 5% and 10% of words.
    dev_scores = {v: np.array([r[v] for r in dev_all]) for v in variants}
    thresholds = {
        v: {f"flag_{q}pct": float(np.quantile(dev_scores[v], q / 100)) for q in (5, 10)} for v in variants
    }
    for name in rows:
        for split in ("dev", "test"):
            rs = rows[name][split]
            wrong = np.array([r["wrong"] for r in rs])
            entry = {"words": len(rs), "wrong": int(wrong.sum()), "error_rate": round(float(wrong.mean()), 4)}
            for v in variants:
                s = np.array([r[v] for r in rs])
                entry[v] = {"auc": round(auc(s, wrong), 4),
                            **{k: flag_stats(s, wrong, t) for k, t in thresholds[v].items()},
                            "calibration": calibration(s, wrong)}  # fmt: skip
            report["sets"][name][split] = entry
            print(name, split, entry["words"], "words", entry["wrong"], "wrong",
                  {v: entry[v]["auc"] for v in variants}, flush=True)  # fmt: skip

    if args.audit:
        random.Random(0).shuffle(audit)
        Path(args.audit).write_text("\n".join(audit[:60]), encoding="utf-8")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
