"""Word timings from `rumblex.align` (given text, no decoding) against LibriSpeech's MFA word alignments.

Two tests on the same utterances as docs/alignment_eval.md (test_clean, 200, seed 0):
  per utterance: each utterance aligned to its reference transcript on its own, and for comparison
                 transcribed + aligned as usual (`rumblex.transcribe`). Both are also scored on the words
                 both of them time, since transcribe + align only times words it recognised;
  long file:     all utterances joined with 1 s of silence into one file, aligned to the joined transcript,
                 so VAD chunking and the assignment of text to chunks are exercised.
Words come from the given text, so word i of an utterance's output is word i of its transcript; errors are
paired by index. Reference words MFA left as "<unk>" are not scored. An utterance where MFA splits one word
in two ("<unk> s") is aligned but not scored. "Gross" counts starts more than 1 s off (a word placed in the
wrong chunk shows up there); those words are listed in the report.

    python scripts/eval_align_text.py --device cpu --out docs/align_text_eval.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import rumblex
from rumblex.audio import SAMPLE_RATE
from rumblex.bench import iter_librispeech, match_words, normalize_word, sample_ids

GAP_S = 1.0


def summary(ref: list, hyp: list) -> dict:
    if len(ref) != len(hyp):
        raise ValueError(f"{len(ref)} reference words but {len(hyp)} aligned words")
    s = np.array([1000 * (h.start - r.start) for r, h in zip(ref, hyp)])
    e = np.array([1000 * (h.end - r.end) for r, h in zip(ref, hyp)])
    a, b = np.abs(s), np.abs(e)
    return {
        "words": len(ref),
        "start_mae_ms": round(float(a.mean()), 1),
        "start_median_ae_ms": round(float(np.median(a)), 1),
        "start_mean_signed_ms": round(float(s.mean()), 1),
        "starts_within_50ms": round(float((a <= 50).mean()), 4),
        "starts_within_100ms": round(float((a <= 100).mean()), 4),
        "end_mae_ms": round(float(b.mean()), 1),
        "end_median_ae_ms": round(float(np.median(b)), 1),
        "end_mean_signed_ms": round(float(e.mean()), 1),
        "ends_within_100ms": round(float((b <= 100).mean()), 4),
        "gross_over_1s": int((a > 1000).sum()),
    }


def words_of(res) -> list:
    return [w for seg in res.segments for w in seg.words]


def score_mask(u) -> list[bool] | None:
    """Per transcript word: score it? None if MFA's words don't line up with the transcript one to one."""
    mfa, text = [w.word for w in u.words], u.transcript.split()
    if len(mfa) != len(text):
        return None
    if any(m != t and m != "<unk>" for m, t in zip(mfa, text)):
        return None
    return [m != "<unk>" for m in mfa]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="float32")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    utts = list(iter_librispeech("test_clean", ids=sample_ids("test_clean", args.n, 0)))
    masks = [score_mask(u) for u in utts]
    model = rumblex.load_model(device=args.device, compute_type=args.compute_type)
    rumblex.align(utts[0].audio, utts[0].transcript, model)  # warm-up
    report: dict = {
        "device": args.device,
        "compute_type": args.compute_type,
        "utterances": len(utts),
        "not_scored": [u.id for u, m in zip(utts, masks) if m is None],
        "unk_words_not_scored": sum(m.count(False) for m in masks if m),
    }

    ref, hyp, t_align = [], [], 0.0
    both_ref, both_given, both_asr = [], [], []
    asr_ref, asr_hyp, t_asr = [], [], 0.0
    for u, mask in zip(utts, masks):
        t0 = time.perf_counter()
        out = words_of(rumblex.align(u.audio, u.transcript, model, language="English"))
        t_align += time.perf_counter() - t0
        t0 = time.perf_counter()
        asr = [
            w
            for w in words_of(rumblex.transcribe(u.audio, model, language="English"))
            if normalize_word(w.word)
        ]
        t_asr += time.perf_counter() - t0
        if mask is None:
            continue
        keep = [i for i, k in enumerate(mask) if k]
        ref += [u.words[i] for i in keep]
        hyp += [out[i] for i in keep]
        # transcribe + align: matched by edit distance, as in docs/alignment_eval.md
        pairs = match_words(
            [normalize_word(u.words[i].word) for i in keep], [normalize_word(w.word) for w in asr]
        )
        for i, j in pairs:
            asr_ref.append(u.words[keep[i]])
            asr_hyp.append(asr[j])
            both_ref.append(u.words[keep[i]])
            both_given.append(out[keep[i]])
            both_asr.append(asr[j])
    report["per_utterance"] = {**summary(ref, hyp), "seconds": round(t_align, 1)}
    report["transcribe_align"] = {**summary(asr_ref, asr_hyp), "seconds": round(t_asr, 1)}
    report["same_words"] = {
        "align": summary(both_ref, both_given),
        "transcribe_align": summary(both_ref, both_asr),
    }
    for k in ("per_utterance", "transcribe_align", "same_words"):
        print(k, report[k], flush=True)

    gap = np.zeros(int(GAP_S * SAMPLE_RATE), np.float32)
    pieces, ref, text, keep, owner, base = [], [], [], [], [], 0.0
    for u, mask in zip(utts, masks):
        pieces += [u.audio, gap]
        n = len(u.transcript.split())
        m = mask if mask is not None else [False] * n
        ref += [type(w)(w.word, w.start + base, w.end + base) for w, k in zip(u.words, m) if k]
        keep += m
        owner += [u.id] * n
        text.append(u.transcript)
        base += len(u.audio) / SAMPLE_RATE + GAP_S
    wav = np.concatenate(pieces)
    t0 = time.perf_counter()
    res = rumblex.align(wav, "\n".join(text), model, language="English")
    secs = time.perf_counter() - t0
    out = words_of(res)
    if len(out) != len(keep):
        raise ValueError(f"{len(keep)} words in the text but {len(out)} aligned words")
    idx = [i for i, k in enumerate(keep) if k]
    got = [out[i] for i in idx]
    gross = [
        {"utterance": owner[i], "word": r.word, "ref_start": round(r.start, 2), "start": round(h.start, 2)}
        for i, r, h in zip(idx, ref, got)
        if abs(h.start - r.start) > 1.0
    ]
    report["long_file"] = {
        **summary(ref, got),
        "audio_seconds": round(len(wav) / SAMPLE_RATE, 1),
        "chunks": res.metadata["chunks"],
        "fallback_chunks": res.metadata["alignment"]["fallback_chunks"],
        "seconds": round(secs, 1),
        "gross_words": gross,
    }
    print("long file", report["long_file"], flush=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
