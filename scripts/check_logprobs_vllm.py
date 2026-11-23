"""The vLLM backend gives every text token a decoder log-prob, and they agree with the transformers backend.

Transcribes the same chunks (AMI and Earnings-22 utterances, chunked as `rumblex.transcribe` does,
English forced, digits off) with both backends and reports, per set: chunks whose vLLM log-probs are
missing (the server's `logprobs` didn't line up with its `token_ids`), chunks where both backends
produced the same tokens, and on those the token log-prob and word probability differences. Then runs
the whole pipeline on the repo's samples through vLLM and counts words without a `probability`.
Needs a running server (`vllm serve Reson8/Rumble-2-Flash`) and a GPU for the transformers backend.

    python scripts/check_logprobs_vllm.py --url http://localhost:8000/v1 --out results/logprobs_vllm.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import rumblex
from rumblex.bench import load_eval_set
from rumblex.transcriber import DEFAULT_MODEL, RumbleTranscriber, speech_chunks
from rumblex.vllm_backend import VLLMTranscriber


def word_probs(ids: list[int], lps: list[float], tok) -> list[float]:
    """exp of the mean log-prob per word, grouped at the leading-space marker as the aligner does."""
    groups: list[list[float]] = []
    for t, lp in zip(tok.convert_ids_to_tokens(ids), lps):
        if not groups or t.startswith("Ġ"):
            groups.append([lp])
        else:
            groups[-1].append(lp)
    return [float(np.exp(np.mean(g))) for g in groups]


def compare(name: str, n: int, hf: RumbleTranscriber, vl: VLLMTranscriber) -> dict:
    utts, _ = load_eval_set(name, n, 0)
    st = {"utterances": len(utts), "chunks": 0, "vllm_missing": 0, "hf_missing": 0, "same_tokens": 0,
          "token_lp_abs_diff": [], "word_prob_abs_diff": []}  # fmt: skip
    for u in utts:
        chunks = speech_chunks(u.audio, 20.0, hf.device)
        kw = {"language": "English", "digits": False}
        a = hf.transcribe_chunks(u.audio, chunks, **kw)
        b = vl.transcribe_chunks(u.audio, chunks, **kw)
        for x, y in zip(a, b):
            if not x.token_ids and not y.token_ids:
                continue
            st["chunks"] += 1
            st["hf_missing"] += x.token_logprobs is None
            st["vllm_missing"] += y.token_logprobs is None
            if x.token_ids != y.token_ids or x.token_logprobs is None or y.token_logprobs is None:
                continue
            st["same_tokens"] += 1
            st["token_lp_abs_diff"] += np.abs(np.subtract(x.token_logprobs, y.token_logprobs)).tolist()
            px = word_probs(x.token_ids, x.token_logprobs, hf.tokenizer)
            py = word_probs(y.token_ids, y.token_logprobs, hf.tokenizer)
            st["word_prob_abs_diff"] += np.abs(np.subtract(px, py)).tolist()
    for k in ("token_lp_abs_diff", "word_prob_abs_diff"):
        d = np.array(st.pop(k))
        st[k] = {"n": len(d)} if not len(d) else {
            "n": len(d), "mean": round(float(d.mean()), 5), "p99": round(float(np.quantile(d, 0.99)), 5),
            "max": round(float(d.max()), 5)}  # fmt: skip
    print(name, st, flush=True)
    return st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--compute-type", default="bfloat16")
    ap.add_argument("--n", type=int, default=25, help="utterances per set")
    ap.add_argument("--samples", default="samples")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    hf = RumbleTranscriber(DEFAULT_MODEL, device=args.device, dtype=args.compute_type)
    vl = VLLMTranscriber(base_url=args.url, device=args.device, dtype=args.compute_type)
    report = {
        "gpu": torch.cuda.get_device_name(hf.device),
        "compute_type": args.compute_type,
        "torch": torch.__version__,
        "sets": {name: compare(name, args.n, hf, vl) for name in ("ami", "earnings22")},
        "samples": {},
    }
    for f in sorted(Path(args.samples).iterdir()):
        res = rumblex.transcribe(str(f), vl)
        words = [w for s in res.segments for w in s.words]
        report["samples"][f.name] = {
            "words": len(words),
            "without_probability": sum(w.probability is None for w in words),
            "fallback_chunks": res.metadata.get("alignment", {}).get("fallback_chunks"),
        }
        print(f.name, report["samples"][f.name], flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
