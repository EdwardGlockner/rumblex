"""Why is Dutch WER higher than expected? Run one eval set through several pipeline variants and save every
transcript, so the differences can be read, not just counted.

Variants (all with the same model, digits on, batched across utterances):
  vad_forced     Silero VAD chunks, language forced (the shipped pipeline)
  whole_forced   each utterance as one chunk (no VAD), language forced
  vad_auto       VAD chunks, language detected by the model
  whole_auto     one chunk, language detected

    python scripts/diagnose_wer.py --dataset fleurs_nl --device cuda:0 --dtype float32 --out diag.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rumblex.bench import load_eval_set, normalizer_for, wer
from rumblex.transcriber import RumbleTranscriber
from rumblex.vad import SpeechChunk, SpeechRegion, vad_chunks

GAP = np.zeros(8000, np.float32)


def chunk_all(utts, use_vad: bool, device):
    """One long array with every utterance (0.5 s gaps) and chunk lists that never cross utterances."""
    pieces, chunks, owner, base = [], [], [], 0
    for ui, u in enumerate(utts):
        n = len(u.audio)
        whole = [SpeechChunk(0, n, (SpeechRegion(0, n),))]
        cs = (vad_chunks(u.audio, device=device) or whole) if use_vad else whole
        for c in cs:
            chunks.append(SpeechChunk(c.start + base, c.end + base,
                                      tuple(SpeechRegion(r.start + base, r.end + base) for r in c.regions)))  # fmt: skip
            owner.append(ui)
        pieces += [u.audio, GAP]
        base += n + len(GAP)
    return np.concatenate(pieces), chunks, owner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fleurs_nl")
    ap.add_argument("--n", type=int, default=364)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--backend", default="transformers", choices=["transformers", "vllm"])
    ap.add_argument("--vllm-url", default="http://localhost:8000/v1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    utts, language = load_eval_set(args.dataset, args.n, seed=0)
    norm_name, norm = normalizer_for(language)
    if args.backend == "vllm":
        from rumblex.vllm_backend import VLLMTranscriber

        tr = VLLMTranscriber(args.vllm_url, concurrency=32, device=args.device, dtype=args.dtype)
    else:
        tr = RumbleTranscriber(device=args.device, dtype=args.dtype, batch_size=args.batch_size)
    refs = [u.transcript for u in utts]
    rep: dict = {"dataset": args.dataset, "n": len(utts), "language": language, "normalizer": norm_name,
                 "backend": args.backend, "dtype": args.dtype, "variants": {}}  # fmt: skip
    for name, use_vad, lang in (
        ("vad_forced", True, language),
        ("whole_forced", False, language),
        ("vad_auto", True, None),
        ("whole_auto", False, None),
    ):
        audio, chunks, owner = chunk_all(utts, use_vad, tr.device)
        items = tr.transcribe_chunks(audio, chunks, language=lang, digits=True)
        hyps = [[] for _ in utts]
        langs = [set() for _ in utts]
        for it, ui in zip(items, owner):
            hyps[ui].append(it.text)
            langs[ui].add(it.language or "?")
        hyp_text = [" ".join(h) for h in hyps]
        w = wer(refs, hyp_text, norm)
        detected = {}
        for s in langs:
            for x in s:
                detected[x] = detected.get(x, 0) + 1
        rep["variants"][name] = {**w, "chunks": len(chunks), "detected_languages": detected,
                                 "hyps": hyp_text}  # fmt: skip
        print(
            name, {k: w[k] for k in ("wer", "sub", "del", "ins")}, "chunks", len(chunks), detected, flush=True
        )
    rep["refs"] = refs
    rep["ids"] = [u.id for u in utts]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
