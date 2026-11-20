"""Transcribe LibriSpeech (and the FLEURS Dutch clips) with the real pipeline and save the encoder states,
so alignment variants can be scored on CPU afterwards (scripts/eval_alignment.py).

Pipeline per utterance: Silero VAD chunks -> RumbleTranscriber (batched across utterances,
keep_encoder_states=True). LibriSpeech is run with language="English" and digits=False, since its
references spell numbers out. Dutch is run with automatic language detection.

    python scripts/collect_alignment_states.py --device cuda:0 --out data/align_states
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from rumblex.audio import SAMPLE_RATE, load_audio
from rumblex.bench import iter_librispeech, sample_ids
from rumblex.transcriber import DEFAULT_MODEL, RumbleTranscriber
from rumblex.vad import SpeechChunk, SpeechRegion, vad_chunks

GAP = np.zeros(SAMPLE_RATE // 2, np.float32)  # between utterances in the concatenated batch audio


def run(tr: RumbleTranscriber, utts: list[dict], **kw) -> tuple[list[dict], float, float]:
    """Chunk each utterance with VAD, transcribe all chunks together, regroup per utterance."""
    pieces, chunks, owner, base = [], [], [], 0
    for ui, u in enumerate(utts):
        cs = vad_chunks(u["audio"]) or [SpeechChunk(0, len(u["audio"]), (SpeechRegion(0, len(u["audio"])),))]
        for c in cs:
            regs = tuple(SpeechRegion(r.start + base, r.end + base) for r in c.regions)
            chunks.append(SpeechChunk(c.start + base, c.end + base, regs))
            owner.append((ui, base))
        pieces += [u["audio"], GAP]
        base += len(u["audio"]) + len(GAP)
    audio = np.concatenate(pieces)
    if tr.device.type == "cuda":
        torch.cuda.synchronize(tr.device)
    t0 = time.perf_counter()
    items = tr.transcribe_chunks(audio, chunks, keep_encoder_states=True, **kw)
    if tr.device.type == "cuda":
        torch.cuda.synchronize(tr.device)
    dt = time.perf_counter() - t0
    out = [{**{k: v for k, v in u.items() if k != "audio"}, "duration": len(u["audio"]) / SAMPLE_RATE, "chunks": []}
           for u in utts]  # fmt: skip
    for it, (ui, b) in zip(items, owner):
        out[ui]["chunks"].append({
            "start": (it.chunk.start - b) / SAMPLE_RATE,
            "end": (it.chunk.end - b) / SAMPLE_RATE,
            "regions": [((r.start - b) / SAMPLE_RATE, (r.end - b) / SAMPLE_RATE) for r in it.chunk.regions],
            "text": it.text,
            "language": it.language,
            "token_ids": it.token_ids,
            "states": it.encoder_states.half(),
        })  # fmt: skip
    speech_s = sum(u["duration"] for u in out)
    return out, dt, speech_s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dev-n", type=int, default=100)
    ap.add_argument("--test-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fleurs", default=None, help="folder with audio/ and transcripts.tsv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tr = RumbleTranscriber(DEFAULT_MODEL, device=args.device, dtype=args.dtype, batch_size=args.batch_size)
    summary = {"device": args.device, "gpu": torch.cuda.get_device_name(tr.device) if tr.device.type == "cuda" else None,
               "dtype": str(tr.dtype), "batch_size": args.batch_size, "seed": args.seed, "torch": torch.__version__}  # fmt: skip

    for split, n in (("dev_clean", args.dev_n), ("test_clean", args.test_n)):
        if n <= 0:
            continue
        ids = sample_ids(split, n, args.seed)
        utts = [
            {"id": u.id, "audio": u.audio, "ref_text": u.transcript,
             "ref_words": [(w.word, w.start, w.end) for w in u.words]}
            for u in iter_librispeech(split, ids=ids)
        ]  # fmt: skip
        res, dt, speech = run(tr, utts, language="English", digits=False)
        torch.save(res, out / f"{split}.pt")
        summary[split] = {"utterances": len(res), "audio_seconds": round(speech, 1), "transcribe_seconds": round(dt, 2),
                          "rtfx_incl_gaps": round(speech / dt, 1)}  # fmt: skip
        print(split, summary[split], flush=True)

    if args.fleurs:
        root = Path(args.fleurs)
        with open(root / "transcripts.tsv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        utts = [{"id": r["file"], "audio": load_audio(root / "audio" / r["file"]), "ref_text": r["transcript"]}
                for r in rows]  # fmt: skip
        res, dt, speech = run(tr, utts)
        torch.save(res, out / "fleurs_nl.pt")
        summary["fleurs_nl"] = {"utterances": len(res), "audio_seconds": round(speech, 1),
                                "transcribe_seconds": round(dt, 2)}  # fmt: skip
        print("fleurs_nl", summary["fleurs_nl"], flush=True)

    if tr.device.type == "cuda":
        summary["peak_vram_gb"] = round(torch.cuda.max_memory_allocated(tr.device) / 1e9, 2)
    (out / "collect_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
