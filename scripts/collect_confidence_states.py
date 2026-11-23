"""Transcribe AMI and Earnings-22 with the real pipeline and save what a word confidence can be built from,
so scripts/eval_confidence.py can score confidence variants on CPU afterwards.

Per chunk it keeps the encoder states (for the CTC head), the text token ids, and the decoder's
log-prob of each text token as generated (`ChunkTranscript.token_logprobs`; the committed run took them
with a logits processor patched in here, the same computation the transcriber now does itself). Chunks are made as `rumblex.transcribe` makes them (the whole
clip up to 20 s, VAD beyond), English forced, digits off (the references write numbers as words).

Each set is a fixed random sample (seed 0) split in two by position: even positions are dev, odd are
test. Choices are made on dev and reported once on test.

    python scripts/collect_confidence_states.py --device cuda:0 --n 1200 --out data/confidence_states
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from rumblex.audio import SAMPLE_RATE
from rumblex.bench import load_eval_set
from rumblex.transcriber import DEFAULT_MODEL, RumbleTranscriber, speech_chunks
from rumblex.vad import SpeechChunk, SpeechRegion

GAP = np.zeros(SAMPLE_RATE // 2, np.float32)  # between utterances in the concatenated batch audio


def run(tr: RumbleTranscriber, utts: list[dict]) -> tuple[list[dict], float]:
    """Chunk each utterance as rumblex.transcribe does, transcribe all chunks together, regroup."""
    pieces, chunks, owner, base = [], [], [], 0
    for ui, u in enumerate(utts):
        for c in speech_chunks(u["audio"], 20.0, tr.device):
            regs = tuple(SpeechRegion(r.start + base, r.end + base) for r in c.regions)
            chunks.append(SpeechChunk(c.start + base, c.end + base, regs))
            owner.append((ui, base))
        pieces += [u["audio"], GAP]
        base += len(u["audio"]) + len(GAP)
    audio = np.concatenate(pieces)
    t0 = time.perf_counter()
    items = tr.transcribe_chunks(audio, chunks, keep_encoder_states=True, language="English", digits=False)
    dt = time.perf_counter() - t0
    out = [{**{k: v for k, v in u.items() if k != "audio"}, "duration": len(u["audio"]) / SAMPLE_RATE,
            "chunks": []} for u in utts]  # fmt: skip
    for it, (ui, b) in zip(items, owner):
        out[ui]["chunks"].append({
            "start": (it.chunk.start - b) / SAMPLE_RATE,
            "end": (it.chunk.end - b) / SAMPLE_RATE,
            "regions": [((r.start - b) / SAMPLE_RATE, (r.end - b) / SAMPLE_RATE) for r in it.chunk.regions],
            "text": it.text,
            "token_ids": it.token_ids,
            "decoder_logprobs": np.asarray(it.token_logprobs, np.float32),
            "states": it.encoder_states.half(),
        })  # fmt: skip
    return out, dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n", type=int, default=1200, help="utterances per set, half dev and half test")
    ap.add_argument("--sets", nargs="+", default=["ami", "earnings22"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tr = RumbleTranscriber(DEFAULT_MODEL, device=args.device, dtype=args.dtype, batch_size=args.batch_size)
    summary = {"device": args.device, "gpu": torch.cuda.get_device_name(tr.device) if tr.device.type == "cuda" else None,
               "dtype": str(tr.dtype), "batch_size": args.batch_size, "seed": args.seed, "torch": torch.__version__}  # fmt: skip
    for name in args.sets:
        utts, _ = load_eval_set(name, args.n, args.seed)
        utts = [{"id": u.id, "audio": u.audio, "ref_text": u.transcript, "split": ("dev", "test")[k % 2]}
                for k, u in enumerate(utts)]  # fmt: skip
        res, dt = run(tr, utts)
        torch.save(res, out / f"{name}.pt")
        speech = sum(u["duration"] for u in res)
        summary[name] = {
            "utterances": len(res),
            "audio_seconds": round(speech, 1),
            "transcribe_seconds": round(dt, 1),
        }
        print(name, summary[name], flush=True)
    if tr.device.type == "cuda":
        summary["peak_vram_gb"] = round(torch.cuda.max_memory_allocated(tr.device) / 1e9, 2)
    (out / "collect_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
