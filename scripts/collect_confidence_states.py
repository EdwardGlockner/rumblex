"""Transcribe AMI and Earnings-22 with the real pipeline and save what a word confidence can be built from,
so scripts/eval_confidence.py can score confidence variants on CPU afterwards.

Per chunk it keeps the encoder states (for the CTC head), the text token ids, and the decoder's
log-prob of each text token as generated. Chunks are made as `rumblex.transcribe` makes them (the whole
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
from rumblex.transcriber import (
    ASR_TEXT_ID,
    DEFAULT_MODEL,
    FIRST_SPECIAL_ID,
    STOP_IDS,
    RumbleTranscriber,
    speech_chunks,
)
from rumblex.vad import SpeechChunk, SpeechRegion

GAP = np.zeros(SAMPLE_RATE // 2, np.float32)  # between utterances in the concatenated batch audio


def text_positions(row: list[int]) -> list[int]:
    """Positions of the text tokens in one generated row, as `parse_generated` picks them."""
    end = next((p for p, t in enumerate(row) if t in STOP_IDS), len(row))
    start = row.index(ASR_TEXT_ID, 0, end) + 1 if ASR_TEXT_ID in row[:end] else 0
    return [p for p in range(start, end) if row[p] < FIRST_SPECIAL_ID]


class _Recorder:
    """Logits processor that keeps, per step, each row's best token and its log-prob. Generation is greedy,
    so that is the token generate() picks; `run` checks this against the generated ids."""

    def __init__(self) -> None:
        self.steps: list[tuple[torch.Tensor, torch.Tensor]] = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        lp, tok = scores.float().log_softmax(-1).max(-1)
        self.steps.append((tok.cpu(), lp.cpu()))
        return scores


def capture_decoder_logprobs(tr: RumbleTranscriber) -> None:
    """Patch the transcriber so every ChunkTranscript gets `.decoder_logprobs`: the decoder's log-prob of
    each of its text tokens when it was generated."""
    from transformers import LogitsProcessorList

    generate, run_batch = tr.model.generate, tr._run_batch
    rec = _Recorder()

    def gen(**kw):
        rec.steps.clear()
        return generate(**kw, logits_processor=LogitsProcessorList([rec]))

    def run(*a, **kw):
        res = run_batch(*a, **kw)
        toks = torch.stack([t for t, _ in rec.steps], 1).tolist()
        lps = torch.stack([lp for _, lp in rec.steps], 1).numpy()
        for r, row, lp in zip(res, toks, lps):
            pos = text_positions(row)
            if [row[p] for p in pos] != list(r.token_ids):
                raise RuntimeError("decoder log-probs don't line up with the text tokens")
            r.decoder_logprobs = lp[pos].astype(np.float32)
        return res

    tr.model.generate, tr._run_batch = gen, run


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
            "decoder_logprobs": it.decoder_logprobs,
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
    capture_decoder_logprobs(tr)
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
