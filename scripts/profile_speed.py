"""Where does the time go, and which batch size / chunk length is fastest without hurting WER?

Builds --seconds of real speech from LibriSpeech test_clean (utterances back to back, 0.5 s gaps), then:
  1. times Silero VAD (CPU) for each --chunk length,
  2. for every (chunk length, batch size): times transcription (encoder states kept, as when aligning),
     alignment with the loudness refinement, peak GPU memory, and WER against the joined references.

    python scripts/profile_speed.py --device cuda:0 --dtype float32 --out profile.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from rumblex.aligner import Aligner
from rumblex.bench import english_normalizer, iter_librispeech, wer
from rumblex.transcriber import RumbleTranscriber
from rumblex.vad import vad_chunks


def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--seconds", type=float, default=600)
    ap.add_argument("--chunks", type=float, nargs="+", default=[15, 20, 30])
    ap.add_argument("--batches", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Real speech plus its reference text, stopping once --seconds is reached.
    pieces, refs, total = [], [], 0
    gap = np.zeros(8000, np.float32)
    for u in iter_librispeech("test_clean"):
        pieces += [u.audio, gap]
        refs.append(u.transcript)
        total += len(u.audio) + len(gap)
        if total >= args.seconds * 16000:
            break
    audio = np.concatenate(pieces)
    dur = len(audio) / 16000
    norm_name, norm = english_normalizer()
    ref_text = " ".join(refs)

    tr = RumbleTranscriber(device=args.device, dtype=args.dtype)
    al = Aligner.from_transcriber(tr)
    rep: dict = {"audio_seconds": round(dur, 1), "device": str(tr.device), "dtype": str(tr.dtype),
                 "gpu": torch.cuda.get_device_name(tr.device) if tr.device.type == "cuda" else None,
                 "torch_threads": torch.get_num_threads(), "wer_normalizer": norm_name, "vad": {}, "runs": []}  # fmt: skip

    chunk_sets = {}
    for c in args.chunks:
        t0 = time.perf_counter()
        chunk_sets[c] = vad_chunks(audio, max_chunk_s=c)
        rep["vad"][str(c)] = {"seconds": round(time.perf_counter() - t0, 2), "chunks": len(chunk_sets[c])}
        print("vad", c, rep["vad"][str(c)], flush=True)

    tr.transcribe_chunks(audio, chunk_sets[args.chunks[0]][:2])  # warm-up
    for c, bs in itertools.product(args.chunks, args.batches):
        tr.batch_size = bs
        if tr.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(tr.device)
        try:
            sync(tr.device)
            t0 = time.perf_counter()
            items = tr.transcribe_chunks(audio, chunk_sets[c], keep_encoder_states=True)
            sync(tr.device)
            t_tr = time.perf_counter() - t0
        except torch.OutOfMemoryError:
            rep["runs"].append({"chunk_s": c, "batch": bs, "oom": True})
            print("oom", c, bs, flush=True)
            continue
        from rumblex.types import Segment, TranscriptResult

        res = TranscriptResult([Segment(i.chunk.start_s, i.chunk.end_s, i.text) for i in items], "English",
                               {"chunk_transcripts": items})  # fmt: skip
        t0 = time.perf_counter()
        al.align(res, audio)
        t_al = time.perf_counter() - t0
        hyp = " ".join(i.text for i in items)
        row = {
            "chunk_s": c,
            "batch": bs,
            "chunks": len(items),
            "transcribe_s": round(t_tr, 2),
            "align_s": round(t_al, 2),
            "rtfx_model": round(dur / (t_tr + t_al), 1),
            "rtfx_with_vad": round(dur / (t_tr + t_al + rep["vad"][str(c)]["seconds"]), 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated(tr.device) / 1e9, 2)
            if tr.device.type == "cuda" else None,
            **{k: v for k, v in wer([ref_text], [hyp], norm).items() if k in ("wer", "ref_words")},
        }  # fmt: skip
        rep["runs"].append(row)
        print(json.dumps(row), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
