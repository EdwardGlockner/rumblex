"""Check RumbleTranscriber on real audio and time it.

Builds one long recording from the repo samples (2 s of silence between them), runs VAD, then:
- transcribes the chunks with batch_size=1 and with --batch-size, and compares text and token ids;
- compares the encoder states captured during generate() with RumbleCTC.ln_post_states (the reference
  encoder pass in the checkpoint's ctc_head.py) on the same chunk;
- checks a forced language comes back by name.

    python scripts/check_transcriber.py --dtype float32 --out docs/transcriber_check_gpu_fp32.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download

from rumblex.audio import SAMPLE_RATE, load_audio
from rumblex.transcriber import DEFAULT_MODEL, RumbleTranscriber, prepare_clip
from rumblex.vad import vad_chunks

SAMPLES = ["librispeech_mr_quilter.wav", "bcn_weather.mp3", "obama_first_45_secs.mp3", "two_speakers.wav"]


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1] / "samples"
    gap = np.zeros(2 * SAMPLE_RATE, np.float32)
    wav = np.concatenate([x for n in SAMPLES for x in (load_audio(root / n), gap)])
    chunks = vad_chunks(wav)
    rep: dict = {
        "device": args.device,
        "torch": torch.__version__,
        "audio_seconds": round(len(wav) / 16000, 2),
        "chunks": [[round(c.start_s, 2), round(c.end_s, 2)] for c in chunks],
    }

    t0 = time.perf_counter()
    tr = RumbleTranscriber(DEFAULT_MODEL, device=args.device, dtype=args.dtype, batch_size=1)
    rep["load_seconds"] = round(time.perf_counter() - t0, 2)
    rep["dtype"] = str(tr.dtype)

    runs = {}
    for bs in (1, args.batch_size):
        tr.batch_size = bs
        if tr.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(tr.device)
        sync(tr.device)
        t0 = time.perf_counter()
        items = tr.transcribe_chunks(wav, chunks, keep_encoder_states=True)
        sync(tr.device)
        dt = time.perf_counter() - t0
        runs[bs] = items
        rep[f"batch{bs}"] = {
            "seconds": round(dt, 3),
            "rtfx": round(len(wav) / 16000 / dt, 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated(tr.device) / 1e9, 2)
            if tr.device.type == "cuda"
            else None,
        }

    one, many = runs[1], runs[args.batch_size]
    rep["texts"] = [
        {
            "chunk": [round(a.chunk.start_s, 2), round(a.chunk.end_s, 2)],
            "language": a.language,
            "text": a.text,
        }
        for a in one
    ]
    rep["batched_vs_single"] = {
        "text_equal": sum(a.text == b.text for a, b in zip(one, many)),
        "token_ids_equal": sum(a.token_ids == b.token_ids for a, b in zip(one, many)),
        "language_equal": sum(a.language == b.language for a, b in zip(one, many)),
        "n": len(one),
        "states_shape_mismatches": [
            [tuple(a.encoder_states.shape), tuple(b.encoder_states.shape)]
            for a, b in zip(one, many)
            if a.encoder_states.shape != b.encoder_states.shape
        ],
        "states_max_abs_diff": max(
            (a.encoder_states - b.encoder_states).abs().max().item()
            for a, b in zip(one, many)
            if a.encoder_states.shape == b.encoder_states.shape
        ),
        "differing_texts": [
            {"single": a.text, "batched": b.text} for a, b in zip(one, many) if a.text != b.text
        ],
    }
    rep["frames_per_second"] = [round(a.encoder_states.shape[0] / a.chunk.duration, 3) for a in one]

    # Reference encoder pass from the checkpoint's own ctc_head.py.
    snap = Path(snapshot_download(DEFAULT_MODEL))
    spec = importlib.util.spec_from_file_location("rumble_ctc_head", snap / "ctc_head.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ctc = mod.RumbleCTC.from_pretrained(str(snap), tr.model, tr.processor)
    diffs, argmax_agree = [], []
    for a in one:
        ref = ctc.ln_post_states(ctc.log_mel(prepare_clip(a.chunk.slice(wav)))).float().cpu()
        if ref.shape != a.encoder_states.shape:
            diffs.append(f"shape {tuple(ref.shape)} vs {tuple(a.encoder_states.shape)}")
            continue
        diffs.append(round((ref - a.encoder_states).abs().max().item(), 5))
        la = ctc.logits(a.encoder_states.to(tr.device)).argmax(-1)
        lr = ctc.logits(ref.to(tr.device)).argmax(-1)
        argmax_agree.append(round((la == lr).float().mean().item(), 4))
    rep["captured_vs_reference_states"] = {"max_abs_diff": diffs, "ctc_argmax_agreement": argmax_agree}

    tr.batch_size = args.batch_size
    forced = tr.transcribe_chunks(wav, chunks[:2], language="en")
    rep["forced_language_en"] = [a.language for a in forced]

    print(json.dumps(rep, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
