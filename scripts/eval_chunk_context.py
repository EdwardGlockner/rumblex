"""Does giving the model the previous chunk's text help on long recordings?

rumblex decodes each chunk of a long file on its own. The model takes free text in its system prompt
(`prompt=`), so the text of the chunk before can be handed to it as context. This script measures what that
does on TED-LIUM long-form (distil-whisper/tedlium-long-form: whole talks with a reference transcript;
`validation` is used as dev, `test` is reported once).

Variants, all on the chunks `rumblex.transcribe` makes (VAD, up to 20 s), English forced:

    none     no context: what the package does today
    prev     chunk k gets the text this same run produced for chunk k-1, so a talk is decoded in order
    twopass  chunk k gets the `none` text of chunk k-1: a second pass that can still be batched
    other    chunk k gets the `none` text of chunk k of another talk: context that has nothing to do with it

`collect` runs the model and saves the chunk texts (not for committing: the talks are CC BY-NC-ND).
`score` reads that file and prints numbers only: WER over whole talks, WER of the words next to a chunk
boundary against the rest, and how the text looks at boundaries.

    python scripts/eval_chunk_context.py collect --split validation --device cuda:0 --out data/chunk_context
    python scripts/eval_chunk_context.py score --split validation --out data/chunk_context \
        --json docs/chunk_context_eval.json
"""

from __future__ import annotations

import argparse
import io
import json
import time
from collections.abc import Callable, Sequence
from itertools import pairwise
from pathlib import Path

import numpy as np

REPO, REVISION = "distil-whisper/tedlium-long-form", "ea3a7847"
VARIANTS = ("none", "prev", "twopass", "other")
GAP_S = 0.5  # silence between talks in the joined audio
ZONE = 3  # words on each side of a chunk boundary that count as "next to" it


def load_talks(split: str, limit: int | None = None) -> list[dict]:
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(
        f for f in HfApi().list_repo_files(REPO, repo_type="dataset", revision=REVISION)
        if f.startswith(f"data/{split}-") and f.endswith(".parquet")
    )  # fmt: skip
    talks = []
    for f in files:
        t = pq.read_table(hf_hub_download(REPO, f, repo_type="dataset", revision=REVISION))
        for i in range(t.num_rows):
            a = t.column("audio")[i].as_py()
            wav, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
            if wav.ndim == 2:
                wav = wav.mean(axis=1)
            if sr != 16000:
                raise ValueError(f"{a['path']}: expected 16 kHz, got {sr}")
            talks.append(
                {"id": a["path"].removesuffix(".wav"), "audio": wav, "ref": t.column("text")[i].as_py()}
            )
    return talks[:limit] if limit else talks


def context_for(variant: str, texts: dict[str, list[list[str]]], talk: int, k: int) -> str | None:
    """The prompt chunk `k` of `talk` gets under `variant`. `texts[variant][talk]` holds the chunk texts
    known so far. No context for the first chunk of a talk, or when the text it would use is empty."""
    if variant == "none":
        return None
    if variant == "prev":
        src = texts["prev"][talk][k - 1] if k else ""
    elif variant == "twopass":
        src = texts["none"][talk][k - 1] if k else ""
    elif variant == "other":
        other = texts["none"][(talk + 1) % len(texts["none"])]
        src = other[k % len(other)] if other else ""
    else:
        raise ValueError(variant)
    return src or None


def decode_order(variant: str, sizes: Sequence[int]) -> list[list[tuple[int, int]]]:
    """Groups of (talk, chunk) that can be decoded together. `prev` needs chunk k-1 of the same talk first,
    so a group is chunk k of every talk that has one; the others have no such order."""
    if variant == "prev":
        return [[(t, k) for t, n in enumerate(sizes) if k < n] for k in range(max(sizes, default=0))]
    return [[(t, k) for t, n in enumerate(sizes) for k in range(n)]]


def collect(args: argparse.Namespace) -> None:
    import torch

    from rumblex.audio import SAMPLE_RATE
    from rumblex.transcriber import DEFAULT_MODEL, RumbleTranscriber, make_batches, speech_chunks
    from rumblex.vad import SpeechChunk

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.split}.json"
    talks = load_talks(args.split, args.limit)
    tr = RumbleTranscriber(DEFAULT_MODEL, device=args.device, dtype=args.dtype, batch_size=args.batch_size)

    gap = np.zeros(int(GAP_S * SAMPLE_RATE), np.float32)
    pieces, chunks, base = [], [], 0
    for t in talks:
        own = speech_chunks(t["audio"], 20.0, tr.device)
        t["chunks"] = [(c.start / SAMPLE_RATE, c.end / SAMPLE_RATE) for c in own]
        chunks.append([SpeechChunk(c.start + base, c.end + base) for c in own])
        pieces += [t["audio"], gap]
        base += len(t["audio"]) + len(gap)
    audio = np.concatenate(pieces)
    sizes = [len(c) for c in chunks]

    state = {"texts": {}, "seconds": {}}
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if [t["id"] for t in old["talks"]] == [t["id"] for t in talks] and old["sizes"] == sizes:
            state = {"texts": old["texts"], "seconds": old["seconds"]}
    meta = {
        "dataset": REPO, "revision": REVISION, "split": args.split, "device": args.device,
        "gpu": torch.cuda.get_device_name(tr.device) if tr.device.type == "cuda" else None,
        "dtype": str(tr.dtype), "batch_size": args.batch_size, "torch": torch.__version__,
        "audio_s": round(sum(len(t["audio"]) for t in talks) / SAMPLE_RATE, 1),
    }  # fmt: skip
    options = {"digits": True, "filler": False, "verbatim": True}

    for variant in args.variants:
        if variant in state["texts"]:
            print(f"{variant}: already done", flush=True)
            continue
        texts = state["texts"] | {variant: [[""] * n for n in sizes]}
        t0 = time.perf_counter()
        for group in decode_order(variant, sizes):
            for idx in make_batches([chunks[t][k].duration for t, k in group], args.batch_size):
                batch = [group[i] for i in idx]
                prompts = [context_for(variant, texts, t, k) for t, k in batch]
                res = tr._run_batch(
                    audio, [chunks[t][k] for t, k in batch], "English",
                    prompts if any(prompts) else None, options, False,
                )  # fmt: skip
                for (t, k), r in zip(batch, res):
                    texts[variant][t][k] = r.text
        state["texts"][variant] = texts[variant]
        state["seconds"][variant] = round(time.perf_counter() - t0, 1)
        print(f"{variant}: {sum(sizes)} chunks in {state['seconds'][variant]} s", flush=True)
        doc = {
            "meta": meta, "sizes": sizes, **state,
            "talks": [{"id": t["id"], "ref": t["ref"], "chunks": t["chunks"]} for t in talks],
        }  # fmt: skip
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc), encoding="utf-8")
        tmp.replace(path)


def edge_distance(sizes: Sequence[int]) -> list[int]:
    """For each word of chunks with `sizes` words, how many words lie between it and the nearest boundary
    between two chunks (0 = right at one). The start and the end of the recording are not boundaries."""
    n = sum(sizes)
    dist = [n] * n
    pos = 0
    for size in sizes[:-1]:
        pos += size
        for i in range(n):
            dist[i] = min(dist[i], pos - 1 - i if i < pos else i - pos)
    return dist


def zone_errors(ref: Sequence[str], hyp_chunks: Sequence[Sequence[str]], zone: int = ZONE) -> dict:
    """Word errors split by where they fall in the hypothesis: within `zone` words of a chunk boundary, or
    not. A deleted reference word counts for the place in the hypothesis where it is missing."""
    import jiwer

    hyp = [w for c in hyp_chunks for w in c]
    dist = edge_distance([len(c) for c in hyp_chunks])
    near = [d < zone for d in dist]
    out = {"near": {"errors": 0, "ref_words": 0}, "far": {"errors": 0, "ref_words": 0}}
    if not ref or not hyp:
        out["far"] = {"errors": max(len(ref), len(hyp)), "ref_words": len(ref)}
        return out

    def side(h: int) -> dict:
        around = [near[i] for i in (h - 1, h) if 0 <= i < len(hyp)]
        return out["near" if any(around) else "far"]

    for op in jiwer.process_words(" ".join(ref), " ".join(hyp)).alignments[0]:
        if op.type in ("equal", "substitute"):
            for h in range(op.hyp_start_idx, op.hyp_end_idx):
                s = out["near" if near[h] else "far"]
                s["ref_words"] += 1
                s["errors"] += op.type == "substitute"
        elif op.type == "delete":
            s = side(op.hyp_start_idx)
            s["ref_words"] += op.ref_end_idx - op.ref_start_idx
            s["errors"] += op.ref_end_idx - op.ref_start_idx
        else:
            for h in range(op.hyp_start_idx, op.hyp_end_idx):
                out["near" if near[h] else "far"]["errors"] += 1
    return out


def boundary_shape(chunk_texts: Sequence[str]) -> dict:
    """How the raw text looks where two non-empty chunks meet."""
    texts = [t.strip() for t in chunk_texts if t.strip()]
    pairs = list(pairwise(texts))
    return {
        "boundaries": len(pairs),
        "ends_sentence": sum(a[-1] in ".?!" for a, _ in pairs),
        "starts_lowercase": sum(b[0].islower() for _, b in pairs),
    }


def score_variant(talks: Sequence[dict], texts: Sequence[Sequence[str]], norm: Callable[[str], str]) -> dict:
    tot = {"errors": 0, "ref_words": 0}
    zones = {"near": {"errors": 0, "ref_words": 0}, "far": {"errors": 0, "ref_words": 0}}
    shape = {"boundaries": 0, "ends_sentence": 0, "starts_lowercase": 0}
    per_talk = {}
    for talk, chunk_texts in zip(talks, texts):
        ref = norm(talk["ref"]).split()
        z = zone_errors(ref, [norm(t).split() for t in chunk_texts])
        errors = z["near"]["errors"] + z["far"]["errors"]
        per_talk[talk["id"]] = round(errors / len(ref), 4)
        tot["errors"] += errors
        tot["ref_words"] += len(ref)
        for side, counts in zones.items():
            for key in counts:
                counts[key] += z[side][key]
        for key, v in boundary_shape(chunk_texts).items():
            shape[key] += v
    rate = lambda d: round(d["errors"] / d["ref_words"], 4) if d["ref_words"] else None
    return {
        "wer": rate(tot), **tot,
        "wer_near_boundary": rate(zones["near"]), "near": zones["near"],
        "wer_elsewhere": rate(zones["far"]), "far": zones["far"],
        "boundary_text": shape, "wer_per_talk": per_talk,
    }  # fmt: skip


def score(args: argparse.Namespace) -> None:
    from rumblex.bench import english_normalizer

    doc = json.loads((Path(args.out) / f"{args.split}.json").read_text(encoding="utf-8"))
    name, norm = english_normalizer()
    result = {
        "script": "scripts/eval_chunk_context.py", **doc["meta"], "normalizer": name, "zone_words": ZONE,
        "talks": len(doc["talks"]), "chunks": sum(doc["sizes"]), "seconds": doc["seconds"],
        "variants": {v: score_variant(doc["talks"], t, norm) for v, t in doc["texts"].items()},
    }  # fmt: skip
    for v, r in result["variants"].items():
        b = r["boundary_text"]
        print(
            f"{v:8s} WER {r['wer']:.4f}  near boundary {r['wer_near_boundary']:.4f}  elsewhere "
            f"{r['wer_elsewhere']:.4f}  ends sentence {b['ends_sentence']}/{b['boundaries']}  "
            f"starts lowercase {b['starts_lowercase']}/{b['boundaries']}  {doc['seconds'].get(v)} s"
        )
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--device", default="cuda:0")
    c.add_argument("--dtype", default="float32")
    c.add_argument("--batch-size", type=int, default=8)
    c.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    c.add_argument("--limit", type=int, default=None, help="only the first N talks (for a quick try)")
    s = sub.add_parser("score")
    s.add_argument("--json", default=None, help="write the numbers here")
    for p in (c, s):
        p.add_argument("--split", default="validation", choices=("validation", "test"))
        p.add_argument("--out", required=True, help="folder for the saved chunk texts")
    args = ap.parse_args()
    {"collect": collect, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
