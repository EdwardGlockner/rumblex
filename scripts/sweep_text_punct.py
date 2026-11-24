"""Optional punctuation when aligning a given transcript: sweep `punct_penalty` on LibriSpeech dev_clean.

The text given to `align_text` is LibriSpeech's reference transcript (no punctuation). Each utterance's
VAD chunks and encoder states come from scripts/collect_alignment_states.py, so nothing is encoded here and
it runs on CPU. Rows:
  model text      the model's own punctuated transcript, aligned as `rumblex.transcribe` does
  given, off      the reference transcript without optional punctuation (what 0.4.1 does)
  given, p=X      the reference transcript with optional punctuation, punct_penalty = X
Word times are scored against the MFA alignments as in docs/alignment_eval.md. With `--test` the same rows
are also scored on test_clean, for the penalties given with `--test-penalties` only.

    python scripts/sweep_text_punct.py --states data/align_states --out docs/text_punct_sweep.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rumblex.aligner import Aligner, AlignOptions, CTCHead
from rumblex.audio import SAMPLE_RATE
from rumblex.bench import TimedWord, TimingScore, iter_librispeech, score_timings
from rumblex.textalign import align_text
from rumblex.transcriber import ChunkTranscript
from rumblex.types import TranscriptResult
from rumblex.vad import SpeechChunk, SpeechRegion

PENALTIES = (0.0, 2.5, 5.0, 7.5, 10.0, 15.0)


class Cached:
    """Stands in for the model: hands `align_text` the saved encoder states of one utterance."""

    device = "cpu"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.states: list[torch.Tensor] = []

    def encode(self, wav, chunks, progress=None):
        return self.states


def chunks_of(u: dict) -> list[SpeechChunk]:
    def n(t: float) -> int:
        return round(t * SAMPLE_RATE)

    return [
        SpeechChunk(n(c["start"]), n(c["end"]), tuple(SpeechRegion(n(a), n(b)) for a, b in c["regions"]))
        for c in u["chunks"]
    ]


def timed(res: TranscriptResult) -> list[TimedWord]:
    return [TimedWord(w.word, w.start, w.end) for seg in res.segments for w in seg.words]


def score(utts, audio, model: Cached, head: CTCHead, penalty: float | None, own_text: bool = False) -> dict:
    aligner = Aligner(head, model.tokenizer, AlignOptions(punct_penalty=penalty or 0.0))
    total = TimingScore()
    for u in utts:
        wav, chunks = audio[u["id"]], chunks_of(u)
        states = [c["states"].float() for c in u["chunks"]]
        if own_text:
            items = [
                ChunkTranscript(chunk=ch, text=c["text"], language=None, token_ids=c["token_ids"],
                                encoder_states=st)
                for ch, c, st in zip(chunks, u["chunks"], states)
            ]  # fmt: skip
            res = aligner.align(TranscriptResult([], "", {"chunk_transcripts": items}), wav)
        else:
            model.states = states
            res = align_text(
                model, aligner, wav, u["ref_text"], chunks=chunks, optional_punct=penalty is not None
            )
        total.add(score_timings([TimedWord(*w) for w in u["ref_words"]], timed(res)))
    return {"words": total.n_matched, **total.summary()}


def row(name: str, s: dict) -> str:
    return (
        f"{name:22s} words {s['words']:5d} | start MAE {s['start_mae_ms']:5.1f} med "
        f"{s['start_median_ae_ms']:4.1f} | end MAE {s['end_mae_ms']:5.1f} med {s['end_median_ae_ms']:4.1f} "
        f"signed {s['end_mean_signed_ms']:+5.1f} <=100 {100 * s['end_within_100ms']:4.1f}%"
    )


def run_split(states: Path, split: str, model, head, penalties) -> dict:
    utts = torch.load(states / f"{split}.pt", weights_only=False)
    audio = {u.id: u.audio for u in iter_librispeech(split, ids=[u["id"] for u in utts])}
    out = {"model text": score(utts, audio, model, head, None, own_text=True)}
    print(row(f"{split} model text", out["model text"]), flush=True)
    for p in (None, *penalties):
        name = "given, off" if p is None else f"given, p={p:g}"
        out[name] = score(utts, audio, model, head, p)
        print(row(f"{split} {name}", out[name]), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--states", type=Path, required=True, help="folder with dev_clean.pt and test_clean.pt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--test", action="store_true", help="also score test_clean")
    ap.add_argument("--test-penalties", type=float, nargs="*", default=[])
    args = ap.parse_args()

    from transformers import AutoProcessor

    model = Cached(AutoProcessor.from_pretrained("Reson8/Rumble-2-Flash").tokenizer)
    head = CTCHead.from_pretrained(device="cpu")
    out = {"dev_clean": run_split(args.states, "dev_clean", model, head, PENALTIES)}
    if args.test:
        out["test_clean"] = run_split(args.states, "test_clean", model, head, args.test_penalties)
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
