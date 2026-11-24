"""Word timestamps for a transcript you already have: no decoding, only the encoder and the CTC head.

The audio is cut into speech chunks as for transcription, and each chunk is encoded. To decide which words
of the text belong to which chunk, each chunk is also decoded greedily with the CTC head; those rough
words are matched against the text (difflib), and matched words pin the text to chunks. Words between two
matches that lie in different chunks are split between those chunks in proportion to the greedy words
there. Where the greedy pass garbles or drops the words at a chunk boundary, that split can be off by a
word or two, so each boundary is then moved by up to `BOUNDARY_SHIFT` words to where both chunks' text has
the best CTC alignment score. Each chunk's words are then force-aligned with the same aligner as
`rumblex.transcribe`.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from collections.abc import Callable, Sequence
from difflib import SequenceMatcher
from itertools import pairwise

import numpy as np

from rumblex.aligner import BLANK, ctc_score
from rumblex.audio import SAMPLE_RATE
from rumblex.transcriber import ChunkTranscript, speech_chunks
from rumblex.types import TranscriptResult
from rumblex.vad import SpeechChunk, SpeechRegion

_NOT_WORD = re.compile(r"[^\w']+")
BOUNDARY_SHIFT = 2  # words a chunk boundary may move when refined on the CTC scores


def word_key(w: str) -> str:
    """Lowercase, no punctuation: what two spellings of a word are compared on."""
    return _NOT_WORD.sub("", w.lower()).strip("'")


def assign_words(words: Sequence[str], chunk_words: Sequence[Sequence[str]], min_block: int = 2) -> list[int]:
    """The chunk index for each of `words`, given rough words per chunk (`chunk_words`). The result is
    non-decreasing, so every chunk gets a contiguous run of the text.

    Matches shorter than `min_block` words are not used as anchors when the text is longer than 20 words,
    since a single common word ("the") can match far from where it belongs."""
    n_chunks = len(chunk_words)
    if n_chunks == 0:
        raise ValueError("no chunks to assign words to")
    if n_chunks == 1 or not words:
        return [0] * len(words)
    hyp_chunk = [c for c, ws in enumerate(chunk_words) for _ in ws]
    hyp = [word_key(w) for ws in chunk_words for w in ws]
    ref = [word_key(w) for w in words]
    blocks = SequenceMatcher(None, ref, hyp, autojunk=False).get_matching_blocks()
    least = min_block if len(words) > 20 else 1
    anchors: list[tuple[int, int]] = [(-1, -1)]  # (ref index, hyp index)
    for b in blocks:
        if b.size >= least:
            anchors += [(b.a + k, b.b + k) for k in range(b.size)]
    anchors.append((len(ref), len(hyp)))

    out = [0] * len(words)
    for (i0, j0), (i1, j1) in pairwise(anchors):
        if 0 <= i0 < len(ref):
            out[i0] = hyp_chunk[j0]
        gap = range(i0 + 1, i1)
        if not gap:
            continue
        c0 = hyp_chunk[j0] if j0 >= 0 else 0
        c1 = hyp_chunk[j1] if j1 < len(hyp) else n_chunks - 1
        if c0 == c1:
            for i in gap:
                out[i] = c0
            continue
        # Split the gap over chunks c0..c1 by how many unmatched rough words each holds between the anchors
        # (at least a little weight each, so a chunk the greedy pass found nothing in still gets a share).
        weights = np.full(c1 - c0 + 1, 0.5)
        for j in range(j0 + 1, j1):
            weights[hyp_chunk[j] - c0] += 1
        bounds = np.cumsum(weights) / weights.sum() * len(gap)
        for k, i in enumerate(gap):
            out[i] = c0 + int(np.searchsorted(bounds, k + 0.5))
    return out


def refine_boundaries(
    words: Sequence[str],
    owner: Sequence[int],
    score: Callable[[int, str], float],
    n_chunks: int,
    shift: int = BOUNDARY_SHIFT,
) -> list[int]:
    """Move each boundary between chunks c and c + 1 by up to `shift` words to where
    score(c, text of c) + score(c + 1, text of c + 1) is highest. `owner` is non-decreasing (from
    `assign_words`); boundaries are refined left to right, and a tie keeps the boundary where it is."""
    owner = list(owner)

    def first(c: int) -> int:  # index of the first word in chunk c or later (owner is sorted)
        return bisect_left(owner, c)

    for c in range(n_chunks - 1):
        lo, b, hi = first(c), first(c + 1), first(c + 2)
        cands = range(max(lo, b - shift), min(hi, b + shift) + 1)
        if len(cands) < 2:
            continue
        totals = {k: score(c, " ".join(words[lo:k])) + score(c + 1, " ".join(words[k:hi])) for k in cands}
        best = max(cands, key=lambda k: (totals[k], k == b))
        for i in range(lo, hi):
            owner[i] = c if i < best else c + 1
    return owner


def align_text(
    model,
    aligner,
    wav: np.ndarray,
    text: str,
    *,
    chunks: list[SpeechChunk] | None = None,
    max_chunk_s: float = 20.0,
    language: str | None = None,
    progress: Callable[[int, int], None] | None = None,
    optional_punct: bool = True,
) -> TranscriptResult:
    """Align `text` to 16 kHz `wav`. `model` is a RumbleTranscriber (or VLLMTranscriber; only its encoder
    is used) and `aligner` an `Aligner`. Returns segments with word timestamps, as `transcribe` does.
    `language` is only recorded in the result. With `optional_punct`, a word without punctuation may be
    aligned as if a punctuation mark followed it (the text itself is not changed)."""
    words = text.split()
    if chunks is None:
        chunks = speech_chunks(wav, max_chunk_s, model.device)
    if not chunks and words:  # VAD found no speech: put the text over the whole file
        chunks = [SpeechChunk(0, len(wav), (SpeechRegion(0, len(wav)),))]
    states = model.encode(wav, chunks, progress) if chunks else []
    tok = model.tokenizer
    if len(chunks) > 1:
        rough = [
            tok.decode(aligner.head.greedy(st), clean_up_tokenization_spaces=False).split() for st in states
        ]
        owner = assign_words(words, rough)
        owner = refine_boundaries(words, owner, _chunk_scorer(aligner.head, tok, states), len(chunks))
    else:
        owner = [0] * len(words)
    per_chunk: list[list[str]] = [[] for _ in chunks]
    for w, c in zip(words, owner):
        per_chunk[c].append(w)
    items = []
    for chunk, st, ws in zip(chunks, states, per_chunk):
        t = " ".join(ws)
        ids = tok.encode(t, add_special_tokens=False) if t else []
        items.append(
            ChunkTranscript(chunk=chunk, text=t, language=language, token_ids=ids, encoder_states=st)
        )
    result = TranscriptResult(
        segments=[],
        language=language or "",
        metadata={
            "model": getattr(getattr(model, "model", None), "name_or_path", None)
            or getattr(model, "model_id", None),
            "duration": len(wav) / SAMPLE_RATE,
            "chunks": len(chunks),
            "aligned_text": True,
            "chunk_transcripts": items,
        },
    )
    aligner.align(result, wav, optional_punct=optional_punct)
    result.metadata.pop("chunk_transcripts", None)
    return result


def _chunk_scorer(head, tok, states) -> Callable[[int, str], float]:
    """score(c, text): CTC alignment log-prob of `text` on chunk c. Each chunk's log-probs are computed once
    and kept for the next call (boundaries are scored left to right, so two chunks are kept at a time)."""
    cache: dict[int, object] = {}

    def score(c: int, text: str) -> float:
        if c not in cache:
            for k in [k for k in cache if k < c - 1]:
                del cache[k]
            cache[c] = head.log_probs(states[c])
        lp = cache[c]
        ids = tok.encode(text, add_special_tokens=False) if text else []
        em = lp[:, [BLANK, *ids]].float().cpu().numpy()
        return ctc_score(em, len(ids), [False] + [a == b for a, b in pairwise(ids)])

    return score
