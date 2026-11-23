"""Batched transcription of VAD chunks with Rumble-2-Flash (transformers backend).

Chunks are sorted by length and run through `model.generate()` in batches. A forward hook on the audio
encoder's `ln_post` captures the encoder states of each batch (the encoder runs once per generate call),
so the CTC aligner can reuse them instead of encoding the audio a second time. A logits processor keeps
the decoder's log-prob of every generated token, which becomes each word's `probability`.
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

from rumblex.audio import SAMPLE_RATE, AudioInput, load_audio
from rumblex.types import Segment, TranscriptResult
from rumblex.vad import SpeechChunk, SpeechRegion, vad_chunks

DEFAULT_MODEL = "Reson8/Rumble-2-Flash"

# Tokenizer layout (docs/model_inspection.md): ids below FIRST_SPECIAL_ID are text, the rest are added tokens.
FIRST_SPECIAL_ID = 99562
EOS_ID = 99564  # <|im_end|>
STOP_IDS = {99562, EOS_ID}  # generation_config.json: <|endoftext|> and <|im_end|> both end generation
ASR_TEXT_ID = 99623  # <asr_text>

# Generation cap per chunk, to stop runaway repetition. Speech produces roughly 6 text tokens per second
# (docs/model_inspection.md), so 15/s leaves wide headroom; a chunk that hits the cap triggers a warning.
MAX_TOKENS_PER_SECOND = 15
MAX_TOKENS_OVERHEAD = 16

# Mel hop in samples. Two things make a padded batch encode a clip differently from the same clip alone:
# a padded clip gets ceil(n / HOP) mel frames instead of floor(n / HOP), and its last STFT window sees the
# batch's zero padding instead of reflect padding. Trimming to whole hops and appending END_PAD samples
# of silence makes a chunk's features identical whatever it is batched with.
HOP = 160
END_PAD = 3 * HOP  # >= n_fft / 2 (200), so the last real frame's window lies inside our own silence


def trim_to_hop(x: np.ndarray) -> np.ndarray:
    return x[: len(x) // HOP * HOP]


def prepare_clip(x: np.ndarray) -> np.ndarray:
    return np.concatenate([trim_to_hop(x), np.zeros(END_PAD, dtype=np.float32)])


def zero_feature_padding(inputs) -> None:
    """Set masked (padding) mel frames to 0.0. A clip alone is padded to its last 100-frame conv block
    with 0.0, but inside a batch with the log-mel of silent audio; the convs read that padding, so the
    clip's last block encoded differently when batched. 0.0 is also what the checkpoint's ctc_head.py uses."""
    mask = inputs["input_features_mask"]
    inputs["input_features"] = inputs["input_features"] * mask[:, None, :].to(inputs["input_features"].dtype)


@dataclass
class ChunkTranscript:
    chunk: SpeechChunk
    text: str
    language: str | None
    token_ids: list[int]  # text tokens only (after <asr_text>, no specials): the CTC alignment targets
    encoder_states: torch.Tensor | None = None  # [frames, 896] ln_post output on CPU, ~13 frames/s
    token_logprobs: list[float] | None = None  # decoder log-prob of each of token_ids when it was generated


def default_dtype(device: str | torch.device) -> torch.dtype:
    """bf16 on Ampere and newer, float32 elsewhere (Pascal has no bf16 and slow fp16)."""
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8:
        return torch.bfloat16
    return torch.float32


_DTYPES = {
    "float32": torch.float32, "fp32": torch.float32,
    "float16": torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
}  # fmt: skip


def resolve_dtype(dtype: torch.dtype | str | None) -> torch.dtype | None:
    """'float16' / 'fp16' / torch.float16 -> torch.float16 (case-insensitive). None stays None."""
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    try:
        return _DTYPES[str(dtype).strip().lower()]
    except KeyError:
        raise ValueError(f"unknown compute type {dtype!r}; use one of: {', '.join(_DTYPES)}") from None


@contextmanager
def quiet_loading() -> Iterator[None]:
    """Hide transformers' "Loading weights" bar and load-time logs, then restore the caller's settings."""
    from transformers.utils import logging as hf_logging

    verbosity, bars = hf_logging.get_verbosity(), hf_logging.is_progress_bar_enabled()
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    try:
        yield
    finally:
        hf_logging.set_verbosity(verbosity)
        if bars:
            hf_logging.enable_progress_bar()


def language_name(language: str | None) -> str | None:
    """'nl' / 'dutch' / 'Dutch' -> 'Dutch', resolved the same way the transformers processor does."""
    if language is None:
        return None
    from transformers.audio_utils import prepare_language_inputs
    from transformers.models.qwen3_asr.processing_qwen3_asr import LANGUAGE_CODE_TO_NAME

    return prepare_language_inputs(language, 1, LANGUAGE_CODE_TO_NAME, return_code=False)[0]


def parse_generated(
    ids: Sequence[int],
    decode: Callable[[list[int]], str],
    forced_language: str | None = None,
) -> tuple[list[int], str | None]:
    """Split one row of generated ids into (text token ids, language).

    Unforced output looks like `language English<asr_text>text...<|im_end|><pad>...`; `decode` turns the
    prefix ids into that `language English` string. With a forced language the prefix is part of the
    prompt, so the row is just the text. `language None` is the model's answer for empty audio.
    """
    ids = list(ids)
    pos = text_positions(ids)
    language = forced_language
    if forced_language is None:
        end = ids.index(EOS_ID) if EOS_ID in ids else len(ids)
        if ASR_TEXT_ID in ids[:end]:
            language = language_from_prefix(decode(ids[: ids.index(ASR_TEXT_ID)]))
    return [ids[p] for p in pos], language


def text_positions(ids: Sequence[int]) -> list[int]:
    """Positions in a generated row of the text tokens `parse_generated` returns: after `<asr_text>` (if
    any), before `<|im_end|>`, specials dropped."""
    ids = list(ids)
    end = ids.index(EOS_ID) if EOS_ID in ids else len(ids)
    start = ids.index(ASR_TEXT_ID, 0, end) + 1 if ASR_TEXT_ID in ids[:end] else 0
    return [p for p in range(start, end) if ids[p] < FIRST_SPECIAL_ID]


class _TokenLogprobs:
    """Logits processor keeping, per step, each row's log-prob of its most likely token. Decoding is greedy
    with no score processors in the checkpoint's generation config, so that is the token generate() picks
    (`_run_batch` checks this against the generated ids)."""

    def __init__(self) -> None:
        self.steps: list[tuple[torch.Tensor, torch.Tensor]] = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        lp, tok = scores.float().log_softmax(-1).max(-1)
        self.steps.append((tok, lp))
        return scores

    def rows(self) -> tuple[list[list[int]], np.ndarray]:
        if not self.steps:
            return [], np.zeros((0, 0), np.float32)
        toks = torch.stack([t for t, _ in self.steps], 1).tolist()
        return toks, torch.stack([lp for _, lp in self.steps], 1).cpu().numpy()


def language_from_prefix(prefix_text: str) -> str | None:
    s = prefix_text.strip()
    if s.lower().startswith("language "):
        s = s[len("language ") :].strip()
    return None if not s or s.lower() == "none" else s


def split_packed(states: torch.Tensor, lengths: Sequence[int]) -> list[torch.Tensor]:
    """The encoder packs all clips of a batch into one [sum(frames), d] tensor; split it per clip."""
    lengths = [int(n) for n in lengths]
    if states.shape[0] != sum(lengths):
        raise ValueError(f"packed states have {states.shape[0]} frames, expected {sum(lengths)}")
    return list(torch.split(states, lengths, dim=0))


def make_batches(durations: Sequence[float], batch_size: int) -> list[list[int]]:
    """Indices grouped into batches, longest first, so each batch pads as little as possible."""
    order = sorted(range(len(durations)), key=lambda i: -durations[i])
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


def speech_chunks(wav: np.ndarray, max_chunk_s: float = 20.0, device=None) -> list[SpeechChunk]:
    """Chunks for 16 kHz `wav`: the whole clip if it is no longer than `max_chunk_s`, VAD otherwise."""
    if 0 < len(wav) <= max_chunk_s * SAMPLE_RATE:
        return [SpeechChunk(0, len(wav), (SpeechRegion(0, len(wav)),))]
    return vad_chunks(wav, max_chunk_s=max_chunk_s, device=device)


def encode_clips(processor, tower, audio_token_id: int, clips, device, dtype) -> list[torch.Tensor]:
    """The audio tower's `ln_post` states [frames, 896] (float32, CPU) for prepared clips (`prepare_clip`),
    the same states the decoder sees during generate()."""
    inputs = processor.apply_transcription_request(audio=list(clips))
    zero_feature_padding(inputs)
    frames = (inputs["input_ids"] == audio_token_id).sum(-1).tolist()
    out = tower(
        input_features=inputs["input_features"].to(device, dtype),
        input_features_mask=inputs["input_features_mask"].to(device),
    ).last_hidden_state
    return [s.float().cpu() for s in split_packed(out, frames)]


def majority_language(items: Sequence[ChunkTranscript]) -> str | None:
    """Language covering the most speech time."""
    votes: Counter[str] = Counter()
    for it in items:
        if it.language:
            votes[it.language] += it.chunk.duration
    return votes.most_common(1)[0][0] if votes else None


class RumbleTranscriber:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        dtype: torch.dtype | str | None = None,
        batch_size: int = 8,
        attn_implementation: str = "sdpa",
    ):
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = resolve_dtype(dtype) or default_dtype(self.device)
        self.batch_size = batch_size
        with quiet_loading():
            self.processor = AutoProcessor.from_pretrained(model)
            self.model = Qwen3ASRForConditionalGeneration.from_pretrained(
                model, dtype=self.dtype, attn_implementation=attn_implementation
            )
        self.tokenizer = self.processor.tokenizer
        self.model = self.model.to(self.device).eval()
        self.audio_tower = self.model.model.audio_tower

    def transcribe(
        self,
        audio: AudioInput,
        *,
        sr: int | None = None,
        chunks: list[SpeechChunk] | None = None,
        max_chunk_s: float = 20.0,
        **kwargs,
    ) -> TranscriptResult:
        """Load audio, run VAD (unless `chunks` is given) and transcribe. Segments are one per chunk,
        on the global timeline; words are filled in by the aligner. kwargs go to `transcribe_chunks`.

        Audio no longer than `max_chunk_s` is passed whole, without VAD: on FLEURS Dutch, VAD chunking of
        single sentences cost about 0.3 WER points against whole utterances
        (docs/bench_2026-11-20_rtx4090_diagnose).
        """
        wav = load_audio(audio, sr=sr)
        if chunks is None:
            chunks = speech_chunks(wav, max_chunk_s, self.device)
        items = self.transcribe_chunks(wav, chunks, **kwargs)
        language = majority_language(items)  # a forced language comes back resolved ("nl" -> "Dutch")
        segments = [
            Segment(start=it.chunk.start_s, end=it.chunk.end_s, text=it.text, language=it.language)
            for it in items
            if it.text
        ]
        return TranscriptResult(
            segments=segments,
            language=language or "",
            metadata={
                "model": self.model.name_or_path,
                "duration": len(wav) / SAMPLE_RATE,
                "chunks": len(chunks),
                "chunk_transcripts": items,
            },
        )

    @torch.inference_mode()
    def transcribe_chunks(
        self,
        audio: np.ndarray,
        chunks: Sequence[SpeechChunk],
        *,
        language: str | None = None,
        prompt: str | None = None,
        digits: bool = True,
        filler: bool = False,
        verbatim: bool = True,
        keep_encoder_states: bool = False,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[ChunkTranscript]:
        """Transcribe each chunk of 16 kHz `audio`. Results come back in chunk order.

        `language` forces the language (name or ISO code); otherwise each chunk is detected separately.
        `prompt` is free-text context / hotwords for the system prompt. `digits`, `filler` and `verbatim`
        are the model's control tokens (defaults match the model's own defaults).
        """
        results: list[ChunkTranscript | None] = [None] * len(chunks)
        batches = make_batches([c.duration for c in chunks], self.batch_size)
        done = 0
        options = {"digits": digits, "filler": filler, "verbatim": verbatim}
        for idx in batches:
            batch = [chunks[i] for i in idx]
            out = self._run_batch(audio, batch, language, prompt, options, keep_encoder_states)
            for i, r in zip(idx, out):
                results[i] = r
            done += len(idx)
            if progress:
                progress(done, len(chunks))
        return results  # type: ignore[return-value]

    @torch.inference_mode()
    def encode(
        self,
        audio: np.ndarray,
        chunks: Sequence[SpeechChunk],
        progress: Callable[[int, int], None] | None = None,
    ) -> list[torch.Tensor]:
        """Encoder states for each chunk, without decoding (for aligning text you already have)."""
        out: list[torch.Tensor | None] = [None] * len(chunks)
        done = 0
        for idx in make_batches([c.duration for c in chunks], self.batch_size):
            clips = [prepare_clip(chunks[i].slice(audio)) for i in idx]
            states = encode_clips(
                self.processor, self.audio_tower, self.model.config.audio_token_id, clips, self.device,
                self.dtype,
            )  # fmt: skip
            for i, st in zip(idx, states):
                out[i] = st
            done += len(idx)
            if progress:
                progress(done, len(chunks))
        return out  # type: ignore[return-value]

    def _run_batch(self, audio, batch, language, prompt, options, keep_states):
        clips = [prepare_clip(c.slice(audio)) for c in batch]
        inputs = self.processor.apply_transcription_request(
            audio=clips, language=language, prompt=prompt, **options
        )
        zero_feature_padding(inputs)
        # Each <|audio_pad|> placeholder receives one encoder frame, so its count per row is that clip's
        # share of the packed encoder output.
        frames = (inputs["input_ids"] == self.model.config.audio_token_id).sum(-1).tolist()
        inputs = inputs.to(self.device, dtype=self.dtype)
        forced = language_name(language)

        from transformers import LogitsProcessorList

        captured: list[torch.Tensor] = []
        rec = _TokenLogprobs()
        hook = self.audio_tower.ln_post.register_forward_hook(lambda m, i, o: captured.append(o.detach()))
        try:
            max_new = int(max(c.duration for c in batch) * MAX_TOKENS_PER_SECOND) + MAX_TOKENS_OVERHEAD
            out = self.model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False, logits_processor=LogitsProcessorList([rec])
            )
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError(f"expected one encoder pass per generate(), saw {len(captured)}")
        states = split_packed(captured[0], frames) if keep_states else [None] * len(batch)

        gen = out[:, inputs["input_ids"].shape[1] :].tolist()
        truncated = sum(1 for row in gen if not set(row) & STOP_IDS)
        if truncated:
            warnings.warn(
                f"{truncated} chunk(s) hit the {max_new}-token cap; their text may be cut short",
                stacklevel=3,
            )
        picked, logprobs = rec.rows()
        results = []
        for b, (chunk, row, st) in enumerate(zip(batch, gen, states)):
            text_ids, lang = parse_generated(row, self._decode_special, forced)
            pos = text_positions(row)
            lps = None
            # a row that already stopped keeps getting scored while others decode, but its pad tokens aren't
            # text positions, so only the positions of text tokens are compared
            if picked and len(picked[b]) == len(row) and all(picked[b][p] == row[p] for p in pos):
                lps = [float(logprobs[b, p]) for p in pos]
            text = self.tokenizer.decode(text_ids, clean_up_tokenization_spaces=False).strip()
            results.append(
                ChunkTranscript(
                    chunk=chunk,
                    text=text,
                    language=lang,
                    token_ids=text_ids,
                    encoder_states=None if st is None else st.float().cpu(),
                    token_logprobs=lps,
                )
            )
        return results

    def _decode_special(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)
