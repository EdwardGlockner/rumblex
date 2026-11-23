"""vLLM backend: decoding on a vLLM server, encoder states for alignment computed locally.

Start a server as on the model card (Linux + a recent NVIDIA GPU; leave room for the local encoder):

    vllm serve Reson8/Rumble-2-Flash --max-model-len 4096 --gpu-memory-utilization 0.5

vLLM does not expose the audio encoder's `ln_post` states, so when alignment needs them they come from a
local run of just the audio tower, through the same processor path as the transformers backend
(`prepare_clip`, `zero_feature_padding`), so the states match that backend's.

Needs the `[vllm]` extra on the client side (`openai`); vLLM itself only where the server runs.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import torch

from rumblex.audio import SAMPLE_RATE
from rumblex.transcriber import (
    DEFAULT_MODEL,
    FIRST_SPECIAL_ID,
    MAX_TOKENS_OVERHEAD,
    MAX_TOKENS_PER_SECOND,
    ChunkTranscript,
    RumbleTranscriber,
    default_dtype,
    encode_clips,
    language_from_prefix,
    language_name,
    parse_generated,
    prepare_clip,
    quiet_loading,
    resolve_dtype,
    text_positions,
)
from rumblex.vad import SpeechChunk


def wav_data_uri(clip: np.ndarray) -> str:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, clip, SAMPLE_RATE, format="WAV", subtype="FLOAT")  # float: no requantisation
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def parse_completion(
    choice,
    encode: Callable[[str], list[int]],
    decode: Callable[[list[int]], str],
    forced: str | None,
) -> tuple[list[int], str | None]:
    """Text token ids and language from one chat completion choice.

    With `return_token_ids` the server's own ids are used (exactly what the model emitted). Otherwise the
    text, requested with special tokens kept so `<asr_text>` survives, is split there and re-tokenised;
    CTC alignment works with any tokenisation of the text.
    """
    ids = getattr(choice, "token_ids", None)
    if ids is None and isinstance(getattr(choice, "model_extra", None), dict):
        ids = choice.model_extra.get("token_ids")
    if ids:
        return parse_generated(list(ids), decode, forced)
    text = choice.message.content or ""
    lang = forced
    if "<asr_text>" in text:
        prefix, text = text.split("<asr_text>", 1)
        if forced is None:
            lang = language_from_prefix(_strip_special(prefix))
    text = _strip_special(text).strip()
    return ([i for i in encode(" " + text) if i < FIRST_SPECIAL_ID] if text else []), lang


def completion_logprobs(choice) -> list[float] | None:
    """Decoder log-prob of each text token `parse_completion` returns, from the server's `logprobs` and
    `token_ids`. None when either is missing or they don't line up (the text-only fallback)."""
    ids = getattr(choice, "token_ids", None)
    if ids is None and isinstance(getattr(choice, "model_extra", None), dict):
        ids = choice.model_extra.get("token_ids")
    content = getattr(getattr(choice, "logprobs", None), "content", None)
    if not ids or not content or len(content) != len(ids):
        return None
    return [float(content[p].logprob) for p in text_positions(list(ids))]


def _strip_special(s: str) -> str:
    for tok in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        s = s.replace(tok, "")
    return s


class EncoderRunner:
    """Just the audio tower, for `ln_post` states. Same input preparation as `RumbleTranscriber`."""

    def __init__(
        self, model: str = DEFAULT_MODEL, device: str | None = None, dtype: torch.dtype | str | None = None
    ):
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = resolve_dtype(dtype) or default_dtype(self.device)
        with quiet_loading():
            self.processor = AutoProcessor.from_pretrained(model)
            full = Qwen3ASRForConditionalGeneration.from_pretrained(model, dtype=self.dtype)
        self.audio_token_id = full.config.audio_token_id
        self.tower = full.model.audio_tower.to(self.device).eval()
        del full  # drop the language model; only the tower is kept

    @torch.inference_mode()
    def states(self, clips: Sequence[np.ndarray]) -> list[torch.Tensor]:
        """[frames, 896] float32 CPU states per prepared clip (see `prepare_clip`)."""
        return encode_clips(self.processor, self.tower, self.audio_token_id, clips, self.device, self.dtype)


class VLLMTranscriber:
    """Same interface as `RumbleTranscriber` (transcribe / transcribe_chunks), backed by a vLLM server."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = DEFAULT_MODEL,
        *,
        concurrency: int = 32,
        encoder: EncoderRunner | None = None,
        client=None,
        tokenizer=None,
        batch_size: int = 8,
        device: str | None = None,
        dtype: torch.dtype | str | None = None,
    ):
        # Device and dtype are for the local encoder (alignment) and VAD; resolved now, so reading them
        # does not load the encoder.
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = resolve_dtype(dtype) or default_dtype(self.device)
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key="EMPTY")
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model)
        self.client, self.model_id, self.tokenizer = client, model, tokenizer
        self.model = SimpleNamespace(name_or_path=model)  # what Aligner.from_transcriber and metadata read
        self.concurrency, self.batch_size = concurrency, batch_size
        self._encoder = encoder

    # same signatures as RumbleTranscriber, so Aligner and the CLI work unchanged
    transcribe = RumbleTranscriber.transcribe

    @property
    def encoder(self) -> EncoderRunner:
        """Loaded on first use, i.e. only when encoder states are asked for."""
        if self._encoder is None:
            self._encoder = EncoderRunner(self.model_id, self.device, self.dtype)
        return self._encoder

    def encode(self, audio: np.ndarray, chunks: Sequence[SpeechChunk], progress=None) -> list[torch.Tensor]:
        """Encoder states for each chunk from the local encoder, without decoding."""
        out = []
        for b in range(0, len(chunks), self.batch_size):
            out += self.encoder.states(
                [prepare_clip(c.slice(audio)) for c in chunks[b : b + self.batch_size]]
            )
            if progress:
                progress(len(out), len(chunks))
        return out

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
        forced = language_name(language)
        clips = [prepare_clip(c.slice(audio)) for c in chunks]
        kwargs = {"digits": digits, "filler": filler, "verbatim": verbatim}
        if forced:
            kwargs["language"] = forced

        def one(i: int):
            msgs = [] if prompt is None else [{"role": "system", "content": prompt}]
            msgs.append(
                {
                    "role": "user",
                    "content": [{"type": "audio_url", "audio_url": {"url": wav_data_uri(clips[i])}}],
                }
            )
            r = self.client.chat.completions.create(
                model=self.model_id,
                messages=msgs,
                temperature=0,
                logprobs=True,
                max_tokens=int(chunks[i].duration * MAX_TOKENS_PER_SECOND) + MAX_TOKENS_OVERHEAD,
                extra_body={
                    "chat_template_kwargs": kwargs,
                    "skip_special_tokens": False,
                    "return_token_ids": True,
                },
            )
            c = r.choices[0]
            ids, lang = parse_completion(
                c,
                lambda t: self.tokenizer.encode(t, add_special_tokens=False),
                lambda ids: self.tokenizer.decode(ids, skip_special_tokens=True),
                forced,
            )
            return ids, lang, completion_logprobs(c)

        def encode_all() -> dict[int, torch.Tensor]:
            states: dict[int, torch.Tensor] = {}
            order = sorted(range(len(chunks)), key=lambda i: -chunks[i].duration)
            for b in range(0, len(order), self.batch_size):
                idx = order[b : b + self.batch_size]
                states.update(zip(idx, self.encoder.states([clips[i] for i in idx])))
            return states

        results: list[ChunkTranscript | None] = [None] * len(chunks)
        # The local encoder pass runs while the server decodes: one thread for it, the rest wait on HTTP.
        with ThreadPoolExecutor(1) as enc_pool, ThreadPoolExecutor(self.concurrency) as pool:
            enc_future = enc_pool.submit(encode_all) if keep_encoder_states else None
            for i, (ids, lang, lps) in enumerate(pool.map(one, range(len(chunks)))):
                text = self.tokenizer.decode(ids, clean_up_tokenization_spaces=False).strip()
                results[i] = ChunkTranscript(chunks[i], text, lang, ids, token_logprobs=lps)
                if progress:
                    progress(i + 1, len(chunks))
            if enc_future is not None:
                for i, st in enc_future.result().items():
                    results[i].encoder_states = st
        return results  # type: ignore[return-value]
