"""rumblex: transcription, word timestamps, diarization and subtitles on Reson8/Rumble-2-Flash.

import rumblex
model = rumblex.load_model()                     # Reson8/Rumble-2-Flash, GPU if available
result = rumblex.transcribe("meeting.mp4", model)  # segments + word timestamps
rumblex.write_result(result, "meeting.mp4", "out/", ["srt", "json"])
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rumblex.types import Segment, TranscriptResult, WordSegment
from rumblex.writers import write_result

if TYPE_CHECKING:
    import torch

    from rumblex.audio import AudioInput
    from rumblex.transcriber import RumbleTranscriber

__version__ = "0.1.1"
__all__ = ["Segment", "TranscriptResult", "WordSegment", "align", "load_model", "transcribe", "write_result"]


def load_model(
    model: str = "Reson8/Rumble-2-Flash",
    device: str | None = None,
    compute_type: str | torch.dtype | None = None,
    batch_size: int = 8,
) -> RumbleTranscriber:
    """Load the model (a Hugging Face repo id or a local folder). `device` defaults to cuda if available.
    `compute_type`: float32 / float16 / bfloat16 (or fp32 / fp16 / bf16); default bfloat16 on Ampere and
    newer GPUs, float32 otherwise. `batch_size` is the number of chunks decoded together."""
    from rumblex.transcriber import RumbleTranscriber, resolve_dtype

    dtype = resolve_dtype(compute_type)
    return RumbleTranscriber(model, device=device, dtype=dtype, batch_size=batch_size)


def transcribe(
    audio: AudioInput,
    model: RumbleTranscriber | None = None,
    *,
    align: bool = True,
    sr: int | None = None,
    **kwargs,
) -> TranscriptResult:
    """Transcribe a file, or an array (16 kHz unless `sr` says otherwise). kwargs: language, prompt, digits,
    filler, verbatim, max_chunk_s, chunks (precomputed SpeechChunks instead of VAD) and progress
    (called with (done, total) chunks). With `align`, words get CTC timestamps (word ends refined on the
    audio) and segments are cut at sentence ends.

    Without `model`, a new model is loaded on every call; load it once with `load_model()` for more files."""
    from rumblex.audio import load_audio

    model = model or load_model()
    wav = load_audio(audio, sr=sr)  # decoded once, reused below
    result = model.transcribe(wav, keep_encoder_states=align, **kwargs)
    if align:
        _aligner_for(model).align(result, wav)
    result.metadata.pop("chunk_transcripts", None)
    return result


def align(
    audio: AudioInput,
    text: str,
    model: RumbleTranscriber | None = None,
    *,
    sr: int | None = None,
    **kwargs,
) -> TranscriptResult:
    """Word timestamps for a transcript you already have, without transcribing: the audio is encoded and
    `text` is force-aligned to it with the CTC head. Works on long files; the text doesn't need line breaks
    or timings. kwargs: language (recorded only), max_chunk_s, chunks, progress.

    Words the audio doesn't contain still get times, squeezed in where the text puts them, so the text should
    match the audio. Without `model`, a new model is loaded on every call."""
    from rumblex.audio import load_audio
    from rumblex.textalign import align_text

    model = model or load_model()
    wav = load_audio(audio, sr=sr)
    return align_text(model, _aligner_for(model), wav, text, **kwargs)


def _aligner_for(model: RumbleTranscriber):
    """One Aligner per model, kept on the model, so repeated calls don't reload the CTC head (357 MB)."""
    from rumblex.aligner import Aligner

    if getattr(model, "_rumblex_aligner", None) is None:
        model._rumblex_aligner = Aligner.from_transcriber(model)
    return model._rumblex_aligner
