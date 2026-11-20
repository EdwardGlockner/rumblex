"""Speaker assignment: put diarization turns onto words and segments.

Works with any diarizer that yields (start, end, speaker) turns; the pyannote wrapper comes separately.
Each word takes the speaker it overlaps most. Segments are split where the speaker changes, so a
subtitle cue never mixes two people.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rumblex.audio import SAMPLE_RATE
from rumblex.types import Segment, TranscriptResult, WordSegment


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str


def _best_speakers(
    spans: Sequence[tuple[float, float]], turns: Sequence[Turn], fill_nearest: bool, max_gap: float
) -> list[str | None]:
    """For each (start, end) span, the speaker with the most overlap (summed over that speaker's turns).
    Without overlap: the nearest turn within `max_gap` seconds if `fill_nearest`, else None."""
    if not spans or not turns:
        return [None] * len(spans)
    speakers = sorted({t.speaker for t in turns})
    sid = {s: i for i, s in enumerate(speakers)}
    ts = np.array([t.start for t in turns])
    te = np.array([t.end for t in turns])
    tk = np.array([sid[t.speaker] for t in turns])
    out: list[str | None] = []
    for block in range(0, len(spans), 2048):  # bounded memory for long files
        sp = np.asarray(spans[block : block + 2048], dtype=float)
        ov = np.clip(np.minimum(sp[:, 1:2], te) - np.maximum(sp[:, 0:1], ts), 0, None)  # [n, turns]
        per = np.zeros((len(sp), len(speakers)))
        np.add.at(per, (slice(None), tk), ov)  # sum overlap per speaker
        best = per.argmax(1)
        has = per.max(1) > 0
        gap = np.maximum(ts - sp[:, 1:2], sp[:, 0:1] - te).clip(min=0)  # distance to each turn
        near = gap.argmin(1)
        for i in range(len(sp)):
            if has[i]:
                out.append(speakers[best[i]])
            elif fill_nearest and gap[i, near[i]] <= max_gap:
                out.append(turns[near[i]].speaker)
            else:
                out.append(None)
    return out


def assign_word_speakers(
    words: Sequence[WordSegment], turns: Sequence[Turn], fill_nearest: bool = True, max_gap: float = 1.0
) -> None:
    for w, s in zip(words, _best_speakers([(w.start, w.end) for w in words], turns, fill_nearest, max_gap)):
        w.speaker = s


def split_by_speaker(seg: Segment) -> list[Segment]:
    """Split a segment into runs of one speaker. Words without a speaker join the run before them
    (or after, at the start)."""
    if not seg.words:
        return [seg]
    spk = [w.speaker for w in seg.words]
    for i in range(1, len(spk)):  # forward fill
        if spk[i] is None:
            spk[i] = spk[i - 1]
    for i in range(len(spk) - 2, -1, -1):  # back fill the leading gap
        if spk[i] is None:
            spk[i] = spk[i + 1]
    runs: list[list[int]] = []
    for i, s in enumerate(spk):
        if runs and spk[runs[-1][0]] == s:
            runs[-1].append(i)
        else:
            runs.append([i])
    out = []
    for r in runs:
        ws = [seg.words[i] for i in r]
        for w in ws:
            w.speaker = spk[r[0]]
        text = seg.text if len(runs) == 1 else " ".join(w.word for w in ws)
        out.append(Segment(ws[0].start, ws[-1].end, text, ws, spk[r[0]], seg.language))
    return out


def assign_speakers(
    result: TranscriptResult, turns: Sequence[Turn], fill_nearest: bool = True, max_gap: float = 1.0
) -> TranscriptResult:
    """Label words and segments with speakers; segments are split at speaker changes.
    Segments without words get the speaker they overlap most."""
    turns = sorted(turns, key=lambda t: t.start)
    segments: list[Segment] = []
    for seg in result.segments:
        if seg.words:
            assign_word_speakers(seg.words, turns, fill_nearest, max_gap)
            segments.extend(split_by_speaker(seg))
        else:
            seg.speaker = _best_speakers([(seg.start, seg.end)], turns, fill_nearest, max_gap)[0]
            segments.append(seg)
    result.segments = segments
    used = {s.speaker for s in segments}  # a turn with no words in it doesn't make a speaker
    result.metadata["speakers"] = [s for s in sorted({t.speaker for t in turns}) if s in used]
    return result


DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


def _disable_pyannote_telemetry() -> None:
    """pyannote.audio 4 sends usage events (incl. audio duration) to otel.pyannote.ai unless
    PYANNOTE_METRICS_ENABLED says otherwise. Off by default here; set the variable to opt in."""
    import os

    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")


class Diarizer:
    """pyannote speaker diarization -> list of Turn. The models are gated on Hugging Face: accept their
    conditions and log in (`hf auth login`), or pass `token`."""

    def __init__(
        self,
        model: str = DEFAULT_DIARIZATION_MODEL,
        device: str | None = None,
        token: str | None = None,
        pipeline=None,
    ):
        if pipeline is None:
            import torch

            _disable_pyannote_telemetry()
            from pyannote.audio import Pipeline

            pipeline = Pipeline.from_pretrained(model, token=token)
            if pipeline is None:
                raise RuntimeError(
                    f"could not load {model}; accept its conditions on Hugging Face and log in"
                )
            pipeline.to(torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")))
        self.pipeline = pipeline

    def __call__(
        self,
        audio: np.ndarray,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        exclusive: bool = True,
    ) -> list[Turn]:
        """Turns for 16 kHz mono `audio`. `exclusive` uses pyannote's no-overlap output, meant for
        aligning with a transcript."""
        import warnings

        import torch

        with warnings.catch_warnings():
            # pyannote warns about this from its statistics pooling on very short segments; there is
            # nothing a user can do about it.
            warnings.filterwarnings("ignore", message=r"std\(\): degrees of freedom is <= 0")
            out = self.pipeline(
                {
                    "waveform": torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None],
                    "sample_rate": SAMPLE_RATE,
                },
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
        ann = getattr(out, "exclusive_speaker_diarization" if exclusive else "speaker_diarization", out)
        return [
            Turn(float(seg.start), float(seg.end), str(spk))
            for seg, _, spk in ann.itertracks(yield_label=True)
        ]
