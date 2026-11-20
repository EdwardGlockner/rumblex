"""Result types shared across the pipeline. All times are seconds on the global (whole-file) timeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class WordSegment:
    word: str
    start: float
    end: float
    score: float
    speaker: str | None = None


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[WordSegment] = field(default_factory=list)
    speaker: str | None = None
    language: str | None = None


@dataclass
class TranscriptResult:
    segments: list[Segment]
    language: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TranscriptResult:
        segs = [
            Segment(**{**s, "words": [WordSegment(**w) for w in s.get("words", [])]}) for s in d["segments"]
        ]
        return cls(segments=segs, language=d["language"], metadata=d.get("metadata", {}))
