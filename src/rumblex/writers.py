"""Exporters: json, srt, vtt, tsv, txt.

Subtitle cues follow segments by default. When segments have word timings and `max_line_width` is
set, long segments are split into cues at word boundaries (at most `max_line_count` lines per cue).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

from rumblex.types import Segment, TranscriptResult, WordSegment

FORMATS = ("json", "srt", "vtt", "tsv", "txt")


@dataclass
class WriterOptions:
    max_line_width: int | None = None  # characters per subtitle line; None = one cue per segment
    max_line_count: int = 2  # lines per cue when splitting
    highlight_words: bool = False  # vtt: per-word <hh:mm:ss.mmm> karaoke tags
    speakers: bool = True  # label speakers when known


@dataclass
class Cue:
    start: float
    end: float
    lines: list[str]
    speaker: str | None = None
    words: list[WordSegment] | None = None  # set when the cue was built from word timings


def format_timestamp(seconds: float, sep: str = ".", always_hours: bool = True) -> str:
    ms = max(0, round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    hh = f"{h:02d}:" if always_hours or h else ""
    return f"{hh}{m:02d}:{s:02d}{sep}{ms:03d}"


def _timed(words: Sequence[WordSegment]) -> bool:
    return bool(words) and all(w.end >= w.start >= 0 for w in words)


def _wrap(words: Sequence[WordSegment], width: int) -> list[list[WordSegment]]:
    """Greedy line fill. A word longer than `width` gets a line of its own."""
    lines: list[list[WordSegment]] = []
    cur: list[WordSegment] = []
    n = 0
    for w in words:
        add = len(w.word) + (1 if cur else 0)
        if cur and n + add > width:
            lines.append(cur)
            cur, n = [], 0
            add = len(w.word)
        cur.append(w)
        n += add
    if cur:
        lines.append(cur)
    return lines


def _text_len(words: Sequence[WordSegment]) -> int:
    return sum(len(w.word) for w in words) + max(0, len(words) - 1)


def _even_split(words: Sequence[WordSegment], parts: int) -> list[list[WordSegment]]:
    """Cut `words` into `parts` contiguous groups whose longest text is as short as possible."""
    n = len(words)
    parts = max(1, min(parts, n))
    inf = float("inf")
    # best[k][i]: smallest possible longest-group length for the first i words in k groups
    best = [[inf] * (n + 1) for _ in range(parts + 1)]
    cut = [[0] * (n + 1) for _ in range(parts + 1)]
    best[0][0] = 0
    for k in range(1, parts + 1):
        for i in range(k, n + 1):
            for j in range(k - 1, i):
                cost = max(best[k - 1][j], _text_len(words[j:i]))
                if cost < best[k][i]:
                    best[k][i], cut[k][i] = cost, j
    groups, i = [], n
    for k in range(parts, 0, -1):
        j = cut[k][i]
        groups.append(list(words[j:i]))
        i = j
    return groups[::-1]


def _cue_lines(words: Sequence[WordSegment], width: int, max_lines: int) -> list[list[WordSegment]]:
    """Lines for one cue: one line if it fits, else up to `max_lines` lines of even length; greedy
    wrapping if no even split fits the width."""
    if _text_len(words) <= width:
        return [list(words)]
    lines = _even_split(words, min(max_lines, len(words)))
    if all(_text_len(line) <= width for line in lines):
        return lines
    return _wrap(words, width)


def segment_cues(seg: Segment, opts: WriterOptions, reserve: int = 0) -> list[Cue]:
    """Cues for one segment. Without `max_line_width`: one cue. With it: as many cues as greedy line
    filling would need, but with the words spread evenly over them, so there are no one-word leftovers.
    `reserve` characters of each line are kept free (for a speaker label)."""
    text = " ".join(seg.text.split())
    if not text:
        return []
    if opts.max_line_width is None or not _timed(seg.words):
        return [Cue(seg.start, seg.end, [text], seg.speaker, list(seg.words) if _timed(seg.words) else None)]
    width, max_lines = max(1, opts.max_line_width - reserve), max(1, opts.max_line_count)
    n_cues = -(-len(_wrap(seg.words, width)) // max_lines)
    while True:  # an even split can still need more lines than allowed; then use more cues
        groups = _even_split(seg.words, n_cues)
        layout = [_cue_lines(g, width, max_lines) for g in groups]
        if all(len(lines) <= max_lines for lines in layout) or n_cues >= len(seg.words):
            break
        n_cues += 1
    cues = []
    for words, lines in zip(groups, layout):
        cues.append(
            Cue(
                start=words[0].start,
                end=words[-1].end,
                lines=[" ".join(w.word for w in line) for line in lines],
                speaker=seg.speaker,
                words=words,
            )
        )
    # keep the segment's outer bounds so the first/last cue don't shrink inside the segment
    cues[0].start = min(cues[0].start, seg.start)
    cues[-1].end = max(cues[-1].end, seg.end)
    return cues


def iter_cues(result: TranscriptResult, opts: WriterOptions, labels: bool = False) -> Iterable[Cue]:
    """`labels`: the speaker label is part of the line text (srt), so it counts towards the width."""
    prev_end = 0.0
    for seg in result.segments:
        reserve = len(f"[{seg.speaker}] ") if labels and opts.speakers and seg.speaker else 0
        for cue in segment_cues(seg, opts, reserve):
            cue.start = max(cue.start, prev_end)  # players dislike overlapping cues
            cue.end = max(cue.end, cue.start + 0.001)
            prev_end = cue.end
            yield cue


def _label(cue: Cue, opts: WriterOptions) -> str:
    return f"[{cue.speaker}] " if opts.speakers and cue.speaker else ""


def write_srt(result: TranscriptResult, f: TextIO, opts: WriterOptions) -> None:
    for i, cue in enumerate(iter_cues(result, opts, labels=True), 1):
        lines = list(cue.lines)
        lines[0] = _label(cue, opts) + lines[0]
        start, end = format_timestamp(cue.start, ","), format_timestamp(cue.end, ",")
        f.write(f"{i}\n{start} --> {end}\n" + "\n".join(lines) + "\n\n")


def _vtt_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _vtt_karaoke(cue: Cue) -> list[str]:
    """Lines with a timestamp tag before every word after the cue's first one."""
    assert cue.words is not None
    out, k = [], 0
    for line in cue.lines:
        parts = []
        for _ in line.split(" "):
            w = cue.words[k]
            tag = f"<{format_timestamp(w.start)}>" if k > 0 and w.start > cue.start else ""
            parts.append(tag + _vtt_escape(w.word))
            k += 1
        out.append(" ".join(parts))
    return out


def write_vtt(result: TranscriptResult, f: TextIO, opts: WriterOptions) -> None:
    f.write("WEBVTT\n\n")
    for cue in iter_cues(result, opts):
        karaoke = opts.highlight_words and cue.words is not None and _line_words_match(cue)
        lines = _vtt_karaoke(cue) if karaoke else [_vtt_escape(x) for x in cue.lines]
        if opts.speakers and cue.speaker:
            lines[0] = f"<v {_vtt_escape(cue.speaker)}>" + lines[0]
        f.write(
            f"{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}\n" + "\n".join(lines) + "\n\n"
        )


def _line_words_match(cue: Cue) -> bool:
    """Karaoke needs a 1:1 match between the displayed tokens and the word list."""
    return cue.words is not None and sum(len(x.split(" ")) for x in cue.lines) == len(cue.words)


def write_tsv(result: TranscriptResult, f: TextIO, opts: WriterOptions) -> None:
    """Start/end in integer milliseconds, like Whisper's tsv."""
    f.write("start\tend\tspeaker\ttext\n")
    for seg in result.segments:
        text = " ".join(seg.text.split())
        if text:
            spk = (seg.speaker or "") if opts.speakers else ""
            f.write(f"{round(seg.start * 1000)}\t{round(seg.end * 1000)}\t{spk}\t{text}\n")


def write_txt(result: TranscriptResult, f: TextIO, opts: WriterOptions) -> None:
    """Plain text. With speakers, consecutive segments of one speaker form a paragraph."""
    if not (opts.speakers and any(s.speaker for s in result.segments)):
        for seg in result.segments:
            if seg.text.strip():
                f.write(seg.text.strip() + "\n")
        return
    paras: list[tuple[str | None, list[str]]] = []
    for seg in result.segments:
        text = seg.text.strip()
        if not text:
            continue
        if paras and paras[-1][0] == seg.speaker:
            paras[-1][1].append(text)
        else:
            paras.append((seg.speaker, [text]))
    f.write("\n\n".join(f"{spk or 'UNKNOWN'}: {' '.join(parts)}" for spk, parts in paras) + "\n")


def _json_safe(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep only metadata values that serialise as JSON (pipelines may stash tensors there)."""
    out = {}
    for k, v in metadata.items():
        try:
            json.dumps(v)
        except (TypeError, ValueError):
            continue
        out[k] = v
    return out


def write_json(result: TranscriptResult, f: TextIO, opts: WriterOptions) -> None:
    d = {
        "language": result.language,
        "segments": [asdict(seg) for seg in result.segments],
        "metadata": _json_safe(result.metadata),
    }
    json.dump(d, f, ensure_ascii=False, indent=2)
    f.write("\n")


WRITERS: dict[str, Callable[[TranscriptResult, TextIO, WriterOptions], None]] = {
    "json": write_json,
    "srt": write_srt,
    "vtt": write_vtt,
    "tsv": write_tsv,
    "txt": write_txt,
}


def write_result(
    result: TranscriptResult,
    audio_path: str | Path,
    output_dir: str | Path,
    formats: Sequence[str] = ("srt",),
    opts: WriterOptions | None = None,
) -> list[Path]:
    """Write `result` once per format as `<output_dir>/<audio stem>.<fmt>`. Returns the paths."""
    opts = opts or WriterOptions()
    formats = list(FORMATS) if "all" in formats else list(formats)
    unknown = [x for x in formats if x not in WRITERS]
    if unknown:
        raise ValueError(f"unknown output format(s) {unknown}; choose from {FORMATS} or 'all'")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        p = out_dir / f"{Path(audio_path).stem}.{fmt}"
        with p.open("w", encoding="utf-8", newline="\n") as f:
            WRITERS[fmt](result, f, opts)
        paths.append(p)
    return paths
