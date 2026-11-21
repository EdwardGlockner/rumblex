"""Render docs/words.gif: an excerpt of a real rumblex JSON result played back in time. Each word lights up
from its start to its end time, coloured by speaker, above a scrolling timeline of the audio with the
word spans drawn on it. All times and speaker labels come from the JSON; nothing is edited by hand.

    rumblex two_speakers.wav --diarize -f srt json -o subs
    python scripts/render_words_gif.py --json subs/two_speakers.json --audio two_speakers.wav \
        --start 28 --end 42 --out docs/words.gif \
        --caption "Word times and speakers exactly as one rumblex run wrote them, in real time."
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rumblex.audio import load_audio

BG, PANEL, FG, DIM, FAINT = (30, 30, 46), (24, 24, 37), (205, 214, 244), (127, 132, 156), (69, 71, 90)
PINK = (243, 139, 168)
SPEAKER_COLOURS = [(249, 226, 175), (137, 220, 235), (166, 227, 161), (203, 166, 247)]
W, PAD = 860, 16
TEXT_LINES, WINDOW_S, FPS = 4, 4.0, 10


def font(size: int, path: str | None) -> ImageFont.FreeTypeFont:
    for p in [path, "C:/Windows/Fonts/consola.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]:
        if p and Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


@dataclass
class Word:
    text: str
    start: float
    end: float
    speaker: str | None


def load_words(path: Path, t0: float, t1: float) -> list[Word]:
    """Words of the segments that overlap [t0, t1], so lines start at a segment boundary."""
    data = json.loads(path.read_text(encoding="utf-8"))
    words = []
    for seg in data["segments"]:
        if seg["end"] < t0 or seg["start"] > t1:
            continue
        for w in seg["words"]:
            if w["start"] < t1:
                words.append(
                    Word(w["word"].strip(), w["start"], w["end"], w.get("speaker") or seg.get("speaker"))
                )
    return words


def layout(words: list[Word], cols: int) -> list[list[tuple[int, str]]]:
    """Wrap words into lines of (word index or -1 for a label, text). A new speaker starts a new line."""
    lines: list[list[tuple[int, str]]] = []
    width = cols
    prev = object()
    for i, w in enumerate(words):
        if w.speaker != prev:
            lines.append([(-1, f"[{w.speaker}]")] if w.speaker else [])
            width = len(lines[-1][0][1]) if w.speaker else 0
            prev = w.speaker
        if width + 1 + len(w.text) > cols and lines[-1]:
            lines.append([])
            width = 0
        lines[-1].append((i, w.text))
        width += len(w.text) + (1 if width else 0)
    return lines


class Renderer:
    def __init__(self, words, audio, sr, fnt, small, caption, cols):
        self.words, self.audio, self.sr, self.f, self.small, self.caption = (
            words,
            audio,
            sr,
            fnt,
            small,
            caption,
        )
        box = fnt.getbbox("M")
        self.cw, self.ch = box[2] - box[0], int((box[3] - box[1]) * 1.6)
        self.lines = layout(words, cols)
        self.line_of = {i: n for n, line in enumerate(self.lines) for i, _ in line if i >= 0}
        speakers = sorted({w.speaker for w in words if w.speaker})
        self.colour = {s: SPEAKER_COLOURS[k % len(SPEAKER_COLOURS)] for k, s in enumerate(speakers)}
        self.text_top = PAD + self.ch + 6
        self.tl_top = self.text_top + TEXT_LINES * self.ch + 14
        self.tl_h = 86
        self.h = self.tl_top + self.tl_h + 18 + self.ch + PAD
        self.env = self._envelope()

    def _envelope(self) -> np.ndarray:
        """Peak level per 10 ms, for the waveform."""
        hop = self.sr // 100
        n = len(self.audio) // hop
        env = np.abs(self.audio[: n * hop]).reshape(n, hop).max(axis=1)
        return env / (np.percentile(env, 99) + 1e-9)

    def col(self, w: Word) -> tuple[int, int, int]:
        return self.colour.get(w.speaker, FG)

    def frame(self, t: float) -> Image.Image:
        img = Image.new("RGB", (W, self.h), BG)
        d = ImageDraw.Draw(img)
        current = next((i for i, w in enumerate(self.words) if w.start <= t < w.end), None)
        spoken = [i for i, w in enumerate(self.words) if w.start <= t]
        anchor = current if current is not None else (spoken[-1] if spoken else 0)

        clock = f"{int(t // 60):02d}:{t % 60:04.1f}"
        d.text((PAD, PAD), "rumblex: word timestamps + speakers", font=self.f, fill=DIM)
        d.text((W - PAD - len(clock) * self.cw, PAD), clock, font=self.f, fill=FG)

        first = max(0, min(self.line_of[anchor] - 1, len(self.lines) - TEXT_LINES))
        for row, line in enumerate(self.lines[first : first + TEXT_LINES]):
            x, y = PAD, self.text_top + row * self.ch
            for i, text in line:
                if i < 0:
                    spk = text[1:-1]
                    d.text((x, y), text, font=self.f, fill=self.colour.get(spk, FG))
                else:
                    w = self.words[i]
                    if i == current:
                        d.rounded_rectangle(
                            [x - 3, y - 2, x + len(text) * self.cw + 3, y + self.ch - 5], 4, fill=self.col(w)
                        )
                        d.text((x, y), text, font=self.f, fill=BG)
                    else:
                        d.text((x, y), text, font=self.f, fill=FG if w.start <= t else FAINT)
                x += (len(text) + 1) * self.cw

        # timeline: WINDOW_S seconds with the playhead in the middle
        top, h = self.tl_top, self.tl_h
        d.rounded_rectangle([PAD, top, W - PAD, top + h], 6, fill=PANEL)
        px_per_s = (W - 2 * PAD) / WINDOW_S
        left = t - WINDOW_S / 2

        def xof(s: float) -> float:
            return PAD + (s - left) * px_per_s

        mid = top + h * 0.62
        for k in range(int(max(0, left) * 100), int(min(len(self.env) / 100, left + WINDOW_S) * 100)):
            amp = float(self.env[k]) * h * 0.3
            x = xof(k / 100)
            d.line([x, mid - amp, x, mid + amp], fill=FAINT)
        scw = self.small_cw()
        for i, w in enumerate(self.words):
            x0, x1 = max(xof(w.start), PAD + 1), min(xof(w.end), W - PAD - 1)
            if x1 <= x0:
                continue
            c = self.col(w) if w.start <= t else FAINT
            d.rectangle([x0, mid - 5, max(x0 + 1, x1 - 1), mid + 5], fill=c)
            lx = xof(w.start)
            if lx > PAD and lx + len(w.text) * scw < W - PAD:  # labels alternate between two rows
                d.text((lx, top + 6 + (i % 2) * 15), w.text, font=self.small, fill=c if w.start <= t else DIM)
        for s in range(int(np.ceil(left)), int(left + WINDOW_S) + 1):
            x = xof(s)
            if PAD < x < W - PAD:
                d.line([x, top + h - 10, x, top + h - 4], fill=DIM)
                d.text((x + 3, top + h - 18), f"{s}s", font=self.small, fill=DIM)
        d.line([W / 2, top + 4, W / 2, top + h - 4], fill=PINK, width=2)

        d.text((PAD, self.h - PAD - self.ch + 6), self.caption, font=self.f, fill=DIM)
        return img.convert("P", palette=Image.Palette.ADAPTIVE, colors=32)

    def small_cw(self) -> int:
        box = self.small.getbbox("M")
        return box[2] - box[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", type=Path, required=True, help="a rumblex JSON result with word times")
    ap.add_argument("--audio", type=Path, required=True, help="the audio that result came from")
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--caption", required=True)
    ap.add_argument("--font", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sr = 16000
    audio = load_audio(args.audio)
    words = load_words(args.json, args.start, args.end)
    fnt, small = font(16, args.font), font(13, args.font)
    cols = (W - 2 * PAD) // (fnt.getbbox("M")[2] - fnt.getbbox("M")[0]) - 1
    r = Renderer(words, audio, sr, fnt, small, args.caption, cols)
    times = np.arange(args.start, args.end, 1 / FPS)
    frames = [r.frame(float(t)) for t in times]
    durations = [1000 // FPS] * len(frames)
    durations[-1] = 2500
    frames[0].save(
        args.out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True
    )
    print(f"{args.out}: {len(frames)} frames, {args.out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
