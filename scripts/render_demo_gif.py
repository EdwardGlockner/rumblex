"""Render docs/demo.gif from a real CLI run: the command is typed out, then the run's captured output and
the start of the subtitle file it wrote are shown. Nothing is invented; waiting is shortened, and a
caption says so and names the hardware.

    rumblex bcn_weather.mp3 two_speakers.wav --max_line_width 42 -f srt json -o subs > cli_output.txt 2>&1
    python scripts/render_demo_gif.py --output cli_output.txt --srt subs/two_speakers.srt \
        --command "rumblex bcn_weather.mp3 two_speakers.wav --max_line_width 42 -f srt json -o subs" \
        --caption "Real run on an RTX 4090 (bf16). Waiting shortened." --out docs/demo.gif
"""

from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BG, FG, DIM = (30, 30, 46), (205, 214, 244), (127, 132, 156)
GREEN, CYAN, YELLOW, PINK = (166, 227, 161), (137, 220, 235), (249, 226, 175), (243, 139, 168)
COLS, ROWS, PAD = 104, 30, 14


def font(size: int, path: str | None) -> ImageFont.FreeTypeFont:
    for p in [path, "C:/Windows/Fonts/consola.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]:
        if p and Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def run_output(path: Path) -> list[str]:
    """The run's own lines, without third-party progress bars and warnings."""
    keep = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.rstrip()
        if not line or "it/s]" in line or line.startswith(("W1", "Loading weights", "Warning")):
            continue
        if keep and not re.match(r"^(model loaded|\S+\.(mp3|wav|mp4|m4a|mkv|flac)\b)", line):
            keep[-1] += " " + line.strip()  # rich wrapped a long line: join it back
        else:
            keep.append(line)
    return keep


def colour(line: str) -> tuple[int, int, int]:
    if line.startswith("$ "):
        return GREEN
    if re.match(r"^\d{2}:\d{2}:\d{2},\d{3} -->", line):
        return CYAN
    if re.match(r"^\d+$", line):
        return DIM
    if line.startswith("model loaded"):
        return DIM
    return FG


class Screen:
    def __init__(self, fnt: ImageFont.FreeTypeFont, caption: str):
        self.f, self.caption = fnt, caption
        box = fnt.getbbox("M")
        self.cw, self.ch = box[2] - box[0], int((box[3] - box[1]) * 1.45)
        self.w = PAD * 2 + self.cw * COLS
        self.h = PAD * 2 + self.ch * (ROWS + 1)
        self.lines: list[str] = []
        self.frames: list[Image.Image] = []
        self.durations: list[int] = []

    def snap(self, ms: int, cursor: bool = False) -> None:
        img = Image.new("RGB", (self.w, self.h), BG)
        d = ImageDraw.Draw(img)
        visible = self.lines[-ROWS:]
        for i, line in enumerate(visible):
            y = PAD + i * self.ch
            tag = re.match(r"^\[[^\]]+\] ", line)
            if line.startswith("$ "):
                d.text((PAD, y), "$ ", font=self.f, fill=PINK)
                d.text((PAD + 2 * self.cw, y), line[2:], font=self.f, fill=FG)
            elif tag:  # speaker label in its own colour
                d.text((PAD, y), tag.group(0), font=self.f, fill=YELLOW)
                d.text((PAD + len(tag.group(0)) * self.cw, y), line[tag.end() :], font=self.f, fill=FG)
            else:
                d.text((PAD, y), line, font=self.f, fill=colour(line))
        if cursor and visible:
            x = PAD + len(visible[-1]) * self.cw
            y = PAD + (len(visible) - 1) * self.ch
            d.rectangle([x, y + 2, x + self.cw - 2, y + self.ch - 4], fill=FG)
        d.text((PAD, self.h - PAD - self.ch + 4), self.caption, font=self.f, fill=DIM)
        self.frames.append(img.convert("P", palette=Image.Palette.ADAPTIVE, colors=16))
        self.durations.append(ms)

    def type(self, text: str) -> None:
        self.lines.append("$ ")
        for i in range(0, len(text) + 1, 3):
            self.lines[-1] = "$ " + text[:i]
            self.snap(45, cursor=True)
        self.lines[-1] = "$ " + text
        self.snap(500, cursor=True)

    def out(self, lines: list[str], ms: int) -> None:
        for line in lines:
            self.lines += textwrap.wrap(line, COLS, subsequent_indent="  ", break_on_hyphens=False) or [""]
            self.snap(ms)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--output", type=Path, required=True, help="captured stdout+stderr of the CLI run")
    ap.add_argument("--srt", type=Path, required=True, help="a subtitle file that run wrote")
    ap.add_argument("--command", required=True, help="the exact command that produced --output")
    ap.add_argument("--caption", required=True)
    ap.add_argument("--font", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    scr = Screen(font(15, args.font), args.caption)
    scr.snap(600, cursor=True)
    scr.type(args.command)
    lines = run_output(args.output)
    scr.out(lines[:1], 900)  # model loaded ...
    scr.out(lines[1:], 1100)  # one line per file
    scr.out([""], 300)
    srt = args.srt.read_text(encoding="utf-8").splitlines()[:18]
    if "" in srt:  # stop at the last whole cue
        srt = srt[: len(srt) - srt[::-1].index("") - 1]
    scr.type(f"head -n {len(srt)} {args.srt.parent.name}/{args.srt.name}")
    scr.out(srt, 120)
    scr.snap(5000)
    scr.frames[0].save(
        args.out, save_all=True, append_images=scr.frames[1:], duration=scr.durations, loop=0, optimize=True
    )
    print(f"{args.out}: {len(scr.frames)} frames, {args.out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
