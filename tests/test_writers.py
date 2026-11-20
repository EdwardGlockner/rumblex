import io
import json

import pytest
import torch

from rumblex.types import Segment, TranscriptResult, WordSegment
from rumblex.writers import (
    FORMATS,
    WriterOptions,
    format_timestamp,
    write_json,
    write_result,
    write_srt,
    write_tsv,
    write_txt,
    write_vtt,
)


def W(word, s, e, spk=None):
    return WordSegment(word, s, e, 0.9, spk)


def result(speakers=True):
    a, b = ("SPEAKER_00", "SPEAKER_01") if speakers else (None, None)
    return TranscriptResult(
        segments=[
            Segment(
                0.5,
                2.0,
                " Hello there, world.",
                [W("Hello", 0.5, 0.9), W("there,", 1.0, 1.4), W("world.", 1.5, 2.0)],
                a,
            ),
            Segment(
                2.2,
                3.0,
                "Fish & <chips>",
                [W("Fish", 2.2, 2.5), W("&", 2.55, 2.6), W("<chips>", 2.6, 3.0)],
                a,
            ),
            Segment(3.1, 3.2, "   ", [], a),
            Segment(3725.25, 3727.0, "Bye.", [W("Bye.", 3725.25, 3727.0)], b),
        ],
        language="English",
        metadata={"model": "m", "states": torch.zeros(2)},
    )


def render(fn, res, **kw):
    buf = io.StringIO()
    fn(res, buf, WriterOptions(**kw))
    return buf.getvalue()


@pytest.mark.parametrize(
    ("t", "sep", "out"),
    [(0, ",", "00:00:00,000"), (1.0006, ".", "00:00:01.001"), (3725.25, ",", "01:02:05,250"),
     (59.9999, ".", "00:01:00.000"), (-0.2, ".", "00:00:00.000")],
)  # fmt: skip
def test_format_timestamp(t, sep, out):
    assert format_timestamp(t, sep) == out


def test_srt():
    out = render(write_srt, result())
    assert out.startswith("1\n00:00:00,500 --> 00:00:02,000\n[SPEAKER_00] Hello there, world.\n\n")
    assert "3\n01:02:05,250 --> 01:02:07,000\n[SPEAKER_01] Bye.\n" in out
    assert out.count("-->") == 3  # whitespace-only segment skipped
    assert "[SPEAKER" not in render(write_srt, result(), speakers=False)


def test_srt_line_splitting():
    out = render(write_srt, result(speakers=False), max_line_width=12, max_line_count=1)
    cues = [c for c in out.strip().split("\n\n")]
    assert cues[0].splitlines()[1:] == ["00:00:00,500 --> 00:00:01,400", "Hello there,"]
    assert cues[1].splitlines()[1:] == ["00:00:01,500 --> 00:00:02,000", "world."]
    two_lines = render(write_srt, result(speakers=False), max_line_width=12, max_line_count=2)
    assert two_lines.split("\n\n")[0].splitlines()[2:] == ["Hello there,", "world."]


def test_vtt_escaping_and_voice():
    out = render(write_vtt, result())
    assert out.startswith("WEBVTT\n\n00:00:00.500 --> 00:00:02.000\n<v SPEAKER_00>Hello there, world.\n")
    assert "Fish &amp; &lt;chips&gt;" in out


def test_vtt_karaoke():
    out = render(write_vtt, result(speakers=False), highlight_words=True)
    assert "Hello <00:00:01.000>there, <00:00:01.500>world." in out
    assert "Fish <00:00:02.550>&amp; <00:00:02.600>&lt;chips&gt;" in out


def test_vtt_karaoke_falls_back_when_words_dont_match_text():
    res = TranscriptResult([Segment(0, 1, "Hello world", [W("Hello", 0, 0.5)])], "English")
    out = render(write_vtt, res, highlight_words=True)
    assert "Hello world" in out and "<00:" not in out


def test_cues_never_overlap():
    res = TranscriptResult([Segment(0, 2.0, "a", []), Segment(1.5, 3.0, "b", [])], "English")
    assert "00:00:02,000 --> 00:00:03,000" in render(write_srt, res)


def test_tsv():
    lines = render(write_tsv, result()).splitlines()
    assert lines[0] == "start\tend\tspeaker\ttext"
    assert lines[1] == "500\t2000\tSPEAKER_00\tHello there, world."
    assert len(lines) == 4
    assert render(write_tsv, result(), speakers=False).splitlines()[1] == "500\t2000\t\tHello there, world."


def test_txt_paragraphs_by_speaker():
    out = render(write_txt, result())
    assert out == "SPEAKER_00: Hello there, world. Fish & <chips>\n\nSPEAKER_01: Bye.\n"
    assert render(write_txt, result(speakers=False)) == "Hello there, world.\nFish & <chips>\nBye.\n"


def test_json_roundtrip_drops_unserialisable_metadata():
    d = json.loads(render(write_json, result()))
    assert d["metadata"] == {"model": "m"}
    back = TranscriptResult.from_dict(d)
    assert back.segments[0].words[1].word == "there," and back.language == "English"


def test_write_result(tmp_path):
    paths = write_result(result(), "some/dir/meeting.mp4", tmp_path / "out", ["all"])
    assert sorted(p.name for p in paths) == sorted(f"meeting.{f}" for f in FORMATS)
    assert b"\r\n" not in (tmp_path / "out" / "meeting.srt").read_bytes()
    with pytest.raises(ValueError):
        write_result(result(), "a.wav", tmp_path, ["docx"])


def test_cues_split_evenly_without_orphans():
    """Greedy filling used to leave a two-word cue ("Wall Street.") after a full two-line cue."""
    text = "Debate is heating up over whether or not the Fearless Girl statue should stay on Wall Street."
    words = [W(w, i * 0.3, i * 0.3 + 0.25) for i, w in enumerate(text.split())]
    res = TranscriptResult([Segment(0, words[-1].end, text, words)], "English")
    cues = [c.split("\n")[2:] for c in render(write_srt, res, max_line_width=42).strip().split("\n\n")]
    assert len(cues) == 2
    lengths = [sum(len(line) for line in c) for c in cues]
    assert max(lengths) - min(lengths) <= 12  # balanced, not 80 characters vs 12
    assert all(len(line) <= 42 for c in cues for line in c)
    assert " ".join(" ".join(c) for c in cues) == text


def test_long_word_falls_back_to_greedy():
    words = [W("x" * 50, 0, 1), W("short", 1, 2)]
    res = TranscriptResult([Segment(0, 2, "x" * 50 + " short", words)], "English")
    out = render(write_srt, res, max_line_width=10, max_line_count=2)
    assert "x" * 50 in out and "short" in out


def test_cues_respect_max_line_count():
    """Words of very different lengths: an even split into the greedy cue count still needed 3 lines."""
    text = "a bbbbbbbbb c dddddddd e ffffffffff g hhhhhhh i jjjjjjjjjj k llllll m nnnnnnnnn o"
    words = [W(w, i * 0.3, i * 0.3 + 0.25) for i, w in enumerate(text.split())]
    res = TranscriptResult([Segment(0, words[-1].end, text, words)], "English")
    for width in range(4, 30):
        for count in (1, 2, 3):
            cues = render(write_srt, res, max_line_width=width, max_line_count=count).strip().split("\n\n")
            assert all(len(c.splitlines()[2:]) <= count for c in cues), (width, count)
            assert " ".join(" ".join(c.splitlines()[2:]) for c in cues) == text


def test_srt_speaker_label_counts_towards_width():
    words = [
        W(w, i * 0.3, i * 0.3 + 0.25, "SPEAKER_00") for i, w in enumerate(["one", "two", "three", "four"])
    ]
    res = TranscriptResult([Segment(0, 1.2, "one two three four", words, "SPEAKER_00")], "English")
    out = render(write_srt, res, max_line_width=20, max_line_count=1)
    lines = [line for c in out.strip().split("\n\n") for line in c.splitlines()[2:]]
    assert lines[0].startswith("[SPEAKER_00] ") and all(len(line) <= 20 for line in lines)


def test_srt_normalises_whitespace():
    res = TranscriptResult([Segment(0, 1, "  Hello \n  world\t again ", [])], "English")
    assert "Hello world again\n" in render(write_srt, res)
    assert "Hello world again\n" in render(write_vtt, res)
