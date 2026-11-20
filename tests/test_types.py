from rumblex.types import Segment, TranscriptResult, WordSegment


def make() -> TranscriptResult:
    return TranscriptResult(
        segments=[
            Segment(
                0.0,
                1.2,
                "Hello there.",
                [WordSegment("Hello", 0.0, 0.5, 0.9), WordSegment("there.", 0.6, 1.2, 0.8)],
                speaker="SPEAKER_00",
                language="English",
            ),
            Segment(1.5, 2.0, "  ", []),
            Segment(2.0, 3.0, "Bye.", [WordSegment("Bye.", 2.0, 3.0, 0.7)]),
        ],
        language="English",
        metadata={"model": "Reson8/Rumble-2-Flash"},
    )


def test_text_skips_empty_segments():
    assert make().text == "Hello there. Bye."


def test_dict_roundtrip():
    r = make()
    back = TranscriptResult.from_dict(r.to_dict())
    assert back == r
    assert isinstance(back.segments[0].words[0], WordSegment)
