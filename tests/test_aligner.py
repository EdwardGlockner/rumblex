from itertools import pairwise
from typing import ClassVar

import numpy as np
import pytest
import torch

from rumblex.aligner import (
    END_RULES,
    Aligner,
    AlignOptions,
    ctc_viterbi,
    frame_times,
    group_words,
    proportional_words,
    split_sentences,
    token_spans,
)
from rumblex.types import Segment, WordSegment


def reference_viterbi(em, targets):
    """The plain double-loop CTC Viterbi from scripts/inspect_rumble_ctc.py, on the same column layout:
    em[:, 0] is blank, em[:, k + 1] is target k. Returns token spans or None."""
    T, L = em.shape[0], len(targets)
    S = 2 * L + 1
    col = [0 if s % 2 == 0 else s // 2 + 1 for s in range(S)]
    lab = [None if s % 2 == 0 else targets[s // 2] for s in range(S)]
    dp = np.full((T, S), -np.inf)
    bp = np.zeros((T, S), dtype=int)
    dp[0, 0] = em[0, col[0]]
    if S > 1:
        dp[0, 1] = em[0, col[1]]
    for t in range(1, T):
        for s in range(S):
            best, arg = dp[t - 1, s], 0
            if s >= 1 and dp[t - 1, s - 1] > best:
                best, arg = dp[t - 1, s - 1], 1
            if s >= 2 and lab[s] is not None and lab[s] != lab[s - 2] and dp[t - 1, s - 2] > best:
                best, arg = dp[t - 1, s - 2], 2
            dp[t, s] = best + em[t, col[s]]
            bp[t, s] = arg
    s = S - 1 if S == 1 or dp[-1, S - 1] >= dp[-1, S - 2] else S - 2
    if not np.isfinite(dp[-1, s]):
        return None
    path = []
    for t in range(T - 1, -1, -1):
        path.append(s)
        s -= bp[t, s]
    return token_spans(np.array(path[::-1]), L)


def _same(targets):
    return [False] + [a == b for a, b in pairwise(targets)]


@pytest.mark.parametrize("seed", range(25))
def test_viterbi_matches_reference(seed):
    rng = np.random.default_rng(seed)
    L = int(rng.integers(1, 8))
    targets = list(rng.integers(0, 3, size=L))  # small alphabet -> plenty of repeats
    T = int(rng.integers(L + sum(_same(targets)), 30))
    em = np.log(rng.dirichlet(np.ones(L + 1), size=T))
    path = ctc_viterbi(em, L, _same(targets))
    assert token_spans(path, L) == reference_viterbi(em, targets)


def test_viterbi_finds_spikes_and_rejects_impossible():
    T, L = 12, 3
    em = np.full((T, L + 1), np.log(0.01))
    em[:, 0] = np.log(0.97)
    for k, f in enumerate([2, 6, 9]):
        em[f, k + 1], em[f, 0] = np.log(0.97), np.log(0.01)
    assert token_spans(ctc_viterbi(em, L, [False] * 3), L) == [(2, 2), (6, 6), (9, 9)]
    # "a a" needs a blank between: 2 tokens + 1 blank > 2 frames
    assert ctc_viterbi(em[:2, :3], 2, [False, True]) is None
    assert ctc_viterbi(em[:0], 1, [False]) is None


def test_frame_times():
    t = frame_times(28)
    assert t[:3] == pytest.approx([0, 0.08, 0.16])
    assert t[12] == pytest.approx(0.96) and t[13] == pytest.approx(1.0) and t[27] == pytest.approx(2.08)
    assert frame_times(14, "uniform")[13] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        frame_times(3, "nope")


def test_group_words():
    tokens = ["ĠHello", ",", "Ġwor", "ld", "Ġ-", "Ġok"]
    ids = list(range(6))
    texts = {0: " Hello", 1: ",", 2: " wor", 3: "ld", 4: " -", 5: " ok"}
    spans = [(0, 0), (1, 1), (3, 3), (4, 4), (6, 6), (8, 8)]
    words = group_words(
        ids, spans, [0.9, 0.5, 0.8, 0.6, 0.7, 1.0], tokens, lambda xs: "".join(texts[i] for i in xs)
    )
    assert [(w.text, w.first, w.last) for w in words] == [("Hello,", 0, 1), ("world-", 3, 4), ("ok", 8, 8)]
    assert words[0].score == pytest.approx(0.7)


def test_end_rules():
    starts = np.array([0.0, 0.5, 3.0])
    spike_end = np.array([0.1, 0.6, 3.1])
    regions = [(0.0, 1.0), (2.9, 4.0)]
    assert END_RULES["spike"](starts, spike_end, regions).tolist() == pytest.approx([0.1, 0.6, 3.1])
    assert END_RULES["next_start"](starts, spike_end, regions).tolist() == pytest.approx([0.5, 3.0, 3.1])
    assert END_RULES["midpoint"](starts, spike_end, regions).tolist() == pytest.approx([0.3, 1.8, 3.1])
    # word 2 stops at its region's end (1.0) instead of running over the pause; the last word fills its region
    assert END_RULES["region"](starts, spike_end, regions).tolist() == pytest.approx([0.5, 1.0, 4.0])


class FakeHead:
    """Spikes for target k at frame 3k + 1 on a blank background."""

    def emissions(self, states, classes):
        T, L = states.shape[0], len(classes) - 1
        em = np.full((T, L + 1), np.log(0.01))
        em[:, 0] = np.log(0.98)
        for k in range(min(L, (T - 2) // 3 + 1)):  # only spikes that fit
            em[3 * k + 1, k + 1], em[3 * k + 1, 0] = np.log(0.98), np.log(0.01)
        return em


class FakeTokenizer:
    vocab: ClassVar[dict] = {7: "ĠThe", 8: "Ġcat", 9: ".", 10: "ĠIt", 11: "Ġsat"}
    text: ClassVar[dict] = {7: " The", 8: " cat", 9: ".", 10: " It", 11: " sat"}

    def convert_ids_to_tokens(self, ids):
        return [self.vocab[i] for i in ids]

    def decode(self, ids, **kw):
        return "".join(self.text[i] for i in ids)


def test_align_tokens_and_sentence_split():
    al = Aligner(FakeHead(), FakeTokenizer(), AlignOptions(end_rule="next_start", start_shift=0.0))
    words = al.align_tokens([7, 8, 9, 10, 11], torch.zeros(20, 4), chunk_start=10.0)
    assert [w.word for w in words] == ["The", "cat.", "It", "sat"]
    # token k spikes at frame 3k+1; start = frame centre - 0.04. Frame 13 opens block 1 at exactly 1.00 s
    assert [w.start for w in words] == pytest.approx([10.04, 10.28, 10.76, 10.96])
    assert words[0].end == pytest.approx(words[1].start)
    segs = split_sentences(Segment(words[0].start, words[-1].end, "The cat. It sat", words))
    assert [s.text for s in segs] == ["The cat.", "It sat"]
    assert al.align_tokens([7] * 30, torch.zeros(5, 4), 0.0) is None  # can't fit


def test_proportional_words_skip_gaps():
    words = proportional_words("aa bb", [(0.0, 1.0), (5.0, 6.0)])
    assert [(w.word, w.start, w.end) for w in words] == [("aa", 0.0, 1.0), ("bb", 5.0, 6.0)]
    assert all(w.score == 0.0 for w in words) and proportional_words("", [(0, 1)]) == []


def test_split_sentences_keeps_unsplit():
    seg = Segment(0, 1, "no words", [])
    assert split_sentences(seg) == [seg]
    w = [WordSegment("Hi!", 0, 0.5, 1), WordSegment('"Yes."', 0.6, 1.0, 1)]
    assert [s.text for s in split_sentences(Segment(0, 1, "", w))] == ["Hi!", '"Yes."']


def test_viterbi_long_sequence():
    """More than 63 tokens (> 127 states): the backtrack once overflowed an int8 (numpy 2 promotion)."""
    L, T = 200, 500
    em = np.full((T, L + 1), np.log(0.001))
    em[:, 0] = np.log(0.99)
    frames = np.arange(L) * 2 + 50
    for k, f in enumerate(frames):
        em[f, k + 1], em[f, 0] = np.log(0.99), np.log(0.001)
    spans = token_spans(ctc_viterbi(em, L, [False] * L), L)
    assert [a for a, _ in spans] == frames.tolist()


def test_word_times_never_overlap():
    from rumblex.aligner import AlignedWord, word_times

    words = [AlignedWord("a", 0, 0, 1.0), AlignedWord("b", 1, 1, 1.0), AlignedWord("c", 5, 6, 1.0)]
    starts, ends = word_times(words, 10, 2.0, [(2.0, 3.0)], "block", "spike", start_shift=-0.03)
    assert (ends[:-1] <= starts[1:] + 1e-9).all() and (ends > starts).all()
    assert starts[0] == pytest.approx(2.0)  # shift can't move a start before the chunk


def test_clamp_to_chunk():
    from rumblex.aligner import clamp_to_chunk

    ws = [
        WordSegment("a", 9.9, 10.2, 1),
        WordSegment("b", 10.3, 12.012, 1),
        WordSegment("c", 12.011, 12.03, 1),
    ]
    clamp_to_chunk(ws, 10.0, 12.0)
    assert [(w.start, w.end) for w in ws] == [(10.0, 10.2), (10.3, 12.0), (11.99, 12.0)]


def test_split_sentences_ignores_titles():
    w = [WordSegment("Mr.", 0, 0.3, 1), WordSegment("Quilter", 0.3, 0.8, 1), WordSegment("spoke.", 0.8, 1.2, 1),
         WordSegment("Dhr.", 1.3, 1.5, 1), WordSegment("Jansen", 1.5, 2.0, 1), WordSegment("ook.", 2.0, 2.3, 1)]  # fmt: skip
    assert [s.text for s in split_sentences(Segment(0, 2.3, "", w))] == [
        "Mr. Quilter spoke.",
        "Dhr. Jansen ook.",
    ]


def _tone_gap_tone():
    """2 s at 16 kHz: tone 0-0.5 s, silence 0.5-1.0 s, tone 1.0-1.5 s, silence after."""
    t = np.arange(32000) / 16000
    a = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    a[(t >= 0.5) & (t < 1.0)] = 0
    a[t >= 1.5] = 0
    return a


def test_refine_ends_pulls_back_to_silence():
    from rumblex.aligner import refine_ends

    a = _tone_gap_tone()
    starts, spike_end = np.array([0.0, 1.0]), np.array([0.45, 1.45])
    ends = np.array([1.0, 1.9])  # next_start for word 0; word 1 runs to 1.9
    out = refine_ends(starts, spike_end, ends, a, 0.0, silence_db=25, min_silence=0.03)
    assert out[0] == pytest.approx(0.5, abs=0.011) and out[1] == pytest.approx(1.5, abs=0.011)
    # no silence in the window -> unchanged
    same = refine_ends(
        np.array([0.0]), np.array([0.1]), np.array([0.4]), a, 0.0, silence_db=25, min_silence=0.03
    )
    assert same[0] == pytest.approx(0.4)


def test_refine_starts_moves_to_onset():
    from rumblex.aligner import refine_starts

    a = _tone_gap_tone()
    starts = np.array([0.02, 0.98])
    out = refine_starts(starts, np.array([0.06, 1.06]), np.array([0.45, 1.45]), a, 0.0, 25, 0.03)
    assert out[0] == pytest.approx(0.02)  # nothing silent before word 0's spike
    assert out[1] == pytest.approx(1.0, abs=0.011)


def test_align_tokens_uses_audio_when_given():
    class Head:  # spikes at frames 1 and 14 (0.08 s and 1.04 s)
        def emissions(self, states, classes):
            em = np.full((states.shape[0], len(classes)), np.log(0.01))
            em[:, 0] = np.log(0.98)
            for k, f in enumerate([1, 14][: len(classes) - 1]):
                em[f, k + 1], em[f, 0] = np.log(0.98), np.log(0.01)
            return em

    al = Aligner(Head(), FakeTokenizer(), AlignOptions(start_shift=0.0))
    without = al.align_tokens([7, 8], torch.zeros(26, 4), 0.0)
    with_audio = al.align_tokens([7, 8], torch.zeros(26, 4), 0.0, audio=_tone_gap_tone())
    assert without[0].end == pytest.approx(without[1].start)  # next_start: runs over the pause
    assert with_audio[0].end == pytest.approx(0.5, abs=0.011)  # stops where the tone stops


def _items(states):
    from rumblex.transcriber import ChunkTranscript
    from rumblex.types import TranscriptResult
    from rumblex.vad import SpeechChunk, SpeechRegion

    chunk = SpeechChunk(0, 32000, (SpeechRegion(0, 32000),))
    it = ChunkTranscript(chunk, "The cat. It sat", "English", [7, 8, 9, 10, 11], states)
    return TranscriptResult([], "English", {"chunk_transcripts": [it]})


def test_align_needs_encoder_states():
    al = Aligner(FakeHead(), FakeTokenizer(), AlignOptions(start_shift=0.0))
    with pytest.raises(ValueError, match="keep_encoder_states=True"):
        al.align(_items(None))
    res = al.align(_items(torch.zeros(3, 4)))  # states too short to align: proportional fallback still works
    assert res.metadata["alignment"]["fallback_chunks"] == 1
    assert [w.word for s in res.segments for w in s.words] == ["The", "cat.", "It", "sat"]


def test_group_words_leading_punctuation_joins_next_word():
    tokens = ['Ġ"', "ĠHello", "Ġworld"]
    texts = {0: ' "', 1: " Hello", 2: " world"}
    words = group_words(
        [0, 1, 2], [(0, 0), (2, 2), (4, 4)], [0.5, 0.9, 0.8], tokens, lambda xs: "".join(texts[i] for i in xs)
    )
    assert [(w.text, w.first, w.last) for w in words] == [('"Hello', 2, 2), ("world", 4, 4)]
    only = group_words([0], [(1, 1)], [0.5], tokens[:1], lambda xs: '"')
    assert [(w.text, w.first) for w in only] == [('"', 1)]


def test_proportional_words_are_plain_floats():
    words = proportional_words("aa bb cc", [(0.0, 1.0), (2.0, 3.5)])
    assert all(type(w.start) is float and type(w.end) is float for w in words)


def test_ctc_head_from_local_folder(tmp_path, monkeypatch):
    import json

    import huggingface_hub
    from safetensors.torch import save_file

    from rumblex.aligner import BLANK, NUM_CLASSES, CTCHead

    (tmp_path / "ctc").mkdir()
    vocab = {"blank": BLANK, "token_ids": list(range(BLANK))}
    (tmp_path / "ctc" / "ctc_vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    save_file({"weight": torch.zeros(NUM_CLASSES, 2), "bias": torch.zeros(NUM_CLASSES)},
              str(tmp_path / "ctc" / "ctc_head.safetensors"))  # fmt: skip

    def no_download(*a, **kw):
        raise AssertionError("should read the local folder")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_download)
    head = CTCHead.from_pretrained(str(tmp_path))
    assert head.weight.shape == (NUM_CLASSES, 2)
