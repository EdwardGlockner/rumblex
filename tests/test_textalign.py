"""Aligning a given transcript (rumblex.align, --text) with small fakes for the model and the CTC head."""

from typing import ClassVar

import numpy as np
import pytest
import torch

from rumblex.aligner import BLANK, NUM_CLASSES, Aligner, AlignOptions, CTCHead
from rumblex.textalign import align_text, assign_words, refine_boundaries, word_key
from rumblex.vad import SpeechChunk, SpeechRegion


def test_word_key():
    assert word_key("Mr.") == "mr" and word_key('"Don\'t,"') == "don't" and word_key("—") == ""


def test_assign_words_follows_matches():
    words = ["The", "cat", "sat", "on", "the", "mat."]
    assert assign_words(words, [["the", "cat", "sat"], ["on", "the", "mat"]]) == [0, 0, 0, 1, 1, 1]
    assert assign_words(words, [["the", "cat", "sat", "on", "the", "mat"], []]) == [0] * 6


def test_assign_words_splits_unmatched_words_between_chunks():
    # the rough pass garbled the words around the chunk boundary ("x", "y"): d and e are split between them
    words = list("abcdefgh")
    assert assign_words(words, [list("abcx"), list("yfgh")]) == [0, 0, 0, 0, 1, 1, 1, 1]
    # text before the first match goes to the first chunk, text after the last match to the last
    assert assign_words(["zz", *words], [list("abcd"), list("efgh")])[0] == 0


def test_assign_words_ignores_single_word_matches_in_long_texts():
    first = [f"w{i}" for i in range(15)]
    second = [f"v{i}" for i in range(15)]
    words = first + ["the"] + second
    # chunk 0's rough words end with a stray "the" that is really chunk 1's: one-word matches don't pin
    rough = [[*first[:13], "the"], ["the", *second]]
    owner = assign_words(words, rough)
    assert owner == sorted(owner) and owner[:13] == [0] * 13 and owner[16:] == [1] * 15


def test_assign_words_edge_cases():
    assert assign_words([], [["a"], ["b"]]) == []
    assert assign_words(["a", "b"], [["x"]]) == [0, 0]
    with pytest.raises(ValueError):
        assign_words(["a"], [])


def test_refine_boundaries_moves_to_best_score():
    words = ["a", "b", "c", "d", "e", "f"]
    # chunk 0 should hold "a b c d"; the rough assignment gave it only "a b"
    truth = {0: "a b c d", 1: "e f"}

    def score(c, text):
        return -abs(len(text.split()) - len(truth[c].split()))

    assert refine_boundaries(words, [0, 0, 1, 1, 1, 1], score, 2) == [0, 0, 0, 0, 1, 1]
    # at most `shift` words per boundary
    assert refine_boundaries(words, [0, 0, 1, 1, 1, 1], score, 2, shift=1) == [0, 0, 0, 1, 1, 1]
    # a tie keeps the boundary
    assert refine_boundaries(words, [0, 0, 1, 1, 1, 1], lambda c, t: 0.0, 2) == [0, 0, 1, 1, 1, 1]


def test_refine_boundaries_with_empty_chunk():
    def score(c, text):
        return 0.0

    assert refine_boundaries(["a", "b", "c"], [0, 0, 2], score, 3) == [0, 0, 2]
    assert refine_boundaries([], [], score, 3) == []


def test_greedy_merges_repeats_and_drops_blanks():
    w = torch.zeros(NUM_CLASSES, 3)
    w[5, 0] = w[BLANK, 1] = w[7, 2] = 10.0
    head = CTCHead(w, torch.zeros(NUM_CLASSES))
    e = torch.eye(3)
    states = torch.stack([e[0], e[0], e[1], e[2], e[2], e[1], e[0]])
    assert head.greedy(states) == [5, 7, 5]
    assert head.greedy(torch.zeros(0, 3)) == []


class WordTokenizer:
    """One token per word ("Ġword"), ids by first appearance."""

    def __init__(self):
        self.ids: dict[str, int] = {}
        self.words: dict[int, str] = {}

    def encode(self, text, add_special_tokens=False):
        out = []
        for w in text.split():
            if w not in self.ids:
                self.ids[w] = len(self.ids) + 1
                self.words[self.ids[w]] = w
            out.append(self.ids[w])
        return out

    def convert_ids_to_tokens(self, ids):
        return ["Ġ" + self.words[i] for i in ids]

    def decode(self, ids, **kw):
        return "".join(" " + self.words[i] for i in ids)


class SpikeHead:
    """Emissions: target k spikes at frame 3k + 1. Greedy: the words this chunk was built with."""

    greedy_ids: ClassVar[dict] = {}

    def emissions(self, states, classes):
        T, L = states.shape[0], len(classes) - 1
        em = np.full((T, L + 1), np.log(0.01))
        em[:, 0] = np.log(0.98)
        for k in range(min(L, (T - 2) // 3 + 1)):
            em[3 * k + 1, k + 1], em[3 * k + 1, 0] = np.log(0.98), np.log(0.01)
        return em

    def greedy(self, states):
        return self.greedy_ids[int(states[0, 0])]

    def log_probs(self, states):  # flat: every split of the text scores the same, so boundaries stay put
        return torch.full((states.shape[0], NUM_CLASSES), -float(np.log(NUM_CLASSES)))


class FakeEncoderModel:
    device = "cpu"

    def __init__(self, tok):
        self.tokenizer = tok
        self.encoded = []

    def encode(self, audio, chunks, progress=None):
        self.encoded.append(len(chunks))
        if progress:
            progress(len(chunks), len(chunks))
        # 13 frames per second of chunk; the first value tells SpikeHead.greedy which chunk it is
        return [torch.full((max(1, round(c.duration * 13)), 4), float(i)) for i, c in enumerate(chunks)]


def _chunk(a, b):
    return SpeechChunk(int(a * 16000), int(b * 16000), (SpeechRegion(int(a * 16000), int(b * 16000)),))


def test_align_text_over_two_chunks():
    tok = WordTokenizer()
    model = FakeEncoderModel(tok)
    head = SpikeHead()
    al = Aligner(head, tok, AlignOptions(start_shift=0.0, blank_penalty=0.0, refine_ends=False))
    first, second = "Hello there, this is", "the second part."
    head.greedy_ids = {0: tok.encode(first.lower()), 1: tok.encode("the second")}
    chunks = [_chunk(0.0, 4.0), _chunk(10.0, 14.0)]
    wav = np.zeros(16000 * 15, np.float32)
    seen = []
    res = align_text(model, al, wav, f"{first}\n{second}", chunks=chunks, language="English",
                     progress=lambda d, t: seen.append((d, t)))  # fmt: skip
    words = [w for s in res.segments for w in s.words]
    assert [w.word for w in words] == (first + " " + second).split()
    assert all(w.start < 4.0 for w in words[:4]) and all(w.start >= 10.0 for w in words[4:])
    assert res.language == "English" and res.metadata["aligned_text"] and seen == [(2, 2)]
    assert "chunk_transcripts" not in res.metadata


def test_align_text_short_audio_needs_no_greedy_pass():
    tok = WordTokenizer()
    head = SpikeHead()
    head.greedy_ids = {}  # would raise KeyError if called
    al = Aligner(head, tok, AlignOptions(start_shift=0.0, blank_penalty=0.0, refine_ends=False))
    res = align_text(FakeEncoderModel(tok), al, np.zeros(16000 * 3, np.float32), "one two three")
    assert [w.word for s in res.segments for w in s.words] == ["one", "two", "three"]
    assert align_text(FakeEncoderModel(tok), al, np.zeros(16000, np.float32), "").segments == []
