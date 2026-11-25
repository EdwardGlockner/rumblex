"""Hotword trie and logits processor, with toy tokens and a fake decoding loop (no model)."""

import numpy as np
import pytest
import torch

from rumblex.hotwords import (
    ROOT,
    Entry,
    HotwordProcessor,
    Hotwords,
    TokenTrie,
    case_forms,
    parse_entries,
    parse_entry,
    title_case,
)
from rumblex.transcriber import ASR_TEXT_ID, EOS_ID, FIRST_SPECIAL_ID, RumbleTranscriber, _TokenLogprobs
from rumblex.vad import SpeechChunk, SpeechRegion

V = 100_000
WS = np.zeros(V, dtype=bool)
WS[[10, 20, 30]] = True  # word-start ("Ġ...") tokens; 11, 12, 21 are word-internal
P = [1, ASR_TEXT_ID]  # a prompt with a forced language: the text starts right away


def toy_encode(text: str) -> list[int]:
    """One id per character; a leading space shifts the first id by 1000."""
    ids = [ord(c) for c in text.lstrip(" ")]
    if text.startswith(" ") and ids:
        ids[0] += 1000
    return ids


class ToyTokenizer:
    """`toy_encode` behind the parts of the tokenizer interface that Hotwords uses."""

    @property
    def added_tokens_decoder(self):
        return {FIRST_SPECIAL_ID: "<|endoftext|>", ASR_TEXT_ID: "<asr_text>"}

    def encode(self, text, add_special_tokens=False):
        return toy_encode(text)

    def convert_tokens_to_ids(self, token):
        return {"<asr_text>": ASR_TEXT_ID}[token]

    def get_vocab(self):
        return {"Ġ" + chr(c): 1000 + c for c in range(32, 127)} | {chr(c): c for c in range(32, 127)}


def toy_trie() -> TokenTrie:
    t = TokenTrie()
    t.insert([10, 11, 12], 0)
    t.insert([20, 21], 1)
    t.insert([11, 12], 2)  # a bare form: only allowed at the start of the text
    return t


def proc(trie, boost, **guards) -> HotwordProcessor:
    """A processor with every guard off unless the test turns it on."""
    kw = {"repeat_decay": 1.0, "abandon_decay": 1.0, "eos_guard": False, **guards}
    return HotwordProcessor(trie, WS, boost, **kw)


def step(p, ids):
    return p(torch.tensor(ids), torch.zeros(len(ids), V))


# ---------------------------------------------------------------------------------------------- trie


def test_insert_walk_and_shared_prefix():
    t = TokenTrie()
    a = t.insert([5, 6, 7], 0)
    b = t.insert([5, 6, 8], 1)
    assert t.num_nodes == 5  # root, 5, 6, 7, 8
    assert t.walk([5, 6, 7]) == a and t.walk([5, 6, 8]) == b
    assert t.walk([5, 9]) == -1
    assert t.child(t.walk([5]), 6) == t.walk([5, 6])
    assert t.terminal == {a: [0], b: [1]}
    toks, nodes = t.children(t.walk([5, 6]))
    assert toks.tolist() == [7, 8] and nodes.tolist() == [a, b]


def test_two_entries_same_path_share_terminal():
    t = TokenTrie()
    n1 = t.insert([1, 2], 0)
    n2 = t.insert([1, 2], 1)
    assert n1 == n2 and t.terminal[n1] == [0, 1]


def test_children_match_the_edges():
    rng = np.random.default_rng(0)
    t = TokenTrie()
    for e in range(200):
        t.insert(rng.integers(0, 6, size=rng.integers(1, 5)).tolist(), e)
    for node in range(t.num_nodes):
        toks, kids = t.children(node)
        assert all(t.child(node, int(k)) == int(c) for k, c in zip(toks, kids))
        assert len(toks) == sum(1 for p in t.parent[1:] if p == node)


def test_empty_path_rejected():
    with pytest.raises(ValueError):
        TokenTrie().insert([], 0)


def test_case_forms_and_parse():
    assert case_forms("ASML") == ["ASML", "asml"]
    assert case_forms("pipeline", ("as_given", "title")) == ["pipeline", "Pipeline"]
    assert parse_entry(" Nvidia | Nvidia's ") == Entry("Nvidia", ("Nvidia's",))


def test_from_entries_expands_forms():
    t = TokenTrie.from_entries(["ab | abs"], toy_encode, cases=("as_given", "upper"))
    # forms: ab, abs; cases: ab/AB, abs/ABS; each spaced + bare -> 8 distinct paths
    assert sum(len(p) for p in t.paths) == 8
    assert t.walk(toy_encode(" ABS")) in t.terminal
    assert t.walk(toy_encode("ab")) in t.terminal
    no_bare = TokenTrie.from_entries(["ab"], toy_encode, cases=("as_given",), bare=False)
    assert no_bare.paths == [[tuple(toy_encode(" ab"))]]


def test_capitals_get_a_title_case_form():
    assert title_case("GOLDMAN SACHS") == "Goldman Sachs"
    got = parse_entries(["GOLDMAN SACHS", "SiTime", "IBM | IBMS", "paul betz"])
    assert got == [
        Entry("GOLDMAN SACHS", ("Goldman Sachs",)),
        Entry("SiTime"),  # mixed case is left as written
        Entry("IBM", ("IBMS", "Ibm", "Ibms")),
        Entry("paul betz"),
    ]
    assert parse_entries(["GOLDMAN SACHS"], title_caps=False) == [Entry("GOLDMAN SACHS")]


def test_parse_entries_drops_blank_and_repeated_lines():
    assert parse_entries(["", "  ", " | ", "Stifel", "Stifel ", Entry("Stifel")]) == [Entry("Stifel")]


def test_hotwords_builds_every_written_form():
    hw = Hotwords(["PAUL BETZ", "stifel | stifels"], ToyTokenizer())
    assert len(hw) == 2 and hw.boost == 2.0
    for text in [" PAUL BETZ", " Paul Betz", " paul betz", "Paul Betz", " Stifel", " STIFELS", "stifel"]:
        assert hw.trie.walk(toy_encode(text)) in hw.trie.terminal, text
    assert hw.trie.walk(toy_encode(" Paul betz")) == -1
    assert hw.word_start[1000 + ord("P")] and not hw.word_start[ord("P")]
    assert len(Hotwords("Stifel", ToyTokenizer())) == 1  # a bare string is one entry, not its letters


def test_hotwords_refuses_another_tokenizer_layout():
    class Other(ToyTokenizer):
        def convert_tokens_to_ids(self, token):
            return 7

    with pytest.raises(ValueError, match="tokenizer layout"):
        Hotwords(["Stifel"], Other())


def test_real_tokenizer_paths():
    tr = pytest.importorskip("transformers")
    try:
        tok = tr.AutoTokenizer.from_pretrained("Reson8/Rumble-2-Flash")
    except OSError:
        pytest.skip("tokenizer not available offline")
    hw = Hotwords(["ziekenhuisopname | ziekenhuisopnames", "paracetamol", "GOLDMAN SACHS"], tok)

    def enc(s):
        return tok.encode(s, add_special_tokens=False)

    for text in [" ziekenhuisopname", "Ziekenhuisopnames", " PARACETAMOL", "paracetamol", " Goldman Sachs"]:
        assert hw.trie.walk(enc(text)) in hw.trie.terminal, text
    assert hw.trie.terminal[hw.trie.walk(enc(" paracetamol"))] == [1]
    assert hw.word_start[enc(" paracetamol")[0]] and not hw.word_start[enc("paracetamol")[0]]


# ---------------------------------------------------------------------------------------- processor


def test_bonus_follows_the_trie():
    p = proc(toy_trie(), 2.0)
    s = step(p, [P])[0]
    # start of text: all root children (spaced and bare)
    assert s[10] == 2 and s[20] == 2 and s[11] == 2 and s[12] == 0
    s = step(p, [P + [10]])[0]
    assert s[11] == 2  # continue the entry
    assert s[10] == 2 and s[20] == 2  # a new word may start an entry
    assert s[21] == 0
    s = step(p, [P + [10, 11]])[0]
    assert s[12] == 2 and s[11] == 0  # bare form not allowed mid-text
    s = step(p, [P + [10, 11, 12]])[0]
    assert s[12] == 0 and s[10] == 2  # entry done: only new word starts


def test_no_bonus_before_asr_text_and_mismatch_resets():
    p = proc(toy_trie(), 1.5)
    s = step(p, [[1, 2]])[0]
    assert s.abs().sum() == 0  # unforced language: still in "language X"
    s = step(p, [[1, 2, 7]])[0]
    assert s.abs().sum() == 0
    s = step(p, [[1, 2, 7, ASR_TEXT_ID]])[0]
    assert s[11] == 1.5  # text start
    step(p, [[1, 2, 7, ASR_TEXT_ID, 20]])
    s = step(p, [[1, 2, 7, ASR_TEXT_ID, 20, 5]])[0]  # 5 breaks the match
    assert s[21] == 0 and s[10] == 1.5


def test_written_entry_gets_no_second_bonus():
    t = TokenTrie()
    t.insert([10, 11], 0)
    t.insert([10, 12], 1)  # shares the first token with entry 0
    t.insert([20], 2)
    p = proc(t, 3.0, repeat_decay=0.0)
    step(p, [P])
    step(p, [P + [10]])
    s = step(p, [P + [10, 11]])[0]  # entry 0 done
    assert s[10] == 3  # entry 1 still lives under the shared first token
    s = step(p, [P + [10, 11, 10]])[0]
    assert s[11] == 0 and s[12] == 3
    s = step(p, [P + [10, 11, 10, 12]])[0]  # entry 1 done too
    assert s[10] == 0 and s[20] == 3


def test_abandon_decay_penalises_the_dead_prefix_only():
    t = TokenTrie()
    t.insert([10, 11, 12], 0)  # e.g. " op" "ge" "wekt"
    t.insert([10, 13], 1)  # another entry under the same first piece
    t.insert([20], 2)
    p = proc(t, 8.0, abandon_decay=0.5)
    step(p, [P])
    s = step(p, [P + [10]])[0]
    assert s[11] == 8 and s[13] == 8
    s = step(p, [P + [10, 30]])[0]  # " op" then an unrelated word: the match died at node " op"
    assert s[10] == 4 and s[20] == 8
    s = step(p, [P + [10, 30, 10]])[0]  # restart from the root: deeper nodes keep their bonus
    assert s[11] == 8 and s[13] == 8
    s = step(p, [P + [10, 30, 10, 10]])[0]  # " op op": dies again, halves again
    assert s[10] == 2


def test_abandon_decay_ignores_completed_matches():
    t = TokenTrie()
    t.insert([20], 0)
    t.insert([10, 11], 1)
    p = proc(t, 3.0, abandon_decay=0.0)
    step(p, [P])
    step(p, [P + [20]])
    s = step(p, [P + [20, 30]])[0]  # entry 0 ended at a terminal node: not abandoned
    assert s[20] == 3 and s[10] == 3


def test_abandon_penalty_survives_a_recompute():
    t = TokenTrie()
    t.insert([10, 11, 12], 0)
    t.insert([10, 13], 1)  # completes, which recomputes the bonuses on its path
    p = proc(t, 8.0, repeat_decay=0.5, abandon_decay=0.5)
    step(p, [P])
    step(p, [P + [10]])
    s = step(p, [P + [10, 30]])[0]  # " op" abandoned: halved
    assert s[10] == 4
    step(p, [P + [10, 30, 10]])
    s = step(p, [P + [10, 30, 10, 13]])[0]  # entry 1 done: " op" = 0.5 (abandon) * max(8, 8 * 0.5)
    assert s[10] == 4


def test_two_matches_one_dies_one_continues():
    t = TokenTrie()
    t.insert([10, 20, 31], 0)  # a two-word entry
    t.insert([20, 21], 1)  # starts inside entry 0
    p = proc(t, 2.0, abandon_decay=0.5)
    step(p, [P])
    step(p, [P + [10]])
    s = step(p, [P + [10, 20]])[0]  # two live matches
    assert s[31] == 2 and s[21] == 2
    s = step(p, [P + [10, 20, 21]])[0]  # entry 0 dies, entry 1 completes
    assert s[10] == 1  # only the dead prefix of entry 0 is halved
    assert s[20] == 2  # entry 1's own path is untouched


def test_rows_keep_their_own_decays():
    p = proc(toy_trie(), 4.0, repeat_decay=0.0, abandon_decay=0.0)
    step(p, [P, P])
    step(p, [P + [20], P + [5]])
    s = step(p, [P + [20, 21], P + [5, 5]])  # row 0 writes entry 1
    assert s[0, 20] == 0 and s[1, 20] == 4
    step(p, [P + [20, 21, 10], P + [5, 5, 5]])
    s = step(p, [P + [20, 21, 10, 30], P + [5, 5, 5, 5]])  # row 0 abandons its match
    assert s[0, 10] == 0 and s[1, 10] == 4
    assert p.base[p.trie.child(ROOT, 10)] == 4  # the shared bonuses are never changed


def test_no_bonus_when_the_model_wants_to_stop():
    p = proc(toy_trie(), 5.0, eos_guard=True)
    scores = torch.zeros(2, V)
    scores[0, EOS_ID] = 10.0  # row 0: <|im_end|> is the argmax
    s = p(torch.tensor([P, P]), scores)
    assert s[0, 10] == 0 and s[1, 10] == 5


def test_stop_guard_is_per_step_and_the_match_keeps_tracking():
    p = proc(toy_trie(), 5.0, eos_guard=True)
    step(p, [P])
    scores = torch.zeros(1, V)
    scores[0, EOS_ID] = 10.0
    s = p(torch.tensor([P + [10]]), scores)[0]  # guarded step: no bonus at all
    assert s[11] == 0 and s[10] == 0
    s = step(p, [P + [10, 11]])[0]  # the match still advanced during the guarded step
    assert s[12] == 5


def test_default_guards_are_the_shipped_ones():
    p = HotwordProcessor(toy_trie(), WS)
    assert (p.boost, p.repeat_decay, p.abandon_decay, p.eos_guard) == (2.0, 0.0, 0.5, True)


def test_more_rows_than_at_the_start_is_an_error():
    p = proc(toy_trie(), 2.0)
    step(p, [P])
    with pytest.raises(ValueError, match="greedy"):
        step(p, [P + [10], P + [20]])


# ------------------------------------------------------------------------------------ decoding loop


def greedy(step_logits, bias=None):
    """generate()'s greedy loop on canned logits: rows of {token: logit} per step, one batch row."""
    rec = _TokenLogprobs(bias)
    ids = torch.tensor([P])
    for want in step_logits:
        scores = torch.zeros(1, V)
        for tok, x in want.items():
            scores[0, tok] = x
        scores = rec(ids, scores)
        ids = torch.cat([ids, scores.argmax(-1, keepdim=True)], 1)
    toks, lps = rec.rows()
    assert toks[0] == ids[0, len(P) :].tolist()
    return toks[0], lps[0]


def test_hotword_wins_a_close_step_and_keeps_its_own_logprob():
    # the model slightly prefers 30 over the entry's first token 20, then is sure about 21 and the stop
    logits = [{30: 5.0, 20: 4.0}, {21: 9.0}, {EOS_ID: 9.0}]
    plain, lp_plain = greedy(logits)
    assert plain == [30, 21, EOS_ID]
    t = TokenTrie()
    t.insert([20, 21], 0)
    biased, lp = greedy(logits, HotwordProcessor(t, WS))
    assert biased == [20, 21, EOS_ID]
    first = torch.zeros(V)
    first[30], first[20] = 5.0, 4.0
    assert lp[0] == pytest.approx(float(first.log_softmax(-1)[20]))  # the log-prob from before the bonus
    assert lp[0] < lp_plain[0]
    assert lp[1] == pytest.approx(lp_plain[1])


def test_bonus_is_too_small_to_override_a_clear_choice():
    t = TokenTrie()
    t.insert([20, 21], 0)
    toks, _ = greedy([{30: 9.0, 20: 4.0}, {EOS_ID: 9.0}], HotwordProcessor(t, WS))
    assert toks == [30, EOS_ID]


# ------------------------------------------------------------------------------------- transcriber


def fake_transcriber(seen):
    tr = object.__new__(RumbleTranscriber)
    tr.batch_size = 2
    tr.tokenizer = ToyTokenizer()
    tr._run_batch = lambda audio, batch, lang, prompt, opts, keep, hotwords=None: (
        seen.append(hotwords) or [None] * len(batch)
    )
    return tr


def chunks(n):
    return [SpeechChunk(0, 16000, (SpeechRegion(0, 16000),)) for _ in range(n)]


def test_transcribe_chunks_compiles_the_list_once():
    seen = []
    tr = fake_transcriber(seen)
    tr.transcribe_chunks(np.zeros(16000, np.float32), chunks(3), hotwords=["Stifel", "PAUL BETZ"])
    assert len(seen) == 2 and seen[0] is seen[1]  # two batches, one compiled list
    assert isinstance(seen[0], Hotwords) and len(seen[0]) == 2
    ready = Hotwords(["Stifel"], tr.tokenizer, boost=3.0)
    tr.transcribe_chunks(np.zeros(16000, np.float32), chunks(1), hotwords=ready)
    assert seen[-1] is ready


def test_transcribe_chunks_without_hotwords():
    seen = []
    tr = fake_transcriber(seen)
    tr.transcribe_chunks(np.zeros(16000, np.float32), chunks(1))
    tr.transcribe_chunks(np.zeros(16000, np.float32), chunks(1), hotwords=[])
    tr.transcribe_chunks(np.zeros(16000, np.float32), chunks(1), hotwords=["", " "])
    assert seen == [None, None, None]


def test_long_list_warns():
    with pytest.warns(UserWarning, match="2001 hotwords"):
        Hotwords([f"term{i}" for i in range(2001)], ToyTokenizer())


def test_vllm_backend_refuses_hotwords():
    pytest.importorskip("openai")
    from rumblex.vllm_backend import VLLMTranscriber

    tr = object.__new__(VLLMTranscriber)
    with pytest.raises(ValueError, match="vLLM backend"):
        tr.transcribe_chunks(np.zeros(16000, np.float32), chunks(1), hotwords=["Stifel"])


def test_list_at_the_limit_does_not_warn(recwarn):
    Hotwords([f"term{i}" for i in range(2000)], ToyTokenizer())
    assert not recwarn.list
