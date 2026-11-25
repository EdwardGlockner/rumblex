"""Hotwords: a list of names and terms that the decoder is nudged towards while it transcribes.

The list is compiled into a prefix tree over the decoder's token ids. At every decoding step the tokens
that would start an entry (at a word start) or continue one that is being written get `boost` added to
their logit. Decoding is greedy, so the bonus only changes which token wins at that step.

One entry needs several token paths: a leading space can re-split the whole word and so can a change of
case. Each entry is therefore expanded into its case forms (as given, lower, first letter capital, upper),
each with and without a leading space, and every form's tokenization goes in as its own path. An entry in
all capitals also gets a Title Case form ("GOLDMAN SACHS" -> "Goldman Sachs"), since that is how a
transcript writes it. A line can add its own forms after a bar: "paracetamol | paracetamols".

Three guards keep the bonus from running away:
- an entry that has been written once in a chunk gets no bonus for the rest of that chunk;
- when a match dies half way (its first pieces were written but the entry didn't follow), the bonus on that
  prefix is halved, which stops loops like "op op op" when a word's first piece is a word by itself;
- no bonus at a step where the model's own choice is a special token, so a finished transcript isn't kept
  going.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from rumblex.transcriber import ASR_TEXT_ID, FIRST_SPECIAL_ID

TOKEN_BITS = 17  # 2**17 = 131,072 > 99,631 classes
ROOT = 0
CASES = ("as_given", "lower", "title", "upper")
BOOST = 2.0
REPEAT_DECAY = 0.0
ABANDON_DECAY = 0.5


def case_forms(w: str, cases: Sequence[str] = CASES) -> list[str]:
    """Case variants of `w`, deduplicated, in the order of `cases`. "title" only raises the first letter."""
    make = {
        "as_given": lambda s: s,
        "lower": str.lower,
        "title": lambda s: s[:1].upper() + s[1:],
        "upper": str.upper,
    }
    out: list[str] = []
    for c in cases:
        v = make[c](w)
        if v not in out:
            out.append(v)
    return out


def title_case(s: str) -> str:
    """'GOLDMAN SACHS' -> 'Goldman Sachs'."""
    return " ".join(w[:1] + w[1:].lower() for w in s.split())


@dataclass
class Entry:
    """One hotword: the term as the user wrote it, plus other written forms (plural, another spelling)."""

    term: str
    variants: tuple[str, ...] = ()

    @property
    def forms(self) -> list[str]:
        return [self.term, *[v for v in self.variants if v != self.term]]


def parse_entry(line: str) -> Entry:
    """'paracetamol | paracetamols' -> Entry('paracetamol', ('paracetamols',))."""
    parts = [p.strip() for p in line.split("|") if p.strip()]
    return Entry(parts[0], tuple(parts[1:]))


def parse_entries(lines: Iterable[str | Entry], title_caps: bool = True) -> list[Entry]:
    """Entries from lines, blank and repeated lines dropped. With `title_caps`, every form written in all
    capitals also gets a Title Case form."""
    out: list[Entry] = []
    seen = set()
    for line in lines:
        if isinstance(line, str):
            if not line.replace("|", "").strip():
                continue
            line = parse_entry(line)
        if title_caps:
            extra = [title_case(f) for f in line.forms if f.isupper()]
            line = Entry(line.term, tuple(dict.fromkeys([*line.variants, *extra])))
        key = (line.term, line.variants)
        if key not in seen:
            seen.add(key)
            out.append(line)
    return out


@dataclass
class TokenTrie:
    """Prefix tree of token-id paths. Node 0 is the root (empty prefix)."""

    _edges: dict[int, int] = field(default_factory=dict)  # node << 17 | token -> child
    parent: list[int] = field(default_factory=lambda: [-1])
    token: list[int] = field(default_factory=lambda: [-1])
    terminal: dict[int, list[int]] = field(default_factory=dict)  # node -> entry ids ending there
    entries: list[Entry] = field(default_factory=list)
    paths: list[list[tuple[int, ...]]] = field(default_factory=list)  # entry id -> its token paths
    frozen: dict[str, np.ndarray] | None = None

    @property
    def num_nodes(self) -> int:
        return len(self.parent)

    def child(self, node: int, tok: int) -> int:
        """Child of `node` along `tok`, or -1."""
        return self._edges.get((node << TOKEN_BITS) | tok, -1)

    def insert(self, path: Sequence[int], entry_id: int) -> int:
        """Add one token path ending in `entry_id`; returns its terminal node."""
        if not path:
            raise ValueError("empty token path")
        node = ROOT
        for t in path:
            key = (node << TOKEN_BITS) | int(t)
            nxt = self._edges.get(key)
            if nxt is None:
                nxt = len(self.parent)
                self._edges[key] = nxt
                self.parent.append(node)
                self.token.append(int(t))
            node = nxt
        ids = self.terminal.setdefault(node, [])
        if entry_id not in ids:
            ids.append(entry_id)
            while len(self.paths) <= entry_id:
                self.paths.append([])
            self.paths[entry_id].append(tuple(int(t) for t in path))
        self.frozen = None
        return node

    def walk(self, path: Iterable[int]) -> int:
        """Node reached by following `path` from the root, or -1."""
        node = ROOT
        for t in path:
            node = self.child(node, int(t))
            if node < 0:
                return -1
        return node

    def children(self, node: int) -> tuple[np.ndarray, np.ndarray]:
        """(tokens, child nodes) of `node`."""
        f = self.freeze()
        a, b = f["child_ptr"][node], f["child_ptr"][node + 1]
        return f["child_tok"][a:b], f["child_node"][a:b]

    def freeze(self) -> dict[str, np.ndarray]:
        """The children of every node as flat arrays (recomputed after any insert)."""
        if self.frozen is not None:
            return self.frozen
        n = self.num_nodes
        parent = np.asarray(self.parent, dtype=np.int64)
        token = np.asarray(self.token, dtype=np.int64)
        order = np.lexsort((token[1:], parent[1:])) + 1  # non-root nodes grouped by parent
        ptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(parent[1:], minlength=n), out=ptr[1:])
        self.frozen = {"child_ptr": ptr, "child_tok": token[order], "child_node": order}
        return self.frozen

    @classmethod
    def from_entries(
        cls,
        entries: Sequence[Entry | str],
        encode: Callable[[str], list[int]],
        *,
        cases: Sequence[str] = CASES,
        bare: bool = True,
    ) -> TokenTrie:
        """Build from entries. `encode(text)` must be the decoder tokenizer without special tokens.

        Each form of an entry is expanded to its case variants, each with a leading space and, if `bare`,
        without one (start of the transcript, after a bracket)."""
        trie = cls()
        for e in entries:
            entry = parse_entry(e) if isinstance(e, str) else e
            eid = len(trie.entries)
            trie.entries.append(entry)
            trie.paths.append([])
            seen: set[tuple[int, ...]] = set()
            for form in entry.forms:
                for f in case_forms(form, cases):
                    for text in (" " + f, f) if bare else (" " + f,):
                        path = tuple(encode(text))
                        if path and path not in seen:
                            seen.add(path)
                            trie.insert(path, eid)
        trie.freeze()
        return trie


def word_start_mask(tokenizer, size: int = FIRST_SPECIAL_ID) -> np.ndarray:
    """True for token ids whose piece starts with the byte-level space marker (Ġ)."""
    mask = np.zeros(size, dtype=bool)
    for piece, i in tokenizer.get_vocab().items():
        if i < size and piece.startswith("Ġ"):
            mask[i] = True
    return mask


def check_token_ids(tokenizer) -> None:
    """The bias relies on two facts about Rumble-2-Flash's tokenizer; refuse a model that differs."""
    asr_text = tokenizer.convert_tokens_to_ids("<asr_text>")
    first_special = min(tokenizer.added_tokens_decoder, default=None)
    if asr_text != ASR_TEXT_ID or first_special != FIRST_SPECIAL_ID:
        raise ValueError(
            "hotwords need Rumble-2-Flash's tokenizer layout (<asr_text> = "
            f"{ASR_TEXT_ID}, special tokens from {FIRST_SPECIAL_ID}); this model has <asr_text> = "
            f"{asr_text} and special tokens from {first_special}"
        )


class Hotwords:
    """A hotword list compiled for one tokenizer. Build it once and pass it to several transcribe calls.

    `entries` are strings ("term" or "term | other form | ...") or `Entry` objects. `boost` is the bonus
    added to a token's logit. `title_caps=False` leaves all-capitals entries without a Title Case form."""

    def __init__(
        self,
        entries: Iterable[str | Entry],
        tokenizer,
        *,
        boost: float = BOOST,
        title_caps: bool = True,
    ):
        check_token_ids(tokenizer)
        if isinstance(entries, str):
            entries = [entries]
        self.entries = parse_entries(entries, title_caps)
        self.trie = TokenTrie.from_entries(
            self.entries, lambda s: tokenizer.encode(s, add_special_tokens=False)
        )
        self.word_start = word_start_mask(tokenizer)
        self.boost = float(boost)

    def __len__(self) -> int:
        return len(self.entries)

    def processor(self) -> HotwordProcessor:
        """A fresh logits processor for one generate() call."""
        return HotwordProcessor(self.trie, self.word_start, self.boost)


class _Row:
    """One batch row's place in the trie and the bonuses it has left."""

    def __init__(self, base: np.ndarray, in_text: bool):
        self.in_text = self.at_start = in_text
        self.active: list[int] = []  # nodes of the partial matches still alive
        self.bonus = base  # shared until this row changes it
        self.owned = False
        self.mult: dict[int, float] = {}  # entry -> gain multiplier after it was written
        self.node_mult: dict[int, float] = {}  # node -> bonus multiplier after abandoned matches
        self.root_all = self.root_ws = None  # (tokens, bonuses) of the root's edges: all / word starts only

    def own(self) -> None:
        if not self.owned:
            self.bonus, self.owned = self.bonus.copy(), True


class HotwordProcessor:
    """Logits processor for greedy generate(): adds each row's hotword bonuses to its scores. A row starts
    once it is in the text part of the output: after `<asr_text>`, which is in the prompt when the language
    is forced and generated otherwise."""

    def __init__(
        self,
        trie: TokenTrie,
        word_start: np.ndarray,
        boost: float = BOOST,
        *,
        repeat_decay: float = REPEAT_DECAY,
        abandon_decay: float = ABANDON_DECAY,
        eos_guard: bool = True,
    ):
        self.trie, self.word_start, self.boost = trie, word_start, float(boost)
        self.repeat_decay, self.abandon_decay, self.eos_guard = repeat_decay, abandon_decay, eos_guard
        self.base = np.full(trie.num_nodes, boost, dtype=np.float32)
        self.base[ROOT] = 0.0
        self.rows: list[_Row] | None = None
        self.start_len = 0
        self._below: dict[int, list[int]] = {}  # node -> entries ending in its subtree

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.rows is None:
            self.start_len = input_ids.shape[1]
            self.rows = [_Row(self.base, ASR_TEXT_ID in ids) for ids in input_ids.tolist()]
            for row in self.rows:
                self._root_edges(row, scores.device)
        else:
            if input_ids.shape[0] != len(self.rows):
                raise ValueError("hotwords work with greedy decoding only (one row per chunk)")
            for row, tok in zip(self.rows, input_ids[:, -1].tolist()):
                self._advance(row, tok, scores.device)
        top = scores.argmax(-1).tolist() if self.eos_guard else None
        for b, row in enumerate(self.rows):
            if not row.in_text or (top is not None and top[b] >= FIRST_SPECIAL_ID):
                continue
            parts = [row.root_all if row.at_start else row.root_ws]
            for n in row.active:
                toks, kids = self.trie.children(n)
                if len(toks):
                    parts.append(_edges(toks, row.bonus[kids], scores.device))
            toks = torch.cat([p[0] for p in parts])
            if not len(toks):
                continue
            bonus = torch.cat([p[1] for p in parts]).to(scores.dtype)
            add = torch.zeros(scores.shape[-1], device=scores.device, dtype=scores.dtype)
            add.scatter_reduce_(0, toks, bonus, reduce="amax")
            scores[b] += add
        return scores

    def _root_edges(self, row: _Row, device) -> None:
        toks, kids = self.trie.children(ROOT)
        ws = self.word_start[toks] if len(toks) else np.zeros(0, dtype=bool)
        row.root_all = _edges(toks, row.bonus[kids], device)
        row.root_ws = _edges(toks[ws], row.bonus[kids[ws]], device)

    def _advance(self, row: _Row, tok: int, device) -> None:
        if not row.in_text:
            if tok == ASR_TEXT_ID:
                row.in_text = row.at_start = True
            return
        trie = self.trie
        alive, dead = [], []
        for n in row.active:
            c = trie.child(n, tok)
            if c >= 0:
                alive.append(c)
            elif n not in trie.terminal:
                dead.append(n)
        if row.at_start or (tok < len(self.word_start) and self.word_start[tok]):
            c = trie.child(ROOT, tok)
            if c >= 0 and c not in alive:
                alive.append(c)
        if dead and self.abandon_decay != 1.0:
            self._abandoned(row, dead)
            self._root_edges(row, device)
        row.active = alive
        row.at_start = False
        if self.repeat_decay != 1.0:
            done = [e for n in alive for e in trie.terminal.get(n, ())]
            if done:
                self._written(row, done)
                self._root_edges(row, device)

    def _abandoned(self, row: _Row, nodes: list[int]) -> None:
        """Partial matches died at `nodes`: scale the bonus of every node on their prefixes. Only the prefix
        is penalised, not every entry below it."""
        row.own()
        seen = set()
        for n in nodes:
            while n > ROOT and n not in seen:
                seen.add(n)
                row.node_mult[n] = row.node_mult.get(n, 1.0) * self.abandon_decay
                row.bonus[n] *= self.abandon_decay
                n = self.trie.parent[n]

    def _written(self, row: _Row, entries: list[int]) -> None:
        """Entries just written: scale their gain and recompute the bonuses on their paths, where a node
        keeps the best gain of the entries still below it."""
        row.own()
        trie = self.trie
        for e in entries:
            row.mult[e] = row.mult.get(e, 1.0) * self.repeat_decay
        nodes = set()
        for e in entries:
            for path in trie.paths[e]:
                node = ROOT
                for t in path:
                    node = trie.child(node, t)
                    nodes.add(node)
        for n in nodes:
            best = max((self.boost * row.mult.get(x, 1.0) for x in self._entries_below(n)), default=0.0)
            row.bonus[n] = row.node_mult.get(n, 1.0) * best

    def _entries_below(self, node: int) -> list[int]:
        if node not in self._below:
            out, stack = [], [node]
            while stack:
                n = stack.pop()
                out.extend(self.trie.terminal.get(n, ()))
                stack.extend(self.trie.children(n)[1].tolist())
            self._below[node] = out
        return self._below[node]


def _edges(toks: np.ndarray, bonus: np.ndarray, device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(np.asarray(toks, dtype=np.int64), device=device),
        torch.as_tensor(np.asarray(bonus, dtype=np.float32), device=device),
    )
