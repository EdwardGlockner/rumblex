"""Word timestamps from Rumble-2-Flash's auxiliary CTC head.

For each chunk, the encoder states captured during transcription go through the CTC head
(`ctc/ctc_head.safetensors`, one Linear 896 -> 99,631; class 99,630 is blank, the rest map 1:1 onto
the decoder tokenizer). The decoder's own text tokens are force-aligned to those log-probs with CTC
Viterbi, and tokens are grouped into words at the leading-space marker.

The head is very peaky (a token is a ~1-frame spike), so the alignment gives good onsets but no word
ends. A word ends where the next one starts (`END_RULES["next_start"]`), pulled back to the first
silence in the audio when the audio is available (`refine_ends`). With the audio, starts also move to
the nearest dip in loudness (`snap_to_dips`). All of it was chosen on LibriSpeech dev_clean and reported on
test_clean in docs/alignment_eval.md.
If alignment fails for a chunk, words are spread over the chunk's VAD speech regions by length.

Words are split at spaces, so languages written without spaces (Chinese, Japanese) get one "word"
per chunk.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

from rumblex.audio import SAMPLE_RATE
from rumblex.types import Segment, TranscriptResult, WordSegment

BLANK = 99_630
NUM_CLASSES = 99_631

# Encoder geometry (docs/model_inspection.md): 100 mel frames (1 s) per conv block -> 13 frames.
# Three stride-2 convs put frame k of a block at mel frame 8k, i.e. 80 ms apart, block-aligned.
BLOCK_SECONDS = 1.0
FRAMES_PER_BLOCK = 13
FRAME_STEP = 0.08


def frame_times(n: int, grid: str = "block") -> np.ndarray:
    """Time in seconds (chunk-local) of each encoder frame's centre.
    `block`: c + 0.08 k for frame k of 1 s block c (the conv geometry). `uniform`: i / 13."""
    i = np.arange(n)
    if grid == "block":
        return (i // FRAMES_PER_BLOCK) * BLOCK_SECONDS + (i % FRAMES_PER_BLOCK) * FRAME_STEP
    if grid == "uniform":
        return i / FRAMES_PER_BLOCK
    raise ValueError(f"unknown frame grid {grid!r}")


def _model_file(model: str, filename: str) -> str:
    local = Path(model) / filename
    if local.is_file():
        return str(local)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(model, filename)


class CTCHead:
    """The checkpoint's CTC head, loaded from its safetensors file (no remote code)."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, device: str | torch.device = "cpu"):
        if weight.shape[0] != NUM_CLASSES:
            raise ValueError(f"CTC head has {weight.shape[0]} classes, expected {NUM_CLASSES}")
        self.device = torch.device(device)
        self.weight = weight.to(self.device, torch.float32)
        self.bias = bias.to(self.device, torch.float32)

    @classmethod
    def from_pretrained(
        cls, model: str = "Reson8/Rumble-2-Flash", device: str | torch.device = "cpu"
    ) -> CTCHead:
        """`model` is a Hugging Face repo id or a local copy of the checkpoint (a folder with `ctc/`)."""
        from safetensors.torch import load_file

        vocab = json.loads(Path(_model_file(model, "ctc/ctc_vocab.json")).read_text(encoding="utf-8"))
        if vocab["blank"] != BLANK or vocab["token_ids"] != list(range(BLANK)):
            raise ValueError("ctc_vocab.json is not the identity mapping this aligner assumes")
        sd = load_file(_model_file(model, "ctc/ctc_head.safetensors"))
        return cls(sd["weight"], sd["bias"], device)

    @torch.inference_mode()
    def emissions(self, states: torch.Tensor, classes: Sequence[int]) -> np.ndarray:
        """Log-probs [frames, len(classes)] of the given classes, normalised over all 99,631."""
        h = states.to(self.device, torch.float32)
        idx = torch.as_tensor(list(classes), device=self.device)
        out = []
        for i in range(0, h.shape[0], 512):  # 512 x 99,631 floats = 200 MB at most
            logits = torch.nn.functional.linear(h[i : i + 512], self.weight, self.bias)
            out.append(logits[:, idx] - torch.logsumexp(logits, dim=-1, keepdim=True))
        return torch.cat(out).cpu().numpy() if out else np.zeros((0, len(idx)), np.float32)

    @torch.inference_mode()
    def log_probs(self, states: torch.Tensor) -> torch.Tensor:
        """Log-probs over the whole vocabulary [frames, 99,631] (on the head's device)."""
        h = states.to(self.device, torch.float32)
        out = [
            torch.log_softmax(torch.nn.functional.linear(h[i : i + 512], self.weight, self.bias), dim=-1)
            for i in range(0, h.shape[0], 512)
        ]
        return torch.cat(out) if out else torch.zeros((0, self.weight.shape[0]), device=self.device)

    @torch.inference_mode()
    def greedy(self, states: torch.Tensor) -> list[int]:
        """Greedy CTC decoding: the best class per frame, repeats merged, blanks dropped. Rougher than the
        decoder's text, but enough to tell which part of a given transcript a chunk holds."""
        h = states.to(self.device, torch.float32)
        best = [
            torch.nn.functional.linear(h[i : i + 512], self.weight, self.bias).argmax(-1)
            for i in range(0, h.shape[0], 512)
        ]
        ids = torch.cat(best).tolist() if best else []
        return [t for k, t in enumerate(ids) if t != BLANK and (k == 0 or t != ids[k - 1])]


def _viterbi(
    em: np.ndarray, n_tokens: int, same_as_prev: Sequence[bool], optional: Sequence[bool] | None = None
):
    """Viterbi over the 2n+1 extended CTC states: (best final state, its log-prob, back pointers), or None
    if the tokens can't fit in the frames. A back pointer is how many states the path jumped. See
    `ctc_viterbi`."""
    T, L = em.shape[0], n_tokens
    S = 2 * L + 1
    opt = np.zeros(L, dtype=bool) if optional is None else np.asarray(optional, dtype=bool)
    same = np.asarray(same_as_prev, dtype=bool)
    if T == 0 or L - int(opt.sum()) + int(same.sum()) > T:
        return None
    cols = np.zeros(S, dtype=np.int64)
    cols[1::2] = np.arange(1, L + 1)
    e = em[:, cols]  # [T, S]
    # Jumps into token k's state s = 2k+1. From s-2 (token k-1) when the two differ; an optional token
    # differs from its neighbours. Past an optional token k-1: from s-3 (the blank before it) and from s-4
    # (token k-2, when it differs from token k).
    after_opt = np.concatenate(([False], opt[:-1]))
    jump = np.zeros((5, S), dtype=bool)
    jump[:2] = True
    jump[2, 3::2] = (~same | opt | after_opt)[1:]
    jump[3, 1::2] = after_opt
    jump[4, 5::2] = (after_opt & ~same)[2:]
    neg = -np.inf
    dp = np.full(S, neg)
    dp[0] = e[0, 0]
    if S > 1:
        dp[1] = e[0, 1]
    if L > 1 and opt[0]:
        dp[3] = e[0, 3]
    back = np.zeros((T, S), dtype=np.int8)
    n_jumps = 5 if opt.any() else 3
    for t in range(1, T):
        cand = np.full((n_jumps, S), neg)
        cand[0] = dp
        for j in range(1, n_jumps):
            cand[j, j:] = np.where(jump[j, j:], dp[:-j], neg)
        arg = cand.argmax(0)
        dp = cand[arg, np.arange(S)] + e[t]
        back[t] = arg
    # ends on the last token or the blank after it; if that token is optional, also one token earlier
    finals = [s for s in ((S - 1, S - 2, S - 3, S - 4) if L and opt[-1] else (S - 1, S - 2)) if s >= 0]
    s = max(finals, key=lambda s: dp[s])
    if not np.isfinite(dp[s]):
        return None
    return s, float(dp[s]), back


def ctc_viterbi(
    em: np.ndarray, n_tokens: int, same_as_prev: Sequence[bool], optional: Sequence[bool] | None = None
) -> np.ndarray | None:
    """CTC forced alignment, vectorised over states.

    `em`: [T, 1 + n_tokens] log-probs, column 0 = blank, column k+1 = target token k.
    `same_as_prev[k]`: token k equals the token before it (then a blank must separate them). When that
    token is optional, compare with the one before the optional token.
    `optional[k]`: token k may be left out of the path. Two optional tokens must not follow each other.
    Returns the state path [T] over the 2n+1 extended states (even = blank, 2k+1 = token k),
    or None if the tokens can't fit in T frames.
    """
    best = _viterbi(em, n_tokens, same_as_prev, optional)
    if best is None:
        return None
    s, _, back = best
    T = em.shape[0]
    path = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(back[t, s])  # int(): numpy 2 would make s an int8 and overflow past 127 states
    return path


def ctc_score(em: np.ndarray, n_tokens: int, same_as_prev: Sequence[bool]) -> float:
    """Log-prob of the best CTC alignment of the tokens (`em` and arguments as for `ctc_viterbi`); -inf if
    they can't fit. With no tokens it is the log-prob of all blanks."""
    if n_tokens == 0:
        return float(em[:, 0].sum())
    best = _viterbi(em, n_tokens, same_as_prev)
    return -np.inf if best is None else best[1]


def token_spans(path: np.ndarray, n_tokens: int) -> list[tuple[int, int]]:
    """(first, last) frame of each token on a Viterbi path."""
    spans = [[-1, -1] for _ in range(n_tokens)]
    for t, s in enumerate(path):
        if s % 2:
            k = s // 2
            if spans[k][0] < 0:
                spans[k][0] = t
            spans[k][1] = t
    return [(a, b) for a, b in spans]


_HAS_ALNUM = re.compile(r"\w")


def _ends_alnum(text: str) -> bool:
    """True when `text` ends in a letter or digit (ignoring trailing combining marks like Arabic tanwin
    or Indic vowel signs, and Arabic tatweel)."""
    s = unicodedata.normalize("NFC", text).rstrip("\u0640")
    while s and unicodedata.category(s[-1]).startswith("M"):
        s = s[:-1]
    return bool(s) and s[-1].isalnum()


@dataclass
class AlignedWord:
    text: str
    first: int  # first frame of its first token
    last: int  # last frame of its last token
    score: float
    probability: float | None = None  # exp of the mean decoder log-prob of its tokens


def group_words(
    token_ids: Sequence[int],
    spans: Sequence[tuple[int, int]],
    token_probs: Sequence[float],
    tokens: Sequence[str],
    decode: Callable[[list[int]], str],
    token_logprobs: Sequence[float] | None = None,
) -> list[AlignedWord]:
    """Group BPE tokens into words at the leading-space marker (`Ġ`); a word's score is its lowest token
    prob, its probability (with `token_logprobs`) exp of its tokens' mean log-prob. Punctuation-only
    groups (e.g. a lone ` -`) are attached to the previous word, or before the first word (an opening
    quote), and don't change its scores."""

    def probability(g: list[int]) -> float | None:
        if token_logprobs is None:
            return None
        return float(np.exp(np.mean([token_logprobs[k] for k in g])))

    groups: list[list[int]] = []
    for k, tok in enumerate(tokens):
        if not groups or tok.startswith("Ġ"):
            groups.append([k])
        else:
            groups[-1].append(k)
    words: list[AlignedWord] = []
    lead: list[int] = []  # punctuation-only groups before the first word
    for g in groups:
        text = decode([token_ids[k] for k in g]).strip()
        if not text:
            continue
        score = float(min(token_probs[k] for k in g))
        if not _HAS_ALNUM.search(text):
            if words:
                words[-1].text += text
            else:
                lead += g
            continue
        if lead and not words:
            text = decode([token_ids[k] for k in lead]).strip() + text
        words.append(AlignedWord(text, spans[g[0]][0], spans[g[-1]][1], score, probability(g)))
    if lead and not words:  # nothing but punctuation: keep it as one word
        text = decode([token_ids[k] for k in lead]).strip()
        score = float(min(token_probs[k] for k in lead))
        words.append(AlignedWord(text, spans[lead[0]][0], spans[lead[-1]][1], score, probability(lead)))
    return words


def word_key(text: str) -> str:
    """A word reduced to its letters and digits (combining marks and tatweel dropped), case-folded."""
    return "".join(
        ch for ch in unicodedata.normalize("NFC", text).casefold() if ch.isalnum() and ch != "\u0640"
    )


def in_common(a: Sequence[str], b: Sequence[str]) -> list[bool]:
    """For each item of `a`, whether it is part of a longest common subsequence of `a` and `b`."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            d[i][j] = d[i + 1][j + 1] + 1 if a[i] == b[j] else max(d[i + 1][j], d[i][j + 1])
    out = [False] * n
    i = j = 0
    while i < n and j < m:
        if a[i] == b[j]:
            out[i] = True
            i, j = i + 1, j + 1
        elif d[i + 1][j] >= d[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


# Each rule maps word starts and spike ends (seconds, [n]) plus the chunk's speech regions to word ends.


def _end_spike(starts, spike_end, regions):
    return spike_end.copy()


def _end_next_start(starts, spike_end, regions):
    ends = spike_end.copy()
    ends[:-1] = np.maximum(ends[:-1], starts[1:])
    return ends


def _end_midpoint(starts, spike_end, regions):
    ends = spike_end.copy()
    ends[:-1] = np.maximum(ends[:-1], (ends[:-1] + starts[1:]) / 2)
    return ends


def _end_region(starts, spike_end, regions):
    """Run on to the next word's start, but not past the end of the VAD speech region the word is in,
    so a word before a pause doesn't swallow the pause. The last word runs to its region's end."""
    nxt = np.append(starts[1:], np.inf)
    region_end = np.full_like(spike_end, np.inf)
    for r0, r1 in regions:
        inside = (spike_end > r0) & (spike_end <= r1 + 1e-6)
        region_end[inside] = r1
    cap = np.minimum(nxt, region_end)
    cap[~np.isfinite(cap)] = spike_end[~np.isfinite(cap)]
    return np.maximum(spike_end, cap)


END_RULES: dict[str, Callable] = {
    "spike": _end_spike,
    "next_start": _end_next_start,
    "midpoint": _end_midpoint,
    "region": _end_region,
}


# Chosen on LibriSpeech dev_clean (docs/alignment_eval.md): 25 dB below the speech level, for 30 ms.
SILENCE_DB = 25.0
MIN_SILENCE = 0.03


@dataclass
class LoudnessRefinement:
    """Audio for refining word times on loudness. `audio` covers the chunk, starting at global time
    `audio_start`. Moving ends to the first silence helped on dev_clean; moving starts to sound onsets
    did not, so `starts` is off by default."""

    audio: np.ndarray
    audio_start: float
    silence_db: float = SILENCE_DB
    min_silence: float = MIN_SILENCE
    starts: bool = False
    ends: bool = True
    # Move each start to the quietest 10 ms frame (loudness smoothed over `dip_smooth` frames) within
    # `dip_window` seconds of its CTC start; 0 turns it off.
    dip_window: float = 0.0
    dip_smooth: int = 1
    # Let the last word of the chunk end inside its last CTC frame, not only after it.
    last_inside: bool = False


def word_times(
    words: Sequence[AlignedWord],
    n_frames: int,
    chunk_start: float,
    regions: Sequence[tuple[float, float]],
    grid: str = "block",
    end_rule: str = "next_start",
    start_shift: float = 0.0,
    loudness: LoudnessRefinement | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Start/end seconds (global) for aligned words. A frame is FRAME_STEP wide around its centre.
    Words never overlap: an end is clamped to the next word's start."""
    times = frame_times(n_frames, grid) + chunk_start
    spike_start = np.array([times[w.first] for w in words])
    spike_end = np.array([times[w.last] for w in words]) + FRAME_STEP / 2
    return spike_times(spike_start, spike_end, chunk_start, regions, end_rule, start_shift, loudness)


def spike_times(
    spike_start: np.ndarray,
    spike_end: np.ndarray,
    chunk_start: float,
    regions: Sequence[tuple[float, float]],
    end_rule: str = "next_start",
    start_shift: float = 0.0,
    loudness: LoudnessRefinement | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """`word_times` from seconds instead of frames: `spike_start` is the centre of each word's first CTC
    frame, `spike_end` the end of its last one (global seconds)."""
    lr = loudness
    starts = spike_start - FRAME_STEP / 2
    if lr is not None and lr.dip_window > 0:
        starts = snap_to_dips(starts, lr.audio, lr.audio_start, lr.dip_window, lr.dip_smooth)
    starts = np.maximum.accumulate(np.maximum(starts + start_shift, chunk_start))
    if lr is not None and lr.starts:
        starts = refine_starts(
            starts, spike_start, spike_end, lr.audio, lr.audio_start, lr.silence_db, lr.min_silence
        )
    ends = END_RULES[end_rule](starts, spike_end, list(regions))
    ends[:-1] = np.minimum(ends[:-1], starts[1:])
    if lr is not None and lr.ends:
        search_from = spike_end.copy()
        if lr.last_inside:
            search_from[-1] -= FRAME_STEP
        ends = refine_ends(starts, search_from, ends, lr.audio, lr.audio_start, lr.silence_db, lr.min_silence)
    ends = np.maximum(ends, starts + 0.01)
    return starts, ends


ENERGY_HOP = 0.01  # seconds per loudness frame


def frame_db(audio: np.ndarray, sr: int = SAMPLE_RATE, hop: float = ENERGY_HOP) -> np.ndarray:
    """Loudness (dB of RMS) per non-overlapping `hop` frame."""
    n = round(hop * sr)
    k = len(audio) // n
    if k == 0:
        return np.zeros(0)
    f = audio[: k * n].astype(np.float64).reshape(k, n)
    return 10 * np.log10((f**2).mean(axis=1) + 1e-12)


def snap_to_dips(
    starts: np.ndarray, audio: np.ndarray, audio_start: float, window: float, smooth: int = 1
) -> np.ndarray:
    """Move each start to the quietest 10 ms frame within `window` seconds of it, on loudness smoothed
    over `smooth` frames. Words that run into each other often meet at a dip in loudness, which is finer
    than the 80 ms CTC frames."""
    db = frame_db(audio)
    if db.size == 0:
        return starts
    if smooth > 1:
        db = np.convolve(db, np.ones(smooth) / smooth, mode="same")
    out = starts.copy()
    for i, t in enumerate(starts):
        lo = max(0, int(np.floor((t - window - audio_start) / ENERGY_HOP)))
        hi = min(len(db), int(np.ceil((t + window - audio_start) / ENERGY_HOP)))
        if hi > lo:
            out[i] = audio_start + (lo + int(np.argmin(db[lo:hi]))) * ENERGY_HOP
    return out


def _silent_frames(audio: np.ndarray, silence_db: float) -> np.ndarray:
    """Per 10 ms frame: is it `silence_db` below the chunk's speech level (95th percentile loudness)?"""
    db = frame_db(audio)
    return db < np.percentile(db, 95) - silence_db if db.size else np.zeros(0, bool)


def refine_ends(
    starts: np.ndarray,
    spike_end: np.ndarray,
    ends: np.ndarray,
    audio: np.ndarray,
    audio_start: float,
    silence_db: float = SILENCE_DB,
    min_silence: float = MIN_SILENCE,
) -> np.ndarray:
    """Pull word ends back to where the audio goes silent.

    `audio` covers the chunk and starts at global time `audio_start`. Between a word's last CTC spike and
    its current end (usually the next word's start), the first run of at least `min_silence` seconds whose
    loudness is `silence_db` below the chunk's speech level (95th percentile) ends the word. Without
    such a run the end is kept, so this only ever moves ends earlier.
    """
    silent = _silent_frames(audio, silence_db)
    if silent.size == 0:
        return ends
    need = max(1, round(min_silence / ENERGY_HOP))
    # run[i] = number of consecutive silent frames starting at frame i
    run = np.zeros(len(silent) + 1, dtype=np.int64)
    for i in range(len(silent) - 1, -1, -1):
        run[i] = run[i + 1] + 1 if silent[i] else 0
    out = ends.copy()
    for i in range(len(ends)):
        a = max(0, int(np.floor((spike_end[i] - audio_start) / ENERGY_HOP)))
        b = min(len(silent), int(np.ceil((ends[i] - audio_start) / ENERGY_HOP)))
        if b - a < need:
            continue
        hits = np.nonzero(run[a:b] >= need)[0]
        if hits.size:
            out[i] = max(audio_start + (a + hits[0]) * ENERGY_HOP, starts[i] + 0.01)
    return out


def refine_starts(
    starts: np.ndarray,
    spike_start: np.ndarray,
    spike_end: np.ndarray,
    audio: np.ndarray,
    audio_start: float,
    silence_db: float = SILENCE_DB,
    min_silence: float = MIN_SILENCE,
) -> np.ndarray:
    """Move a word's start to where sound resumes after a silence, if there is one.

    Searches between the previous word's last spike (or the chunk start) and the word's first spike for
    the last run of at least `min_silence` seconds of silence (same definition as `refine_ends`); the
    word then starts where that run ends. Words with no such silence before them are left alone.
    """
    silent = _silent_frames(audio, silence_db)
    if silent.size == 0:
        return starts
    need = max(1, round(min_silence / ENERGY_HOP))
    back = np.zeros(len(silent), dtype=np.int64)  # back[i] = consecutive silent frames ending at frame i
    for i in range(len(silent)):
        back[i] = (back[i - 1] + 1 if i else 1) if silent[i] else 0
    out = starts.copy()
    for i in range(len(starts)):
        lo = spike_end[i - 1] if i else audio_start
        a = max(0, int(np.floor((lo - audio_start) / ENERGY_HOP)))
        b = min(len(silent), int(np.ceil((spike_start[i] - audio_start) / ENERGY_HOP)))
        hits = np.nonzero(back[a:b] >= need)[0]
        if hits.size:
            onset = audio_start + (a + hits[-1] + 1) * ENERGY_HOP
            out[i] = min(onset, spike_start[i])
    return out


class BoundaryHead:
    """Corrections to word starts and ends, read from the encoder states around a word's first and last
    CTC frame: one linear map each, in milliseconds. Fitted on LibriSpeech training utterances with ridge
    regression, settings chosen on dev_clean (scripts/boundary_head.py, docs/boundary_head.md). The
    weights ship with the package."""

    FILE = Path(__file__).with_name("boundary_head.safetensors")

    def __init__(
        self,
        start: tuple[torch.Tensor, float, int],
        end: tuple[torch.Tensor, float, int] | None = None,
        start_shift: float = 0.0,
    ):
        """`start` and `end` are (weights, bias, frames used on each side of the word's frame)."""
        self.start, self.end, self.start_shift = start, end, start_shift
        self.dim = len(start[0]) // (2 * start[2] + 1)  # width of the encoder states it was fitted on

    @classmethod
    def load(cls, path: str | Path | None = None) -> BoundaryHead:
        from safetensors import safe_open

        with safe_open(str(path or cls.FILE), "pt") as f:
            sd = {k: f.get_tensor(k).float() for k in f.keys()}  # noqa: SIM118
            chosen = json.loads(f.metadata()["chosen"])
        start = (sd["start.weight"], float(sd["start.bias"]), chosen["start"]["width"])
        end = (
            (sd["end.weight"], float(sd["end.bias"]), chosen["end"]["width"]) if "end.weight" in sd else None
        )
        return cls(start, end, chosen["start"]["start_shift_ms"] / 1000)

    @staticmethod
    def _apply(head: tuple[torch.Tensor, float, int], states: torch.Tensor, idx: Sequence[int]) -> np.ndarray:
        w, b, width = head
        rows = torch.as_tensor(list(idx))[:, None] + torch.arange(-width, width + 1)[None, :]
        x = states[rows.clamp(0, len(states) - 1)].flatten(1)  # past the chunk's edges: the edge frame
        return ((x @ w + b) / 1000).numpy().astype(np.float64)

    def offsets(
        self, states: torch.Tensor, first: Sequence[int], last: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Seconds to add to each word's raw start (the start of its first frame) and raw end (the end of
        its last frame). No end offsets if the file has no end weights."""
        st = states.detach().to("cpu", torch.float32)
        return self._apply(self.start, st, first), None if self.end is None else self._apply(
            self.end, st, last
        )


def boundary_times(
    words: Sequence[AlignedWord],
    n_frames: int,
    chunk_start: float,
    regions: Sequence[tuple[float, float]],
    start_offset: np.ndarray,
    end_offset: np.ndarray | None,
    grid: str = "block",
    start_shift: float = 0.0,
    loudness: LoudnessRefinement | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """`word_times` with a BoundaryHead's offsets. Ends come from `end_offset`, never past the next word's
    start; without it they follow the next-start rule and `loudness`."""
    times = frame_times(n_frames, grid) + chunk_start
    spike_start = np.array([times[w.first] for w in words]) + start_offset
    spike_end = np.array([times[w.last] for w in words]) + FRAME_STEP / 2
    starts, ends = spike_times(
        spike_start, spike_end, chunk_start, regions, "next_start", start_shift, loudness
    )
    if end_offset is not None:
        ends = spike_end + end_offset
        ends[:-1] = np.minimum(ends[:-1], starts[1:])
        ends = np.maximum(ends, starts + 0.01)
    return starts, ends


@dataclass
class AlignOptions:
    # Word starts and ends corrected by the BoundaryHead; it replaces the start shift and the loudness
    # rules below. False gives the timings of 0.5.0 and earlier.
    boundary_head: bool = True
    end_rule: str = "next_start"  # chosen on dev_clean, see docs/alignment_eval.md
    grid: str = "block"
    start_shift: float = -0.010  # added to every start; median start error on dev_clean (block grid)
    # Subtracted from the blank's log-prob before Viterbi, so tokens hold more frames. Chosen on dev_clean
    # (docs/alignment_eval.md); it mostly helps word starts. Scaling the logits (temperature) can't do this:
    # it scales every path's score alike, so the best path stays the same.
    blank_penalty: float = 5.0
    split_sentences: bool = True
    # Loudness refinement, used when the audio is passed to align() (see LoudnessRefinement).
    refine_ends: bool = True
    refine_starts: bool = False
    silence_db: float = SILENCE_DB
    min_silence: float = MIN_SILENCE
    # Starts moved to a loudness dip, and the last word of a chunk may end inside its last frame. Chosen on
    # dev_clean (docs/alignment_eval.md); with them the start shift is recalibrated to dip_start_shift.
    dip_window: float = 0.02
    dip_smooth: int = 7
    last_inside: bool = True
    dip_start_shift: float = -0.004
    # For text given without punctuation (`rumblex.align`): the path may hold one of PUNCTUATION after a
    # word, as it does when the model's own punctuated text is aligned. Subtracted from that token's
    # log-prob, as blank_penalty is from the blank's.
    punct_penalty: float = 2.5  # chosen on dev_clean, see docs/align_text_eval.md
    # Mark each transcribed word with whether the CTC head's own greedy text has it too (`ctc_match`).
    # Costs a second pass of the head over the chunk's frames. Not done for text that was given, like
    # `probability`. See docs/word_trust_eval.md.
    ctc_match: bool = True


PUNCTUATION = (",", ".", "?", "!")


class Aligner:
    def __init__(self, head: CTCHead, tokenizer, opts: AlignOptions | None = None):
        self.head = head
        self.tokenizer = tokenizer
        self.opts = opts or AlignOptions()
        self._punct_ids: list[int] | None = None
        self.boundary = BoundaryHead.load() if self.opts.boundary_head else None
        if self.opts.end_rule not in END_RULES:
            raise ValueError(f"end_rule must be one of {list(END_RULES)}")

    @classmethod
    def from_transcriber(
        cls, transcriber, opts: AlignOptions | None = None, model: str | None = None
    ) -> Aligner:
        head = CTCHead.from_pretrained(model or transcriber.model.name_or_path, device=transcriber.device)
        return cls(head, transcriber.tokenizer, opts)

    def align_tokens(
        self,
        token_ids: Sequence[int],
        states: torch.Tensor,
        chunk_start: float,
        regions: Sequence[tuple[float, float]] = (),
        audio: np.ndarray | None = None,
        token_logprobs: Sequence[float] | None = None,
        optional_punct: bool = False,
    ) -> list[WordSegment] | None:
        """Word timings (global seconds) for one chunk's tokens, or None if CTC alignment fails.
        `regions` are the chunk's speech regions in global seconds. With the decoder's `token_logprobs`
        (one per token) words also get a `probability` and a `ctc_match`. `optional_punct` lets the path
        put a punctuation mark after words that have none (for text that was given without punctuation)."""
        token_ids = list(token_ids)
        if token_logprobs is not None and len(token_logprobs) != len(token_ids):
            raise ValueError("token_logprobs must have one value per token")
        if not token_ids or states is None:
            return None
        n = len(token_ids)
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        slots = self._punct_slots(token_ids, tokens) if optional_punct else []
        em = self.head.emissions(states, [BLANK, *token_ids, *(self._punctuation() if slots else [])])
        same = [False] + [a == b for a, b in pairwise(token_ids)]
        o = self.opts
        punct = em[:, n + 1 :].max(axis=1) - o.punct_penalty if slots else None
        em = em[:, : n + 1]
        vit = em
        if o.blank_penalty:
            vit = em.copy()
            vit[:, 0] -= o.blank_penalty
        if punct is None:
            path = ctc_viterbi(vit, n, same)
            if path is None:
                return None
            spans = token_spans(path, n)
        else:
            # extended sequence: every token, with an optional punctuation token after each slot
            after, cols, opt, same_x, at = set(slots), [0], [], [], []
            for k in range(n):
                at.append(len(opt))
                cols.append(k + 1)
                opt.append(False)
                same_x.append(same[k])
                if k in after:
                    cols.append(n + 1)
                    opt.append(True)
                    same_x.append(False)
            path = ctc_viterbi(np.column_stack([vit, punct])[:, cols], len(opt), same_x, opt)
            if path is None:
                return None
            spans_x = token_spans(path, len(opt))
            spans = [spans_x[i] for i in at]
        # a token's highest CTC probability on its frames; a word scores its weakest token
        # (best of the CTC-only word scores in docs/confidence_eval.md)
        probs = [float(np.exp(em[a : b + 1, k + 1]).max()) for k, (a, b) in enumerate(spans)]
        if punct is not None:  # a word runs to the end of the punctuation the path put after it
            for k in slots:
                last = spans_x[at[k] + 1][1]
                if last >= 0:
                    spans[k] = (spans[k][0], last)
        words = group_words(token_ids, spans, probs, tokens, self._decode, token_logprobs)
        if not words:
            return None
        match: list[bool | None] = [None] * len(words)
        if o.ctc_match and token_logprobs is not None:  # only for the model's own text
            match = self._ctc_match(states, words)
        loudness = None
        shift = o.start_shift
        boundary = self.boundary
        if boundary is not None and states.shape[-1] != boundary.dim:  # states from another encoder
            boundary = None
        if boundary is not None:
            ds, de = boundary.offsets(states, [w.first for w in words], [w.last for w in words])
            if de is None and audio is not None and o.refine_ends:
                loudness = LoudnessRefinement(
                    audio, chunk_start, o.silence_db, o.min_silence, last_inside=True
                )
            starts, ends = boundary_times(
                words, len(em), chunk_start, regions, ds, de, o.grid, boundary.start_shift, loudness
            )
        elif audio is not None and (o.refine_ends or o.refine_starts or o.dip_window > 0):
            loudness = LoudnessRefinement(
                audio, chunk_start, o.silence_db, o.min_silence, starts=o.refine_starts, ends=o.refine_ends,
                dip_window=o.dip_window, dip_smooth=o.dip_smooth, last_inside=o.last_inside,
            )  # fmt: skip
            if o.dip_window > 0:
                shift = o.dip_start_shift
        if boundary is None:
            starts, ends = word_times(
                words, len(em), chunk_start, regions, o.grid, o.end_rule, shift, loudness
            )
        return [
            WordSegment(
                w.text,
                round(float(s), 3),
                round(float(e), 3),
                round(w.score, 3),
                probability=None if w.probability is None else round(w.probability, 3),
                ctc_match=m,
            )
            for w, s, e, m in zip(words, starts, ends, match)
        ]

    def _ctc_match(self, states: torch.Tensor, words: Sequence[AlignedWord]) -> list[bool | None]:
        """For each word, whether the head's greedy text has it too: the word is part of a longest common
        subsequence of the two word lists, compared by `word_key`. None for a word without letters or
        digits."""
        ids = self.head.greedy(states)
        groups: list[list[int]] = []
        for i, tok in zip(ids, self.tokenizer.convert_ids_to_tokens(ids)):
            if not groups or tok.startswith("Ġ"):
                groups.append([i])
            else:
                groups[-1].append(i)
        heard = [k for k in (word_key(self._decode(g)) for g in groups) if k]
        keys = [word_key(w.text) for w in words]
        hit = iter(in_common([k for k in keys if k], heard))
        return [next(hit) if k else None for k in keys]

    def align(
        self, result: TranscriptResult, audio: np.ndarray | None = None, optional_punct: bool = False
    ) -> TranscriptResult:
        """Fill in words for a result from `RumbleTranscriber.transcribe` (needs keep_encoder_states=True).
        Pass the 16 kHz `audio` that was transcribed to enable the loudness refinement of word ends.
        Segments are re-cut at sentence ends when `split_sentences` is on. `optional_punct`: see
        `align_tokens`."""
        items = result.metadata.get("chunk_transcripts")
        if items is None:
            raise ValueError("result has no chunk_transcripts; transcribe with keep_encoder_states=True")
        if any(it.text and it.encoder_states is None for it in items):
            raise ValueError(
                "chunk_transcripts have no encoder states; transcribe with keep_encoder_states=True"
            )
        segments: list[Segment] = []
        n_fallback = 0
        for it in items:
            if not it.text:
                continue
            regions = [(r.start / SAMPLE_RATE, r.end / SAMPLE_RATE) for r in it.chunk.regions] or [
                (it.chunk.start_s, it.chunk.end_s)
            ]
            chunk_audio = None if audio is None else it.chunk.slice(audio)
            words = self.align_tokens(
                it.token_ids, it.encoder_states, it.chunk.start_s, regions, chunk_audio, it.token_logprobs,
                optional_punct,
            )  # fmt: skip
            if words is None:
                n_fallback += 1
                words = proportional_words(it.text, regions)
            clamp_to_chunk(words, it.chunk.start_s, it.chunk.end_s)
            seg = Segment(words[0].start, words[-1].end, it.text, words, language=it.language)
            segments.extend(split_sentences(seg) if self.opts.split_sentences else [seg])
        result.segments = segments
        result.metadata["alignment"] = {
            **vars(self.opts),
            "audio_refinement": audio is not None,
            "fallback_chunks": n_fallback,
        }
        return result

    def _decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, clean_up_tokenization_spaces=False)

    def _punctuation(self) -> list[int]:
        if self._punct_ids is None:
            ids = self.tokenizer.convert_tokens_to_ids(list(PUNCTUATION))
            self._punct_ids = [i for i in ids if i is not None and 0 <= i < BLANK]
        return self._punct_ids

    def _punct_slots(self, token_ids: Sequence[int], tokens: Sequence[str]) -> list[int]:
        """Indices of the tokens that end a word whose text ends in a letter or digit."""
        if not self._punctuation():
            return []
        slots, first = [], 0
        for k in range(len(tokens)):
            if k + 1 < len(tokens) and not tokens[k + 1].startswith("Ġ"):
                continue
            if _ends_alnum(self._decode(list(token_ids[first : k + 1])).strip()):
                slots.append(k)
            first = k + 1
        return slots


def clamp_to_chunk(words: list[WordSegment], start: float, end: float) -> None:
    """Keep word times inside the chunk's real audio. The silence appended before encoding
    (transcriber.END_PAD) can otherwise put the last word's end a few ms past the end."""
    for w in words:
        w.end = round(min(w.end, end), 3)
        w.start = round(min(max(w.start, start), max(w.end - 0.01, start)), 3)


def proportional_words(text: str, regions: Sequence[tuple[float, float]]) -> list[WordSegment]:
    """Spread words over the speech regions in proportion to their length (score 0: not aligned)."""
    words = text.split()
    if not words:
        return []
    total = sum(e - s for s, e in regions)
    weights = np.array([len(w) + 1 for w in words], dtype=float)
    cum = np.concatenate(([0.0], np.cumsum(weights) / weights.sum() * total))

    def at(x: float, start: bool) -> float:
        """Speech-time offset -> wall time. A start exactly on a region boundary goes to the next region."""
        for s, e in regions:
            if x < e - s or (not start and x <= e - s):
                return s + x
            x -= e - s
        return regions[-1][1]

    return [
        WordSegment(w, round(float(at(cum[i], True)), 3), round(float(at(cum[i + 1], False)), 3), 0.0)
        for i, w in enumerate(words)
    ]


_SENTENCE_END = re.compile(r"[.!?؟۔।。？！]['\")\]]*$")
# Words ending in "." that rarely end a sentence (English, Dutch and Arabic titles and common abbreviations).
_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "jr.", "sr.", "vs.", "etc.", "e.g.", "i.e.", "no.",
    "dhr.", "mevr.", "mw.", "bijv.", "o.a.", "z.g.a.n.", "nr.", "ca.", "د.", "أ.", "م.",
}  # fmt: skip


def split_sentences(seg: Segment) -> list[Segment]:
    """Cut a segment after words that end a sentence."""
    if not seg.words:
        return [seg]
    out, cur = [], []
    for w in seg.words:
        cur.append(w)
        if _SENTENCE_END.search(w.word) and w.word.lower() not in _ABBREVIATIONS:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return [
        Segment(ws[0].start, ws[-1].end, " ".join(w.word for w in ws), ws, seg.speaker, seg.language)
        for ws in out
    ]
