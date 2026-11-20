"""Benchmark building blocks, used by scripts/benchmark.py and the evaluation scripts.

- Word-timing scoring against LibriSpeech's word alignments: `gilkeyio/librispeech-alignments` on
  Hugging Face (CC-BY-4.0), which repackages Loren Lugosch's Montreal Forced Aligner alignments
  (zenodo.org/record/2619474) with the audio. These are MFA output, not human labels; in them a word's
  end usually equals the next word's start.
- WER-only evaluation sets (`EVAL_SETS`): AMI and Earnings-22 from the Open ASR Leaderboard's test sets,
  FLEURS Dutch. Revisions are pinned.
- WER with Whisper's normalizers (`normalizer_for`), as the Open ASR Leaderboard does.
- The systems under test (rumblex, rumblex with vLLM, WhisperX) behind one `System.run` interface, and
  the accuracy and long-file speed runs.

pyarrow, jiwer and whisper-normalizer come with the `[dev]` extra; WhisperX is not a rumblex dependency.
"""

from __future__ import annotations

import io
import re
import statistics
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np

from rumblex.audio import SAMPLE_RATE

ALIGNMENT_REPO = "gilkeyio/librispeech-alignments"
ALIGNMENT_REVISION = "0daa1eb43dda38ee6ce752e785555380e5628f5c"
TOLERANCES_MS = (20, 50, 100, 200)


@dataclass(frozen=True)
class TimedWord:
    word: str
    start: float
    end: float


@dataclass
class Utterance:
    id: str
    audio: np.ndarray  # 16 kHz mono float32
    transcript: str
    words: list[TimedWord]


_PUNCT = re.compile(r"[^\w']+")


def normalize_word(w: str) -> str:
    """Lowercase, drop punctuation (keep apostrophes: LibriSpeech has "don't", "o'clock")."""
    return _PUNCT.sub("", w.lower()).strip("'")


def match_words(ref: Sequence[str], hyp: Sequence[str]) -> list[tuple[int, int]]:
    """Index pairs (i_ref, j_hyp) of equal words on a minimum-edit-distance alignment.
    Among alignments of equal cost, the one with the most matches wins."""
    n, m = len(ref), len(hyp)
    big = n + m + 1  # score = edits * big - matches: fewer edits first, then more matches
    d = np.zeros((n + 1, m + 1), dtype=np.int64)
    d[:, 0] = np.arange(n + 1) * big
    d[0, :] = np.arange(m + 1) * big
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = d[i - 1, j - 1] + (-1 if ref[i - 1] == hyp[j - 1] else big)
            d[i, j] = min(diag, d[i - 1, j] + big, d[i, j - 1] + big)
    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        eq = ref[i - 1] == hyp[j - 1]
        if d[i, j] == d[i - 1, j - 1] + (-1 if eq else big):
            if eq:
                pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif d[i, j] == d[i - 1, j] + big:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


@dataclass
class TimingScore:
    n_ref: int = 0
    n_hyp: int = 0
    start_err_ms: list[float] = field(default_factory=list)  # signed hyp - ref, matched words only
    end_err_ms: list[float] = field(default_factory=list)

    @property
    def n_matched(self) -> int:
        return len(self.start_err_ms)

    def add(self, other: TimingScore) -> TimingScore:
        self.n_ref += other.n_ref
        self.n_hyp += other.n_hyp
        self.start_err_ms += other.start_err_ms
        self.end_err_ms += other.end_err_ms
        return self

    def summary(self) -> dict:
        out: dict = {"n_ref": self.n_ref, "n_hyp": self.n_hyp, "n_matched": self.n_matched}
        for name, errs in (("start", self.start_err_ms), ("end", self.end_err_ms)):
            if not errs:
                continue
            a = [abs(e) for e in errs]
            out[f"{name}_mae_ms"] = round(statistics.fmean(a), 1)
            out[f"{name}_median_ae_ms"] = round(statistics.median(a), 1)
            out[f"{name}_mean_signed_ms"] = round(statistics.fmean(errs), 1)
            for tol in TOLERANCES_MS:
                out[f"{name}_within_{tol}ms"] = round(sum(x <= tol for x in a) / len(a), 4)
        return out


def score_timings(ref: Sequence[TimedWord], hyp: Sequence[TimedWord]) -> TimingScore:
    """Compare word timings on the words both sides agree on (after normalisation).
    Words that differ (ASR errors) are counted in n_ref/n_hyp but not timed."""
    ref = [w for w in ref if normalize_word(w.word)]
    hyp = [w for w in hyp if normalize_word(w.word)]
    pairs = match_words([normalize_word(w.word) for w in ref], [normalize_word(w.word) for w in hyp])
    s = TimingScore(n_ref=len(ref), n_hyp=len(hyp))
    for i, j in pairs:
        # rounded to µs so float noise doesn't push e.g. 50 ms over a 50 ms tolerance
        s.start_err_ms.append(round((hyp[j].start - ref[i].start) * 1000, 3))
        s.end_err_ms.append(round((hyp[j].end - ref[i].end) * 1000, 3))
    return s


def alignment_parquet(split: str = "test_clean") -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        ALIGNMENT_REPO,
        f"data/{split}-00000-of-00001.parquet",
        repo_type="dataset",
        revision=ALIGNMENT_REVISION,
    )


def librispeech_ids(split: str = "test_clean", path: str | None = None) -> list[str]:
    import pyarrow.parquet as pq

    return sorted(pq.read_table(path or alignment_parquet(split), columns=["id"]).column("id").to_pylist())


def sample_ids(split: str, n: int, seed: int = 0, path: str | None = None) -> list[str]:
    """A fixed random subset of utterance ids (sorted), so runs are comparable."""
    ids = librispeech_ids(split, path)
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(ids, size=min(n, len(ids)), replace=False).tolist())


def iter_librispeech(
    split: str = "test_clean",
    limit: int | None = None,
    path: str | None = None,
    ids: Sequence[str] | None = None,
) -> Iterator[Utterance]:
    """Utterances in id order (deterministic), audio decoded to float32 16 kHz. `ids` restricts to
    those utterances. Empty-word entries (MFA's silence markers) are dropped."""
    import pyarrow.parquet as pq
    import soundfile as sf

    table = pq.read_table(path or alignment_parquet(split), columns=["id", "audio", "transcript", "words"])
    all_ids = np.asarray(table.column("id").to_pylist())
    order = np.argsort(all_ids)
    if ids is not None:
        wanted = set(ids)
        order = np.array([k for k in order if all_ids[k] in wanted], dtype=np.int64)
    if limit is not None:
        order = order[:limit]
    for k in order:
        row = table.slice(int(k), 1).to_pylist()[0]
        audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if sr != SAMPLE_RATE:
            raise ValueError(f"{row['id']}: expected 16 kHz, got {sr}")
        words = [TimedWord(w["word"], w["start"], w["end"]) for w in row["words"] if w["word"].strip()]
        yield Utterance(row["id"], audio, row["transcript"], words)


def normalize_text(s: str) -> str:
    """Lowercase, drop punctuation except apostrophes, collapse spaces (LibriSpeech refs are already so)."""
    s = re.sub(r"[^\w' ]+", " ", s.lower())
    return " ".join(w.strip("'") for w in s.split() if w.strip("'"))


def english_normalizer() -> tuple[str, Callable[[str], str]]:
    """Whisper's English normalizer (numbers, spellings, contractions) when the `whisper-normalizer`
    package is installed, as on the Open ASR Leaderboard; otherwise `normalize_text`. Returns (name, fn)."""
    try:
        from whisper_normalizer.english import EnglishTextNormalizer
    except ImportError:
        return "basic", normalize_text
    return "whisper-english", EnglishTextNormalizer()


def normalizer_for(language: str | None) -> tuple[str, Callable[[str], str]]:
    """English: Whisper's English normalizer. Other languages: Whisper's basic normalizer (lowercase, drop
    punctuation and bracketed text, keep diacritics), as the Open ASR Leaderboard's multilingual track."""
    if language is None or language.lower() in ("english", "en"):
        return english_normalizer()
    try:
        from whisper_normalizer.basic import BasicTextNormalizer
    except ImportError:
        return "basic", normalize_text
    return "whisper-basic", BasicTextNormalizer()


# Extra evaluation sets: WER (and speed) only, no reference word timings.
ESB_REPO, ESB_REVISION = "hf-audio/esb-datasets-test-only-sorted", "de79e57adf9983993bcc72586d7313460825f5eb"
FLEURS_REPO, FLEURS_REVISION = "google/fleurs", "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
EVAL_SETS = {
    # name: (repo, revision, parquet prefix, id column, text column, language)
    "ami": (ESB_REPO, ESB_REVISION, "ami/test-", "id", "text", "English"),  # meetings, CC-BY-4.0
    "earnings22": (ESB_REPO, ESB_REVISION, "earnings22/test-", "id", "text", "English"),  # CC-BY-SA-4.0
    "fleurs_nl": (
        FLEURS_REPO,
        FLEURS_REVISION,
        "parquet-data/nl_nl/test-",
        "id",
        "raw_transcription",
        "Dutch",
    ),
}


def load_eval_set(name: str, n: int, seed: int = 0) -> tuple[list[Utterance], str]:
    """A fixed random sample of `n` utterances (all if fewer) and the set's language. Utterances whose
    reference is empty are skipped before sampling."""
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import HfApi, hf_hub_download

    repo, rev, prefix, id_col, text_col, language = EVAL_SETS[name]
    files = sorted(
        f for f in HfApi().list_repo_files(repo, repo_type="dataset", revision=rev)
        if f.startswith(prefix) and f.endswith(".parquet")
    )  # fmt: skip
    tables = [pq.read_table(hf_hub_download(repo, f, repo_type="dataset", revision=rev)) for f in files]
    rows = [(t, i) for t in tables for i in range(t.num_rows) if t.column(text_col)[i].as_py().strip()]
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(rows), size=min(n, len(rows)), replace=False).tolist())
    utts = []
    for k in pick:
        t, i = rows[k]
        audio, sr = sf.read(io.BytesIO(t.column("audio")[i].as_py()["bytes"]), dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            from rumblex.audio import load_audio

            audio = load_audio(audio, sr=sr)
        utts.append(Utterance(str(t.column(id_col)[i].as_py()), audio, t.column(text_col)[i].as_py(), []))
    return utts, language


def wer(refs: Sequence[str], hyps: Sequence[str], norm: Callable[[str], str] = normalize_text) -> dict:
    import jiwer

    r, h = [norm(x) for x in refs], [norm(x) for x in hyps]
    keep = [i for i, x in enumerate(r) if x.strip()]  # jiwer rejects empty references
    o = jiwer.process_words([r[i] for i in keep], [h[i] for i in keep])
    return {
        "wer": round(o.wer, 4),
        "sub": o.substitutions,
        "del": o.deletions,
        "ins": o.insertions,
        "ref_words": o.hits + o.substitutions + o.deletions,
    }


@dataclass
class SystemOutput:
    text: str
    words: list[TimedWord]
    timings: dict[str, float] = field(default_factory=dict)  # seconds per stage


class System:
    """One ASR + word-alignment system under test. `load` is timed separately from `run`."""

    name = "system"

    def load(self) -> None:
        pass

    def run(self, audio: np.ndarray, language: str | None) -> SystemOutput:
        raise NotImplementedError

    def info(self) -> dict:
        return {}


class RumbleXSystem(System):
    name = "rumblex"

    def __init__(
        self,
        model: str = "Reson8/Rumble-2-Flash",
        device: str | None = None,
        compute_type: str | None = None,
        batch_size: int = 8,
        digits: bool = False,
        max_chunk_s: float = 20.0,
    ):
        self.kw = {"model": model, "device": device, "compute_type": compute_type, "batch_size": batch_size}
        self.digits = digits
        self.max_chunk_s = max_chunk_s

    def load(self) -> None:
        from rumblex import load_model
        from rumblex.aligner import Aligner

        self.tr = load_model(**self.kw)
        self.al = Aligner.from_transcriber(self.tr)

    def run(self, audio: np.ndarray, language: str | None) -> SystemOutput:
        t0 = _now()
        res = self.tr.transcribe(
            audio,
            language=language,
            digits=self.digits,
            keep_encoder_states=True,
            max_chunk_s=self.max_chunk_s,
        )
        t1 = _now()
        self.al.align(res, audio)
        t2 = _now()
        words = [TimedWord(w.word, w.start, w.end) for s in res.segments for w in s.words]
        text = " ".join(s.text for s in res.segments)
        return SystemOutput(text, words, {"transcribe": t1 - t0, "align": t2 - t1})

    def info(self) -> dict:
        return {
            **self.kw,
            "dtype": str(self.tr.dtype),
            "digits": self.digits,
            "max_chunk_s": self.max_chunk_s,
        }


class RumbleXVLLMSystem(RumbleXSystem):
    """rumblex with decoding on a vLLM server (`rumblex.vllm_backend`); alignment as in `RumbleXSystem`."""

    name = "rumblex-vllm"

    def __init__(self, base_url: str = "http://localhost:8000/v1", concurrency: int = 32, **kw):
        super().__init__(**kw)
        self.base_url, self.concurrency = base_url, concurrency

    def load(self) -> None:
        from rumblex.aligner import Aligner, AlignOptions, CTCHead
        from rumblex.vllm_backend import VLLMTranscriber

        self.tr = VLLMTranscriber(
            self.base_url,
            self.kw["model"],
            concurrency=self.concurrency,
            batch_size=self.kw["batch_size"],
            device=self.kw["device"],
            dtype=self.kw["compute_type"],
        )
        head = CTCHead.from_pretrained(self.kw["model"], device=self.tr.device)
        self.al = Aligner(head, self.tr.tokenizer, AlignOptions())

    def info(self) -> dict:
        return {
            **super().info(),
            "backend": "vllm",
            "base_url": self.base_url,
            "concurrency": self.concurrency,
        }


class WhisperXSystem(System):
    """WhisperX (faster-whisper + wav2vec2 alignment). Needs `pip install whisperx` (not a rumblex dependency)."""

    name = "whisperx"

    def __init__(
        self,
        model: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        batch_size: int = 16,
    ):
        self.kw = {"model": model, "device": device, "compute_type": compute_type, "batch_size": batch_size}
        self._align: dict = {}

    def load(self) -> None:
        import whisperx

        self.wx = whisperx
        self.model = whisperx.load_model(
            self.kw["model"], self.kw["device"], compute_type=self.kw["compute_type"]
        )

    def run(self, audio: np.ndarray, language: str | None) -> SystemOutput:
        lang = _iso(language)
        t0 = _now()
        res = self.model.transcribe(audio, batch_size=self.kw["batch_size"], language=lang)
        t1 = _now()
        code = res.get("language") or lang or "en"
        if code not in self._align:  # alignment model load is a one-off, kept out of the timing
            self._align[code] = self.wx.load_align_model(language_code=code, device=self.kw["device"])
        t2 = _now()
        model_a, meta = self._align[code]
        aligned = self.wx.align(
            res["segments"], model_a, meta, audio, self.kw["device"], return_char_alignments=False
        )
        t3 = _now()
        words = [
            TimedWord(w["word"], float(w["start"]), float(w["end"]))
            for s in aligned["segments"]
            for w in s.get("words", [])
            if "start" in w and "end" in w  # WhisperX leaves some tokens (e.g. bare numbers) untimed
        ]
        text = " ".join(s["text"].strip() for s in res["segments"])
        return SystemOutput(text, words, {"transcribe": t1 - t0, "align": t3 - t2})

    def info(self) -> dict:
        import importlib.metadata as md

        return {**self.kw, "whisperx": md.version("whisperx")}


_ISO = {"english": "en", "dutch": "nl", "german": "de", "french": "fr", "spanish": "es"}


def _iso(language: str | None) -> str | None:
    if language is None:
        return None
    return _ISO.get(language.lower(), language.lower())


def _now() -> float:
    import time

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass
    return time.perf_counter()


class GpuMemorySampler:
    """Peak memory used on the GPU while running, sampled from nvidia-smi. It counts memory torch doesn't
    see (e.g. CTranslate2's), so both systems are measured the same way. None without nvidia-smi."""

    def __init__(self, interval: float = 0.2, index: int = 0):
        self.interval, self.index, self.peak_mb = interval, index, None

    def _read(self) -> int | None:
        import subprocess

        cmd = ["nvidia-smi", f"--id={self.index}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=True).stdout
            return int(out.strip().splitlines()[0])
        except Exception:  # noqa: BLE001 - no GPU / no nvidia-smi: just no reading
            return None

    def __enter__(self):
        import threading

        self._stop = threading.Event()

        def loop():
            while not self._stop.is_set():
                v = self._read()
                if v is not None:
                    self.peak_mb = max(self.peak_mb or 0, v)
                self._stop.wait(self.interval)

        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def run_accuracy(
    system: System,
    utts: Sequence[Utterance],
    language: str | None,
    norm: Callable[[str], str] = normalize_text,
) -> dict:
    """One run per utterance: WER over all of them, and word timing errors against the references."""
    hyps, score, secs = [], TimingScore(), 0.0
    for u in utts:
        out = system.run(u.audio, language)
        secs += sum(out.timings.values())
        hyps.append(out.text)
        score.add(score_timings(u.words, out.words))
    audio_s = sum(len(u.audio) for u in utts) / SAMPLE_RATE
    return {
        "utterances": len(utts),
        "audio_seconds": round(audio_s, 1),
        **wer([u.transcript for u in utts], hyps, norm),
        "timing": score.summary(),
        "rtfx_per_utterance": round(audio_s / secs, 1) if secs else None,
    }


def run_speed(system: System, audio: np.ndarray, language: str | None) -> dict:
    """One long recording end to end: stage times, real-time factor, peak GPU memory."""
    with GpuMemorySampler() as mem:
        t0 = _now()
        out = system.run(audio, language)
        total = _now() - t0
    dur = len(audio) / SAMPLE_RATE
    return {
        "audio_seconds": round(dur, 1),
        "total_seconds": round(total, 2),
        "rtfx": round(dur / total, 1) if total else None,
        "stages": {k: round(v, 2) for k, v in out.timings.items()},
        "words": len(out.words),
        "peak_gpu_mb": mem.peak_mb,
    }


def long_audio_from_librispeech(utts: Sequence[Utterance], seconds: float, gap: float = 0.5) -> np.ndarray:
    """Real speech for speed runs: utterances back to back with short gaps, cycled up to `seconds`."""
    if not utts:
        return np.zeros(0, np.float32)
    target = int(seconds * SAMPLE_RATE)
    g = np.zeros(int(gap * SAMPLE_RATE), np.float32)
    pieces, total = [], 0
    while total < target:
        for u in utts:
            pieces += [u.audio, g]
            total += len(u.audio) + len(g)
            if total >= target:
                break
    return np.concatenate(pieces)[:target]


def _pct(x) -> str:
    return f"{100 * x:.1f}%" if isinstance(x, (int, float)) else "n/a"


def _unit(x, unit: str) -> str:
    return "n/a" if x is None else f"{x}{unit}"


def markdown_table(results: dict) -> str:
    rows = [
        "| system | WER | start MAE | starts ≤ 100 ms | end MAE | ends ≤ 100 ms | long file speed | peak GPU |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        a, s = r.get("accuracy", {}), r.get("speed", {})
        t = a.get("timing", {})
        rows.append(
            f"| {name} | {_pct(a.get('wer'))} | {_unit(t.get('start_mae_ms'), ' ms')}"
            f" | {_pct(t.get('start_within_100ms'))} | {_unit(t.get('end_mae_ms'), ' ms')}"
            f" | {_pct(t.get('end_within_100ms'))} | {_unit(s.get('rtfx'), 'x')} | {_unit(s.get('peak_gpu_mb'), ' MB')} |"
        )
    return "\n".join(rows)
