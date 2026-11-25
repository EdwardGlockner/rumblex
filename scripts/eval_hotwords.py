"""Do hotwords help on a real list? Earnings-21 clips transcribed with and without Rev's biasing list.

Data (downloaded by you, nothing of it is in this repo):
- Earnings-21 from github.com/revdotcom/speech-datasets (`earnings21/`): the calls' audio (`media/*.mp3`) and
  Rev's biasing list (`bias_lists/oracle_list.txt`, 1,013 words and phrases in capitals, from arXiv
  2209.01250). The transcripts are CC BY-SA 4.0.
- ConEC from github.com/huangruizhe/ConEC (`earnings21/`): the same transcripts with word times
  (`transcripts/timestamps/*.nlp`), which Rev's own files don't have. They are only used to cut clips.

Clips: each call is cut in pauses of at least 0.3 s between two timed words, at a speaker change, at a
sentence end once the clip is 5 s long, or before it would pass 20 s. `--clips-per-call` of them are drawn
per call (seeded), from the calls ConEC has context files for.

`collect` (GPU) transcribes the clips once per setting, forced English, and saves every transcript:

    python scripts/eval_hotwords.py collect --earnings21 D:/data/speech-datasets/earnings21 \\
        --conec D:/data/ConEC/earnings21 --out data/hotwords_e21/run.json

Settings: no hotwords; the list as rumblex takes it (capitals also get a Title Case form); the same but
single words of up to four letters keep only their capitals ("IBM" gets no "Ibm"); and the list with no
Title Case forms at all. Last, `prompt`: no hotwords, the whole list as given, comma-separated, in the
prompt. That is thousands of tokens in front of every clip, so it runs one clip at a time.

`score` (CPU) reports, per setting: WER, recall of the list words that are spoken, false triggers (list
words written where the reference has something else) per 100 clips, and the change against no hotwords
with a paired bootstrap 95% interval over clips. Text goes through Whisper's English normaliser. A list
word is a word of a list entry that is alphabetic, longer than two letters and outside the 5,000 most
frequent English words (wordfreq), so "co" and "zoom" don't count. With `--compare`, the transcripts are
also checked against another run's saved ones.

    python scripts/eval_hotwords.py score --run data/hotwords_e21/run.json \
        --earnings21 D:/data/speech-datasets/earnings21 --out docs/hotwords_eval.json

Needs whisper-normalizer (in the dev extra) and, for `score`, wordfreq.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

MIN_S, MAX_S = 5.0, 20.0
MIN_GAP, PAD = 0.3, 0.2
SHORT = 4  # "acronyms" setting: single words up to this many letters get no Title Case form
SETTINGS = ("none", "hotwords", "hotwords_acronyms", "hotwords_as_given", "prompt")


def read_nlp(path: Path) -> list[dict]:
    rows = [line.rstrip("\n").split("|") for line in path.read_text(encoding="utf-8").splitlines()][1:]
    out = []
    for r in rows:
        ts = float(r[2]) if r[2].strip() else None
        end = float(r[3]) if r[3].strip() else None
        out.append({"tok": r[0], "spk": r[1], "ts": ts, "end": end, "punct": r[4]})
    return out


def segment(rows: list[dict]) -> list[dict]:
    """Clips {start, end, text} cut only between two timed words (about two thirds of the words have
    times). The clip edges sit inside the pause, up to PAD from the words."""
    clips, cur = [], []
    pre = PAD  # padding before the current clip's first word

    def close(post):
        if cur and cur[0]["ts"] is not None and cur[-1]["end"] is not None:
            text = " ".join(r["tok"] + r["punct"] for r in cur)
            start = max(cur[0]["ts"] - pre, 0.0)
            clips.append({"start": round(start, 3), "end": round(cur[-1]["end"] + post, 3), "text": text})

    for i, r in enumerate(rows):
        if cur:
            prev = cur[-1]
            gap = r["ts"] - prev["end"] if prev["end"] is not None and r["ts"] is not None else None
            if gap is not None and gap >= MIN_GAP and cur[0]["ts"] is not None:
                dur = prev["end"] - cur[0]["ts"]
                nxt_known = next((x["end"] for x in rows[i:] if x["end"] is not None), None)
                too_long = nxt_known is not None and nxt_known - cur[0]["ts"] > MAX_S
                if r["spk"] != prev["spk"] or (dur >= MIN_S and prev["punct"] in ".?!") or too_long:
                    pad = min(PAD, gap / 2)
                    close(pad)
                    cur, pre = [], pad
            elif cur[0]["ts"] is None:  # a clip must start on a timed word
                cur, pre = [], PAD
        cur.append(r)
    close(PAD)
    return [c for c in clips if 1.0 <= c["end"] - c["start"] <= 30.0]


def sample_clips(earnings21: Path, conec: Path, seed: int, k: int) -> dict[str, list[dict]]:
    """call id -> its drawn clips, each with an id `call/index`."""
    rng = np.random.default_rng(seed)
    out = {}
    for nlp in sorted((conec / "transcripts" / "timestamps").glob("*.nlp")):
        cid = nlp.stem
        if not (
            (earnings21 / "media" / f"{cid}.mp3").exists() and (conec / "contexts" / f"{cid}.txt").exists()
        ):
            continue
        clips = segment(read_nlp(nlp))
        idx = sorted(rng.choice(len(clips), size=min(k, len(clips)), replace=False).tolist())
        out[cid] = [{**clips[i], "id": f"{cid}/{i}"} for i in idx]
    return out


def read_list(earnings21: Path) -> list[str]:
    lines = (earnings21 / "bias_lists" / "oracle_list.txt").read_text(encoding="utf-8").splitlines()
    return list(dict.fromkeys(ln.strip() for ln in lines if ln.strip()))


def entries_for(setting: str, phrases: list[str]):
    """(entries, title_caps) to build the setting's Hotwords from."""
    from rumblex.hotwords import Entry, title_case

    if setting == "hotwords":
        return phrases, True
    if setting == "hotwords_as_given":
        return phrases, False
    if setting == "hotwords_acronyms":
        acronym = lambda p: len(p.split()) == 1 and len(p) <= SHORT
        return [
            Entry(p, (title_case(p),)) if p.isupper() and not acronym(p) else Entry(p) for p in phrases
        ], False
    raise ValueError(setting)


def collect(args) -> None:
    import torch

    from rumblex.audio import SAMPLE_RATE, load_audio
    from rumblex.hotwords import Hotwords
    from rumblex.transcriber import RumbleTranscriber
    from rumblex.vad import SpeechChunk, SpeechRegion

    class Timed(Hotwords):
        """Hotwords that add up the host time spent in their logits processor."""

        seconds = 0.0

        def processor(self):
            inner = super().processor()

            def call(input_ids, scores):
                t = time.perf_counter()
                try:
                    return inner(input_ids, scores)
                finally:
                    self.seconds += time.perf_counter() - t

            return call

    picked = sample_clips(args.earnings21, args.conec, args.seed, args.clips_per_call)
    if args.calls:
        picked = dict(list(picked.items())[: args.calls])
    phrases = read_list(args.earnings21)
    out = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {"settings": {}}
    n_clips = sum(len(v) for v in picked.values())
    audio_s = round(sum(c["end"] - c["start"] for v in picked.values() for c in v), 1)
    tr = RumbleTranscriber(device=args.device, dtype=args.dtype, batch_size=args.batch_size)
    out.update(
        device=torch.cuda.get_device_name(tr.device) if tr.device.type == "cuda" else "cpu",
        dtype=str(tr.dtype).removeprefix("torch."), batch_size=args.batch_size, seed=args.seed,
        clips_per_call=args.clips_per_call, calls=len(picked), clips=n_clips, audio_s=audio_s,
        list_entries=len(phrases), language="English",
    )  # fmt: skip
    print(f"{len(picked)} calls, {n_clips} clips, {audio_s / 3600:.2f} h, list of {len(phrases)}", flush=True)

    pieces = {}
    for cid, clips in picked.items():
        full = load_audio(args.earnings21 / "media" / f"{cid}.mp3")
        pieces[cid] = [full[int(c["start"] * SAMPLE_RATE) : int(c["end"] * SAMPLE_RATE)] for c in clips]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for name in args.only or SETTINGS:
        if name in out["settings"]:
            print("have", name, flush=True)
            continue
        hw, build_s, prompt = None, 0.0, None
        if name == "prompt":
            prompt = ", ".join(phrases)
            tr.batch_size = 1
        elif name != "none":
            entries, title_caps = entries_for(name, phrases)
            t0 = time.perf_counter()
            hw = Timed(entries, tr.tokenizer, title_caps=title_caps)
            build_s = time.perf_counter() - t0
        utts, wall = [], 0.0
        for cid, clips in picked.items():
            audio = np.concatenate(pieces[cid])
            chunks, pos = [], 0
            for p in pieces[cid]:
                chunks.append(SpeechChunk(pos, pos + len(p), (SpeechRegion(pos, pos + len(p)),)))
                pos += len(p)
            t0 = time.perf_counter()
            items = tr.transcribe_chunks(audio, chunks, language="English", hotwords=hw, prompt=prompt)
            wall += time.perf_counter() - t0
            utts += [{"id": c["id"], "ref": c["text"], "hyp": it.text} for c, it in zip(clips, items)]
        out["settings"][name] = {
            "decode_s": round(wall, 1), "rtf": round(wall / audio_s, 4),
            "processor_ms_per_clip": round(1000 * hw.seconds / n_clips, 1) if hw else 0.0,
            "list_build_s": round(build_s, 2), "token_paths": sum(map(len, hw.trie.paths)) if hw else 0,
            "utts": utts,
        }  # fmt: skip
        if prompt:
            out["settings"][name].update(
                batch_size=1, prompt_tokens=len(tr.tokenizer(prompt, add_special_tokens=False).input_ids)
            )
            tr.batch_size = args.batch_size
        s = out["settings"][name]
        print(f"{name}: decode {s['decode_s']} s, rtf {s['rtf']}, processor {s['processor_ms_per_clip']} ms/clip, "
              f"list built in {s['list_build_s']} s", flush=True)  # fmt: skip
        args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


def align_words(ref: list[str], hyp: list[str]) -> list[tuple[str | None, str | None]]:
    """Levenshtein word alignment: (ref word or None, hyp word or None) pairs, in order."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]), d[i - 1][j] + 1, d[i][j - 1] + 1)
    out = []
    i, j = n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            out.append((ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            out.append((ref[i - 1], None))
            i -= 1
        else:
            out.append((None, hyp[j - 1]))
            j -= 1
    return out[::-1]


def clip_counts(ref: str, hyp: str, listed: set[str]) -> tuple[int, int, int, int, int]:
    """(errors, ref words, list words written right, list words in the ref, false triggers) for one clip."""
    errors = ref_words = hits = in_ref = false_triggers = 0
    for rw, hw in align_words(ref.split(), hyp.split()):
        err = rw != hw
        errors += err
        if rw is not None:
            ref_words += 1
            if rw in listed:
                in_ref += 1
                hits += not err
        if hw is not None and hw in listed and err:
            false_triggers += 1
    return errors, ref_words, hits, in_ref, false_triggers


def bootstrap(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0) -> dict:
    """95% intervals of (b - a) over resampled clips; rows are `clip_counts`."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    sa, sb = a[idx].sum(1), b[idx].sum(1)
    wer = 100 * (sb[:, 0] / sb[:, 1] - sa[:, 0] / sa[:, 1])
    rec = 100 * (sb[:, 2] / np.maximum(sb[:, 3], 1) - sa[:, 2] / np.maximum(sa[:, 3], 1))
    ft = 100 * (sb[:, 4] - sa[:, 4]) / len(a)
    return {k: [round(float(x), 2) for x in np.percentile(v, [2.5, 97.5])]
            for k, v in (("wer", wer), ("list_recall", rec), ("false_triggers_per_100", ft))}  # fmt: skip


def totals(c: np.ndarray) -> dict:
    s = c.sum(0)
    return {"wer": round(100 * s[0] / s[1], 2), "list_recall": round(100 * s[2] / s[3], 1),
            "false_triggers_per_100": round(100 * s[4] / len(c), 1)}  # fmt: skip


def score(args) -> None:
    from whisper_normalizer.english import EnglishTextNormalizer
    from wordfreq import top_n_list

    norm = EnglishTextNormalizer()
    run = json.loads(args.run.read_text(encoding="utf-8"))
    top5k = set(top_n_list("en", 5000))
    words = {x for ph in read_list(args.earnings21) for x in norm(ph).split()}
    listed = {w for w in words if w.isalpha() and len(w) > 2 and w not in top5k}

    hyps, counts = {}, {}
    for name, s in run["settings"].items():
        hyps[name] = [norm(u["hyp"]) for u in s["utts"]]
        counts[name] = np.array(
            [clip_counts(norm(u["ref"]), h, listed) for u, h in zip(s["utts"], hyps[name])], dtype=np.float64
        )
    base = counts["none"]
    out = {k: v for k, v in run.items() if k != "settings"}
    out.update(
        script="scripts/eval_hotwords.py", normaliser="whisper-english", list_words=len(listed),
        ref_words=int(base[:, 1].sum()), list_words_spoken=int(base[:, 3].sum()), bootstrap_resamples=2000,
        settings={},
    )  # fmt: skip
    for name, s in run["settings"].items():
        r = {**totals(counts[name]), **{k: v for k, v in s.items() if k != "utts"}}
        if name != "none":
            t, b = totals(counts[name]), totals(base)
            r["delta"] = {k: round(t[k] - b[k], 2) for k in t}
            r["delta_ci95"] = bootstrap(base, counts[name])
            r["clips_changed"] = sum(a != b for a, b in zip(hyps["none"], hyps[name]))
        out["settings"][name] = r
        d = ""
        if name != "none":
            ci = r["delta_ci95"]
            d = (f" | dWER {r['delta']['wer']:+.2f} {ci['wer']} | drecall {r['delta']['list_recall']:+.1f} "
                 f"{ci['list_recall']} | {r['clips_changed']} clips changed")  # fmt: skip
        print(f"{name:18s} WER {r['wer']:.2f} | list recall {r['list_recall']:.1f} | FT/100 "
              f"{r['false_triggers_per_100']:.1f} | rtf {r['rtf']} | processor {r['processor_ms_per_clip']} ms{d}",
              flush=True)  # fmt: skip

    if (
        args.compare
    ):  # another run of the same clips: {"configs": {name: {"utts": [{"id", "hyp"}]}}}, normalised
        other = json.loads(args.compare.read_text(encoding="utf-8"))["configs"]
        out["compare"] = {}
        for pair in args.pairs:
            mine, theirs = pair.split("=")
            ids = [u["id"] for u in run["settings"][mine]["utts"]]
            ref = {u["id"]: u["hyp"] for u in other[theirs]["utts"]}
            diff = [i for i, h in zip(ids, hyps[mine]) if ref.get(i) != h]
            out["compare"][pair] = {"clips": len(ids), "missing": sum(i not in ref for i in ids),
                                    "different": len(diff)}  # fmt: skip
            print(f"compare {pair}: {len(diff)} of {len(ids)} clips differ", " ".join(diff[:20]), flush=True)
    if args.out:
        args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--earnings21", type=Path, required=True)
    c.add_argument("--conec", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--clips-per-call", type=int, default=20)
    c.add_argument("--calls", type=int, default=0, help="first N calls only (smoke test)")
    c.add_argument("--only", nargs="+", choices=SETTINGS)
    c.add_argument("--device", default=None)
    c.add_argument("--dtype", default=None)
    c.add_argument("--batch-size", type=int, default=4)
    c.set_defaults(fn=collect)
    s = sub.add_parser("score")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--earnings21", type=Path, required=True)
    s.add_argument("--out", type=Path)
    s.add_argument("--compare", type=Path)
    s.add_argument("--pairs", nargs="+", default=[], help="mine=theirs setting names for --compare")
    s.set_defaults(fn=score)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
