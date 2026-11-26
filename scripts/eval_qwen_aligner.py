"""`rumblex.align` next to Qwen3-ForcedAligner-0.6B: both get the audio and the reference transcript of the
same LibriSpeech test_clean utterances (200, seed 0, as in docs/align_text_eval.md) and are scored against
the MFA word times.

Three steps, because `qwen-asr` needs its own environment:

    python scripts/eval_qwen_aligner.py prepare --out data/qwen_fa
    python scripts/eval_qwen_aligner.py rumblex --dir data/qwen_fa --device cuda:0
    <qwen venv>/python scripts/qwen_aligner_worker.py data/qwen_fa/jobs.jsonl data/qwen_fa/qwen.json
    python scripts/eval_qwen_aligner.py score --dir data/qwen_fa --out docs/qwen_aligner_eval.json

`prepare` writes each utterance as a wav file with a jobs.jsonl holding the transcript and the MFA words.
`rumblex` and the worker each write words as [text, start, end] and the seconds per call, with the file
read included for both. `score` (CPU, seconds) pairs word i of an output with word i of the transcript.
An utterance is scored only if both systems returned exactly the transcript's words, so the two columns
are over the same words; the ones left out are listed with the reason. Reference words MFA left as
"<unk>" are not scored. The difference in start and end error comes with a bootstrap interval over
utterances.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

SYSTEMS = ("rumblex", "qwen")
TOLERANCES_MS = (20, 50, 100)


def score_mask(mfa: list[str], transcript: str) -> list[bool] | None:
    """Per transcript word: score it? None if MFA's words don't line up with the transcript one to one."""
    text = transcript.split()
    if len(mfa) != len(text):
        return None
    if any(m != t and m != "<unk>" for m, t in zip(mfa, text)):
        return None
    return [m != "<unk>" for m in mfa]


def utterance_errors(job: dict, result: dict | None, norm) -> tuple[dict | None, str | None]:
    """Signed start and end errors in ms for the scored words of one utterance, or the reason there are none."""
    if result is None:
        return None, "no output"
    if "error" in result:
        return None, "failed: " + result["error"]
    mask = score_mask([w[0] for w in job["ref_words"]], job["transcript"])
    if mask is None:
        return None, "MFA words differ from the transcript"
    words, text = result["words"], job["transcript"].split()
    if len(words) != len(text):
        return None, f"{len(words)} words returned for {len(text)} in the transcript"
    if any(norm(w[0]) != norm(t) for w, t in zip(words, text)):
        return None, "returned words differ from the transcript"
    if max(w[2] for w in words) > job["duration"] + 1.0:
        raise ValueError(
            f"{job['id']}: a word ends at {max(w[2] for w in words)} in a {job['duration']} s file; not seconds?"
        )
    keep = [i for i, k in enumerate(mask) if k]
    return {
        "start": [1000 * (words[i][1] - job["ref_words"][i][1]) for i in keep],
        "end": [1000 * (words[i][2] - job["ref_words"][i][2]) for i in keep],
        "zero_duration": sum(words[i][2] <= words[i][1] for i in keep),
    }, None


def summary(errs: list[dict]) -> dict:
    out: dict = {"words": sum(len(e["start"]) for e in errs)}
    for name in ("start", "end"):
        signed = np.array([x for e in errs for x in e[name]])
        a = np.abs(signed)
        out[f"{name}_mae_ms"] = round(float(a.mean()), 1)
        out[f"{name}_median_ae_ms"] = round(float(np.median(a)), 1)
        out[f"{name}_mean_signed_ms"] = round(float(signed.mean()), 1)
        for tol in TOLERANCES_MS:
            out[f"{name}s_within_{tol}ms"] = round(float((a <= tol).mean()), 4)
    out["zero_duration_words"] = sum(e["zero_duration"] for e in errs)
    return out


def mae_difference(a: list[dict], b: list[dict], name: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean absolute error of a minus that of b over the same utterances, with a 95% bootstrap interval
    from resampling utterances."""
    sa = np.array([np.abs(e[name]).sum() for e in a])
    sb = np.array([np.abs(e[name]).sum() for e in b])
    n = np.array([len(e[name]) for e in a], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n), size=(n_boot, len(n)))
    boot = (sa[idx].sum(1) - sb[idx].sum(1)) / n[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "difference_ms": round(float((sa.sum() - sb.sum()) / n.sum()), 1),
        "ci95_ms": [round(float(lo), 1), round(float(hi), 1)],
    }


def speed(jobs: list[dict], utts: dict) -> dict:
    done = [j for j in jobs if "seconds" in utts.get(j["id"], {})]
    secs = sum(utts[j["id"]]["seconds"] for j in done)
    audio = sum(j["duration"] for j in done)
    return {
        "utterances_timed": len(done),
        "audio_seconds": round(audio, 1),
        "seconds": round(secs, 1),
        "times_real_time": round(audio / secs, 1) if secs else None,
    }


def score(jobs: list[dict], results: dict[str, dict], norm) -> dict:
    """`results`: system name -> {"meta": ..., "utterances": {id: {"words": ..., "seconds": ...}}}."""
    errs: dict[str, dict] = {s: {} for s in results}
    not_scored = []
    for job in jobs:
        reasons = {}
        for s, res in results.items():
            e, why = utterance_errors(job, res["utterances"].get(job["id"]), norm)
            if e is None:
                reasons[s] = why
            else:
                errs[s][job["id"]] = e
        if reasons:
            not_scored.append({"id": job["id"], **reasons})
    common = [j["id"] for j in jobs if all(j["id"] in errs[s] for s in results)]
    report: dict = {"utterances": len(jobs), "utterances_scored": len(common), "not_scored": not_scored}
    for s, res in results.items():
        report[s] = {
            "meta": res["meta"],
            **summary([errs[s][i] for i in common]),
            "speed": speed(jobs, res["utterances"]),
        }
    names = list(results)
    if len(names) == 2 and common:
        a, b = ([errs[s][i] for i in common] for s in names)
        report[f"{names[0]}_minus_{names[1]}"] = {
            "start_mae": mae_difference(a, b, "start"),
            "end_mae": mae_difference(a, b, "end"),
        }
    return report


def read_jobs(folder: Path) -> list[dict]:
    lines = (folder / "jobs.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def cmd_prepare(args) -> None:
    import soundfile as sf

    from rumblex.audio import SAMPLE_RATE
    from rumblex.bench import iter_librispeech, sample_ids

    audio = args.out / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    with open(args.out / "jobs.jsonl", "w", encoding="utf-8") as f:
        for u in iter_librispeech("test_clean", ids=sample_ids("test_clean", args.n, args.seed)):
            path = (audio / f"{u.id}.wav").resolve()
            sf.write(path, u.audio, SAMPLE_RATE, subtype="PCM_16")
            job = {
                "id": u.id,
                "audio_path": path.as_posix(),
                "transcript": u.transcript,
                "duration": round(len(u.audio) / SAMPLE_RATE, 3),
                "ref_words": [[w.word, w.start, w.end] for w in u.words],
            }
            f.write(json.dumps(job) + "\n")
    print(f"wrote {args.out / 'jobs.jsonl'}")


def cmd_rumblex(args) -> None:
    import torch

    import rumblex

    jobs = read_jobs(args.dir)
    model = rumblex.load_model(device=args.device, compute_type=args.compute_type)
    cuda = model.device.type == "cuda"

    def run(job: dict) -> list[list]:
        res = rumblex.align(job["audio_path"], job["transcript"], model, language="English")
        return [[w.word, w.start, w.end] for seg in res.segments for w in seg.words]

    run(jobs[0])  # warm-up
    if cuda:
        torch.cuda.reset_peak_memory_stats(model.device)
    utts: dict = {}
    for job in jobs:
        if cuda:
            torch.cuda.synchronize(model.device)
        t0 = time.perf_counter()
        words = run(job)
        if cuda:
            torch.cuda.synchronize(model.device)
        utts[job["id"]] = {"words": words, "seconds": round(time.perf_counter() - t0, 4)}
    meta = {
        "system": "rumblex.align",
        "rumblex": rumblex.__version__,
        "device": str(model.device),
        "dtype": str(model.dtype).removeprefix("torch."),
        "gpu": torch.cuda.get_device_name(model.device) if cuda else None,
        "torch": torch.__version__,
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated(model.device) / 2**20) if cuda else None,
    }
    (args.dir / "rumblex.json").write_text(json.dumps({"meta": meta, "utterances": utts}), encoding="utf-8")
    print(json.dumps(meta, indent=2))


def cmd_score(args) -> None:
    from rumblex.bench import normalize_word

    results = {s: json.loads((args.dir / f"{s}.json").read_text(encoding="utf-8")) for s in SYSTEMS}
    report = score(read_jobs(args.dir), results, normalize_word)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "not_scored"}, indent=2))
    print(f"{len(report['not_scored'])} utterances not scored")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_prepare)
    p = sub.add_parser("rumblex")
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--compute-type", default=None)
    p.set_defaults(fn=cmd_rumblex)
    p = sub.add_parser("score")
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(fn=cmd_score)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
