"""Worker that runs rumblex inside FA-Bench (github.com/olewave/fa-bench), in rumblex's own interpreter.

FA-Bench starts it once per cell as `fabench_worker.py <jobs.jsonl> <model> <device> <task>` and reads one
JSON object per line from stdout:

    in : {"item_id": ..., "audio_path": ..., "transcript": ...}    (no transcript for task `transcribe`)
    out: {"item_id": ..., "words": [[word, start, end, score], ...]}
         {"item_id": ..., "error": "..."}

Task `align` gives the reference transcript to `rumblex.align` (FA-Bench track 1). Task `transcribe` runs
`rumblex.transcribe` on the audio alone (track 2), in English, with numbers spelled out and fillers kept,
since the references spell numbers out and include "um" and "uh". Everything else is the package default.

The recipe and adapter that call this are in scripts/fabench.patch; docs/fabench_eval.md has the setup.
"""

from __future__ import annotations

import json
import sys
import time


def main(argv: list[str]) -> int:
    jobs_path, model_id = argv[0], argv[1]
    device = argv[2] if len(argv) > 2 else "cuda"
    task = argv[3] if len(argv) > 3 else "align"
    if task not in ("align", "transcribe"):
        raise SystemExit(f"unknown task {task!r}")

    import torch

    import rumblex

    if not (device.startswith("cuda") and torch.cuda.is_available()):
        device = "cpu"
    model = rumblex.load_model(model_id, device=device)
    with open(jobs_path, encoding="utf-8") as f:
        jobs = [json.loads(line) for line in f if line.strip()]
    t0 = time.time()
    for n, job in enumerate(jobs, 1):
        rec = {"item_id": job["item_id"]}
        try:
            if task == "align":
                result = rumblex.align(job["audio_path"], job.get("transcript", ""), model)
            else:
                result = rumblex.transcribe(
                    job["audio_path"], model, language="en", digits=False, filler=True, verbatim=True
                )
            rec["words"] = [[w.word, w.start, w.end, w.score] for s in result.segments for w in s.words]
        except Exception as e:  # noqa: BLE001  one bad item shouldn't lose the batch
            rec["error"] = f"{type(e).__name__}: {e}"
        print(json.dumps(rec), flush=True)
        if n % 100 == 0 or n == len(jobs):
            print(f"[rumblex {task}] {n}/{len(jobs)} items, {time.time() - t0:.0f} s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
