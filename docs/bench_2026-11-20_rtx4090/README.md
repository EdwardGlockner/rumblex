# rumblex vs WhisperX, RTX 4090, 2026-11-20

Raw output of `scripts/runpod_bench.sh` at commit `817f678`, run on a rented Runpod pod (RTX 4090 24 GB,
Secure Cloud EU-RO-1, driver 595.112.07). Files here: one JSON per system (settings, environment, all
metrics), `table.md` (the script's report), `envs.txt` (torch/CUDA per venv, WhisperX 3.8.6, vLLM 0.34.0)
and `session.log`. The session log covers 7 minutes from environment setup to the last result.

Reproduce: on a pod set up as in [runpod_session.md](../runpod_session.md), `bash scripts/runpod_bench.sh`.
The code has changed since this run (see the provenance note in [docs/README.md](../README.md)).

Superseded for word ends: the loudness refinement added after this run brings the end MAE from 75 to
57 ms ([tune run](../bench_2026-11-20_rtx4090_tune/README.md), [alignment_eval.md](../alignment_eval.md)).

## Setup

- Accuracy: 200 LibriSpeech `test_clean` utterances (seed 0, 4,099 reference words, 4,092 of them with
  MFA timings), one run per utterance, `language="English"`. WER uses Whisper's English normalizer
  (`whisper-normalizer`) for every system. Word timings are compared with the MFA alignments
  (`gilkeyio/librispeech-alignments` @ `0daa1eb`) on the words each system got right.
- Speed: one 3,600 s file made from LibriSpeech `test_clean` utterances back to back (0.5 s gaps),
  processed end to end. Peak GPU memory comes from `nvidia-smi` (whole device).
- Systems, as configured:
  - rumblex: transformers backend, bf16, batch 8, `digits=False`.
  - rumblex-vllm: vLLM 0.34.0 server (`--gpu-memory-utilization 0.5`, 32 concurrent requests); encoder
    states and alignment run locally.
  - WhisperX: 3.8.6, `large-v3`, float16, batch 16, its default wav2vec2 English aligner.

  None of them was tuned for throughput.

## Results

| | rumblex | rumblex + vLLM | WhisperX large-v3 |
|---|---|---|---|
| WER | 2.12% | 2.10% | 1.61% |
| words timed (of 4,092) | 3,989 | 3,989 | 3,984 |
| start: MAE / median / mean signed | 57.0 / 42 / +22.3 ms | 57.1 / 42 / +22.3 ms | 74.8 / 67 / +70.7 ms |
| starts ≤ 50 ms / ≤ 100 ms | 57.9% / 85.4% | 57.9% / 85.4% | 26.3% / 82.5% |
| end: MAE / median / mean signed | 75.3 / 46 / +45.8 ms | 75.4 / 46 / +45.8 ms | 33.9 / 22 / −9.6 ms |
| ends ≤ 50 ms / ≤ 100 ms | 53.4% / 80.1% | 53.3% / 80.0% | 82.3% / 94.1% |
| 1 h file: total (transcribe + align) | 37.8 s (37.0 + 0.8) | 28.8 s (28.1 + 0.8) | 38.1 s (21.8 + 16.4) |
| 1 h file: × real time | 95.4× | 124.8× | 94.4× |
| peak GPU memory | 4,778 MB | 17,235 MB* | 11,324 MB |
| model load | 13.8 s | 4.9 s | 12.5 s |

\* Includes the ~12 GB that vLLM reserves up front at `--gpu-memory-utilization 0.5`. It is not what
the model needs.

## Reading it

- On clean read English, WhisperX large-v3 has the lower WER (1.6% vs 2.1%).
- Word starts are closer to the reference with rumblex. WhisperX starts land about 70 ms late on average.
- Word ends are much closer with WhisperX: its wav2vec2 aligner works at 20 ms, while the Rumble encoder
  frames are 80 ms apart inside each 1 s block (13 per second). rumblex's end rule (end = next
  word's start) is the weak part.
- Alignment costs 0.8 s per hour for rumblex against 16.4 s for WhisperX, because rumblex reuses the
  encoder pass.
- End-to-end speed on the transformers backend is the same as WhisperX here. vLLM gave 1.3×. Batch sizes
  and vLLM concurrency were not tuned, so these are not ceilings for any of the three.

## Not covered

One GPU type; clean read speech only (no noisy, conversational or non-English audio); a forced aligner
as reference, not human labels; WhisperX's default English align model only.
