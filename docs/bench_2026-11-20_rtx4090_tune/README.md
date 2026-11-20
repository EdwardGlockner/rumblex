# Throughput tuning, RTX 4090, 2026-11-20

Raw output of `scripts/runpod_tune.sh` at commit `514bd41`, on a rented Runpod pod (RTX 4090 24 GB,
Secure Cloud EU-RO-1, 8 vCPUs allocated; torch ran 16 threads). The session log covers 12 minutes from
environment setup to the last result. The VAD in this run is still the sequential Silero pass on the CPU;
the batched VAD came after (see the end). WhisperX 3.8.6 (in `whisperx.json`), vLLM 0.34.0 (server log,
not committed).

Reproduce: on a pod set up as in [runpod_session.md](../runpod_session.md), `bash scripts/runpod_tune.sh`.
The code has changed since this run (see the provenance note in [docs/README.md](../README.md)).

## 1. Transformers backend: batch size × chunk length (`profile.json`)

The source is 1,213 s of LibriSpeech `test_clean` speech, bf16, encoder states kept, and alignment with
the loudness refinement. WER is computed on the whole file against the joined references (Whisper English
normalizer). VAD (Silero, CPU, 16 torch threads) took 8.6 s for both chunk lengths.

| max chunk | batch | transcribe | align | × real time (model only) | × real time (with VAD) | peak VRAM (torch) | WER |
|---|---|---|---|---|---|---|---|
| 20 s | 8 | 5.51 s | 0.41 s | 205 | 83 | 2.63 GB | 1.43% |
| 20 s | 16 | 3.61 s | 0.33 s | 308 | 96 | 3.87 GB | 1.53% |
| 20 s | 32 | 2.82 s | 0.34 s | 384 | 103 | 6.36 GB | 1.50% |
| 20 s | 64 | 2.41 s | 0.34 s | 442 | 107 | 11.33 GB | 1.53% |
| 30 s | 8 | 5.52 s | 0.36 s | 206 | 83 | 3.25 GB | 1.60% |
| 30 s | 16 | 3.39 s | 0.35 s | 325 | 98 | 5.12 GB | 1.57% |
| 30 s | 32 | 2.80 s | 0.35 s | 385 | 103 | 8.84 GB | 1.60% |
| 30 s | 64 | 2.31 s | 0.34 s | 458 | 107 | 11.87 GB | 1.57% |

- VAD is the bottleneck. The model alone runs at up to 458× real time; with VAD it is 107×.
- In bf16, WER moves by about 0.1–0.2 points between batch sizes. The decoder sees the same chunks, so
  this is numerical: bf16 kernels round differently for different batch shapes. In float32 on a local
  Pascal GPU, batch 1 and batch 8 gave identical output
  ([transcriber_check_gpu_fp32.json](../transcriber_check_gpu_fp32.json)).

## 2. vLLM: client concurrency × local encoder batch × chunk length (`sweep_vllm_*.json`)

Speed only, on a 1,200 s file built by `benchmark.py` (not the 1,213 s file above). All eight settings
landed at 109–116× real time. Each one spent about 10 s in "transcribe", and the sequential VAD
(≈ 8.6 s) is inside that, so the settings could not show.

## 3. Full comparison with the chosen settings (`table.md`, `rumblex*.json`, `whisperx.json`)

| | rumblex (batch 64, 30 s) | rumblex + vLLM | WhisperX large-v3 |
|---|---|---|---|
| WER (200 utt.) | 2.1% | 2.1% | 1.6% |
| start MAE / ≤ 100 ms | 57.1 ms / 85.4% | 57.1 ms / 85.4% | 74.8 ms / 82.5% |
| end MAE / ≤ 100 ms | 57.7 ms / 85.6% | 57.1 ms / 85.7% | 33.9 ms / 94.1% |
| 1 h file, × real time | 108× | 113× | 78.6× |
| peak GPU memory (nvidia-smi) | 20.4 GB | 20.3 GB | 11.4 GB |

- The word ends now include the loudness refinement (end MAE 75 → 57 ms compared with the first run).
- WhisperX ran at 78.6× here against 94.4× in the first session. Pods differ in CPU, and WhisperX has
  CPU-bound stages too. Speed numbers from different pods are not directly comparable.
- Batch 64 uses a lot of memory. With VAD as the bottleneck it buys little. Batch 16 with 20 s chunks gets
  most of the model speed-up at 3.9 GB.

## Follow-up: batched VAD (not part of this pod run)

The VAD now computes Silero's probabilities for 64 parts of the audio at once, each starting 8 s early so
the model state settles (`rumblex.vad.batched_speech_probs`), and feeds them to Silero's own timestamp
logic. Where a long stretch has to be cut at 20 s, the batched and sequential passes can pick
different pauses.

It was measured later on a 4090 pod, on 1,213 s of LibriSpeech with one torch CPU thread
([Dutch re-run](../bench_2026-11-20_rtx4090_dutch/README.md), `vad_gpu.json`): 13.47 s for the
sequential CPU pass, 2.27 s batched on the CPU (same 232 regions, 99.96% of 10 ms frames agree on
speech) and 0.60 s batched on the GPU (231 regions, 99.92%).
