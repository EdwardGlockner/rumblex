# FLEURS Dutch re-run + GPU VAD check, RTX 4090, 2026-11-20

Raw output of `scripts/runpod_dutch.sh` at commit `b00a9da`, on a rented Runpod pod (RTX 4090 24 GB,
Secure Cloud US-NC-1, 16 vCPUs allocated, driver 595.112.07). The session log covers 9 minutes from
environment setup to the last result. WhisperX 3.8.6 (in `whisperx_fleurs_nl.json`).

Reproduce: on a pod set up as in [runpod_session.md](../runpod_session.md), `bash scripts/runpod_dutch.sh`.
The code has changed since this run (see the provenance note in [docs/README.md](../README.md)).

## Dutch, with rumblex writing numbers as digits

The first run ([four-dataset run](../bench_2026-11-20_rtx4090_eval/README.md)) had rumblex writing numbers as words, while FLEURS
references use digits and Whisper's basic normalizer does not convert between the two. This run uses
`--digits` (the model's default). Everything else is as before: all 364 FLEURS Dutch test utterances,
`language="Dutch"`, Whisper basic normalizer, bf16, batch 32. rumblex now uses the batched VAD on the GPU.

| | WER | sub / del / ins (of 8,256 words) | speed, 20 min file |
|---|---|---|---|
| rumblex, digits on | 9.50% | 480 / 70 / 234 | 222× real time |
| rumblex, digits off (first run) | 10.10% | 512 / 66 / 256 | 140× |
| WhisperX large-v3 | 7.05% | 319 / 65 / 198 | 56× |

The 140× is from the earlier pod, where the VAD still ran on the CPU. Compare 222× only with the 56× for
WhisperX, which ran on this pod.

- WhisperX is more accurate on this Dutch set. The gap comes from substitutions and insertions, not
  deletions, so rumblex is not dropping words at chunk edges.
- The Rumble-2-Flash model card reports 6.05% on FLEURS Dutch (vLLM, Open ASR Leaderboard setup). We
  measure 9.50% with rumblex's pipeline. A later run looked into the difference:
  [diagnose](../bench_2026-11-20_rtx4090_diagnose/README.md).

## Batched VAD on GPU (`vad_gpu.json`)

The test audio was 1,213 s of LibriSpeech speech. Only one torch CPU thread was in use in this job, so the
CPU timings are pessimistic.

| VAD | seconds | regions | speech/non-speech agreement with sequential |
|---|---|---|---|
| CPU, sequential (Silero as is) | 13.47 | 232 | |
| CPU, batched | 2.27 | 232 | 99.96% |
| GPU, batched | 0.60 | 231 | 99.92% |
