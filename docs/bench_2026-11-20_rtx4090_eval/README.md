# rumblex vs WhisperX on four datasets, RTX 4090, 2026-11-20

Raw output of `scripts/runpod_eval.sh` at commit `f085f0a`, on a rented Runpod pod (RTX 4090 24 GB,
Secure Cloud, 12 vCPUs allocated, driver 595.112.07; `nproc.txt` shows the host's 96 threads). The session
log covers 38 minutes from environment setup to the last result. This is the first run with the batched
Silero VAD, still on the CPU.

Reproduce: on a pod set up as in [runpod_session.md](../runpod_session.md), `bash scripts/runpod_eval.sh`.
The code has changed since this run (see the provenance note in [docs/README.md](../README.md)).
The README's numbers now come from the [release run](../bench_2026-11-20_rtx4090_release/README.md).

## Setup

- rumblex: transformers backend, bf16, batch 32, max chunk 20 s, `digits=False`, word ends refined on
  the audio.
- rumblex + vLLM: vLLM 0.34.0 server (`--gpu-memory-utilization 0.5`, 32 concurrent requests), local
  encoder batch 8. LibriSpeech only.
- WhisperX: 3.8.6, `large-v3`, float16, batch 16, default alignment models (English and Dutch wav2vec2).
- Data:
  - LibriSpeech `test_clean`: 200 utterances, seed 0, with MFA word alignments.
  - AMI (meetings, CC-BY-4.0) and Earnings-22 (earnings calls, CC-BY-SA-4.0): 500 random utterances each
    (seed 0) from the Open ASR Leaderboard's test sets (`hf-audio/esb-datasets-test-only-sorted` @ `de79e57`).
  - FLEURS Dutch test (CC-BY-4.0): all 364 utterances (`google/fleurs` @ `70bb2e84`).
- WER: Whisper's English normalizer for the English sets and Whisper's basic normalizer for Dutch
  (`whisper-normalizer`). This is the Open ASR Leaderboard's choice.
- Speed: one long file per dataset built from that dataset's own audio (LibriSpeech 1 h, AMI and
  Earnings-22 30 min, FLEURS 20 min), processed end to end. Peak memory comes from `nvidia-smi`.

## Results

| dataset | system | WER | speed (× real time) | peak GPU memory |
|---|---|---|---|---|
| LibriSpeech (read, clean) | rumblex | 2.12% | 169× | 9.3 GB |
| | rumblex + vLLM | 2.10% | 210× | 17.5 GB* |
| | WhisperX | 1.61% | 53× | 11.3 GB |
| AMI (meetings) | rumblex | 14.96% | 158× | 8.2 GB |
| | WhisperX | 17.22% | 59× | 10.0 GB |
| Earnings-22 (earnings calls) | rumblex | 10.02% | 162× | 9.0 GB |
| | WhisperX | 10.57% | 64× | 10.4 GB |
| FLEURS Dutch | rumblex | 10.10% (see below) | 140× | 9.0 GB |
| | WhisperX | 7.05% | 48× | 12.5 GB |

\* Includes the GPU memory vLLM reserves up front.

LibriSpeech word timestamps (MFA reference, mean absolute error):
- rumblex: starts 57.0 ms, ends 57.1 ms.
- WhisperX: starts 74.8 ms, ends 33.9 ms.

The Dutch rumblex WER in this run is not a fair number. rumblex ran with `digits=False` (numbers as words),
because LibriSpeech references spell numbers out. 72 of the 364 FLEURS Dutch references contain numbers
written as digits (106 numbers in all), and Whisper's basic normalizer does not convert between the two
forms. Every number therefore counted against rumblex, while WhisperX writes digits. Re-run with
`--digits` (the model's default) before comparing Dutch. The English sets are not affected: the English
normalizer converts both forms.

## Notes

rumblex was 2.5–3.2× faster than WhisperX end to end on every dataset, and about 4× with vLLM on
LibriSpeech. WhisperX spends about half its time in alignment (14–37 s); rumblex's alignment took
0.5–1.6 s.

WhisperX was more accurate on clean read English. rumblex was more accurate on meetings (15.0% vs
17.2%) and slightly on earnings calls (10.0% vs 10.6%). On Dutch, WhisperX was more accurate after the
re-run, 7.05% vs 9.50% ([Dutch re-run](../bench_2026-11-20_rtx4090_dutch/README.md), with a
follow-up in [diagnose](../bench_2026-11-20_rtx4090_diagnose/README.md)). rumblex has better word starts,
WhisperX better word ends.

WhisperX's speed varies between pods (on the 1 h LibriSpeech file 94×, 79× and 53× on three pods; on
FLEURS Dutch 48× here and 56× on the Dutch re-run pod), so only compare speeds within one pod run.
With a few hundred utterances per set, differences under about 0.5 WER points (the Earnings-22 gap)
could change with another sample.
