# rumblex 0.1.0 vs WhisperX, RTX 4090, 2026-11-20

The comparison in the top-level README. Raw output of `scripts/runpod_release.sh` on the code released as
v0.1.0, on a rented Runpod pod: RTX 4090 24 GB, Secure Cloud, driver 595.112.07 (`nproc.txt` shows 32
threads). `commit.txt` names the pre-release commit it ran at, which is not in the public history; its
`src/` is identical to v0.1.0. The session log covers 18 minutes from environment setup to the last
result.

Reproduce: on a pod set up as in [runpod_session.md](../runpod_session.md), `bash scripts/runpod_release.sh`.

## Setup

- rumblex: transformers backend, bf16, batch 32, max chunk 20 s, word ends refined on the audio, VAD on
  the GPU. `digits=False` for the English sets, whose references spell numbers out; `digits=True` (the
  model's default) for FLEURS Dutch, whose references mostly use digits.
- rumblex + vLLM: vLLM 0.34.0 server (`--gpu-memory-utilization 0.5`, 32 concurrent requests), local
  encoder batch 8. LibriSpeech only.
- WhisperX: 3.8.6, `large-v3`, float16, batch 16, default alignment models (English and Dutch wav2vec2).
- Library versions per environment: `envs.txt`.
- The language is given to both systems.
- Data, the same as in [bench_2026-11-20_rtx4090_eval](../bench_2026-11-20_rtx4090_eval/README.md):
  - LibriSpeech `test_clean`: 200 utterances, seed 0, with MFA word alignments.
  - AMI (meetings, CC-BY-4.0) and Earnings-22 (earnings calls, CC-BY-SA-4.0): 500 random utterances each
    (seed 0) from the Open ASR Leaderboard's test sets (`hf-audio/esb-datasets-test-only-sorted`).
  - FLEURS Dutch test (CC-BY-4.0): all 364 utterances (`google/fleurs`).
- WER: Whisper's English normalizer for the English sets, Whisper's basic normalizer for Dutch
  (`whisper-normalizer`).
- Speed: one long file per dataset built from that dataset's own audio (LibriSpeech 1 h, AMI and
  Earnings-22 30 min, FLEURS Dutch 20 min), timed from decoded audio to aligned words with the models
  already loaded. Peak GPU memory is nvidia-smi's reading for the whole device during that pass.

## Results

| dataset | rumblex WER | WhisperX WER | rumblex speed | WhisperX speed | speed-up |
|---|---|---|---|---|---|
| AMI (500 utt.) | 10.94% | 17.22% | 329.0× | 104.4× | 3.2× |
| Earnings-22 (500 utt.) | 9.28% | 10.57% | 430.3× | 102.0× | 4.2× |
| LibriSpeech test-clean (200 utt.) | 2.22% | 1.61% | 466.9× (601.0× with vLLM) | 92.4× | 5.1× |
| FLEURS Dutch (364 utt.) | 9.18% | 7.05% | 379.9× | 78.7× | 4.8× |

Time per stage on the long files, in seconds (rumblex's transcribe stage includes the VAD):

| dataset | rumblex transcribe | rumblex align | WhisperX transcribe | WhisperX align |
|---|---|---|---|---|
| AMI | 5.04 | 0.43 | 9.82 | 7.43 |
| Earnings-22 | 3.76 | 0.42 | 9.61 | 8.03 |
| LibriSpeech | 6.84 | 0.86 | 22.32 | 16.66 |
| FLEURS Dutch | 2.90 | 0.26 | 8.42 | 6.84 |

Word timings on LibriSpeech against MFA (mean absolute error): rumblex 57.6 ms for starts and 57.3 ms for
ends, WhisperX 74.8 ms and 33.9 ms. The vLLM run gives the same WER and timings as the transformers run.

Peak GPU memory: rumblex 8,216–9,956 MiB at batch 32, WhisperX 10,088–12,422 MiB at batch 16, rumblex
with vLLM 18,753 MiB (the server reserves half the card).

## Compared with the earlier run

WhisperX's WER is identical to [bench_2026-11-20_rtx4090_eval](../bench_2026-11-20_rtx4090_eval/README.md),
so the data and scoring are the same. rumblex's WER changed:

| dataset | earlier | now | deletions earlier → now |
|---|---|---|---|
| AMI | 14.96% | 10.94% | 333 → 177 |
| Earnings-22 | 10.02% | 9.28% | 294 → 231 |
| LibriSpeech | 2.12% | 2.22% | 10 → 11 |
| FLEURS Dutch | 9.50% (digits on, [dutch run](../bench_2026-11-20_rtx4090_dutch/README.md)) | 9.18% | |

Since the earlier runs, audio of up to 20 s goes to the model whole instead of through the VAD. Most of
the AMI and Earnings-22 gain is fewer deleted words, most likely short or quiet stretches that the VAD
had cut. Both systems also ran faster on this pod than on the four-dataset pod (WhisperX 79–104× here
against 48–64× there), so speeds are only comparable within one pod.
