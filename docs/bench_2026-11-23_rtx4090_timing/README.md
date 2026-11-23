# LibriSpeech word timings with loudness-dip starts, RTX 4090, 2026-11-23

Raw output of `scripts/runpod_timing.sh` at commit `ceb9657` (0.4.0 plus starts moved to loudness dips and
the last word of a chunk ending inside its frame, see
[alignment_eval.md](../alignment_eval.md#starts-on-loudness-dips-and-the-last-words-end-added-2026-11-23)),
on a rented Runpod pod (RTX 4090 24 GB, Secure Cloud, driver 595.112.07, 128 vCPUs). Same 200 LibriSpeech
test_clean utterances and settings as the [previous timing run](../bench_2026-11-22_rtx4090_timing/README.md):
rumblex in bf16 at batch 32, WhisperX 3.8.6 with `large-v3` in float16 at batch 16. Word times are scored
against the Montreal Forced Aligner reference alignments.

| | rumblex, previous run | rumblex (this run) | WhisperX (this run) |
|---|---|---|---|
| WER | 2.2% | 2.2% | 1.6% |
| word start MAE | 52.6 ms | 50.6 ms | 74.8 ms |
| starts ≤ 100 ms | 88.4% | 87.5% | 82.5% |
| word end MAE | 57.7 ms | 54.6 ms | 33.9 ms |
| ends ≤ 100 ms | 85.4% | 85.9% | 94.1% |

WhisperX's WER and timings match the previous run exactly, so the change in rumblex's times is the aligner,
not the pod. Mean errors are lower; the share of starts within 100 ms is 0.9 points lower. Speeds on this
host were lower again for both systems (rumblex 223×, WhisperX 59×); the README keeps the release run's
speeds.

Package versions: `envs.txt`. Session log: `session.log`.
