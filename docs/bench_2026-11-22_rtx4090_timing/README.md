# LibriSpeech word timings after the blank penalty, RTX 4090, 2026-11-22

Raw output of `scripts/runpod_timing.sh` at commit `5ae0e4d` (v0.1.0 plus the blank penalty before Viterbi,
see [alignment_eval.md](../alignment_eval.md#blank-penalty-added-2026-11-21)), on a rented Runpod pod
(RTX 4090 24 GB, Secure Cloud, driver 595.112.07, 64 vCPUs). Same 200 LibriSpeech test_clean utterances and
settings as the [release run](../bench_2026-11-20_rtx4090_release/README.md): rumblex in bf16 at batch 32,
WhisperX 3.8.6 with `large-v3` in float16 at batch 16. Word times are scored against the Montreal Forced
Aligner reference alignments.

| | rumblex 0.1.0 (release run) | rumblex, blank penalty (this run) | WhisperX (this run) |
|---|---|---|---|
| WER | 2.2% | 2.2% | 1.6% |
| word start MAE | 57.6 ms | 52.6 ms | 74.8 ms |
| starts ≤ 100 ms | 85.3% | 88.4% | 82.5% |
| word end MAE | 57.3 ms | 57.7 ms | 33.9 ms |
| ends ≤ 100 ms | 85.9% | 85.4% | 94.1% |

WhisperX's WER and timings match the release run exactly, so the change in rumblex's starts is the
aligner, not the pod. Speeds are lower than in the release run for both systems (rumblex 357× vs 467×,
WhisperX 80× vs 92×), on a different host; the README keeps the release run's speeds.

Package versions: `envs.txt`. Session log: `session.log`.
