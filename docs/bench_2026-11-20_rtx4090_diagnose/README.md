# FLEURS Dutch WER follow-up, RTX 4090, 2026-11-20

Raw output of `scripts/runpod_diagnose.sh` at commit `546bf10`, on a rented Runpod pod (RTX 4090 24 GB,
Secure Cloud, 12 vCPUs, driver 595.112.07). The session log covers 13 minutes from environment setup to
the last result. Each `diag_*.json` holds the WER summary per variant, the references (`refs`, `ids`) and
every transcript (`variants.*.hyps`). Unlike the `benchmark.py` output, these files do not record the
environment or commit; `commit.txt` and `gpu.txt` do.

Reproduce: on a pod set up as in [runpod_session.md](../runpod_session.md), `bash scripts/runpod_diagnose.sh`.
The code has changed since this run (see the provenance note in [docs/README.md](../README.md)).

All 364 FLEURS Dutch test utterances, `digits=True`, Whisper basic normalizer. Batch 32 for bf16 and 16
for float32 and for vLLM's local encoder; the vLLM 0.34.0 server ran at `--gpu-memory-utilization 0.6`.
Each setup ran four variants:

| WER | VAD + language forced (shipped at the time) | no VAD + forced | VAD + detected | no VAD + detected |
|---|---|---|---|---|
| transformers, bf16 | 9.44% | 9.17% | 9.56% | 9.24% |
| transformers, float32 | 9.50% | 9.19% | 9.50% | 9.19% |
| vLLM 0.34.0 (bf16) | 9.47% | 9.23% | 9.48% | 9.21% |

The shipped bf16 cell (9.44%) differs from the 9.50% in the
[Dutch re-run](../bench_2026-11-20_rtx4090_dutch/README.md) with the same settings. The two runs were at
different commits on different pods. Since then, v0.1.0 passes audio of up to 20 s whole, which is the
"no VAD" column.

- Not the pipeline. Precision (bf16/fp32), backend (transformers/vLLM) and language forcing make no
  real difference. Language detection said Dutch on all 364. VAD chunking costs about 0.3 points against
  passing each utterance whole.
- Numbers. The largest single error source (transformers bf16, no VAD): FLEURS Dutch references write
  some numbers in digits ("2006", "34") and others in words ("twee", "vijf", "drie", "één", "duizend").
  The basic normalizer does not reconcile them. With `digits=True` the model writes "2" where the
  reference has "twee" (20 times for "twee" alone). With `digits=False` it misses the digit references
  instead (10.10% in the first run).
- Real recognition errors (reference → rumblex). Compound words split in two (21 adjacent word pairs
  that join into a word of the reference, e.g. "zagros gebergte", "vice premier"). Names ("angel" →
  "angèle", "zagrosgebergte" → "gebergte"). Word forms ("uitstierven" → "uitsterven", "ingehuldigd" →
  "ingelicht"). Function words inserted, deleted or swapped ("de", "een").
- The Rumble-2-Flash model card reports 6.05% on FLEURS Dutch under the Open ASR Leaderboard's
  setup. Most of the gap is scoring, not the pipeline: see the rescoring below.
- WhisperX large-v3, scored the same way on the same references, measured 7.05%
  ([Dutch re-run](../bench_2026-11-20_rtx4090_dutch/README.md)).

The counts come from `diag_transformers_bf16.json`, variant `whole_forced`: both sides through the
Whisper basic normalizer, aligned word by word with `jiwer.process_words`.

## Rescored with the Open ASR Leaderboard's scoring (2026-11-21)

The same saved transcripts, scored on CPU with the leaderboard's own code
([huggingface/open_asr_leaderboard](https://github.com/huggingface/open_asr_leaderboard) at `c469619`):
its multilingual normaliser (diacritics kept, digits written out in Dutch with num2words on both sides)
and its alignment, which counts a compound split on one side and joined on the other as correct.
Script: `scripts/rescore_leaderboard.py`; output: `leaderboard_rescore.json`.

| WER, leaderboard scoring | VAD + language forced | no VAD + forced | VAD + detected | no VAD + detected |
|---|---|---|---|---|
| transformers, bf16 | 6.25% | 6.18% | 6.39% | 6.20% |
| transformers, float32 | 6.33% | 6.20% | 6.33% | 6.20% |
| vLLM 0.34.0 (bf16) | 6.28% | 6.19% | 6.31% | 6.24% |

The model card's 6.05% is within 0.13–0.34 points of these. The references here are google/fleurs; the
leaderboard scores against its own copy (hf-audio/open-asr-leaderboard-multilingual-datasets), which was
not checked against them. The transcripts come from commit `546bf10`, not the 0.1.0 release run (which
did not save transcripts); v0.1.0 corresponds to the "no VAD + forced" column. WhisperX's transcripts
were not saved, so its 7.05% has not been rescored and the two numbers should not be compared.
