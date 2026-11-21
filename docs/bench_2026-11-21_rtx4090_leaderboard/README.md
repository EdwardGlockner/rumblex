# FLEURS Dutch under the Open ASR Leaderboard's scoring, RTX 4090, 2026-11-21

Raw output of `scripts/runpod_leaderboard.sh` at commit `bde8a3d` (v0.1.0 plus the `--save-transcripts`
option), on a rented Runpod pod (RTX 4090 24 GB, Secure Cloud, driver 595.112.07). Both systems ran all
364 FLEURS Dutch test utterances with the settings of the
[release run](../bench_2026-11-20_rtx4090_release/README.md): rumblex in bf16 at batch 32 with digits on,
WhisperX 3.8.6 with `large-v3` in float16 at batch 16. Each result JSON holds every reference and
transcript (`accuracy.refs`, `accuracy.hyps`). The speed file was kept to 120 s, so the speed numbers in
these files are not comparable with the release run's.

| FLEURS Dutch WER | rumblex | WhisperX large-v3 |
|---|---|---|
| Whisper basic normalizer (as in the README table) | 9.18% | 7.05% |
| Open ASR Leaderboard normaliser and alignment | 6.24% | 4.43% |

The leaderboard scoring is `scripts/rescore_leaderboard.py` with
[huggingface/open_asr_leaderboard](https://github.com/huggingface/open_asr_leaderboard) at `c469619`:
diacritics kept, digits written out in Dutch with num2words on both sides, and a compound split on one
side and joined on the other counted as correct. Output: `leaderboard_rescore.json`.

- Both systems gain about 2.6–2.9 points from the scoring, and WhisperX stays ahead on Dutch.
- rumblex's 6.24% is 0.19 points above the 6.05% on the model card. The references here are google/fleurs;
  the leaderboard scores against its own copy (hf-audio/open-asr-leaderboard-multilingual-datasets),
  which was not checked against them.
- The WhisperX job failed on the first try because the pod image sets `HF_HUB_ENABLE_HF_TRANSFER=1`
  without the `hf_transfer` package; it was rerun on the same pod with that variable set to 0
  (`session.log`). The table and rescoring were then rerun over both results.

Package versions: `envs.txt`.
