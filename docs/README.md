# docs

Every number in the top-level README comes from one of these. Each `bench_*` folder holds the raw JSON
of a run and a README describing it. Most JSON files record the GPU, torch version, git commit and
settings; the READMEs say where one does not.

## Provenance

The comparison in the top-level README is
[bench_2026-11-20_rtx4090_release](bench_2026-11-20_rtx4090_release/README.md), run on the code released
as v0.1.0. Everything else here was measured during development, and that
history was squashed into the first public commit, so the commits named in these files are not in this
repository. Re-running those scripts on v0.1.0 will not reproduce every number exactly:

- Audio of up to 20 s is now transcribed whole instead of going through VAD. Every per-utterance accuracy
  pass is affected: between the four-dataset run and the release run, rumblex's WER went from 14.96% to
  10.94% on AMI and from 10.02% to 9.28% on Earnings-22.
- The earlier speed rows ran with the VAD on the CPU: the sequential Silero pass in the first two runs,
  the batched one in the four-dataset run. The VAD now runs on the GPU when the model is there; it moved
  there before the Dutch re-run.

| | what | produced by |
|---|---|---|
| [model_inspection.md](model_inspection.md) | What is in the Rumble-2-Flash checkpoint: CTC head, frame rate, options, tokenizer, single-clip timings | `scripts/inspect_rumble_ctc.py` |
| [alignment_eval.md](alignment_eval.md) | Word timestamp accuracy on LibriSpeech; how the end rule, frame grid, start shift, loudness refinement, blank penalty and loudness-dip starts were chosen | `scripts/eval_alignment.py`, `scripts/sweep_energy.py`, `scripts/sweep_smoothing.py`, `scripts/sweep_word_ends.py` |
| [align_text_eval.md](align_text_eval.md) | Word timing accuracy of `rumblex.align` on a given transcript, per utterance and as one long file; optional punctuation for text without it | `scripts/eval_align_text.py`, `scripts/sweep_text_punct.py` |
| [boundary_head.md](boundary_head.md) | The learned correction of word starts and ends: what it is, how it was fitted, timing error on LibriSpeech with and without it | `scripts/collect_boundary_train.py`, `scripts/boundary_head.py` |
| [fabench_eval.md](fabench_eval.md) | Word timing accuracy of `rumblex.align` and WhisperX against Buckeye's human word labels, scored by FA-Bench | `scripts/fabench_worker.py`, `scripts/fabench.patch` |
| [confidence_eval.md](confidence_eval.md) | Whether word scores tell right words from wrong ones on AMI and Earnings-22: CTC vs decoder scores, AUC, thresholds | `scripts/collect_confidence_states.py`, `scripts/eval_confidence.py` |
| [hotwords.md](hotwords.md) | How hotwords work and what they change on Earnings-21 clips with Rev's biasing list: WER, recall of list words, false triggers, decode time | `scripts/eval_hotwords.py` |
| [chunk_context_eval.md](chunk_context_eval.md) | Whether giving the model the previous chunk's text as context helps on long recordings (TED-LIUM long-form, dev split): it doesn't | `scripts/eval_chunk_context.py` |
| [bench_2026-11-20_rtx4090](bench_2026-11-20_rtx4090/README.md) | First comparison with WhisperX (LibriSpeech) | `scripts/runpod_bench.sh` |
| [bench_2026-11-20_rtx4090_tune](bench_2026-11-20_rtx4090_tune/README.md) | Batch size, chunk length and vLLM settings; found VAD to be the bottleneck | `scripts/runpod_tune.sh` |
| [bench_2026-11-20_rtx4090_eval](bench_2026-11-20_rtx4090_eval/README.md) | rumblex vs WhisperX on LibriSpeech, AMI, Earnings-22, FLEURS Dutch | `scripts/runpod_eval.sh` |
| [bench_2026-11-20_rtx4090_dutch](bench_2026-11-20_rtx4090_dutch/README.md) | FLEURS Dutch re-run with digits on; batched VAD on GPU vs CPU | `scripts/runpod_dutch.sh` |
| [bench_2026-11-20_rtx4090_diagnose](bench_2026-11-20_rtx4090_diagnose/README.md) | Where the Dutch WER gap comes from: pipeline variants, error analysis, leaderboard rescoring | `scripts/runpod_diagnose.sh`, `scripts/rescore_leaderboard.py` |
| [bench_2026-11-20_rtx4090_release](bench_2026-11-20_rtx4090_release/README.md) | rumblex 0.1.0 vs WhisperX on LibriSpeech, AMI, Earnings-22, FLEURS Dutch: the README's numbers | `scripts/runpod_release.sh` |
| [bench_2026-11-21_rtx4090_leaderboard](bench_2026-11-21_rtx4090_leaderboard/README.md) | FLEURS Dutch for rumblex and WhisperX with transcripts saved, rescored with the Open ASR Leaderboard's scoring | `scripts/runpod_leaderboard.sh` |
| [bench_2026-11-22_rtx4090_timing](bench_2026-11-22_rtx4090_timing/README.md) | LibriSpeech word timings for rumblex with the blank penalty and WhisperX | `scripts/runpod_timing.sh` |
| [bench_2026-11-23_rtx4090_timing](bench_2026-11-23_rtx4090_timing/README.md) | LibriSpeech word timings for rumblex with loudness-dip starts and WhisperX | `scripts/runpod_timing.sh` |
| [bench_2026-11-22_rtx4090_languages](bench_2026-11-22_rtx4090_languages/README.md) | FLEURS in 13 languages, rumblex vs WhisperX: WER under both scorings, speed, CTC alignment coverage | `scripts/runpod_languages.sh` |
| [bench_2026-11-23_a40_logprobs](bench_2026-11-23_a40_logprobs/README.md) | vLLM returns a decoder log-prob for every token, matching the transformers backend's | `scripts/runpod_logprobs.sh` |
| [server.md](server.md) | `rumblex serve`: the OpenAI-compatible endpoint, its formats and flags | |
| [runpod_session.md](runpod_session.md) | How to run the benchmark scripts on a rented GPU | |
| `transcriber_check_gpu_fp32.json` | Batch 8 vs batch 1 give identical output; captured encoder states match the reference | `scripts/check_transcriber.py` |
| `inspection_cpu.json`, `inspection_gpu_fp32.json` | Raw output behind `model_inspection.md` | `scripts/inspect_rumble_ctc.py` |
| `alignment_eval.json` | Raw output behind `alignment_eval.md`: transcription run, WER, every grid / end rule / start shift variant | `scripts/collect_alignment_states.py`, `scripts/eval_alignment.py` |
| `confidence_eval.json` | Raw output behind `confidence_eval.md`, with calibration bins for every score | `scripts/eval_confidence.py` |
| `hotwords_eval.json` | Raw output behind `hotwords.md` (no transcripts) | `scripts/eval_hotwords.py score` |
| `chunk_context_eval.json` | Raw output behind `chunk_context_eval.md` (numbers only, no transcripts) | `scripts/eval_chunk_context.py` |
| `align_text_eval.json` | Raw output behind `align_text_eval.md` | `scripts/eval_align_text.py` |
| `text_punct_sweep.json` | Raw output of the optional punctuation sweep in `align_text_eval.md` | `scripts/sweep_text_punct.py` |
| `boundary_head_sweep.json` | Raw output behind `boundary_head.md`: every setting on dev_clean, the chosen one on test_clean | `scripts/boundary_head.py fit --test` |
| `fabench_eval.csv` | The leaderboard rows FA-Bench wrote for `fabench_eval.md` | `scripts/fabench_worker.py` |
| `alignment_energy_sweep.json` | Raw output of the word-end refinement sweep in `alignment_eval.md` | `scripts/sweep_energy.py` |
| `alignment_shift_sweep.json` | Raw dev_clean output of the shifted-encoding sweep (several encoder passes per chunk); not written up, the gain was small | `scripts/sweep_shifted_frames.py` |
| `alignment_word_ends_sweep.json` | Raw output of the loudness-dip start and last-word end sweep in `alignment_eval.md` | `scripts/sweep_word_ends.py` |
| `alignment_smoothing_sweep.json` | Raw output of the logit temperature and blank penalty sweep in `alignment_eval.md` | `scripts/sweep_smoothing.py` |
| `demo.gif` | The README's terminal demo: the captured output of one CLI run on an RTX 4090 (bf16, commit `da3aea8`), replayed with the waiting shortened | `scripts/render_demo_gif.py` |
| `words.gif`, `words_gif.json` | The README's word-timing animation: 28–42 s of `words_gif.json`, the JSON a CPU run of the 0.1.0 package wrote for a 60 s two-speaker clip with `--diarize`, played back in real time | `scripts/render_words_gif.py` |
