# scripts

Research and benchmark scripts. None of them is needed to use rumblex. Each GPU script prints JSON and
writes it with `--out`; the results committed under `docs/` name the script and run that produced them.
Run from the repo root with `src/` (and, for the two that import each other, `scripts/`) importable,
e.g. `PYTHONPATH=src:scripts python scripts/<name>.py --help`.

| script | what it does | committed result |
|---|---|---|
| `inspect_rumble_ctc.py` | First look at the checkpoint and its CTC head: frame rate, vocab mapping, peakiness, reuse of `generate()` states, a rough Viterbi alignment | `docs/model_inspection.md`, `docs/inspection_*.json` |
| `check_transcriber.py` | Batch 8 vs batch 1 must give identical output; captured encoder states vs the checkpoint's reference encoder pass | `docs/transcriber_check_gpu_fp32.json` |
| `check_encoder_runner.py` | The vLLM backend's local encoder gives the same states as the transformers backend | (log only) |
| `check_vad_gpu.py` | Batched Silero VAD on GPU and CPU vs the sequential pass: time and speech/non-speech agreement | `docs/bench_2026-11-20_rtx4090_dutch/vad_gpu.json` |
| `collect_alignment_states.py` | Transcribes LibriSpeech dev/test (and FLEURS Dutch) and saves encoder states, so alignment variants can be scored on CPU | input for the next two |
| `eval_alignment.py` | Scores frame grids, end rules and start shift on the saved states; dev chooses, test reports. Writes `eval_alignment.json` into the `--states` folder | `docs/alignment_eval.md`, `docs/alignment_eval.json` |
| `sweep_energy.py` | Sweeps the loudness refinement of word ends and starts on the saved states | `docs/alignment_energy_sweep.json` |
| `profile_speed.py` | Where the time goes: VAD vs transcription vs alignment over batch size and chunk length | `docs/bench_2026-11-20_rtx4090_tune/` |
| `benchmark.py` | rumblex vs rumblex+vLLM vs WhisperX: WER, word timing error, long-file speed, peak GPU memory | `docs/bench_2026-11-20_rtx4090*/` |
| `diagnose_wer.py` | One eval set through pipeline variants (VAD on/off, language forced/detected), saving every transcript | `docs/bench_2026-11-20_rtx4090_diagnose/` |
| `rescore_leaderboard.py` | Rescores `diagnose_wer.py` transcripts with the Open ASR Leaderboard's normaliser and alignment (CPU only) | `docs/bench_2026-11-20_rtx4090_diagnose/leaderboard_rescore.json` |
| `render_demo_gif.py` | Turns the captured output of a real CLI run and the subtitles it wrote into a terminal GIF | `docs/demo.gif` |
| `render_words_gif.py` | Plays back a rumblex JSON result: words light up at their times, coloured by speaker, over the audio's waveform | `docs/words.gif` |

The `runpod_*.sh` scripts run those benchmarks on a rented Linux GPU in one go; see
`docs/runpod_session.md`.
