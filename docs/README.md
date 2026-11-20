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
| [alignment_eval.md](alignment_eval.md) | Word timestamp accuracy on LibriSpeech; how the end rule, frame grid, start shift and loudness refinement were chosen | `scripts/eval_alignment.py`, `scripts/sweep_energy.py` |
| [bench_2026-11-20_rtx4090](bench_2026-11-20_rtx4090/README.md) | First comparison with WhisperX (LibriSpeech) | `scripts/runpod_bench.sh` |
| [bench_2026-11-20_rtx4090_tune](bench_2026-11-20_rtx4090_tune/README.md) | Batch size, chunk length and vLLM settings; found VAD to be the bottleneck | `scripts/runpod_tune.sh` |
| [bench_2026-11-20_rtx4090_eval](bench_2026-11-20_rtx4090_eval/README.md) | rumblex vs WhisperX on LibriSpeech, AMI, Earnings-22, FLEURS Dutch | `scripts/runpod_eval.sh` |
| [bench_2026-11-20_rtx4090_dutch](bench_2026-11-20_rtx4090_dutch/README.md) | FLEURS Dutch re-run with digits on; batched VAD on GPU vs CPU | `scripts/runpod_dutch.sh` |
| [bench_2026-11-20_rtx4090_diagnose](bench_2026-11-20_rtx4090_diagnose/README.md) | Where the Dutch WER gap comes from: pipeline variants and error analysis | `scripts/runpod_diagnose.sh` |
| [bench_2026-11-20_rtx4090_release](bench_2026-11-20_rtx4090_release/README.md) | rumblex 0.1.0 vs WhisperX on LibriSpeech, AMI, Earnings-22, FLEURS Dutch: the README's numbers | `scripts/runpod_release.sh` |
| [runpod_session.md](runpod_session.md) | How to run the benchmark scripts on a rented GPU | |
| `transcriber_check_gpu_fp32.json` | Batch 8 vs batch 1 give identical output; captured encoder states match the reference | `scripts/check_transcriber.py` |
| `inspection_cpu.json`, `inspection_gpu_fp32.json` | Raw output behind `model_inspection.md` | `scripts/inspect_rumble_ctc.py` |
| `alignment_eval.json` | Raw output behind `alignment_eval.md`: transcription run, WER, every grid / end rule / start shift variant | `scripts/collect_alignment_states.py`, `scripts/eval_alignment.py` |
| `alignment_energy_sweep.json` | Raw output of the word-end refinement sweep in `alignment_eval.md` | `scripts/sweep_energy.py` |
| `demo.gif` | The README's terminal demo: the captured output of one CLI run on an RTX 4090 (bf16, commit `da3aea8`), replayed with the waiting shortened | `scripts/render_demo_gif.py` |
