# Changelog

## 0.2.0 (2026-11-23)

- `rumblex.align` and `--text`: word times for a transcript you already have, without decoding. Accuracy
  on LibriSpeech in `docs/align_text_eval.md` (`scripts/eval_align_text.py`).
- Benchmarks: FLEURS in 13 languages for rumblex and WhisperX (`scripts/runpod_languages.sh`,
  `docs/bench_2026-11-22_rtx4090_languages`), with CTC alignment coverage per language. `rumblex.bench`
  has `fleurs_<code>` sets for them.

## 0.1.1 (2026-11-22)

- Word starts: a blank penalty before the Viterbi search (`AlignOptions.blank_penalty`, default 5) with the
  start shift recalibrated to −10 ms. Start error on LibriSpeech test_clean went from 57.0 to 52.2 ms; ends
  are unchanged (`docs/alignment_eval.md`). On an RTX 4090 in bf16: 57.6 to 52.6 ms
  (`docs/bench_2026-11-22_rtx4090_timing`).

## 0.1.0 (2026-11-21)

First release.

- Transcription with Reson8/Rumble-2-Flash: Silero VAD chunking (batched, on the GPU when there is one),
  batched decoding whose output doesn't depend on the batch, and the model's `language`, `prompt`,
  `digits`, `filler` and `verbatim` options.
- Word timestamps from the model's CTC head, reusing the decoder's encoder pass. Word ends are refined
  on the audio's silences. A proportional fallback keeps a long job going if a chunk fails to align.
- Optional speaker labels with pyannote, per word.
- JSON, SRT, VTT (optionally with per-word karaoke tags), TSV and TXT output; subtitle cues split evenly at
  word boundaries.
- Optional vLLM backend for decoding (`--vllm_url`).
- `rumblex` command line tool and a Python API (`load_model`, `transcribe`, `write_result`).
- Benchmarks against WhisperX on LibriSpeech, AMI, Earnings-22 and FLEURS Dutch, with scripts and raw
  results in `docs/`.
