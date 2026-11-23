# Changelog

## Unreleased

- Word times: starts move to a loudness dip within 20 ms of the CTC start, and the last word of a chunk can
  end inside its last CTC frame. On LibriSpeech test_clean (CPU) start error went from 52.2 to 49.5 ms and
  end error from 57.2 to 53.7 ms (`docs/alignment_eval.md`, `scripts/sweep_word_ends.py`). Needs the audio,
  like the existing end refinement; `AlignOptions(dip_window=0, last_inside=False)` gives the old times.

## 0.4.0 (2026-11-23)

- `rumblex serve`: an OpenAI-compatible transcription server (`POST /v1/audio/transcriptions`) with the
  `json`, `text`, `srt`, `vtt`, `verbose_json` and (with `--diarize`) `diarized_json` formats. Words in
  `verbose_json` carry `probability` and `score`. Optional `--api_key`; needs the `server` extra. Tested
  with the official `openai` client.

## 0.3.0 (2026-11-23)

- Words have a `probability`: the decoder's, exp of the mean log-prob of the word's tokens, taken during
  decoding on both backends (vLLM is asked for `logprobs`; on 50 AMI/Earnings-22 utterances it returned
  them for every chunk, matching the transformers backend to bf16 rounding,
  `docs/bench_2026-11-23_a40_logprobs`). It separates wrong words from right ones best:
  AUC 0.882 on AMI and 0.837 on Earnings-22 (test halves, `docs/confidence_eval.md`). Empty (`null`) with
  `--text` and on words that got proportional times.
- Word `score` is now the lowest over the word's tokens of each token's peak CTC probability, instead of
  the mean of mean probabilities. It separates wrong words from right ones better: AUC 0.856 vs 0.825 on
  AMI and 0.829 vs 0.795 on Earnings-22 (test halves, `docs/confidence_eval.md`,
  `scripts/eval_confidence.py`). Word times are unchanged, but `score` values are not comparable with
  0.2.0's: correct words can score near 0, so use it to rank words, or use `probability`.

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
