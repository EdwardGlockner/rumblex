# Changelog

## Unreleased

- Transcribed words have `ctc_match`: false when the CTC head's greedy text for the chunk doesn't
  contain the word, compared as casefolded letters and digits. On AMI and Earnings-22 test words,
  unmatched words were wrong 21.6% and 30.7% of the time and matched ones 1.8% and 3.9%; added to
  `probability` it raises AUC for wrong words from 0.882 to 0.899 and from 0.837 to 0.848
  (`docs/word_trust_eval.md`, `scripts/eval_word_trust.py`). It is in the JSON output and in the
  server's `verbose_json`, and empty for `--text` and for words with proportional times. It costs one
  more pass of the CTC head per chunk; `AlignOptions(ctc_match=False)` turns it off.

## 0.6.0 (2026-11-26)

- Word starts and ends are corrected by a small learned head on the encoder states around each word
  (43 KB, shipped in the package). On LibriSpeech test_clean, start error went from 49.5 to 20.3 ms and
  end error from 53.7 to 31.1 ms against MFA word times (`docs/boundary_head.md`,
  `scripts/boundary_head.py`). It replaces the start shift and the loudness rules, so word times no longer
  depend on the audio being passed to the aligner. Against Buckeye's human labels (FA-Bench, test split,
  `rumblex.align` on the reference words) start error went from 42.1 to 25.1 ms and end error from 41.1
  to 29.6 ms (`docs/fabench_eval.md`). Fitted on read English only; not measured in other languages.
  `--no_boundary_head` or `AlignOptions(boundary_head=False)` gives the old times.
- Hotwords: `--hotwords "Stifel, Paul Betz"`, `--hotwords_file terms.txt` and `hotwords=` in Python. The
  list becomes a prefix tree over the decoder's tokens, and tokens that start or continue an entry get a
  bonus of 2 on their logit during greedy decoding. An entry in capitals also gets a Title Case form. On
  857 Earnings-21 clips with Rev's 1,013-entry list, recall of the spoken list words went from 67.9% to
  76.9% (+9.0 [5.7, 12.5] points) and WER from 5.84% to 6.02% (+0.18 [0.06, 0.31])
  (`docs/hotwords.md`, `scripts/eval_hotwords.py`). Transformers backend only: `--vllm_url` with hotwords
  is an error, and `rumblex serve` doesn't take them yet.
- A word's `probability` is now the log-prob of the token that was written, taken before any hotword
  bonus. Without hotwords the values are the same as before.
- `--prompt` is described as free-text context. It still works as before.

## 0.5.0 (2026-11-24)

- `rumblex.align` / `--text`: word ends on text without punctuation are no longer early. The CTC path may
  hold a punctuation mark after a word that has none, as it does for the model's own text; the text is not
  changed. On LibriSpeech test_clean, end error on the unpunctuated reference transcript went from 67.0 to
  51.4 ms, with starts unchanged (`docs/align_text_eval.md`, `scripts/sweep_text_punct.py`). Against
  Buckeye's human labels (FA-Bench, test split) it went from 56.5 to 41.1 ms (`docs/fabench_eval.md`).
  `optional_punct=False` or `AlignOptions(punct_penalty=...)` control it.
- Docs: giving the model the previous chunk's text as `prompt` was tried on TED-LIUM long-form and made
  WER worse (2.55% to 5.31% on the dev split), so chunks are still decoded without context
  (`docs/chunk_context_eval.md`, `scripts/eval_chunk_context.py`).

## 0.4.1 (2026-11-23)

- Word times: starts move to a loudness dip within 20 ms of the CTC start, and the last word of a chunk can
  end inside its last CTC frame. On LibriSpeech test_clean (CPU) start error went from 52.2 to 49.5 ms and
  end error from 57.2 to 53.7 ms (`docs/alignment_eval.md`, `scripts/sweep_word_ends.py`); on an RTX 4090 in
  bf16, 52.6 to 50.6 ms and 57.7 to 54.6 ms (`docs/bench_2026-11-23_rtx4090_timing`). Needs the audio,
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
