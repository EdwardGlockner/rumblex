# Changelog

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
