# rumblex.align and Qwen3-ForcedAligner-0.6B on LibriSpeech, RTX 4090, 2026-11-27

`rumblex.align` (rumblex 0.6.1) next to `Qwen/Qwen3-ForcedAligner-0.6B` (qwen-asr 0.0.6) on a rented
Runpod pod (RTX 4090 24 GB, Secure Cloud, driver 595.112.07, 32 vCPUs). Both get the audio and
LibriSpeech's reference transcript of the same 200 test_clean utterances (seed 0, as in
[align_text_eval.md](../align_text_eval.md)) and are scored against the Montreal Forced Aligner word
times. One utterance where MFA splits a word in two is left out, and 14 words MFA marked `<unk>` are not
scored, which leaves 4,043 words in 199 utterances, the same words for both.

In float32:

| | rumblex.align | Qwen3-ForcedAligner | rumblex minus Qwen (95% interval) |
|---|---|---|---|
| word start MAE | 20.1 ms | 28.6 ms | -8.5 ms (-9.6 to -7.4) |
| word start, median error | 13 ms | 20 ms | |
| starts ≤ 20 ms | 66.9% | 39.1% | |
| starts ≤ 50 ms | 94.0% | 86.6% | |
| starts ≤ 100 ms | 98.1% | 98.1% | |
| word end MAE | 28.9 ms | 26.0 ms | 2.9 ms (1.1 to 4.7) |
| word end, median error | 17 ms | 20 ms | |
| ends ≤ 20 ms | 55.3% | 41.3% | |
| ends ≤ 50 ms | 87.7% | 89.6% | |
| ends ≤ 100 ms | 95.0% | 99.1% | |
| mean signed end error | -22.3 ms | -7.9 ms | |
| words with zero length | 0 | 59 | |
| times real time, one utterance per call | 129× | 353× | |
| peak GPU memory | 2.8 GB | 3.9 GB | |

The intervals are bootstrap intervals over utterances (2,000 draws). In bfloat16 the errors are the same
to within 0.1 ms for both; rumblex ran at 133× with 1.6 GB and Qwen at 389× with 2.0 GB.

What it says:

- rumblex places word starts closer to MFA's: 8.5 ms less mean error, and two thirds of starts within
  20 ms against two fifths.
- Qwen places word ends slightly closer on average (2.9 ms) and has fewer ends far off: 99.1% within
  100 ms against 95.0%. rumblex's ends are early by 22 ms on average. More of rumblex's ends are within
  20 ms.
- Qwen's times come in steps of 80 ms, and it gave 59 words the same start and end, all short words like
  "a" and "the" in the ones looked at.
- Qwen is about 2.7 times faster per utterance here. Both were called one utterance at a time, and the
  time includes reading the file.

Limits:

- One set, read English, 200 utterances; the transcript given is the exact reference. Nothing here about
  other languages, long files, or transcripts with errors.
- The reference is MFA's output, not hand labels. rumblex's boundary head was fitted on MFA times of
  LibriSpeech training utterances ([boundary_head.md](../boundary_head.md)), so this reference favours
  rumblex in a way a hand-labelled set would not.
- The environments differ: rumblex with torch 2.14.1+cu126 and transformers 5.24.1, Qwen with torch 2.15.0
  and transformers 4.57.6, which qwen-asr pins (`envs.txt`).
- Speed is one pass over 200 short files with no batching for either; the first utterance is aligned once
  before timing starts.

Raw output of `scripts/runpod_qwen_aligner.sh` at commit `650846e`: the two score files
(`qwen_aligner_float32.json`, `qwen_aligner_bfloat16.json`), each system's words and seconds per
utterance, `envs.txt` and `session.log`.
