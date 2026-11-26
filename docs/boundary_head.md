# Boundary head

A small learned correction for word starts and ends. It is on by default; `--no_boundary_head` or
`AlignOptions(boundary_head=False)` gives the word times of 0.5.0.

Measured 2026-11-26 on CPU, float32, with `scripts/boundary_head.py fit --test`; the raw output is
[boundary_head_sweep.json](boundary_head_sweep.json).

## What it is

The CTC path puts a word on 80 ms frames, and the frame where the model's spike sits is at a fairly steady
but word-dependent distance from where the word really starts. The encoder states around that frame say
more: a linear map from the states of the word's first frame and the three frames on each side predicts
how far the real start is from the frame's start, in milliseconds. A second map does the same for the
word's end, from the last frame and two frames on each side. An end is never placed after the next word's
start.

That is two vectors, 6,272 and 4,480 numbers, 43 KB, shipped inside the package
(`rumblex/boundary_head.safetensors`). It replaces the constant start shift and the loudness rules
(starts moved to a dip in loudness, ends pulled back to silence), so it needs no audio. It costs one small
matrix product per chunk.

## How it was fitted

- Training: 2,500 LibriSpeech utterances from `train_clean_360` and `train_other_500`, transcribed with
  `scripts/collect_boundary_train.py`. The 82,814 words that match the reference are the training rows.
  Targets are the distance from the raw CTC time to the Montreal Forced Aligner time, clipped at 150 ms.
- Model: ridge regression on standardised states.
- Chosen on `dev_clean` (100 utterances, the set used in [alignment_eval.md](alignment_eval.md)): how many
  frames on each side (0 to 3) and the penalty (1 to 30,000 per 1,000 rows), by start error plus end
  error through the whole aligner. Below a penalty of 300 the choice hardly matters: start error stays
  between 17.7 and 18.3 ms with one or more frames on each side.
- `test_clean` (200 utterances) was scored once, after the choice.

## Results

Mean and median absolute error in ms against the MFA word times, and the share of words within 20 ms.

| split | aligner | words | start mean | start median | starts within 20 ms | end mean | end median | ends within 20 ms |
|---|---|---|---|---|---|---|---|---|
| dev_clean | 0.5.0 | 1,915 | 43.0 | 30.0 | 38.3% | 48.0 | 32.0 | 37.0% |
| dev_clean | boundary head | 1,915 | 17.7 | 11.9 | 71.0% | 28.1 | 16.6 | 56.5% |
| test_clean | 0.5.0 | 3,989 | 49.5 | 32.0 | 35.5% | 53.7 | 34.0 | 34.1% |
| test_clean | boundary head | 3,989 | 20.3 | 13.2 | 67.1% | 31.1 | 17.1 | 56.2% |

Adding the loudness dip on top of the corrected starts made them worse on dev (17.7 to 18.6 ms with a
10 ms window), so it is off when the head is on.

## Limits

- The reference is forced-aligner output, not human labels, and the head was fitted to that same kind of
  reference. It may have learned MFA's habits along with the real boundaries.
- Read English only. Nothing here says how it does on conversational speech or other languages. There is
  no Dutch set with word times in this repository.
- Ends are still worse than starts and lean early: the mean signed end error is -23.4 ms on test_clean,
  and the mean (31.1 ms) is far above the median (17.1 ms).
- Fitted on the model's own transcripts. `rumblex.align` with a given text uses the same head; that
  combination was not scored here.
