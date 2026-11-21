# Word timestamp accuracy

Measured 2026-11-20 at commit `ea627a1` (see the provenance note in [docs/README.md](README.md)). The
numbers come from two runs, both in [alignment_eval.json](alignment_eval.json), plus the word-end sweep
further down:

- States: `scripts/collect_alignment_states.py` on a GTX 1060 6GB, float32, batch size 8 (`collect` in
  the JSON). It runs the real pipeline (Silero VAD chunks → batched transcription, encoder states kept).
- Scoring: `scripts/eval_alignment.py` on CPU (`dev_clean`, `test_clean`, `variants` in the JSON).

## Setup

- Reference: LibriSpeech word alignments made by the Montreal Forced Aligner
  (`gilkeyio/librispeech-alignments` @ `0daa1eb`, CC-BY-4.0, from Lugosch's zenodo.org/record/2619474).
  These are forced-aligner output, not human labels. In them a word's end is usually the next word's start.
- Data: 100 `dev_clean` and 200 `test_clean` utterances, sampled with seed 0 (`rumblex.bench.sample_ids`).
  That is 682 s and 1,515 s of audio.
- Transcription settings: `language="English"`, `digits=False` (the references spell numbers out).
- Scoring: words are matched by edit distance after lowercasing and stripping punctuation. Only words
  the model got right are timed. Errors are signed hypothesis − reference.
- Selection: settings were picked on `dev_clean` and then reported once on `test_clean`. The start
  shift is the median signed start error on dev (−22 ms).

## Result (0.1.0 defaults: block grid, `next_start` ends, −22 ms start shift)

| split | words timed | start MAE | start median | starts ≤ 50 ms | starts ≤ 100 ms | end MAE | end median | ends ≤ 50 ms | ends ≤ 100 ms |
|---|---|---|---|---|---|---|---|---|---|
| dev_clean | 1,915 | 50.5 ms | 38 ms | 62.9% | 89.3% | 67.6 ms | 42 ms | 57.8% | 83.7% |
| test_clean | 3,989 | 57.0 ms | 42 ms | 57.9% | 85.4% | 75.4 ms | 46 ms | 53.2% | 80.1% |

Viterbi alignment failed on 0 of 306 chunks, so the proportional fallback was never used.

## Alternatives (test_clean, dev start shift)

| end rule | end MAE | end median | ends ≤ 100 ms |
|---|---|---|---|
| `next_start` (end = next word's start) | 75.4 ms | 46 ms | 80.1% |
| `midpoint` (halfway to the next word) | 70.1 ms | 55 ms | 76.5% |
| `region` (next start, capped at the VAD region end) | 83.5 ms | 50 ms | 75.9% |
| `spike` (end of the last CTC spike) | 115.2 ms | 84 ms | 57.3% |

On dev too, `next_start` had the lowest median and the most ends within 100 ms, which is why it is the
default. `midpoint` has a lower mean (66.8 vs 67.6 ms on dev) but a higher median; without the start
shift it had the lowest mean end error on dev (61.8 ms).

The frame grid made no measurable difference: block (`c + 0.08 k`) and uniform (`i / 13`) were within
0.5 ms on start MAE once each was calibrated on dev. Block is kept because it matches the encoder's
conv geometry.

## Word-end refinement on the audio (added later the same day)

With `next_start`, ends are late on average (+33.5 ms on dev, +45.8 ms on test). Looking at the dev
errors by context, the worst ones were words followed by a pause, because "end = next word's start" runs
across the pause. That breakdown was not saved, so only the fix and its sweep are reported here.

The fix is `refine_ends` in `rumblex.aligner`. Between a word's last CTC spike and its `next_start` end,
the first run of at least `min_silence` whose 10 ms loudness is `silence_db` below the chunk's speech
level (95th percentile) ends the word. It only ever moves ends earlier. A matching `refine_starts` (start at
the sound onset after a silence) was tried as well.

Swept on `dev_clean` over `silence_db` ∈ {20, 25, 30, 35} and `min_silence` ∈ {30, 50, 80} ms, for ends only
and for starts + ends. Chosen by start MAE + end MAE on dev: ends only, 25 dB, 30 ms. Moving starts as
well never lowered that sum on dev, so it is off by default. Same states as above; script
`scripts/sweep_energy.py`, output [alignment_energy_sweep.json](alignment_energy_sweep.json).

| test_clean | start MAE | end MAE | end median | end mean signed | ends ≤ 50 ms | ends ≤ 100 ms |
|---|---|---|---|---|---|---|
| `next_start` (previous default) | 57.0 ms | 75.4 ms | 46 ms | +45.8 ms | 53.2% | 80.1% |
| `next_start` + end refinement (new default) | 57.0 ms | 57.1 ms | 40 ms | −1.6 ms | 59.3% | 85.8% |

The refinement needs the audio. It runs when the audio is passed to `Aligner.align(result, audio)`, which
the CLI, `rumblex.transcribe()` and the benchmark all do. Starts are unchanged.

## Blank penalty (added 2026-11-21)

Suggested in a Reddit thread: smooth the CTC logits before Viterbi so that tokens hold more frames. Dividing
the logits by a temperature cannot change a Viterbi path, because it scales every path's score alike (the
log-softmax normaliser is the same for every state within a frame). The sweep confirms it: T from 1 to 8 gave
identical times. What does move the path is lowering the blank's score: `blank_penalty` is subtracted from
the blank's log-prob before Viterbi. Word scores still use the unpenalised log-probs.

Swept on `dev_clean` over penalties {0.5, 1, 2, 3, 5, 8}, with the start shift recalibrated on dev for each
(median signed start error), and chosen by start MAE + end MAE with the shipped end rule: penalty 5, start
shift −10 ms. Reported once on `test_clean`. Script `scripts/sweep_smoothing.py`, output
[alignment_smoothing_sweep.json](alignment_smoothing_sweep.json). The dev states are the ones above; the
`test_clean` states were collected again with the same script on CPU (float32), since the GPU copies were no
longer around. On them the old default gives the same 57.0 / 57.1 ms as the table above.

| test_clean | start MAE | start median | starts ≤ 50 ms | end MAE | end median | ends ≤ 100 ms |
|---|---|---|---|---|---|---|
| penalty 0, shift −22 ms (previous default) | 57.0 ms | 42 ms | 57.9% | 57.1 ms | 40 ms | 85.8% |
| penalty 5, shift −10 ms (new default) | 52.2 ms | 36 ms | 64.6% | 57.2 ms | 38 ms | 84.5% |

On the path, blank frames drop from 65% to 49% and a token holds 1.9 frames instead of 1.3. Starts improve;
ends with the shipped rule stay the same, since that rule mostly takes them from the next word's start and
the audio. Ends taken straight from the last CTC frame (`spike`) improve from 115 to 101 ms, still far behind.

## Transcription quality on the same data

| set | words | WER |
|---|---|---|
| LibriSpeech dev_clean (100 utt.) | 1,980 | 2.17% |
| LibriSpeech test_clean (200 utt.) | 4,091 | 2.30% |
| FLEURS Dutch dev, 12 clips (spot check) | 246 | 5.69% |

WER uses lowercase text with punctuation stripped (`normalize_text` in the eval script), with no other
normalisation. The Dutch set is far too small to be a benchmark. Language detection returned only Dutch
on the 12 clips.

## Speed (same runs)

- Transcription: GTX 1060 6GB, float32, batch 8, with encoder states kept. 36× real time on dev and 39×
  on test, counting the 0.5 s gaps between utterances. A separate run of `scripts/check_transcriber.py`
  gave 17× on 130 s of audio ([transcriber_check_gpu_fp32.json](transcriber_check_gpu_fp32.json)); the
  GPU is shared, so timings vary between runs.
- Alignment on CPU (CTC head + Viterbi): 50–58× real time. In the pipeline the head runs on the
  transcriber's device.

## Limits

- Frames are 80 ms apart inside each 1 s block, so starts are at best about ±40 ms. A wav2vec2 aligner
  like WhisperX's works at 20 ms. This was not compared with WhisperX here; the
  [first comparison](bench_2026-11-20_rtx4090/README.md) did that later.
- LibriSpeech is clean read speech, and the reference is another aligner. Spontaneous or noisy speech has
  not been measured.
- Ends are judged against MFA's convention (end = next start). Against human labels that mark silence
  inside words differently, `next_start` may look worse.
