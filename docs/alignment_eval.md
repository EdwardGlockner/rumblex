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

## Starts on loudness dips, and the last word's end (added 2026-11-23)

Most words run straight into the next one, and there a word's end is the next word's start, so its end
error is that start's error. Better ends therefore need better starts. The CTC head gives nothing finer
than its 80 ms frames: weighting a start by the token's probability on the neighbouring frames changed
nothing on dev, because the frames around a spike carry almost no probability for the token. Those checks
were not saved; what follows is the part that helped.

Two refinements on the audio, both in `rumblex.aligner` and on by default when the audio is passed:

- `dip_window`, `dip_smooth`: each start moves to the quietest 10 ms frame within `dip_window` of its CTC
  start, on loudness smoothed over `dip_smooth` frames (`snap_to_dips`).
- `last_inside`: the silence search for the last word of a chunk begins at the start of its last CTC frame
  instead of after it, so that word can end inside the frame.

Swept on `dev_clean` over windows {5, 10, 20, 30, 40, 50} ms, smoothing {1, 3, 5, 7, 9, 11, 15} frames and
`last_inside` on or off, with the start shift recalibrated on dev for each (median signed start error), and
chosen by start MAE + end MAE: 20 ms, 7 frames, `last_inside` on, start shift −4 ms. Reported once on
`test_clean`. Same states as the blank penalty sweep; script `scripts/sweep_word_ends.py`, output
[alignment_word_ends_sweep.json](alignment_word_ends_sweep.json).

| dev_clean | start MAE | start median | end MAE | end median |
|---|---|---|---|---|
| previous default | 45.4 ms | 34 ms | 51.0 ms | 36 ms |
| `last_inside` only | 45.4 ms | 34 ms | 49.3 ms | 34 ms |
| dips only (20 ms, 7 frames) | 43.0 ms | 30 ms | 49.7 ms | 32 ms |
| both (new default) | 43.0 ms | 30 ms | 48.0 ms | 32 ms |

| test_clean | start MAE | start median | starts ≤ 50 ms | end MAE | end median | ends ≤ 50 ms | ends ≤ 100 ms |
|---|---|---|---|---|---|---|---|
| previous default | 52.2 ms | 36 ms | 64.6% | 57.2 ms | 38 ms | 60.3% | 84.5% |
| new default | 49.5 ms | 32 ms | 69.0% | 53.7 ms | 34 ms | 65.2% | 85.6% |

Scored through `Aligner.align_tokens` with default options, `test_clean` gives the same 49.5 / 53.7 ms. The
gain is small next to the gap to a 20 ms aligner. On an RTX 4090 in bf16, next to WhisperX: starts 52.6 to
50.6 ms, ends 57.7 to 54.6 ms, against WhisperX's 74.8 and 33.9 ms
([run](bench_2026-11-23_rtx4090_timing/README.md)).

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
