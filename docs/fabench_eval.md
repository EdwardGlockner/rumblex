# Word timings against human labels (FA-Bench, Buckeye)

The other timing numbers in this repository are scored against Montreal Forced Aligner output, which is
itself an aligner. Here `rumblex.align` and WhisperX's aligner are scored against the hand-corrected word
boundaries of the Buckeye corpus, using [FA-Bench](https://github.com/olewave/fa-bench)
(arXiv 2609.32396) to cut the corpus, run both tools and compute the metrics.

Measured 2026-11-24 on CPU (Intel i5-6600K, 4 threads, float32), FA-Bench at commit `892e8ba` with
[scripts/fabench.patch](../scripts/fabench.patch) applied. rumblex was run through
`scripts/fabench_worker.py` at the commit that added this file; the aligner code is that of 0.4.1.
The rows FA-Bench wrote are in [fabench_eval.csv](fabench_eval.csv).

This is FA-Bench's track 1: both tools are handed the reference words and only have to place them.
Transcribing from audio alone (track 2) has not been run; see [Track 2](#track-2-not-run) below.

## Setup

- Data: Buckeye, FA-Bench's `paper` protocol. Recordings are cut at silences of 300 ms or more, each
  piece keeps 200 ms of padding on both sides, and pieces with fewer than four words or with unknown or
  cut-off words are dropped. That gives 4,456 utterances from 8 speakers in dev and 4,513 from 8 other
  speakers in test, the same counts FA-Bench reports. Speech is spontaneous interviews: fillers,
  reduced forms, restarts.
- rumblex: `rumblex.align(audio, transcript, model)` with `Reson8/Rumble-2-Flash` and package defaults.
  Transcripts are lower case without punctuation.
- WhisperX 3.8.6: FA-Bench's own recipe and worker, unchanged except for `device: cpu`. It aligns with
  `WAV2VEC2_ASR_BASE_960H`.
- Nothing was tuned on either split. Dev and test are both reported.

FA-Bench's columns, as used below:

- start / end MAE (`wts_start_mae_ms`, `wts_end_mae_ms`): mean absolute error of word starts and of word
  ends.
- timestamp MAE (`wts_mae_ms`): the two pooled.
- boundary MAE (`wbe_ms`): one error per word boundary. Where two words touch, the time is taken from
  the end of the left word, so this mostly follows end error. It is the "MAE" column of FA-Bench's
  leaderboard.
- F1 @ 20 ms (`wts_f1`): share of word starts and ends within 20 ms of the label.

## Results

| split | tool | words | start MAE | end MAE | timestamp MAE | boundary MAE | F1 @ 20 ms |
|---|---|---|---|---|---|---|---|
| dev | rumblex | 47,401 | 42.9 ms | 57.4 ms | 50.2 ms | 56.8 ms | 0.331 |
| dev | WhisperX | 47,386 | 58.8 ms | 36.7 ms | 47.8 ms | 40.3 ms | 0.284 |
| test | rumblex | 45,921 | 42.4 ms | 56.5 ms | 49.5 ms | 56.0 ms | 0.332 |
| test | WhisperX | 45,921 | 58.4 ms | 38.1 ms | 48.2 ms | 41.7 ms | 0.281 |

"words" is the number of words scored (timestamps matched, divided by two), out of 47,401 in dev and
45,922 in test. WhisperX returned nothing for one dev utterance (15 words), and one test word is missing
from both tools' output.

rumblex places word starts about 16 ms better than WhisperX, and WhisperX places word ends about 19 ms
better than rumblex. Pooled over starts and ends the two are within 3 ms of each other, with WhisperX
ahead. On boundary MAE, which follows the ends, WhisperX is ahead by about 15 ms. On F1 at 20 ms rumblex
is ahead, 0.33 against 0.28. The same split between starts and ends shows on LibriSpeech against MFA
references ([bench_2026-11-23_rtx4090_timing](bench_2026-11-23_rtx4090_timing/README.md)).

Dev and test agree to within 1 ms for rumblex, so the result does not depend on the speakers.

Compared with LibriSpeech and MFA references ([alignment_eval.md](alignment_eval.md), start 49.5 ms and
end 53.7 ms on `test_clean` for transcribe + align), starts are better here and ends a little worse.
The data, the labels and the task all differ, so this is only a rough check that the two evaluations
tell the same story about rumblex. Ends being the weaker side matches what
[align_text_eval.md](align_text_eval.md) found for text without punctuation.

## Check against FA-Bench's published row

FA-Bench's leaderboard (olewave.com/en/leaderboard/fa-bench, as updated 2026-11-17) lists WhisperX on
Buckeye clean at F1 0.281 and MAE 41.7 ms; the paper gives the same MAE for the test split. Our CPU run
of the same recipe gives 0.281 and 41.7 ms on test, so the setup here reproduces theirs. Their rows for other aligners
on the same data, not re-run here: nyra-FA 0.731 / 20.2 ms, Montreal Forced Aligner 0.706 / 21.3 ms,
MMS-FA 0.344 / 37.1 ms. rumblex's 0.332 / 56.0 ms on test sits next to MMS-FA on F1 and behind it on
MAE, and far behind the aligners built on phone models. With 13 frames a second the CTC head cannot do
much better at a 20 ms tolerance.

## Track 2 (not run)

FA-Bench scores tools in two ways. Track 1, above, hands the tool the audio and the correct words and
asks only when each word was said. Track 2 hands it the audio alone: the tool transcribes and times its
own words, which for rumblex is the normal `rumblex.transcribe` and for WhisperX is faster-whisper
large-v3 followed by its aligner. A word the tool gets wrong has no label to match, so the track 2
score mixes recognition errors with timing errors, and FA-Bench reports WER next to it.

It was left out for now because it would mostly combine two things already measured separately: timing
against human labels (above; the timing code is the same in both modes) and recognition (WER on other
sets, in the top-level README). It would add one number for normal use on spontaneous speech. The
WhisperX side also needs a GPU to finish in reasonable time.

What is in place, and what has and hasn't been tried:

- `scripts/fabench_worker.py` has a `transcribe` task: English forced, numbers spelled out
  (`digits=False`), fillers kept (`filler=True`, `verbatim=True`), since Buckeye's transcripts spell
  numbers out and include "um" and "uh". Otherwise package defaults. Utterances here are under 20 s, so
  they go in whole without VAD. These options are a first guess and have not been checked against the
  dev split.
- `scripts/fabench.patch` adds the `rumblex_asr` adapter and its recipe
  (`evals/timestamp_asrs/rumblex_asr/config.yaml`).
- The worker's `transcribe` task is covered by a unit test with the model mocked. The `rumblex_asr`
  recipe has not been run inside FA-Bench, not even on a few utterances, so expect to fix something on
  the first try.
- FA-Bench's WhisperX counterpart is `whisperx_asr`. It shares the venv of the track 1 `whisperx`
  recipe and its recipe is set to `device: cuda`.

To run it, with the data already ingested as below:

```
python evals/gen_config.py --corpus buckeye --subset dev --tools rumblex_asr
C=evals/timestamp_asrs/rumblex_asr/en/buckeye/dev/origin/config.yaml
python -m fabench.cli align --config $C --aligner rumblex_asr
python -m fabench.cli score --config $C
```

and the same with `whisperx_asr`. Any choice of options should be made on dev and reported once on
test. When reading the result, compare `wer` first: a timing column only covers the words that matched.

## Reproducing

Buckeye has to be requested from Ohio State (buckeyecorpus.osu.edu). Its licence allows research use
and does not allow passing the data on, so only the scores are kept here: no audio, transcripts or
per-word output.

```
git clone https://github.com/olewave/fa-bench && cd fa-bench && git checkout 892e8ba
git apply /path/to/rumblex/scripts/fabench.patch
# set the three paths in evals/aligners/rumblex/config.yaml, FABENCH_BUCKEYE_ROOT in .fabench.env,
# and install WhisperX with evals/aligners/whisperx/download_and_install.sh
python evals/gen_config.py --corpus buckeye --subset dev --tools rumblex
C=evals/aligners/rumblex/en/buckeye/dev/origin/config.yaml
python -m fabench.cli ingest --config $C
python -m fabench.cli mix --config $C
python -m fabench.cli align --config $C --aligner rumblex
python -m fabench.cli score --config $C
```

The same with `--subset test` and with `whisperx` in place of `rumblex`. The patch adds the rumblex
adapter and recipe, lets FA-Bench find the interpreter of a venv made on Windows, and sets WhisperX to
CPU; on Windows the commands also need `PYTHONUTF8=1`.

Each split took about an hour per tool on this CPU, with the two tools running at the same time, so
these are not speed measurements.
