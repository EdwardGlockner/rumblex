# Word timings against human labels (FA-Bench, Buckeye)

The other timing numbers in this repository are scored against Montreal Forced Aligner output, which is
itself an aligner. Here `rumblex.align` and WhisperX's aligner are scored against the hand-corrected word
boundaries of the Buckeye corpus, using [FA-Bench](https://github.com/olewave/fa-bench)
(arXiv 2609.32396) to cut the corpus, run both tools and compute the metrics.

Measured 2026-11-24 on CPU (Intel i5-6600K, 4 threads, float32), FA-Bench at commit `892e8ba` with
[scripts/fabench.patch](../scripts/fabench.patch) applied. rumblex was run through
`scripts/fabench_worker.py`, twice on the same day: first with the aligner of 0.4.1, then at the commit
that added optional punctuation for text given without it ([align_text_eval.md](align_text_eval.md)).
WhisperX was run once. rumblex was run a third time on 2026-11-26, same machine and FA-Bench checkout, at
the commit that added the boundary head ([boundary_head.md](boundary_head.md)), which is the aligner of
0.6.0. The rows FA-Bench wrote are in [fabench_eval.csv](fabench_eval.csv); its `run` column tells the
three rumblex runs apart.

Qwen3-ForcedAligner-0.6B was added on 2026-11-27 through FA-Bench's own `qwen3_fa` recipe, same checkout
and same data, on a GTX 1070. Its two rows have an empty `run` column, like WhisperX's.

This is FA-Bench's track 1: the tools are handed the reference words and only have to place them.
Transcribing from audio alone (track 2) has been run for rumblex on dev only; see
[Track 2](#track-2-dev-only) below.

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
- Qwen3-ForcedAligner-0.6B: FA-Bench's `qwen3_fa` recipe and worker, with qwen-asr 0.0.6 and
  torch 2.14.1+cu126 in a venv of its own, the model read from a local copy, language `English`, float32
  on a GTX 1070 (8 GB).
- Nothing was tuned on either split. Dev and test are both reported. The penalty on the optional
  punctuation was chosen on LibriSpeech `dev_clean` before the second run. The boundary head was fitted
  on LibriSpeech training utterances against Montreal Forced Aligner times and its settings chosen on
  `dev_clean`; it has seen no Buckeye audio or labels.

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
| dev | rumblex 0.6.0 | 47,401 | 24.5 ms | 29.7 ms | 27.1 ms | 29.8 ms | 0.540 |
| dev | Qwen3-ForcedAligner | 47,401 | 31.3 ms | 35.6 ms | 33.4 ms | 34.7 ms | 0.425 |
| dev | rumblex 0.5.0 | 47,401 | 42.6 ms | 40.4 ms | 41.5 ms | 41.5 ms | 0.356 |
| dev | WhisperX | 47,386 | 58.8 ms | 36.7 ms | 47.8 ms | 40.3 ms | 0.284 |
| dev | rumblex 0.4.1 | 47,401 | 42.9 ms | 57.4 ms | 50.2 ms | 56.8 ms | 0.331 |
| test | rumblex 0.6.0 | 45,921 | 25.1 ms | 29.6 ms | 27.4 ms | 30.3 ms | 0.536 |
| test | Qwen3-ForcedAligner | 45,921 | 30.9 ms | 34.0 ms | 32.5 ms | 33.8 ms | 0.426 |
| test | rumblex 0.5.0 | 45,921 | 42.1 ms | 41.1 ms | 41.6 ms | 42.5 ms | 0.358 |
| test | WhisperX | 45,921 | 58.4 ms | 38.1 ms | 48.2 ms | 41.7 ms | 0.281 |
| test | rumblex 0.4.1 | 45,921 | 42.4 ms | 56.5 ms | 49.5 ms | 56.0 ms | 0.332 |

In the CSV, 0.6.0 is the run `boundary head` and 0.5.0 is `optional punctuation`.

"words" is the number of words scored (timestamps matched, divided by two), out of 47,401 in dev and
45,922 in test. WhisperX returned nothing for one dev utterance (15 words), and one test word is missing
from every tool's output.

Qwen3-ForcedAligner sits between rumblex 0.6.0 and the rest. rumblex 0.6.0 is ahead of it on every
column: 6 to 7 ms on starts, 4 to 6 ms on ends, 3.5 to 5 ms on boundary MAE, and 0.54 against 0.43 on F1
at 20 ms. There are no intervals on these differences; FA-Bench reports one number per split. Qwen is
ahead of rumblex 0.5.0 and of WhisperX on every column. Its worker counted 3,133 of 47,530 output items
in dev and 2,731 of 46,004 in test with the same start and end time (6.6% and 5.9%); FA-Bench scores
them as they are. On LibriSpeech against Montreal Forced Aligner times
([bench_2026-11-27_rtx4090_qwen_aligner](bench_2026-11-27_rtx4090_qwen_aligner/README.md)) Qwen's ends
were 2.9 ms closer than rumblex's; against hand labels on spontaneous speech they are not.

With the boundary head (0.6.0), starts are 17 to 18 ms better than in 0.5.0 and ends about 11 ms better,
and the share of starts and ends within 20 ms goes from 0.36 to 0.54. Against WhisperX that is 33 to
34 ms better on starts, 7 to 8 ms better on ends and about 11 ms better on boundary MAE. The head was
fitted to forced-aligner times on read speech, so this is also the first check of it against human
labels and on spontaneous speech: the gain is smaller than on LibriSpeech (where starts went from 49.5
to 20.3 ms, [boundary_head.md](boundary_head.md)) but it holds.

Without the head (0.5.0), rumblex places word starts about 16 ms better than WhisperX, and WhisperX
places word ends 3 to 4 ms better than rumblex. Pooled over starts and ends rumblex is ahead by about
6 ms. On boundary MAE, which follows the ends, WhisperX is ahead by about 1 ms. On F1 at 20 ms rumblex
is ahead, 0.36 against 0.28.

With 0.4.1 the ends were 16 to 17 ms worse (57.4 and 56.5 ms), which put WhisperX ahead on timestamp MAE
and about 15 ms ahead on boundary MAE. Buckeye's transcripts have no punctuation, and that was the
early-ends problem [align_text_eval.md](align_text_eval.md) describes; optional punctuation removed it
here as it did on LibriSpeech. Starts moved by less than half a millisecond.

Dev and test agree to within about 1 ms for rumblex, so the result does not depend on the speakers.

Compared with LibriSpeech and MFA references ([align_text_eval.md](align_text_eval.md), start 49.5 ms and
end 51.9 ms on `test_clean` for the same task with 0.5.0), both are better here for 0.5.0. The data and the labels differ,
so this is only a rough check that the two evaluations tell the same story about rumblex.

## Check against FA-Bench's published row

FA-Bench's leaderboard (olewave.com/en/leaderboard/fa-bench, as updated 2026-11-17) lists WhisperX on
Buckeye clean at F1 0.281 and MAE 41.7 ms; the paper gives the same MAE for the test split. Our CPU run
of the same recipe gives 0.281 and 41.7 ms on test, so the setup here reproduces theirs. Their rows for other aligners
on the same data, not re-run here: nyra-FA 0.731 / 20.2 ms, Montreal Forced Aligner 0.706 / 21.3 ms,
MMS-FA 0.344 / 37.1 ms. rumblex 0.5.0, at 0.358 / 42.5 ms on test, sat next to MMS-FA on F1 and next to
WhisperX on MAE. 0.6.0, at 0.536 / 30.3 ms, is ahead of MMS-FA and WhisperX on both and still clearly
behind the aligners built on phone models.

## Track 2 (dev only)

`rumblex.transcribe` on the dev split, 2026-11-27, rumblex 0.6.1 at commit `deb1d8c` with the worker of
this branch, GTX 1070, float32. Three results, `track 2 first run`, `track 2 fillers kept` and `track 2`
in the CSV:

| dev | WER | words matched | start MAE | end MAE | timestamp MAE | boundary MAE | F1 @ 20 ms |
|---|---|---|---|---|---|---|---|
| first run, fillers kept | 11.2% | 43,621 of 47,401 | 24.9 ms | 33.6 ms | 29.2 ms | 32.4 ms | 0.496 |
| same output, edge punctuation removed | 10.9% | 43,798 | 24.9 ms | 33.7 ms | 29.3 ms | 32.4 ms | 0.498 |
| second run, fillers left out | 10.7% | 43,282 | 23.4 ms | 30.0 ms | 26.7 ms | 28.5 ms | 0.507 |

The first run asked the model to keep fillers (`filler=True`), a guess made before any run. Its output
had a few hundred words with a quotation mark or the punctuation inside one attached (`"well`, `want."`),
which FA-Bench compares as they come; the worker now removes punctuation at the edges of a word, and the
second row is the first run's output rescored with that applied. The second run leaves fillers out, which
is the package default. It has the lower WER and the better timing, so it is the setting kept, and the
only choice made on dev. With fillers kept the model wrote "uh" 886 times, "um" 265 times and "mm-hmm"
308 times; many of those are not in the reference as words.

The MAE columns cover only the words that matched the reference. On those, with fillers left out, timing
is as good as in track 1 (start 23.4 against 24.5 ms, end 30.0 against 29.7 ms). F1 also counts the
words that were missed or added, which is why it is lower than track 1's 0.540.

A rough count of the first run's word errors (difflib alignment of the output against the reference
words, not FA-Bench's scorer) says part of the WER is spelling convention and not recognition: Buckeye
writes `yknow`, `gonna` and `wanna` where rumblex writes "you know" (251 times), "going to" (85) and
"want to" (49), each counted as two errors. That has been left as it is.

The test split is to be run once with the kept setting. The WhisperX side (`whisperx_asr`) has not been
run. The runs took 61 to 79 minutes on a shared card and are not speed measurements.

Background and how to run it:

FA-Bench scores tools in two ways. Track 1, above, hands the tool the audio and the correct words and
asks only when each word was said. Track 2 hands it the audio alone: the tool transcribes and times its
own words, which for rumblex is the normal `rumblex.transcribe` and for WhisperX is faster-whisper
large-v3 followed by its aligner. A word the tool gets wrong has no label to match, so the track 2
score mixes recognition errors with timing errors, and FA-Bench reports WER next to it.

It mostly combines two things already measured separately: timing against human labels (above; the
timing code is the same in both modes) and recognition (WER on other sets, in the top-level README). What
it adds is one number for normal use on spontaneous speech. The WhisperX side needs a GPU to finish in
reasonable time.

What is in place:

- `scripts/fabench_worker.py` has a `transcribe` task: English forced and numbers spelled out
  (`digits=False`), since Buckeye's transcripts spell them out. Otherwise package defaults; setting
  `FABENCH_RUMBLEX_FILLER: "1"` under `env` in the recipe keeps fillers. Utterances here are under 20 s,
  so they go in whole without VAD.
- `scripts/fabench.patch` adds the `rumblex_asr` adapter and its recipe
  (`evals/timestamp_asrs/rumblex_asr/config.yaml`).
- The worker's `transcribe` task is covered by a unit test with the model mocked. The `rumblex_asr`
  recipe ran inside FA-Bench without changes, with `device: cuda` set in the generated config.
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

The same with `--subset test` and with `whisperx` or `qwen3_fa` in place of `rumblex`. The patch adds the
rumblex adapter and recipe, lets FA-Bench find the interpreter of a venv made on Windows (also for the
`qwen3_fa` adapter), and sets WhisperX to CPU; on Windows the commands also need `PYTHONUTF8=1`.

For `qwen3_fa`, make a venv with qwen-asr at `evals/aligners/qwen3_fa/venv` (FA-Bench's
`download_and_install.sh` does this on Linux). On Windows, with that path a junction to a venv elsewhere,
`gen_config.py` wrote the junction's target into the config and the adapter then could not find its
worker; the `venv` line of the two generated configs was set back to the path under
`evals/aligners/qwen3_fa/` by hand, and `model` to the local copy. `ingest` exits with code 1 on Buckeye
because some labels end a few milliseconds outside their clip; it still writes its output, so carry on.

Each split took about an hour per tool on this CPU, with the two tools running at the same time, so
these are not speed measurements. The 0.6.0 run took about the same, 115 minutes for both splits. The
Qwen run took 52 minutes for both splits on the GTX 1070, scoring included.
