# Context across chunks

rumblex decodes each chunk of a long file on its own. The model accepts free text in its system prompt
(`prompt=`), so the text of the chunk before could be handed to it as context. This measures whether that
helps. On the dev split it does not: it roughly doubles the word error rate.

Measured 2026-11-24 on a GTX 1070, float32, batch size 8, with `scripts/eval_chunk_context.py` (`collect`
then `score`); the numbers are in [chunk_context_eval.json](chunk_context_eval.json). The chunk texts
themselves are not kept here: the talks are CC BY-NC-ND.

## Setup

- Data: TED-LIUM long-form (`distil-whisper/tedlium-long-form`, revision `ea3a7847`), whole talks with a
  reference transcript. The `validation` split is used as dev: 8 talks, 93.8 min, 17,705 reference words.
  The `test` split (11 talks) has not been run.
- Chunks are the ones `rumblex.transcribe` makes (VAD, up to 20 s): 378 over the 8 talks. English is
  forced; other options are the package defaults.
- WER is over whole talks with Whisper's English normaliser.

Variants:

| variant | context given to chunk k |
|---|---|
| none | nothing: what the package does |
| prev | the text this same run produced for chunk k−1, so a talk is decoded in order |
| twopass | the `none` text of chunk k−1: a second pass that can still be batched |
| other | the `none` text of a chunk from another talk: context that has nothing to do with the audio |

## Results

| variant | errors | WER | WER within 3 words of a chunk boundary | WER elsewhere | time |
|---|---|---|---|---|---|
| none | 452 | 2.55% | 3.22% | 2.46% | 175 s |
| prev | 941 | 5.31% | 7.82% | 4.95% | 335 s |
| twopass | 855 | 4.83% | 7.36% | 4.46% | 184 s |
| other | 573 | 3.24% | 4.64% | 3.04% | 185 s |

Context from the same talk makes the transcript worse, and by more than unrelated context does. `prev`
is also about twice as slow, since a talk's chunks can no longer be batched together.

Per talk, `prev` is worse than `none` on seven of the eight talks, and the size of the loss varies a lot:
from 5.2% to 16.9% on one talk and from 2.2% to 6.7% on another, against 3.2% to 3.7% on a third. The
one talk where it is better goes from 2.3% to 1.8%.

## Where the errors come from

Most chunks come out the same with or without context. The loss is in a few chunks that change
completely (`against_none` in the JSON; a chunk "changed a lot" when 10 or more words differ from its
`none` text, and "repeats its context" when it shares a run of 5 or more words with the context it was
given that its `none` text does not share):

| variant | chunks changed | words changed | chunks changed a lot | words changed in those | chunks repeating their context |
|---|---|---|---|---|---|
| prev | 62 of 378 | 670 | 14 | 602 | 14 |
| twopass | 59 of 378 | 558 | 11 | 485 | 11 |
| other | 46 of 378 | 192 | 4 | 117 | 5 |

In those chunks the model writes out the context again, in part or whole, in place of what is said in
the audio, and in some it then loops on a phrase. This happens with unrelated context too (5 chunks of
`other` repeat text from a different talk), so it is the prompt being copied, not the audio being
misheard. The chunks it happens to are of every length, from 0.5 s to nearly 20 s. With `prev` a bad
chunk's text becomes the next chunk's context, which is why `prev` loses a little more than `twopass`.

## The sentence-break question

The reason to try context was that a sentence crossing a chunk boundary can be broken there. In the raw
text of `none`, 313 of 369 boundaries follow a chunk that ends in `.`, `?` or `!`, and 14 chunks start
with a lower-case letter. With `prev` that is 309 of 370 and 7; with `twopass` 310 of 370 and 9. So
context halves the lower-case starts, but the words next to a boundary get worse, not better (3.22% to
7.82% WER), because that count includes the chunks that repeat their context.

## What this means

The previous chunk's text should not go into `prompt` as it is. Not tried here, and each would need its
own dev run before anything changes in the package:

- a short context (the last few words or the last sentence), which leaves less to copy;
- dropping a chunk's output when it repeats its context, and decoding that chunk again without context;
- context only for chunks that start mid-sentence.

The package is unchanged: chunks are still decoded without context.
