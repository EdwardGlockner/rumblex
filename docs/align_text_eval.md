# Aligning a given transcript

How accurate the word times from `rumblex.align` (CLI: `--text`) are. It takes a transcript you already
have, runs only the encoder and the CTC head, and gives every word of that text a time.

Measured 2026-11-23 on CPU, float32, with `scripts/eval_align_text.py`; the raw output is
[align_text_eval.json](align_text_eval.json). The JSON doesn't record the commit; it is the commit that
added this file.

## Setup

Same data and references as [alignment_eval.md](alignment_eval.md): 200 LibriSpeech `test_clean`
utterances (seed 0), Montreal Forced Aligner word alignments as reference (forced-aligner output, not
human labels). The text given to the aligner is LibriSpeech's reference transcript: upper case, no
punctuation.

Two tests:

- **Per utterance**: each utterance aligned to its own transcript. For comparison the same utterances
  are also transcribed and aligned the usual way (`rumblex.transcribe`). Transcribe + align only times
  words the model recognised, so both are also scored on the 3,945 words both of them time.
- **Long file**: all 200 utterances joined with 1 s of silence into one 28.6 min file and aligned to the
  joined transcript. This exercises VAD chunking (102 chunks) and the step that decides which words go
  in which chunk.

Words come from the given text, so errors are paired by index. 14 words MFA left as `<unk>` are not
scored, and one utterance (`61-70970-0004`) where MFA splits a word in two is aligned but not scored.
"Gross" counts word starts more than 1 s off.

## Results

| | words | start MAE | start median | starts ≤ 50 ms | starts ≤ 100 ms | end MAE | end median | ends ≤ 100 ms | gross |
|---|---|---|---|---|---|---|---|---|---|
| align, per utterance | 4,043 | 51.9 ms | 40 ms | 63.8% | 87.1% | 68.4 ms | 40 ms | 80.8% | 0 |
| align, long file | 4,043 | 53.2 ms | 37 ms | 62.8% | 86.9% | 66.2 ms | 40 ms | 81.8% | 1 |
| transcribe + align | 3,945 | 52.7 ms | 38 ms | 64.1% | 87.3% | 58.0 ms | 40 ms | 84.2% | 0 |
| align, same 3,945 words | 3,945 | 51.9 ms | 40 ms | 63.8% | 87.0% | 68.4 ms | 40 ms | 80.9% | 0 |

Starts are as good as with transcribe + align. Ends are about 10 ms worse on average: the mean signed
end error is −23.2 ms against +5.7 ms for transcribe + align on the same words, so ends come out early.

The cause is the missing punctuation. The model writes punctuation, and in the CTC path a punctuation
mark sits right after the word it follows. Without those tokens, the word before them ends sooner. In a quick check (not saved here), aligning the model's own
transcript with its punctuation gave the same times as transcribe + align, and the same transcript with
punctuation removed gave early ends; casing made no difference. Letting the Viterbi search insert
optional punctuation tokens should fix this, and isn't done yet.

The one gross word in the long file is "whenever" (`3570-5694-0021`), placed 1.35 s early. Its onset was
clipped by the VAD, so the aligner had no audio for it where it was said.

## Time

On the CPU used here, aligning the 200 utterances one by one took 303.6 s against 1,060.3 s for
transcribing and aligning them, and the 28.6 min long file took 360.6 s. This is one run on a machine
that wasn't kept idle, so take these as rough.
