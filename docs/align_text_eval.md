# Aligning a given transcript

How accurate the word times from `rumblex.align` (CLI: `--text`) are. It takes a transcript you already
have, runs only the encoder and the CTC head, and gives every word of that text a time.

Measured 2026-11-23 on CPU, float32, with `scripts/eval_align_text.py`; the raw output is
[align_text_eval.json](align_text_eval.json). Run again the same day after starts were moved to loudness
dips ([alignment_eval.md](alignment_eval.md)); the numbers here are from that second run. The JSON doesn't
record the commit; it is the commit that last changed this file.

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
| align, per utterance | 4,043 | 49.5 ms | 34 ms | 67.3% | 87.7% | 67.3 ms | 36 ms | 80.9% | 0 |
| align, long file | 4,043 | 50.2 ms | 31 ms | 67.8% | 87.2% | 64.5 ms | 36 ms | 81.8% | 1 |
| transcribe + align | 3,945 | 50.6 ms | 34 ms | 67.3% | 87.6% | 54.6 ms | 34 ms | 85.4% | 0 |
| align, same 3,945 words | 3,945 | 49.5 ms | 34 ms | 67.2% | 87.5% | 67.2 ms | 36 ms | 80.9% | 0 |

Starts are as good as with transcribe + align. Ends are about 13 ms worse on average: the mean signed
end error is −21.9 ms against +3.6 ms for transcribe + align on the same words, so ends come out early.

The cause is the missing punctuation. The model writes punctuation, and in the CTC path a punctuation
mark sits right after the word it follows. Without those tokens, the word before them ends sooner. In a quick check (not saved here), aligning the model's own
transcript with its punctuation gave the same times as transcribe + align, and the same transcript with
punctuation removed gave early ends; casing made no difference. Letting the Viterbi search insert
optional punctuation tokens should fix this, and isn't done yet.

The one gross word in the long file is "whenever" (`3570-5694-0021`), placed 1.32 s early. Its onset was
clipped by the VAD, so the aligner had no audio for it where it was said.

## Time

On the CPU used here, aligning the 200 utterances one by one took 318.9 s against 1,102.2 s for
transcribing and aligning them, and the 28.6 min long file took 365.8 s. This is one run on a machine
that wasn't kept idle, so take these as rough.
