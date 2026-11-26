# Hotwords

A hotword list is a list of names and terms you expect in a recording: the people on a call, product
names, drug names, a company's jargon. rumblex nudges the decoder towards them while it transcribes.

```bash
rumblex call.wav --hotwords "Stifel, Paul Betz, SiTime"
rumblex call.wav --hotwords_file terms.txt
```

```python
result = rumblex.transcribe("call.wav", model, hotwords=["Stifel", "Paul Betz", "SiTime"])
```

For several files with the same list, compile it once:

```python
from rumblex.hotwords import Hotwords

terms = Hotwords(open("terms.txt", encoding="utf-8").read().splitlines(), model.tokenizer)
for path in files:
    result = rumblex.transcribe(path, model, hotwords=terms)
```

## Writing a list

One entry per line in a file, or separated by commas on the command line. An entry can be several words.

```
Stifel
Paul Betz
SiTime
paracetamol | paracetamols
```

- Write an entry the way it should appear in the transcript. Case variants are added for you: as given,
  lowercase, first letter capital and all capitals.
- An entry in all capitals also gets a Title Case form, so `GOLDMAN SACHS` covers "Goldman Sachs". Lists
  exported from other systems are often in capitals, and the model doesn't write names that way.
  `Hotwords(..., title_caps=False)` turns this off.
- Mixed-case names such as `SiTime` or `iPhone` can't be rebuilt from capitals. Write them as they are.
- Other written forms go after a bar: plurals, inflections, another spelling. They are not guessed.
- Keep the list to what this recording or this customer needs. A list above 2,000 entries gets a warning.

## What it does

The list is turned into a prefix tree over the decoder's tokens. At each decoding step, the tokens that
would start an entry at a word start, or continue an entry that is being written, get a fixed bonus of 2
added to their logit. Decoding stays greedy, so the bonus only decides close calls: a word the model
clearly heard is left alone.

Three rules keep the bonus from doing damage:

- an entry that has been written once in a chunk gets no bonus for the rest of that chunk;
- when the first pieces of an entry are written but the rest doesn't follow, the bonus on that beginning is
  halved, which stops loops when the first piece of an entry is a word by itself;
- there is no bonus at a step where the model wants to stop.

A word's `probability` in the output is the model's own, from before the bonus. A hotword that only got
written because of its bonus therefore shows up with a low probability, which is a useful thing to check
when a transcript has a list word where you didn't expect one.

Word timestamps, speaker labels and the writers are unchanged: the aligner works on whatever text was
decoded.

## Measured

2026-11-26, GTX 1070, float32, batch size 4, forced English, with `scripts/eval_hotwords.py` (`collect` then
`score`); the numbers are in [hotwords_eval.json](hotwords_eval.json).

- Audio: Earnings-21, 44 earnings calls. Each call is cut into clips in pauses, and 20 clips per call are
  drawn at random (seed 0): 857 clips, 3.69 h, 33,667 reference words.
- List: Rev's Earnings-21 biasing list, 1,013 words and phrases, all in capitals. It is one list for all
  calls and was made from the reference transcripts, so it is a favourable list: every entry is said
  somewhere in the 39 hours, though most are not said in any given clip.
- A list word is a word of a list entry that is alphabetic, longer than two letters and outside the 5,000
  most frequent English words. There are 887, and 321 of them are spoken in the clips. List recall is the share of those
  that the transcript gets right. False triggers are list words written where the reference has something
  else, per 100 clips.
- WER uses Whisper's English normaliser. Changes are against no hotwords, with a paired bootstrap 95%
  interval over clips (2,000 resamples).

| setting | WER | list recall | false triggers per 100 clips | WER change | list recall change |
|---|---|---|---|---|---|
| no hotwords | 5.84 | 67.9 | 1.8 | | |
| hotwords | 6.02 | 76.9 | 2.1 | +0.18 [0.06, 0.31] | +9.0 [5.7, 12.5] |
| hotwords, no Title Case for words of up to four letters | 6.02 | 76.9 | 2.1 | +0.18 [0.06, 0.31] | +9.0 [5.7, 12.5] |
| hotwords, list as given (no Title Case forms) | 6.01 | 70.7 | 1.8 | +0.17 [0.04, 0.30] | +2.8 [0.9, 5.0] |
| no hotwords, the whole list in `--prompt` (batch size 1) | 24.13 | 52.0 | 4.6 | +18.29 [15.94, 21.13] | -15.9 [-20.6, -11.6] |

Hotwords get 9 more of every 100 spoken list words right, and cost 0.18 points of WER; neither interval
includes zero. Most of the gain comes from the Title Case forms: with the list as Rev
ships it, in capitals, recall moves 2.8 points for the same WER cost. Leaving words of up to four letters
out of the Title Case rule changed no transcript in a way the scores see, so the plain rule ships: every
all-capitals entry gets a Title Case form. The same list in the prompt is much worse than no list at all.
It is 6,286 tokens of prompt, WER goes to 24.13% and decoding takes 5.6 times as long.

Speed: decoding took 518.0 s without hotwords and 525.2 s with them (real-time factor 0.0390 and 0.0396).
The logits processor itself took 69.3 ms per clip on the host. Compiling the list took 1.32 s.

## Limits

- One sample of clips from one kind of audio, English earnings calls, with one list. Other languages have
  not been measured here.
- The list was built from the references. A list that comes from somewhere else (slides, a CRM export)
  will cover fewer of the words that are said, and gain less.
- Nothing above 1,013 entries was measured. The warning above 2,000 is a caution, not a measured limit.
  During development I also tried general term lists of ten thousand entries and more, and they made
  transcripts worse; those runs are not in this repository, so there are no numbers for them here.
- The bonus was not tuned on this data. It is fixed at 2.
- Hotwords need the default backend. They are not available with `--vllm_url`, in `rumblex serve`, or
  with `--text` (which doesn't decode).
- One list per call. Chunks of a file can't have their own lists.
- `--prompt` still works and can be combined with hotwords, but the two were not measured together here.
  A list doesn't belong in the prompt: the last row of the table is this list as a prompt, and the model
  can copy its prompt into the transcript ([chunk_context_eval.md](chunk_context_eval.md)). That row is
  one long list; a prompt of a few names was not measured.

## Data

Nothing from these sources is in this repository; the script reads local copies.

- Earnings-21 and the biasing list: [revdotcom/speech-datasets](https://github.com/revdotcom/speech-datasets)
  (`earnings21/`). The transcripts are CC BY-SA 4.0. The list is from "Improving Contextual Recognition of
  Rare Words with an Alternate Spelling Prediction Model" (arXiv 2209.01250).
- Rev's transcripts have no word times, so the clips are cut with the timed copies of the same transcripts
  in [ConEC](https://github.com/huangruizhe/ConEC) (Huang et al., LREC-COLING 2024).
