# Word confidence

Whether a word's score says anything about whether the word is right, and which score says it best.
rumblex 0.2.0 writes `score` on every word: the mean over the word's tokens of each token's mean CTC
probability over its frames. This compares it with other CTC scores and with the decoder's own token
probabilities.

Measured 2026-11-23 on CPU, float32. States were saved by `scripts/collect_confidence_states.py`
(`collect_summary.json` next to them records device, torch and timings) and scored by
`scripts/eval_confidence.py`; the raw output is [confidence_eval.json](confidence_eval.json). Neither JSON
records the commit; it is the commit that added this file.

## Setup

1,200 utterances each from AMI and Earnings-22 (seed 0), transcribed the way `rumblex.transcribe` chunks
them, English forced, digits off (47 min and 2.3 h of audio). Even positions are dev, odd positions are
test. Scores and thresholds are chosen on dev; test is reported once.

A hypothesis word is wrong when a minimum-edit-distance alignment to the reference doesn't match it (a
substitution or an insertion). Both sides go through Whisper's English normaliser without its number
step, "ok" counts as "okay", and dotted letters ("U.S.", "D. V. D.") are joined. Earnings-22 utterances
whose reference has a digit are left out (241 of them), since with digits off every number would count
as wrong. Words that normalise to nothing (fillers, a lone dash) are not scored.

Scores, each over the tokens of one word:

| score | what |
|---|---|
| `ctc_mean` | mean of each token's mean CTC probability over its frames (0.2.0's `score`) |
| `ctc_min` | the same, lowest token |
| `ctc_peak_min` | lowest over tokens of each token's highest CTC probability on its frames |
| `dec_mean` | exp of the mean decoder log-prob of the word's tokens |
| `dec_min` | lowest decoder token probability |
| `combined` | logistic regression on log `ctc_mean` and log `dec_min`, fitted on dev |

AUC is the chance that a random wrong word scores lower than a random right one (0.5 is chance).

## Results

| | words | wrong | `ctc_mean` | `ctc_min` | `ctc_peak_min` | `dec_mean` | `dec_min` | `combined` |
|---|---|---|---|---|---|---|---|---|
| AMI dev | 3,736 | 6.5% | 0.793 | 0.797 | 0.815 | **0.841** | 0.840 | 0.836 |
| Earnings-22 dev | 7,578 | 7.0% | 0.790 | 0.789 | 0.827 | **0.836** | 0.835 | 0.834 |
| AMI test | 3,689 | 5.5% | 0.825 | 0.837 | 0.856 | **0.882** | 0.879 | 0.876 |
| Earnings-22 test | 7,318 | 7.0% | 0.795 | 0.797 | 0.829 | **0.837** | 0.837 | 0.827 |

The decoder's mean token probability is best on both dev sets, and the order holds on test. Among
CTC-only scores, `ctc_peak_min` beats 0.2.0's `ctc_mean` by 0.02 to 0.04 everywhere. Adding the CTC score
to the decoder score this way doesn't help. Following this, rumblex now writes `dec_mean` as each word's
`probability` and `ctc_peak_min` as its `score`.

Flagging words below a threshold chosen on dev (both sets pooled) to flag 10% of dev words:

| test set | score | flagged | errors caught | flagged words that are wrong |
|---|---|---|---|---|
| AMI | `ctc_mean` | 12.2% | 54.2% | 24.4% |
| AMI | `ctc_peak_min` | 13.1% | 55.7% | 23.4% |
| AMI | `dec_mean` | 12.1% | 61.1% | 27.8% |
| Earnings-22 | `ctc_mean` | 7.7% | 38.5% | 35.1% |
| Earnings-22 | `ctc_peak_min` | 8.2% | 43.2% | 36.8% |
| Earnings-22 | `dec_mean` | 8.7% | 44.0% | 35.4% |

Read as a probability of being right, `dec_mean` is close but a little high in the middle. On Earnings-22
test, words scoring ≥ 0.95 are right 96.6% of the time (6,231 words), 0.9–0.95: 83.9% (280), 0.8–0.9: 78.1%
(306), 0.7–0.8: 72.6% (230), 0.5–0.7: 58.7% (196), below 0.5: 38.7% (75). Per-bin numbers for every score
are in the JSON.

## Label noise

The labels are only as good as the references. Reading 60 randomly drawn wrong dev words (`--audit`)
under a first version of the labelling that only lowercased and dropped punctuation, by hand about 18
were right. Four were spelling and contraction variants (organisation, it's / it is, 'cause, gotta) that
the normaliser above now handles, two were compounds it doesn't handle (onto / on to, multiannual), and
the rest were reference typos or gaps ("parch" for patch, Sierra "Garda", phrases missing from the
reference). About 18 more were unclear: AMI crosstalk from another speaker, repeated words the reference
may have cleaned up, segment edges. The normaliser change raised every AUC by 0.003 to 0.07 and kept
the decoder scores ahead of the CTC scores and `ctc_peak_min` ahead of `ctc_mean`. The remaining noise
makes some right words count as wrong, so the AUCs above understate every score.
