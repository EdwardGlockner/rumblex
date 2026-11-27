# Word trust: does the CTC head know more than `probability`?

rumblex writes two scores per word: `probability` (the decoder's mean token probability, `dec_mean` below)
and `score` (the CTC head's `ctc_peak_min`). [confidence_eval.md](confidence_eval.md) found the decoder
score the better of the two. This asks two further questions. Does disagreement between the CTC head and
the decoder say which words are wrong, beyond what `probability` already says? And does the CTC head hear
words that the decoder left out?

Measured 2026-11-27 on CPU, float32, on the states saved for `confidence_eval.md` (same utterances, same
dev and test split, same labelling of words as right or wrong). `scripts/eval_word_trust.py features` wrote
the per-word features (gpushift job 421, 638 s) and `score` produced
[word_trust_eval.json](word_trust_eval.json). The JSON doesn't record the commit; it is the commit that
added this file.

## Setup

1,200 AMI utterances and 959 Earnings-22 utterances (the 241 with a digit in the reference are left out).
Even positions are dev, odd positions are test. Everything fitted or chosen is fitted on the two dev
splits pooled; test is reported once.

Features per transcribed word, next to the two existing scores:

| feature | what |
|---|---|
| `ctc_margin_min` | per token, the best frame of (log-prob of the token minus log-prob of the best other non-blank token); lowest token of the word |
| `ctc_top1` | share of the word's tokens that are the head's first choice on at least one of their frames |
| `ctc_gap` | how far the forced path's log-prob is below the best class, averaged over the word's frames |
| `ctc_word_match` | 1 when the word also appears in the greedy CTC text (the head's own transcript, aligned to the decoder's words), else 0 |

AUC is the chance that a random wrong word scores lower than a random right one (0.5 is chance).

## Wrong words

Each score alone:

| | words | wrong | `dec_mean` | `ctc_peak_min` | `ctc_margin_min` | `ctc_top1` | `ctc_gap` | `ctc_word_match` |
|---|---|---|---|---|---|---|---|---|
| AMI dev | 3,736 | 243 | 0.841 | 0.815 | 0.792 | 0.745 | 0.722 | 0.747 |
| Earnings-22 dev | 7,578 | 528 | 0.836 | 0.827 | 0.816 | 0.733 | 0.703 | 0.708 |
| AMI test | 3,689 | 203 | 0.882 | 0.856 | 0.841 | 0.790 | 0.753 | 0.797 |
| Earnings-22 test | 7,318 | 512 | 0.837 | 0.829 | 0.817 | 0.749 | 0.715 | 0.721 |

No CTC feature beats `dec_mean` on its own. A logistic regression on all five CTC features together
(`ctc_all` in the JSON) doesn't either: 0.861 on AMI test and 0.828 on Earnings-22 test.

`dec_mean` plus one feature, as a logistic regression. The last two columns are the gain over `dec_mean`
alone on test, with a 95% bootstrap interval over utterances (1,000 draws):

| added to `dec_mean` | AMI test | Earnings-22 test | gain, AMI | gain, Earnings-22 |
|---|---|---|---|---|
| nothing | 0.882 | 0.837 | | |
| `ctc_peak_min` | 0.887 | 0.848 | -0.007 to 0.017 | -0.001 to 0.021 |
| `ctc_margin_min` | 0.881 | 0.840 | -0.015 to 0.012 | -0.010 to 0.017 |
| `ctc_gap` | 0.861 | 0.804 | -0.037 to -0.007 | -0.047 to -0.019 |
| `ctc_word_match` | 0.902 | 0.855 | 0.008 to 0.032 | 0.008 to 0.027 |
| all five | 0.892 | 0.846 | -0.004 to 0.026 | -0.006 to 0.022 |

Only `ctc_word_match` helps on both test sets with an interval that stays above zero. It is the crudest
of the features: whether the head's own greedy transcript contains the word at all. The frame-level
features add nothing reliable, and `ctc_gap` makes things worse.

Flagging words below a threshold chosen to flag 10% of the pooled dev words, on test:

| test set | score | flagged | errors caught | flagged words that are wrong |
|---|---|---|---|---|
| AMI | `dec_mean` | 12.1% | 61.1% | 27.8% |
| AMI | `dec_mean` + `ctc_word_match` | 13.4% | 71.9% | 29.5% |
| Earnings-22 | `dec_mean` | 8.7% | 44.0% | 35.4% |
| Earnings-22 | `dec_mean` + `ctc_word_match` | 7.6% | 48.2% | 44.3% |

On Earnings-22 the combination flags fewer words, catches more of the errors, and more of what it flags
is wrong. On AMI it flags more words, so part of the extra errors caught comes from that.

## Words the decoder is sure of

Words with `dec_mean` of 0.95 or more are about 80% of AMI words and 85% of Earnings-22 words, and hold
a quarter to two fifths of all the errors. `probability` can't rank these much further, so this is where
a second opinion would be worth most.

| | words | wrong | share of all errors | `dec_mean` | `ctc_peak_min` | all five CTC features |
|---|---|---|---|---|---|---|
| AMI dev | 2,965 | 78 | 32.1% | 0.742 | 0.737 | 0.739 |
| Earnings-22 dev | 6,438 | 201 | 38.1% | 0.716 | 0.744 | 0.726 |
| AMI test | 2,916 | 47 | 23.2% | 0.799 | 0.797 | 0.803 |
| Earnings-22 test | 6,231 | 210 | 41.0% | 0.743 | 0.743 | 0.736 |

Nothing here. Inside this group the CTC features rank words about as well as `dec_mean` does and no
better; the one dev difference (`ctc_peak_min` on Earnings-22) is gone on test. Fitting the regression on
the sure words only doesn't change that (`ctc_all_fitted_on_sure` in the JSON).

## Deleted words

A place between two transcribed words is flagged when the greedy CTC text has a word there that the
decoder's text lacks. Against the places where the reference has a word the decoder left out:

| | places with a deletion | flagged | flagged that have one | deletions found |
|---|---|---|---|---|
| AMI dev | 168 of 4,523 | 56 | 17.9% | 6.0% |
| Earnings-22 dev | 178 of 8,245 | 141 | 17.0% | 13.5% |
| AMI test | 141 of 4,474 | 48 | 18.8% | 6.4% |
| Earnings-22 test | 180 of 7,941 | 113 | 12.4% | 7.8% |

The head finds few of the deleted words and most of what it flags is not a deletion. Requiring the extra
CTC word to be confident flags fewer places without becoming dependable: at 0.7 on test, 10 places are
flagged on AMI and none has a deletion, and 23 on Earnings-22 of which 26.1% do. The other thresholds are
in the JSON.

## What this says

- Whether the greedy CTC text contains the word is a small, real addition to `probability` for finding
  wrong words: about 0.02 AUC on both test sets.
- The finer CTC features don't add to it, and nothing separates right from wrong among the words the
  decoder is already sure of.
- The CTC head can't be used to find words the decoder dropped.

## Limits

- English only, two sets (meetings and earnings calls), 1,200 utterances each before the digit filter.
- The label noise described in `confidence_eval.md` applies here too: some words counted as wrong are
  right, which lowers every AUC.
- The flag rates of the two scores differ on test (same dev quantile, different test rate), so the
  flagging table is not a comparison at an equal number of flags.
