# FLEURS in 13 languages, rumblex vs WhisperX, RTX 4090, 2026-11-22

Raw output of `scripts/runpod_languages.sh` at commit `ffd5d3e`, on a rented Runpod pod (RTX 4090 24 GB,
Secure Cloud, driver 595.112.07, 16 vCPUs). Each language is a fixed random sample of 350 utterances
(seed 0) from its FLEURS test set (`google/fleurs`, revision pinned in `rumblex.bench`). Settings as in the
[release run](../bench_2026-11-20_rtx4090_release/README.md): rumblex in bf16 at batch 32 with digits on,
WhisperX 3.8.6 with `large-v3` in float16 at batch 16 and its default wav2vec2 aligner for each language.
The language was given to both. Every reference and transcript is saved in the result JSONs.

WER is scored twice. "Whisper" is Whisper's English normalizer for English and its basic normalizer for the
rest, as in the README. "Leaderboard" is the Open ASR Leaderboard's multilingual normaliser and alignment
(`scripts/rescore_leaderboard.py`, open_asr_leaderboard at `c469619`; numbers written out with num2words),
which is the scoring the model card uses. English has no leaderboard rescore here.

| language | FLEURS set | rumblex WER (Whisper) | WhisperX WER (Whisper) | rumblex WER (leaderboard) | WhisperX WER (leaderboard) |
|---|---|---|---|---|---|
| English | en_us | 6.3% | 5.2% | | |
| German | de_de | 7.9% | 6.3% | 5.0% | 3.3% |
| Spanish | es_419 | 5.8% | 5.7% | 3.1% | 3.4% |
| French | fr_fr | 7.8% | 7.6% | 5.5% | 5.6% |
| Italian | it_it | 5.0% | 4.2% | 2.3% | 1.8% |
| Portuguese | pt_br | 6.3% | 5.4% | 4.0% | 3.3% |
| Dutch | nl_nl | 9.2% | 7.1% | 6.2% | 4.5% |
| Swedish | sv_se | 14.0% | 8.5% | 10.8% | 5.8% |
| Polish | pl_pl | 9.9% | 6.3% | 7.0% | 4.1% |
| Russian | ru_ru | 10.4% | 7.7% | 6.7% | 4.6% |
| Turkish | tr_tr | 10.2% | 7.4% | 6.3% | 3.8% |
| Arabic | ar_eg | 23.2% | 17.4% | 19.1% | 14.7% |
| Hindi | hi_in | 9.9% | 19.0% | 12.6% | 26.5% |

Speed and alignment, one 20-minute file per language made from that set's audio, models loaded:

| language | rumblex speed | WhisperX speed | rumblex align (s) | WhisperX align (s) | rumblex chunks aligned by CTC |
|---|---|---|---|---|---|
| English | 158× | 47× | 0.65 | 18.7 | 351 / 352 |
| German | 138× | 37× | 0.63 | 24.9 | 363 / 363 |
| Spanish | 134× | 42× | 0.79 | 20.2 | 355 / 355 |
| French | 128× | 41× | 0.74 | 20.9 | 352 / 352 |
| Italian | 151× | 46× | 0.68 | 18.5 | 387 / 387 |
| Portuguese | 134× | 44× | 0.75 | 19.9 | 359 / 359 |
| Dutch | 119× | 38× | 0.74 | 21.6 | 353 / 353 |
| Swedish | 129× | 35× | 0.69 | 26.2 | 353 / 353 |
| Polish | 103× | 33× | 0.86 | 26.7 | 351 / 351 |
| Russian | 119× | 44× | 0.77 | 18.4 | 354 / 354 |
| Turkish | 141× | 42× | 0.71 | 20.8 | 350 / 350 |
| Arabic | 126× | 37× | 0.81 | 21.3 | 359 / 359 |
| Hindi | 72× | 39× | 1.01 | 16.5 | 330 / 360 |

The chunk counts are from the 350-utterance accuracy pass (`accuracy.alignment`); a chunk that is not
aligned by CTC gets proportional word times.

Notes:

- WhisperX large-v3 has the lower WER in 9 of the 12 rescored languages. rumblex is ahead on Hindi by a wide
  margin under both scorings, and level with WhisperX on Spanish and French (within 0.3 points).
- Under the leaderboard scoring, rumblex's FLEURS numbers are close to the model card's (card: German 4.77,
  Spanish 2.82, French 5.95, Italian 2.53, Dutch 6.05, Portuguese 5.31; here 5.0, 3.1, 5.5, 2.3, 6.2, 4.0).
  These are 350-utterance samples, and the leaderboard scores against its own copy of FLEURS.
- Dutch reproduces the release run (9.2% vs 7.0%) and the leaderboard rescore of 2026-11-21 (6.24% vs 4.43%).
- Both systems ran 2.5–3× slower than in the release run (Dutch: 119× vs 380× for rumblex, 38× vs 79× for
  WhisperX) on what is nominally the same pod type, so absolute speeds depend on the host. The ratio between
  the two (2.7–3.7×, 1.9× on Hindi) is the comparable number. rumblex's alignment took 0.6–1.0 s per file,
  WhisperX's 16–27 s.
- Hindi: 30 of 360 chunks fell back to proportional times. Rumble-2-Flash's tokenizer gives about 10 tokens per
  second of Hindi audio (English 2.8, Swedish and Arabic 3.8, counted over the saved transcripts), close to
  the encoder's 13 frames per second, so in fast speech a CTC path doesn't fit and the chunk falls back.
  The mean CTC word score is also lowest for Hindi (0.20; 0.44–0.61 for the others). Word times in Hindi are
  less reliable than in the other languages.
- FLEURS is read speech, mostly clean. These numbers say nothing about conversational audio in these
  languages.
- Before this run, the pod's first start stalled on a package download and was restarted with the
  environments reused, and a 20-utterance check on Swedish and German ran on the same pod. Neither is part of
  these results; `session.log` covers this run only.

Package versions: `envs.txt`. GPU: `gpu.txt`. Combined table from `benchmark.py report`: `table.md`.
