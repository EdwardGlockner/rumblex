# rumblex and Parakeet TDT 0.6b v3 on LibriSpeech, RTX 4090, 2026-11-27

rumblex 0.6.1 next to `nvidia/parakeet-tdt-0.6b-v3` on a rented Runpod pod (RTX 4090 24 GB, Secure Cloud,
driver 595.112.07, 32 vCPUs). Same data and scoring as the other LibriSpeech runs here: 200 test_clean
utterances (seed 0) for WER and word times against the Montreal Forced Aligner reference alignments, and
one hour of concatenated test_clean speech for speed and peak GPU memory.

Parakeet was run two ways: through NeMo 3.0.0, the toolkit it was released with, and through
transformers 5.24.1 (`ParakeetForTDT`) in rumblex's own environment. Every system gets the same chunks
(rumblex's VAD chunking, at most 20 s) at batch size 32, and each ran in bfloat16 and in float32. Parakeet's
word times are its own: each token starts on the frame it was emitted on and lasts its predicted duration.

| | rumblex | Parakeet, NeMo | Parakeet, transformers |
|---|---|---|---|
| WER | 2.2% | 1.7% | 1.8% |
| word start MAE | 21.0 ms | 67.5 ms | 67.3 ms |
| word start, median error | 13 ms | 50 ms | 50 ms |
| starts ≤ 50 ms | 94.0% | 50.5% | 50.7% |
| starts ≤ 100 ms | 97.9% | 77.8% | 77.9% |
| word end MAE | 31.7 ms | 74.8 ms | 74.8 ms |
| word end, median error | 17 ms | 60 ms | 60 ms |
| ends ≤ 50 ms | 86.1% | 47.3% | 47.4% |
| ends ≤ 100 ms | 93.5% | 74.4% | 74.4% |
| 1 h file, times real time | 462× | 1,364× | 83× |
| 1 h file, seconds | 7.8 | 2.6 | 43.6 |
| peak GPU memory | 9.0 GB | 7.6 GB | 3.9 GB |

These are the bfloat16 runs. In float32:

| | rumblex | Parakeet, NeMo | Parakeet, transformers |
|---|---|---|---|
| WER | 2.1% | 1.8% | 1.8% |
| word start MAE | 20.9 ms | 67.3 ms | 67.4 ms |
| word end MAE | 31.8 ms | 74.7 ms | 74.8 ms |
| 1 h file, times real time | 387× | 931× | 80× |
| peak GPU memory | 15.7 GB | 11.6 GB | 7.5 GB |

What it says:

- Parakeet has the lower WER on read English: 71 to 74 word errors of 4,099 against rumblex's 88 to 90.
- rumblex's word times are closer to the reference: about a third of Parakeet's start error and under half
  of its end error. The two Parakeet implementations agree to within 0.2 ms, so this is the model's
  timing, not a port's. Parakeet's times move in 80 ms steps (its frame), and rumblex's encoder has the
  same frame length but corrects boundaries inside the frame ([boundary_head.md](../boundary_head.md)).
- Parakeet through NeMo is the fastest: about three times rumblex's speed in bfloat16 and 2.4 times in
  float32. Through transformers it is about five times slower than rumblex; that port decodes much more
  slowly than NeMo does, and its number should not be read as Parakeet's speed.
- rumblex's time includes the CTC alignment (1.1 s of the 7.8 s) and the `ctc_match` pass.

Limits:

- One set, read English, 200 utterances. No conversational speech, no other language.
- The speed is one pass over one file per run, with no warm-up pass and no repeat. The NeMo runs take
  under 4 s, so their speed figures are the least stable ones here.
- NeMo ran with torch 2.15.0, the other two with torch 2.14.1+cu126 (`envs.txt`).
- bfloat16 for NeMo means `torch.autocast` around `transcribe`; for the other two the weights are loaded
  in bfloat16.
- The reference times are MFA's, not hand labels.
- Chunking by rumblex's VAD is not how NeMo would cut a long file itself.

## The same on a GTX 1070

The same data and script on a local GTX 1070 (8 GB), float32, 2026-11-28, at commit `30641e9`, each
system with the machine to itself. NeMo is not installed here, so Parakeet ran through the transformers
port only. Batch size was 8 for rumblex and 16 for Parakeet instead of 32.

| | rumblex | Parakeet, transformers |
|---|---|---|
| WER | 2.1% | 1.8% |
| word start MAE | 20.9 ms | 67.4 ms |
| starts ≤ 100 ms | 98.0% | 77.8% |
| word end MAE | 31.8 ms | 74.8 ms |
| ends ≤ 100 ms | 93.5% | 74.4% |
| 1 h file, times real time | 39× | 30× |
| 1 h file, seconds | 91.6 | 120.6 |
| peak GPU memory | 6.5 GB | 6.1 GB |

WER and word times are the same as in the float32 runs on the RTX 4090. The speed column compares rumblex
with the slow port, as in the third column above; on the 4090 NeMo was about 12 times faster in float32 than the
port, so this says nothing about how fast Parakeet can run on this card. rumblex's 91.6 s include 13.4 s
of alignment. Files: `rumblex_gtx1070_float32_librispeech.json` and
`parakeet_gtx1070_float32_librispeech.json`.

## Files

Raw output of `scripts/runpod_parakeet.sh`: one JSON per run, `table.md`, `envs.txt`, `session.log`. The
rumblex and transformers runs are at commit `0169240`. The two NeMo runs were added in the same pod session
afterwards, with the `parakeet-nemo` system of the next commit copied to the pod and the same arguments;
their JSON still records `0169240`, and they are not in `session.log`. The script now includes them.
