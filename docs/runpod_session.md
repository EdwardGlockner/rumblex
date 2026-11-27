# Running the benchmarks on a rented GPU

The comparisons in `docs/bench_*` ran on rented RTX 4090 pods (Runpod) with the `scripts/runpod_*.sh`
scripts. Each script does a whole session on the pod: it creates its own environments with `uv`, runs,
and packs everything into `results.tar.gz`. Nothing needs a Hugging Face token: the model, LibriSpeech
alignments, AMI, Earnings-22 and FLEURS are public.

| script | runs | about |
|---|---|---|
| `runpod_bench.sh` | rumblex, WhisperX and rumblex+vLLM on 200 LibriSpeech utterances + 1 h of speech | 7 min |
| `runpod_tune.sh` | batch size × chunk length sweep, vLLM settings sweep, then the full comparison | 12 min |
| `runpod_eval.sh` | rumblex vs WhisperX on LibriSpeech, AMI, Earnings-22 and FLEURS Dutch (+ vLLM on LibriSpeech) | 38 min |
| `runpod_dutch.sh` | FLEURS Dutch with digits on, plus the GPU VAD check | 9 min |
| `runpod_diagnose.sh` | FLEURS Dutch through VAD/language variants, transformers bf16/fp32 and vLLM, saving transcripts | 13 min |
| `runpod_release.sh` | rumblex 0.1.0 vs WhisperX on LibriSpeech, AMI, Earnings-22 and FLEURS Dutch (+ vLLM on LibriSpeech): the README's numbers | 18 min |
| `runpod_leaderboard.sh` | FLEURS Dutch for rumblex and WhisperX with every transcript saved, rescored with the Open ASR Leaderboard's scoring | 16 min |
| `runpod_timing.sh` | LibriSpeech (WER, word timings, 1 h speed) for rumblex and WhisperX, to re-measure timings after an aligner change | 6 min |
| `runpod_parakeet.sh` | LibriSpeech (WER, word timings, 1 h speed) for rumblex and Parakeet TDT 0.6b v3 through NeMo and transformers, in bfloat16 and float32 | 5 min without the NeMo runs |
| `runpod_qwen_aligner.sh` | `rumblex.align` and Qwen3-ForcedAligner-0.6B on 200 LibriSpeech utterances, in float32 and bfloat16 | 2 min |
| `runpod_languages.sh` | FLEURS in 13 languages (350 utt. each, WER under Whisper's and the leaderboard's scoring, 20 min speed file) for rumblex and WhisperX; `LANGS`, `N`, `SKIP_ENV` to narrow or resume it | 2 h 40 min, plus setup |
| `runpod_logprobs.sh` | decoder log-probs through vLLM vs the transformers backend on 50 AMI/Earnings-22 utterances | 4 min on an A40 |

Times are from the session logs in `docs/bench_*` (RTX 4090, from environment setup to the last result).

## Pod

- One NVIDIA GPU with at least 24 GB (vLLM reserves part of the memory up front).
- A host driver that supports CUDA 13.0: the vLLM environment gets torch built for CUDA 13.0, the other
  two CUDA 12.6 and 12.8 (`bench_2026-11-20_rtx4090/envs.txt`). The runs used driver 595.112.07 on
  Runpod's PyTorch 2.8 template.
- 80 GB of container disk: three virtual environments with torch, plus the model, Whisper large-v3, the
  wav2vec2 alignment models and the datasets. No network volume is needed.
- SSH access only.

## Session

```bash
# on the pod
cd /root && git clone https://github.com/EdwardGlockner/rumblex && cd rumblex && git checkout v0.1.0
nohup bash scripts/runpod_release.sh > session.out 2>&1 &
tail -f logs/session.log

# local, when the log says "done: results.tar.gz"
scp -P <port> root@<pod-ip>:/root/rumblex/results.tar.gz .
```

Then terminate the pod. A stopped pod still bills for its disk.

If a system fails (e.g. a WhisperX install problem), the scripts log it and go on with the rest. Each
result JSON from `benchmark.py` records the GPU, torch version, git commit and settings. The runs in
`docs/bench_*` other than `bench_2026-11-20_rtx4090_release` predate v0.1.0; see the provenance note in
[docs/README.md](README.md).
