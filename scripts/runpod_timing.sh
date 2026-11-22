#!/usr/bin/env bash
# LibriSpeech test_clean (WER + word timings + 1 h speed) for rumblex and WhisperX in one pod session, to
# re-measure word timings after an aligner change. Same settings as runpod_release.sh, without vLLM.
#
#   bash scripts/runpod_timing.sh        # from the repo root on the pod
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=0  # the PyTorch image sets it without installing hf_transfer
mkdir -p results logs
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a logs/session.log; }
run() { local name=$1; shift; if "$@" > "logs/$name.log" 2>&1; then log "ok: $name"; else log "FAILED: $name"; tail -20 "logs/$name.log"; fi; }

command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee results/gpu.txt
nproc | tee results/nproc.txt
git rev-parse --short HEAD | tee results/commit.txt

log "env"
uv venv -q -p 3.11 .venv && uv pip install -q -p .venv -e ".[dev]" whisper-normalizer
uv venv -q -p 3.11 .venv-wx && { uv pip install -q -p .venv-wx whisperx==3.8.6 whisper-normalizer jiwer pyarrow soundfile huggingface_hub \
  && uv pip install -q -p .venv-wx --no-deps -e . ; } || log "FAILED: whisperx install"
{ for v in .venv .venv-wx; do echo "## $v"; uv pip list -p $v 2>/dev/null | grep -iE '^(torch|transformers|whisperx|ctranslate2|faster-whisper) '; done; } > results/envs.txt

B="scripts/benchmark.py run --dataset librispeech --n 200 --long-seconds 3600 --long-utts 600"
log "dataset: librispeech"
run rumblex_librispeech .venv/bin/python $B --system rumblex --device cuda:0 --compute-type bfloat16 --batch-size 32 \
  --max-chunk 20 --out results/rumblex_librispeech.json
run whisperx_librispeech .venv-wx/bin/python $B --system whisperx --device cuda --compute-type float16 \
  --out results/whisperx_librispeech.json

{ echo "## librispeech"; .venv/bin/python scripts/benchmark.py report results/*_librispeech.json; } | tee results/table.md
tar czf results.tar.gz results logs
log "done: results.tar.gz"
