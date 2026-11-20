#!/usr/bin/env bash
# FLEURS Dutch re-run with rumblex writing numbers as digits (the model's default), plus WhisperX on the
# same pod for a like-for-like speed number. Also checks the batched VAD on GPU vs CPU first.
# See docs/bench_2026-11-20_rtx4090_eval/README.md for why.
#
#   bash scripts/runpod_dutch.sh        # from the repo root on the pod
set -euo pipefail
mkdir -p results logs
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a logs/session.log; }
run() { local name=$1; shift; if "$@" > "logs/$name.log" 2>&1; then log "ok: $name"; else log "FAILED: $name"; tail -20 "logs/$name.log"; fi; }

command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee results/gpu.txt
git rev-parse --short HEAD | tee results/commit.txt

log "env"
uv venv -q -p 3.11 .venv && uv pip install -q -p .venv -e ".[dev]" whisper-normalizer
uv venv -q -p 3.11 .venv-wx && { uv pip install -q -p .venv-wx whisperx==3.8.6 whisper-normalizer jiwer pyarrow soundfile huggingface_hub \
  && uv pip install -q -p .venv-wx --no-deps -e . ; } || log "FAILED: whisperx install"

run vad_gpu .venv/bin/python scripts/check_vad_gpu.py --device cuda:0 --seconds 1200 --out results/vad_gpu.json

B="scripts/benchmark.py run --dataset fleurs_nl --n 364 --long-seconds 1200"
run rumblex_fleurs_nl_digits .venv/bin/python $B --system rumblex --device cuda:0 --compute-type bfloat16 \
  --batch-size 32 --max-chunk 20 --digits --out results/rumblex_fleurs_nl_digits.json
run whisperx_fleurs_nl .venv-wx/bin/python $B --system whisperx --device cuda --compute-type float16 \
  --out results/whisperx_fleurs_nl.json

files=()
for f in results/rumblex_fleurs_nl_digits.json results/whisperx_fleurs_nl.json; do [ -f "$f" ] && files+=("$f"); done
if [ ${#files[@]} -gt 0 ]; then .venv/bin/python scripts/benchmark.py report "${files[@]}" | tee results/table.md; fi
tar czf results.tar.gz results logs
log "done: results.tar.gz"
