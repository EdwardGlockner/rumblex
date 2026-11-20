#!/usr/bin/env bash
# Dutch WER diagnosis on a rented GPU pod: VAD on/off x language forced/detected, through transformers in
# bf16 and float32 and through a vLLM server (closest to the model card's Open ASR Leaderboard setup).
# Every transcript is saved. See scripts/diagnose_wer.py.
#
#   bash scripts/runpod_diagnose.sh        # from the repo root on the pod
set -euo pipefail
mkdir -p results logs
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a logs/session.log; }
run() { local name=$1; shift; if "$@" > "logs/$name.log" 2>&1; then log "ok: $name"; else log "FAILED: $name"; tail -20 "logs/$name.log"; fi; }

command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee results/gpu.txt
git rev-parse --short HEAD | tee results/commit.txt

log "env"
uv venv -q -p 3.11 .venv && uv pip install -q -p .venv -e ".[dev]" openai whisper-normalizer
uv venv -q -p 3.11 .venv-vllm && uv pip install -q -p .venv-vllm "vllm[audio]==0.34.0" || log "FAILED: vllm install"

D="scripts/diagnose_wer.py --dataset fleurs_nl --n 364 --device cuda:0"
run diag_bf16 .venv/bin/python $D --dtype bfloat16 --batch-size 32 --out results/diag_transformers_bf16.json
run diag_fp32 .venv/bin/python $D --dtype float32 --batch-size 16 --out results/diag_transformers_fp32.json

log "start: vllm server"
.venv-vllm/bin/vllm serve Reson8/Rumble-2-Flash --max-model-len 4096 --gpu-memory-utilization 0.6 > logs/vllm_server.log 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
for i in $(seq 1 180); do curl -sf http://localhost:8000/health >/dev/null && break; sleep 5; done
if curl -sf http://localhost:8000/health >/dev/null; then
  run diag_vllm .venv/bin/python $D --backend vllm --dtype bfloat16 --out results/diag_vllm.json
else
  log "vllm server not up; skipping"
fi
kill $VLLM_PID 2>/dev/null || true
tar czf results.tar.gz results logs
log "done: results.tar.gz"
