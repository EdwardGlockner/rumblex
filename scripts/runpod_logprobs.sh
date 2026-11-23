#!/usr/bin/env bash
# Decoder log-probs through vLLM vs the transformers backend, in one pod session
# (scripts/check_logprobs_vllm.py). Same vLLM version and server settings as runpod_release.sh.
#
#   bash scripts/runpod_logprobs.sh        # from the repo root on the pod
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=0  # the PyTorch image sets it without installing hf_transfer
# the image keeps these caches on /workspace, which may be a small volume; the container disk has room
export UV_CACHE_DIR=/root/.cache/uv HF_HOME=/root/.cache/huggingface
mkdir -p results logs
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a logs/session.log; }
run() { local name=$1; shift; if "$@" > "logs/$name.log" 2>&1; then log "ok: $name"; else log "FAILED: $name"; tail -20 "logs/$name.log"; fi; }

command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee results/gpu.txt
git rev-parse --short HEAD | tee results/commit.txt

log "env"
# an environment that is already there (a resumed session) is kept
[ -x .venv/bin/python ] || { uv venv -q -p 3.11 .venv && uv pip install -q -p .venv -e ".[dev,vllm]" whisper-normalizer; }
[ -x .venv-vllm/bin/vllm ] || { rm -rf .venv-vllm && uv venv -q -p 3.11 .venv-vllm   && uv pip install -q -p .venv-vllm "vllm[audio]==0.34.0"; } || log "FAILED: vllm install"
[ -x .venv-vllm/bin/vllm ] || { log "no vllm; stopping"; exit 1; }
{ for v in .venv .venv-vllm; do echo "## $v"; uv pip list -p $v 2>/dev/null | grep -iE '^(torch|transformers|vllm|openai) '; done; } > results/envs.txt

log "start: vllm server"
.venv-vllm/bin/vllm serve Reson8/Rumble-2-Flash --max-model-len 4096 --gpu-memory-utilization 0.5 > logs/vllm_server.log 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
for i in $(seq 1 180); do curl -sf http://localhost:8000/health >/dev/null && break; sleep 5; done
if curl -sf http://localhost:8000/health >/dev/null; then
  run check_logprobs .venv/bin/python scripts/check_logprobs_vllm.py --url http://localhost:8000/v1 --device cuda:0 \
    --compute-type bfloat16 --n 25 --out results/logprobs_vllm.json
  cp logs/check_logprobs.log results/ || true
else
  log "vllm server not up; skipping"
fi
kill $VLLM_PID 2>/dev/null || true

tar czf results.tar.gz results logs
log "done: results.tar.gz"
