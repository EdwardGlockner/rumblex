#!/usr/bin/env bash
# One-shot benchmark session on a rented Linux GPU pod (see docs/runpod_session.md).
#
#   bash scripts/runpod_bench.sh            # from the repo root on the pod
#
# Three separate environments, because vLLM and WhisperX each pin their own torch:
#   .venv        rumblex (transformers backend) + benchmark client
#   .venv-wx     WhisperX + rumblex installed without deps (for the shared benchmark code)
#   .venv-vllm   vLLM server only
# Results land in results/ and results.tar.gz. No Hugging Face token is needed: the model and
# LibriSpeech alignments are public, and diarization is not part of this run.
set -euo pipefail

N=${N:-200}                  # LibriSpeech test_clean utterances for WER / timestamps
LONG=${LONG:-3600}           # seconds of real speech for the speed run
VLLM_MEM=${VLLM_MEM:-0.5}    # vLLM's share of GPU memory; the rest is for the local encoder + CTC head
mkdir -p results logs

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a logs/session.log; }

command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee results/gpu.txt
git rev-parse --short HEAD | tee results/commit.txt

log "env: rumblex"
uv venv -q -p 3.11 .venv
uv pip install -q -p .venv -e ".[dev]" openai whisper-normalizer

log "env: whisperx"
uv venv -q -p 3.11 .venv-wx
{ uv pip install -q -p .venv-wx whisperx==3.8.6 whisper-normalizer jiwer pyarrow soundfile huggingface_hub \
  && uv pip install -q -p .venv-wx --no-deps -e . ; } || log "FAILED: whisperx install"

log "env: vllm"
uv venv -q -p 3.11 .venv-vllm
uv pip install -q -p .venv-vllm "vllm[audio]==0.34.0" || log "FAILED: vllm install"
for v in .venv .venv-wx .venv-vllm; do
  $v/bin/python -c "import torch, sys; print(sys.argv[1], torch.__version__, torch.version.cuda)" $v \
    | tee -a results/envs.txt || true
done
.venv-wx/bin/python -c "import importlib.metadata as m; print('whisperx', m.version('whisperx'))" \
  | tee -a results/envs.txt || true
.venv-vllm/bin/python -c "import importlib.metadata as m; print('vllm', m.version('vllm'))" \
  | tee -a results/envs.txt || true

BENCH="scripts/benchmark.py run --n $N --long-seconds $LONG --long-utts 600"
# A failing system is logged and skipped; the report covers whatever finished.
run() { local name=$1; shift; if "$@" > "logs/$name.log" 2>&1; then log "ok: $name"; else log "FAILED: $name (logs/$name.log)"; tail -20 "logs/$name.log"; fi; }

log "run: rumblex (transformers, bf16)"
run rumblex .venv/bin/python $BENCH --system rumblex --device cuda:0 --compute-type bfloat16 --out results/rumblex.json

log "run: whisperx (large-v3, float16)"
run whisperx .venv-wx/bin/python $BENCH --system whisperx --device cuda --compute-type float16 --out results/whisperx.json

log "start: vllm server"
.venv-vllm/bin/vllm serve Reson8/Rumble-2-Flash --max-model-len 4096 --gpu-memory-utilization "$VLLM_MEM" \
  > logs/vllm_server.log 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
for i in $(seq 1 180); do
  curl -sf http://localhost:8000/health >/dev/null && break
  kill -0 $VLLM_PID 2>/dev/null || { log "vllm server died, see logs/vllm_server.log"; tail -40 logs/vllm_server.log; break; }
  sleep 5
done
curl -sf http://localhost:8000/health >/dev/null || log "vllm server not up after 15 min; skipping rumblex-vllm"

log "run: rumblex-vllm"
curl -sf http://localhost:8000/health >/dev/null && run rumblex_vllm .venv/bin/python $BENCH --system rumblex-vllm --device cuda:0 --compute-type bfloat16 --out results/rumblex_vllm.json
kill $VLLM_PID 2>/dev/null || true

log "report"
files=()
for f in results/rumblex.json results/whisperx.json results/rumblex_vllm.json; do [ -f "$f" ] && files+=("$f"); done
if [ ${#files[@]} -gt 0 ]; then .venv/bin/python scripts/benchmark.py report "${files[@]}" | tee results/table.md; fi
tar czf results.tar.gz results logs
log "done: results.tar.gz"
