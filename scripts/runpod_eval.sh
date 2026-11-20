#!/usr/bin/env bash
# Multi-dataset comparison on a rented GPU pod (see docs/runpod_session.md):
#   librispeech (WER + word timings + 1 h speed; also rumblex-vllm)  -- measures the batched VAD end to end
#   ami, earnings22 (English meetings / earnings calls) and fleurs_nl (Dutch): WER + speed, rumblex vs WhisperX
#
#   bash scripts/runpod_eval.sh        # from the repo root on the pod
set -euo pipefail
mkdir -p results logs
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a logs/session.log; }
run() { local name=$1; shift; if "$@" > "logs/$name.log" 2>&1; then log "ok: $name"; else log "FAILED: $name"; tail -20 "logs/$name.log"; fi; }

command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee results/gpu.txt
nproc | tee results/nproc.txt
git rev-parse --short HEAD | tee results/commit.txt

log "env"
uv venv -q -p 3.11 .venv && uv pip install -q -p .venv -e ".[dev]" openai whisper-normalizer
uv venv -q -p 3.11 .venv-wx && { uv pip install -q -p .venv-wx whisperx==3.8.6 whisper-normalizer jiwer pyarrow soundfile huggingface_hub \
  && uv pip install -q -p .venv-wx --no-deps -e . ; } || log "FAILED: whisperx install"
uv venv -q -p 3.11 .venv-vllm && uv pip install -q -p .venv-vllm "vllm[audio]==0.34.0" || log "FAILED: vllm install"

RX="--system rumblex --device cuda:0 --compute-type bfloat16 --batch-size 32 --max-chunk 20"
WX="--system whisperx --device cuda --compute-type float16"
B="scripts/benchmark.py run"

# name  dataset  n  long-seconds
for spec in "librispeech librispeech 200 3600" "ami ami 500 1800" "earnings22 earnings22 500 1800" "fleurs_nl fleurs_nl 364 1200"; do
  set -- $spec
  name=$1; ds=$2; n=$3; long=$4
  extra=""; [ "$ds" = "librispeech" ] && extra="--long-utts 600"
  log "dataset: $name"
  run "rumblex_$name" .venv/bin/python $B $RX --dataset "$ds" --n "$n" --long-seconds "$long" $extra --out "results/rumblex_$name.json"
  run "whisperx_$name" .venv-wx/bin/python $B $WX --dataset "$ds" --n "$n" --long-seconds "$long" $extra --out "results/whisperx_$name.json"
done

log "start: vllm server"
.venv-vllm/bin/vllm serve Reson8/Rumble-2-Flash --max-model-len 4096 --gpu-memory-utilization 0.5 > logs/vllm_server.log 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
for i in $(seq 1 180); do curl -sf http://localhost:8000/health >/dev/null && break; sleep 5; done
if curl -sf http://localhost:8000/health >/dev/null; then
  run rumblex_vllm_librispeech .venv/bin/python $B --system rumblex-vllm --device cuda:0 --compute-type bfloat16 \
    --batch-size 8 --max-chunk 20 --dataset librispeech --n 200 --long-seconds 3600 --long-utts 600 \
    --out results/rumblex_vllm_librispeech.json
else
  log "vllm server not up; skipping"
fi
kill $VLLM_PID 2>/dev/null || true

for name in librispeech ami earnings22 fleurs_nl; do
  files=()
  for f in "results/rumblex_$name.json" "results/rumblex_vllm_$name.json" "results/whisperx_$name.json"; do
    [ -f "$f" ] && files+=("$f")
  done
  if [ ${#files[@]} -gt 0 ]; then
    { echo "## $name"; .venv/bin/python scripts/benchmark.py report "${files[@]}"; echo; } | tee -a results/table.md
  fi
done
tar czf results.tar.gz results logs
log "done: results.tar.gz"
