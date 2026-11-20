#!/usr/bin/env bash
# Throughput tuning + refreshed comparison on a rented GPU pod (see docs/runpod_session.md).
#
#   bash scripts/runpod_tune.sh        # from the repo root on the pod
#
# 1. profile_speed.py: transformers backend, bf16, batch size x max chunk length, with WER per setting.
# 2. vLLM: rumblex-vllm speed runs over client concurrency x local encoder batch size x chunk length.
# 3. Full comparison (accuracy + 1 h speed) with the best settings: rumblex, rumblex-vllm and WhisperX.
# "Best" = fastest setting whose WER is within 0.2 percentage points of the lowest WER seen in that sweep.
set -euo pipefail
mkdir -p results logs
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a logs/session.log; }
run() { local name=$1; shift; if "$@" > "logs/$name.log" 2>&1; then log "ok: $name"; else log "FAILED: $name"; tail -20 "logs/$name.log"; fi; }

command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee results/gpu.txt
git rev-parse --short HEAD | tee results/commit.txt

log "env"
uv venv -q -p 3.11 .venv && uv pip install -q -p .venv -e ".[dev]" openai whisper-normalizer
uv venv -q -p 3.11 .venv-wx && { uv pip install -q -p .venv-wx whisperx==3.8.6 whisper-normalizer jiwer pyarrow soundfile huggingface_hub \
  && uv pip install -q -p .venv-wx --no-deps -e . ; } || log "FAILED: whisperx install"
uv venv -q -p 3.11 .venv-vllm && uv pip install -q -p .venv-vllm "vllm[audio]==0.34.0" || log "FAILED: vllm install"

# 1. transformers backend sweep
log "sweep: transformers"
run profile .venv/bin/python scripts/profile_speed.py --device cuda:0 --dtype bfloat16 --seconds 1200 \
  --chunks 20 30 --batches 8 16 32 64 --out results/profile.json
read -r BS CHUNK < <(.venv/bin/python - <<'EOF'
import json
runs = [r for r in json.load(open("results/profile.json"))["runs"] if not r.get("oom")]
best_wer = min(r["wer"] for r in runs)
ok = [r for r in runs if r["wer"] <= best_wer + 0.002]
b = max(ok, key=lambda r: r["rtfx_model"])
print(b["batch"], b["chunk_s"])
EOF
) || { BS=8; CHUNK=20; }
log "chosen transformers: batch $BS, chunk $CHUNK"

# 2. vLLM sweep (speed only)
log "start: vllm server"
.venv-vllm/bin/vllm serve Reson8/Rumble-2-Flash --max-model-len 4096 --gpu-memory-utilization 0.5 > logs/vllm_server.log 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
for i in $(seq 1 180); do curl -sf http://localhost:8000/health >/dev/null && break; sleep 5; done
SWEEP="scripts/benchmark.py run --system rumblex-vllm --device cuda:0 --compute-type bfloat16 --speed-only --n 1 --long-seconds 1200 --long-utts 250"
if curl -sf http://localhost:8000/health >/dev/null; then
  for conc in 32 128; do for ebs in 8 32; do for ch in 20 30; do
    run "vllm_c${conc}_b${ebs}_ch${ch}" .venv/bin/python $SWEEP --vllm-concurrency $conc --batch-size $ebs --max-chunk $ch \
      --out "results/sweep_vllm_c${conc}_b${ebs}_ch${ch}.json"
  done; done; done
  read -r VC VB VCH < <(.venv/bin/python - <<'EOF'
import glob, json
rows = []
for f in glob.glob("results/sweep_vllm_*.json"):
    r = json.load(open(f)); s = r["settings"]
    rows.append((r["speed"]["rtfx"], s["concurrency"], s["batch_size"], s["max_chunk_s"]))
best = max(rows)
print(best[1], best[2], int(best[3]))
EOF
  ) || { VC=32; VB=8; VCH=20; }
  log "chosen vllm: concurrency $VC, encoder batch $VB, chunk $VCH"
  # 3a. full comparison, vLLM
  run rumblex_vllm .venv/bin/python scripts/benchmark.py run --system rumblex-vllm --device cuda:0 --compute-type bfloat16 \
    --vllm-concurrency "$VC" --batch-size "$VB" --max-chunk "$VCH" --long-utts 600 --out results/rumblex_vllm.json
else
  log "vllm server not up; skipping vLLM"
fi
kill $VLLM_PID 2>/dev/null || true
sleep 5

# 3b. full comparison, transformers + WhisperX
run rumblex .venv/bin/python scripts/benchmark.py run --system rumblex --device cuda:0 --compute-type bfloat16 \
  --batch-size "$BS" --max-chunk "$CHUNK" --long-utts 600 --out results/rumblex.json
run whisperx .venv-wx/bin/python scripts/benchmark.py run --system whisperx --device cuda --compute-type float16 \
  --long-utts 600 --out results/whisperx.json

files=()
for f in results/rumblex.json results/rumblex_vllm.json results/whisperx.json; do [ -f "$f" ] && files+=("$f"); done
if [ ${#files[@]} -gt 0 ]; then .venv/bin/python scripts/benchmark.py report "${files[@]}" | tee results/table.md; fi
tar czf results.tar.gz results logs
log "done: results.tar.gz"
