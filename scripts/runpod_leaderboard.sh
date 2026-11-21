#!/usr/bin/env bash
# FLEURS Dutch for rumblex and WhisperX with every transcript saved, then both rescored with the Open ASR
# Leaderboard's normaliser and alignment, so the two can be compared under the scoring the model card uses.
# Settings as in runpod_release.sh. The speed file is kept short: speed is not the point of this run.
#
#   bash scripts/runpod_leaderboard.sh        # from the repo root on the pod
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=0  # some pod images turn it on without installing hf_transfer
mkdir -p results logs
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a logs/session.log; }
run() { local name=$1; shift; if "$@" > "logs/$name.log" 2>&1; then log "ok: $name"; else log "FAILED: $name"; tail -20 "logs/$name.log"; fi; }

command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee results/gpu.txt
git rev-parse --short HEAD | tee results/commit.txt

log "env"
uv venv -q -p 3.11 .venv && uv pip install -q -p .venv -e ".[dev]" whisper-normalizer regex kaldialign num2words
uv venv -q -p 3.11 .venv-wx && { uv pip install -q -p .venv-wx whisperx==3.8.6 whisper-normalizer jiwer pyarrow soundfile huggingface_hub \
  && uv pip install -q -p .venv-wx --no-deps -e . ; } || log "FAILED: whisperx install"
git clone -q https://github.com/huggingface/open_asr_leaderboard /tmp/open_asr_leaderboard

B="scripts/benchmark.py run --dataset fleurs_nl --n 364 --long-seconds 120 --save-transcripts"
run rumblex_fleurs_nl .venv/bin/python $B --system rumblex --device cuda:0 --compute-type bfloat16 \
  --batch-size 32 --max-chunk 20 --digits --out results/rumblex_fleurs_nl.json
run whisperx_fleurs_nl .venv-wx/bin/python $B --system whisperx --device cuda --compute-type float16 \
  --out results/whisperx_fleurs_nl.json

files=()
for f in results/rumblex_fleurs_nl.json results/whisperx_fleurs_nl.json; do [ -f "$f" ] && files+=("$f"); done
if [ ${#files[@]} -gt 0 ]; then
  .venv/bin/python scripts/benchmark.py report "${files[@]}" | tee results/table.md
  .venv/bin/python scripts/rescore_leaderboard.py --leaderboard /tmp/open_asr_leaderboard --language nl \
    --out results/leaderboard_rescore.json "${files[@]}" | tee -a results/table.md
fi
tar czf results.tar.gz results logs
log "done: results.tar.gz"
