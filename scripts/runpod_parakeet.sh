#!/usr/bin/env bash
# LibriSpeech test_clean (WER + word timings + 1 h speed) for rumblex and Parakeet TDT 0.6b v3 in one pod
# session. Both run in the same environment and get the same chunks; each runs in bfloat16 and float32.
#
#   bash scripts/runpod_parakeet.sh        # from the repo root on the pod
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=0  # the PyTorch image sets it without installing hf_transfer
export TQDM_DISABLE=1
mkdir -p results logs
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a logs/session.log; }
run() { local name=$1; shift; if "$@" > "logs/$name.log" 2>&1; then log "ok: $name"; else log "FAILED: $name"; tail -20 "logs/$name.log"; fi; }

command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee results/gpu.txt
nproc | tee results/nproc.txt
git rev-parse --short HEAD | tee results/commit.txt

log "env"
uv venv -q -p 3.11 .venv && uv pip install -q -p .venv -e ".[dev]" librosa
uv pip list -p .venv 2>/dev/null | grep -iE '^(torch|torchaudio|transformers|librosa) ' > results/envs.txt

B="scripts/benchmark.py run --dataset librispeech --n 200 --long-seconds 3600 --long-utts 600 --save-transcripts"
for dt in bfloat16 float32; do
  run rumblex_$dt .venv/bin/python $B --system rumblex --device cuda:0 --compute-type $dt --batch-size 32 \
    --max-chunk 20 --out results/rumblex_${dt}_librispeech.json
  run parakeet_$dt .venv/bin/python $B --system parakeet --device cuda:0 --compute-type $dt --batch-size 32 \
    --max-chunk 20 --out results/parakeet_${dt}_librispeech.json
done

{ echo "## librispeech"; .venv/bin/python scripts/benchmark.py report results/*_librispeech.json; } | tee results/table.md
tar czf results.tar.gz results logs
log "done: results.tar.gz"
