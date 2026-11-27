#!/usr/bin/env bash
# rumblex.align next to Qwen3-ForcedAligner-0.6B on 200 LibriSpeech test_clean utterances in one pod
# session, each in float32 and bfloat16. qwen-asr gets its own environment.
#
#   bash scripts/runpod_qwen_aligner.sh        # from the repo root on the pod
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
uv venv -q -p 3.11 .venv && uv pip install -q -p .venv -e ".[dev]"
uv venv -q -p 3.11 .venv-qwen && uv pip install -q -p .venv-qwen torch torchaudio qwen-asr
{ for v in .venv .venv-qwen; do echo "## $v"; uv pip list -p $v 2>/dev/null | grep -iE '^(torch|transformers|qwen-asr) '; done; } > results/envs.txt

D=data/qwen_fa
run prepare .venv/bin/python scripts/eval_qwen_aligner.py prepare --out $D
for dt in float32 bfloat16; do
  mkdir -p $D/$dt && cp $D/jobs.jsonl $D/$dt/
  run rumblex_$dt .venv/bin/python scripts/eval_qwen_aligner.py rumblex --dir $D/$dt --device cuda:0 --compute-type $dt
  run qwen_$dt .venv-qwen/bin/python scripts/qwen_aligner_worker.py $D/$dt/jobs.jsonl $D/$dt/qwen.json \
    --device cuda:0 --dtype $dt
  run score_$dt .venv/bin/python scripts/eval_qwen_aligner.py score --dir $D/$dt --out results/qwen_aligner_$dt.json
  cp $D/$dt/rumblex.json results/rumblex_words_$dt.json 2>/dev/null || true
  cp $D/$dt/qwen.json results/qwen_words_$dt.json 2>/dev/null || true
done

tar czf results.tar.gz results logs
log "done: results.tar.gz"
