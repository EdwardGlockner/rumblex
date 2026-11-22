#!/usr/bin/env bash
# FLEURS test sets in 13 languages, rumblex vs WhisperX large-v3, with release settings (runpod_release.sh):
# WER with Whisper's normalisers, transcripts saved and rescored with the Open ASR Leaderboard's scoring,
# rumblex's alignment counters (fallback chunks, mean word score), and speed on a 20-minute file per language.
#
#   bash scripts/runpod_languages.sh          # from the repo root on the pod
#   LANGS="sv de" N=50 bash scripts/runpod_languages.sh   # a subset, e.g. to check the setup
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=0  # some pod images turn it on without installing hf_transfer
LANGS=${LANGS:-"en de es fr it pt nl sv pl ru tr ar hi"}
N=${N:-350}
mkdir -p results logs
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a logs/session.log; }
run() { local name=$1; shift; if "$@" > "logs/$name.log" 2>&1; then log "ok: $name"; else log "FAILED: $name"; tail -20 "logs/$name.log"; fi; }

command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee results/gpu.txt
git rev-parse --short HEAD | tee results/commit.txt

export UV_HTTP_TIMEOUT=120  # a stalled download fails and is retried instead of hanging
if [ -z "${SKIP_ENV:-}" ]; then  # SKIP_ENV=1 reuses .venv and .venv-wx from an earlier start
  log "env"
  uv venv -q -p 3.11 .venv && uv pip install -q -p .venv -e ".[dev]" whisper-normalizer regex kaldialign num2words
  uv venv -q -p 3.11 .venv-wx && { uv pip install -q -p .venv-wx whisperx==3.8.6 whisper-normalizer jiwer pyarrow soundfile huggingface_hub \
    && uv pip install -q -p .venv-wx --no-deps -e . ; } || log "FAILED: whisperx install"
fi
{ for v in .venv .venv-wx; do echo "## $v"; uv pip list -p $v 2>/dev/null | grep -iE '^(torch|transformers|whisperx|ctranslate2|faster-whisper) '; done; } > results/envs.txt
[ -d /tmp/open_asr_leaderboard ] || git clone -q https://github.com/huggingface/open_asr_leaderboard /tmp/open_asr_leaderboard

for lang in $LANGS; do
  B="scripts/benchmark.py run --dataset fleurs_$lang --n $N --long-seconds 1200 --save-transcripts"
  run "rumblex_fleurs_$lang" .venv/bin/python $B --system rumblex --device cuda:0 --compute-type bfloat16 \
    --batch-size 32 --max-chunk 20 --digits --out "results/rumblex_fleurs_$lang.json"
  run "whisperx_fleurs_$lang" .venv-wx/bin/python $B --system whisperx --device cuda --compute-type float16 \
    --out "results/whisperx_fleurs_$lang.json"
  files=()
  for f in "results/rumblex_fleurs_$lang.json" "results/whisperx_fleurs_$lang.json"; do [ -f "$f" ] && files+=("$f"); done
  if [ ${#files[@]} -gt 0 ] && [ "$lang" != en ]; then
    run "rescore_$lang" .venv/bin/python scripts/rescore_leaderboard.py --leaderboard /tmp/open_asr_leaderboard \
      --language "$lang" --out "results/leaderboard_$lang.json" "${files[@]}"
  fi
  tar czf results.tar.gz results logs  # after every language, so a stopped pod still leaves results
done

shopt -s nullglob
all=(results/rumblex_fleurs_*.json results/whisperx_fleurs_*.json)
[ ${#all[@]} -gt 0 ] && .venv/bin/python scripts/benchmark.py report "${all[@]}" > results/table.md
tar czf results.tar.gz results logs
log "done: results.tar.gz"
