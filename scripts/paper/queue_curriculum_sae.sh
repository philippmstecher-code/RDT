#!/bin/zsh
# SAE / RDT analysis on all finished curriculum lanes; runs after queue_curriculum_noise.sh (controlled curriculum
# and static label noise) finishes.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY=${PYTHON:-python}
TPL=${RDT_DATA:-/data/rdt}/core/experiments/noise_curriculum_arch/_config/template.json
export DEVTRAIN_MPS_MAX_BATCH=128
say() { echo "[$(date -u +%FT%TZ)] QUEUE $*"; }
say "runs after queue_curriculum_noise.sh (curriculum / static noise)"
while pgrep -f queue_curriculum_noise.sh >/dev/null; do sleep 120; done
say "curriculum / static noise finished; starting curriculum SAE analysis"
$PY -u $REPO/scripts/paper/run_curriculum_sae.py --experiment-id curriculum --template-json $TPL \
  || say "curriculum SAE FAILED"
say "SAE ALL DONE"
