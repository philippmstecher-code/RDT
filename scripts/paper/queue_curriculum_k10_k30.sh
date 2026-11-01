#!/bin/zsh
# Curriculum lanes K = 30 and K = 10 (sc_true, sc_rand, matched scratch) for seeds 7, 42, 137, 256, 314,
# with SAE. Waits for queue_curriculum_sae.sh first; K = 30 runs for all seeds, then K = 10.
# Launch: nohup zsh scripts/paper/queue_curriculum_k10_k30.sh > logs/paper/curriculum_k10_k30.out 2>&1 &
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY=${PYTHON:-python}
TPL=${RDT_DATA:-/data/rdt}/core/experiments/noise_curriculum_arch/_config/template.json
export DEVTRAIN_MPS_MAX_BATCH=128
say() { echo "[$(date -u +%FT%TZ)] QUEUE $*"; }
say "waiting for queue_curriculum_sae.sh to finish"
while pgrep -f queue_curriculum_sae.sh >/dev/null; do sleep 120; done
for K in 30 10; do
  for S in 7 42 137 256 314; do
    say "K=$K seed $S START"
    $PY -u $REPO/scripts/paper/run_curriculum.py --experiment-id curriculum --seed $S --K $K --epochs 50 \
      --arms sc_true,sc_rand,scratch --template-json $TPL --with-sae \
      || say "K=$K seed $S FAILED"
    say "K=$K seed $S END"
  done
done
say "ALL DONE"
