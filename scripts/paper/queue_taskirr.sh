#!/bin/zsh
# Task-irrelevant superclasses: ResNet-18 / CIFAR-100, 30 epochs, clean-run recipe, labels = index of the
# class within its superclass (5 outputs), with SAE / RDT analysis. Seeds 7, 42, 137, 256, 314, sequentially.
# Launch: nohup zsh scripts/paper/queue_taskirr.sh > logs/paper/taskirr.out 2>&1 &
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY=${PYTHON:-python}
TPL=${RDT_DATA:-/data/rdt}/core/experiments/noise_curriculum_arch/_config/template.json
export DEVTRAIN_MPS_MAX_BATCH=128
export PYTHONHASHSEED=0
say() { echo "[$(date -u +%FT%TZ)] QUEUE $*"; }
for S in 7 42 137 256 314; do
  say "taskirr seed $S START"
  if $PY -u $REPO/scripts/paper/run_taskirr.py --experiment-id taskirr --seed $S --epochs 30 \
      --every 5 --template-json $TPL --with-sae; then
    say "taskirr seed $S LANE_DONE"
  else
    say "taskirr seed $S FAILED rc=$?"
  fi
  say "taskirr seed $S END"
done
say "ALL DONE"
