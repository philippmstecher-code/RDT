#!/bin/zsh
# Runs after run_prevention.sh finishes, as two parallel streams: controlled curriculum (K = 20) and
# static label noise (30 epochs, p = 0.3), seeds 42 137 7 256 314.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
WT=${RDT_DATA:-/data/rdt}/checkouts/lesion
PY=${PYTHON:-python}
TPL=${RDT_DATA:-/data/rdt}/core/experiments/noise_curriculum_arch/_config/template.json
export DEVTRAIN_MPS_MAX_BATCH=128
SEEDS=(42 137 7 256 314)
say() { echo "[$(date -u +%FT%TZ)] QUEUE $*"; }
say "runs after run_prevention.sh"
while pgrep -f run_prevention.sh >/dev/null; do sleep 120; done
say "prevention finished; starting curriculum and static noise"
(
  for S in $SEEDS; do
    say "curriculum seed $S START"
    $PY -u $REPO/scripts/paper/run_curriculum.py --experiment-id curriculum --seed $S --K 20 --epochs 50 \
       --arms sc_true,sc_rand,scratch --template-json $TPL || say "curriculum seed $S FAILED"
    say "curriculum seed $S END"
  done
) > $REPO/logs/paper/curriculum.out 2>&1 &
(
  cd $WT
  for S in $SEEDS; do
    say "static noise seed $S START"
    $PY -u scripts/run_noise_curriculum_experiment.py --experiment-id static_noise_s$S --arms between_sc,within_sc \
       --switch-epochs "" --archs resnet18 --epochs 30 --noise-prob 0.3 --seed $S --skip-sae --template-json $TPL || say "static noise seed $S FAILED"
    say "static noise seed $S END"
  done
) > $REPO/logs/paper/static_noise.out 2>&1 &
wait
say "ALL DONE"
