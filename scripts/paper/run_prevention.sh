#!/bin/zsh
# Prevention-from-birth suppression (superclass code at layer4 + avgpool, epochs 1-10) and its random control,
# branched from each seed's clean 30-epoch lane (the label-noise experiment's clean lanes, same recipe).
# Usage: run_prevention.sh SEED [SEED ...]
set -u
WT=${RDT_DATA:-/data/rdt}/checkouts/lesion
PY=${PYTHON:-python}
D=${RDT_DATA:-/data/rdt}/core/experiments
LN=$D/0c28799c-ec9d-4e45-ac30-302dff810a8d/lanes
export DEVTRAIN_MPS_MAX_BATCH=128
typeset -A CLEAN
CLEAN=(7 498bb1e8-47b0-4e99-be8d-e35e566b1aa7 137 2cc9ed75-ab1f-444f-a59b-5cc9ac6c8f9f 256 5a1e6fe5-6b82-4532-8766-0895df9d2799 314 1da28ac9-4274-4ecd-a2b4-96a6adffc13a)
cd $WT
for S in "$@"; do
  echo "[$(date -u +%FT%TZ)] SEED $S START"
  $PY -u scripts/run_scaffold_lesion_experiment.py --experiment-id prevention \
    --clean-lane $LN/${CLEAN[$S]} --clean-lane-seed $S --seeds $S --epochs 30 \
    --branch-epochs 0 --window 10 --top-n 8 --conditions scaffold,random --matchings variance \
    --layers layer4,avgpool --no-ssi-gate --reproject-every 65 --reproject-rank-policy sliding \
    --reproject-max-rank-frac 0.7 --skip-sae \
    --template-json $D/noise_curriculum_arch/_config/template.json || echo "SEED $S FAILED rc=$?"
  echo "[$(date -u +%FT%TZ)] SEED $S END"
done
