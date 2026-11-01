#!/bin/zsh
# Random-grouping control for the last-block suppression, epochs 1-10 (Extended Data Fig. 6): exactly the pipeline of the
# prevention superclass arm (branch epoch 0, window 10, re-estimation every 65 updates, q_max 19, EVR floor 2%,
# sliding rank cap 0.7, layer4 + avgpool), with the between-GROUP code of a fixed random grouping of the 100 classes
# into 20 groups of five (RandomState(10000 + seed)) instead of the superclass code, initialised with the leading
# between-group directions at the superclass arm's initial rank (8 per layer).
# Each seed branches from the same clean lane as its superclass arm.
# Usage: run_randgroup.sh SEED [SEED ...]
set -u
WT=${RDT_DATA:-/data/rdt}/checkouts/random_grouping
PY=${PYTHON:-python}
D=${RDT_DATA:-/data/rdt}/core/experiments
LN=$D/0c28799c-ec9d-4e45-ac30-302dff810a8d/lanes
export DEVTRAIN_MPS_MAX_BATCH=128
typeset -A CLEAN
CLEAN=(7 $LN/498bb1e8-47b0-4e99-be8d-e35e566b1aa7 137 $LN/2cc9ed75-ab1f-444f-a59b-5cc9ac6c8f9f
       256 $LN/5a1e6fe5-6b82-4532-8766-0895df9d2799 314 $LN/1da28ac9-4274-4ecd-a2b4-96a6adffc13a
       42 $D/noise_curriculum_arch/lanes/standard_resnet18)
cd $WT
for S in "$@"; do
  echo "[$(date -u +%FT%TZ)] SEED $S START"
  $PY -u scripts/run_scaffold_lesion_experiment.py --experiment-id randgroup \
    --clean-lane ${CLEAN[$S]} --clean-lane-seed $S --seeds $S --epochs 30 \
    --branch-epochs 0 --window 10 --top-n 8 --conditions randgroup --matchings variance \
    --layers layer4,avgpool --no-ssi-gate --reproject-every 65 --reproject-rank-policy sliding \
    --reproject-max-rank-frac 0.7 --skip-sae \
    --template-json $D/noise_curriculum_arch/_config/template.json || echo "SEED $S FAILED rc=$?"
  echo "[$(date -u +%FT%TZ)] SEED $S END"
done
