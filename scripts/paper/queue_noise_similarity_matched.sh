#!/bin/zsh
# Similarity-matched between-superclass noise lanes only (seeds 7..314, added to static_noise_s{S}).
set -u
PY=${PYTHON:-python}
TPL=${RDT_DATA:-/data/rdt}/core/experiments/noise_curriculum_arch/_config/template.json
TGT=${REPO:-.}/analysis_outputs/paper/NOISE_matched/targets.json
LN=${RDT_DATA:-/data/rdt}/core/experiments/0c28799c-ec9d-4e45-ac30-302dff810a8d/lanes
export DEVTRAIN_MPS_MAX_BATCH=128 PYTHONHASHSEED=0
typeset -A CLEAN
CLEAN=(7 498bb1e8-47b0-4e99-be8d-e35e566b1aa7 42 0bd1c27f-38f5-449d-8159-144392a92026)
say() { echo "[$(date -u +%FT%TZ)] NOISE $*"; }
cd ${RDT_DATA:-/data/rdt}/checkouts/lesion
for S in 7 42 137 256 314; do
  say "MATCHED seed $S START"
  $PY -u scripts/run_noise_curriculum_experiment.py --experiment-id static_noise_s$S --arms between_sc_matched \
     --switch-epochs "" --archs resnet18 --epochs 30 --noise-prob 0.3 --seed $S --skip-sae \
     --matched-targets-json $TGT --template-json $TPL || say "MATCHED seed $S FAILED rc=$?"
  say "MATCHED seed $S END"
done
say "MATCHED ALL DONE"
