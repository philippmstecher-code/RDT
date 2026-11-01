#!/bin/zsh
# Critical period of the scaffold: suppression of the superclass code during epochs 11-20 and 21-30,
# branched from each seed's clean 30-epoch run, with the same settings as the epochs 1-10 runs (prevention)
# and with SAE / RDT analysis; plus sham lanes. Usage: zsh run_critical.sh 7 42 137 256 314
set -u
WT=${RDT_DATA:-/data/rdt}/checkouts/lesion
PY=${PYTHON:-python}
D=${RDT_DATA:-/data/rdt}/core/experiments
LN=$D/0c28799c-ec9d-4e45-ac30-302dff810a8d/lanes
export DEVTRAIN_MPS_MAX_BATCH=128
export PYTHONHASHSEED=0
typeset -A CLEAN
CLEAN=(7 498bb1e8-47b0-4e99-be8d-e35e566b1aa7 42 0bd1c27f-38f5-449d-8159-144392a92026 137 2cc9ed75-ab1f-444f-a59b-5cc9ac6c8f9f 256 5a1e6fe5-6b82-4532-8766-0895df9d2799 314 1da28ac9-4274-4ecd-a2b4-96a6adffc13a)
cd $WT
for S in "$@"; do
  echo "[$(date -u +%FT%TZ)] CRIT SEED $S START"
  $PY -u scripts/run_scaffold_lesion_experiment.py --experiment-id critical \
    --clean-lane $LN/${CLEAN[$S]} --clean-lane-seed $S --seeds $S --epochs 30 \
    --branch-epochs 10,20 --window 10 --top-n 8 --conditions scaffold,random --matchings variance \
    --layers layer4,avgpool --no-ssi-gate --reproject-every 65 --reproject-rank-policy sliding \
    --reproject-max-rank-frac 0.7 \
    --template-json $D/noise_curriculum_arch/_config/template.json || echo "CRIT SEED $S FAILED rc=$?"
  # Sham lanes (same branch checkpoint and continuation config, no projection), branches
  # 0 (baseline for prevention, window 1-10) and 10 (window 11-20); training only, no SAE.
  echo "[$(date -u +%FT%TZ)] CRIT SEED $S SHAM START"
  $PY -u scripts/run_critical_sham.py --experiment-id critical_sham \
    --clean-lane $LN/${CLEAN[$S]} --clean-lane-seed $S --seeds $S --branch-epochs 0,10 --window 10 \
    --epochs 30 --skip-sae \
    --template-json $D/noise_curriculum_arch/_config/template.json || echo "CRIT SEED $S SHAM FAILED rc=$?"
  echo "[$(date -u +%FT%TZ)] CRIT SEED $S END"
done
echo "[$(date -u +%FT%TZ)] CRIT ALL DONE"
