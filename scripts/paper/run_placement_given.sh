#!/bin/zsh
# Placement given (placement_given): training loss = cross-entropy over the five classes of each image's TRUE
# superclass only (orchestrator --masked-loss; devtrain masked_loss_policy). Three arms per seed, run sequentially in one
# orchestrator call: no block (sham_e00), superclass block (gradsc), jointly matched block (gradjoint). Settings of
# run_gradblock.sh / run_gradjoint.sh (branch epoch 0, block epochs 1-10 at layer1-4 and the pooled layer,
# re-estimation every 65 updates, q_max 19, EVR floor 2%, sliding rank cap 0.7, same clean lane per seed), 15 epochs.
# Seed 42 runs first; seeds 7 137 256 314 run only if the technical checks of placement_given_analysis.py --gate
# are met (its output line starts with 'checks: criterion met'); then the final analysis.
# Usage: nohup scripts/paper/run_placement_given.sh > logs/paper/placement_given_wrapper.out 2>&1 &
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
WT=${RDT_DATA:-/data/rdt}/checkouts/placement
PY=${PYTHON:-python}
LOG=$REPO/logs/paper
D=${RDT_DATA:-/data/rdt}/core/experiments
LN=$D/0c28799c-ec9d-4e45-ac30-302dff810a8d/lanes
export DEVTRAIN_MPS_MAX_BATCH=128
typeset -A CLEAN
CLEAN=(7 $LN/498bb1e8-47b0-4e99-be8d-e35e566b1aa7 137 $LN/2cc9ed75-ab1f-444f-a59b-5cc9ac6c8f9f
       256 $LN/5a1e6fe5-6b82-4532-8766-0895df9d2799 314 $LN/1da28ac9-4274-4ecd-a2b4-96a6adffc13a
       42 $D/noise_curriculum_arch/lanes/standard_resnet18)
ts() { date -u +%FT%TZ }

run_seed() {
  local S=$1
  echo "[$(ts)] SEED $S START"
  (cd $WT && $PY -u scripts/run_scaffold_lesion_experiment.py --experiment-id placement_given \
    --clean-lane ${CLEAN[$S]} --clean-lane-seed $S --seeds $S --epochs 15 \
    --branch-epochs 0 --window 10 --top-n 8 --conditions none,gradsc,gradjoint --matchings variance \
    --layers layer1,layer2,layer3,layer4,avgpool --no-ssi-gate --reproject-every 65 --reproject-rank-policy sliding \
    --reproject-max-rank-frac 0.7 --skip-sae --masked-loss \
    --template-json $D/noise_curriculum_arch/_config/template.json) || echo "SEED $S FAILED rc=$?"
  echo "[$(ts)] SEED $S END"
}

echo "[$(ts)] SEED 42 START (first seed)"
run_seed 42 > $LOG/placement_given_pilot.out 2>&1
echo "[$(ts)] SEED 42 END; technical checks"
$PY -u $REPO/scripts/paper/placement_given_analysis.py --gate > $LOG/placement_given_gate.out 2>&1
cat $LOG/placement_given_gate.out
if ! grep -q '^checks: criterion met' $LOG/placement_given_gate.out; then
  echo "[$(ts)] technical checks not met: seeds 7 137 256 314 NOT started"
  exit 1
fi
echo "[$(ts)] technical checks met: seeds 7 137 256 314 START"
for S in 7 137 256 314; do
  run_seed $S >> $LOG/placement_given_full.out 2>&1
done
echo "[$(ts)] SEEDS 7 137 256 314 END; final analysis"
$PY -u $REPO/scripts/paper/placement_given_analysis.py > $LOG/placement_given_final.out 2>&1
echo "[$(ts)] FINAL ANALYSIS rc=$?"
