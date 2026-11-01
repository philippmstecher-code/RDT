#!/bin/zsh
# Continuation of the acquisition-blocking runs to epoch 30: each blocking lane (gradsc, gradrand, gradenergy; block in
# epochs 1-10, trained to epoch 15 on the 30-epoch schedule) is resumed from its own epoch-15 checkpoint
# (checkpoint.pt + training_state.pt; cosine T_max 30) without any block, through epoch 30, one checkpoint per epoch
# (run_critical_sham.py with branch 15, window 0, post-release-dense 15).
# Usage: [ARMS="gradsc gradrand gradenergy"] [EPOCHS=30] run_gradcont.sh SEED [SEED ...]
set -u
WT=${RDT_DATA:-/data/rdt}/checkouts/random_grouping
PY=${PYTHON:-python}
D=${RDT_DATA:-/data/rdt}/core/experiments
E=${RDT_DATA:-/data/rdt}/extended/experiments
L5=Llayer1-layer2-layer3-layer4-avgpool
export DEVTRAIN_MPS_MAX_BATCH=128
typeset -A SRC
SRC=(gradsc gradblock gradrand gradblock gradenergy gradenergy)
cd $WT
for S in "$@"; do
  for A in ${=ARMS:-gradsc gradrand gradenergy}; do
    echo "[$(date -u +%FT%TZ)] CONT $A SEED $S START"
    $PY -u scripts/run_critical_sham.py --experiment-id ${EXP_PREFIX:-gradcont}_${A}_r1 \
      --clean-lane $E/${SRC[$A]}/lanes/lesion_${A}_e00_seed${S}_rp65_${L5} --clean-lane-seed $S --seeds $S \
      --branch-epochs 15 --window 0 --post-release-dense 15 --epochs ${EPOCHS:-30} --skip-sae \
      --template-json $D/noise_curriculum_arch/_config/template.json || echo "CONT $A SEED $S FAILED rc=$?"
    echo "[$(date -u +%FT%TZ)] CONT $A SEED $S END"
  done
done
