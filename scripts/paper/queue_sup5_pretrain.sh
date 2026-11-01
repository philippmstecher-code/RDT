#!/bin/zsh
# Label-count control aligned with the hierarchy (sup5_pretrain): arm sc_sup5, 20 epochs on five groups of four
# whole superclasses (coarse_labels 'superclass_groups', seed 3000 + seed), then 30 fine-class epochs; protocol and
# seeds of the superclass-blind arm; training code from the checkout superclass_groups_pretraining (variants/); SAE analysis after
# training.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY=${PYTHON:-python}
LOG=$REPO/logs/paper
TPL=${RDT_DATA:-/data/rdt}/core/experiments/noise_curriculum_arch/_config/template.json
WT=${RDT_DATA:-/data/rdt}/checkouts/superclass_groups_pretraining
export DEVTRAIN_MPS_MAX_BATCH=128
say() { echo "[$(date -u +%FT%TZ)] QUEUE $*"; }
cd $REPO

say "TRAINING START (seeds 7 42 137 256 314)"
for S in 7 42 137 256 314; do
  say "train seed $S START"
  CURRICULUM_WT=$WT $PY -u scripts/paper/run_curriculum.py \
    --experiment-id sup5_pretrain --seed $S --K 20 --epochs 50 --arms sc_sup5 --template-json $TPL \
    > $LOG/sup5_pretrain_train_seed$S.out 2>&1 || say "train seed $S FAILED rc=$?"
  say "train seed $S END"
done
say "TRAINING END"

say "SAE START"
CURRICULUM_WT=$WT $PY -u scripts/paper/run_curriculum_sae.py --experiment-id sup5_pretrain --template-json $TPL \
  --arms "sc_sup5_K20_seed{}" > $LOG/sup5_pretrain_sae.out 2>&1 || say "SAE FAILED rc=$?"
say "SAE END"

n=0
for S in 7 42 137 256 314; do
  L=${RDT_DATA:-/data/rdt}/extended/experiments/sup5_pretrain/lanes/sc_sup5_K20_seed$S
  [[ -e $L/.lane_done && -e $L/.sae_done ]] && n=$((n + 1)) || say "NOT DONE: $L"
done
say "lanes trained + SAE: $n/5"
say "ALL END"
