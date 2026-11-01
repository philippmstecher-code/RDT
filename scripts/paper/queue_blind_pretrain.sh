#!/bin/zsh
# Superclass-blind pre-training (blind_pretrain): arm sc_blind, 20 epochs on the five-way
# within-superclass-index task, then 30 fine-class epochs (same protocol as sc_true_K20 / sc_rand_K20),
# seeds 7 42 137 256 314 sequentially. The SAE analysis runs after training and after run_placement_given_full.sh
# finishes; blind_pretrain_analysis.py runs once all five lanes are complete.
# Launch: nohup scripts/paper/queue_blind_pretrain.sh > logs/paper/blind_pretrain_queue.out 2>&1 &
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY=${PYTHON:-python}
LOG=$REPO/logs/paper
TPL=${RDT_DATA:-/data/rdt}/core/experiments/noise_curriculum_arch/_config/template.json
export DEVTRAIN_MPS_MAX_BATCH=128
say() { echo "[$(date -u +%FT%TZ)] QUEUE $*"; }
cd $REPO

say "TRAINING START (seeds 7 42 137 256 314)"
for S in 7 42 137 256 314; do
  say "train seed $S START"
  CURRICULUM_WT=${RDT_DATA:-/data/rdt}/checkouts/blind_pretraining $PY -u scripts/paper/run_curriculum.py \
    --experiment-id blind_pretrain --seed $S --K 20 --epochs 50 --arms sc_blind --template-json $TPL \
    > $LOG/blind_pretrain_train_seed$S.out 2>&1 || say "train seed $S FAILED rc=$?"
  say "train seed $S END"
done
say "TRAINING END"

say "SAE runs after run_placement_given_full.sh finishes"
while pgrep -f run_placement_given_full.sh >/dev/null; do sleep 120; done
say "SAE START"
$PY -u scripts/paper/run_curriculum_sae.py --experiment-id blind_pretrain --template-json $TPL \
  --arms "sc_blind_K20_seed{}" > $LOG/blind_pretrain_sae.out 2>&1 || say "SAE FAILED rc=$?"
say "SAE END"

n=0
for S in 7 42 137 256 314; do
  L=${RDT_DATA:-/data/rdt}/extended/experiments/blind_pretrain/lanes/sc_blind_K20_seed$S
  [[ -e $L/.lane_done && -e $L/.sae_done ]] && n=$((n + 1)) || say "NOT DONE: $L"
done
say "lanes trained + SAE: $n/5"
if (( n == 5 )); then
  say "ANALYSIS START"
  $PY -u scripts/paper/blind_pretrain_analysis.py > $LOG/blind_pretrain_analysis.out 2>&1
  say "ANALYSIS rc=$?"
else
  say "analysis NOT run (incomplete lanes)"
fi
say "ALL END"
