#!/bin/zsh
# Frozen 100-class read-out (Extended Data Fig. 9d measure, curriculum_probe.py) for the superclass-blind arm
# (blind_pretrain, sc_blind_K20, seeds 7 42 137 256 314): scripts/paper/blind_frozen_readout.py.
# Read-only on the lanes; runs alongside training (queue_blind_pretrain.sh), one process at a time.
#   1. reproduction check: sc_rand_K20 seed 42 against CURRICULUM_probe/probe.csv (stops on mismatch)
#   2. per seed, once its switch activations exist (epoch 20, milestone_4): every milestone written so far
#   3. once every lane has .lane_done: the remaining milestones (fine-class epochs up to 30)
# Outputs: analysis_outputs/paper/BLIND_PRETRAIN/frozen_readout.{csv,md}
# Launch: nohup scripts/paper/queue_blind_frozen_readout.sh > logs/paper/blind_frozen_readout.out 2>&1 &
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY=${PYTHON:-python}
SCRIPT=$REPO/scripts/paper/blind_frozen_readout.py
LANES=${RDT_DATA:-/data/rdt}/extended/experiments/blind_pretrain/lanes
SEEDS=(7 42 137 256 314)
MIN_FREE_PCT=20
MAX_WAIT_S=$((12 * 3600))
say() { echo "[$(date -u +%FT%TZ)] FROZEN_READOUT $*"; }
cd $REPO
export PYTHONWARNINGS=ignore

free_pct() { memory_pressure 2>/dev/null | awk -F': ' '/free percentage/ {gsub("%", "", $2); print $2}'; }
wait_mem() {
  local f
  while true; do
    f=$(free_pct); f=${f:-100}
    (( f >= MIN_FREE_PCT )) && { say "memory free ${f}%"; return; }
    say "memory free ${f}% < ${MIN_FREE_PCT}%; waiting"; sleep 120
  done
}
switch_ready() {   # epoch-20 activations (milestone_4) written and older than 60 s
  local f=$LANES/sc_blind_K20_seed$1/dev_snapshots/milestone_4/individual_multilayer_activations.pt
  [[ -e $f ]] && (( $(date +%s) - $(stat -f %m $f) > 60 ))
}

say "START (pid $$)"
wait_mem
say "reproduction gate START"
$PY -u $SCRIPT --repro || { say "REPRO FAILED rc=$?; stopping"; exit 1; }
say "reproduction gate PASS"

t0=$(date +%s)
for S in $SEEDS; do
  until switch_ready $S; do
    (( $(date +%s) - t0 > MAX_WAIT_S )) && { say "seed $S: switch not ready after ${MAX_WAIT_S}s; skipping"; continue 2; }
    sleep 120
  done
  wait_mem
  say "seed $S START (switch ready)"
  $PY -u $SCRIPT --seeds $S || say "seed $S FAILED rc=$?"
  say "seed $S END"
done

say "waiting for .lane_done of every seed (fine-class checkpoints)"
while true; do
  n=0; for S in $SEEDS; do [[ -e $LANES/sc_blind_K20_seed$S/.lane_done ]] && n=$((n + 1)); done
  (( n == ${#SEEDS} )) && break
  (( $(date +%s) - t0 > MAX_WAIT_S )) && { say "only $n/${#SEEDS} lanes done after ${MAX_WAIT_S}s; computing what exists"; break; }
  sleep 120
done
sleep 90   # let the last activation files settle
wait_mem
say "all seeds, remaining milestones START"
$PY -u $SCRIPT --seeds $SEEDS || say "remaining milestones FAILED rc=$?"
say "all seeds, remaining milestones END"
$PY -c "
import pandas as pd
d = pd.read_csv('$REPO/analysis_outputs/paper/BLIND_PRETRAIN/frozen_readout.csv')
print('rows per seed:', d.groupby('seed').size().to_dict()); print('switch rows:', sorted(d[d.epoch == 20].seed))"
say "ALL END"
