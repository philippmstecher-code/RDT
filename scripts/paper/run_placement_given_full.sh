#!/bin/zsh
# Placement given (placement_given): seeds 7 137 256 314 (seed 42 run by run_placement_given.sh), then the
# final analysis once all 15 lanes (5 seeds x 3 arms) have .devtrain_done.
# Uses the orchestrator call of run_placement_given.sh::run_seed, sourced from that file.
# Usage: nohup scripts/paper/run_placement_given_full.sh >> logs/paper/placement_given_full.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
source <(sed -n "/^REPO=/,/^}/p" scripts/paper/run_placement_given.sh)
for S in 7 137 256 314; do run_seed $S; done
L=${RDT_DATA:-/data/rdt}/extended/experiments/placement_given/lanes
n=0
for S in 7 42 137 256 314; do
  for A in sham_e00_seed$S lesion_gradsc_e00_seed${S}_rp65_Llayer1-layer2-layer3-layer4-avgpool \
           lesion_gradjoint_e00_seed${S}_rp65_Llayer1-layer2-layer3-layer4-avgpool; do
    [[ -e $L/$A/.devtrain_done ]] && n=$((n + 1)) || echo "[$(ts)] NOT DONE: $A"
  done
done
echo "[$(ts)] lanes done: $n/15"
if (( n == 15 )); then
  echo "[$(ts)] FINAL ANALYSIS START"
  $PY -u $REPO/scripts/paper/placement_given_analysis.py > $LOG/placement_given_final.out 2>&1
  echo "[$(ts)] FINAL ANALYSIS rc=$?"
else
  echo "[$(ts)] final analysis NOT run (incomplete lanes)"
fi
touch $LOG/.placement_given_full_finished
echo "[$(ts)] PLACEMENT FULL END"
