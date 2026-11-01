#!/bin/zsh
# RDT completion driver: a second SAE stream beside rdt_completion_queue.sh, then all analyses.
#  1. runs after rdt_completion_analysis.py finishes
#  2. stream B (layer3+ pair; run_blocking_rdt.py lane locks prevent overlap with rdt_completion_queue.sh) starts if
#     at least 50% of memory and 150 GB of disk are free
#  3. resource check every 5 min: stream B stops if free disk falls below 60 GB or free memory below 15%
#  4. once rdt_completion_queue.sh (QUEUE_END) and stream B have ended: a final pass, then rdt_completion_finalize.sh
cd "$(dirname "$0")/../.."
PY=${PYTHON:-python}
LOG=logs/paper/rdt_completion_driver.out
QLOG=logs/paper/rdt_completion.out
say() { echo "[$(date '+%F %T')] $*" | tee -a $LOG; }
disk_free_gb() { df -g ${RDT_DATA:-/data/rdt}/extended | awk 'NR==2{print $4}'; }
mem_free() { memory_pressure | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}'; }

say "driver start; runs after rdt_completion_analysis.py"
while pgrep -f rdt_completion_analysis.py >/dev/null; do sleep 60; done
say "rdt_completion_analysis.py finished; disk free $(disk_free_gb) GB, memory free $(mem_free)%"
# partial pattern analyses on the complete jointly matched comparison
$PY scripts/paper/rdt_completion_patterns.py routing --arms sham,gradsc,gradjoint > analysis_outputs/paper/RDT_completion/run_routing_partial.log 2>&1
$PY scripts/paper/rdt_completion_patterns.py dose --arms sham,gradsc,gradjoint > analysis_outputs/paper/RDT_completion/run_dose_partial.log 2>&1
$PY scripts/paper/rdt_completion_patterns.py turnover --arms sham,gradsc,gradjoint,gradenergy,gradrand,gradrg > analysis_outputs/paper/RDT_completion/run_turnover_partial.log 2>&1
say "partial pattern analyses done"

BPID=""
if [[ $(disk_free_gb) -ge 150 && $(mem_free) -ge 50 ]]; then
  PYTHONHASHSEED=0 DEVTRAIN_MPS_MAX_BATCH=128 $PY -u scripts/paper/run_blocking_rdt.py \
      --arms gradjoint_l3,gradsc_l3 --seeds 314,256,137,42,7 > logs/paper/rdt_streamB.out 2>&1 &
  BPID=$!
  say "stream B started (pid $BPID)"
else
  say "stream B NOT started (not enough free disk or memory)"
fi

while true; do
  sleep 300
  if [[ -n $BPID ]] && kill -0 $BPID 2>/dev/null; then
    f=$(disk_free_gb); m=$(mem_free)
    if [[ $f -lt 60 || $m -lt 15 ]]; then
      say "resource check: disk $f GB, memory $m% -> stopping stream B"
      kill $BPID; BPID=""
    fi
  fi
  qdone=$(grep -c QUEUE_END $QLOG)
  bdone=1; [[ -n $BPID ]] && kill -0 $BPID 2>/dev/null && bdone=0
  if [[ $qdone -ge 1 && $bdone -eq 1 ]]; then break; fi
done
say "queue and stream B ended; disk free $(disk_free_gb) GB"
# final sequential pass over lanes that rdt_completion_queue.sh skipped (held by stream B) or that failed (lane locks apply)
PYTHONHASHSEED=0 DEVTRAIN_MPS_MAX_BATCH=128 $PY -u scripts/paper/run_blocking_rdt.py \
    --arms gradsc,gradenergy,gradrand,gradrg,gradjoint,gradsc_l3,gradjoint_l3 --seeds 7,42,137,256,314 >> $LOG 2>&1
say "final pass done; finalize"
zsh scripts/paper/rdt_completion_finalize.sh >> $LOG 2>&1
say "driver end"
