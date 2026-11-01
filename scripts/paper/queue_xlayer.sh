#!/bin/zsh
# Cross-layer superclass suppression (xlayer), then cross-layer task-general suppression in an early and a
# late window (tgxl), seeds 7 42 137 256 314. The script starts once a run_pretrained.py process (not part of this
# release) is running, stops that process (SIGSTOP) for the duration of these runs and resumes it afterwards.
set -u
WT=${RDT_DATA:-/data/rdt}/checkouts/lesion
PY=${PYTHON:-python}
D=${RDT_DATA:-/data/rdt}/core/experiments
LN=$D/0c28799c-ec9d-4e45-ac30-302dff810a8d/lanes
TPL=$D/noise_curriculum_arch/_config/template.json
ALL=layer1,layer2,layer3,layer4,avgpool
export DEVTRAIN_MPS_MAX_BATCH=128
export PYTHONHASHSEED=0
typeset -A CLEAN
CLEAN=(7 498bb1e8-47b0-4e99-be8d-e35e566b1aa7 42 0bd1c27f-38f5-449d-8159-144392a92026 137 2cc9ed75-ab1f-444f-a59b-5cc9ac6c8f9f 256 5a1e6fe5-6b82-4532-8766-0895df9d2799 314 1da28ac9-4274-4ecd-a2b4-96a6adffc13a)
COMMON=(--epochs 30 --window 10 --top-n 8 --matchings variance --layers $ALL --no-ssi-gate --reproject-every 65
        --reproject-rank-policy sliding --reproject-max-rank-frac 0.7 --template-json $TPL)
say() { echo "[$(date -u +%FT%TZ)] XL $*"; }
say "waiting for run_pretrained.py"
until PID=$(pgrep -f "run_pretrained.py" | head -1) && [ -n "$PID" ]; do sleep 2; done
kill -STOP $PID && say "stopped run_pretrained.py (pid $PID)"
cd $WT
for S in 7 42 137 256 314; do
  say "SUPERCLASS SEED $S START"
  $PY -u scripts/run_scaffold_lesion_experiment.py --experiment-id xlayer --clean-lane $LN/${CLEAN[$S]} \
    --clean-lane-seed $S --seeds $S --branch-epochs 0 --conditions scaffold,random $COMMON || say "SUPERCLASS SEED $S FAILED rc=$?"
  say "SUPERCLASS SEED $S END"
done
for S in 7 42 137 256 314; do
  say "TG EARLY SEED $S START"
  $PY -u scripts/run_scaffold_lesion_experiment.py --experiment-id tgxl --clean-lane $LN/${CLEAN[$S]} \
    --clean-lane-seed $S --seeds $S --branch-epochs 0 --conditions taskgeneral,random $COMMON || say "TG EARLY SEED $S FAILED rc=$?"
  say "TG LATE SEED $S START"
  $PY -u scripts/run_scaffold_lesion_experiment.py --experiment-id tgxl --clean-lane $LN/${CLEAN[$S]} \
    --clean-lane-seed $S --seeds $S --branch-epochs 10 --conditions taskgeneral,random --skip-sae $COMMON || say "TG LATE SEED $S FAILED rc=$?"
  say "TG SEED $S END"
done
kill -CONT $PID && say "resumed run_pretrained.py (pid $PID)"
say "ALL DONE"
