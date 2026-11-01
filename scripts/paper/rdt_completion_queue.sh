#!/bin/zsh
# RDT completion: per-epoch SAE runs for the sham, blocking and matched-noise lanes, sequentially.
cd "$(dirname "$0")/../.."
PY=${PYTHON:-python}
TPL=${RDT_DATA:-/data/rdt}/core/experiments/noise_curriculum_arch/_config/template.json
export PYTHONHASHSEED=0 DEVTRAIN_MPS_MAX_BATCH=128
echo "[1/5] sham 256,314 $(date)";      $PY -u scripts/paper/run_sham_perepoch_sae.py 256,314
echo "[2/5] gradjoint $(date)";         $PY -u scripts/paper/run_blocking_rdt.py --arms gradjoint --seeds 7,42,137,256,314
echo "[3/5] seeds 256,314 $(date)";     $PY -u scripts/paper/run_blocking_rdt.py --arms gradsc,gradenergy,gradrand,gradrg --seeds 256,314
echo "[4/5] matched noise $(date)";     $PY -u scripts/paper/run_sae_lanes.py --template-json $TPL --only matched
echo "[5/5] layer3+ pair $(date)";      $PY -u scripts/paper/run_blocking_rdt.py --arms gradsc_l3,gradjoint_l3 --seeds 7,42,137,256,314
echo "QUEUE_END $(date)"
