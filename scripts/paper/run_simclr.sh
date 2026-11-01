#!/bin/zsh
# Self-supervised control (simclr_train): SimCLR ResNet-18 on the CIFAR-100 training images without labels, 100 epochs,
# checkpoints and pooled test-set activations at epochs 0, 1, 2, 3, 5, 10, 20, ..., 100 (scripts/paper/simclr_train.py).
# Seeds run one after another; each run resumes from <OUT>/seed<S>/train_state.pt if present. Writes only to OUT.
# Analysis afterwards: python scripts/paper/simclr_analysis.py
# Usage: [OUT=...] [EPOCHS=...] [LR_REF=128 for the Lightly lr scaling] run_simclr.sh SEED [SEED ...]
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY=${PYTHON:-python}
OUT=${OUT:-${RDT_DATA:-/data/rdt}/extended/experiments/simclr_train}
cd $REPO
for S in "$@"; do
  echo "[$(date -u +%FT%TZ)] SEED $S START"
  $PY -u scripts/paper/simclr_train.py --seed $S --epochs ${EPOCHS:-100} --lr-ref ${LR_REF:-256} --out $OUT || echo "SEED $S FAILED rc=$?"
  echo "[$(date -u +%FT%TZ)] SEED $S END"
done
