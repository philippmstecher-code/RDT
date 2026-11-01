"""Shared helpers for the robustness check of the core census (robust_census_analyze_storedsai.py).

Runs: the 11 census runs (9 core, Tiny ImageNet, independent-init SAE) and the 5 clean 30-epoch ResNet-18 lanes,
with run lists, lane metadata and threshold sets taken from census_strict.census_runs / clean_runs.

Excluded lanes (label-noise experiment 0c28799c, curriculum, tg_suppression, critical) are rejected by `guard`.

CPU only (CUDA/MPS hidden), <= 3 BLAS / torch threads. Read-only on all experiment data. Writes only to
/data/rdt/extended/cache_census and analysis_outputs/paper/ROBUST_census.
"""
from __future__ import annotations

import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "3"

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import census_strict as DATA_ORDER  # noqa: E402  (sets thread env to 2 if imported first; ours is set above)
import torch  # noqa: E402

torch.set_num_threads(3)

OUT = REPO / "analysis_outputs" / "paper" / "ROBUST_census"
CACHE = Path(RDT_DATA + "/extended/cache_census")
TRANSFORMED_CACHE = REPO / "analysis_outputs" / "paper" / "TRANSFORMED" / "cache"
EXCLUDED = ("0c28799c", "curriculum", "tg_suppression", "critical")
READ_ONLY_ROOTS = (RDT_DATA + "/core",)

N_SAI_NULL = 100          # row-shuffled class assignments (variant a)
SAI_NULL_PCT = 95.0
SAI_NULL_SEED = 20261007
HEALTHY_FRAC = 0.10       # variant c: alive >= 10% of the dictionary ...
# ... at a majority (> half) of the real checkpoints
SAI_FIG4A = 0.9           # Fig. 4a: task-general = SAI > 0.9 (stored t_ssi / t_csi)


def guard(p) -> None:
    s = str(p)
    for f in EXCLUDED:
        if f in s:
            raise RuntimeError(f"excluded lane (not part of the census check): {s}")


def assert_writable(p) -> None:
    s = str(Path(p).resolve())
    for r in READ_ONLY_ROOTS:
        if s.startswith(r):
            raise RuntimeError(f"read-only volume, not writing: {s}")


def runs() -> List[Dict[str, Any]]:
    """11 census runs + 5 clean lanes, as in census_strict, with real checkpoints only."""
    out = []
    for r in DATA_ORDER.census_runs() + DATA_ORDER.clean_runs():
        guard(r["dir"])
        r = dict(r)
        labs = list(r["labels"])
        if r["group"] != "clean30":
            assert labs[-1] == "terminal", (r["run"], labs)
            labs = labs[:-1]            # 'terminal' = the last real milestone again (duplicate network)
        r["real"] = labs
        r["lane_id"] = Path(r["dir"]).name
        r["tsets"] = list(r["thr"].keys())
        out.append(r)
    return out


def record_layers(r: Dict[str, Any]) -> List[str]:
    """Lineage layers as in Fig. 4a (persist_lib.records_real, dedup): ResNet avgpool dropped (layer4 kept)."""
    return [L for L in r["layers"] if not (r["arch"] == "ResNet-18" and L == "avgpool")]


def sai_from_means(M: np.ndarray) -> np.ndarray:
    """Pipeline SAI (saeanalysis._phase4_ssi_csi): normalised entropy of the class-mean |code| distribution.
    M: (n_classes, F) class means of |codes|. Returns (F,)."""
    n_cls = M.shape[0]
    tot = M.sum(0)
    p = M / (tot[None, :] + 1e-10)
    ent = -np.where(p > 0, p * np.log(p + 1e-10), 0.0).sum(0)
    return np.where(tot > 1e-10, ent / np.log(n_cls), 0.0)


def shuffle_operators(n_cls: int, n_per: int, n_perm: int, seed: int):
    """Sparse (n_cls x N) averaging operators for n_perm row-shuffled class assignments (same for every matrix of
    a run: the samples are the same images at every checkpoint and layer)."""
    from scipy import sparse
    N = n_cls * n_per
    y = np.repeat(np.arange(n_cls), n_per)
    rng = np.random.default_rng(seed)
    ops = []
    for _ in range(n_perm):
        yp = y[rng.permutation(N)]
        ops.append(sparse.csr_matrix((np.full(N, 1.0 / n_per, dtype=np.float32), (yp, np.arange(N))), shape=(n_cls, N)))
    return ops
