#!/usr/bin/env python3
"""Population geometry and probes of the taskirr lanes with the dictionary-free geometry code (hierarchy_geometry.run).

Two analyses:
  * linear probes for true superclasses vs random partitions at every checkpoint (p_true20 vs p_null20),
    all five seeds;
  * held-out seeds 256 / 314: true-superclass nesting AUC (avgpool) at epoch 30 below its epoch-0 value.
hierarchy_lib's seed list (SEEDS / EXCLUDED) is set to all five seeds.
CPU only, 2 threads; read-only on the data. Writes analysis_outputs/paper/TASK_irrelevant/geo/ and
analysis_outputs/paper/DATA_order/taskirr_share_test/H-DS_taskirr_secondary.json.

Usage: task_irrelevant_geometry.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import hierarchy_lib as L  # noqa: E402

SEEDS = [7, 42, 137, 256, 314]
HELDOUT_SEEDS = [256, 314]
L.SEEDS = list(SEEDS)
L.EXCLUDED = set()

import hierarchy_geometry as G  # noqa: E402

OUT = L.REPO / "analysis_outputs" / "paper" / "TASK_irrelevant" / "geo"
HDS = L.REPO / "analysis_outputs" / "paper" / "DATA_order" / "taskirr_share_test"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for s in SEEDS:
        f = OUT / f"geo_taskirr_s{s}.csv"
        if f.exists():
            continue
        assert (L.lane_dir("taskirr", s) / ".lane_done").exists(), s
        G.run("taskirr", s).to_csv(f, index=False)
    g = pd.concat([pd.read_csv(OUT / f"geo_taskirr_s{s}.csv") for s in SEEDS])
    ap = g[g.layer == "avgpool"]
    per = {}
    for s in HELDOUT_SEEDS:
        q = ap[ap.seed == s].set_index("epoch")
        a0, a30 = float(q.loc[0, "true_auc"]), float(q.loc[30, "true_auc"])
        per[s] = {"true_auc_epoch0": a0, "true_auc_epoch30": a30, "diff": a30 - a0, "below": a30 < a0}
    res = {"test": "taskirr true-superclass nesting AUC at epoch 30 < epoch 0 (held-out seeds 256, 314)",
           "layer": "avgpool", "per_seed": per, "holds": all(p["below"] for p in per.values()),
           "open_seeds_descriptive": {
               int(s): {"true_auc_epoch0": float(ap[(ap.seed == s) & (ap.epoch == 0)].true_auc.iloc[0]),
                        "true_auc_epoch30": float(ap[(ap.seed == s) & (ap.epoch == 30)].true_auc.iloc[0])}
               for s in (7, 42, 137)}}
    json.dump(res, open(HDS / "H-DS_taskirr_secondary.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
