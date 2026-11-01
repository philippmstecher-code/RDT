#!/usr/bin/env python3
"""Random-partition null for the nesting ratio of Extended Data Fig. 6a (pooled-layer SAE).

Observed (avgpool SAE, 5 standard lanes, milestones 1-6 = epochs 5-30; stored thresholds; exclusive rule):
  per class-selective feature f (CSI > t, alive, SAI <= t) with best superclass c: e_same = squared norm of the projection
  of its unit decoder direction onto span{decoders of superclass-selective features (SSI > t, CSI <= t) with best
  superclass c}; e_other = mean over other superclasses' equally sized SSI subsets; ratio = e_same / e_other;
  per seed: median over features and milestones.
Null: identical feature sets, decoders and thresholds, but the 'superclass' of every feature is recomputed under a random
partition of the 100 classes into 20 groups of five (best group = argmax of class-mean |code| summed per group).
100 partitions (RandomState(p), p = 1..100). Per seed: null mean, 95% interval and p = (1 + #null >= observed) / (1 + 100).
Writes analysis_outputs/paper/DEPENDENCY/T4_partition_null.json.
Usage: python scripts/paper/nesting_partition_null.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dependency_lib import COARSE_TO_FINE, OUT, LaneData, selectivity, standard_lanes  # noqa: E402

MS = [1, 2, 3, 4, 5, 6]
NPERM = 100
LAYER = "avgpool"


def groups_true():
    g = np.empty(100, int)
    for c in range(20):
        g[COARSE_TO_FINE[c]] = c
    return g


def ratio_stat(D, M, csi_idx, ssi_idx, grp, rng):
    """Per-feature ratios e_same / e_other, with each feature's best group taken under partition grp."""
    G = np.stack([M[:, grp == k].sum(1) for k in range(20)], 1)
    best = G.argmax(1)
    by = {c: ssi_idx[best[ssi_idx] == c] for c in range(20)}
    r = []
    for f in csi_idx:
        c = best[f]
        same = by[c]
        k = len(same)
        if k == 0:
            continue
        Q, _ = np.linalg.qr(D[:, same])
        e_same = float(((Q.T @ D[:, f]) ** 2).sum())
        eo = []
        for c2 in range(20):
            if c2 == c or len(by[c2]) < k:
                continue
            for _ in range(5 if len(by[c2]) > k else 1):
                Qo, _ = np.linalg.qr(D[:, rng.choice(by[c2], k, replace=False)])
                eo.append(float(((Qo.T @ D[:, f]) ** 2).sum()))
        if eo:
            r.append(e_same / np.mean(eo))
    return r


def main():
    res = {}
    gt = groups_true()
    parts = []
    for p in range(1, NPERM + 1):
        g = np.empty(100, int)
        for k, row in enumerate(np.random.RandomState(p).permutation(100).reshape(20, 5)):
            g[row] = k
        parts.append(g)
    for r in standard_lanes():
        lane = LaneData(r["dir"])
        rng = np.random.default_rng(r["seed"])
        obs, null = [], [[] for _ in range(NPERM)]
        for m in MS:
            H = lane.codes(m, LAYER)
            s = selectivity(H)
            D = torch.nn.functional.normalize(lane.sae(m, LAYER)["decoder.weight"].double(), dim=0).numpy()
            alive = (H != 0).mean(0) >= 0.01
            sai = s["sai"] > lane.sai_t
            csi_idx = np.where((s["csi"] > lane.csi_t) & alive & ~sai)[0]
            ssi_idx = np.where((s["ssi"] > lane.ssi_t) & (s["csi"] <= lane.csi_t) & alive & ~sai)[0]
            obs += ratio_stat(D, s["M"], csi_idx, ssi_idx, gt, rng)
            for i, g in enumerate(parts):
                null[i] += ratio_stat(D, s["M"], csi_idx, ssi_idx, g, rng)
        o = float(np.median(obs))
        nm = np.array([np.median(x) if x else np.nan for x in null])
        nm = nm[np.isfinite(nm)]
        res[int(r["seed"])] = {"observed_median_ratio": o, "n_features": len(obs), "null_mean": float(nm.mean()),
                               "null_95": [float(np.percentile(nm, 2.5)), float(np.percentile(nm, 97.5))],
                               "null_max": float(nm.max()), "n_null": int(len(nm)),
                               "p": float((1 + (nm >= o).sum()) / (1 + len(nm)))}
        print(r["seed"], json.dumps(res[int(r["seed"])]), flush=True)
        json.dump(res, open(OUT / "T4_partition_null.json", "w"), indent=1)


if __name__ == "__main__":
    main()
