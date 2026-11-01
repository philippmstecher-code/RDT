#!/usr/bin/env python3
"""Similarity-matched null for the nesting ratio of Extended Data Fig. 6a (pooled-layer SAE): is the observed nesting of class-selective features
in the superclass-selective subspace of their own superclass explained by class means clustering by similarity?

Statistic: identical to nesting_partition_null.py: pooled-layer SAE, the 5 standard lanes of Extended Data Fig. 6a,
milestones 1-6 (epochs 5-30), stored thresholds, exclusive rule; per seed the median over class-selective features
and milestones of e_same / e_other, with each feature's 'superclass' recomputed under the partition (best group by
class-mean |code| summed per group).
Null: 100 partitions of the 100 classes into 20 groups of five whose mean within-group cosine similarity of class
means equals that of the true superclasses (within tol), in the same network: class means of the stored pooled
activations of the 10,000 test images at the final checkpoint (milestone 6, epoch 30) of the lane. Each partition is
built by annealed pair swaps from a random partition (RandomState(p)), accepting a swap when it moves the mean
within-group similarity towards the target and otherwise with probability exp(-delta / T), stopping as soon as the
gap is below tol = 1% of (target - random-partition mean). Reported per partition: achieved similarity, adjusted Rand
index against the true superclasses and the share of within-group class pairs that are also true-superclass pairs.
Writes analysis_outputs/paper/DEPENDENCY/T4_similarity_null.json.
Usage: python scripts/paper/nesting_similarity_null.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nesting_partition_null import LAYER, MS, groups_true, ratio_stat  # noqa: E402
from dependency_lib import OUT, LaneData, selectivity, standard_lanes  # noqa: E402

NPERM = 100
FINAL = 6


def within_mean(S, g):
    tot, n = 0.0, 0
    for k in range(20):
        idx = np.where(g == k)[0]
        sub = S[np.ix_(idx, idx)]
        tot += (sub.sum() - np.trace(sub)) / 2
        n += len(idx) * (len(idx) - 1) // 2
    return tot / n


def matched_partition(S, target, tol, rs, max_iter=200000):
    g = np.empty(100, int)
    for k, row in enumerate(rs.permutation(100).reshape(20, 5)):
        g[row] = k
    cur = within_mean(S, g)
    T = 0.002
    for it in range(max_iter):
        if abs(cur - target) < tol:
            return g, cur, it
        i, j = rs.randint(100), rs.randint(100)
        if g[i] == g[j]:
            continue
        g2 = g.copy()
        g2[i], g2[j] = g[j], g[i]
        new = within_mean(S, g2)
        d = abs(new - target) - abs(cur - target)
        if d < 0 or rs.rand() < np.exp(-d / T):
            g, cur = g2, new
        T *= 0.9999
    return g, cur, max_iter


def pair_overlap(g, gt):
    same = g[:, None] == g[None, :]
    true = gt[:, None] == gt[None, :]
    iu = np.triu_indices(100, 1)
    return float((same[iu] & true[iu]).sum() / same[iu].sum())


def main():
    gt = groups_true()
    res = {}
    for r in standard_lanes():
        lane = LaneData(r["dir"])
        X = lane.raw(FINAL)["avgpool"].astype(np.float64)
        mu = X.reshape(100, 100, -1).mean(1)
        mu /= np.linalg.norm(mu, axis=1, keepdims=True)
        S = mu @ mu.T
        target = within_mean(S, gt)
        rand_mean = np.mean([within_mean(S, np.random.RandomState(p).permutation(np.repeat(np.arange(20), 5)))
                             for p in range(200)])
        tol = 0.01 * (target - rand_mean)
        parts, info = [], []
        for p in range(1, NPERM + 1):
            g, ach, it = matched_partition(S, target, tol, np.random.RandomState(p))
            parts.append(g)
            info.append({"achieved": float(ach), "iters": it, "ari_vs_true": float(adjusted_rand_score(gt, g)),
                         "pair_overlap_true": pair_overlap(g, gt)})
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
        res[int(r["seed"])] = {
            "observed_median_ratio": o, "n_features": len(obs),
            "matched_null_mean": float(nm.mean()), "matched_null_95": [float(np.percentile(nm, 2.5)),
                                                                       float(np.percentile(nm, 97.5))],
            "matched_null_max": float(nm.max()), "n_null": int(len(nm)),
            "rank_of_observed": int((nm >= o).sum()) + 1, "p": float((1 + (nm >= o).sum()) / (1 + len(nm))),
            "similarity_target": float(target), "similarity_random_mean": float(rand_mean), "tol": float(tol),
            "similarity_achieved_mean": float(np.mean([x["achieved"] for x in info])),
            "similarity_achieved_range": [float(min(x["achieved"] for x in info)),
                                          float(max(x["achieved"] for x in info))],
            "ari_vs_true_mean": float(np.mean([x["ari_vs_true"] for x in info])),
            "ari_vs_true_max": float(max(x["ari_vs_true"] for x in info)),
            "pair_overlap_true_mean": float(np.mean([x["pair_overlap_true"] for x in info])),
            "pair_overlap_true_max": float(max(x["pair_overlap_true"] for x in info)),
            "n_not_converged": int(sum(x["iters"] >= 200000 for x in info))}
        print(r["seed"], json.dumps(res[int(r["seed"])]), flush=True)
        json.dump(res, open(OUT / "T4_similarity_null.json", "w"), indent=1)


if __name__ == "__main__":
    main()
