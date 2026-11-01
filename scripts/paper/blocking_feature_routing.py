#!/usr/bin/env python3
"""Do the class-selective features that still form under blocking sit on the remaining coarse structure, and are they
misrouted? Feature-level tests on the RDT of the acquisition-blocking runs (no training; CPU only).

Lanes (SAEs at epochs 0-15, settings of the sham per-epoch runs): sham (sham_perepoch sham_e00_seed{S}) and the
blocking arms gradsc (superclass code), gradrg (random-grouping code), gradenergy (energy-matched)
(blocking_rdt/lanes/{arm}_seed{S}); seeds 7, 42, 137; pooled-layer SAE (avgpool).
Class-selective features: alive (active on >= 1% of the 10,000 test images), CSI above threshold, SAI not above
threshold; thresholds = one set per seed, the sham run's stored set (as blocking_rdt_analysis.py). Each feature's
class = argmax of its class-mean |code|; its superclass = that class's superclass. Selectivity recomputed from the
stored codes (dependency_lib.selectivity, class-major 100 images per class).
Measures per network and epoch (mean over class-selective features):
  off_superclass_share   share of the feature's summed |code| that comes from images of other superclasses
                         (chance 0.95; a feature confined to its superclass gives 0)
  own_direction_ratio    cos^2 between the feature's unit decoder direction and the mean direction of its own
                         superclass in the pooled activations (superclass mean minus grand mean), over the mean cos^2
                         with the other 19 superclasses' mean directions (1 = no preference)
  n_class_selective      number of class-selective features
Expected under the routing account: class features in coarse-blocked networks (gradsc, gradrg) have a higher
off-superclass share and a lower own-direction ratio than in energy-matched and sham networks.
Contrasts paired over the 3 seeds (mean, t-based 95% CI, seeds in direction): gradsc - gradenergy, gradrg -
gradenergy, gradsc - sham, gradrg - sham, gradenergy - sham; windows epochs 1-3, 1-10 (block), 11-15 (release).
Writes analysis_outputs/paper/RDT_blocking/feature_routing_long.csv and feature_routing_summary.json.
Usage: python scripts/paper/blocking_feature_routing.py
"""
import json
import os
import sys
from pathlib import Path
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curriculum_probe import SUPER_OF, load  # noqa: E402
from curriculum_recount import load_lane, thr_set  # noqa: E402
from dependency_lib import selectivity  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis_outputs/paper/RDT_blocking"
E = Path(RDT_DATA + "/extended/experiments")
SEEDS = [7, 42, 137]
LANES = {"sham": "sham_perepoch/lanes/sham_e00_seed{s}", "gradsc": "blocking_rdt/lanes/gradsc_seed{s}",
         "gradrg": "blocking_rdt/lanes/gradrg_seed{s}", "gradenergy": "blocking_rdt/lanes/gradenergy_seed{s}",
         "gradrand": "blocking_rdt/lanes/gradrand_seed{s}",
         "gradsc_l2": "blocking_rdt/lanes/gradsc_l2_seed{s}",
         "gradenergy_l2": "blocking_rdt/lanes/gradenergy_l2_seed{s}"}
LAYER = "avgpool"
EPOCHS = range(1, 16)
CONTRASTS = [("gradsc", "gradenergy"), ("gradrg", "gradenergy"), ("gradsc", "sham"), ("gradrg", "sham"),
             ("gradenergy", "sham"), ("gradrand", "sham"), ("gradsc", "gradrand"), ("gradrg", "gradrand"),
             ("gradsc_l2", "gradenergy_l2"), ("gradsc_l2", "sham"), ("gradenergy_l2", "sham")]
WINDOWS = {"epochs 1-3": range(1, 4), "block (1-10)": range(1, 11), "release (11-15)": range(11, 16)}


def measure(lane_dir, e, t):
    sae = lane_dir / "sae_analysis"
    H = np.load(sae / "activations" / str(e) / f"{LAYER}.npy").astype(np.float64)
    s = selectivity(H)
    alive = (np.abs(H) > 1e-6).mean(0) >= 0.01
    cls = alive & (s["csi"] > t["csi_differentiation_thresh"]) & ~(s["sai"] > t["sai_task_general_thresh"])
    idx = np.where(cls)[0]
    out = {"n_class_selective": int(len(idx))}
    if len(idx) == 0:
        return {**out, "off_superclass_share": np.nan, "own_direction_ratio": np.nan}
    M = s["M"]                                            # (F, 100) class-mean |code|
    best = M[idx].argmax(1)
    own_sc = SUPER_OF[best]
    A = np.abs(H[:, idx])                                 # images x features
    img_sc = SUPER_OF[np.repeat(np.arange(100), 100)]
    off = np.array([A[img_sc != own_sc[j], j].sum() / A[:, j].sum() for j in range(len(idx))])
    X, y = load(lane_dir / "dev_snapshots" / f"milestone_{e}", LAYER)
    X = X.astype(np.float64)
    mu = X.mean(0)
    D_sc = np.stack([X[SUPER_OF[y] == c].mean(0) - mu for c in range(20)])
    D_sc /= np.linalg.norm(D_sc, axis=1, keepdims=True)
    dec = torch.load(sae / "saes" / str(e) / f"{LAYER}.pt", map_location="cpu", weights_only=True)["decoder.weight"]
    dec = torch.nn.functional.normalize(dec.double(), dim=0).numpy()[:, idx]     # (C, n)
    cos2 = (D_sc @ dec) ** 2                                                       # (20, n)
    own = cos2[own_sc, np.arange(len(idx))]
    other = (cos2.sum(0) - own) / 19
    return {**out, "off_superclass_share": float(off.mean()), "own_direction_ratio": float(np.median(own / other))}


def tci(d):
    d = np.asarray(d, float)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return [float(d.mean()), float(d.mean() - h), float(d.mean() + h)]


def main():
    rows = []
    for s in SEEDS:
        t = thr_set(load_lane(E / LANES["sham"].format(s=s)))["thresholds"]
        for arm, pat in LANES.items():
            d = E / pat.format(s=s)
            for e in EPOCHS:
                rows.append({"arm": arm, "seed": s, "epoch": e, **measure(d, e, t)})
            print(arm, s, rows[-1], flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "feature_routing_long.csv", index=False)
    res = {}
    for w, eps in WINDOWS.items():
        M = D[D.epoch.isin(list(eps))].groupby(["arm", "seed"]).mean(numeric_only=True)
        res[f"levels|{w}"] = M.groupby("arm").mean().round(4).to_dict()
        for a, b in CONTRASTS:
            for k in ("off_superclass_share", "own_direction_ratio", "n_class_selective"):
                d = (M.loc[a][k] - M.loc[b][k]).loc[SEEDS].values
                res[f"{w}|{a} - {b}|{k}"] = {"mean_ci": tci(d), "n_pos": int((d > 0).sum()),
                                             "per_seed": np.round(d, 4).tolist()}
    json.dump(res, open(OUT / "feature_routing_summary.json", "w"), indent=1)
    for k, v in res.items():
        if k.startswith("levels"):
            print(k, json.dumps(v))
        else:
            print(f"{k:70s} {v['mean_ci'][0]:+8.4f} [{v['mean_ci'][1]:+8.4f}, {v['mean_ci'][2]:+8.4f}] pos {v['n_pos']}/3")


if __name__ == "__main__":
    main()
