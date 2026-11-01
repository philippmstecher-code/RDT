#!/usr/bin/env python3
"""Within-network dose-response under blocking: do the superclasses whose coarse structure is damaged most also lose
the most class-level features?

Per seed (7, 42, 137), arm (gradsc, gradrg, gradenergy; blocking_rdt) and its sham (sham_perepoch sham_e00),
per superclass c (20 CIFAR-100 superclasses), averaged over the checkpoints of a window (block = epochs 1-10,
release = 11-15):
  misassign_c   share of test images of superclass c that the network's own classifier assigns to another
                superclass (stored predictions, sample_predictions.pt; 500 images per superclass)
  n_sc_c        superclass-selective features whose best superclass is c (SSI above, CSI not above threshold, SAI
                not above threshold, alive), census layers (layer1-3, pooled)
  n_cls_c       class-selective features whose class lies in superclass c, census layers
Thresholds: one set per seed, the sham run's stored set (as blocking_rdt_analysis.py). Selectivity recomputed from
the stored SAE codes (dependency_lib.selectivity).
Per superclass, damage and loss relative to the sham of the same seed:
  d_misassign = misassign(arm) - misassign(sham)                    (percentage points; damage to coarse routing)
  d_sc        = log((n_sc(arm) + 1) / (n_sc(sham) + 1))              (loss of superclass features)
  d_cls       = log((n_cls(arm) + 1) / (n_cls(sham) + 1))            (loss of class features)
Per seed and arm, Spearman rho over the 20 superclasses: d_misassign vs d_cls (expected negative: more misrouting,
more class-feature loss), d_sc vs d_cls (expected positive), and partial rho of d_misassign vs d_cls controlling
for the sham class-feature count (baseline size). Pooled: Fisher-z mean over seeds.
Writes analysis_outputs/paper/RDT_blocking/superclass_doseresponse_long.csv and _summary.json.
Usage: python scripts/paper/blocking_superclass_doseresponse.py
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
from curriculum_probe import SUPER_OF  # noqa: E402
from curriculum_recount import load_lane, thr_set  # noqa: E402
from dependency_lib import selectivity  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis_outputs/paper/RDT_blocking"
E = Path(RDT_DATA + "/extended/experiments")
SEEDS = [7, 42, 137]
LANES = {"sham": "sham_perepoch/lanes/sham_e00_seed{s}", "gradsc": "blocking_rdt/lanes/gradsc_seed{s}",
         "gradrg": "blocking_rdt/lanes/gradrg_seed{s}", "gradenergy": "blocking_rdt/lanes/gradenergy_seed{s}"}
CENSUS = ("layer1", "layer2", "layer3", "avgpool")
WINDOWS = {"block": range(1, 11), "release": range(11, 16)}
Y = np.repeat(np.arange(100), 100)


def per_checkpoint(d, e, t):
    p = torch.load(d / "dev_snapshots" / f"milestone_{e}" / "sample_predictions.pt", map_location="cpu",
                   weights_only=False)
    pred = np.concatenate([p[c]["predictions"].numpy() for c in range(100)])
    assert len(pred) == 10000
    wrong = SUPER_OF[pred] != SUPER_OF[Y]
    mis = np.array([wrong[SUPER_OF[Y] == c].mean() for c in range(20)])
    n_sc, n_cls = np.zeros(20), np.zeros(20)
    for L in CENSUS:
        H = np.load(d / "sae_analysis" / "activations" / str(e) / f"{L}.npy").astype(np.float64)
        s = selectivity(H)
        alive = (np.abs(H) > 1e-6).mean(0) >= 0.01
        tg = s["sai"] > t["sai_task_general_thresh"]
        cs = s["csi"] > t["csi_differentiation_thresh"]
        sc = alive & (s["ssi"] > t["ssi_abstraction_thresh"]) & ~cs & ~tg
        cl = alive & cs & ~tg
        M = s["M"]
        sc_of = np.stack([M[:, SUPER_OF == c].sum(1) for c in range(20)], 1).argmax(1)
        cls_sc = SUPER_OF[M.argmax(1)]
        n_sc += np.bincount(sc_of[sc], minlength=20)
        n_cls += np.bincount(cls_sc[cl], minlength=20)
    return mis, n_sc, n_cls


def partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    def res(a, b):
        b1 = np.c_[np.ones_like(b), b]
        return a - b1 @ np.linalg.lstsq(b1, a, rcond=None)[0]
    return float(stats.pearsonr(res(rx, rz), res(ry, rz))[0])


def main():
    rows = []
    for s in SEEDS:
        t = thr_set(load_lane(E / LANES["sham"].format(s=s)))["thresholds"]
        for arm, pat in LANES.items():
            d = E / pat.format(s=s)
            for e in range(1, 16):
                mis, nsc, ncl = per_checkpoint(d, e, t)
                for c in range(20):
                    rows.append({"arm": arm, "seed": s, "epoch": e, "superclass": c, "misassign": 100 * mis[c],
                                 "n_sc": nsc[c], "n_cls": ncl[c]})
            print(arm, s, flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "superclass_doseresponse_long.csv", index=False)
    res = {}
    for w, eps in WINDOWS.items():
        A = D[D.epoch.isin(list(eps))].groupby(["arm", "seed", "superclass"]).mean(numeric_only=True)
        for arm in ("gradsc", "gradrg", "gradenergy"):
            per = []
            for s in SEEDS:
                a, b = A.loc[(arm, s)], A.loc[("sham", s)]
                dm = (a.misassign - b.misassign).values
                dsc = np.log((a.n_sc + 1) / (b.n_sc + 1)).values
                dcl = np.log((a.n_cls + 1) / (b.n_cls + 1)).values
                per.append({"seed": s,
                            "rho_misassign_vs_cls": float(stats.spearmanr(dm, dcl)[0]),
                            "p_misassign_vs_cls": float(stats.spearmanr(dm, dcl)[1]),
                            "partial_rho_misassign_vs_cls_given_baseline": partial_spearman(dm, dcl, b.n_cls.values),
                            "rho_sc_vs_cls": float(stats.spearmanr(dsc, dcl)[0]),
                            "p_sc_vs_cls": float(stats.spearmanr(dsc, dcl)[1]),
                            "mean_d_misassign": float(dm.mean()), "mean_d_cls": float(dcl.mean())})
            z = lambda k: float(np.tanh(np.mean([np.arctanh(np.clip(p[k], -0.999, 0.999)) for p in per])))
            res[f"{w}|{arm}"] = {"per_seed": per,
                                 "pooled_rho_misassign_vs_cls": z("rho_misassign_vs_cls"),
                                 "pooled_partial_rho": z("partial_rho_misassign_vs_cls_given_baseline"),
                                 "pooled_rho_sc_vs_cls": z("rho_sc_vs_cls"),
                                 "n_seeds_neg_misassign_vs_cls": int(sum(p["rho_misassign_vs_cls"] < 0 for p in per)),
                                 "n_seeds_pos_sc_vs_cls": int(sum(p["rho_sc_vs_cls"] > 0 for p in per))}
    json.dump(res, open(OUT / "superclass_doseresponse_summary.json", "w"), indent=1)
    for k, v in res.items():
        print(k, "| rho(misassign, cls) per seed", [round(p["rho_misassign_vs_cls"], 2) for p in v["per_seed"]],
              "pooled", round(v["pooled_rho_misassign_vs_cls"], 2),
              "| partial", [round(p["partial_rho_misassign_vs_cls_given_baseline"], 2) for p in v["per_seed"]],
              "| rho(sc, cls)", [round(p["rho_sc_vs_cls"], 2) for p in v["per_seed"]],
              "| mean dmis", round(np.mean([p["mean_d_misassign"] for p in v["per_seed"]]), 1))


if __name__ == "__main__":
    main()
