#!/usr/bin/env python3
"""Class hierarchy, step 3: why does training on random groupings give more true-superclass-selective SAE features
than training on true superclasses? Open seeds only.

Per alive feature (stored avgpool SAE codes, and raw avgpool units as the dictionary-free analogue):
  breadth   effective number of classes exp(H(p)), p = class-mean |activation| / sum
  top1      largest class fraction
  ssi_true  Phase-4 SSI on the true superclasses (identical formula: normalised best-group mass fraction)
  ssi_imp   same on the seed's imposed random partition
  ssi_null  mean / p95 of SSI over 30 random 20x5 partitions (a feature-specific null: what its breadth
            alone gives); ssi_exc = ssi_true - ssi_null_mean
  cover     within the best true superclass, number of its 5 classes carrying >= 50% of the best class's mass
Writes cache/sae_feat_<lane>_s<seed>_ms<ms>_<src>.csv.gz and sae_summary.csv.
"""
from __future__ import annotations

import ast
import json

import numpy as np
import pandas as pd

from hierarchy_lib import OUT, SEEDS, imposed_vec, lane_dir, load_acts, null_partitions, snapshots, true_vec

EPS = 1e-10
THR = json.load(open(OUT.parents[1] / "SAE_outcomes" / "thresholds_EC.json"))["common"]
LANES = ["sc_true_K20", "sc_rand_K20", "scratch_E30", "taskirr"]
N_NULL = 30


def masks(g):
    return np.stack([g == k for k in np.unique(g)]).astype(np.float64)


def ssi(M, mk):
    tot = M.sum(1)
    frac = (M @ mk.T) / (tot[:, None] + EPS)
    n = mk.shape[0]
    return np.maximum((frac.max(1) - 1 / n) / (1 - 1 / n), 0), frac.argmax(1)


def feature_table(H, y, tv, iv, nulls):
    act = np.abs(H) > 1e-6
    alive = act.sum(0) >= 100
    M = np.stack([np.abs(H[y == c]).mean(0) for c in range(100)], 1)[alive]  # f x 100
    dens = act[:, alive].mean(0)
    p = M / (M.sum(1, keepdims=True) + EPS)
    breadth = np.exp(-(np.where(p > 0, p * np.log(p + EPS), 0)).sum(1))
    st, best = ssi(M, masks(tv))
    si, _ = ssi(M, masks(iv))
    nl = np.stack([ssi(M, masks(g))[0] for g in nulls], 1)
    mt = masks(tv).astype(bool)
    cover = np.array([(M[i, mt[b]] >= 0.5 * M[i, mt[b]].max()).sum() for i, b in enumerate(best)])
    return pd.DataFrame({"density": dens, "breadth": breadth, "top1": p.max(1), "ssi_true": st, "ssi_imp": si,
                         "ssi_null_mean": nl.mean(1), "ssi_null_p95": np.percentile(nl, 95, axis=1),
                         "cover": cover})


def summarise(ft, thr):
    hi = ft.ssi_true > thr
    out = {"n_alive": len(ft), "census_ssi": hi.mean(), "census_ssi_imp": (ft.ssi_imp > thr).mean(),
           "census_ssi_null": (ft.ssi_null_mean > thr).mean(),
           "census_genuine": (hi & (ft.ssi_true > ft.ssi_null_p95)).mean(),
           "census_imp_genuine": ((ft.ssi_imp > thr) & (ft.ssi_imp > ft.ssi_null_p95)).mean(),
           "mean_ssi_exc": (ft.ssi_true - ft.ssi_null_mean).mean(),
           "mean_breadth": ft.breadth.mean(), "median_breadth": ft.breadth.median(),
           "frac_narrow": (ft.breadth < 5).mean(), "mean_density": ft.density.mean(),
           "hi_frac_single_class": ((ft.cover == 1) & hi).sum() / max(hi.sum(), 1),
           "hi_frac_multi_class": ((ft.cover >= 3) & hi).sum() / max(hi.sum(), 1),
           "hi_mean_breadth": ft.breadth[hi].mean() if hi.any() else np.nan}
    return out


def main():
    tv = true_vec()
    nulls = null_partitions(N_NULL, tv, 70000)
    rows = []
    for s in SEEDS:
        iv = imposed_vec(s)
        thr = THR[str(s)]["thresholds"]["ssi_abstraction_thresh"]
        for kind in LANES:
            d = lane_dir(kind, s)
            if not d.exists():
                continue
            have_sae = (d / "sae_analysis" / "sae_results.json").exists()
            for sn in snapshots(d):
                X, y = load_acts(d, sn["ms"], ["avgpool"])
                srcs = {"raw": X["avgpool"]}
                ap = d / "sae_analysis" / "activations" / str(sn["ms"]) / "avgpool.npy"
                if have_sae and ap.exists():
                    srcs["sae"] = np.load(ap)
                for src, H in srcs.items():
                    ft = feature_table(H, y, tv, iv, nulls)
                    if sn["epoch"] in (0, 20, 50) or (kind == "scratch_E30" and sn["epoch"] == 30):
                        ft.to_csv(OUT / "cache" / f"sae_feat_{kind}_s{s}_ms{sn['ms']}_{src}.csv.gz", index=False)
                    rows.append({"lane": kind, "seed": s, "ms": sn["ms"], "epoch": sn["epoch"], "src": src,
                                 "thr": thr, **summarise(ft, thr)})
                print(f"{kind} s{s} ms{sn['ms']} {list(srcs)}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "cache" / "sae_summary.csv", index=False)


if __name__ == "__main__":
    main()
