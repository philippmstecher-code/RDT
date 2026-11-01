#!/usr/bin/env python3
"""Pace of coarse and fine structure in RDT terms, as epochs behind the control.

Raw class-feature counts cannot carry pace: under damage the network grows extra, misrouted class features (blocking:
+50% class events against the jointly matched block). This script counts only class features confined to their
superclass and reads coarse and fine structure, both from RDT, in epochs behind the control.

Definitions:
  superclass-selective, class-selective   RDT census categories under one common threshold set per seed (the sham's;
                                          lineage_lib.masks), census layers layer1-3 + pooled, alive = active on >= 1%
  confined class feature                  class-selective feature whose leave-own-class-out sibling enrichment
                                          (lineage_lib.sib_enrich: share of its non-own-class code on the 4 sibling
                                          classes, over 4/99) exceeds the 95th percentile of the same feature's
                                          enrichment under 40 random groupings of the classes into 20 groups of five
  coarse measures                         number of superclass-selective features; cumulative superclass events
                                          (Ab-E, all layers, common thresholds; RDT_completion/per_lane recount)
  fine measure                            number of confined class features
  epochs behind                           e - the epoch at which the control trajectory (running maximum, linear
                                          interpolation) reaches the arm's value at epoch e; controls: jointly matched
                                          block and sham; windows end of block (8-10) and release (11-15)
Data: lineage_lib cache (class-mean |code| per feature, epochs 0-15) for sham, gradsc, gradjoint, gradrg, 5 seeds.
Writes analysis_outputs/paper/RDT_pace/{long.csv, summary.json}.
Usage: python scripts/paper/rdt_pace.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lineage"))
import lineage_lib as X  # noqa: E402

OUT = X.REPO / "analysis_outputs/paper/RDT_pace"
ARMS = ["sham", "gradsc", "gradjoint", "gradrg"]
NULL = X.random_partitions(40, seed0=101)


def census(cache, e, t):
    n_sc = n_cl = n_conf = 0
    sib_all = []
    for L in X.LAYERS:
        M = cache.M(L, e)                                        # (100, F)
        sel = X.sel_from_M(M)
        m = X.masks(sel, cache.alive(L, e), t)
        n_sc += int(m["sc"].sum())
        idx = np.where(m["cl"])[0]
        n_cl += len(idx)
        if len(idx):
            k = sel["best_class"][idx]
            obs = X.sib_enrich(M, idx, k)
            nul = np.stack([X.sib_enrich(M, idx, k, g) for g in NULL])   # (40, n)
            thr = np.quantile(nul, 0.95, axis=0)
            n_conf += int((obs > thr).sum())
            sib_all += obs.tolist()
    return n_sc, n_cl, n_conf, float(np.median(sib_all)) if sib_all else np.nan


def cum_abe(arm, s):
    d = json.load(open(X.PER_LANE / f"{arm}_seed{s}.json"))
    ev = {int(k.split("->")[1]): sum(v["common"][L]["ab_h"] for L in v["common"]) for k, v in d["events"].items()}
    c, out = 0, {0: 0}
    for e in range(1, 16):
        c += ev[e]
        out[e] = c
    return out


def behind(arm_vals, ctrl_vals):
    """epochs behind: for each epoch e, e minus the epoch at which the control (running max) reaches arm_vals[e]."""
    eps = np.arange(16)
    c = np.maximum.accumulate(np.asarray(ctrl_vals, float))
    out = []
    for e, v in zip(eps, arm_vals):
        if v >= c[-1]:
            out.append(e - 15.0 if v == c[-1] else np.nan)
            continue
        j = np.searchsorted(c, v, side="left")
        if j == 0:
            out.append(float(e))
            continue
        x0, x1, y0, y1 = eps[j - 1], eps[j], c[j - 1], c[j]
        out.append(e - (x0 + (v - y0) / (y1 - y0) if y1 > y0 else x1))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in X.SEEDS:
        t = X.common_thresholds(s)
        for arm in ARMS:
            cache, cab = X.Cache(arm, s), cum_abe(arm, s)
            for e in X.EPOCHS:
                n_sc, n_cl, n_conf, sib = census(cache, e, t)
                rows.append({"arm": arm, "seed": s, "epoch": e, "n_sc": n_sc, "n_cl": n_cl, "n_cl_confined": n_conf,
                             "share_confined": n_conf / n_cl if n_cl else np.nan, "sib_median": sib,
                             "cum_AbE": cab[e]})
            print(arm, s, rows[-1], flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "long.csv", index=False)
    res = {"levels_by_arm_epoch": D.groupby(["arm", "epoch"])[["n_sc", "n_cl", "n_cl_confined", "share_confined",
                                                               "cum_AbE"]].mean().round(2).reset_index()
           .to_dict("records")}
    lag_rows = []
    for arm in ("gradsc", "gradrg", "gradjoint"):
        for ctrl in ("gradjoint", "sham"):
            if arm == ctrl:
                continue
            for s in X.SEEDS:
                a = D[(D.arm == arm) & (D.seed == s)].sort_values("epoch")
                c = D[(D.arm == ctrl) & (D.seed == s)].sort_values("epoch")
                for m in ("n_sc", "cum_AbE", "n_cl_confined", "n_cl"):
                    lb = behind(a[m].values, c[m].values)
                    for e, v in zip(a.epoch, lb):
                        lag_rows.append({"arm": arm, "control": ctrl, "seed": s, "measure": m, "epoch": int(e),
                                         "epochs_behind": v})
    Lg = pd.DataFrame(lag_rows)
    Lg.to_csv(OUT / "epochs_behind_long.csv", index=False)
    for (arm, ctrl, m), g in Lg.groupby(["arm", "control", "measure"]):
        for w, (lo, hi) in {"end of block (8-10)": (8, 10), "release (11-15)": (11, 15), "block (1-10)": (1, 10)}.items():
            per = g[g.epoch.between(lo, hi)].groupby("seed").epochs_behind.mean()
            res[f"{arm} vs {ctrl}|{m}|{w}"] = {**X.tci(per.values), "per_seed": per.round(2).to_dict(),
                                               "n_nan_points": int(g[g.epoch.between(lo, hi)].epochs_behind.isna().sum())}
    json.dump(res, open(OUT / "summary.json", "w"), indent=1, default=str)
    for k, v in res.items():
        if "|" in k and "vs gradjoint" in k or ("vs sham" in k and "gradjoint vs" in k):
            print(k, {x: v.get(x) for x in ("mean", "ci95", "n_pos", "n_neg", "n_nan_points")})


if __name__ == "__main__":
    main()
