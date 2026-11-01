#!/usr/bin/env python3
"""Pre-building, RDT read-out of class features: confinement of class features to their superclass in pre-built
networks from the first fine-class epoch, and their lead over scratch in epochs.

Same definitions as rdt_pace.py (census categories under one common threshold set per seed, census layers layer1-3 +
pooled, alive = active on >= 1%; confined class feature = leave-own-class-out sibling enrichment above the 95th
percentile of the same feature under 40 random groupings). Thresholds: the scratch run's stored set per seed.
Lanes (curriculum, five seeds): sc_true_K20 and sc_rand_K20 (20 epochs on the superclasses or a random grouping,
then fine classes; milestones 4, 5, 6, 7, 8 = fine epochs 0, 1, 5, 10, 15) and scratch_E30 (epochs 0, 5, 10, 15).
Measures per lane and fine epoch: n class-selective, n confined, share confined, n superclass-selective; epochs ahead
of scratch in confined class features (scratch trajectory, running maximum, linear interpolation).
Writes analysis_outputs/paper/RDT_pace/curriculum_{long.csv, summary.json}.
Usage: python scripts/paper/rdt_pace_curriculum.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lineage"))
sys.path.insert(0, str(HERE))
import lineage_lib as X  # noqa: E402

OUT = X.REPO / "analysis_outputs/paper/RDT_pace"
L = Path(RDT_DATA + "/extended/experiments/curriculum/lanes")
NULL = X.random_partitions(40, seed0=101)


def snaps(d):
    s = json.load(open(d / "dev_snapshots" / "metrics" / "devtrain_snapshots.json"))
    s = s.get("snapshots", s) if isinstance(s, dict) else s
    return {int(x["milestone_index"]): int(x["epoch"]) for x in s if x.get("snapshot_type") != "final"}


def thresholds(s):
    r = json.load(open(L / f"scratch_E30_seed{s}" / "sae_analysis" / "sae_results.json"))["metadata"]["analysis_thresholds"]
    r = r if isinstance(r, dict) else json.loads(r)
    return {k: float(r[k]) for k in ("ssi_abstraction_thresh", "csi_differentiation_thresh", "sai_task_general_thresh")}


def census(d, m, t):
    n_sc = n_cl = n_conf = 0
    for layer in X.LAYERS:
        H = np.load(d / "sae_analysis" / "activations" / str(m) / f"{layer}.npy")
        M = np.abs(H).reshape(100, 100, -1).mean(1)            # (100 classes, F), class-major codes
        alive = (np.abs(H) > 1e-6).sum(0) >= 100
        sel = X.sel_from_M(M)
        mk = X.masks(sel, alive, t)
        n_sc += int(mk["sc"].sum())
        idx = np.where(mk["cl"])[0]
        n_cl += len(idx)
        if len(idx):
            k = sel["best_class"][idx]
            obs = X.sib_enrich(M, idx, k)
            nul = np.stack([X.sib_enrich(M, idx, k, g) for g in NULL])
            n_conf += int((obs > np.quantile(nul, 0.95, axis=0)).sum())
    return n_sc, n_cl, n_conf


def main():
    rows = []
    for s in X.SEEDS:
        t = thresholds(s)
        for arm, lane, pre in (("true", f"sc_true_K20_seed{s}", 20), ("rand", f"sc_rand_K20_seed{s}", 20),
                               ("scratch", f"scratch_E30_seed{s}", 0)):
            d = L / lane
            for m, ep in sorted(snaps(d).items()):
                fe = ep - pre
                if fe not in (0, 1, 5, 10, 15) or not (d / "sae_analysis" / "activations" / str(m)).exists():
                    continue
                n_sc, n_cl, n_conf = census(d, m, t)
                rows.append({"arm": arm, "seed": s, "fine_epoch": fe, "milestone": m, "n_sc": n_sc, "n_cl": n_cl,
                             "n_cl_confined": n_conf, "share_confined": n_conf / n_cl if n_cl else np.nan})
            print(arm, s, [r for r in rows if r["arm"] == arm and r["seed"] == s], flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "curriculum_long.csv", index=False)
    res = {"levels": D.groupby(["arm", "fine_epoch"])[["n_sc", "n_cl", "n_cl_confined", "share_confined"]].mean()
           .round(3).reset_index().to_dict("records")}
    for arm in ("true", "rand"):
        for fe in (5, 10):
            for mtr in ("share_confined", "n_cl_confined", "n_cl"):
                a = D[(D.arm == arm) & (D.fine_epoch == fe)].set_index("seed")[mtr]
                b = D[(D.arm == "scratch") & (D.fine_epoch == fe)].set_index("seed")[mtr]
                res[f"{arm} - scratch|fine epoch {fe}|{mtr}"] = X.tci((a - b).loc[X.SEEDS].values)
        ahead = []
        for s in X.SEEDS:
            sc = D[(D.arm == "scratch") & (D.seed == s)].sort_values("fine_epoch")
            x, y = sc.fine_epoch.values.astype(float), np.maximum.accumulate(sc.n_cl_confined.values.astype(float))
            for fe in (1, 5):
                v = float(D[(D.arm == arm) & (D.seed == s) & (D.fine_epoch == fe)].n_cl_confined.iloc[0])
                eq = float(np.interp(v, y, x)) if v <= y[-1] else np.nan
                ahead.append({"seed": s, "fine_epoch": fe, "scratch_epoch_equivalent": eq, "epochs_ahead": eq - fe})
        A = pd.DataFrame(ahead)
        for fe in (1, 5):
            res[f"{arm}|epochs ahead in confined class features|fine epoch {fe}"] = X.tci(
                A[A.fine_epoch == fe].epochs_ahead.values)
    json.dump(res, open(OUT / "curriculum_summary.json", "w"), indent=1, default=str)
    for k, v in res.items():
        if k != "levels":
            print(f"{k:62s} {v['mean']:+.3f} {np.round(v['ci95'], 3).tolist()} pos {v.get('n_pos')} neg {v.get('n_neg')}")
    print(pd.DataFrame(res["levels"]).to_string(index=False))


if __name__ == "__main__":
    main()
