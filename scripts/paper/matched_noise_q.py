#!/usr/bin/env python3
"""Similarity-matched between-superclass label noise: is the cost of between-superclass noise explained by
label similarity?

Lanes: static_noise_s{7,42,137,256,314}: standard_resnet18 (clean), within_sc / between_sc / between_sc_matched
_resnet18_throughout (fixed labels, p = 0.3, same flipped images; matched = flip targets outside the superclass, each
matched in similarity to one sibling; NOISE_matched/targets.json; closes 78% of the similarity gap).
Measures:
  fine-class validation accuracy (devtrain_metrics.json, epochs 0..30; as noise_lib._val_curve), mean over epochs 0-10
  (trapezoid, as noise_lib 'fine_acc_epochs0_10') and at epoch 30;
  20-way superclass probe on avgpool (curriculum_probe.load / probe; identical to static_probe.py) at epochs 5 and 30.
  The probe of clean / between / within is read from STATIC_probe/static_probe.csv; the matched arm is computed here with
  the same code; check: one stored value is recomputed.
Main measure: Q = (cost_matched - cost_within) / (cost_between - cost_within), cost = clean - arm, per seed on fine
accuracy epochs 0-10. Similarity alone predicts Q ~ 0.22; hierarchy predicts Q ~ 1.
Classification: HIERARCHY if 95% t-CI > 0.5 and Q > 0.5 in >= 4/5; SIMILARITY if CI < 0.5 and Q < 0.5 in >= 4/5; else MIXED.
Also: Q on probe epoch 5 and final accuracy; plain contrasts (matched - within, between - matched).
Writes analysis_outputs/paper/NOISE_matched/noise_matched_results.json and noise_matched_per_seed.csv.
Usage: python scripts/paper/matched_noise_q.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curriculum_probe import SUPER_OF, load, probe  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "analysis_outputs" / "paper"
OUT = RESULTS / "NOISE_matched"
EXP = Path(RDT_DATA + "/extended/experiments")
SEEDS = [7, 42, 137, 256, 314]
LANES = {"clean": "standard_resnet18", "within": "within_sc_resnet18_throughout",
         "between": "between_sc_resnet18_throughout", "matched": "between_sc_matched_resnet18_throughout"}


def tci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean()), float(x.mean() - h), float(x.mean() + h)]


def val_curve(lane):
    mm = json.load(open(lane / "dev_snapshots/metrics/devtrain_metrics.json"))
    c = {int(e["epoch"]): float(e["val_accuracy"]) for e in mm}
    snaps = json.load(open(lane / "dev_snapshots/metrics/devtrain_snapshots.json"))
    a0 = [float(z["val_accuracy"]) for z in snaps if z.get("is_t0_baseline") or z.get("epoch") == 0][0]
    return np.array([a0] + [c[e] for e in range(1, 31)])


def probe_at(lane, epoch):
    snaps = json.load(open(lane / "dev_snapshots/metrics/devtrain_snapshots.json"))
    sn = [z for z in snaps if z.get("epoch") == epoch and z.get("snapshot_type") != "final"][0]
    X, y = load(lane / "dev_snapshots" / f"milestone_{sn['milestone_index']}", "avgpool")
    return 100 * probe(X, SUPER_OF[y])


def main():
    sp = pd.read_csv(RESULTS / "STATIC_probe" / "static_probe.csv")
    stored = {(r.seed, {"between_sc": "between", "within_sc": "within"}.get(r.arm, r.arm), int(r.epoch)): 100 * r.super20
              for r in sp.itertuples()}
    rows = []
    for s in SEEDS:
        for arm, nm in LANES.items():
            lane = EXP / f"static_noise_s{s}" / "lanes" / nm
            acc = val_curve(lane)
            row = {"seed": s, "arm": arm, "acc_0_10": float(np.trapezoid(acc[:11], dx=1.0) / 10), "acc_30": float(acc[30])}
            for e in (5, 30):
                row[f"probe_{e}"] = probe_at(lane, e) if arm == "matched" else stored[(s, arm, e)]
            rows.append(row)
            print(row, flush=True)
    check = probe_at(EXP / "static_noise_s7" / "lanes" / LANES["between"], 5)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "noise_matched_per_seed.csv", index=False)
    res = {"check_probe_recomputed_vs_stored_between_s7_ep5": [check, stored[(7, "between", 5)]],
           "means": df.groupby("arm")[["acc_0_10", "acc_30", "probe_5", "probe_30"]].mean().round(3).to_dict("index")}
    P = {m: df.pivot(index="seed", columns="arm", values=m).loc[SEEDS] for m in ("acc_0_10", "acc_30", "probe_5", "probe_30")}
    for m, p in P.items():
        cost = {a: p["clean"] - p[a] for a in ("within", "between", "matched")}
        q = ((cost["matched"] - cost["within"]) / (cost["between"] - cost["within"])).values
        ci = tci(q)
        verdict = ("HIERARCHY" if ci[1] > 0.5 and (q > 0.5).sum() >= 4 else
                   "SIMILARITY" if ci[2] < 0.5 and (q < 0.5).sum() >= 4 else "MIXED")
        res[f"Q_{m}"] = {"per_seed": q.round(3).tolist(), "mean_ci": ci, "median": float(np.median(q)),
                         "pooled": float(cost["matched"].mean() - cost["within"].mean()) / float(cost["between"].mean() - cost["within"].mean()),
                         "verdict": verdict}
        for a, b in (("between", "within"), ("matched", "within"), ("between", "matched"), ("matched", "clean")):
            d = (p[a] - p[b]).values
            res[f"{m}: {a} - {b}"] = {"mean_ci": tci(d), "n_neg": int((d < 0).sum()), "per_seed": d.round(3).tolist()}
    res["primary"] = res["Q_acc_0_10"]
    json.dump(res, open(OUT / "noise_matched_results.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k.startswith("Q_") or k in ("check_probe_recomputed_vs_stored_between_s7_ep5", "means")}, indent=1))
    for k, v in res.items():
        if ":" in k:
            print(k, [round(x, 2) for x in v["mean_ci"]], "neg", v["n_neg"])


if __name__ == "__main__":
    main()
