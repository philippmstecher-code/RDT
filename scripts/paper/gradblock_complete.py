#!/usr/bin/env python3
"""Acquisition blocking, completed experiment: all arms to epoch 30, a random-grouping arm, and a stage-matched dose
control.

Arms (ResNet-18, CIFAR-100, seeds 7, 42, 137, 256, 314; branch epoch 0; block during epochs 1-10; the forward pass is
never changed; learning-rate schedule of the 30-epoch runs):
  gradsc      learning signal blocked along the superclass code (layer1-4 + pooled)    epochs 1-15 gradblock,
  gradrand    along as many random directions                                           epochs 16-30 from the
  gradenergy  along directions outside the code carrying a matched share of the signal  continuation lanes
                                                                                         gradcont_<arm>_r1 (resumed
                                                                                         from epoch 15, no block)
  gradrg      along the code of one fixed random grouping per seed (RandomState(10000 + seed), as the suppression
              arm), re-estimated like the superclass arm; trained straight to epoch 30 (gradrg)
  gradsc_l2, gradenergy_l2   stage-matched dose control: superclass and energy-matched blocks at layer2-pooled only
              (where the energy-matched arm matches the superclass arm's share), epochs 1-15 (gradl2)
  sham        critical_sham sham_e00 (same branch and schedule, no block)
Measures per arm, seed, epoch: the network's fine-class validation accuracy (every epoch; valid because the forward
pass is unchanged) and, at stored checkpoints, a fresh 100-way ridge read-out (5-fold CV) of the pooled activations of
the 10,000 test images split into superclass assignment and within-superclass accuracy (as gradblock_analysis.py).
Contrasts: per-seed means over windows (epochs 1-3, 1-10, 11-15, 16-20, 21-25, 26-30, and epoch 30), paired over
seeds (t-based 95% CI, seeds in the predicted direction).
Writes analysis_outputs/paper/GRADBLOCK/final_long.csv, final_summary.json; Source Data gradblock_final_long.csv.
Usage: python scripts/paper/gradblock_complete.py
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
from scipy import stats
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coarse_overlap import milestones  # noqa: E402
from class_level import FINE2SC  # noqa: E402
from curriculum_probe import load  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis_outputs/paper/GRADBLOCK"
SD = REPO / "analysis_outputs/paper/figures/final/source_data"
E = Path(RDT_DATA + "/extended/experiments")
SEEDS = [7, 42, 137, 256, 314]
L5 = "Llayer1-layer2-layer3-layer4-avgpool"
L4 = "Llayer2-layer3-layer4-avgpool"


def lanes(arm, s):
    """Lane directories of an arm and seed, in training order (a continued arm has two)."""
    if arm == "sham":
        return [E / "critical_sham/lanes" / f"sham_e00_seed{s}"]
    if arm in ("gradsc", "gradrand", "gradenergy"):
        src = "gradenergy" if arm == "gradenergy" else "gradblock"
        return [E / src / "lanes" / f"lesion_{arm}_e00_seed{s}_rp65_{L5}",
                E / f"gradcont_{arm}_r1/lanes" / f"sham_e15_seed{s}"]
    if arm == "gradrg":
        return [E / "gradrg/lanes" / f"lesion_gradrg_e00_seed{s}_rp65_{L5}"]
    if arm in ("gradsc_l2", "gradenergy_l2"):
        return [E / "gradl2/lanes" / f"lesion_{arm[:-3]}_e00_seed{s}_rp65_{L4}"]
    raise KeyError(arm)


ARMS = ["sham", "gradsc", "gradrand", "gradenergy", "gradrg", "gradsc_l2", "gradenergy_l2"]


def tci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean()), float(x.mean() - h), float(x.mean() + h)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    cache = OUT / "final_long.csv"
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame()
    have = set(zip(done.arm, done.seed, done.epoch)) if len(done) else set()
    rows = done.to_dict("records") if len(done) else []
    for arm in ARMS:
        for s in SEEDS:
            acc, ms = {}, {}
            for d in lanes(arm, s):
                f = d / "dev_snapshots/metrics/devtrain_metrics.json"
                if not f.exists():
                    continue
                for e in json.load(open(f)):
                    acc[int(e["epoch"])] = float(e["val_accuracy"])
                ms.update(milestones(d))
            if not acc:
                print("missing", arm, s, flush=True)
                continue
            for e in sorted(acc):
                if (arm, s, e) in have:
                    continue
                r = {"arm": arm, "seed": s, "epoch": e, "val_accuracy": acc[e]}
                if e in ms and (ms[e] / "individual_multilayer_activations.pt").exists():
                    X, y = load(ms[e], "avgpool")
                    pred = cross_val_predict(make_pipeline(StandardScaler(), RidgeClassifier(alpha=10.0)), X, y, cv=cv)
                    ok = FINE2SC[pred] == FINE2SC[y]
                    r.update(fresh_fine=100 * float((pred == y).mean()), fresh_assignment=100 * float(ok.mean()),
                             fresh_within=100 * float((pred == y)[ok].mean()))
                rows.append(r)
            print(arm, s, "done", flush=True)
    D = pd.DataFrame(rows).sort_values(["arm", "seed", "epoch"]).reset_index(drop=True)
    D.to_csv(cache, index=False)
    D.to_csv(SD / "gradblock_final_long.csv", index=False)
    res = {}
    wins = {"epochs_1_3": range(1, 4), "epochs_1_10": range(1, 11), "epochs_11_15": range(11, 16),
            "epochs_16_20": range(16, 21), "epochs_21_25": range(21, 26), "epochs_26_30": range(26, 31),
            "epoch_30": [30]}
    pairs = [("gradsc", "sham"), ("gradsc", "gradenergy"), ("gradsc", "gradrand"), ("gradsc", "gradrg"),
             ("gradrg", "sham"), ("gradrg", "gradenergy"), ("gradrg", "gradrand"), ("gradenergy", "sham"),
             ("gradrand", "sham"), ("gradsc_l2", "gradenergy_l2"), ("gradsc_l2", "sham"), ("gradenergy_l2", "sham")]
    for w, eps in wins.items():
        for key in ("val_accuracy", "fresh_fine", "fresh_assignment", "fresh_within"):
            if key not in D:
                continue
            M = D[D.epoch.isin(list(eps))].groupby(["arm", "seed"])[key].mean()
            for a, b in pairs:
                if a not in M.index.get_level_values(0) or b not in M.index.get_level_values(0):
                    continue
                seeds = sorted(set(M.loc[a].dropna().index) & set(M.loc[b].dropna().index))
                if len(seeds) < 2:
                    continue
                d = (M.loc[a].loc[seeds] - M.loc[b].loc[seeds]).values
                res[f"{w}|{a} - {b}|{key}"] = {"mean_ci": tci(d), "n_negative": int((d < 0).sum()), "seeds": seeds}
    json.dump(res, open(OUT / "final_summary.json", "w"), indent=1)
    for k, v in res.items():
        if "val_accuracy" in k or "fresh_assignment" in k:
            print(k, [round(z, 2) for z in v["mean_ci"]], f"neg {v['n_negative']}/{len(v['seeds'])}")


if __name__ == "__main__":
    main()
