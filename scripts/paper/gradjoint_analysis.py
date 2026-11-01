#!/usr/bin/env python3
"""Acquisition blocking against fully dose-matched controls (gradjoint, gradrandfresh).

Arms analysed here (ResNet-18, CIFAR-100, seeds 7, 42, 137, 256, 314; branch epoch 0; block epochs 1-10 at layer1-4 and the pooled
layer; forward pass unchanged; trained straight to epoch 30):
  gradjoint      at every re-estimation, as many directions as the superclass arm blocks (shadow rank), outside the
                 current superclass code, chosen as the window of consecutive learning-signal eigenvectors whose share
                 lies closest to the superclass arm's share (rank- and share-matched); rebuilt at every re-estimation
  gradrandfresh  as many random directions outside the current superclass code, redrawn at every re-estimation
                 (rank-matched, not accumulated)
  gradsc_l3, gradjoint_l3   the superclass block and the jointly matched block at layer3, layer4 and the pooled layer
                 only, where rank and share can be matched over the whole window; 15 epochs (gradl3)
The other arms (sham, gradsc, gradrand, gradenergy, gradrg) are read from GRADBLOCK/final_long.csv (gradblock_complete.py).
Measures per arm, seed, epoch: the network's fine-class validation accuracy and, at stored checkpoints, the fresh
100-way ridge read-out of the pooled activations split into superclass assignment and within-superclass accuracy
(identical to gradblock_complete.py). Dose: in-run target (superclass arm's share in the same network) and achieved share
of the learning signal per stage, and blocked rank against the superclass arm's rank, from lesion_log.jsonl.
Contrasts: per-seed means over windows, paired over seeds (t-based 95% CI).
Writes analysis_outputs/paper/GRADJOINT/{long.csv, summary.json, dose.csv, dose_summary.json}.
Usage: python scripts/paper/gradjoint_analysis.py
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
OUT = REPO / "analysis_outputs/paper/GRADJOINT"
E = Path(RDT_DATA + "/extended/experiments")
SEEDS = [7, 42, 137, 256, 314]
L5 = "Llayer1-layer2-layer3-layer4-avgpool"
L3 = "Llayer3-layer4-avgpool"
# arm: (experiment, condition, layer tag, last epoch)
NEW = {"gradjoint": ("gradjoint", "gradjoint", L5, 30), "gradrandfresh": ("gradrandfresh", "gradrandfresh", L5, 30),
       "gradsc_l3": ("gradl3", "gradsc", L3, 15), "gradjoint_l3": ("gradl3", "gradjoint", L3, 15)}
WINS = {"epochs_1_3": range(1, 4), "epochs_1_10": range(1, 11), "epochs_11_15": range(11, 16),
        "epochs_16_20": range(16, 21), "epochs_26_30": range(26, 31), "epoch_30": [30]}
PAIRS = [("gradsc", "gradjoint"), ("gradsc", "gradrandfresh"), ("gradjoint", "sham"), ("gradrandfresh", "sham"),
         ("gradjoint", "gradenergy"), ("gradjoint", "gradrand"), ("gradrandfresh", "gradrand"),
         ("gradsc", "gradenergy"), ("gradsc", "gradrand"), ("gradsc", "sham"), ("gradrg", "gradjoint"),
         ("gradsc_l3", "gradjoint_l3"), ("gradsc_l3", "sham"), ("gradjoint_l3", "sham")]
KEYS = ("val_accuracy", "fresh_fine", "fresh_assignment", "fresh_within")


def lane(arm, s):
    exp, cond, tag, _ = NEW[arm]
    return E / exp / "lanes" / f"lesion_{cond}_e00_seed{s}_rp65_{tag}"


def done(ld, arm):
    f = ld / "dev_snapshots/metrics/devtrain_metrics.json"
    return f.exists() and max(int(e["epoch"]) for e in json.load(open(f))) >= NEW[arm][3]


def tci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return [float(x.mean()), float(x.mean() - h), float(x.mean() + h)]


def new_rows():
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    cache = OUT / "long_new.csv"
    old = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["arm", "seed", "epoch"])
    have = set(zip(old.arm, old.seed, old.epoch))
    rows = old.to_dict("records")
    for arm in NEW:
        for s in SEEDS:
            ld = lane(arm, s)
            if not done(ld, arm):
                print("not finished", arm, s, flush=True)
                continue
            acc = {int(e["epoch"]): float(e["val_accuracy"])
                   for e in json.load(open(ld / "dev_snapshots/metrics/devtrain_metrics.json"))}
            ms = milestones(ld)
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
            print(arm, s, "rows done", flush=True)
            pd.DataFrame(rows).to_csv(cache, index=False)
    return pd.DataFrame(rows)


def dose():
    rows = []
    for arm in NEW:
        for s in SEEDS:
            f = lane(arm, s) / "dev_snapshots/metrics/lesion_log.jsonl"
            if not f.exists():
                continue
            for line in open(f):
                d = json.loads(line)
                for layer, r in (d.get("report") or {}).items() if isinstance(d.get("report"), dict) else []:
                    if "target_grad_share" in r:
                        rows.append({"arm": arm, "seed": s, "epoch": d.get("epoch"), "stage": layer,
                                     "target_share": r["target_grad_share"], "achieved_share": r["achieved_grad_share"],
                                     "shadow_rank": r["shadow_rank"], "rank": r["total_rank"]})
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "dose.csv", index=False)
    res = {}
    for w, eps in {"epochs_1_3": range(1, 4), "epochs_1_10": range(1, 11)}.items():
        g = D[D.epoch.isin(list(eps))].groupby(["arm", "stage", "seed"])[["target_share", "achieved_share", "shadow_rank", "rank"]].mean()
        g = g.groupby(["arm", "stage"]).mean().round(3)
        res[w] = {f"{a}|{st}": v for (a, st), v in g.to_dict("index").items()}
    json.dump(res, open(OUT / "dose_summary.json", "w"), indent=1)
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(REPO / "analysis_outputs/paper/GRADBLOCK/final_long.csv")
    D = pd.concat([old, new_rows()], ignore_index=True).sort_values(["arm", "seed", "epoch"])
    D.to_csv(OUT / "long.csv", index=False)
    res = {}
    for w, eps in WINS.items():
        for key in KEYS:
            M = D[D.epoch.isin(list(eps))].groupby(["arm", "seed"])[key].mean()
            for a, b in PAIRS:
                if a not in M.index.get_level_values(0) or b not in M.index.get_level_values(0):
                    continue
                sd = sorted(set(M.loc[a].dropna().index) & set(M.loc[b].dropna().index))
                if len(sd) < 2:
                    continue
                d = (M.loc[a].loc[sd] - M.loc[b].loc[sd]).values
                res[f"{w}|{a} - {b}|{key}"] = {"mean_ci": tci(d), "n_negative": int((d < 0).sum()), "seeds": sd,
                                               "per_seed": [round(float(v), 3) for v in d]}
    json.dump(res, open(OUT / "summary.json", "w"), indent=1)
    ds = dose()
    for k, v in res.items():
        if k.split("|")[2] in ("val_accuracy", "fresh_assignment"):
            print(k, [round(z, 2) for z in v["mean_ci"]], f"neg {v['n_negative']}/{len(v['seeds'])}")
    for w, d in ds.items():
        print("dose", w, {k: (v["target_share"], v["achieved_share"], v["shadow_rank"], v["rank"]) for k, v in d.items()})


if __name__ == "__main__":
    main()
