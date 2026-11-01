#!/usr/bin/env python3
"""Within-superclass discrimination given the true superclass ("oracle" read-out) under blocking and suppression.

For every test image (10,000 CIFAR-100 test images, 100 per class; stored pooled-layer activations of each network),
the 5-way accuracy among the five classes of the image's TRUE superclass, regardless of the predicted superclass:
  ridge   100-way ridge read-out (standardised features, alpha 10, StratifiedKFold(5, shuffle, random_state 0), as the
          fresh read-out of gradblock_complete.py), decision scores restricted to the five true-superclass classes
  logits  the network's own classifier (W x + b of the stored checkpoint) restricted to the five true-superclass classes
Arms (paired over seeds 7, 42, 137, 256, 314; mean, t-based 95% CI, seeds lower):
  blocking       gradsc - gradenergy, epochs 1-3, 11-15 and 30 (epochs 16-30 from the continuation lanes);
                 gradsc - gradrand (random directions) and gradrg (random-grouping directions) - gradenergy / -
                 gradrand, epochs 1-3
  jointly matched gradsc - gradjoint (rank and learning-signal share matched), epochs 1-3, 8-10, 11-15 and 30
  layer3+ pair   gradsc_l3 - gradjoint_l3 (confined to layer3 to the pooled layer, both matched), epochs 1-3, 8-10, 13-15
  stage-matched  gradsc_l2 - gradenergy_l2, epochs 1-3
  suppression    last-block superclass code (prevention lesion_scaffold) - sham (critical_sham sham_e00) and
                 - random directions (lesion_random_variance), epochs 1-3. The stored activations of the suppression
                 networks are those computed with the projection in place (the network's own forward pass).
Writes analysis_outputs/paper/ORACLE_within/oracle_long.csv and oracle_summary.json.
Usage: python scripts/paper/oracle_within.py
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
from gradient_routing import classifier  # noqa: E402
from curriculum_probe import SUPER_OF, load  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis_outputs/paper/ORACLE_within"
E = Path(RDT_DATA + "/extended/experiments")
SEEDS = [7, 42, 137, 256, 314]
L5, L4, L3 = "Llayer1-layer2-layer3-layer4-avgpool", "Llayer2-layer3-layer4-avgpool", "Llayer3-layer4-avgpool"
MEMBERS = np.array([np.where(SUPER_OF == SUPER_OF[c])[0] for c in range(100)])   # (100, 5): classes of c's superclass


def lanes(arm, s):
    """List of lane directories (in training order) of an arm and seed."""
    if arm == "sham":
        return [E / f"critical_sham/lanes/sham_e00_seed{s}"]
    if arm in ("gradsc", "gradenergy", "gradrand"):
        src = "gradenergy" if arm == "gradenergy" else "gradblock"
        return [E / f"{src}/lanes/lesion_{arm}_e00_seed{s}_rp65_{L5}", E / f"gradcont_{arm}_r1/lanes/sham_e15_seed{s}"]
    if arm == "gradrg":
        return [E / f"gradrg/lanes/lesion_gradrg_e00_seed{s}_rp65_{L5}"]
    if arm == "gradjoint":
        return [E / f"gradjoint/lanes/lesion_gradjoint_e00_seed{s}_rp65_{L5}"]
    if arm in ("gradsc_l3", "gradjoint_l3"):
        return [E / f"gradl3/lanes/lesion_{arm[:-3]}_e00_seed{s}_rp65_{L3}"]
    if arm in ("gradsc_l2", "gradenergy_l2"):
        return [E / f"gradl2/lanes/lesion_{arm[:-3]}_e00_seed{s}_rp65_{L4}"]
    kind = {"supp_sc": "scaffold", "supp_rand": "random_variance"}[arm]
    if s == 42:   # the RUNS_EXTENDED copy holds SAE inputs only; the original lane (read only) has the checkpoints
        return [REPO / f"data/experiments/scaffold_lesion/lanes/lesion_{kind}_e00_seed42_rp65_Llayer4-avgpool"]
    return [E / f"prevention/lanes/lesion_{kind}_e00_seed{s}_rp65_Llayer4-avgpool"]


def ms_of(arm, s):
    out = {}
    for d in lanes(arm, s):
        out.update(milestones(d))
    return out


def oracle(ms_dir, cv):
    X, y = load(ms_dir, "avgpool")
    X = X.astype(np.float64)
    W, b = classifier(ms_dir)
    Z = X @ W.T + b
    S = cross_val_predict(make_pipeline(StandardScaler(), RidgeClassifier(alpha=10.0)), X, y, cv=cv,
                          method="decision_function")
    mem = MEMBERS[y]
    def acc(M):
        sub = np.take_along_axis(M, mem, 1)
        return 100 * float((mem[np.arange(len(y)), sub.argmax(1)] == y).mean())
    pred = Z.argmax(1)
    return {"oracle_ridge": acc(S), "oracle_logits": acc(Z),
            "net_assignment": 100 * float((SUPER_OF[pred] == SUPER_OF[y]).mean()),
            "net_fine": 100 * float((pred == y).mean())}


PLAN = {"gradsc": [1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 30], "gradenergy": [1, 2, 3, 11, 12, 13, 14, 15, 30],
        "gradjoint": [1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 30],
        "gradsc_l3": [1, 2, 3, 8, 9, 10, 13, 14, 15], "gradjoint_l3": [1, 2, 3, 8, 9, 10, 13, 14, 15],
        "gradsc_l2": [1, 2, 3], "gradenergy_l2": [1, 2, 3], "supp_sc": [1, 2, 3], "supp_rand": [1, 2, 3],
        "sham": [1, 2, 3], "gradrand": [1, 2, 3], "gradrg": [1, 2, 3]}
CONTRASTS = [("blocking", "gradsc", "gradenergy", {"epochs 1-3": [1, 2, 3], "epochs 11-15": [11, 12, 13, 14, 15],
                                                    "epoch 30": [30]}),
             ("jointly matched", "gradsc", "gradjoint", {"epochs 1-3": [1, 2, 3], "epochs 8-10": [8, 9, 10],
                                                         "epochs 11-15": [11, 12, 13, 14, 15], "epoch 30": [30]}),
             ("layer3+ pair", "gradsc_l3", "gradjoint_l3", {"epochs 1-3": [1, 2, 3], "epochs 8-10": [8, 9, 10],
                                                           "epochs 13-15": [13, 14, 15]}),
             ("stage-matched", "gradsc_l2", "gradenergy_l2", {"epochs 1-3": [1, 2, 3]}),
             ("blocking vs random directions", "gradsc", "gradrand", {"epochs 1-3": [1, 2, 3]}),
             ("random-grouping blocking", "gradrg", "gradenergy", {"epochs 1-3": [1, 2, 3]}),
             ("random-grouping blocking vs random directions", "gradrg", "gradrand", {"epochs 1-3": [1, 2, 3]}),
             ("suppression vs sham", "supp_sc", "sham", {"epochs 1-3": [1, 2, 3]}),
             ("suppression vs random", "supp_sc", "supp_rand", {"epochs 1-3": [1, 2, 3]})]


def tci(d):
    d = np.asarray(d, float)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return [float(d.mean()), float(d.mean() - h), float(d.mean() + h)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "oracle_long.csv"
    done = pd.read_csv(f) if f.exists() else pd.DataFrame()
    have = set(zip(done.arm, done.seed, done.epoch)) if len(done) else set()
    rows = done.to_dict("records") if len(done) else []
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for arm, eps in PLAN.items():
        for s in SEEDS:
            if all((arm, s, e) in have for e in eps):
                continue
            ms = ms_of(arm, s)
            for e in eps:
                if (arm, s, e) in have:
                    continue
                if e not in ms:
                    print("missing", arm, s, e, flush=True)
                    continue
                rows.append({"arm": arm, "seed": s, "epoch": e, **oracle(ms[e], cv)})
                print(rows[-1], flush=True)
                pd.DataFrame(rows).to_csv(f, index=False)
    D = pd.DataFrame(rows)
    res = {}
    for name, a, b, wins in CONTRASTS:
        for w, eps in wins.items():
            M = D[D.epoch.isin(eps)].groupby(["arm", "seed"]).mean(numeric_only=True)
            for k in ("oracle_ridge", "oracle_logits", "net_assignment", "net_fine"):
                d = (M.loc[a][k] - M.loc[b][k]).loc[SEEDS].values
                res[f"{name}|{w}|{k}"] = {"mean_ci": tci(d), "n_lower": int((d < 0).sum()),
                                          "per_seed": np.round(d, 3).tolist(),
                                          "level_a": float(M.loc[a][k].loc[SEEDS].mean()),
                                          "level_b": float(M.loc[b][k].loc[SEEDS].mean())}
    json.dump(res, open(OUT / "oracle_summary.json", "w"), indent=1)
    for k, v in res.items():
        print(f"{k:60s} {v['mean_ci'][0]:+6.2f} [{v['mean_ci'][1]:+6.2f}, {v['mean_ci'][2]:+6.2f}] lower "
              f"{v['n_lower']}/5  ({v['level_a']:.1f} vs {v['level_b']:.1f})")


if __name__ == "__main__":
    main()
