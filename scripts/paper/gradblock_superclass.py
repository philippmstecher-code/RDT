#!/usr/bin/env python3
"""Per-superclass view of the acquisition-blocking control: for which superclasses was the most learning signal blocked,
and where did their classes end up?

Per arm (gradsc, gradrand, sham), seed and epoch, from the stored pooled-layer activations (100 test images per class;
the forward pass of the blocking arms is unchanged) and the network's classifier:
  blocked share   per superclass: share of the summed squared fine-class gradient of its images that lies in the
                  rank-19 superclass code of that network (the directions the gradsc arm blocks; estimated from the
                  activations as in the re-estimation) - computed on the gradsc network itself; for comparison the share
                  in a random subspace of equal rank (what gradrand blocks)
  outcome         per superclass: the network's fine accuracy, superclass assignment (prediction in the right
                  superclass) and accuracy within the right superclass, and the same for a fresh 100-way linear
                  read-out (ridge, 5-fold CV) of the activations; gradsc - sham and gradrand - sham
Across the 20 superclasses: Spearman rho between the blocked share and the deficit (gradsc - sham), per seed and epoch.
Writes analysis_outputs/paper/GRADBLOCK/superclass_<exp>.csv and superclass_<exp>_summary.json.
Usage: python scripts/paper/gradblock_superclass.py [--exp gradblock] [--seeds 7 ...]
       [--epochs 1 2 3 ...]
"""
import argparse
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
from coarse_overlap import between_scatter, milestones, top  # noqa: E402
from gradient_routing import classifier  # noqa: E402
from curriculum_probe import SUPER_OF, load  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis_outputs/paper/GRADBLOCK"
E = Path(RDT_DATA + "/extended/experiments")
L5 = "Llayer1-layer2-layer3-layer4-avgpool"


def per_image(ms_dir):
    X, y = load(ms_dir, "avgpool")
    X = X.astype(np.float64)
    W, b = classifier(ms_dir)
    Z = X @ W.T + b
    pred = Z.argmax(1)
    P = np.exp(Z - Z.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    P[np.arange(len(y)), y] -= 1.0
    return X, y, pred, P @ W


def outcome(y, pred):
    ok = SUPER_OF[pred] == SUPER_OF[y]
    rows = {}
    for k in range(20):
        m = SUPER_OF[y] == k
        rows[k] = {"fine": 100 * float((pred[m] == y[m]).mean()), "assignment": 100 * float(ok[m].mean()),
                   "within": 100 * float((pred[m] == y[m])[ok[m]].mean()) if ok[m].any() else np.nan}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="gradblock")
    ap.add_argument("--seeds", nargs="+", type=int, default=[7])
    ap.add_argument("--epochs", nargs="+", type=int, default=[1, 2, 3])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lanes = {"sham": lambda s: E / "critical_sham/lanes" / f"sham_e00_seed{s}",
             "gradsc": lambda s: E / a.exp / "lanes" / f"lesion_gradsc_e00_seed{s}_rp65_{L5}",
             "gradrand": lambda s: E / a.exp / "lanes" / f"lesion_gradrand_e00_seed{s}_rp65_{L5}"}
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    rng = np.random.default_rng(0)
    rows = []
    for s in a.seeds:
        ms = {arm: milestones(f(s)) for arm, f in lanes.items()}
        for e in a.epochs:
            res = {}
            for arm in lanes:
                X, y, pred, G = per_image(ms[arm][e])
                fresh = cross_val_predict(make_pipeline(StandardScaler(), RidgeClassifier(alpha=10.0)), X, y, cv=cv)
                res[arm] = (outcome(y, pred), outcome(y, fresh))
                if arm == "gradsc":
                    Q = top(between_scatter(X, y, SUPER_OF))
                    Qr, _ = np.linalg.qr(rng.standard_normal((X.shape[1], Q.shape[1])))
                    blk = {k: float(((G[SUPER_OF[y] == k] @ Q) ** 2).sum() / (G[SUPER_OF[y] == k] ** 2).sum())
                           for k in range(20)}
                    blk_r = {k: float(((G[SUPER_OF[y] == k] @ Qr) ** 2).sum() / (G[SUPER_OF[y] == k] ** 2).sum())
                             for k in range(20)}
            for k in range(20):
                r = {"seed": s, "epoch": e, "superclass": k, "blocked_share": blk[k], "random_share": blk_r[k]}
                for i, src in enumerate(("net", "fresh")):
                    for m in ("fine", "assignment", "within"):
                        r[f"{src}_{m}_sham"] = res["sham"][i][k][m]
                        r[f"{src}_{m}_gradsc_minus_sham"] = res["gradsc"][i][k][m] - res["sham"][i][k][m]
                        r[f"{src}_{m}_gradrand_minus_sham"] = res["gradrand"][i][k][m] - res["sham"][i][k][m]
                rows.append(r)
            print("seed", s, "epoch", e, "done", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / f"superclass_{a.exp}.csv", index=False)
    summ = {}
    for (s, e), g in D.groupby(["seed", "epoch"]):
        out = {}
        for col in ("fresh_fine", "fresh_assignment", "fresh_within", "net_fine", "net_assignment"):
            r1 = stats.spearmanr(g.blocked_share, g[f"{col}_gradsc_minus_sham"])
            r0 = stats.spearmanr(g.blocked_share, g[f"{col}_gradrand_minus_sham"])
            out[col] = {"rho_blocked_vs_gradsc_deficit": [round(float(r1[0]), 3), round(float(r1[1]), 4)],
                        "rho_blocked_vs_gradrand_deficit": [round(float(r0[0]), 3), round(float(r0[1]), 4)]}
        summ[f"seed{s}_epoch{e}"] = out
    json.dump(summ, open(OUT / f"superclass_{a.exp}_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1)[:3000])


if __name__ == "__main__":
    main()
