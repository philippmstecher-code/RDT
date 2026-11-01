#!/usr/bin/env python3
"""Class-level routing of the learning signal: is each class's learning signal first shared with its superclass
siblings, and does the end of that shared phase predict when the class itself is learned?

Sham networks (critical_sham sham_e00_seed{7,42,137,256,314}), checkpoints at epochs 1-15, 20, 25, 30. Per image,
the fine-class cross-entropy gradient at the pooled features, g = W'(softmax(Wx + b) - onehot(y)) (exact; the
network's own classifier). Per class c, the class-mean gradient G_c over its 100 stored test images.
  shared_c      cos^2(G_c, mean of G over c's four superclass siblings), sibling mean excluding c itself;
  null_c        the same with c's four group-mates in each of 20 random groupings (RandomState(p), p = 0..19), mean;
  excess_c      shared_c - null_c: how much more c's learning signal points with its siblings than with random classes.
  shared phase  centre of mass over epochs of the positive part of excess_c (com_shared), and its last epoch with
                excess_c > 0 (end_shared).
  learning time t50_c: first epoch (linear interpolation over checkpoints) at which the class's accuracy on its stored
                test images reaches half of its accuracy at epoch 30 (accuracy from the same logits).
Prediction (coarse structure routes learning): excess_c > 0 early for most classes and falling; across classes,
classes whose shared phase lasts longer are learned later (Spearman rho(com_shared, t50) > 0 per seed); no such
relation for the random-grouping version of the same measure.
Writes analysis_outputs/paper/RANDGROUP_control/class_routing_long.csv, class_routing_per_class.csv,
class_routing_summary.json, and Source Data figures/final/source_data/class_routing.csv.
Usage: python scripts/paper/class_routing.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coarse_overlap import grouping, milestones  # noqa: E402
from gradient_routing import classifier  # noqa: E402
from curriculum_probe import SUPER_OF, load  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis_outputs/paper/RANDGROUP_control"
SD = REPO / "analysis_outputs/paper/figures/final/source_data"
E = Path(RDT_DATA + "/extended/experiments")
SEEDS = [7, 42, 137, 256, 314]
NGROUP = 20


def cos2(a, b):
    return float((a @ b) ** 2 / ((a @ a) * (b @ b) + 1e-30))


def loo_share(Gc, groups):
    """Per class: cos^2 between its class-mean gradient and the mean over its group-mates (itself excluded)."""
    out = np.zeros(len(Gc))
    for c in range(len(Gc)):
        mates = np.where((groups == groups[c]) & (np.arange(len(Gc)) != c))[0]
        out[c] = cos2(Gc[c], Gc[mates].mean(0))
    return out


def t_half(epochs, acc):
    """First epoch (linear interpolation) at which acc reaches half of its final value."""
    target = 0.5 * acc[-1]
    for i in range(len(acc)):
        if acc[i] >= target:
            if i == 0:
                return float(epochs[0])
            e0, e1, a0, a1 = epochs[i - 1], epochs[i], acc[i - 1], acc[i]
            return float(e0 + (target - a0) / (a1 - a0) * (e1 - e0)) if a1 > a0 else float(e1)
    return float("nan")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rgroups = [grouping(np.random.RandomState(p)) for p in range(NGROUP)]
    rows = []
    for s in SEEDS:
        ms = milestones(E / "critical_sham/lanes" / f"sham_e00_seed{s}")
        for e in sorted(ms):
            X, y = load(ms[e], "avgpool")
            X = X.astype(np.float64)
            W, b = classifier(ms[e])
            Z = X @ W.T + b
            acc = np.array([(Z[y == c].argmax(1) == c).mean() for c in range(100)])
            P = np.exp(Z - Z.max(1, keepdims=True))
            P /= P.sum(1, keepdims=True)
            P[np.arange(len(y)), y] -= 1.0
            G = P @ W
            Gc = np.stack([G[y == c].mean(0) for c in range(100)])
            sh = loo_share(Gc, SUPER_OF)
            nulls = np.stack([loo_share(Gc, g) for g in rgroups])
            for c in range(100):
                rows.append({"seed": s, "epoch": e, "cls": c, "superclass": int(SUPER_OF[c]), "acc": float(acc[c]),
                             "shared": float(sh[c]), "null_mean": float(nulls[:, c].mean()),
                             "null_p95": float(np.percentile(nulls[:, c], 95)), "excess": float(sh[c] - nulls[:, c].mean()),
                             # one random grouping as a matched 'control class measure'
                             "excess_rg0": float(nulls[0, c] - np.delete(nulls, 0, 0)[:, c].mean())})
        print("seed", s, "done", flush=True)
    L = pd.DataFrame(rows)
    L.to_csv(OUT / "class_routing_long.csv", index=False)

    per = []
    for (s, c), g in L.groupby(["seed", "cls"]):
        g = g.sort_values("epoch")
        ep, ex, exr = g.epoch.values.astype(float), g.excess.values, g.excess_rg0.values
        pos, posr = np.clip(ex, 0, None), np.clip(exr, 0, None)
        per.append({"seed": s, "cls": c, "superclass": int(g.superclass.iloc[0]),
                    "t50": t_half(ep, g.acc.values), "final_acc": float(g.acc.values[-1]),
                    "com_shared": float((ep * pos).sum() / pos.sum()) if pos.sum() > 0 else float("nan"),
                    "end_shared": float(ep[ex > 0].max()) if (ex > 0).any() else float("nan"),
                    "early_excess": float(ex[ep <= 3].mean()), "late_excess": float(ex[ep >= 20].mean()),
                    "com_shared_rg0": float((ep * posr).sum() / posr.sum()) if posr.sum() > 0 else float("nan")})
    Pc = pd.DataFrame(per)
    Pc.to_csv(OUT / "class_routing_per_class.csv", index=False)
    Pc.to_csv(SD / "class_routing.csv", index=False)

    res = {"n_seeds": len(SEEDS), "n_classes": 100}
    byep = L.groupby("epoch").agg(excess_mean=("excess", "mean"),
                                  frac_classes_excess_pos=("excess", lambda v: float((v > 0).mean())),
                                  frac_above_null_p95=("shared", "size"))
    byep["frac_above_null_p95"] = L.assign(a=L.shared > L.null_p95).groupby("epoch").a.mean()
    res["by_epoch"] = byep.round(4).reset_index().to_dict("records")
    res["early_vs_late_excess_per_seed"] = {int(s): [round(float(g.early_excess.mean()), 4),
                                                     round(float(g.late_excess.mean()), 4)]
                                            for s, g in Pc.groupby("seed")}
    rho = {}
    for s, g in Pc.groupby("seed"):
        g = g.dropna(subset=["t50", "com_shared"])
        r1 = stats.spearmanr(g.com_shared, g.t50)
        # partial: control the class's final accuracy (rank-residualised)
        rk = lambda v: stats.rankdata(v)
        def resid(a, b):
            A = np.column_stack([np.ones(len(b)), rk(b)])
            return rk(a) - A @ np.linalg.lstsq(A, rk(a), rcond=None)[0]
        pr = stats.pearsonr(resid(g.com_shared.values, g.final_acc.values), resid(g.t50.values, g.final_acc.values))
        gr = g.dropna(subset=["com_shared_rg0"])
        r0 = stats.spearmanr(gr.com_shared_rg0, gr.t50)
        rho[int(s)] = {"rho_com_shared_t50": [round(float(r1[0]), 3), float(r1[1])],
                       "partial_rho_given_final_acc": [round(float(pr[0]), 3), float(pr[1])],
                       "null_rho_random_grouping": [round(float(r0[0]), 3), float(r0[1])], "n": int(len(g))}
    res["shared_phase_predicts_learning_time"] = rho
    json.dump(res, open(OUT / "class_routing_summary.json", "w"), indent=1)
    print(json.dumps(res, indent=1)[:5000])


if __name__ == "__main__":
    main()
