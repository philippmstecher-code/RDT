#!/usr/bin/env python3
"""Graded random removal: robustness of superclass (coarse) vs within-superclass (fine) information.
Clean ResNet-18 seeds, epochs 5 and 30. Random orthonormal directions are added until a fraction f of the centred
variance of the pooled stage activations is removed (mean-preserving projection at every position):
  setting (a) at avgpool (the classifier input; via the layer4 hook, since avgpool is the spatial mean of layer4)
  setting (b) at layer3, then forward to avgpool.
Readout-free S_sc and S_wsc at avgpool (decodability), 3 draws per f.
Writes analysis_outputs/paper/REDUNDANCY/results.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tg_dominant as E  # noqa: E402
from decodability import measures_from_acts  # noqa: E402
T = E.T
OUT = T.REPO / "analysis_outputs" / "paper" / "REDUNDANCY"
OUT.mkdir(parents=True, exist_ok=True)
FRACS = (0.1, 0.25, 0.5, 0.75, 0.9)
NDRAW = 3


def dirs_for_fraction(x, f, g):
    """Random orthonormal directions in fixed random order, as many as needed to remove >= f of centred variance."""
    xc = x - x.mean(0)
    tot = (xc ** 2).sum()
    Q, _ = torch.linalg.qr(torch.randn(x.shape[1], x.shape[1], generator=g, dtype=torch.float64))
    v = ((xc.double() @ Q) ** 2).sum(0) / tot
    k = int((torch.cumsum(v, 0) < f).sum().item()) + 1
    return Q[:, :k].float(), float(torch.cumsum(v, 0)[k - 1])


def acts_by_class(F, y):
    return {c: F[y == c] for c in range(100)}


def main():
    cf = T.cifar_test()
    y = cf["y"]
    res = {}
    for s in T.SEEDS:
        d = T.RUNS_EXTENDED / f"static_noise_s{s}" / "lanes" / "standard_resnet18"
        lab_of = {c["epoch"]: c["label"] for c in T.real_ckpts(T.checkpoint_info(d, T._snap_labels(d)))}
        for ep in (30, 5):
            net = T.load_net({"dir": str(d), "arch": "ResNet-18", "group": "extended"}, lab_of[ep])
            F, _ = E.forward(net, cf["X"])
            base = measures_from_acts(acts_by_class(F["layer4"], y))
            key = f"{s}|{ep}"
            res[key] = {"base": base, "a": {}, "b": {}}
            for setting, stage in (("a", "layer4"), ("b", "layer3")):
                x = F[stage].float()
                mu = x.mean(0)
                for f in FRACS:
                    rows = []
                    for dr in range(NDRAW):
                        g = torch.Generator().manual_seed(100000 * s + 1000 * ep + 10 * dr + (0 if setting == "a" else 5))
                        Q, got = dirs_for_fraction(x, f, g)
                        Fa, _ = E.forward(net, cf["X"], {stage: E.MeanPreservingProjection(Q, mu)})
                        m = measures_from_acts(acts_by_class(Fa["layer4"], y))
                        rows.append({"r_sc": m["S_sc"] / base["S_sc"], "r_wsc": m["S_wsc"] / base["S_wsc"],
                                     "k": int(Q.shape[1]), "var_removed": got})
                    res[key][setting][f] = {k: float(np.mean([r[k] for r in rows])) for k in ("r_sc", "r_wsc", "var_removed")} | {
                        "k": int(np.mean([r["k"] for r in rows]))}
                    print(s, ep, setting, f, {k: round(v, 3) for k, v in res[key][setting][f].items()}, flush=True)
            json.dump(res, open(OUT / "results.json", "w"), indent=1, default=float)
    summ = {}
    for ep in (30, 5):
        for setting in ("a", "b"):
            R = [np.mean([res[f"{s}|{ep}"][setting][f]["r_sc"] - res[f"{s}|{ep}"][setting][f]["r_wsc"] for f in FRACS]) for s in T.SEEDS]
            R = np.array(R)
            h = stats.t.ppf(0.975, len(R) - 1) * R.std(ddof=1) / np.sqrt(len(R))
            summ[f"ep{ep}_{setting}"] = {"R_mean_ci": [float(R.mean()), float(R.mean() - h), float(R.mean() + h)],
                                         "n_positive": int((R > 0).sum()), "per_seed": R.tolist()}
    summ["PRIMARY_supported"] = bool(summ["ep30_b"]["R_mean_ci"][1] > 0 and summ["ep30_b"]["n_positive"] >= 4)
    res["_summary"] = summ
    json.dump(res, open(OUT / "results.json", "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
