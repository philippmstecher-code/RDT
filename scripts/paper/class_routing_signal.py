#!/usr/bin/env python3
"""Class-level routing of the learning signal: where the shared signal comes from, whether it follows each class's
state rather than the epoch, and whether removing the coarse structure prolongs it.

For every arm, seed and checkpoint (epochs 1-15, 20, 25, 30), per stored test image (100 per class) at the pooled
layer: the fine-class cross-entropy gradient g = W'(softmax(Wx + b) - onehot(y)) with the network's classifier at that
checkpoint (for the suppression arms the stored activations were recorded with the projection on, i.e. as the head
saw them; g is the signal the head sends back), and the image's error type:
  correct      predicted class = true class
  within       wrong class in the right superclass
  misrouted    predicted class in another superclass
Per class c: G_c = class-mean gradient = G_mis + G_within + G_correct (sums over the image subsets / 100).
  shared(v)    cos^2(v, u_c), u_c = mean class-mean gradient of c's four superclass siblings (c excluded)
  null(v)      the same with c's group-mates in 20 random groupings (mean)
  excess(v)    shared(v) - null(v)
1) Error-type decomposition: excess of G_mis vs G_within (sham); energy shares of the components; per-class misrouting
   rate. Prediction: the shared signal comes from misrouted images.
2) State vs clock: mixed model excess(G_c) ~ accuracy + C(epoch) + (1 | class), per seed (statsmodels MixedLM);
   also with the misrouting rate. Prediction: accuracy (or misrouting) predicts sharing beyond the epoch.
3) Causal: arms sham, superclass (code removed, epochs 1-10), random directions, random grouping; per seed, the mean
   over classes of the misrouting rate and of excess(G_c) during epochs 1-10 and 11-15, arm minus sham, paired over
   seeds. Prediction: removing the coarse code keeps classes misrouted longer and their learning signal shared longer.
Writes analysis_outputs/paper/RANDGROUP_control/class_routing_mech_long.csv and class_routing_mech_summary.json.
Usage: python scripts/paper/class_routing_signal.py
"""
import json
import os
import sys
import warnings
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
E = Path(RDT_DATA + "/extended/experiments")
FEST = Path(RDT_DATA + "/core/experiments")
SEEDS = [7, 42, 137, 256, 314]
RG = [grouping(np.random.RandomState(p)) for p in range(20)]


def lane(arm, s):
    """Lane with both stored activations and checkpoints."""
    if arm == "sham":
        cands = [E / "critical_sham/lanes" / f"sham_e00_seed{s}"]
    elif arm == "randgroup":
        cands = [E / "randgroup/lanes" / f"lesion_randgroup_e00_seed{s}_rp65_Llayer4-avgpool"]
    else:
        nm = {"superclass": "lesion_scaffold_e00_seed{}_rp65_Llayer4-avgpool",
              "random": "lesion_random_variance_e00_seed{}_rp65_Llayer4-avgpool"}[arm].format(s)
        cands = [E / "prevention/lanes" / nm, E / "prevention_s42_copy/lanes" / nm, FEST / "scaffold_lesion/lanes" / nm]
    for c in cands:
        m1 = c / "dev_snapshots/milestone_1"
        if (m1 / "individual_multilayer_activations.pt").exists() and (m1 / "checkpoint.pt").exists():
            return c
    return None


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def excess_of(V, Gc):
    """Per class: cos^2(V_c, sibling-mean of Gc) minus the random-grouping mean of the same."""
    def share(groups):
        out = np.zeros(100)
        for c in range(100):
            mates = np.where((groups == groups[c]) & (np.arange(100) != c))[0]
            u = unit(Gc[mates].mean(0))
            v = V[c]
            nv = v @ v
            out[c] = (v @ u) ** 2 / nv if nv > 0 else np.nan
        return out
    sh = share(SUPER_OF)
    nl = np.nanmean(np.stack([share(g) for g in RG]), 0)
    return sh - nl


def analyse(arm, s):
    ld = lane(arm, s)
    if ld is None:
        return []
    ms = milestones(ld)
    if 30 not in ms:
        return []
    rows = []
    for e in sorted(ms):
        X, y = load(ms[e], "avgpool")
        X = X.astype(np.float64)
        W, b = classifier(ms[e])
        Z = X @ W.T + b
        pred = Z.argmax(1)
        P = np.exp(Z - Z.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        P[np.arange(len(y)), y] -= 1.0
        G = P @ W
        kind = np.where(pred == y, "correct", np.where(SUPER_OF[pred] == SUPER_OF[y], "within", "misrouted"))
        Gc = np.stack([G[y == c].mean(0) for c in range(100)])
        comp = {k: np.stack([G[(y == c) & (kind == k)].sum(0) / (y == c).sum() for c in range(100)])
                for k in ("misrouted", "within", "correct")}
        ex_all = excess_of(Gc, Gc)
        ex_mis = excess_of(comp["misrouted"], Gc)
        ex_win = excess_of(comp["within"], Gc)
        for c in range(100):
            m = y == c
            en = {k: float((comp[k][c] ** 2).sum()) for k in comp}
            tot = sum(en.values())
            rows.append({"arm": arm, "seed": s, "epoch": e, "cls": c, "acc": float((pred[m] == c).mean()),
                         "misrouted_rate": float((kind[m] == "misrouted").mean()),
                         "within_rate": float((kind[m] == "within").mean()),
                         "excess_all": float(ex_all[c]), "excess_misrouted": float(ex_mis[c]),
                         "excess_within": float(ex_win[c]),
                         "energy_misrouted": en["misrouted"] / tot if tot else np.nan,
                         "energy_within": en["within"] / tot if tot else np.nan})
    print(arm, s, "done", flush=True)
    return rows


def tci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean()), float(x.mean() - h), float(x.mean() + h)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm in ("sham", "superclass", "random", "randgroup"):
        for s in SEEDS:
            rows += analyse(arm, s)
    L = pd.DataFrame(rows)
    L.to_csv(OUT / "class_routing_mech_long.csv", index=False)
    res = {}

    # 1) error-type decomposition (sham)
    S = L[L.arm == "sham"]
    res["1_decomposition_sham_by_epoch"] = S.groupby("epoch")[
        ["misrouted_rate", "within_rate", "energy_misrouted", "energy_within", "excess_misrouted", "excess_within"]
    ].mean().round(4).reset_index().to_dict("records")
    d = (S.groupby(["seed"])[["excess_misrouted", "excess_within"]].mean())
    res["1_excess_misrouted_minus_within_per_seed"] = (d.excess_misrouted - d.excess_within).round(4).to_dict()

    # 2) state vs clock (sham), mixed model per seed
    import statsmodels.formula.api as smf
    res["2_state_vs_clock"] = {}
    for s, g in S.groupby("seed"):
        g = g.dropna(subset=["excess_all"])
        out = {}
        for pred in ("acc", "misrouted_rate"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = smf.mixedlm(f"excess_all ~ {pred} + C(epoch)", g, groups=g["cls"]).fit()
            out[pred] = {"coef": float(m.params[pred]), "z": float(m.tvalues[pred]), "p": float(m.pvalues[pred])}
        res["2_state_vs_clock"][int(s)] = out

    # 3) causal: arm minus sham, paired over seeds
    res["3_arm_minus_sham"] = {}
    for win, eps in (("epochs_1_10", range(1, 11)), ("epochs_11_15", range(11, 16))):
        W_ = L[L.epoch.isin(list(eps))].groupby(["arm", "seed"])[["misrouted_rate", "excess_all", "acc"]].mean()
        for arm in ("superclass", "random", "randgroup"):
            if arm not in W_.index.get_level_values("arm"):
                continue
            seeds = sorted(set(W_.loc[arm].index) & set(W_.loc["sham"].index))
            for k in ("misrouted_rate", "excess_all", "acc"):
                dlt = (W_.loc[arm].loc[seeds, k] - W_.loc["sham"].loc[seeds, k]).values
                res["3_arm_minus_sham"][f"{win}|{arm}|{k}"] = {"mean_ci": tci(dlt), "n_positive": int((dlt > 0).sum()),
                                                              "n": len(dlt)}
    json.dump(res, open(OUT / "class_routing_mech_summary.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
