#!/usr/bin/env python3
"""Routing of the learning signal, measured: which images supply the learning signal, and what that signal teaches.

Per arm, seed and stored checkpoint, from the stored pooled-layer activations of the 10,000 test images (100 per
class) and the network's own classifier (W, b): prediction p = softmax(Wx + b) and the per-image learning signal at
the pooled layer g = W'(p - onehot(y)) (the gradient of the fine-class cross-entropy with respect to the pooled
activations; the forward pass of the blocking arms is unchanged, so these are the networks' real predictions and
learning signal). Each image is in one of three states:
  misrouted   predicted class in another superclass
  routed      predicted class in the right superclass, not the right class
  correct     predicted class right
Measures per state:
  image_share     share of images in the state
  signal_share    share of the summed learning-signal energy sum ||g||^2 that images in the state supply
  code_share      share of the state's learning-signal energy inside the network's current superclass code (rank-19
                  top eigenvectors of the between-superclass scatter of the activations, as gradblock_analysis.py):
                  sum ||Q'g||^2 / sum ||g||^2 over the state's images
  random_share    the same for a random subspace of equal rank (one draw per network)
Arms: sham (critical_sham sham_e00, epochs 0-15, 20, 25, 30; epoch 0 = branch checkpoint), and the acquisition-
blocking arms (block in epochs 1-10; epochs 1-15): gradsc (superclass code), gradenergy (energy-matched), gradrand
(random directions), gradrg (random-grouping code).
Contrasts (paired over seeds, t-based 95% CI, seeds in the stated direction): epochs 1-3 and 1-10, gradsc - gradenergy,
gradrg - gradenergy, gradsc - sham, gradenergy - sham; and in sham, misrouted - routed code_share per epoch window.
Writes analysis_outputs/paper/SIGNAL_routing/signal_routing_long.csv and signal_routing_summary.json, and Source
Data figures/final/source_data/signal_routing_long.csv.
Usage: python scripts/paper/signal_routing.py
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
from coarse_overlap import between_scatter, milestones, top  # noqa: E402
from gradient_routing import classifier  # noqa: E402
from curriculum_probe import SUPER_OF, load  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis_outputs/paper/SIGNAL_routing"
SD = REPO / "analysis_outputs/paper/figures/final/source_data"
E = Path(RDT_DATA + "/extended/experiments")
SEEDS = [7, 42, 137, 256, 314]
L5 = "Llayer1-layer2-layer3-layer4-avgpool"
LANES = {"sham": lambda s: E / "critical_sham/lanes" / f"sham_e00_seed{s}",
         "gradsc": lambda s: E / "gradblock/lanes" / f"lesion_gradsc_e00_seed{s}_rp65_{L5}",
         "gradrand": lambda s: E / "gradblock/lanes" / f"lesion_gradrand_e00_seed{s}_rp65_{L5}",
         "gradenergy": lambda s: E / "gradenergy/lanes" / f"lesion_gradenergy_e00_seed{s}_rp65_{L5}",
         "gradrg": lambda s: E / "gradrg/lanes" / f"lesion_gradrg_e00_seed{s}_rp65_{L5}"}
STATES = ("misrouted", "routed", "correct")


def tci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean()), float(x.mean() - h), float(x.mean() + h)]


def measure(ms_dir, rng):
    X, y = load(ms_dir, "avgpool")
    X = X.astype(np.float64)
    W, b = classifier(ms_dir)
    Z = X @ W.T + b
    P = np.exp(Z - Z.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    pred = P.argmax(1)
    P[np.arange(len(y)), y] -= 1.0
    G = P @ W
    Q = top(between_scatter(X, y, SUPER_OF))
    Qr, _ = np.linalg.qr(rng.standard_normal((X.shape[1], Q.shape[1])))
    e_all, e_q, e_r = (G ** 2).sum(1), ((G @ Q) ** 2).sum(1), ((G @ Qr) ** 2).sum(1)
    right_sc = SUPER_OF[pred] == SUPER_OF[y]
    state = np.where(pred == y, "correct", np.where(right_sc, "routed", "misrouted"))
    out = {"code_share_all": float(e_q.sum() / e_all.sum()), "random_share_all": float(e_r.sum() / e_all.sum())}
    for st in STATES:
        m = state == st
        out[f"image_share_{st}"] = float(m.mean())
        out[f"signal_share_{st}"] = float(e_all[m].sum() / e_all.sum())
        out[f"code_share_{st}"] = float(e_q[m].sum() / e_all[m].sum()) if m.any() else np.nan
        out[f"random_share_{st}"] = float(e_r[m].sum() / e_all[m].sum()) if m.any() else np.nan
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "signal_routing_long.csv"
    done = pd.read_csv(f) if f.exists() else pd.DataFrame()
    have = set(zip(done.arm, done.seed, done.epoch)) if len(done) else set()
    rows = done.to_dict("records") if len(done) else []
    rng = np.random.default_rng(0)
    for arm, lf in LANES.items():
        for s in SEEDS:
            d = lf(s)
            ms = milestones(d)
            ms[0] = d / "dev_snapshots" / "milestone_0"
            eps = sorted(ms) if arm == "sham" else [e for e in sorted(ms) if 1 <= e <= 15]
            for e in eps:
                if (arm, s, e) in have:
                    continue
                rows.append({"arm": arm, "seed": s, "epoch": e, **measure(ms[e], rng)})
                print(arm, s, e, {k: round(v, 3) for k, v in rows[-1].items() if k.startswith(("image_share_mis",
                      "signal_share_mis", "code_share_mis", "code_share_rou"))}, flush=True)
    D = pd.DataFrame(rows).sort_values(["arm", "seed", "epoch"]).reset_index(drop=True)
    D.to_csv(f, index=False)
    D.to_csv(SD / "signal_routing_long.csv", index=False)
    res = {}
    keys = [c for c in D.columns if c not in ("arm", "seed", "epoch")]
    for win, eps in (("epochs_1_3", [1, 2, 3]), ("epochs_1_10", list(range(1, 11)))):
        M = D[D.epoch.isin(eps)].groupby(["arm", "seed"])[keys].mean()
        for a, b in (("gradsc", "gradenergy"), ("gradrg", "gradenergy"), ("gradsc", "sham"), ("gradenergy", "sham"),
                     ("gradrand", "sham")):
            for k in keys:
                dl = (M.loc[a][k] - M.loc[b][k]).loc[SEEDS].values
                res[f"{win}|{a} - {b}|{k}"] = {"mean_ci": tci(dl), "n_positive": int((dl > 0).sum()),
                                               "per_seed": np.round(dl, 4).tolist()}
    S = D[D.arm == "sham"]
    for win, eps in (("epoch_1", [1]), ("epochs_1_3", [1, 2, 3]), ("epochs_4_10", list(range(4, 11))),
                     ("epochs_11_30", [e for e in S.epoch.unique() if e >= 11])):
        M = S[S.epoch.isin(eps)].groupby("seed").mean(numeric_only=True)
        dl = (M.code_share_misrouted - M.code_share_routed).loc[SEEDS].values
        res[f"sham|{win}|code_share misrouted - routed"] = {"mean_ci": tci(dl), "n_positive": int((dl > 0).sum()),
                                                            "per_seed": np.round(dl, 4).tolist()}
        for k in ("image_share_misrouted", "signal_share_misrouted", "code_share_misrouted", "code_share_routed",
                  "code_share_correct", "random_share_all"):
            res[f"sham|{win}|{k}"] = tci(M[k].loc[SEEDS].values)
    res["sham_by_epoch_mean"] = S.groupby("epoch")[keys].mean().round(4).reset_index().to_dict("records")
    json.dump(res, open(OUT / "signal_routing_summary.json", "w"), indent=1)
    for k, v in res.items():
        if "|" in k and isinstance(v, dict) and any(t in k for t in ("signal_share_mis", "image_share_mis",
                                                                     "code_share mis", "code_share_misrouted")):
            print(k, [round(z, 3) for z in v["mean_ci"]], "pos", v["n_positive"])


if __name__ == "__main__":
    main()
