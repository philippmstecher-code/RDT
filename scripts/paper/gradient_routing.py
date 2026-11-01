#!/usr/bin/env python3
"""Does the coarse shared structure organise where fine-class learning happens ("routing of learning")?

Sham networks (critical_sham sham_e00_seed{7,42,137,256,314}; standard training), checkpoints at epochs 1-15, 20, 25,
30. For every stored test image (100 per class) the gradient of the fine-class cross-entropy with respect to the pooled
features x is g = W'(softmax(Wx + b) - onehot(y)) (W, b: the network's classifier at that checkpoint; exact, no
approximation). This is the learning signal that arrives at the last block.
Measures per checkpoint:
  coarse_share     share of the summed squared gradient that lies in the coarse subspace (19 leading eigenvectors of
                   the between-superclass scatter of the activations, as removed by the suppression arm); null: random
                   19-dim subspace (expected 19/512) and the code of a random grouping of the classes (RandomState(p)).
  shared_share     share of the class-mean gradients' energy that is shared within superclasses: ||mean over the
                   superclass of the class-mean gradients||^2 summed over superclasses (x5) / total class-mean gradient
                   energy; null: the same for 20 random groupings of the classes (RandomState(p), p = 0..19).
  same_sc_cos      mean cosine between class-mean gradients of classes in the same superclass minus the mean over
                   classes in different superclasses.
Prediction (coarse structure routes learning): early in training most of the learning signal lies in the coarse
subspace and is shared within superclasses, more than for random groupings; the share falls as training proceeds.
Writes analysis_outputs/paper/RANDGROUP_control/gradient_routing.csv and gradient_routing_summary.json.
Usage: python scripts/paper/gradient_routing.py
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coarse_overlap import between_scatter, grouping, milestones, top  # noqa: E402
from curriculum_probe import SUPER_OF, load  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis_outputs/paper/RANDGROUP_control"
E = Path(RDT_DATA + "/extended/experiments")
SEEDS = [7, 42, 137, 256, 314]
NGROUP = 20


def classifier(ms_dir):
    sd = torch.load(ms_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd.get("state_dict", sd)) if isinstance(sd, dict) else sd
    return sd["fc.weight"].double().numpy(), sd["fc.bias"].double().numpy()


def shared_share(Gc, groups):
    """Energy of the within-group mean of the class-mean gradients (x group size) / total energy."""
    tot = (Gc ** 2).sum()
    s = 0.0
    for k in np.unique(groups):
        m = Gc[groups == k]
        s += len(m) * (m.mean(0) ** 2).sum()
    return float(s / tot)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rand_groups = [grouping(np.random.RandomState(p)) for p in range(NGROUP)]
    rows = []
    for s in SEEDS:
        ms = milestones(E / "critical_sham/lanes" / f"sham_e00_seed{s}")
        for e in sorted(ms):
            X, y = load(ms[e], "avgpool")
            X = X.astype(np.float64)
            W, b = classifier(ms[e])
            Z = X @ W.T + b
            P = np.exp(Z - Z.max(1, keepdims=True))
            P /= P.sum(1, keepdims=True)
            P[np.arange(len(y)), y] -= 1.0
            G = P @ W                                   # per-image gradient w.r.t. the pooled features
            Gc = np.stack([G[y == c].mean(0) for c in range(100)])
            Qsc = top(between_scatter(X, y, SUPER_OF))
            tot = (G ** 2).sum()
            coarse = float(((G @ Qsc) ** 2).sum() / tot)
            Qr, _ = np.linalg.qr(rng.standard_normal((X.shape[1], Qsc.shape[1])))
            rnd = float(((G @ Qr) ** 2).sum() / tot)
            rg = [float(((G @ top(between_scatter(X, y, g))) ** 2).sum() / tot) for g in rand_groups[:5]]
            C = Gc / np.linalg.norm(Gc, axis=1, keepdims=True)
            cos = C @ C.T
            same = SUPER_OF[:, None] == SUPER_OF[None, :]
            off = ~np.eye(100, dtype=bool)
            rows.append({"seed": s, "epoch": e, "coarse_share": coarse, "random_subspace_share": rnd,
                         "randgroup_code_share_mean": float(np.mean(rg)),
                         "shared_share_superclass": shared_share(Gc, SUPER_OF),
                         "shared_share_randgroups_mean": float(np.mean([shared_share(Gc, g) for g in rand_groups])),
                         "shared_share_randgroups_max": float(np.max([shared_share(Gc, g) for g in rand_groups])),
                         "same_sc_cos_minus_diff": float(cos[same & off].mean() - cos[~same].mean())})
            print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "gradient_routing.csv", index=False)
    summ = d.groupby("epoch").mean(numeric_only=True).drop(columns="seed").round(4)
    n_above = d.assign(above=d.shared_share_superclass > d.shared_share_randgroups_max).groupby("epoch").above.sum()
    res = {"mean_by_epoch": summ.reset_index().to_dict("records"),
           "n_seeds_superclass_shared_share_above_all_20_groupings": {int(k): int(v) for k, v in n_above.items()}}
    json.dump(res, open(OUT / "gradient_routing_summary.json", "w"), indent=1)
    print(summ.to_string())


if __name__ == "__main__":
    main()
