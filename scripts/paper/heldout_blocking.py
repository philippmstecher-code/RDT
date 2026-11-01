#!/usr/bin/env python3
"""Held-out check of the acquisition-blocking and suppression contrasts.

The blocked or projected directions were re-estimated during training from the first 20 CIFAR-100 test images of each
class (and, for the energy-matched arm, the learning-signal second moment from the first 2,000 test images in loader
order). The published contrasts are measured on all 10,000 test images. This script recomputes them on
  all   the 10,000 test images (must reproduce the published values),
  ho    the 8,000 images never used to estimate directions (per-class positions 20-99),
  ho2   ho without the first 2,000 test images (also never used for the energy-matched arm's learning signal).
Measures per arm, seed and epoch 1-3 (stored checkpoints; nothing is retrained):
  net_fine        the network's own fine-class accuracy (sample_predictions.pt, captured as evaluated in training)
  net_assignment  share of images whose predicted class lies in the true superclass
  fresh_fine, fresh_assignment, fresh_within   100-way ridge read-out of the pooled activations, 5-fold CV over all
                  10,000 images exactly as gradblock_complete.py, scored on each image subset
  oracle_within   5-way accuracy among the classes of the true superclass, from the same ridge scores
Contrasts: per-seed means over epochs 1-3, paired over seeds (t-based 95% CI), as in gradblock_complete.py.
Order check: one sham checkpoint is re-evaluated on the CIFAR-100 test set in file order; its predictions must equal
the stored per-class predictions, which fixes the mapping from per-class position to test-set index.
Writes analysis_outputs/paper/HELDOUT_blocking/{long.csv, contrasts.csv, summary.json, order_check.json}.
Usage: python scripts/paper/heldout_blocking.py
"""
import json
import os
import pickle
import sys
from pathlib import Path
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coarse_overlap import milestones  # noqa: E402
from class_level import FINE2SC  # noqa: E402
from gradblock_complete import lanes as block_lanes  # noqa: E402
from curriculum_probe import load  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis_outputs/paper/HELDOUT_blocking"
E = Path(RDT_DATA + "/extended/experiments")
CIFAR = E / "_cifar_src/datasets/cifar100/cifar-100-python"
SEEDS = [7, 42, 137, 256, 314]
EPOCHS = [1, 2, 3]
N_EST, N_GRAD = 20, 2000
BLOCK_ARMS = ["sham", "gradsc", "gradrand", "gradenergy", "gradrg",
              "gradjoint", "gradrandfresh", "gradsc_l3", "gradjoint_l3"]
# the layer2-to-pooled pair is not analysed here (its activations were not retained)
# dose-matched arms trained in one lane from epoch 0 (as gradjoint_analysis.NEW)
NEW = {"gradjoint": ("gradjoint", "gradjoint", "Llayer1-layer2-layer3-layer4-avgpool"),
       "gradrandfresh": ("gradrandfresh", "gradrandfresh", "Llayer1-layer2-layer3-layer4-avgpool"),
       "gradsc_l3": ("gradl3", "gradsc", "Llayer3-layer4-avgpool"), "gradjoint_l3": ("gradl3", "gradjoint", "Llayer3-layer4-avgpool")}
SUPP_ARMS = ["supp_sc", "supp_random", "supp_randgroup"]
PAIRS = [("gradsc", "gradrand"), ("gradsc", "gradenergy"), ("gradsc", "gradrg"), ("gradsc", "sham"),
         ("gradenergy", "gradrand"),
         ("supp_sc", "supp_random"), ("supp_sc", "sham"), ("supp_sc", "supp_randgroup"),
         ("gradsc", "gradjoint"), ("gradsc", "gradrandfresh"), ("gradrg", "gradjoint"), ("gradsc_l3", "gradjoint_l3")]
MEASURES = ["net_fine", "net_assignment", "fresh_fine", "fresh_assignment", "fresh_within", "oracle_within"]


def supp_lane(arm, s):
    if arm == "supp_randgroup":
        return E / "randgroup/lanes" / f"lesion_randgroup_e00_seed{s}_rp65_Llayer4-avgpool"
    kind = "scaffold" if arm == "supp_sc" else "random_variance"
    root = "prevention_s42_copy" if s == 42 else "prevention"
    return E / root / "lanes" / f"lesion_{kind}_e00_seed{s}_rp65_Llayer4-avgpool"


def arm_lanes(arm, s):
    if arm in NEW:
        exp, cond, tag = NEW[arm]
        return [E / exp / "lanes" / f"lesion_{cond}_e00_seed{s}_rp65_{tag}"]
    return [supp_lane(arm, s)] if arm in SUPP_ARMS else block_lanes(arm, s)[:1]   # epochs 1-3 lie in the first lane


def test_labels():
    with open(CIFAR / "test", "rb") as f:
        return np.array(pickle.load(f, encoding="latin1")["fine_labels"])


def masks(labels):
    """Per-class-position masks in the stored order (class-major, 100 positions per class)."""
    pos_global = np.stack([np.flatnonzero(labels == c) for c in range(100)])      # (100, 100): test index
    pos = np.tile(np.arange(100), (100, 1))
    ho = pos >= N_EST
    return {"all": np.ones((100, 100), bool), "ho": ho, "ho2": ho & (pos_global >= N_GRAD)}


def order_check(labels):
    """Re-evaluate sham seed 7, epoch 1 on the test set in file order and compare with the stored predictions."""
    import torchvision
    ld = E / "critical_sham/lanes/sham_e00_seed7"
    ms = milestones(ld)[1]
    net = torchvision.models.resnet18(num_classes=100)
    net.load_state_dict(torch.load(ms / "checkpoint.pt", map_location="cpu", weights_only=False))
    net.eval()
    with open(CIFAR / "test", "rb") as f:
        x = pickle.load(f, encoding="latin1")["data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    cfg = json.load(open(E / "critical_sham/_config/template.json"))["transform_config"]   # CIFAR mean/std
    x = (torch.from_numpy(x) - torch.tensor(cfg["normalize_mean"])[:, None, None]) / torch.tensor(cfg["normalize_std"])[:, None, None]
    with torch.no_grad():
        pred = torch.cat([net(x[i:i + 500]).argmax(1) for i in range(0, len(x), 500)]).numpy()
    p = torch.load(ms / "sample_predictions.pt", map_location="cpu", weights_only=False)
    stored = np.stack([p[c]["predictions"].numpy() for c in range(100)])
    mine = np.stack([pred[labels == c] for c in range(100)])
    res = {"lane": ld.name, "epoch": 1, "agreement": float((stored == mine).mean()),
           "stored_acc": float((stored == np.arange(100)[:, None]).mean()), "recomputed_acc": float((mine == np.arange(100)[:, None]).mean())}
    json.dump(res, open(OUT / "order_check.json", "w"), indent=1)
    return res


def tci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = test_labels()
    oc = order_check(labels)
    print("order check", oc, flush=True)
    M = masks(labels)
    y_grid = np.repeat(np.arange(100), 100).reshape(100, 100)
    sc_cls = [np.flatnonzero(FINE2SC == g) for g in range(20)]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    rows = []
    for arm in BLOCK_ARMS + SUPP_ARMS:
        for s in SEEDS:
            ld = arm_lanes(arm, s)[0]
            if not ld.exists():
                print("missing", arm, s, flush=True)
                continue
            ms = milestones(ld)
            for e in EPOCHS:
                if e not in ms:
                    print("no checkpoint", arm, s, e, flush=True)
                    continue
                p = torch.load(ms[e] / "sample_predictions.pt", map_location="cpu", weights_only=False)
                P = np.stack([p[c]["predictions"].numpy() for c in range(100)])
                X, y = load(ms[e], "avgpool")
                assert len(y) == 10000 and (y.reshape(100, 100) == y_grid).all()
                pipe = make_pipeline(StandardScaler(), RidgeClassifier(alpha=10.0))
                S = cross_val_predict(pipe, X, y, cv=cv, method="decision_function")
                R = S.argmax(1)
                orc = np.empty_like(R)
                for i in range(len(y)):
                    cl = sc_cls[FINE2SC[y[i]]]
                    orc[i] = cl[S[i, cl].argmax()]
                R, orc = R.reshape(100, 100), orc.reshape(100, 100)
                for mk, m in M.items():
                    yt, pn, pr, po = y_grid[m], P[m], R[m], orc[m]
                    okn, okr = FINE2SC[pn] == FINE2SC[yt], FINE2SC[pr] == FINE2SC[yt]
                    rows.append({"arm": arm, "seed": s, "epoch": e, "subset": mk, "n": int(m.sum()),
                                 "net_fine": 100 * float((pn == yt).mean()), "net_assignment": 100 * float(okn.mean()),
                                 "fresh_fine": 100 * float((pr == yt).mean()), "fresh_assignment": 100 * float(okr.mean()),
                                 "fresh_within": 100 * float((pr == yt)[okr].mean()), "oracle_within": 100 * float((po == yt).mean())})
            print(arm, s, "done", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "long.csv", index=False)
    W = D.groupby(["subset", "arm", "seed"])[MEASURES].mean()
    out = []
    for mk in M:
        for a, b in PAIRS:
            for key in MEASURES:
                try:
                    A, B = W.loc[(mk, a)][key], W.loc[(mk, b)][key]
                except KeyError:
                    continue
                sd = sorted(set(A.index) & set(B.index))
                d = (A.loc[sd] - B.loc[sd]).values
                m_, lo, hi = tci(d)
                out.append({"subset": mk, "contrast": f"{a} - {b}", "measure": key, "mean": m_, "ci_lo": lo, "ci_hi": hi,
                            "n_negative": int((d < 0).sum()), "n_seeds": len(sd), "per_seed": [round(v, 3) for v in d]})
    C = pd.DataFrame(out)
    C.to_csv(OUT / "contrasts.csv", index=False)
    json.dump({"order_check": oc, "contrasts": out}, open(OUT / "summary.json", "w"), indent=1)
    show = C[C.measure.isin(["net_fine", "fresh_assignment", "oracle_within"])]
    for (c, k), g in show.groupby(["contrast", "measure"], sort=False):
        print(f"{c:30s} {k:16s} " + "  ".join(f"{r.subset}: {r['mean']:+.2f} [{r.ci_lo:+.2f},{r.ci_hi:+.2f}] {r.n_negative}/{r.n_seeds}"
                                             for _, r in g.iterrows()))


if __name__ == "__main__":
    main()
