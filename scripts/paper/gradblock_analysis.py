#!/usr/bin/env python3
"""Acquisition-blocking control: does the representation learn fine classes more slowly when the network receives no
learning signal to build the coarse (superclass) structure?

Arms (ResNet-18, CIFAR-100, seeds 7, 42, 137, 256, 314; branch epoch 0; 15 epochs; scripts/paper/run_gradblock.sh):
  gradsc    epochs 1-10: the gradient at every stage (layer1-4, avgpool) is projected off the current superclass code
            (re-estimated every 65 updates); the forward pass is unchanged
  gradrand  the same with random directions of equal rank (dose-matched in number of directions)
  gradenergy the leading learning-signal directions outside the current superclass code, as large a share of the
            learning signal as the superclass code would carry in that network (dose-matched in learning signal;
            lesion._gradenergy_step; scripts/paper/run_gradenergy.sh)
  sham      critical_sham sham_e00 (standard training, same branch and schedule)
Because the forward pass is unchanged, the stored activations are the network's real features and its own validation
accuracy is a valid measure (no projection artefact).
Measures, per arm, seed and epoch (0-15; epoch 0 = the branch checkpoint, identical forward pass in all arms):
  acquisition    20-way superclass linear read-out (curriculum_probe.probe) on the stored pooled-layer and layer3 activations
  accuracy       the network's fine-class validation accuracy (devtrain_metrics.json)
  fresh read-out 100-way linear read-out (ridge, 5-fold CV) on the stored pooled-layer activations, decomposed into
                 superclass assignment and within-superclass accuracy
  blocked signal share of the per-image fine-class gradient energy at the pooled layer that lies in a rank-19 superclass
                 code (as gradient_routing.py) vs a random subspace of equal rank: the dose of learning signal each arm
                 removes (approximate: the arms' own accumulated bases are not stored)
Contrasts: gradsc - sham, gradsc - gradrand, gradrand - sham; epochs 1-3, 1-10 (window), 11-15 (after release); paired
over seeds, t-based 95% CI, number of seeds with a negative difference.
Writes analysis_outputs/paper/GRADBLOCK/ (long table, summary json) and Source Data
analysis_outputs/paper/figures/final/source_data/gradblock_*.csv.
Usage: python scripts/paper/gradblock_analysis.py
       subset: GB_SEEDS="7 256" GB_ARMS="sham gradsc gradrand" GB_OUT=<subdir> ... (writes only to
       GRADBLOCK/<GB_OUT>, no Source Data)
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
from coarse_overlap import between_scatter, milestones, top  # noqa: E402
from class_level import FINE2SC  # noqa: E402
from gradient_routing import classifier  # noqa: E402
from curriculum_probe import SUPER_OF, load, probe  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis_outputs/paper/GRADBLOCK"
SD = REPO / "analysis_outputs/paper/figures/final/source_data"
E = Path(RDT_DATA + "/extended/experiments")
SEEDS = [int(x) for x in os.environ.get("GB_SEEDS", "7 42 137 256 314").split()]
ARMS = os.environ.get("GB_ARMS", "sham gradsc gradrand gradenergy").split()
PARTIAL = os.environ.get("GB_OUT")
if PARTIAL:
    OUT = OUT / PARTIAL
EPOCHS = list(range(0, 16))  # 0 = branch checkpoint (milestone_0, t0_resume; no validation record)
L5 = "Llayer1-layer2-layer3-layer4-avgpool"
LANES = {"sham": lambda s: E / "critical_sham/lanes" / f"sham_e00_seed{s}",
         "gradsc": lambda s: E / "gradblock/lanes" / f"lesion_gradsc_e00_seed{s}_rp65_{L5}",
         "gradrand": lambda s: E / "gradblock/lanes" / f"lesion_gradrand_e00_seed{s}_rp65_{L5}",
         "gradenergy": lambda s: E / "gradenergy/lanes" / f"lesion_gradenergy_e00_seed{s}_rp65_{L5}"}


def tci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean()), float(x.mean() - h), float(x.mean() + h)]


def val_acc(d):
    m = json.load(open(d / "dev_snapshots/metrics/devtrain_metrics.json"))
    return {int(e["epoch"]): float(e["val_accuracy"]) for e in m}


def blocked_share_log():
    """Energy-matched arm: the learning-signal share its blocked subspace removed (achieved) and the share the
    superclass code carried in the same network (target), per stage and epoch, from the lesion log; mean over seeds."""
    rows = []
    for s in SEEDS:
        f = LANES["gradenergy"](s) / "dev_snapshots/metrics/lesion_log.jsonl"
        if not f.exists():
            continue
        for ev in map(json.loads, open(f)):
            if ev.get("event") != "reproject":
                continue
            for layer, r in ev["report"].items():
                rows.append({"seed": s, "epoch": ev["epoch"], "layer": layer, "rank": r["total_rank"],
                             "target": r["target_grad_share"], "achieved": r["achieved_grad_share"]})
    if not rows:
        return []
    L = pd.DataFrame(rows)
    return L.groupby(["layer", "epoch"])[["rank", "target", "achieved"]].mean().round(3).reset_index().to_dict("records")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    rng = np.random.default_rng(0)
    done = pd.read_csv(OUT / "gradblock_long.csv") if (OUT / "gradblock_long.csv").exists() else pd.DataFrame()
    have = set(zip(done.arm, done.seed, done.epoch)) if len(done) else set()
    rows = done.to_dict("records") if len(done) else []  # incremental: rows already computed are kept
    for arm, lf in LANES.items():
        if arm not in ARMS:
            continue
        for s in SEEDS:
            d = lf(s)
            if not (d / "dev_snapshots/metrics/devtrain_metrics.json").exists():
                print("missing", arm, s, flush=True)
                continue
            ms, acc = milestones(d), val_acc(d)
            ms[0] = d / "dev_snapshots" / "milestone_0"
            for e in EPOCHS:
                if e not in ms or (arm, s, e) in have:
                    continue
                X, y = load(ms[e], "avgpool")
                X3, _ = load(ms[e], "layer3")
                pred = cross_val_predict(make_pipeline(StandardScaler(), RidgeClassifier(alpha=10.0)), X, y, cv=cv)
                ok = FINE2SC[pred] == FINE2SC[y]
                Xd = X.astype(np.float64)
                W, b = classifier(ms[e])
                Z = Xd @ W.T + b
                P = np.exp(Z - Z.max(1, keepdims=True))
                P /= P.sum(1, keepdims=True)
                P[np.arange(len(y)), y] -= 1.0
                G = P @ W
                Q = top(between_scatter(Xd, y, SUPER_OF))
                Qr, _ = np.linalg.qr(rng.standard_normal((X.shape[1], Q.shape[1])))
                tot = (G ** 2).sum()
                rows.append({"arm": arm, "seed": s, "epoch": e, "val_accuracy": acc.get(e, np.nan),
                             "superclass_probe_avgpool": 100 * float(probe(X, SUPER_OF[y])),
                             "superclass_probe_layer3": 100 * float(probe(X3, SUPER_OF[y])),
                             "fresh_fine": 100 * float((pred == y).mean()),
                             "fresh_assignment": 100 * float(ok.mean()),
                             "fresh_within": 100 * float((pred == y)[ok].mean()),
                             "grad_share_superclass_code": float(((G @ Q) ** 2).sum() / tot),
                             "grad_share_random": float(((G @ Qr) ** 2).sum() / tot)})
                print(rows[-1], flush=True)
    D = pd.DataFrame(rows).sort_values(["arm", "seed", "epoch"]).reset_index(drop=True)
    D = D[D.arm.isin(ARMS) & D.seed.isin(SEEDS)]
    D.to_csv(OUT / "gradblock_long.csv", index=False)
    if not PARTIAL:
        D.to_csv(SD / "gradblock_long.csv", index=False)
    res = {}
    keys = ["val_accuracy", "fresh_fine", "fresh_assignment", "fresh_within", "superclass_probe_avgpool",
            "superclass_probe_layer3"]
    for win, eps in (("epochs_1_3", [1, 2, 3]), ("epochs_1_10", list(range(1, 11))), ("epochs_11_15", list(range(11, 16)))):
        M = D[D.epoch.isin(eps)].groupby(["arm", "seed"])[keys].mean()
        present = set(M.index.get_level_values("arm"))
        for a, b in (("gradsc", "sham"), ("gradsc", "gradrand"), ("gradrand", "sham"), ("gradsc", "gradenergy"),
                     ("gradenergy", "sham"), ("gradenergy", "gradrand")):
            if a not in present or b not in present:
                continue
            seeds = sorted(set(M.loc[a].index) & set(M.loc[b].index))
            for k in keys:
                dl = (M.loc[a][k] - M.loc[b][k]).loc[seeds].values
                res[f"{win}|{a} - {b}|{k}"] = {"mean_ci": tci(dl), "n_negative": int((dl < 0).sum()),
                                               "per_seed": np.round(dl, 3).tolist(), "seeds": seeds}
    res["grad_share_by_epoch_sham"] = D[D.arm == "sham"].groupby("epoch")[
        ["grad_share_superclass_code", "grad_share_random"]].mean().round(3).reset_index().to_dict("records")
    res["blocked_share_by_epoch"] = blocked_share_log()
    json.dump(res, open(OUT / "gradblock_summary.json", "w"), indent=1)
    for k, v in res.items():
        if "|" in k:
            print(k, [round(z, 2) for z in v["mean_ci"]], "neg", v["n_negative"])


if __name__ == "__main__":
    main()
