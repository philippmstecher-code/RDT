#!/usr/bin/env python3
"""SAE-free linear probes of fine-class and superclass structure in the curriculum runs (curriculum).

Linear read-out of the stored curriculum checkpoint activations (100 test images per class,
10,000 total) at every milestone of the sc_true / sc_rand / scratch arms:
  fine100     100-way fine-class probe (chance 1%)
  within_sc   5-way fine-class probe inside each TRUE superclass, averaged over the 20 superclasses
              (chance 20%); removes the trivial contribution of superclass separability
  super20     20-way true-superclass probe (chance 5%)
Probe = RidgeClassifier on standardised features, 5-fold stratified CV (closed form, no tuning
beyond a fixed alpha). Writes analysis_outputs/paper/CURRICULUM_probe/probe.csv.

Usage: python scripts/paper/curriculum_probe.py [--layers avgpool layer3]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "packages" / "ml" / "src"))
from cifar100_hierarchy import CIFAR100_SUPERCLASSES  # noqa: E402

LANES = Path(RDT_DATA + "/extended/experiments/curriculum/lanes")
OUT = REPO / "analysis_outputs" / "paper" / "CURRICULUM_probe"
FINE = sorted(f for fs in CIFAR100_SUPERCLASSES.values() for f in fs)  # torchvision order (alphabetical)
SUP = {f: i for i, s in enumerate(sorted(CIFAR100_SUPERCLASSES)) for f in CIFAR100_SUPERCLASSES[s]}
SUPER_OF = np.array([SUP[f] for f in FINE])


def probe(X, y, alpha=10.0):
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    return cross_val_score(make_pipeline(StandardScaler(), RidgeClassifier(alpha=alpha)), X, y, cv=cv).mean()


def load(ms_dir, layer):
    a = torch.load(ms_dir / "individual_multilayer_activations.pt", map_location="cpu", weights_only=False)[layer]
    X = np.concatenate([a[c].float().numpy().reshape(len(a[c]), -1) for c in sorted(a)])
    y = np.concatenate([np.full(len(a[c]), c) for c in sorted(a)])
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", default=["avgpool"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for lane in sorted(p for p in LANES.iterdir() if (p / ".lane_done").exists()):
        snaps = json.load(open(lane / "dev_snapshots" / "metrics" / "devtrain_snapshots.json"))
        arm, seed = lane.name.rsplit("_", 2)[0], int(lane.name.rsplit("seed", 1)[1])
        for s in snaps:
            ms = lane / "dev_snapshots" / f"milestone_{s['milestone_index']}"
            for layer in args.layers:
                X, y = load(ms, layer)
                ysc = SUPER_OF[y]
                within = np.mean([probe(X[ysc == g], y[ysc == g]) for g in range(20)])
                row = dict(arm=arm, seed=seed, epoch=s["epoch"], layer=layer, fine100=probe(X, y),
                           within_sc=within, super20=probe(X, ysc), val_acc=s["val_accuracy"])
                rows.append(row)
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "probe.csv", index=False)


if __name__ == "__main__":
    main()
