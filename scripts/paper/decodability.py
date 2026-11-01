#!/usr/bin/env python3
"""Readout-free decodability measures on a lane, per milestone, from the stored (projected) activations with ridge
probes trained on test images 40-69 and tested on 70-99 of every class: S_sc (true superclasses minus the mean over
partitions RandomState(1..10)), S_grp (partition 0 minus the same reference), S_wsc (mean 5-way within-superclass
accuracy minus 0.2), and the head's fine validation accuracy. Usage: import and call lane_measures(lane_dir, stage)."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transfer_lib as T  # noqa: E402

_d = pickle.load(open(T.CIFAR / "test", "rb"), encoding="latin1")
F2S = np.zeros(100, int)
for f, c in zip(_d["fine_labels"], _d["coarse_labels"]):
    F2S[f] = c


def part(p):
    a = np.empty(100, int)
    for g, row in enumerate(np.random.RandomState(p).permutation(100).reshape(20, 5)):
        a[row] = g
    return a


P0, PREF = part(0), [part(p) for p in range(1, 11)]


def ridge(Xtr, ytr, Xte, yte, lam=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    Y = np.eye(ytr.max() + 1)[ytr]
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y)
    return float(((B @ W).argmax(1) == yte).mean())


def measures_from_acts(a):
    cls = np.array(sorted(a))
    X = np.stack([a[c].numpy() for c in cls]).astype(np.float64)
    tr, te = X[:, 40:70].reshape(-1, X.shape[-1]), X[:, 70:].reshape(-1, X.shape[-1])
    ytr, yte = np.repeat(cls, 30), np.repeat(cls, 30)
    ref = np.mean([ridge(tr, p[ytr], te, p[yte]) for p in PREF])
    s_sc = ridge(tr, F2S[ytr], te, F2S[yte]) - ref
    s_grp = ridge(tr, P0[ytr], te, P0[yte]) - ref
    ws = []
    for k in range(20):
        cs = cls[F2S[cls] == k]
        mtr, mte = np.isin(ytr, cs), np.isin(yte, cs)
        ws.append(ridge(tr[mtr], np.searchsorted(cs, ytr[mtr]), te[mte], np.searchsorted(cs, yte[mte])) - 0.2)
    return {"S_sc": s_sc, "S_grp": s_grp, "S_wsc": float(np.mean(ws))}


def lane_measures(lane, stage="avgpool", only_wu=None):
    lane = Path(lane)
    snaps = json.load(open(lane / "dev_snapshots" / "metrics" / "devtrain_snapshots.json"))
    snaps = snaps.get("snapshots", snaps) if isinstance(snaps, dict) else snaps
    out = {}
    for sn in snaps:
        if sn.get("snapshot_type") == "final":
            continue
        wu = int(sn.get("weight_updates", 0))
        if only_wu is not None and wu not in only_wu:
            continue
        a = torch.load(lane / "dev_snapshots" / Path(sn["snapshot_dir"]).name / "individual_multilayer_activations.pt",
                       map_location="cpu", weights_only=False)[stage]
        out[wu] = {**measures_from_acts(a), "acc": sn.get("val_accuracy")}
    return out
