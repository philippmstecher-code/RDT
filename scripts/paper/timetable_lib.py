"""Shared helpers for the developmental-timetable analysis (timetable_*.py), open networks only (held-out seeds 256 and 314 excluded).

Canonical image order: row = c*100 + j, c = CIFAR-100 fine index (alphabetical, torchvision order),
j = position of the image among class-c images in the CIFAR-100 test set (file order).
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ML_SRC = REPO / "packages" / "ml" / "src"
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))
from cifar100_hierarchy import CIFAR100_SUPERCLASSES  # noqa: E402

OUT = REPO / "analysis_outputs" / "paper" / "DICTFREE" / "B_timetable"
CACHE = OUT / "cache"
CIFAR = Path(RDT_DATA + "/extended/experiments/_cifar_src/datasets/cifar100/cifar-100-python")
NOISE_EXP = Path(RDT_DATA + "/core/experiments/0c28799c-ec9d-4e45-ac30-302dff810a8d/lanes")

HELDOUT_SEEDS = {256, 314}


def cifar_meta():
    with open(CIFAR / "meta", "rb") as f:
        m = pickle.load(f, encoding="latin1")
    return m["fine_label_names"], m["coarse_label_names"]


FINE_NAMES, _ = cifar_meta()
SUPER_NAMES = list(CIFAR100_SUPERCLASSES.keys())
FINE2SUPER = {f: s for s, fs in CIFAR100_SUPERCLASSES.items() for f in fs}
SC_OF_CLASS = np.array([SUPER_NAMES.index(FINE2SUPER[n]) for n in FINE_NAMES])  # canonical class -> sc idx
SC_OF_IMG = np.repeat(SC_OF_CLASS, 100)
CLS_OF_IMG = np.repeat(np.arange(100), 100)


def load_split(train: bool):
    """Return (X uint8 (N,32,32,3), fine labels) in file order."""
    with open(CIFAR / ("train" if train else "test"), "rb") as f:
        d = pickle.load(f, encoding="latin1")
    X = d["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return X, np.array(d["fine_labels"])


def test_canonical():
    """Test images in canonical order (class-major, file order within class)."""
    X, y = load_split(False)
    order = np.argsort(y, kind="stable")
    return X[order], y[order]


def lanes():
    """Open lanes. key -> dict(dir, arch, seed, group, epochs, ckpt labels)."""
    L = json.load(open(REPO / "analysis_outputs/paper/CORE_lanes/lanes.json"))["lanes"]
    out = {}
    for l in L:
        if l["group"] == "core" and l["seed"] != 42:
            continue
        if l["seed"] in HELDOUT_SEEDS:
            continue
        out[l["key"]] = dict(dir=l["dir"], arch=l["arch"], seed=l["seed"], group=l["group"],
                             labels=[c["label"] for c in l["ckpts"]],
                             epochs=[float(c["epoch"]) for c in l["ckpts"]],
                             val_acc=[c["val_acc"] for c in l["ckpts"]])
    import pandas as pd
    m = pd.read_csv(REPO / "analysis_outputs/paper/NOISE_scaffold/per_lane_metrics.csv")
    m = m[(m.condition == "standard") & (m.seed.isin([7, 42, 137]))]
    for (lid, seed), g in m.groupby(["lane_id", "seed"]):
        g = g.sort_values("milestone")
        out[f"noise_std_ResNet-18_s{seed}"] = dict(dir=str(NOISE_EXP / lid), arch="ResNet-18", seed=int(seed),
                                                   group="noise_std", labels=[str(x) for x in g.milestone],
                                                   epochs=[float(x) for x in g.epoch],
                                                   val_acc=[float(x) * 100 for x in g.val_acc_logged])
    for k, v in out.items():
        assert v["seed"] not in HELDOUT_SEEDS, k
    return out


def lane_class_names(lane):
    """Lane class idx -> bare CIFAR fine name."""
    if lane["group"] == "core":
        meta = json.load(open(Path(lane["dir"]) / "sae_analysis" / "sae_results.json"))["metadata"]
        return [c.split("_", 1)[1] if c.startswith("cifar100_") else c for c in meta["selected_classes"]]
    return list(FINE_NAMES)


def to_canonical_perm(lane):
    """lane class idx -> canonical class idx."""
    names = lane_class_names(lane)
    return np.array([FINE_NAMES.index(n) for n in names])
