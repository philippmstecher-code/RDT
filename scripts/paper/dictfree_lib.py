"""Shared helpers for the dictionary-free developmental-program analysis (dictfree_*.py).

Restricted to the open networks: core seed 42 and clean / label-noise
standard ResNet-18 seeds 7, 42, 137; loading a held-out lane raises. Read-only on the data volumes.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from noise_null_lib import Lane  # noqa: E402

REPO = HERE.parents[1]
OUT = REPO / "analysis_outputs" / "paper" / "DICTFREE" / "population"
LANES_JSON = REPO / "analysis_outputs" / "paper" / "CORE_lanes" / "lanes.json"
LN_EXP = Path(RDT_DATA + "/core/experiments/0c28799c-ec9d-4e45-ac30-302dff810a8d/lanes")
LN_STANDARD = {7: "498bb1e8-47b0-4e99-be8d-e35e566b1aa7", 42: "0bd1c27f-38f5-449d-8159-144392a92026",
               137: "2cc9ed75-ab1f-444f-a59b-5cc9ac6c8f9f"}
HELDOUT_SEEDS = {256, 314}
OPEN_KEYS = ["core_ResNet-18_s42", "core_ViT-Small_s42", "core_CCT-7_s42",
                  "clean_ResNet-18_s7", "clean_ResNet-18_s42", "clean_ResNet-18_s137"]
POOLED = {"ResNet-18": "avgpool", "ViT-Small": "encoder.ln", "CCT-7": "norm"}


def _guard(seed: int) -> None:
    if int(seed) in HELDOUT_SEEDS:
        raise RuntimeError(f"seed {seed} is a held-out seed (256, 314)")


def open_lanes() -> List[Dict[str, Any]]:
    d = json.load(open(LANES_JSON))
    by = {ln["key"]: ln for ln in d["lanes"]}
    out = []
    for k in OPEN_KEYS:
        ln = dict(by[k])
        _guard(ln["seed"])
        layers = [L for L in ln["layers"] if L != "layer4"]  # layer4 == avgpool bit-for-bit (verified in compute)
        out.append({"key": k, "group": ln["group"], "arch": ln["arch"], "seed": ln["seed"], "dir": ln["dir"],
                    "pooled": ln["pooled"], "layers": layers,
                    "ckpts": [{"label": c["label"], "epoch": float(c["epoch_field"]), "val_acc": c["val_acc"]}
                              for c in ln["ckpts"]]})
    for s, lid in LN_STANDARD.items():
        _guard(s)
        d_ = LN_EXP / lid
        snaps = json.load(open(d_ / "dev_snapshots" / "metrics" / "devtrain_snapshots.json"))
        ck, seen = [], set()
        for sn in sorted(snaps, key=lambda z: z["milestone_index"]):
            if sn["weight_updates"] in seen:
                continue  # final milestone duplicates its predecessor
            seen.add(sn["weight_updates"])
            ck.append({"label": str(sn["milestone_index"]), "epoch": float(sn["epoch"]),
                       "val_acc": sn.get("val_accuracy")})
        out.append({"key": f"lnstd_ResNet-18_s{s}", "group": "lnstd", "arch": "ResNet-18", "seed": s, "dir": str(d_),
                    "pooled": "avgpool", "layers": ["layer1", "layer2", "layer3", "avgpool"], "ckpts": ck})
    for ln in out:
        n = len(ln["ckpts"])
        for i, c in enumerate(ln["ckpts"]):
            c["idx"], c["tau"] = i, i / (n - 1)
    return out


def superclass_of_class(lane_dir: str) -> np.ndarray:
    L = Lane(Path(lane_dir), load_results=True)
    sc = np.full(L.n_classes, -1, dtype=int)
    for j, name in enumerate(sorted(L.superclass_groups)):
        for c in L.superclass_groups[name]:
            sc[c] = j
    assert (sc >= 0).all() and np.bincount(sc).tolist() == [5] * 20
    return sc


def load_acts(ln: Dict[str, Any], label: str) -> Dict[str, np.ndarray]:
    _guard(ln["seed"])
    p = Path(ln["dir"]) / "dev_snapshots" / f"milestone_{label}" / "individual_multilayer_activations.pt"
    a = torch.load(p, map_location="cpu", weights_only=False)
    out = {}
    for L, v in a.items():
        assert sorted(v.keys()) == list(range(100))
        X = torch.cat([v[c].float() for c in range(100)], 0).numpy().astype(np.float64)
        assert X.shape[0] == 10000
        out[L] = X
    return out


def load_preds(ln: Dict[str, Any], label: str) -> np.ndarray:
    _guard(ln["seed"])
    p = Path(ln["dir"]) / "dev_snapshots" / f"milestone_{label}" / "sample_predictions.pt"
    sp = torch.load(p, map_location="cpu", weights_only=False)
    return np.concatenate([sp[c]["predictions"].numpy().ravel() for c in range(100)])
