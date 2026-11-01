"""Shared helpers for the RDT lineage analyses (CPU only, read-only on the data drive).

Cache: compact per-lane, per-layer summaries extracted once from the stored SAE codes by extract.py
(class-mean |code|, active counts, Phase-5 matches between consecutive epochs, pooled-layer decoders).
Everything downstream (selectivity under the true or a random partition, lineage, routing) is computed from the cache.
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path

import numpy as np
from scipy import stats
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from dependency_lib import COARSE_TO_FINE, FINE_TO_COARSE  # noqa: E402

REPO = HERE.parents[2]
OUT = REPO / "analysis_outputs/paper/RDT_lineage/analyses"
E = Path(RDT_DATA + "/extended/experiments")
CACHE = Path(os.environ.get("RDT_CACHE", str(Path(__file__).resolve().parents[3] / ".cache/lineage")))
PER_LANE = REPO / "analysis_outputs/paper/RDT_completion/per_lane"
SEEDS = [7, 42, 137, 256, 314]
LAYERS = ("avgpool", "layer3", "layer2", "layer1")      # census layers (layer4 == avgpool in this ResNet-18)
EPOCHS = list(range(16))
TRUE_G = FINE_TO_COARSE.copy()                           # class index -> superclass (class-major codes, CIFAR order)


def lane_dir(arm: str, s: int) -> Path:
    if arm == "sham":
        return E / f"sham_perepoch/lanes/sham_e00_seed{s}"
    return E / f"blocking_rdt/lanes/{arm}_seed{s}"


def common_thresholds(s: int) -> dict:
    """One common set per seed: the sham's stored analysis thresholds (as RDT_completion)."""
    d = json.load(open(PER_LANE / f"sham_seed{s}.json"))
    return d["tsets"]["common"]["thresholds"]


def random_partitions(n: int, seed0: int = 1) -> list:
    out = []
    for p in range(seed0, seed0 + n):
        g = np.empty(100, int)
        for k, row in enumerate(np.random.RandomState(p).permutation(100).reshape(20, 5)):
            g[row] = k
        out.append(g)
    return out


def sel_from_M(M: np.ndarray, g: np.ndarray = TRUE_G) -> dict:
    """Phase-4 selectivity (dependency_lib.selectivity logic) from class-mean |code| M (100, F) under grouping g
    (class -> group of five). Returns ssi, csi, sai, best group, best class."""
    eps = 1e-10
    Mt = M.T.astype(np.float64)                               # (F, 100)
    tot = Mt.sum(1)
    S = np.stack([Mt[:, g == c].sum(1) for c in range(20)], 1)
    frac = S / (tot[:, None] + eps)
    best = frac.argmax(1)
    ssi = np.maximum((frac.max(1) - 1 / 20) / (1 - 1 / 20 + eps), 0)
    p = Mt / (tot[:, None] + eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.where(p > 0, p * np.log(p + eps), 0).sum(1)
    sai = ent / np.log(100)
    members = np.stack([np.where(g == c)[0] for c in range(20)])   # (20, 5), sorted class idx
    W = np.take_along_axis(Mt, members[best], 1)
    csi = np.maximum((W.max(1) / (W.sum(1) + eps) - 0.2) / (0.8 + eps), 0)
    act = tot > eps
    return {"ssi": np.where(act, ssi, 0), "csi": np.where(act, csi, 0), "sai": np.where(act, sai, 0),
            "best_g": best, "best_class": Mt.argmax(1)}


def masks(sel: dict, alive: np.ndarray, t: dict) -> dict:
    """RDT census categories (blocking_rdt_analysis.census_ex): task-general, superclass-selective (SSI > t,
    CSI <= t, not task-general), class-selective (CSI > t, not task-general)."""
    tg = alive & (sel["sai"] > t["sai_task_general_thresh"])
    cs = sel["csi"] > t["csi_differentiation_thresh"]
    sc = alive & (sel["ssi"] > t["ssi_abstraction_thresh"]) & ~cs & ~tg
    cl = alive & cs & ~tg
    return {"tg": tg, "sc": sc, "cl": cl}


class Cache:
    """Per-lane cache (written by extract.py)."""

    def __init__(self, arm: str, s: int):
        self.arm, self.seed = arm, s
        self.dir = CACHE / f"{arm}_seed{s}"
        self._z = {}

    def z(self, layer: str):
        if layer not in self._z:
            self._z[layer] = np.load(self.dir / f"{layer}.npz")
        return self._z[layer]

    def M(self, layer, e):
        return self.z(layer)["M"][e]

    def alive(self, layer, e):
        return self.z(layer)["cnt"][e] >= 100

    def stable(self, layer, e):
        """Stable Phase-5 pairs for the transition e-1 -> e: (idx at e-1, idx at e, r)."""
        a = self.z(layer)[f"st_{e}"]
        return a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2]

    def transformed(self, layer, e):
        a = self.z(layer)[f"tr_{e}"]
        return a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2]

    def born(self, layer, e):
        return self.z(layer)[f"born_{e}"].astype(int)


def tci(d) -> dict:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return {"mean": float(d.mean()) if n else float("nan"), "ci95": [float("nan")] * 2, "n": n}
    h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return {"mean": float(d.mean()), "ci95": [float(d.mean() - h), float(d.mean() + h)], "n": n,
            "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()), "per_seed": np.round(d, 4).tolist()}


def fmt(r: dict, nd=3) -> str:
    return (f"{r['mean']:+.{nd}f} (CI {r['ci95'][0]:+.{nd}f} to {r['ci95'][1]:+.{nd}f}; "
            f"{r.get('n_pos', 0)}+/{r.get('n_neg', 0)}- of {r['n']})")


def off_share(M: np.ndarray, cols: np.ndarray, k: np.ndarray, g: np.ndarray = TRUE_G) -> np.ndarray:
    """Share of each feature's summed class-mean |code| that falls on classes outside the group of class k
    (chance 0.95; blocking_feature_routing.measure computes the same from per-image codes)."""
    X = M[:, cols].astype(np.float64)                     # (100, n)
    same = g[None, :] == g[k][:, None]                    # (n, 100)
    return 1 - (X.T * same).sum(1) / (X.sum(0) + 1e-12)


def sib_enrich(M: np.ndarray, cols: np.ndarray, k: np.ndarray, g: np.ndarray = TRUE_G) -> np.ndarray:
    """Leave-own-class-out sibling share over its chance value: of a feature's class-mean |code| outside its own
    class k, the share on the 4 other classes of k's group, divided by 4/99. 1 = no preference for siblings.
    Removes the by-construction part of off_share (mass on k itself)."""
    X = M[:, cols].astype(np.float64).T                   # (n, 100)
    same = g[None, :] == g[k][:, None]
    own = np.zeros_like(same); own[np.arange(len(k)), k] = True
    sib = (X * (same & ~own)).sum(1)
    rest = (X * ~own).sum(1)
    return (sib / (rest + 1e-12)) / (4 / 99)
