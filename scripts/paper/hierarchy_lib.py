"""Shared helpers for the class-hierarchy analysis (hierarchy_geometry.py, hierarchy_sae.py, hierarchy_report.py),
open seeds only (7, 42, 137).

Read-only on all data. `lane_dir` accepts only the open seeds.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from imposed_grouping_lib import (canonical_classes, cifar100_superclasses, imposed_label_map,  # noqa: E402
                           label_vector, partition_seed_of, true_label_map)

torch.set_num_threads(2)

REPO = HERE.parents[1]
OUT = REPO / "analysis_outputs" / "paper" / "DICTFREE" / "hierarchy"
EXPS = Path(RDT_DATA + "/extended/experiments")
SEEDS = [7, 42, 137]
EXCLUDED = {256, 314}
LAYERS = ["layer1", "layer2", "layer3", "layer4", "avgpool"]


def lane_dir(kind: str, seed: int) -> Path:
    assert seed not in EXCLUDED and seed in SEEDS, f"seed {seed} is not in the open set"
    if kind == "taskirr":
        return EXPS / "taskirr" / "lanes" / f"taskirr_within_index_seed{seed}"
    return EXPS / "curriculum" / "lanes" / f"{kind}_seed{seed}"


def snapshots(d: Path) -> List[dict]:
    """Real checkpoints (duplicate final dropped), with epoch from weight updates (391/epoch)."""
    snaps = json.load(open(d / "dev_snapshots" / "metrics" / "devtrain_snapshots.json"))
    out = []
    for i, sn in enumerate(snaps):
        if i and sn["weight_updates"] == snaps[i - 1]["weight_updates"]:
            continue
        out.append({"ms": sn["milestone_index"], "epoch": int(round(sn["weight_updates"] / 391)),
                    "val_acc": sn["val_accuracy"]})
    return out


def load_acts(d: Path, ms: int, layers=LAYERS) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    a = torch.load(d / "dev_snapshots" / f"milestone_{ms}" / "individual_multilayer_activations.pt",
                   map_location="cpu", weights_only=False)
    X = {L: np.concatenate([a[L][c].float().numpy().reshape(len(a[L][c]), -1) for c in sorted(a[L])])
         .astype(np.float64) for L in layers}
    y = np.concatenate([np.full(len(a[layers[0]][c]), c) for c in sorted(a[layers[0]])])
    return X, y


_CLASSES = None


def classes() -> List[str]:
    global _CLASSES
    if _CLASSES is None:
        _CLASSES = canonical_classes()
    return _CLASSES


def true_vec() -> np.ndarray:
    return label_vector(true_label_map(classes()))


def imposed_vec(seed: int) -> np.ndarray:
    assert seed in SEEDS
    return label_vector(imposed_label_map(classes(), partition_seed_of(seed)))


def taskirr_vec() -> np.ndarray:
    """Label of each class in taskirr: index of the class within its superclass (alphabetical fine names)."""
    sc = cifar100_superclasses()
    idx = {f: i for s, fs in sc.items() for i, f in enumerate(sorted(fs))}
    return np.array([idx[c] for c in classes()], dtype=int)


def null_partitions(n: int, template: np.ndarray, base: int = 50000) -> List[np.ndarray]:
    return [np.random.RandomState(base + i).permutation(template) for i in range(n)]


# geometry

class Geo:
    """Class-centroid geometry of one (N x d) activation matrix with balanced classes."""

    def __init__(self, X: np.ndarray, y: np.ndarray, shrink: float = 0.1):
        self.C = int(y.max()) + 1
        self.n = np.bincount(y, minlength=self.C)
        mu = np.stack([X[y == c].mean(0) for c in range(self.C)])
        m = mu.mean(0)
        self.mu = mu - m  # centred centroids
        R = X - mu[y]
        self.W = float((R ** 2).sum())
        self.B = float(self.n[0] * (self.mu ** 2).sum())
        # within-class covariance, shrunk toward scaled identity, for the whitened (Fisher) geometry
        Sw = R.T @ R / len(X)
        t = np.trace(Sw) / Sw.shape[0]
        Sw = (1 - shrink) * Sw + shrink * t * np.eye(Sw.shape[0])
        ev, V = np.linalg.eigh(Sw)
        self.muw = self.mu @ V @ np.diag(ev ** -0.5) @ V.T
        cn = self.mu / (np.linalg.norm(self.mu, axis=1, keepdims=True) + 1e-12)
        self.cos = cn @ cn.T
        self.iu = np.triu_indices(self.C, 1)

    def class_fisher(self) -> Dict[str, float]:
        ev = np.linalg.eigvalsh(self.mu.T @ self.mu)
        ev = np.clip(ev, 0, None)
        pr = float(ev.sum() ** 2 / (ev ** 2).sum())
        return {"between_frac": self.B / (self.B + self.W),
                "fisher_w": float((self.muw ** 2).sum(1).mean()),  # tr(Sw^-1 Sb) for centroids
                "centroid_pr": pr}

    @staticmethod
    def _share(M: np.ndarray, g: np.ndarray) -> float:
        G = np.stack([M[g == k].mean(0) for k in np.unique(g)])
        sizes = np.array([(g == k).sum() for k in np.unique(g)])
        return float((sizes[:, None] * G ** 2).sum() / (M ** 2).sum())

    def auc(self, g: np.ndarray) -> float:
        s = self.cos[self.iu]
        same = (g[:, None] == g[None, :])[self.iu]
        from scipy.stats import rankdata
        r = rankdata(s)
        n1, n0 = same.sum(), (~same).sum()
        return float((r[same].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    def part(self, g: np.ndarray) -> Dict[str, float]:
        """share: fraction of between-class SS carried by the group means (raw / whitened);
        auc: P(cos(same-group centroid pair) > cos(different-group pair));
        wg_sep: whitened within-group between-class spread per class (Fisher units) = how well classes
        inside a group stay apart (low = coarse-level collapse)."""
        sh_w = self._share(self.muw, g)
        return {"share": self._share(self.mu, g), "share_w": sh_w, "auc": self.auc(g),
                "wg_sep": float((self.muw ** 2).sum(1).mean() * (1 - sh_w))}


# probes

def _ridge_fit_predict(Xtr, ytr, Xte, alpha=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A, Bm = (Xtr - mu) / sd, (Xte - mu) / sd
    cl = np.unique(ytr)
    Y = (ytr[:, None] == cl[None, :]).astype(np.float64) * 2 - 1
    A1 = np.hstack([A, np.ones((len(A), 1))])
    reg = alpha * np.eye(A1.shape[1])
    reg[-1, -1] = 0
    Wt = np.linalg.solve(A1.T @ A1 + reg, A1.T @ Y)
    return cl[np.argmax(np.hstack([Bm, np.ones((len(Bm), 1))]) @ Wt, 1)]


def cv_probe(X, lab, k=5, seed=0) -> float:
    """Stratified-ish k-fold (per-label round robin after shuffling), ridge alpha 10 on standardised features."""
    rng = np.random.RandomState(seed)
    fold = np.empty(len(lab), int)
    for c in np.unique(lab):
        ix = np.where(lab == c)[0]
        rng.shuffle(ix)
        fold[ix] = np.arange(len(ix)) % k
    acc = []
    for f in range(k):
        te = fold == f
        acc.append((_ridge_fit_predict(X[~te], lab[~te], X[te]) == lab[te]).mean())
    return float(np.mean(acc))


def within_group_probe(X, y, g) -> float:
    """Mean over groups of the CV accuracy of a class probe among the classes of one group."""
    return float(np.mean([cv_probe(X[np.isin(y, np.where(g == k)[0])], y[np.isin(y, np.where(g == k)[0])])
                          for k in np.unique(g)]))


def heldout_class_transfer(X, y, g) -> float:
    """Group probe trained on 4/5 of each group's classes, tested on the held-out classes (5 folds).
    Measures group structure that generalises to unseen member classes; chance = 1/n_groups."""
    order = np.zeros(len(g), int)
    for k in np.unique(g):
        mem = np.where(g == k)[0]
        order[mem] = np.arange(len(mem))
    nf = int(order.max()) + 1
    acc = []
    for f in range(nf):
        te_cls = order == f
        te = te_cls[y]
        acc.append((_ridge_fit_predict(X[~te], g[y[~te]], X[te]) == g[y[te]]).mean())
    return float(np.mean(acc))
