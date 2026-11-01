"""Shared helpers for the dependency tests T1-T4 (class-selective features built on superclass-level features).

Substrate: the 5 clean ('standard') lanes of the CIFAR-100 ResNet-18 label-noise
experiment (seeds 7, 42, 137, 256, 314; 30 epochs; milestones every 5 epochs).
Read-only with respect to the data drive.
"""
from __future__ import annotations

import json
import os
import pickle
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

REPO = Path(__file__).resolve().parents[2]
ML_SRC = REPO / "packages" / "ml" / "src"
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))

DATA = Path(RDT_DATA + "/core")
EXP_ID = "0c28799c-ec9d-4e45-ac30-302dff810a8d"
EXP = DATA / "experiments" / EXP_ID
DB = DATA / "rcx.db"
OUT = REPO / "analysis_outputs" / "paper" / "DEPENDENCY"
CACHE = Path(os.environ.get("RDT_CACHE_DEPENDENCY", "/tmp/dependency_cache"))  # local cache, never on the data drive

MILESTONES = list(range(7))  # milestone_7 == milestone_6 (terminal re-capture)
EPOCH = {0: 0, 1: 5, 2: 10, 3: 15, 4: 20, 5: 25, 6: 30}

COARSE_TO_FINE = {
    0: [4, 30, 55, 72, 95], 1: [1, 32, 67, 73, 91], 2: [54, 62, 70, 82, 92],
    3: [9, 10, 16, 28, 61], 4: [0, 51, 53, 57, 83], 5: [22, 39, 40, 86, 87],
    6: [5, 20, 25, 84, 94], 7: [6, 7, 14, 18, 24], 8: [3, 42, 43, 88, 97],
    9: [12, 17, 37, 68, 76], 10: [23, 33, 49, 60, 71], 11: [15, 19, 21, 31, 38],
    12: [34, 63, 64, 66, 75], 13: [26, 45, 77, 79, 99], 14: [2, 11, 35, 46, 98],
    15: [27, 29, 44, 78, 93], 16: [36, 50, 65, 74, 80], 17: [47, 52, 56, 59, 96],
    18: [8, 13, 48, 58, 90], 19: [41, 69, 81, 85, 89],
}
FINE_TO_COARSE = np.zeros(100, dtype=int)
for _c, _fs in COARSE_TO_FINE.items():
    FINE_TO_COARSE[_fs] = _c
Y_FINE = np.repeat(np.arange(100), 100)  # stored activations are class-major, 100/class
Y_SC = FINE_TO_COARSE[Y_FINE]


def standard_lanes() -> List[Dict]:
    con = sqlite3.connect(f"file:{DB}?immutable=1", uri=True)
    rows = con.execute("select id, name from lanes where experiment_id=? and name like '%-standard-%'",
                       (EXP_ID,)).fetchall()
    con.close()
    out = [{"lane_id": lid, "seed": int(n.rsplit("-seed", 1)[1]), "dir": EXP / "lanes" / lid} for lid, n in rows]
    return sorted(out, key=lambda r: r["seed"])


class LaneData:
    def __init__(self, lane_dir: Path):
        self.dir = Path(lane_dir)
        self.sae_dir = self.dir / "sae_analysis"
        cache = CACHE / self.dir.name / "sel.pkl"
        if cache.exists():
            with open(cache, "rb") as f:
                self.sel, self.meta = pickle.load(f)
        else:
            with open(self.sae_dir / "sae_results.json") as f:
                r = json.load(f)
            self.meta = r["metadata"]
            self.sel = {lab: {ly: {k: np.asarray(v[k], dtype=np.float64) for k in ("feature_ssi", "feature_csi", "feature_sai")}
                              for ly, v in d.items()} for lab, d in r["selectivity"].items()}
            del r
            cache.parent.mkdir(parents=True, exist_ok=True)
            with open(cache, "wb") as f:
                pickle.dump((self.sel, self.meta), f)
        t = self.meta["analysis_thresholds"]
        self.ssi_t = t["ssi_abstraction_thresh"]
        self.csi_t = t["csi_differentiation_thresh"]
        self.sai_t = t["sai_task_general_thresh"]
        a = self.meta["adaptive_thresholds"]
        self.null = {k: (a[f"{k}_null_mean"], a[f"{k}_null_std"]) for k in ("ssi", "csi", "sai")}

    def codes(self, m: int, layer: str) -> np.ndarray:
        return np.load(self.sae_dir / "activations" / str(m) / f"{layer}.npy")

    def sae(self, m: int, layer: str) -> Dict[str, torch.Tensor]:
        return torch.load(self.sae_dir / "saes" / str(m) / f"{layer}.pt", map_location="cpu", weights_only=True)

    def raw(self, m: int) -> Dict[str, np.ndarray]:
        """Stored spatially-averaged activations (class-major, 100/class)."""
        cache = CACHE / self.dir.name / f"raw_m{m}.npz"
        if cache.exists():
            z = np.load(cache)
            out = {k: z[k] for k in z.files}
            out["layer4"] = out["avgpool"]  # identical in this ResNet-18 (1x1 map); verified
            return out
        d = torch.load(self.dir / "dev_snapshots" / f"milestone_{m}" / "individual_multilayer_activations.pt",
                       map_location="cpu", weights_only=False)
        out = {ly: torch.cat([d[ly][c].float() for c in range(100)], 0).numpy() for ly in ("layer3", "layer4", "avgpool")}
        assert np.array_equal(out["layer4"], out["avgpool"])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, layer3=out["layer3"], avgpool=out["avgpool"])
        return out

    def checkpoint(self, m: int):
        return torch.load(self.dir / "dev_snapshots" / f"milestone_{m}" / "checkpoint.pt", map_location="cpu",
                          weights_only=False)

    def masks(self, m: int, layer: str, alive: np.ndarray) -> Dict[str, np.ndarray]:
        s = self.sel[str(m)][layer]
        sai = (s["feature_sai"] > self.sai_t) & alive
        ssi_any = (s["feature_ssi"] > self.ssi_t) & ~sai & alive
        csi = (s["feature_csi"] > self.csi_t) & alive
        return {
            "SAI": sai,
            "SSI": ssi_any & ~csi,          # superclass-level, not class-specific (primary)
            "SSI_pipe": ssi_any,            # pipeline Ab definition (sensitivity)
            "CSI": csi & ~sai,
        }


def encode(sae: Dict[str, torch.Tensor], X: np.ndarray, k: int = 32) -> np.ndarray:
    with torch.no_grad():
        h = torch.from_numpy(X).float() @ sae["encoder.weight"].T + sae["encoder.bias"]
        v, i = h.topk(k, dim=-1)
        out = torch.zeros_like(h).scatter_(-1, i, v)
    return out.numpy()


def selectivity(H: np.ndarray) -> Dict[str, np.ndarray]:
    """Vectorised reproduction of saeanalysis._phase4_ssi_csi (class-major H, 100/class)."""
    eps = 1e-10
    M = np.abs(H).reshape(100, 100, -1).mean(1).T  # (F, 100)
    tot = M.sum(1)
    S = np.stack([M[:, COARSE_TO_FINE[c]].sum(1) for c in range(20)], 1)  # (F,20)
    frac = S / (tot[:, None] + eps)
    best_sc = frac.argmax(1)
    ssi = np.maximum((frac.max(1) - 1 / 20) / (1 - 1 / 20 + eps), 0)
    p = M / (tot[:, None] + eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.where(p > 0, p * np.log(p + eps), 0).sum(1)
    sai = ent / np.log(100)
    cols = np.array([COARSE_TO_FINE[c] for c in best_sc])  # (F,5)
    W = np.take_along_axis(M, cols, 1)
    wt = W.sum(1)
    csi = np.maximum((W.max(1) / (wt + eps) - 0.2) / (0.8 + eps), 0)
    active = tot > eps
    return {"ssi": np.where(active, ssi, 0), "csi": np.where(active, csi, 0), "sai": np.where(active, sai, 0),
            "best_sc": best_sc, "best_class": M.argmax(1), "M": M}


# layer4 of torchvision ResNet-18 on a 2x2 layer3 map (32x32 input, ImageNet stem)
# is exactly a small MLP; BN folded (eval mode).

def _fold(sd, conv, bn):
    W = sd[f"{conv}.weight"].double()
    g, b, mu, var = (sd[f"{bn}.{k}"].double() for k in ("weight", "bias", "running_mean", "running_var"))
    s = g / torch.sqrt(var + 1e-5)
    return W * s[:, None, None, None], b - mu * s


class Layer4:
    """Exact layer4 for a (N,256,2,2) input -> (N,512) (== avgpool, 1x1 map)."""

    def __init__(self, sd, dtype=torch.float32, device="cpu"):
        W, b = _fold(sd, "layer4.0.conv1", "layer4.0.bn1")
        # 3x3 stride2 pad1 conv -> output (0,0) sees input (i,j) via kernel (i+1, j+1)
        self.W1 = torch.cat([W[:, :, i + 1, j + 1] for i in range(2) for j in range(2)], 1)  # (512, 4*256) pos-major
        self.b1 = b
        W, self.b2 = _fold(sd, "layer4.0.conv2", "layer4.0.bn2"); self.W2 = W[:, :, 1, 1]
        W, self.bd = _fold(sd, "layer4.0.downsample.0", "layer4.0.downsample.1"); self.Wd = W[:, :, 0, 0]
        W, self.b3 = _fold(sd, "layer4.1.conv1", "layer4.1.bn1"); self.W3 = W[:, :, 1, 1]
        W, self.b4 = _fold(sd, "layer4.1.conv2", "layer4.1.bn2"); self.W4 = W[:, :, 1, 1]
        for k in ("W1", "b1", "W2", "b2", "Wd", "bd", "W3", "b3", "W4", "b4"):
            setattr(self, k, getattr(self, k).to(dtype=dtype, device=device))

    def from_pre(self, z1, zd):
        """z1 = W1 x + b1 (pre-ReLU), zd = Wd x00 + bd."""
        h = F.relu(z1) @ self.W2.T + self.b2
        y = F.relu(h + zd)
        h = F.relu(y @ self.W3.T + self.b3) @ self.W4.T + self.b4
        return F.relu(h + y)

    def pre(self, Xflat):
        """Xflat: (N, 4*256) positions-major [(0,0),(0,1),(1,0),(1,1)] x channels."""
        return Xflat @ self.W1.T + self.b1, Xflat[:, :256] @ self.Wd.T + self.bd

    def __call__(self, Xflat):
        return self.from_pre(*self.pre(Xflat))


def layer3_spatial(lane: LaneData, m: int, device="cpu") -> np.ndarray:
    """(10000, 4, 256) layer3 output (positions-major), class-major order. Cached in CACHE."""
    cache = CACHE / lane.dir.name / f"l3sp_m{m}.npy"
    if cache.exists():
        return np.load(cache)
    import torchvision
    import torchvision.transforms as T
    from models import create_network
    net = create_network("resnet18", num_classes=100)
    net.load_state_dict(lane.checkpoint(m))
    net.eval().to(device)
    ds = torchvision.datasets.CIFAR100(root=str(EXP / "datasets" / "cifar100"), train=False, download=False,
                                       transform=T.Compose([T.ToTensor(), T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]))
    imgs = torch.stack([ds[i][0] for i in range(len(ds))])
    labels = np.array(ds.targets)
    order = np.argsort(labels, kind="stable")  # class-major, original order within class
    feats = []
    with torch.no_grad():
        for s in range(0, 10000, 1000):
            x = imgs[order[s:s + 1000]].to(device)
            x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
            x = net.layer3(net.layer2(net.layer1(x)))  # (B,256,2,2)
            feats.append(x.permute(0, 2, 3, 1).reshape(x.shape[0], 4, 256).cpu())
    out = torch.cat(feats).numpy().astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    return out


def boot_ci(x, n=2000, stat=np.median, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = stat(x[rng.integers(0, len(x), (n, len(x)))], axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def t_ci(x):
    """Mean and 95% t-CI across seeds."""
    from scipy import stats
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float(np.mean(x)) if len(x) else np.nan, np.nan, np.nan
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    h = se * stats.t.ppf(0.975, len(x) - 1)
    return float(m), float(m - h), float(m + h)
