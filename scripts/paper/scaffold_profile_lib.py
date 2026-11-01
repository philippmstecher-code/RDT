"""Scaffold profile: shared helpers.

Networks: static_noise ResNet-18 lanes only -- clean (standard_resnet18), between_sc_resnet18_throughout and
within_sc_resnet18_throughout, seeds 7, 42, 137, 256, 314; all real checkpoints (epochs 0, 5, ..., 30; duplicate
final dropped via sae_outcomes.checkpoint_info / real_ckpts).

Data outside this analysis, which the module does not open (every path it hands out passes `guard`): label-noise
experiment 0c28799c, curriculum, tg_suppression, critical.

CPU only (CUDA/MPS hidden), <= 3 torch threads. Read-only on all data. Writes only to
analysis_outputs/paper/SCAFFOLD_profile and the cache directory CACHE.
"""
from __future__ import annotations

import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "3"

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(3)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "packages" / "ml" / "src"))

from sae_outcomes import checkpoint_info, real_ckpts  # noqa: E402

OUT = REPO / "analysis_outputs" / "paper" / "SCAFFOLD_profile"
CACHE = Path(RDT_DATA + "/extended/cache_lineage")
RUNS_EXTENDED = Path(RDT_DATA + "/extended/experiments")
A4EN = REPO / "analysis_outputs" / "paper" / "SAE_outcomes" / "cache" / "EN"
TRANSFER = REPO / "analysis_outputs" / "paper" / "TRANSFER"

SEEDS = [7, 42, 137, 256, 314]
CONDS = {"clean": "standard_resnet18", "between": "between_sc_resnet18_throughout",
         "within": "within_sc_resnet18_throughout"}
LAYERS = ["layer1", "layer2", "layer3", "layer4", "avgpool"]
SAI_T = 0.9
ACT_EPS = 1e-6
N_CLS, N_PER = 100, 100
EXCLUDED = ("0c28799c", "curriculum", "tg_suppression", "critical")


def guard(p) -> None:
    s = str(p)
    for f in EXCLUDED:
        if f in s:
            raise RuntimeError(f"excluded path (not part of this analysis): {s}")


def lanes() -> List[Dict[str, Any]]:
    out = []
    for cond, nm in CONDS.items():
        for s in SEEDS:
            d = RUNS_EXTENDED / f"static_noise_s{s}" / "lanes" / nm
            guard(d)
            out.append({"key": f"{cond}__s{s}", "cond": cond, "seed": s, "dir": d})
    return out


def ckpts(d: Path) -> List[Dict[str, Any]]:
    guard(d)
    snaps = json.load(open(d / "dev_snapshots" / "metrics" / "devtrain_snapshots.json"))
    labels = [str(i) for i in sorted(int(s["milestone_index"]) for s in snaps)]
    return real_ckpts(checkpoint_info(d, labels))


def null_json(key: str) -> Dict[str, Any]:
    return json.load(open(A4EN / f"{key}_null.json"))


def sel_npz(key: str):
    return np.load(A4EN / f"{key}_sel.npz")


def thresholds(nl: Dict[str, Any], layer: str, kind: str) -> Dict[str, float]:
    """kind 'layer': per-layer alive-only null p95 (cache 'by_layer'); 'pooled': pooled p95 (all layers x ckpts)."""
    src = nl["by_layer"][layer] if kind == "layer" else nl["pooled"]
    return {"ssi": float(src["ssi"]["p95"]), "csi": float(src["csi"]["p95"]), "sai": SAI_T,
            "ssi_mean": float(src["ssi"]["mean"]), "ssi_std": float(src["ssi"]["std"]),
            "csi_mean": float(src["csi"]["mean"]), "csi_std": float(src["csi"]["std"])}


def roles(cnt, ssi, csi, sai, alive_min, thr) -> Dict[str, np.ndarray]:
    """Exclusive rule: tg = SAI > 0.9; cs = CSI > t_csi
    (NaN -> not class-selective); ss = SSI > t_ssi & ~cs & ~tg; scaffold = tg | ss."""
    alive = cnt >= alive_min
    csi_ok = np.nan_to_num(csi, nan=0.0)
    tg = alive & (sai > thr["sai"])
    cs = alive & (csi_ok > thr["csi"])
    ss = alive & (ssi > thr["ssi"]) & ~cs & ~tg
    ss_incl = alive & (ssi > thr["ssi"])
    return {"alive": alive, "tg": tg, "cs": cs, "ss": ss, "scaf": tg | ss, "ss_incl": ss_incl}


def load_raw_acts(d: Path, label: str) -> Dict[str, np.ndarray]:
    """Stored pooled activations (10,000 CIFAR-100 test images, 100 per class, class-major), float64."""
    guard(d)
    a = torch.load(d / "dev_snapshots" / f"milestone_{label}" / "individual_multilayer_activations.pt",
                   map_location="cpu", weights_only=False)
    out = {}
    for L in LAYERS:
        assert sorted(a[L].keys()) == list(range(N_CLS))
        out[L] = np.concatenate([a[L][c].float().numpy().reshape(len(a[L][c]), -1) for c in range(N_CLS)]).astype(np.float64)
        assert out[L].shape[0] == N_CLS * N_PER
    return out


def load_codes(d: Path, label: str, layer: str) -> np.ndarray:
    guard(d)
    return np.load(d / "sae_analysis" / "activations" / label / f"{layer}.npy")


def load_decoder(d: Path, label: str, layer: str) -> np.ndarray:
    """Decoder dictionary (d_in x n_features)."""
    guard(d)
    sd = torch.load(d / "sae_analysis" / "saes" / label / f"{layer}.pt", map_location="cpu")
    return sd["decoder.weight"].numpy().astype(np.float64)


Y = np.repeat(np.arange(N_CLS), N_PER)
