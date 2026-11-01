"""Re-implementation of RDT Phases 4-6 on stored SAE codes, used for recounts and noise nulls.

Phase 4 (selectivity) and Phase 5 (activation correlation + Hungarian matching) are called directly from
``packages/ml/src/saeanalysis.py``. Phase 6 (per-sample process classification) is re-implemented in vectorised
form (``classify_counts``), because the original is a per-sample loop that writes ~30 MB of JSON per
(transition, layer). ``classify_counts_original`` runs the original Phase 6 in a temporary directory and is used
to check equality on selected transitions. Read-only on the data drive.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

REPO = Path(__file__).resolve().parents[2]
ML_SRC = REPO / "packages" / "ml" / "src"
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))

import saeanalysis as sa  # noqa: E402
from cifar100_hierarchy import get_superclass_groups  # noqa: E402

DATA = Path(RDT_DATA + "/core")
EXP = DATA / "experiments"
CORE_EXP = EXP / "abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb"
OUT = REPO / "analysis_outputs" / "paper" / "NOISE_null"

PROCS = ["tg_h", "ab_h", "di_h", "as_h", "de_h"]
COUNT_KEYS = PROCS + ["stable"]


# Lane metadata

def lane_names(db_copy: Path) -> Dict[str, str]:
    """lane_id -> lane name from a copy of rcx.db."""
    c = sqlite3.connect(str(db_copy))
    return {r[0]: r[1] for r in c.execute("select id, name from lanes")}


def experiment_dataset(db_copy: Path) -> Dict[str, str]:
    c = sqlite3.connect(str(db_copy))
    return {r[0]: r[1] for r in c.execute("select id, dataset_id from experiments")}


class Lane:
    """Stored RDT artefacts of one lane (read-only)."""

    def __init__(self, lane_dir: Path, dataset_id: str = "cifar100", load_results: bool = True):
        self.dir = Path(lane_dir)
        self.sae_dir = self.dir / "sae_analysis"
        self.dataset_id = dataset_id
        self.results: Dict[str, Any] = {}
        if load_results:
            with open(self.sae_dir / "sae_results.json") as f:
                self.results = json.load(f)
        meta = self.results.get("metadata", {})
        self.layers: List[str] = meta.get("layers", [])
        self.labels: List[str] = meta.get("checkpoint_labels", [])
        self.n_classes: int = int(meta.get("n_classes", 100))
        self.n_per_class: int = int(meta.get("n_samples_per_class", 100))
        self.selected_classes: List[str] = meta.get("selected_classes", [])
        self.ati: Dict[str, Any] = meta.get("adaptive_thresholds") or {}
        self.thresholds: Dict[str, float] = meta.get("analysis_thresholds") or dict(sa.DEFAULT_THRESHOLDS)
        build = sa._resolve_build_superclass_map(dataset_id)
        c2i = {n: i for i, n in enumerate(self.selected_classes)}
        self.superclass_map = build(self.selected_classes, c2i)
        self.superclass_groups = get_superclass_groups(self.superclass_map)

    # class boundaries: rows ordered by sorted class idx, n_per_class rows each
    def fake_layer_data(self, n_rows: int) -> Dict[int, torch.Tensor]:
        assert n_rows == self.n_classes * self.n_per_class, (n_rows, self.n_classes, self.n_per_class)
        return {ci: torch.empty((self.n_per_class, 1)) for ci in range(self.n_classes)}

    def act_path(self, label: str, layer: str) -> Path:
        return self.sae_dir / "activations" / label / f"{sa._safe_layer_filename(layer)}.npy"

    def load_codes(self, label: str, layer: str) -> np.ndarray:
        return np.load(self.act_path(label, layer))

    def stored_selectivity(self, label: str, layer: str) -> Dict[str, Any]:
        return self.results["selectivity"][label][layer]


# Phase 4 / 5 wrappers (original code)

def selectivity(lane: Lane, H: np.ndarray) -> Dict[str, Any]:
    """Phase 4 (original) on one code matrix."""
    key = ("X", "L")
    out = sa._phase4_ssi_csi({"X": {"L": H}}, {"X": {"L": lane.fake_layer_data(H.shape[0])}},
                             lane.superclass_map, lane.superclass_groups, ["X"], ["L"], None)
    e = out[key]
    return {k: e[k] for k in ("feature_ssi", "feature_csi", "feature_sai", "n_active_features")}


def match(H_a: np.ndarray, H_b: np.ndarray) -> Dict[str, Any]:
    """Phase 5 (original) between two code matrices on the same samples."""
    sa._apply_thresholds(None)  # MATCH_STABLE_THRESH=0.5, MATCH_DEATH_THRESH=0.2 (as stored runs)
    res = sa._phase5_feature_matching({"A": {"L": H_a}, "B": {"L": H_b}}, {}, {}, ["A", "B"], ["L"], None)
    return res["A->B"]["L"]


# Phase 6, vectorised exact re-implementation

def classify_counts(H_a: np.ndarray, H_b: np.ndarray, md: Dict[str, Any],
                    sel_a: Dict[str, Any], sel_b: Dict[str, Any],
                    thresholds: Dict[str, float], ati: Dict[str, Any]) -> Dict[str, int]:
    """Totals over samples of Phase-6 process counts (identical logic to
    saeanalysis._phase6_process_classification, summed over samples)."""
    eps_z = 1e-10
    ssi_t = thresholds["ssi_abstraction_thresh"]
    sai_t = thresholds["sai_task_general_thresh"]
    csi_t = thresholds["csi_differentiation_thresh"]
    z_delta_min = 0.5
    if ati:
        ssi_m, ssi_s = ati.get("ssi_null_mean", 0.0), max(ati.get("ssi_null_std", 1.0), eps_z)
        csi_m, csi_s = ati.get("csi_null_mean", 0.0), max(ati.get("csi_null_std", 1.0), eps_z)
        sai_m, sai_s = ati.get("sai_null_mean", 0.0), max(ati.get("sai_null_std", 1.0), eps_z)
    else:
        ssi_m = csi_m = sai_m = 0.0
        ssi_s = csi_s = sai_s = 1.0

    act_a = np.abs(H_a) > 1e-6
    act_b = np.abs(H_b) > 1e-6
    fa = {k: np.asarray(sel_a[k], dtype=np.float64) for k in ("feature_ssi", "feature_csi", "feature_sai")}
    fb = {k: np.asarray(sel_b[k], dtype=np.float64) for k in ("feature_ssi", "feature_csi", "feature_sai")}
    alive_a, alive_b = np.asarray(md["alive_a"]), np.asarray(md["alive_b"])
    out = {k: 0 for k in COUNT_KEYS}

    # stable
    if md["stable"]:
        ia = np.array([s[0] for s in md["stable"]], dtype=int)
        ib = np.array([s[1] for s in md["stable"]], dtype=int)
        keep = alive_a[ia]
        ia, ib = ia[keep], ib[keep]
        co = (act_a[:, ia] & act_b[:, ib]).sum(axis=0).astype(np.int64)  # samples per pair
        ssi_a, ssi_b = fa["feature_ssi"][ia], fb["feature_ssi"][ib]
        csi_a, csi_b = fa["feature_csi"][ia], fb["feature_csi"][ib]
        sai_a, sai_b = fa["feature_sai"][ia], fb["feature_sai"][ib]
        dz_ssi = (ssi_b - ssi_m) / ssi_s - (ssi_a - ssi_m) / ssi_s
        dz_csi = (csi_b - csi_m) / csi_s - (csi_a - csi_m) / csi_s
        dz_sai = (sai_b - sai_m) / sai_s - (sai_a - sai_m) / sai_s
        has_ab = (ssi_b > ssi_t) & (dz_ssi > z_delta_min)
        has_tg = (sai_b > sai_t) & (dz_sai > z_delta_min)
        has_di = (csi_b > csi_t) & (dz_csi > z_delta_min)
        out["stable"] += int(co.sum())
        out["tg_h"] += int(co[has_tg].sum())
        out["ab_h"] += int(co[has_ab & ~has_tg].sum())
        out["di_h"] += int(co[has_di].sum())

    # died (alive in A, active in sample)
    died = np.array(sorted(set(md["died"])), dtype=int)
    if died.size:
        died = died[alive_a[died]]
        out["de_h"] += int(act_a[:, died].sum())

    # born
    born = np.array(sorted(set(md["born"])), dtype=int)
    if born.size:
        born = born[alive_b[born]]
        cb = act_b[:, born].sum(axis=0).astype(np.int64)
        ssi_v, csi_v, sai_v = fb["feature_ssi"][born], fb["feature_csi"][born], fb["feature_sai"][born]
        has_tg = sai_v > sai_t
        has_ab = ssi_v > ssi_t
        out["as_h"] += int(cb.sum())
        out["tg_h"] += int(cb[has_tg].sum())
        out["ab_h"] += int(cb[has_ab & ~has_tg].sum())
        out["di_h"] += int(cb[csi_v > csi_t].sum())
    return out


def classify_counts_original(lane: Lane, H_a: np.ndarray, H_b: np.ndarray, md: Dict[str, Any],
                             sel_a: Dict[str, Any], sel_b: Dict[str, Any]) -> Dict[str, int]:
    """Original Phase 6 (slow; writes per-sample JSON into a temp dir that is deleted)."""
    fake = lane.fake_layer_data(H_a.shape[0])
    with tempfile.TemporaryDirectory() as td:
        ev = sa._phase6_process_classification(
            {"A": {"L": H_a}, "B": {"L": H_b}}, {"A": {"L": fake}, "B": {"L": fake}},
            {"A->B": {"L": md}}, {("A", "L"): sel_a, ("B", "L"): sel_b},
            ["A", "B"], ["L"], Path(td), None,
            thresholds=lane.thresholds, adaptive_threshold_info=lane.ati or None)
    tot = {k: 0 for k in COUNT_KEYS}
    for e in ev["A->B"]["L"]:
        for k in COUNT_KEYS:
            tot[k] += e["process_counts"].get(k, 0)
    return tot


def transition_counts(lane: Lane, H_a: np.ndarray, H_b: np.ndarray,
                      sel_a: Optional[Dict[str, Any]] = None,
                      sel_b: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """Phase 4+5+6 for one (pseudo-)transition; returns (counts, matching summary)."""
    if sel_a is None:
        sel_a = selectivity(lane, H_a)
    if sel_b is None:
        sel_b = selectivity(lane, H_b)
    md = match(H_a, H_b)
    c = classify_counts(H_a, H_b, md, sel_a, sel_b, lane.thresholds, lane.ati)
    n_alive_a = int(np.asarray(md["alive_a"]).sum())
    msum = {"n_stable": md["n_stable"], "n_born": md["n_born"], "n_died": md["n_died"],
            "n_transformed": md["n_transformed"], "n_alive_a": n_alive_a,
            "n_alive_b": int(np.asarray(md["alive_b"]).sum())}
    return c, msum


def census(lane: Lane, sel: Dict[str, Any], H: np.ndarray) -> Dict[str, float]:
    """Feature-composition state at one checkpoint: alive features (>=1% samples) that exceed
    the run's SAI/SSI/CSI thresholds (Tg/Ab exclusive as in Phase 6), plus activation-weighted
    versions (sum of active-sample counts)."""
    t = lane.thresholds
    n = H.shape[0]
    act = np.abs(H) > 1e-6
    cnt = act.sum(axis=0)
    alive = cnt >= max(1, n * 0.01)
    ssi = np.asarray(sel["feature_ssi"], dtype=np.float64)
    csi = np.asarray(sel["feature_csi"], dtype=np.float64)
    sai = np.asarray(sel["feature_sai"], dtype=np.float64)
    tg = alive & (sai > t["sai_task_general_thresh"])
    ab = alive & (ssi > t["ssi_abstraction_thresh"]) & ~tg
    di = alive & (csi > t["csi_differentiation_thresh"])
    w = cnt.astype(np.float64)
    return {
        "n_alive": int(alive.sum()), "n_tg": int(tg.sum()), "n_ab": int(ab.sum()), "n_di": int(di.sum()),
        "w_alive": float(w[alive].sum()), "w_tg": float(w[tg].sum()), "w_ab": float(w[ab].sum()),
        "w_di": float(w[di].sum()),
    }


def cr(d: Dict[str, float]) -> float:
    den = d["di_h"]
    return float("nan") if den == 0 else (d["tg_h"] + d["ab_h"]) / den


def _quiet_progress(*_a, **_k):
    return None


sa._progress = _quiet_progress  # silence saeanalysis progress prints
