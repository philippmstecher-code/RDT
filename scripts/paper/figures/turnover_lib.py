"""Shared loaders for Fig. 3 (fig3.py), seed_events.py and granger_stats.py.

Rule (as in process_composition.py): in the main runs the last milestone checkpoint and the
``terminal`` state are the same network, so the stored ``k->terminal`` transition (and the ``terminal``
checkpoint column) is dropped everywhere.

Sources (data drive, read only):
  <lane>/sae_analysis/sae_results.json
      process_intensity[]   per-transition event counts (tg_h, ab_h, di_h, ...) and ``churn``
      feature_matching      per-transition, per-layer n_stable / n_born / n_died / n_transformed
      feature_landscape     per-checkpoint, per-layer mean_ssi / mean_csi
  <lane>/dev_snapshots/metrics/devtrain_snapshots.json
      weight_updates per milestone checkpoint (the per-1,000-update normalisation)
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

EXP = Path(RDT_DATA + "/core/experiments")
CORE_EXP = "abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb"
CORE = [  # (arch, seed, lane-id prefix)
    ("ResNet-18", 42, "7db28a9a"), ("ResNet-18", 137, "a0bcbcba"), ("ResNet-18", 256, "935c13ae"),
    ("ViT-Small", 42, "10d31151"), ("ViT-Small", 137, "909d7f68"), ("ViT-Small", 256, "abad8975"),
    ("CCT-7", 42, "4bebc329"), ("CCT-7", 137, "edf8275e"), ("CCT-7", 256, "445dd4d4"),
]
ARCHS = ["ResNet-18", "ViT-Small", "CCT-7"]
SEEDS = [42, 137, 256]
REPO = Path(__file__).resolve().parents[3]
STORE = REPO / "analysis_outputs/paper/SUPERCLASS_series"


def lane_dir(prefix: str, exp: str = CORE_EXP) -> Path:
    hits = glob.glob(str(EXP / exp / "lanes" / f"{prefix}*"))
    assert len(hits) == 1, (exp, prefix, hits)
    return Path(hits[0])


def load_run(arch: str, seed: int, prefix: str) -> dict:
    """One core run with the duplicated final transition and the terminal checkpoint removed."""
    lane = lane_dir(prefix)
    d = json.load(open(lane / "sae_analysis" / "sae_results.json"))
    snaps = json.load(open(lane / "dev_snapshots" / "metrics" / "devtrain_snapshots.json"))
    wu = {str(s["milestone_index"]): int(s["weight_updates"]) for s in snaps}
    pi = d["process_intensity"]
    assert pi[-1]["transition"].endswith("->terminal"), pi[-1]["transition"]
    pi_real = pi[:-1]
    assert all("terminal" not in t["transition"] for t in pi_real)
    labels = [c for c in d["metadata"]["checkpoint_labels"] if c != "terminal"]
    assert labels == [str(i) for i in range(len(labels))] and len(labels) == len(pi_real) + 1
    fm_keys = [k for k in d["feature_matching"] if "terminal" not in k]
    assert fm_keys == [t["transition"] for t in pi_real]
    trs = []
    for t in pi_real:
        a, b = t["transition"].split("->")
        dt = wu[b] - wu[a]
        assert dt > 0
        fm = d["feature_matching"][t["transition"]]
        trs.append({
            "transition": t["transition"], "dt_updates": dt,
            "tg_h": t["tg_h"], "ab_h": t["ab_h"], "di_h": t["di_h"], "churn": t["churn"],
            "born": sum(v["n_born"] for v in fm.values()), "died": sum(v["n_died"] for v in fm.values()),
            "fm": fm,
        })
    land = {c: d["feature_landscape"][c] for c in labels}
    return {"arch": arch, "seed": seed, "layers": d["metadata"]["layers"], "labels": labels,
            "weight_updates": [wu[c] for c in labels], "tr": trs, "landscape": land,
            "dropped": pi[-1]["transition"]}


def load_core() -> list[dict]:
    return [load_run(a, s, p) for a, s, p in CORE]


def per1k(count: float, dt: int) -> float:
    return count / dt * 1000.0
