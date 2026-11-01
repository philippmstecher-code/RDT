"""Shared helpers for ed5.py, the Supplementary figure scripts and extended/ed1.py, extended/supp4.py.

Data loaders only read stored results (analysis_outputs/paper and the read-only experiment drive).
"""
import glob
import json
from pathlib import Path

import numpy as np

import nmi_style as ns
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "analysis_outputs" / "paper"
EXP = Path(RDT_DATA + "/core/experiments")
CORE_EXP = "abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb"
INDEP_EXP, INDEP_LANE = "independent_sae", "independent_sae_seed256"
ARCHS = ["ResNet-18", "ViT-Small", "CCT-7"]
CORE = [  # (arch, seed, lane-id prefix), the 9 standard CIFAR-100 runs
    ("ResNet-18", 42, "7db28a9a"), ("ResNet-18", 137, "a0bcbcba"), ("ResNet-18", 256, "935c13ae"),
    ("ViT-Small", 42, "10d31151"), ("ViT-Small", 137, "909d7f68"), ("ViT-Small", 256, "abad8975"),
    ("CCT-7", 42, "4bebc329"), ("CCT-7", 137, "edf8275e"), ("CCT-7", 256, "445dd4d4"),
]


def lane_dir(exp, prefix):
    hits = glob.glob(str(EXP / exp / "lanes" / f"{prefix}*"))
    assert len(hits) == 1, (exp, prefix, hits)
    return Path(hits[0])


def sae_results(exp, prefix):
    return json.loads((lane_dir(exp, prefix) / "sae_analysis" / "sae_results.json").read_text())


def cr(t):
    """Construction/refinement ratio (Tg-E + Ab-E) / Di-E with the 1-event floor of the pipeline."""
    return (t["tg_h"] + t["ab_h"]) / max(t["di_h"], 1)


def a2_lane_pi(lane_id):
    """Per-transition Tg/Ab/Di counts summed over layers, as recomputed in NOISE_null (source of headline_cr.csv)."""
    d = json.loads((RESULTS / "NOISE_null" / "per_lane" / f"{lane_id}.json").read_text())
    out = []
    for tr, layers in d["real"].items():
        if "terminal" in tr:
            continue
        out.append({"transition": tr, **{k: sum(v[k] for v in layers.values()) for k in ("tg_h", "ab_h", "di_h")}})
    return out


class MMCanvas:
    """Axes placed in mm from the top-left corner of a fixed-size canvas."""

    def __init__(self, fig, w_mm, h_mm):
        self.fig, self.w, self.h = fig, w_mm, h_mm

    def ax(self, x, y_top, w, h, **kw):
        return self.fig.add_axes([x / self.w, 1 - (y_top + h) / self.h, w / self.w, h / self.h], **kw)


def spearman(x, y):
    from scipy import stats
    return float(stats.spearmanr(x, y).statistic)


def arch_of(lane):
    for a in ARCHS:
        if a in lane:
            return a
    raise KeyError(lane)


def ols_line(x, y):
    b, a = np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)
    return a, b


TEXT_DARK = ns.REF_DARK
