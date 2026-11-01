"""Shared data loaders and layout helpers for Fig. 1, Fig. 2, Fig. 4 and the Extended Data and Supplementary
figures built on the core runs; every figure is drawn and saved by its own script through nmi_style.save_final.

Data (read only):
  * core CIFAR-100 lanes and the Tiny ImageNet lane: process_composition.CORE / TINY_LANE
    (<lane>/sae_analysis/sae_results.json)
  * ResNet-18 variants: 8x SAE expansion (578b5803/abfc645c; seed-256 network, compared with the
    4x SAE of seed 256, 935c13ae) and 200-epoch training (ce93d3f6/c645c39e, seed 42, = persist_200ep.RUNS['200ep'])
The duplicate final transition (last milestone -> 'terminal', the same network) and the 'terminal'
checkpoint are dropped everywhere, unless DROP_TERMINAL is set to False.
"""
from __future__ import annotations

import glob
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nmi_style as ns  # noqa: E402
import process_composition as f12  # noqa: E402  (loaders only; its main() is not run)

EXP = f12.EXP
CORE, CORE_EXP, TINY_LANE = f12.CORE, f12.CORE_EXP, f12.TINY_LANE
ED4_LANES = {  # ResNet-18, CIFAR-100
    "4x": (CORE_EXP, "7db28a9a"),          # seed 42: 4x expansion, 50 epochs (main run); reference for 200ep
    "4x_s256": (CORE_EXP, "935c13ae"),     # seed 256: 4x SAE of the network the 8x SAE decomposes
    # 8x SAE: its checkpoints are those of the seed-256 standard run (initial.pt byte-identical to 935c13ae,
    # same milestone epochs), so it is compared with the 4x SAE of seed 256, not seed 42
    "8x": ("578b5803-1f2f-49c4-99b8-bf7071f43917", "abfc645c"),
    "200ep": ("ce93d3f6-282b-4811-84f8-be6c589c0500", "c645c39e"),  # seed 42 (initial.pt = 7db28a9a)
}
DROP_TERMINAL = True
PROCS5 = ("as_h", "de_h", "ab_h", "di_h", "tg_h")
PROC5_LAB = {"as_h": "As-E", "de_h": "De-E", "ab_h": "Ab-E", "di_h": "Di-E", "tg_h": "Tg-E"}
MM = 1 / 25.4


def lane_dir(exp: str, prefix: str) -> Path:
    hits = glob.glob(str(EXP / exp / "lanes" / f"{prefix}*"))
    assert len(hits) == 1, (exp, prefix, hits)
    return Path(hits[0])


@lru_cache(maxsize=None)
def sae_results(exp: str, prefix: str) -> dict:
    return json.loads((lane_dir(exp, prefix) / "sae_analysis" / "sae_results.json").read_text())


def process_intensity(exp, prefix, drop=None):
    """Per-transition process counts; the duplicate k->terminal transition removed when drop."""
    drop = DROP_TERMINAL if drop is None else drop
    pi = sae_results(exp, prefix)["process_intensity"]
    assert pi[-1]["transition"].endswith("->terminal")
    return pi[:-1] if drop else pi


def selectivity_evolution(exp, prefix, drop=None):
    """Per-checkpoint mean SSI / CSI over the SAE layers; the 'terminal' checkpoint removed when drop."""
    drop = DROP_TERMINAL if drop is None else drop
    se = sae_results(exp, prefix)["selectivity_evolution"]
    assert se[-1]["checkpoint"] == "terminal"
    return se[:-1] if drop else se


def checkpoint_epochs(exp, prefix):
    f = lane_dir(exp, prefix) / "dev_snapshots" / "metrics" / "devtrain_snapshots.json"
    if not f.exists():
        return None
    return [int(s["epoch"]) for s in json.loads(f.read_text())]


def superclass_fractions(exp, prefix, drop=None):
    """Fig. 2b: per superclass, the share (%) of each of the five processes among all events,
    summed over the classes of the superclass (class_process_summary.per_transition) and over
    transitions; the duplicate final transition removed when drop. Validated: with drop=False the
    sums equal the stored superclass_summary.process_breakdown exactly."""
    drop = DROP_TERMINAL if drop is None else drop
    d = sae_results(exp, prefix)
    by_name = {c["class_name"]: c for c in d["class_process_summary"].values()}
    out = {}
    for sc, v in d["superclass_summary"].items():
        tot, tot_all = np.zeros(5), np.zeros(5)
        for cn in v["per_class_process"]:
            for t in by_name[cn]["per_transition"]:
                a = np.array([t[p] for p in PROCS5], float)
                tot_all += a
                if not (drop and t["transition"].endswith("->terminal")):
                    tot += a
        ref = np.array([v["process_breakdown"][p] for p in PROCS5], float)
        assert np.abs(ref - tot_all).max() == 0, (prefix, sc)
        out[sc] = 100 * tot / tot.sum()
    return out


# layout helpers
def add_axes_mm(fig, x, y, w, h, **kw):
    """Axes at (x, y) mm from the bottom-left corner of the canvas, size w x h mm."""
    W, H = fig.get_size_inches() / MM
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)


def panel_mm(fig, x, y_top, letter):
    """Bold panel letter with its top-left corner at (x, y_top) mm (canvas coordinates)."""
    W, H = fig.get_size_inches() / MM
    fig.text(x / W, y_top / H, letter, fontsize=8, fontweight="bold", va="top", ha="left")


def apply_style():
    """ns.apply() plus Arial for mathtext (log-axis tick labels, subscripts), so that no DejaVu glyphs
    end up in the PDF."""
    import matplotlib.pyplot as plt
    ns.apply()
    plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                         "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial"})


def plain_log_ticks(ax, ticks, axis="y"):
    """Plain-number tick labels on a log axis (0.1, 1, 10, 100, 1,000 ...) instead of mathtext exponents, whose
    superscripts would print below 6 pt; minor ticks kept, unlabelled."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    a.set_minor_formatter(NullFormatter())
