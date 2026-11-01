"""Shared data paths, loaders and plotting helpers for the figure scripts.

All drawing goes through nmi_style (colours, fonts, panel letters, save). Recurring idioms:
  * contrast bars: bar = mean, grey jittered seed dots, dark 95%-CI error bar with caps, thin dashed zero line,
    value label "x.x pp";
  * trajectories: line + small markers, shaded band (SEM or CI), legend top-right.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

import nmi_style as ns
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "analysis_outputs" / "paper"
RUNS_EXTENDED = Path(RDT_DATA + "/extended/experiments")
RUNS_CORE = Path(RDT_DATA + "/core/experiments")
SEEDS = [7, 42, 137, 256, 314]

DOT = "#9CA3AF"     # seed dots
ERR = "#374151"     # error bars
REF = "#6B7280"     # dashed reference lines
BAND_ALPHA = 0.15   # shaded SEM/CI bands
WINDOW = "#F3F4F6"  # light grey window shading


# data
def lane_done(lane):
    return (Path(lane) / ".lane_done").exists()


def val_curve(lane, key="val_accuracy"):
    f = Path(lane) / "dev_snapshots" / "metrics" / "devtrain_metrics.json"
    if not f.exists():
        return None
    return {int(e["epoch"]): float(e[key]) for e in json.loads(f.read_text())}


def t0_acc(lane):
    f = Path(lane) / "dev_snapshots" / "metrics" / "devtrain_snapshots.json"
    if not f.exists():
        return None
    for s in json.loads(f.read_text()):
        if s.get("is_t0_baseline") or s.get("epoch") == 0:
            return float(s["val_accuracy"])
    return None


# stats
def mean_sem(a, axis=0):
    a = np.asarray(a, float)
    m = a.mean(axis)
    n = a.shape[axis]
    s = a.std(axis, ddof=1) / math.sqrt(n) if n > 1 else np.zeros_like(m)
    return m, s


def t_ci(x):
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    if n < 2:
        return m, m, m
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / math.sqrt(n)
    return m, m - h, m + h


def fmt_pp(v):
    return ("+" if v > 0 else "−" if v < 0 else "") + f"{abs(v):.1f} pp"


# drawing
def contrast_bars(ax, xs, groups, colors, width=0.55, label_vals=True, alphas=None, unit="pp", seed=0):
    """Contrast bars. groups: list of 1-D arrays of per-seed values. Returns (mean, lo, hi, n) per bar."""
    rng = np.random.default_rng(seed)
    out = []
    for i, (x, vals, c) in enumerate(zip(xs, groups, colors)):
        vals = np.asarray(vals, float)
        m, lo, hi = t_ci(vals)
        a = ns.FILL_ALPHA if alphas is None else alphas[i]
        ax.bar(x, m, width=width, color=c, alpha=a, lw=0, zorder=1)
        jit = rng.uniform(-0.22, 0.22, len(vals)) * width
        ax.scatter(x + jit, vals, s=9, color=DOT, lw=0, zorder=3)
        if len(vals) > 1:
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=ERR, elinewidth=0.9,
                        capsize=3, capthick=0.9, zorder=4)
        out.append((m, lo, hi, len(vals)))
    ax.axhline(0, color=REF, lw=0.6, ls="--", zorder=0)
    return out


def label_bar(ax, x, y, text, va="bottom", color=ERR, fs=6.3):
    ax.text(x, y, text, ha="center", va=va, fontsize=fs, color=color)


def traj(ax, x, Y, color, ls="-", marker="o", label=None, band="sem", ms=2.6, lw=1.1, markevery=None):
    """Line (mean) + shaded band over seeds. Y: (n_seeds, len(x)). band 'sem' or 'ci'."""
    Y = np.asarray(Y, float)
    m = np.nanmean(Y, 0)
    n = Y.shape[0]
    if n > 1:
        se = np.nanstd(Y, 0, ddof=1) / math.sqrt(n)
        h = se * (stats.t.ppf(0.975, n - 1) if band == "ci" else 1.0)
        ax.fill_between(x, m - h, m + h, color=color, alpha=BAND_ALPHA, lw=0)
    ax.plot(x, m, color=color, ls=ls, marker=marker, ms=ms, lw=lw, label=label, markevery=markevery)
    return m


def nlabel(ax, text, loc="lower left", x=None, y=None):
    pos = {"lower left": (0.02, 0.03, "left", "bottom"), "lower right": (0.98, 0.03, "right", "bottom"),
           "upper left": (0.02, 0.97, "left", "top"), "upper right": (0.98, 0.97, "right", "top")}[loc]
    ax.text(pos[0] if x is None else x, pos[1] if y is None else y, text, transform=ax.transAxes,
            ha=pos[2], va=pos[3], fontsize=6, color=REF)
