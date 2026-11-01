#!/usr/bin/env python3
"""Process composition (Fig. 1b), construction/refinement ratio (Fig. 1c) and initial vs final C/R per architecture
(Fig. 2a) from the stored per-transition process counts of the core runs.

The last stored transition ``k->terminal`` is dropped everywhere: the last milestone checkpoint and ``terminal`` are
the same network, so that transition only measures SAE re-training noise (see NOISE_null).

  * source: <lane>/sae_analysis/sae_results.json['process_intensity'] (tg_h, ab_h, di_h)
  * C/R = (tg_h + ab_h) / max(di_h, 1)   (1-event floor when di_h = 0)
  * Fig. 1c clips C/R at the 1e3 axis ceiling
  * Fig. 1b: share of tg+ab+di per run, mean over the 9 CIFAR-100 runs
  * Fig. 2a: mean over seeds per architecture

Transitions used: Figs 1b/1c show transitions 0-7 (0->1 ... 7->8), the transitions that exist
as real (non-duplicate) transitions in all 9 runs (ViT-Small has 8 real transitions; ResNet-18 10,
CCT-7 9).  Fig. 2a uses each run's own first (0->1) and last real transition.

Data drive is read only.  Outputs: analysis_outputs/paper/figures/{fig1bc,fig1b,fig1c,fig2a}.*
and fig1bc_fig2a_values.csv. The loaders are reused by core_runs_lib.py.
Usage: python scripts/paper/figures/process_composition.py
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nmi_style  # noqa: E402
from nmi_style import ARCH, ARCH_LS, FILL_ALPHA, NEUTRAL, PROC, TINY  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

EXP = Path(RDT_DATA + "/core/experiments")
CORE_EXP = "abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb"
TIN_EXP = "821eaadb-0160-48f8-9669-ec3d9185c428"

CORE = [  # (arch, seed, lane-id prefix)
    ("ResNet-18", 42, "7db28a9a"), ("ResNet-18", 137, "a0bcbcba"), ("ResNet-18", 256, "935c13ae"),
    ("ViT-Small", 42, "10d31151"), ("ViT-Small", 137, "909d7f68"), ("ViT-Small", 256, "abad8975"),
    ("CCT-7", 42, "4bebc329"), ("CCT-7", 137, "edf8275e"), ("CCT-7", 256, "445dd4d4"),
]
TINY_LANE = (TIN_EXP, "45476b66")  # ResNet18-200class-10ms-seed42
ARCHS = ["ResNet-18", "ViT-Small", "CCT-7"]
CR_CEIL = 1e3          # Fig. 1c display ceiling
GREY_LEG = "#9CA3AF"   # legend swatches of Fig. 2a
SMALL = 5.5            # legend / phase-label size
LW = 1.0

# Reference values (final-transition C/R per run) that the loader must reproduce
REF_LAST = {("CCT-7", 137): 1.566, ("CCT-7", 256): 0.793, ("CCT-7", 42): 0.818,
            ("ResNet-18", 137): 1.960, ("ResNet-18", 256): 0.245, ("ResNet-18", 42): 1.446,
            ("ViT-Small", 137): 0.525, ("ViT-Small", 256): 0.535, ("ViT-Small", 42): 1.765}


def load_pi(exp: str, prefix: str):
    """Stored process intensity of one lane with the duplicate k->terminal transition removed."""
    lane = glob.glob(str(EXP / exp / "lanes" / f"{prefix}*"))
    assert len(lane) == 1, (exp, prefix, lane)
    pi = json.load(open(Path(lane[0]) / "sae_analysis" / "sae_results.json"))["process_intensity"]
    assert pi[-1]["transition"].endswith("->terminal"), pi[-1]["transition"]
    real = pi[:-1]
    assert all("terminal" not in t["transition"] for t in real)
    return real


def cr(t) -> float:
    return (t["tg_h"] + t["ab_h"]) / max(t["di_h"], 1)


def load_all():
    runs = []
    for arch, seed, pre in CORE:
        pi = load_pi(CORE_EXP, pre)
        runs.append({"arch": arch, "seed": seed, "pi": pi})
        got = cr(pi[-1])
        assert abs(got - REF_LAST[(arch, seed)]) < 1e-3, (arch, seed, got)
    tiny = load_pi(*TINY_LANE)
    assert abs(cr(tiny[0]) - 34.807) < 1e-3 and abs(cr(tiny[-1]) - 2.903) < 1e-3
    return runs, tiny


# panels

def draw_1b(ax, runs, n_tr):
    x = np.arange(n_tr)
    shares = []
    for r in runs:
        s = np.array([[t["tg_h"], t["ab_h"], t["di_h"]] for t in r["pi"][:n_tr]], float)
        shares.append(100 * s / s.sum(1, keepdims=True))
    m = np.mean(shares, 0)  # (n_tr, 3) mean over runs
    top = np.full(n_tr, 100.0)

    # phase shading (boundaries at transitions 2 and 5)
    xmax = n_tr - 0.5
    phases = [(-0.5, 2, PROC["Tg-E"], "Phase I\nTask-general"),
              (2, 5, PROC["Ab-E"], "Phase II\nAbstraction"),
              (5, xmax, PROC["Di-E"], "Phase III\nDifferentiation")]
    for x0, x1, c, lab in phases:
        ax.axvspan(x0, x1, color=c, alpha=0.06, lw=0, zorder=0)
        ax.text((x0 + x1) / 2, 110, lab, ha="center", va="center", fontsize=SMALL, color="#111111",
                linespacing=1.05)
    for xb, c in ((2, PROC["Tg-E"]), (5, PROC["Ab-E"])):
        ax.axvline(xb, color=c, alpha=0.12, lw=0.8, zorder=0)

    c1, c2 = m[:, 0], m[:, 0] + m[:, 1]
    for lo, hi, k in ((np.zeros(n_tr), c1, "Tg-E"), (c1, c2, "Ab-E"), (c2, top, "Di-E")):
        ax.fill_between(x, lo, hi, facecolor=PROC[k], edgecolor=PROC[k], alpha=FILL_ALPHA, lw=0.8,
                        label=k, zorder=2)
    ax.plot(x, c1, color="white", lw=0.7, zorder=3)
    ax.plot(x, c2, color="white", lw=0.7, zorder=3)

    ax.set_xlim(-0.5, xmax)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xticks(np.arange(0, n_tr, 2))
    ax.set_xlabel("Transition")
    ax.set_ylabel("Process composition (%)")
    h = [Patch(facecolor=PROC[k], edgecolor=PROC[k], alpha=FILL_ALPHA, label=k) for k in ("Tg-E", "Ab-E", "Di-E")]
    leg = ax.legend(handles=h, loc="center right", bbox_to_anchor=(1.0, 0.70), frameon=True, fontsize=SMALL,
                    framealpha=0.8, edgecolor="none", facecolor="white", handlelength=1.6,
                    borderpad=0.4, labelspacing=0.35)
    leg.get_frame().set_linewidth(0)
    return m


def draw_1c(ax, runs, n_tr):
    x = np.arange(n_tr)
    C = np.array([[min(cr(t), CR_CEIL) for t in r["pi"][:n_tr]] for r in runs])  # (9, n_tr)
    med = np.median(C, 0)
    q1, q3 = np.percentile(C, 25, 0), np.percentile(C, 75, 0)
    ax.fill_between(x, q1, q3, facecolor=NEUTRAL, alpha=0.12, edgecolor=NEUTRAL, lw=1.0, zorder=1)
    ax.plot(x, med, color="#374151", lw=LW + 0.1, zorder=4)
    am = {}
    for a in ARCHS:
        am[a] = C[[i for i, r in enumerate(runs) if r["arch"] == a]].mean(0)
        ax.plot(x, am[a], color=ARCH[a], ls=ARCH_LS[a], lw=LW, alpha=0.8, zorder=3)
    ax.axhline(1, color="#999999", ls="--", lw=0.5, zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(0.3, 2e3)
    ax.set_xlim(-0.5, n_tr - 0.5)
    ax.set_xticks(np.arange(0, n_tr, 2))
    ax.set_xlabel("Transition")
    ax.set_ylabel("C/R ratio")
    n = len(runs)
    h = [Patch(facecolor=NEUTRAL, alpha=0.2, edgecolor=NEUTRAL, lw=0.8, label=f"IQR (n = {n})"),
         Line2D([], [], color="#374151", lw=LW + 0.1, label=f"Median (n = {n})")]
    h += [Line2D([], [], color=ARCH[a], ls=ARCH_LS[a], lw=LW, alpha=0.8, label=a) for a in ARCHS]
    h += [Line2D([], [], color="#999999", ls="--", lw=0.5, label="Ratio = 1")]
    ax.legend(handles=h, loc="upper right", handlelength=2.2, labelspacing=0.35, fontsize=SMALL)
    return C, med, q1, q3, am


def draw_2a(ax, runs, tiny):
    groups = [(a, a) for a in ARCHS] + [("ResNet-18\n(TinyImageNet)", "tiny")]
    vals = []
    for lab, key in groups:
        if key == "tiny":
            ini, fin, col = cr(tiny[0]), cr(tiny[-1]), TINY
        else:
            rr = [r for r in runs if r["arch"] == key]
            ini = float(np.mean([cr(r["pi"][0]) for r in rr]))
            fin = float(np.mean([cr(r["pi"][-1]) for r in rr]))
            col = ARCH[key]
        vals.append((lab, ini, fin, col))
    w = 0.3
    for i, (lab, ini, fin, col) in enumerate(vals):
        ax.bar(i - w / 2, ini, w, color=col, alpha=0.35, edgecolor=col, lw=0.8, zorder=2)
        ax.bar(i + w / 2, fin, w, color=col, alpha=0.85, lw=0, zorder=2)
    ax.axhline(1, color="#555555", ls="--", lw=0.5, alpha=0.8, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.38, 1e5)
    ax.set_xlim(-0.5, len(vals) - 0.5)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([v[0] for v in vals])
    ax.set_ylabel("(Tg-E + Ab-E) / Di-E ratio (log)")
    ax.set_title("Construction / refinement ratio")
    h = [Patch(facecolor=GREY_LEG, alpha=0.35, edgecolor=GREY_LEG, lw=0.8, label="Initial"),
         Patch(facecolor=GREY_LEG, alpha=0.85, lw=0, label="Final")]
    ax.legend(handles=h, loc="upper right", handlelength=2.2, fontsize=7)
    return vals


def main():
    nmi_style.apply()
    runs, tiny = load_all()
    n_tr = min(len(r["pi"]) for r in runs)  # 8 (ViT-Small) -> transitions 0..7
    trs = [t["transition"] for t in runs[0]["pi"][:n_tr]]
    print(f"n = {len(runs)} runs; transitions used in Fig 1b/c: {trs[0]} ... {trs[-1]} ({n_tr})")

    # combined Fig. 1b,c: b left, c right
    fig, (ax_b, ax_c) = plt.subplots(1, 2, figsize=(6.0, 2.25))
    fig.subplots_adjust(wspace=0.3)
    m = draw_1b(ax_b, runs, n_tr)
    C, med, q1, q3, am = draw_1c(ax_c, runs, n_tr)
    nmi_style.panel(ax_b, "b", x=-0.2, y=1.06)
    nmi_style.panel(ax_c, "c", x=-0.2, y=1.06)
    nmi_style.save(fig, "fig1bc")

    for name, fn, letter in (("fig1b", lambda ax: draw_1b(ax, runs, n_tr), "b"),
                             ("fig1c", lambda ax: draw_1c(ax, runs, n_tr), "c")):
        fig, ax = plt.subplots(figsize=(3.0, 2.25))
        fn(ax)
        nmi_style.panel(ax, letter, x=-0.2, y=1.06)
        nmi_style.save(fig, name)

    fig, ax = plt.subplots(figsize=(2.6, 2.15))
    vals = draw_2a(ax, runs, tiny)
    nmi_style.panel(ax, "a", x=-0.2, y=1.04)
    nmi_style.save(fig, "fig2a")

    # values shown
    out = nmi_style.OUT / "fig1bc_fig2a_values.csv"
    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["panel", "item", "transition", "value"])
        for i, t in enumerate(trs):
            for j, k in enumerate(("Tg-E", "Ab-E", "Di-E")):
                wr.writerow(["1b", f"mean share {k} (%)", t, f"{m[i, j]:.2f}"])
        for i, t in enumerate(trs):
            wr.writerow(["1c", "median C/R", t, f"{med[i]:.4g}"])
            wr.writerow(["1c", "IQR low", t, f"{q1[i]:.4g}"])
            wr.writerow(["1c", "IQR high", t, f"{q3[i]:.4g}"])
            for a in ARCHS:
                wr.writerow(["1c", f"mean C/R {a}", t, f"{am[a][i]:.4g}"])
            wr.writerow(["1c", "n runs C/R<1", t, int((C[:, i] < 1).sum())])
        for r in runs:
            wr.writerow(["2a", f"{r['arch']} s{r['seed']} initial", r["pi"][0]["transition"], f"{cr(r['pi'][0]):.4g}"])
            wr.writerow(["2a", f"{r['arch']} s{r['seed']} final", r["pi"][-1]["transition"], f"{cr(r['pi'][-1]):.4g}"])
        for lab, ini, fin, _ in vals:
            lab = lab.replace("\n", " ")
            wr.writerow(["2a", f"{lab} initial (bar)", "0->1", f"{ini:.4g}"])
            wr.writerow(["2a", f"{lab} final (bar)", "last real", f"{fin:.4g}"])
    print(open(out).read())


if __name__ == "__main__":
    main()
