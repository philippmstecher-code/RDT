#!/usr/bin/env python3
"""Spread of the process fractions per superclass between architectures (heatmaps, values printed in every cell;
final transition excluded): the drawing and data of Extended Data Fig. 4, which routing/ed4.py saves.
Heatmaps in Blues; the process names keep their process colours.
a  population s.d. (pp) of the three architecture means for As-E and De-E (Blues, 0-4).
b  as a for Tg-E, Ab-E and Di-E.
Each architecture mean is the mean over its three seeds (42, 137, 256); n = 9 runs.
Source: analysis_outputs/paper/SI/superclass_process_fractions_{arch_means,per_run}.csv.

Usage: python scripts/paper/figures/architecture_spread.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nmi_style as ns  # noqa: E402
from checks_lib import RESULTS, MMCanvas  # noqa: E402

FIG = "supp3"
W_MM, H_MM = 180, 190
SRC = RESULTS / "SI"
COLS = {"As-E": "as_h", "De-E": "de_h", "Tg-E": "tg_h", "Ab-E": "ab_h", "Di-E": "di_h"}
DOMINANT, MINOR = ["As-E", "De-E"], ["Tg-E", "Ab-E", "Di-E"]
CELL_FS = 6.0
SPREAD_MAP = "Blues"
DARK_TXT = "#333333"
# Layout (mm from top-left)
LX, BX = 39, 109            # left edges of the left and right heatmaps
AW, BW = 50, 54             # heatmap widths
HM_H = 174                  # heatmap height (20 rows)
ROW1 = 9                    # top edge of the heatmap row
CB_GAP, CB_W = 2, 2.5


def label(sc):
    return sc.replace("_", " ")


def load():
    am = pd.read_csv(SRC / "superclass_process_fractions_arch_means.csv")
    pr = pd.read_csv(SRC / "superclass_process_fractions_per_run.csv")
    order = list(dict.fromkeys(am["superclass"]))
    mean = am.groupby("superclass")[list(COLS.values())].mean().loc[order]
    sd = am.groupby("superclass")[list(COLS.values())].std(ddof=0).loc[order]
    # consistency checks: architecture means = mean of their runs; fractions sum to 100 per run
    chk = pr.groupby(["arch", "superclass"])[list(COLS.values())].mean()
    diff = (chk - am.set_index(["arch", "superclass"])[list(COLS.values())]).abs().to_numpy().max()
    assert diff < 1e-9, diff
    assert np.allclose(pr[list(COLS.values())].sum(axis=1), 100), "fractions do not sum to 100"
    assert pr.groupby("arch")["seed"].nunique().eq(3).all() and len(pr) == 180
    mean.columns = sd.columns = list(COLS)
    return am, pr, order, mean, sd


def heatmap(ax, cax, M, procs, cmap, vmax, ticks, show_labels, rows):
    im = ax.imshow(M[procs].to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    cm = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    for i in range(M.shape[0]):
        for j, p in enumerate(procs):
            v = M[p].iloc[i]
            r, g, b = to_rgb(cm(v / vmax))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=CELL_FS,
                    color="white" if lum < 0.62 else DARK_TXT)
    ax.set_xticks(range(len(procs)))
    ax.set_xticklabels(procs)
    for t, p in zip(ax.get_xticklabels(), procs):
        t.set_color(ns.PROC[p])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label(r) for r in rows] if show_labels else [])
    ax.tick_params(axis="y", labelsize=6.5, length=2)
    ax.tick_params(axis="x", labelsize=6.5, length=2)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
        s.set_color("#9CA3AF")
    cb = plt.colorbar(im, cax=cax, ticks=ticks)
    cb.outline.set_linewidth(0.5)
    if vmax <= 4:
        cb.ax.set_yticklabels([f"{t:.1f}" for t in ticks])
    cb.ax.tick_params(labelsize=6.5, length=2, width=0.5)
    return im


def main():
    ns.apply()
    am, pr, order, mean, sd = load()
    fig = plt.figure(figsize=(W_MM * ns.MM, H_MM * ns.MM))
    C = MMCanvas(fig, W_MM, H_MM)
    specs = [  # letter, x, w, y, matrix, procs, cmap, vmax, ticks, title, labels
        ("a", LX, AW, ROW1, sd, DOMINANT, SPREAD_MAP, 4, [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], "s.d. across architecture means (pp)", True),
        ("b", BX, BW, ROW1, sd, MINOR, SPREAD_MAP, 4, [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], "s.d. across architecture means (pp)", False),
    ]
    for letter, x, w, y, M, procs, cmap, vmax, ticks, title, lab in specs:
        ax = C.ax(x, y, w, HM_H)
        cax = C.ax(x + w + CB_GAP, y, CB_W, HM_H)
        heatmap(ax, cax, M, procs, cmap, vmax, ticks, lab, order)
        ax.set_title(title, fontsize=7, pad=4)
        lx = 2 if lab else x - 9
        fig.text(lx / W_MM, 1 - (y - 6.5) / H_MM, letter, fontsize=8, fontweight="bold", va="top", ha="left")
        src = M[procs].round(4).reset_index()
        src.insert(1, "superclass_label", src["superclass"].map(label))
        src.columns = ["superclass", "superclass_label"] + [
            f"{p} {'mean fraction (%)' if M is mean else 's.d. of architecture means (pp)'}" for p in procs]
        ns.source_data(src, FIG, letter)

    ns.source_data(am.rename(columns={v: f"{k} (%)" for k, v in COLS.items()}), FIG, "arch_means")
    ns.source_data(pr.rename(columns={v: f"{k} (%)" for k, v in COLS.items()}), FIG, "per_run")
    out = ns.save_final(fig, FIG, W_MM, ed=False)

    print(out)
    print("mean fraction range per process (%):")
    for p in COLS:
        print(f"  {p}: {mean[p].min():.2f}-{mean[p].max():.2f}")
    print("population s.d. of architecture means (pp):")
    for p in COLS:
        print(f"  {p}: {sd[p].min():.2f}-{sd[p].max():.2f}, mean {sd[p].mean():.2f}")
    print(f"  overall max {sd.to_numpy().max():.3f} ({sd.max().idxmax()}), overall mean {sd.to_numpy().mean():.3f}")
    print(f"  sample s.d. (ddof=1) max {am.groupby('superclass')[list(COLS.values())].std(ddof=1).to_numpy().max():.3f}")


if __name__ == "__main__":
    main()
