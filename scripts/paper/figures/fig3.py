#!/usr/bin/env python3
"""Fig. 3: feature turnover across training transitions. The stored ``k->terminal`` transition compares the last
checkpoint with itself and is dropped.

a  Churn rate per 1,000 weight updates: 100 x process_intensity[t].churn / delta weight updates x 1,000.
   ``churn`` = non-stable feature events / (non-stable + stable) in the transition (births, deaths,
   abstraction and differentiation events; saeanalysis.py).  Faint lines: the 9 runs; solid lines:
   architecture means (3 seeds); transitions 0->1 ... 7->8, which all 9 runs have.
b  Feature births and deaths per 1,000 weight updates (feature_matching n_born / n_died summed over the
   analysed layers), mean over the 9 runs; same transitions.
Shaded: early window (transitions 0-2).
The loader is checked against reference values for all non-terminal points.
Usage: python scripts/paper/figures/fig3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nmi_style as ns  # noqa: E402
from turnover_lib import ARCHS, load_core, per1k  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

NAME = "fig3"
BORN, DIED = "#0891B2", "#EF4444"
WIN_FILL, WIN_EDGE, WIN_TEXT = "#FEF3C7", "#FDE68A", "#92400E"
WINDOW = (-0.5, 2.5)
N_COMMON = 8                               # transitions 0..7 exist in all 9 runs

# reference values (non-terminal points) for the loader check
SUB_A = {"ResNet-18": [235, 82, 56, 40, 27, 19, 10, 9.5], "ViT-Small": [224, 89, 61, 53, 50, 41, 11, 5],
         "CCT-7": [204, 131, 97, 72, 75, 58, 39, 8]}
SUB_B_BORN = [531, 446, 347, 262, 289, 319, 223, 109]
SUB_B_DIED = [515, 55, 41, 64, 50, 56, 55, 49]


def churn_rate(t):
    return 100.0 * per1k(t["churn"], t["dt_updates"])


def shade(ax, ytop, label):
    ax.axvspan(*WINDOW, facecolor=WIN_FILL, edgecolor=WIN_EDGE, lw=0.6, alpha=0.6, zorder=0)
    if label:
        ax.text(np.mean(WINDOW), ytop, "Early window", color=WIN_TEXT, fontsize=6.5, ha="center", va="top")


def main():
    ns.apply()
    runs = load_core()

    # data
    rows_a, rows_b = [], []
    for r in runs:
        for i, t in enumerate(r["tr"][:N_COMMON]):
            rows_a.append({"arch": r["arch"], "seed": r["seed"], "transition_index": i, "transition": t["transition"],
                           "delta_weight_updates": t["dt_updates"], "churn_fraction": t["churn"],
                           "churn_pct_per_1k_updates": churn_rate(t)})
            rows_b.append({"arch": r["arch"], "seed": r["seed"], "transition_index": i, "transition": t["transition"],
                           "delta_weight_updates": t["dt_updates"], "born": t["born"], "died": t["died"],
                           "born_per_1k_updates": per1k(t["born"], t["dt_updates"]),
                           "died_per_1k_updates": per1k(t["died"], t["dt_updates"])})
    A = pd.DataFrame(rows_a)
    B = pd.DataFrame(rows_b)
    amean = {a: A[A.arch == a].groupby("transition_index")["churn_pct_per_1k_updates"].mean().values for a in ARCHS}
    Bc = B
    born = Bc.groupby("transition_index")["born_per_1k_updates"].mean().values
    died = Bc.groupby("transition_index")["died_per_1k_updates"].mean().values
    assert all(Bc.groupby("transition_index").size() == 9)
    assert all(A.groupby("transition_index").size() == 9) and A.transition_index.max() == N_COMMON - 1

    for a in ARCHS:
        n = min(len(SUB_A[a]), len(amean[a]))
        assert np.allclose(amean[a][:n], SUB_A[a][:n], atol=1.0), (a, amean[a][:n])
    assert np.allclose(born, SUB_B_BORN, atol=1.0) and np.allclose(died, SUB_B_DIED, atol=1.0)

    # figure
    fig = plt.figure(figsize=(180 * ns.MM, 68 * ns.MM))
    axa = fig.add_axes([0.075, 0.17, 0.39, 0.66])
    axb = fig.add_axes([0.585, 0.17, 0.39, 0.66])

    # a
    shade(axa, 268, True)
    for r in runs:
        y = [churn_rate(t) for t in r["tr"][:N_COMMON]]
        axa.plot(range(len(y)), y, color=ns.ARCH[r["arch"]], lw=0.6, alpha=0.18, zorder=2)
    for a in ARCHS:
        axa.plot(range(len(amean[a])), amean[a], color=ns.ARCH[a], lw=1.3, zorder=3, solid_capstyle="round")
    axa.set_xlim(-0.75, N_COMMON - 0.25)
    axa.set_ylim(-8, 272)
    axa.set_xticks(range(N_COMMON))
    axa.set_yticks([0, 50, 100, 150, 200, 250])
    axa.set_xlabel("Training transition")
    axa.set_ylabel("Churn rate (% per 1k updates)")
    h = [Line2D([], [], color=ns.ARCH[a], lw=1.3, label=a) for a in ARCHS]
    axa.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=3, frameon=False,
               handlelength=2.0, columnspacing=1.6, fontsize=6.5)

    # b
    shade(axb, 1100, False)
    x = np.arange(N_COMMON)
    axb.bar(x, born, 0.8, color=BORN, alpha=0.8, lw=0, zorder=2)
    axb.bar(x, died, 0.8, bottom=born, color=DIED, alpha=0.8, lw=0, zorder=2)
    axb.set_xlim(-0.75, N_COMMON - 0.25)
    axb.set_ylim(0, 1100)
    axb.set_xticks(x)
    axb.set_yticks([0, 200, 400, 600, 800, 1000])
    axb.set_xlabel("Training transition")
    axb.set_ylabel("Feature events per 1k updates")
    h = [Patch(facecolor=BORN, alpha=0.8, label="Born"), Patch(facecolor=DIED, alpha=0.8, label="Died")]
    axb.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=2, frameon=False,
               handlelength=2.0, columnspacing=1.6, fontsize=6.5)

    ns.panel(axa, "a", x=-0.16, y=1.04)
    ns.panel(axb, "b", x=-0.16, y=1.04)
    for _ax in (axa, axb):
        ns.rdt_tag(_ax)

    # source data and numbers
    ns.source_data(A, NAME, "a")
    summ_a = [{"arch": a, "transition_index": i, "mean_churn_pct_per_1k_updates": v}
              for a in ARCHS for i, v in enumerate(amean[a])]
    ns.source_data(pd.DataFrame(summ_a), NAME, "a_arch_means")
    ns.source_data(B, NAME, "b")
    ns.source_data(pd.DataFrame({"transition_index": x, "born_per_1k_mean9": born, "died_per_1k_mean9": died,
                                 "total_per_1k_mean9": born + died}), NAME, "b_means")
    out = ns.save_final(fig, NAME, 180)

    # numbers quoted in the text
    rn = A[(A.arch == "ResNet-18")]
    t0 = rn[rn.transition_index == 0].set_index("seed")["churn_fraction"]
    tot = born + died
    print(f"saved {out}")
    print("ResNet-18 churn at 0->1 (fraction):", t0.round(4).to_dict(), "mean", round(t0.mean(), 4))
    print("panel b totals:", np.round(tot, 0), "ratio t0/max(later):", round(tot[0] / tot[1:].max(), 2))
    for a in ARCHS:
        print(a, "mean churn:", np.round(amean[a], 1), "t0/max(later):", round(amean[a][0] / amean[a][1:].max(), 2))
    print("births share of events at t0:", round(born[0] / tot[0], 3))


if __name__ == "__main__":
    main()
