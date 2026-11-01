#!/usr/bin/env python3
"""Extended Data Fig. 3: the process ordering in every run of the three architectures.

a  one small multiple per architecture (ResNet-18, ViT-Small, CCT-7): event rate per training transition (events per
   1,000 weight updates) scaled to each process's peak in each run; Tg-E purple, Ab-E blue, Di-E amber (ns.PROC) with
   markers triangle / circle / square; seeds 42 / 137 / 256 as solid / dashed / dotted.
b  transition of the peak rate of each process in each of the 9 runs; 'in order or tied' = Tg <= Ab <= Di.
Data: figures/final/source_data_ed/seed_events.csv (stored per-transition event counts), read-only.
Usage: python scripts/paper/figures/extended/ed3.py  (ES_FIG_OUT=<dir> redirects the output)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import nmi_style as ns  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

NAME = "ed3"
FINAL_SD = ns.OUT / "final" / "source_data_ed"
ARCHS = ["ResNet-18", "ViT-Small", "CCT-7"]
SEEDS = [42, 137, 256]
PROCS = ["Tg-E", "Ab-E", "Di-E"]
SEED_LS = {42: "-", 137: (0, (4, 1.6)), 256: (0, (1, 1.3))}
SEED_LAB = {42: "seed 42", 137: "seed 137", 256: "seed 256"}
OFFSET = {"Tg-E": -0.22, "Ab-E": 0.0, "Di-E": 0.22}
# purple and blue collapse for deuteranopes, so each process also has its own marker shape in a and b
MARK = {"Tg-E": "^", "Ab-E": "o", "Di-E": "s"}
MS_A = {"Tg-E": 2.6, "Ab-E": 2.3, "Di-E": 2.1}       # a: per-point markers, sizes matched for area
MS_B = {"Tg-E": 4.6, "Ab-E": 4.0, "Di-E": 3.6}       # b: peak markers


def run_label(arch, seed):
    lab = f"{arch} s{seed}"
    return lab + " (224 px)" if (arch, seed) == ("ViT-Small", 256) else lab


def main():
    ns.apply()
    df = pd.read_csv(FINAL_SD / "seed_events.csv")
    assert len(df) == 81 and set(df.arch) == set(ARCHS)
    for p in PROCS:
        assert np.allclose(df[f"{p}_per_1k_updates"], df[f"{p}_events"] / df.delta_weight_updates * 1000)
        df[f"{p}_scaled_to_run_peak"] = df[f"{p}_per_1k_updates"] / df.groupby(["arch", "seed"])[
            f"{p}_per_1k_updates"].transform("max")

    peaks = []
    for a in ARCHS:
        for s in SEEDS:
            g = df[(df.arch == a) & (df.seed == s)]
            pk = {p: int(g.loc[g[f"{p}_per_1k_updates"].idxmax(), "transition_index"]) for p in PROCS}
            peaks.append({"arch": a, "seed": s, "run": run_label(a, s), "n_transitions": len(g),
                          **{f"peak_{p}": pk[p] for p in PROCS},
                          "in_order_or_tied": pk["Tg-E"] <= pk["Ab-E"] <= pk["Di-E"],
                          "strictly_in_order": pk["Tg-E"] < pk["Ab-E"] < pk["Di-E"]})
    peaks = pd.DataFrame(peaks)
    n_ok, n_runs = int(peaks.in_order_or_tied.sum()), len(peaks)
    print(peaks.to_string())

    fig = plt.figure(figsize=(180 * ns.MM, 170 * ns.MM))
    # a: small multiples
    L, R, W, GAP = 0.075, 0.985, None, 0.035
    W = (R - L - 2 * GAP) / 3
    ya0, ya1 = 0.60, 0.935
    fig.text(0.008, 0.975, "a", fontsize=8, fontweight="bold", va="center")
    fig.text(L, 0.975, "Task-general, then superclass, then class events peak in every architecture",
             fontsize=7, va="center")
    for i, a in enumerate(ARCHS):
        ax = fig.add_axes([L + i * (W + GAP), ya0, W, ya1 - ya0])
        for p in PROCS:
            for s in SEEDS:
                g = df[(df.arch == a) & (df.seed == s)].sort_values("transition_index")
                ax.plot(g.transition_index, g[f"{p}_scaled_to_run_peak"], color=ns.PROC[p], ls=SEED_LS[s],
                        lw=1.0, alpha=0.9, dash_capstyle="butt", marker=MARK[p], ms=MS_A[p], mew=0)
        ax.set_xlim(-0.3, 9.3)
        ax.set_xticks(range(0, 10, 2))
        ax.set_ylim(-0.03, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.set_title(a, fontsize=7, pad=3)
        ax.set_xlabel("Training transition")
        if i == 0:
            ax.set_ylabel("Event rate\n(fraction of run peak)")
        else:
            ax.set_yticklabels([])

    # b: peak transition per run
    fig.text(0.008, 0.535, "b", fontsize=8, fontweight="bold", va="center")
    xb0, xb1 = 0.205, 0.665
    fig.text(L, 0.535, "Peak transitions follow the order task-general, superclass, class", fontsize=7,
             va="center")
    ax = fig.add_axes([xb0, 0.06, xb1 - xb0, 0.44])
    ypos, y = [], 0.0
    for i, r in peaks.iterrows():
        if i and r.arch != peaks.arch[i - 1]:
            y += 0.6
        ypos.append(y)
        y += 1.0
    peaks["y"] = ypos
    for _, r in peaks.iterrows():
        xs = [r[f"peak_{p}"] for p in PROCS]
        ax.plot([min(xs), max(xs)], [r.y, r.y], "-", color="#9CA3AF", lw=0.6, zorder=1)
        for p in PROCS:
            col = ns.PROC[p]
            ax.plot(r[f"peak_{p}"], r.y + OFFSET[p], MARK[p], ms=MS_B[p], mec=col, mew=0.9,
                    mfc="white" if not r.in_order_or_tied else col, zorder=3)
    ax.set_yticks(peaks.y)
    ax.set_yticklabels(peaks.run)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(peaks.y.max() + 0.6, -0.6)
    ax.set_xlim(-0.4, 9.4)
    ax.set_xticks(range(0, 10))
    ax.set_xlabel("Transition of peak event rate")
    ax.text(0.99, 0.97, f"in order or tied: {n_ok} of {n_runs} runs", transform=ax.transAxes, ha="right",
            va="top", fontsize=6.5)

    # shared legend strip (right of b)
    H = [Line2D([], [], color=ns.PROC[p], lw=1.2, marker=MARK[p], ms=MS_B[p] - 0.5, label=lab) for p, lab in
         zip(PROCS, ["task-general (Tg-E)", "superclass (Ab-E)", "class (Di-E)"])]
    H += [Line2D([], [], color="#4B5563", lw=1.0, ls=SEED_LS[s], label=SEED_LAB[s]) for s in SEEDS]
    H += [Line2D([], [], color="#4B5563", lw=0, marker="o", mfc="white", mew=0.9, ms=3.5,
                 label="peaks out of order (b)")]
    fig.legend(handles=H, loc="center left", bbox_to_anchor=(0.71, 0.28), fontsize=6.5, frameon=False,
               handlelength=2.6, labelspacing=0.55)

    # Source Data
    cols = ["arch", "seed", "transition_index", "transition", "delta_weight_updates"]
    cols += [f"{p}_{k}" for p in PROCS for k in ("events", "per_1k_updates", "scaled_to_run_peak")]
    ns.source_data(df[cols], NAME, "a")
    ns.source_data(peaks.drop(columns="y"), NAME, "b")
    out = ns.save_final(fig, NAME, 180, ed=True)
    print("saved", out, "| in order or tied:", n_ok, "of", n_runs)


if __name__ == "__main__":
    main()
