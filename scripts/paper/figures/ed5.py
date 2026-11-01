#!/usr/bin/env python3
"""Extended Data Fig. 5: superclass mode strength in the images and the time at which networks learn
each superclass, in the seven held-out networks.

a-g  one scatter per network: mode strength s of each of the 20 superclasses (distance of the superclass mean from the
     grand mean of the training images after whitening with their first 64 principal components) against its learning
     lead (mean normalised superclass probability over checkpoints minus the mean over superclasses); Spearman rho per
     panel; grey line, least-squares fit (#6B7280).
h    Spearman rho per network (diamonds, architecture colours) against 30 random partitions of the 100 classes into
     20 groups of five (grey dots), and the pooled partial rho controlling for the final probability (random-effects
     meta-analysis of Fisher-transformed values, 95% CI).
Source: analysis_outputs/paper/DATA_order/mode_strength/{saxe_superclass_units.csv, mode_strength_per_network.csv, saxe_partition_test.csv,
mode_strength_results.json}.
Usage: python scripts/paper/figures/ed5.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nmi_style as ns  # noqa: E402
from checks_lib import RESULTS, MMCanvas, arch_of, ols_line, spearman  # noqa: E402

FIG = "ed5"
W_MM, H_MM = 180, 211
SRC = RESULTS / "DATA_order" / "mode_strength"
FIT = "#6B7280"
NULL_DOT = "#9CA3AF"
DARK = ns.REF_DARK
SMALL = 6.0
ORDER = ["core_ResNet-18_s137", "core_ViT-Small_s137", "core_CCT-7_s137", "clean_ResNet-18_s256",
         "clean_ResNet-18_s314", "noise_std_ResNet-18_s256", "noise_std_ResNet-18_s314"]


def net_title(lane):
    arch = arch_of(lane)
    seed = lane.rsplit("_s", 1)[1]
    if lane.startswith("core"):
        ep = 100 if arch == "ViT-Small" else 50
        return f"{arch}, seed {seed}, {ep} epochs"
    if lane.startswith("clean"):
        return f"{arch}, seed {seed}, 30 epochs"
    return f"{arch}, seed {seed}, 30 epochs\nlabel-noise set (clean)"


def main():
    ns.apply()
    U = pd.read_csv(SRC / "saxe_superclass_units.csv")
    P = pd.read_csv(SRC / "mode_strength_per_network.csv").set_index("lane")
    PT = pd.read_csv(SRC / "saxe_partition_test.csv")
    J = json.loads((SRC / "mode_strength_results.json").read_text())
    assert sorted(U.lane.unique()) == sorted(ORDER) and len(U) == 140

    fig = plt.figure(figsize=(W_MM * ns.MM, H_MM * ns.MM))
    C = MMCanvas(fig, W_MM, H_MM)
    AW, AH = 46, 48
    xs = [13, 71, 129]
    ys = [11, 80, 150]
    axes = [C.ax(xs[i % 3], ys[i // 3], AW, AH) for i in range(7)]
    axes.append(C.ax(xs[1], ys[2], xs[2] + AW - xs[1], AH))
    rows, rho = [], {}
    xlim = (U.s.min() - 0.08, U.s.max() + 0.08)
    ylim = (U["lead"].min() - 0.03, U["lead"].max() + 0.03)
    for ax, lane, L in zip(axes, ORDER, "abcdefg"):
        g = U[U.lane == lane]
        arch = arch_of(lane)
        r = spearman(g.s, g["lead"])
        assert abs(r - P.loc[lane, "rho_lead"]) < 1e-6, (lane, r)
        rho[lane] = r
        a, b = ols_line(g.s, g["lead"])
        xx = np.array(xlim)
        ax.plot(xx, a + b * xx, color=FIT, lw=0.8, zorder=1)
        ax.axhline(0, color="#D1D5DB", lw=0.5, zorder=0)
        ax.scatter(g.s, g["lead"], s=9, color=ns.ARCH[arch], lw=0, alpha=0.9, zorder=3)
        ax.text(0.04, 0.96, f"ρ = {r:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=DARK)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(net_title(lane), fontsize=6.3, pad=2.5, linespacing=1.1)
        ax.tick_params(labelsize=6.2, pad=1.5)
        if L in "adg":
            ax.set_ylabel("Learning lead", fontsize=6.5)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Mode strength (whitened images)", fontsize=6.5)
        ns.panel(ax, L, x=-0.22 if L in "adg" else -0.08, y=1 + (8 if L in "abc" else 10) / AH)
        for _, rr in g.iterrows():
            rows.append(dict(panel=L, lane=lane, arch=arch, superclass=int(rr.sc), mode_strength=rr.s,
                             learning_lead=rr["lead"], final_probability=rr.final))

    # h: rho per network vs random partitions, plus the pooled partial rho
    ax = axes[7]
    hrows = []
    PCT = {}
    for i, (lane, L) in enumerate(zip(ORDER, "abcdefg")):
        arch = arch_of(lane)
        rnd = PT[(PT.lane == lane) & (PT.partition != "true")].rho_lead.values
        assert len(rnd) == 30
        jit = np.random.default_rng(i).uniform(-0.18, 0.18, len(rnd))
        ax.scatter(i + jit, rnd, s=3, color=NULL_DOT, lw=0, zorder=2)
        ax.scatter([i], [rho[lane]], s=20, marker="D", color=ns.ARCH[arch], lw=0, zorder=3)
        pct = 100 * float((rnd < rho[lane]).mean())
        PCT[lane] = pct
        ax.text(i, 0.97, f"{pct:.0f}", ha="center", va="bottom", fontsize=SMALL, color=ns.ARCH[arch], rotation=90)
        hrows += [dict(network=L, lane=lane, partition=p, rho_lead=v) for p, v in
                  zip(PT[(PT.lane == lane)].partition, PT[(PT.lane == lane)].rho_lead)]
    c2 = J["criterion2"]
    pr, (lo, hi) = c2["pooled_rho"], c2["ci95_rho"]
    xp = 7.9
    ax.errorbar([xp], [pr], yerr=[[pr - lo], [hi - pr]], fmt="o", ms=3.2, mfc="white", mec=DARK, ecolor=DARK,
                elinewidth=0.8, capsize=2, capthick=0.8, zorder=3)
    ax.axvline(7.0, color="#E5E7EB", lw=0.6)
    ax.axhline(0, color="#D1D5DB", lw=0.5, zorder=0)
    ax.set_xticks(list(range(7)) + [xp])
    ax.set_xticklabels(list("abcdefg") + ["pooled\npartial"], fontsize=6.2)
    ax.set_xlim(-0.6, 8.6)
    ax.set_ylim(-0.5, 1.57)
    ax.set_yticks([-0.25, 0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Spearman ρ", fontsize=6.5, labelpad=1)
    ax.tick_params(labelsize=6.2, pad=1.5)
    ax.set_title("True superclasses vs\nrandom partitions", fontsize=6.3, pad=2.5, linespacing=1.1)
    ax.text(3.0, 1.55, "% of random partitions\nbelow true", ha="center", va="top", fontsize=SMALL, color=DARK,
            linespacing=1.0)
    med_rnd = float(PT[PT.partition != "true"].rho_lead.median())
    ax.set_xlabel("Network", fontsize=6.5)
    h = [Line2D([], [], marker="D", color=DARK, ls="none", ms=3.2, label="true superclasses"),
         Line2D([], [], marker="o", color=NULL_DOT, ls="none", ms=2.0, label=f"random partitions (median {med_rnd:.2f})")]
    ax.legend(handles=h, fontsize=SMALL, loc="lower left", handletextpad=0.3, borderaxespad=0.1, labelspacing=0.2)
    ns.panel(ax, "h", x=-0.10, y=1 + 10 / AH)

    ns.source_data(pd.DataFrame(rows), FIG, "a_g")
    ns.source_data(pd.DataFrame(hrows), FIG, "h_random_partitions")
    summ = [dict(network=L, lane=lane, arch=arch_of(lane), rho_lead=rho[lane],
                 partial_rho_given_final=P.loc[lane, "partial_lead"],
                 frac_random_partitions_rho_ge_true=P.loc[lane, "frac_rand_rho_lead_ge_true"],
                 pct_random_partitions_below_true=PCT[lane])
            for lane, L in zip(ORDER, "abcdefg")]
    summ.append(dict(network="pooled", lane="random-effects meta-analysis (Fisher z)", arch="",
                     rho_lead=np.nan, partial_rho_given_final=pr, frac_random_partitions_rho_ge_true=np.nan,
                     ci95_lo=lo, ci95_hi=hi))
    ns.source_data(pd.DataFrame(summ), FIG, "h")
    out = ns.save_final(fig, FIG, W_MM, ed=False)
    print(out)
    print("rho range %.3f-%.3f; pooled partial %.3f (%.3f-%.3f)" % (min(rho.values()), max(rho.values()), pr, lo, hi))


if __name__ == "__main__":
    main()
