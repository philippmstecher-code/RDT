#!/usr/bin/env python3
"""Supplementary Fig. 6: class-selective features under acquisition blocking.

Blocking arms (gradient projected off a set of directions in epochs 1-10): coarse block (gradsc), random grouping
(gradrg), random directions (gradrand), energy-matched block (gradenergy); no block (sham). In a the arms take amber
shades (coarse block darkest circles, random grouping mid squares, random directions light triangles, energy-matched
block open dark diamonds); b is amber (class-selective features). Windows: block = epochs 1-10, release = epochs 11-15
(means over the checkpoints of the window, per seed).
a  per arm and window: change vs sham in superclass assignment (fresh ridge read-out of the pooled layer, 5-fold CV;
   GRADBLOCK/final_long.csv, seeds 7, 42, 137, 256, 314) against the change vs sham in the share of class-selective
   features' activity that comes from images of other superclasses (pooled-layer SAE, sham thresholds;
   RDT_blocking/feature_routing_long.csv, seeds 7, 42, 137). Large markers: arm means (x over 5 seeds, y over 3) with
   t-based 95% CIs; small markers: the 3 seeds with both measures.
b  share of alive features that are class-selective (census layers layer1-3 + pooled, common thresholds), % change vs
   sham per seed (RDT_blocking/rdt_blocking_long.csv); bars, mean of 3 seeds; dots, seeds. Values equal the
   pct_cls contrasts of rdt_blocking_summary.json (asserted).

Usage: python scripts/paper/figures/supp6.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nmi_style as ns  # noqa: E402
import noise_lib as L  # noqa: E402
from core_runs_lib import apply_style  # noqa: E402
from checks_lib import RESULTS, MMCanvas  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

FIG = "supp6"
W_MM, H_MM = 180, 66
HUE = ns.INDEX["CSI"]                      # class-selective features (b)
GREY = "#6B7280"
RDT_SEEDS = (7, 42, 137)
ACC_SEEDS = (7, 42, 137, 256, 314)
ARMS = [("gradsc", "Coarse block", "o", True), ("gradrg", "Random grouping", "s", True),
        ("gradrand", "Random directions", "^", True), ("gradenergy", "Energy-matched block", "D", False)]
ARM_HUE = {"gradsc": ns.AMBER[0], "gradrg": ns.AMBER[1], "gradrand": ns.AMBER[2], "gradenergy": ns.AMBER[0]}
YLIM_A, YTICKS_A = (-11.0, 17.5), [-8, -4, 0, 4, 8, 12, 16]   # range covers the whole 95% CIs
WINDOWS = {"block": ("Block (epochs 1–10)", range(1, 11)), "release": ("After release (epochs 11–15)", range(11, 16))}
AX = {"a1": (16.0, 9.0, 44.0), "a2": (70.0, 9.0, 44.0), "b": (134.0, 9.0, 44.0)}
AH = 40.0


def window_mean(D, col, eps, seeds):
    M = D[D.epoch.isin(list(eps))].groupby(["arm", "seed"])[col].mean()
    return {a: M.loc[a].loc[list(seeds)].to_numpy(float) for a in D.arm.unique()}


def load():
    G = pd.read_csv(RESULTS / "GRADBLOCK" / "final_long.csv")
    F = pd.read_csv(RESULTS / "RDT_blocking" / "feature_routing_long.csv")
    R = pd.read_csv(RESULTS / "RDT_blocking" / "rdt_blocking_long.csv", low_memory=False)
    R = R[(R.kind == "census") & (R.layer == "census_pooled") & (R.tset == "common") & (R.measure == "pct_cls")]
    R = R.rename(columns={"value": "pct_cls"})[["arm", "seed", "epoch", "pct_cls"]]
    out = {}
    for w, (_, eps) in WINDOWS.items():
        acc = window_mean(G, "fresh_assignment", eps, ACC_SEEDS)
        off = window_mean(F, "off_superclass_share", eps, RDT_SEEDS)
        cls = window_mean(R, "pct_cls", eps, RDT_SEEDS)
        out[w] = {a: {"assign": acc[a] - acc["sham"], "off": 100 * (off[a] - off["sham"]),
                      "cls_pct": 100 * (cls[a] / cls["sham"] - 1)} for a, *_ in ARMS}
    S = json.loads((RESULTS / "RDT_blocking" / "rdt_blocking_summary.json").read_text())
    for c in S["contrasts"]:
        a, b = c["contrast"].split(" - ")
        if c["tset"] == "common" and c["measure"] == "pct_cls mean" and b == "sham" and c["window"] in out \
                and a in out[c["window"]]:
            assert abs(out[c["window"]][a]["cls_pct"].mean() - c["pct_change_mean"]) < 1e-6, (a, c["window"])
    return out


def ci(v):
    m, lo, hi = L.t_ci(np.asarray(v, float)[:, None])
    return float(m[0]), float(lo[0]), float(hi[0])


def draw_a(ax, V, title, show_ylabel):
    ax.axhline(0, color=GREY, lw=0.5, zorder=1)
    ax.axvline(0, color=GREY, lw=0.5, zorder=1)
    rows = []
    for arm, lab, mk, filled in ARMS:
        c = ARM_HUE[arm]
        v = V[arm]
        x = v["assign"]
        y = v["off"]
        xi = x[[ACC_SEEDS.index(s) for s in RDT_SEEDS]]
        ax.plot(xi, y, marker=mk, ms=2.6, ls="none", color=c, mfc=c if filled else "white", mec=c, mew=0.6,
                alpha=0.55, zorder=3)
        mx, xlo, xhi = ci(x)
        my, ylo, yhi = ci(y)
        ax.errorbar(mx, my, xerr=[[mx - xlo], [xhi - mx]], yerr=[[my - ylo], [yhi - my]], fmt="none", ecolor=c,
                    elinewidth=0.7, capsize=0, zorder=4)
        ax.plot(mx, my, marker=mk, ms=5.0, ls="none", color=c, mfc=c if filled else "white", mec="white"
                if filled else c, mew=0.6 if filled else 0.9, zorder=5)
        rows.append(dict(arm=arm, label=lab, assignment_change_mean=mx, assignment_ci_lo=xlo, assignment_ci_hi=xhi,
                         n_assignment=len(x), off_superclass_change_mean=my, off_superclass_ci_lo=ylo,
                         off_superclass_ci_hi=yhi, n_off_superclass=len(y),
                         **{f"assignment_change_seed_{s}": x[i] for i, s in enumerate(ACC_SEEDS)},
                         **{f"off_superclass_change_seed_{s}": y[i] for i, s in enumerate(RDT_SEEDS)}))
    ax.set_xlim(-17, 2)
    ax.set_xticks([-15, -10, -5, 0])
    ax.set_ylim(*YLIM_A)
    ax.set_yticks(YTICKS_A)
    ax.set_xlabel("Superclass assignment,\nchange vs no block (points)", labelpad=1.5)
    if show_ylabel:
        ax.set_ylabel("Class-feature activity from other\nsuperclasses, change vs no block (points)")
    ax.set_title(title, fontsize=7, pad=3)
    return rows


def draw_b(ax, out):
    ax.axhline(0, color=GREY, lw=0.5, zorder=1)
    xs = np.arange(len(ARMS))
    rows = []
    for j, (w, off, alpha) in enumerate((("block", -0.19, 0.85), ("release", 0.19, 0.35))):
        for i, (arm, lab, mk, filled) in enumerate(ARMS):
            v = out[w][arm]["cls_pct"]
            m, lo, hi = ci(v)
            ax.bar(xs[i] + off, m, width=0.36, color=HUE, alpha=alpha, lw=0, zorder=2)
            ax.plot(np.full(len(v), xs[i] + off), v, marker=mk, ms=2.4, ls="none", color=REF, mfc="white", mew=0.6,
                    zorder=3)
            rows.append(dict(window=WINDOWS[w][0], arm=arm, label=lab, pct_change_mean=m, ci_lo=lo, ci_hi=hi,
                             n_seeds_below_zero=int((v < 0).sum()),
                             **{f"pct_change_seed_{s}": v[k] for k, s in enumerate(RDT_SEEDS)}))
    ax.set_xticks(xs)
    ax.set_xticklabels(["Coarse\nblock", "Random\ngrouping", "Random\ndirections", "Energy-\nmatched\nblock"],
                       fontsize=6)
    ax.set_xlim(-0.6, len(ARMS) - 0.4)
    ax.set_ylim(-80, 80)
    ax.set_yticks([-80, -40, 0, 40, 80])
    ax.set_ylabel("Class-selective features,\nchange vs no block (%)")
    ax.set_title("Class-selective features", fontsize=7, pad=3)
    ax.legend(handles=[Patch(facecolor=HUE, alpha=0.85, lw=0, label="Block (1–10)"),
                       Patch(facecolor=HUE, alpha=0.35, lw=0, label="After release (11–15)")],
              loc="upper left", fontsize=6, frameon=False, handlelength=1.2, borderaxespad=0.2)
    return rows


REF = ns.REF_DARK


def main():
    apply_style()
    out = load()
    fig = plt.figure(figsize=(W_MM * ns.MM, H_MM * ns.MM))
    C = MMCanvas(fig, W_MM, H_MM)
    axs = {k: C.ax(x, y, w, AH) for k, (x, y, w) in AX.items()}
    ra = draw_a(axs["a1"], out["block"], WINDOWS["block"][0], True)
    rb = draw_a(axs["a2"], out["release"], WINDOWS["release"][0], False)
    axs["a2"].legend(handles=[Line2D([], [], marker=mk, ms=4.0, ls="none", color=ARM_HUE[a],
                                     mfc=ARM_HUE[a] if f else "white", mec=ARM_HUE[a], mew=0.9, label=lab)
                              for a, lab, mk, f in ARMS],
                     loc="upper left", fontsize=6, frameon=False, handletextpad=0.3, borderaxespad=0.2,
                     labelspacing=0.25)
    rc = draw_b(axs["b"], out)
    for k, lx in (("a", AX["a1"][0] - 14.0), ("b", AX["b"][0] - 14.0)):
        fig.text(lx / W_MM, 1 - (AX["a1"][1] - 6.5) / H_MM, k, fontsize=8, fontweight="bold", va="top", ha="left")

    a_df = pd.concat([pd.DataFrame(ra).assign(window=WINDOWS["block"][0]),
                      pd.DataFrame(rb).assign(window=WINDOWS["release"][0])], ignore_index=True)
    ns.source_data(a_df.round(4), FIG, "a")
    ns.source_data(pd.DataFrame(rc).round(4), FIG, "b")
    path = ns.save_final(fig, FIG, W_MM, ed=False)
    print(path, f"{W_MM} x {H_MM} mm")
    print(a_df[["window", "arm", "assignment_change_mean", "off_superclass_change_mean"]].round(2).to_string())
    print(pd.DataFrame(rc)[["window", "arm", "pct_change_mean", "ci_lo", "ci_hi", "n_seeds_below_zero"]]
          .round(1).to_string())


if __name__ == "__main__":
    main()
