#!/usr/bin/env python3
"""Routing of the learning signal: three data panels (stand-alone; intervention_panels.py imports the panel functions).

i    standard training (sham networks, n = 5): share of images in the wrong superclass (dashed) and share of the
     learning signal they supply (solid), epochs 0-30; mean and t-based 95% CI band.
ii   standard training: share of each image state's learning signal that lies in the network's current superclass
     directions, for images in the wrong superclass (blue), in the right superclass but the wrong class (amber) and
     correct (grey), with as many random directions (dotted).
iii  superclass-direction and random-grouping-direction blocking minus energy-matched blocking over epochs 1-3: share
     of images in the wrong superclass and share of the learning signal they supply (bars, means; dots, seeds; 95% CI);
     colour = arm (ns.ARM), filled = learning signal, hatched = images.
Data: analysis_outputs/paper/SIGNAL_routing/signal_routing_long.csv (scripts/paper/signal_routing.py).
Usage: python scripts/paper/figures/signal_routing_panels.py [out.png]
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nmi_style as ns  # noqa: E402
import noise_lib as L  # noqa: E402
from fig_common import RESULTS  # noqa: E402

D = pd.read_csv(RESULTS / "SIGNAL_routing" / "signal_routing_long.csv")
SEEDS = [7, 42, 137, 256, 314]
SC, CL, GREY, ERR, DOT = ns.INDEX["SSI"], ns.INDEX["CSI"], "#9CA3AF", "#374151", "#9CA3AF"
W, H = 180, 52


def mm_axes(fig, x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def course(ax, col, color, ls, label, lw=1.2):
    S = D[D.arm == "sham"].pivot(index="seed", columns="epoch", values=col).loc[SEEDS]
    x = S.columns.values
    m, lo, hi = L.t_ci(S.values)
    ax.fill_between(x, 100 * lo, 100 * hi, color=color, alpha=0.18, lw=0)
    ax.plot(x, 100 * m, color=color, ls=ls, lw=lw, label=label)
    return x, m


def panel_i(ax):
    course(ax, "signal_share_misrouted", SC, "-", "share of the learning signal")
    course(ax, "image_share_misrouted", SC, "--", "share of images")
    ax.set_xlim(0, 30)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Epoch", labelpad=1.5)
    ax.set_ylabel("Misrouted images:\nshare (%)")
    ax.legend(loc="lower left", fontsize=6, frameon=False, handlelength=2.2, borderaxespad=0.2)
    ax.set_title("Misrouted images supply\nmost of the learning signal", fontsize=7, pad=3)


def panel_ii(ax):
    for st, color, lab in (("misrouted", SC, "wrong superclass"), ("routed", CL, "right superclass, wrong class"),
                           ("correct", GREY, "correct")):
        course(ax, f"code_share_{st}", color, "-", lab)
    course(ax, "random_share_all", GREY, ":", "random directions", lw=1.0)
    ax.set_xlim(0, 30)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_ylim(0, 78)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_xlabel("Epoch", labelpad=1.5)
    ax.set_ylabel("Learning signal in the\nsuperclass directions (%)")
    ax.legend(loc="upper right", fontsize=6.0, frameon=False, handlelength=1.6, borderaxespad=0.1, labelspacing=0.2)
    ax.set_title("Misrouted images' learning signal\nlies more in the superclass directions", fontsize=7, pad=3)


def panel_iii(ax):
    rng = np.random.default_rng(3)
    M = D[D.epoch.isin([1, 2, 3])].groupby(["arm", "seed"]).mean(numeric_only=True)
    # colour = intervention arm (ns.ARM); filled = learning signal, hatched = images
    arms = (("gradsc", "superclass\ndirections", ns.ARM["superclass"]),
            ("gradrg", "random-grouping\ndirections", ns.ARM["random_grouping"]))
    meas = (("signal_share_misrouted", -0.21, True), ("image_share_misrouted", 0.21, False))
    rows = []
    alpha = ns.BAR_ALPHA_CORE
    for i, (arm, lab, col_arm) in enumerate(arms):
        for col, dx, solid in meas:
            v = 100 * (M.loc[arm][col] - M.loc["gradenergy"][col]).loc[SEEDS].values
            m, lo, hi = L.t_ci(v)
            x = i + dx
            if solid:
                ax.bar(x, m, width=0.38, color=col_arm, alpha=alpha, lw=0)
            else:
                ax.bar(x, m, width=0.38, color=col_arm, alpha=alpha * 0.35, lw=0)
                ax.bar(x, m, width=0.38, facecolor="none", edgecolor=col_arm, lw=0.8, hatch="//////")
            ax.scatter(x + rng.uniform(-0.07, 0.07, len(v)), v, s=6, color=DOT, lw=0, zorder=3)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=ERR, elinewidth=0.9, capsize=2.2,
                        capthick=0.9, zorder=4)
            ax.text(x, max(hi, v.max()) + 0.3, f"+{m:.1f}", fontsize=6.2, color=ERR, ha="center", va="bottom")
            rows.append(dict(arm=arm, measure=col, mean=m, ci_lo=lo, ci_hi=hi, n_above_zero=int((v > 0).sum())))
    ax.axhline(0, color="#6B7280", ls="--", lw=0.6, zorder=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([a[1] for a in arms], fontsize=6.2)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-0.8, 12.5)
    ax.set_yticks([0, 4, 8])
    ax.set_ylabel("Misrouted, minus energy-\nmatched, epochs 1–3 (pp)")
    from matplotlib.patches import Patch
    h = [Patch(facecolor="#9CA3AF", alpha=0.8, lw=0, label="learning signal"),
         Patch(facecolor="white", edgecolor="#6B7280", hatch="//////", lw=0.8, label="images")]
    ax.legend(handles=h, loc="upper center", fontsize=6.0, frameon=False, handlelength=1.2, borderaxespad=0.1)
    ax.set_title("Blocking the directions keeps\nthe learning signal misrouted", fontsize=7, pad=3)
    for r in rows:
        print(f"[iii] {r['arm']:7s} {r['measure']:24s} {r['mean']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] "
              f"{r['n_above_zero']}/5 > 0")


def main():
    ns.apply()
    fig = plt.figure(figsize=(W * ns.MM, H * ns.MM))
    a1, a2, a3 = mm_axes(fig, 16, 9, 42, 30), mm_axes(fig, 77, 9, 46, 30), mm_axes(fig, 142, 9, 36, 30)
    panel_i(a1)
    panel_ii(a2)
    panel_iii(a3)
    for ax, letter, x in ((a1, "i", 1), (a2, "ii", 62), (a3, "iii", 127)):
        b = ax.get_position()
        ax.text((x / W - b.x0) / b.width, (H - 1) / H / b.height - b.y0 / b.height, letter, transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top")
    out = sys.argv[1] if len(sys.argv) > 1 else "signal_routing_panels.png"
    fig.savefig(out, dpi=300)
    print("saved", out)


if __name__ == "__main__":
    main()
