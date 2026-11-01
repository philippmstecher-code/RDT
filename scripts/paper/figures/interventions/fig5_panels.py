#!/usr/bin/env python3
"""Fig. 5: blocking the learning signal along the superclass directions. 2 x 2 layout (180 x 107 mm).

  a  RDT superclass-selective features per checkpoint under the coarse block, the matched block and no block
  b  fine-class accuracy, epochs behind no block, for the coarse, matched and energy-matched blocks
  c  epochs behind no block at epoch 10, cumulative superclass events (RDT; blue end) and fine-class accuracy (amber
     end), coarse block and random grouping (coarse structure damaged) against the matched and energy-matched blocks
     (spared); thin grey connectors, seeds; thick grey connector, mean
  d  RDT share of class-selective features confined to their superclass, epochs 1-10 (chance 5%, dashed); counts of
     class-selective features per checkpoint go to Source Data (fig5_d.csv) only
Colour = level measured (superclass blue, class amber, references grey); within a level, coarse block darkest solid,
matched block mid dashed, energy-matched light dotted, no block thin grey solid.
Data: RDT_pace/long.csv, PLACEMENT_curve/long.csv, RDT_completion/rdt_completion_long.csv.

Usage: [ES_FIG_OUT=<dir>] python scripts/paper/figures/interventions/fig5_panels.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # scripts/paper/figures
sys.path.insert(0, str(HERE.parents[1]))   # scripts/paper
import nmi_style as ns  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

RESULTS = ns.OUT.parent
NAME, W, H = "fig5", 180, 107
SEEDS = [7, 42, 137, 256, 314]
# level ramps; local fallbacks if nmi_style lacks them
BLUE = getattr(ns, "BLUE", ["#1E40AF", "#3B82F6", "#60A5FA", "#93C5FD"])
AMBER = getattr(ns, "AMBER", ["#92400E", "#D97706", "#F59E0B", "#FCD34D"])
GREYS = getattr(ns, "GREYS", ["#374151", "#6B7280", "#9CA3AF", "#E5E7EB"])
BLUE_T, AMBER_T, GREY_T = "#1E40AF", "#92400E", "#6B7280"
DARK = GREYS[0]
WINDOW = "#F3F4F6"
# arm -> (label, ramp index or None for the grey reference, line style, mean line width)
ARMS = {"gradsc": ("coarse block", 0, "-", 1.7), "gradjoint": ("matched block", 1, "--", 1.4),
        "gradenergy": ("energy-matched", 2, ":", 1.5), "sham": ("no block", None, "-", 1.0)}


def col(arm, ramp):
    k = ARMS[arm][1]
    return GREYS[2] if k is None else ramp[k]


def seedlines(ax, x, Y, color, ls="-", lw=1.4, label=None, alpha=0.3, thin=0.45, z=2):
    """Thin solid seed lines under the mean, which carries the arm's line style."""
    Y = np.asarray(Y, float)
    for y in Y:
        ax.plot(x, y, color=color, lw=thin, alpha=alpha, ls="-", zorder=z)
    ax.plot(x, np.nanmean(Y, 0), color=color, lw=lw, ls=ls, label=label, zorder=z + 1,
            dash_capstyle="round" if ls == ":" else "butt")


def behind(a, c, eps):
    c = np.maximum.accumulate(np.asarray(c, float))
    return [e - float(np.interp(v, c, eps)) if v <= c[-1] else np.nan for e, v in zip(eps, a)]


TITLES = {}


def title(ax, s, pad=None, **kw):
    """Panel titles are placed by main() beside the panel letter (ED style: bold letter, then the claim)."""
    TITLES[ax] = s


def panel_a(ax, P):
    rows, end = [], {}
    for arm in ("sham", "gradjoint", "gradsc"):
        lab, _, ls, lw = ARMS[arm]
        c = col(arm, BLUE)
        Y = np.array([P[(P.arm == arm) & (P.seed == s)].sort_values("epoch").n_sc.values for s in SEEDS])
        seedlines(ax, np.arange(16), Y, c, ls, lw)
        end[arm] = Y[:, 10].mean()
        rows += [dict(arm=arm, arm_label=lab, seed=s, epoch=e, n_superclass_selective=v)
                 for s, y in zip(SEEDS, Y) for e, v in enumerate(y)]
    ax.axvspan(0.5, 10.5, color=WINDOW, zorder=0, lw=0)
    ax.text(5.5, 43.5, "block", fontsize=6, color=GREY_T, ha="center", va="top")
    ax.set_xlim(0, 15); ax.set_ylim(0, 44); ax.set_xticks([0, 5, 10, 15]); ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_xlabel("Epoch", labelpad=1.5); ax.set_ylabel("Superclass-selective\nfeatures per checkpoint")
    for arm, y_, tc in (("sham", 24.5, GREY_T), ("gradjoint", 16.0, BLUE[1]), ("gradsc", 8.5, BLUE_T)):
        ax.text(15.4, y_, ARMS[arm][0].replace(" block", "\nblock"), fontsize=6, color=tc, va="center",
                linespacing=0.95, clip_on=False)
    title(ax, "The coarse block removes the coarse structure")
    ns.source_data(pd.DataFrame(rows), NAME, "a")
    print("a epoch-10 means: " + ", ".join(f"{k} {v:.1f}" for k, v in end.items()))


def panel_b(ax, Pl):
    """Fine-class accuracy behind no block: coarse block against the matched and energy-matched blocks."""
    eps = list(range(16))
    rows, e10 = [], {}
    for arm in ("gradsc", "gradjoint", "gradenergy"):
        lab, _, ls, lw = ARMS[arm]
        Y = []
        for s in SEEDS:
            a = Pl[(Pl.family == "block") & (Pl.arm == arm) & (Pl.seed == s)].set_index("epoch").loc[eps]
            b = Pl[(Pl.family == "block") & (Pl.arm == "sham") & (Pl.seed == s)].set_index("epoch").loc[eps]
            Y.append(behind(a.fine.values, b.fine.values, eps))
        seedlines(ax, np.arange(16), Y, col(arm, AMBER), ls, lw, lab, alpha=0.3)
        e10[arm] = np.nanmean(np.asarray(Y)[:, 10])
        rows += [dict(arm=arm, arm_label=lab, seed=s, epoch=e, fine_accuracy_epochs_behind_no_block=v)
                 for s, y in zip(SEEDS, Y) for e, v in enumerate(y)]
    ax.axvspan(0.5, 10.5, color=WINDOW, zorder=0, lw=0)
    ax.axhline(0, color=GREYS[2], lw=0.6, ls="-", zorder=1)
    ax.text(15, 0.15, "no block", fontsize=6, color=GREY_T, ha="right", va="bottom")
    ax.set_xlim(1, 15); ax.set_ylim(-0.5, 9); ax.set_xticks([1, 5, 10, 15])
    ax.set_xlabel("Epoch", labelpad=1.5); ax.set_ylabel("Fine-class accuracy,\nepochs behind no block")
    ax.legend(fontsize=6, frameon=False, loc="upper left", handlelength=2.2, borderaxespad=0.1, labelspacing=0.25)
    title(ax, "Specificity: fine-grained learning slows beyond matched damage")
    ns.source_data(pd.DataFrame(rows), NAME, "b")
    print("b epoch-10 means: " + ", ".join(f"{k} {v:.2f}" for k, v in e10.items()))


def panel_c(ax, Pl, R):
    """Epochs behind no block at epoch 10: superclass events (RDT, cumulative; blue end) and fine-class accuracy
    (amber end). Thin grey connectors, seeds; thick grey connector and dots, mean."""
    b16 = list(range(16))
    q = R[(R.kind == "events") & (R.layer == "all") & (R.tset == "common") & (R.measure == "Ab-E")]

    def cum(arm, s):
        v = q[(q.arm == arm) & (q.seed == s)].set_index("epoch").value.reindex(range(1, 16)).fillna(0)
        return np.r_[0, np.cumsum(v.values)]

    def fine(arm, s):
        a = Pl[(Pl.family == "block") & (Pl.arm == arm) & (Pl.seed == s)].set_index("epoch").loc[b16]
        b = Pl[(Pl.family == "block") & (Pl.arm == "sham") & (Pl.seed == s)].set_index("epoch").loc[b16]
        return behind(a.fine.values, b.fine.values, b16)[10]

    def block(arm):
        return np.array([(behind(cum(arm, s), cum("sham", s), b16)[10], fine(arm, s)) for s in SEEDS])

    items = [("coarse\nblock", block("gradsc")), ("random\ngrouping", block("gradrg")),
             ("matched\nblock", block("gradjoint")), ("energy-\nmatched", block("gradenergy"))]
    SCD, CLD = BLUE[1], AMBER[1]
    rows, xs = [], []
    gap, step, split = 0.62, 1.25, 0.5
    tr = ax.get_xaxis_transform()
    for k, (lab, XY) in enumerate(items):
        x0 = k * step + (split if k >= 2 else 0)
        xs.append(x0)
        for (sx, fy) in XY:
            ax.plot([x0, x0 + gap], [sx, fy], color=GREYS[2], lw=0.5, alpha=0.8, zorder=2)
        m = XY.mean(0)
        ax.plot([x0, x0 + gap], m, color=GREYS[1], lw=1.6, zorder=3, solid_capstyle="round")
        ax.scatter([x0], [m[0]], s=18, color=SCD, edgecolor="white", lw=0.5, zorder=4)
        ax.scatter([x0 + gap], [m[1]], s=18, color=CLD, edgecolor="white", lw=0.5, zorder=4)
        ax.text(x0 + gap / 2, -0.04, lab, fontsize=6, ha="center", va="top", color=DARK, linespacing=0.95,
                transform=tr)
        name = lab.replace("-\n", "-").replace("\n", " ")
        rows += [dict(manipulation=name, reference="no block", coarse_measure="cumulative superclass events (RDT)",
                      fine_measure="fine-class accuracy", epoch=10, seed=s, coarse_lag_epochs=a, fine_lag_epochs=b)
                 for s, (a, b) in zip(SEEDS, XY)]
        rows.append(dict(manipulation=name, reference="no block", coarse_measure="cumulative superclass events (RDT)",
                         fine_measure="fine-class accuracy", epoch=10, seed="mean", coarse_lag_epochs=m[0],
                         fine_lag_epochs=m[1]))
    xm = (xs[1] + gap + xs[2]) / 2
    ax.axvline(xm, color="#D1D5DB", lw=0.6, zorder=0)
    ax.text((xs[0] + xs[1] + gap) / 2, 1.0, "coarse structure damaged", fontsize=6, ha="center", va="bottom",
            color=GREY_T, transform=tr)
    ax.text((xs[2] + xs[3] + gap) / 2, 1.0, "coarse structure spared", fontsize=6, ha="center", va="bottom",
            color=GREY_T, transform=tr)
    h = [Line2D([], [], marker="o", ls="none", ms=4, color=SCD, label="superclass events (RDT)"),
         Line2D([], [], marker="o", ls="none", ms=4, color=CLD, label="fine-class accuracy")]
    ax.legend(handles=h, fontsize=6, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.0), handletextpad=0.1,
              borderaxespad=0.1, labelspacing=0.3, handlelength=1.0)
    ax.axhline(0, color="#D1D5DB", lw=0.5, zorder=0)
    ax.set_xlim(-0.3, xs[-1] + gap + 0.3); ax.set_ylim(-3, 10.2)
    ax.set_yticks([-2, 0, 2, 4, 6, 8, 10])
    ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("Epochs behind no block,\nat epoch 10")
    title(ax, "Asymmetry: only the coarse delay crosses over")
    ns.source_data(pd.DataFrame(rows), NAME, "c")
    return {lab.replace("-\n", "-").replace("\n", " "): XY for lab, XY in items}


def panel_d(ax, P):
    """Share of class-selective features confined to their superclass, epochs 1-10 (bars, mean; dots, seeds);
    chance 5% dashed. Counts of class-selective features per checkpoint go to Source Data only."""
    W_ = P[P.epoch.between(1, 10)]
    rng = np.random.default_rng(3)
    rows = []
    order = (("sham", GREYS[2]), ("gradjoint", AMBER[1]), ("gradsc", AMBER[0]))
    for i, (arm, c) in enumerate(order):
        g = W_[W_.arm == arm].groupby("seed")
        n = g.n_cl.mean().loc[SEEDS].values
        sh = (100 * g.n_cl_confined.sum() / g.n_cl.sum()).loc[SEEDS].values
        ax.bar(i, sh.mean(), color=c, alpha=0.9, width=0.62, lw=0, zorder=1)
        ax.scatter(i + rng.uniform(-0.13, 0.13, 5), sh, s=5, color="white", edgecolor=DARK, lw=0.45, zorder=3)
        ax.text(i, sh.max() + 1.5, f"{sh.mean():.0f}%", fontsize=6.5, ha="center", va="bottom", color=DARK)
        rows += [dict(arm=arm, arm_label=ARMS[arm][0], epochs="1-10", seed=s, class_selective_per_checkpoint=a,
                      class_selective_confined=int(W_[(W_.arm == arm) & (W_.seed == s)].n_cl_confined.sum()),
                      class_selective_total=int(W_[(W_.arm == arm) & (W_.seed == s)].n_cl.sum()),
                      share_confined_pct=b) for s, a, b in zip(SEEDS, n, sh)]
        print(f"d {arm}: confined {sh.mean():.1f}% (seeds " + ", ".join(f"{v:.1f}" for v in sh)
              + f"); class-selective per checkpoint {n.mean():.1f}")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["no\nblock", "matched\nblock", "coarse\nblock"], fontsize=6.5,
                                                  linespacing=0.95)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylabel("Class-selective features\nconfined to their superclass (%)")
    ax.set_ylim(0, 50); ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.axhline(5, color=DARK, lw=0.7, ls=(0, (3, 2)), zorder=2.5)
    ax.text(2.6, 5, "chance", fontsize=6, color=DARK, ha="left", va="center", clip_on=False)
    title(ax, "Prediction: class features form\noutside their superclass")
    ns.source_data(pd.DataFrame(rows), NAME, "d")


def main():
    ns.apply()
    for f in (ns.FINAL_OUT / "source_data").glob(f"{NAME}_*.csv"):
        f.unlink()
    P = pd.read_csv(RESULTS / "RDT_pace" / "long.csv")
    Pl = pd.read_csv(RESULTS / "PLACEMENT_curve" / "long.csv")
    R = pd.read_csv(RESULTS / "RDT_completion" / "rdt_completion_long.csv")
    fig = plt.figure(figsize=(W * ns.MM, H * ns.MM))
    mm = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
    y1, h1 = 63, 35          # row 1: a | b
    y2, h2 = 9, 36          # row 2: c | d
    a, b = mm(16, y1, 62, h1), mm(112, y1, 64, h1)
    c = mm(16, y2, 92, h2 - 4)
    d = mm(136, y2, 34, h2 - 4)
    panel_a(a, P)
    panel_b(b, Pl)
    F = panel_c(c, Pl, R)
    panel_d(d, P)
    for k, XY in F.items():
        print(f"c {k:16s} mean coarse {XY[:, 0].mean():.3f} fine {XY[:, 1].mean():.3f} | per seed {SEEDS}: coarse "
              + ", ".join(f"{v:.3f}" for v in XY[:, 0]) + " ; fine " + ", ".join(f"{v:.3f}" for v in XY[:, 1]))
    t1, t2 = H - 1.5, y2 + h2 + 4
    for ax, letter, x, y in ((a, "a", 0.5, t1), (b, "b", 96, t1), (c, "c", 0.5, t2), (d, "d", 118, t2)):
        fig.text(x / W, y / H, letter, fontsize=8, fontweight="bold", va="top", ha="left")
        fig.text((x + 3.8) / W, (y - 0.25) / H, TITLES[ax], fontsize=7, va="top", ha="left", linespacing=1.1)
    print("saved", ns.save_final(fig, NAME, 180, strict=True))


if __name__ == "__main__":
    main()
