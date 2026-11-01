#!/usr/bin/env python3
"""Fig. 5, drawn with the effects stated on the panels. Uses fig5_panels.py for loaders, colours and panels a/d.

  a  RDT superclass-selective features (as fig5_panels.panel_a) with the epoch-10 means at the line ends
  b  specificity at the end of the block (epoch 10): fine-class accuracy, epochs behind no block, for energy-matched,
     matched and coarse block; thin lines join each seed's three values; brackets give the extra cost of the coarse
     block against matched and energy-matched
  c  asymmetry: per block, the delay of the variable it targets (left bar) and of the other variable (right bar),
     epochs behind no block at epoch 10, with the share that crosses over (ratio of the means) on the arrow;
     blue = coarse structure (cumulative superclass events, RDT), amber = fine-class accuracy
  d  share of class-selective features confined to their superclass (as fig5_panels.panel_d), with the number of
     class-selective features per checkpoint under each bar
Data: RDT_pace/long.csv, PLACEMENT_curve/long.csv, RDT_completion/rdt_completion_long.csv. Writes fig5.* and Source Data.

Usage: [ES_FIG_OUT=<dir>] python scripts/paper/figures/interventions/fig5.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
import nmi_style as ns  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
import fig5_panels as F  # noqa: E402

NAME, W, H = "fig5", 180, 110
SEEDS, BLUE, AMBER, GREYS, DARK = F.SEEDS, F.BLUE, F.AMBER, F.GREYS, F.DARK
GREY_T, BLUE_T, AMBER_T = F.GREY_T, F.BLUE_T, F.AMBER_T
TITLES = {}


def title(ax, s):
    TITLES[ax] = s


def bracket(ax, x0, x1, y, text, color=DARK, h=0.25, fs=6.5, bold=True):
    ax.plot([x0, x0, x1, x1], [y - h, y, y, y - h], color=color, lw=0.8, clip_on=False, zorder=5)
    ax.text((x0 + x1) / 2, y + 0.12, text, ha="center", va="bottom", fontsize=fs, color=color,
            fontweight="normal", clip_on=False, zorder=5)


def lag10_fine(Pl, arm):
    eps = list(range(16)); out = []
    for s in SEEDS:
        a = Pl[(Pl.family == "block") & (Pl.arm == arm) & (Pl.seed == s)].set_index("epoch").loc[eps]
        b = Pl[(Pl.family == "block") & (Pl.arm == "sham") & (Pl.seed == s)].set_index("epoch").loc[eps]
        out.append(F.behind(a.fine.values, b.fine.values, eps)[10])
    return np.array(out)


# a
def panel_a(ax, P):
    F.panel_a(ax, P)
    for t in list(ax.texts):            # drop the line-end labels of fig5_panels; replaced below
        if t.get_position()[0] > 15:
            t.remove()
    ys = {"sham": 22.5, "gradjoint": 14.5, "gradsc": 4.5}
    for arm, tc in (("sham", GREY_T), ("gradjoint", BLUE[1]), ("gradsc", BLUE_T)):
        m = np.mean([P[(P.arm == arm) & (P.seed == s) & (P.epoch == 10)].n_sc.values[0] for s in SEEDS])
        ax.scatter([10], [m], s=16, color=tc if arm != "sham" else F.GREYS[1], edgecolor="white", lw=0.5, zorder=6)
        ax.text(15.6, ys[arm], f"{m:.0f}", fontsize=9, color=tc, ha="left", va="center", fontweight="bold",
                clip_on=False)
        ax.text(17.4, ys[arm], F.ARMS[arm][0], fontsize=6, color=tc, ha="left", va="center", clip_on=False)
    ax.text(15.6, 31, "at epoch 10", fontsize=6, color=GREY_T, ha="left", va="center", clip_on=False)
    for ln in ax.lines:                     # quieter seed traces
        if abs(ln.get_linewidth() - 0.45) < 1e-6:
            ln.set_alpha(0.13)
    title(ax, "Under the coarse block the coarse structure barely forms")


# b
def panel_b(ax, Pl):
    arms = ("gradenergy", "gradjoint", "gradsc")
    L = {a: lag10_fine(Pl, a) for a in arms}
    xs = {"gradenergy": 0, "gradjoint": 1, "gradsc": 2.25}
    for i, s in enumerate(SEEDS):
        ax.plot([xs[a] for a in arms], [L[a][i] for a in arms], color=GREYS[2], lw=0.55, alpha=0.9, zorder=1)
    for a in arms:
        c = F.col(a, AMBER)
        ax.scatter(np.full(5, xs[a]), L[a], s=9, color="white", edgecolor=c, lw=0.7, zorder=2)
        m = L[a].mean()
        ax.scatter([xs[a]], [m], s=60, color=c, edgecolor="white", lw=0.7, zorder=3)
        if a == "gradsc":
            ax.text(xs[a] + 0.2, m, f"{m:.1f}", fontsize=8, ha="left", va="center", color=c, fontweight="bold")
        else:
            ax.text(xs[a], L[a].max() + 0.45, f"{m:.1f}", fontsize=7, ha="center", va="bottom", color=AMBER_T)
    d_j, d_e = L["gradsc"] - L["gradjoint"], L["gradsc"] - L["gradenergy"]
    bracket(ax, xs["gradjoint"], xs["gradsc"], 9.2, f"+{d_j.mean():.1f} epochs, {int((d_j > 0).sum())}/5 seeds")
    ax.text(xs["gradenergy"], 3.9, f"coarse block\n+{d_e.mean():.1f} against this,\n{int((d_e > 0).sum())}/5 seeds",
            fontsize=6, color=GREY_T, ha="center", va="top", linespacing=1.0)
    ax.axhline(0, color=GREYS[3], lw=0.6, zorder=0)
    ax.text(2.6, 0.1, "no block", fontsize=6, color=GREY_T, ha="right", va="bottom")
    ax.set_xticks([xs[a] for a in arms])
    ax.set_xticklabels(["energy-\nmatched", "matched\nblock", "coarse\nblock"], fontsize=6.5, linespacing=0.95)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlim(-0.45, 2.75); ax.set_ylim(-0.3, 11.2); ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_ylabel("Fine-class accuracy, epochs\nbehind no block at epoch 10")
    title(ax, "Specificity: withholding the coarse structure costs\nhalf again as much as matched damage")
    rows = [dict(arm=a, arm_label=F.ARMS[a][0], epoch=10, seed=s, fine_accuracy_epochs_behind_no_block=v)
            for a in arms for s, v in zip(SEEDS, L[a])]
    rows += [dict(arm="gradsc minus " + r, arm_label="extra cost", epoch=10, seed=s, fine_accuracy_epochs_behind_no_block=v)
             for r, D in (("gradjoint", d_j), ("gradenergy", d_e)) for s, v in zip(SEEDS, D)]
    ns.source_data(pd.DataFrame(rows), NAME, "b")
    print(f"b epoch 10: energy {L['gradenergy'].mean():.2f}, matched {L['gradjoint'].mean():.2f}, coarse "
          f"{L['gradsc'].mean():.2f}; coarse-matched {d_j.mean():.2f} ({(d_j > 0).sum()}/5), coarse-energy "
          f"{d_e.mean():.2f} ({(d_e > 0).sum()}/5)")


# c
def panel_c(ax, Pl, R):
    tmp = plt.figure(); tax = tmp.add_subplot(111)
    XY = F.panel_c(tax, Pl, R)            # per block, per seed: (coarse lag, fine lag) at epoch 10
    plt.close(tmp)
    blocks = [("coarse block", "coarse", XY["coarse block"]), ("random grouping", "coarse", XY["random grouping"]),
              ("matched block", "fine", XY["matched block"]), ("energy-matched", "fine", XY["energy-matched"])]
    bw, step, split = 0.36, 1.15, 0.45
    rng = np.random.default_rng(5)
    rows = []
    for k, (lab, target, M) in enumerate(blocks):
        x0 = k * step + (split if k >= 2 else 0)
        coarse, fine = M[:, 0], M[:, 1]
        order = [("coarse", coarse, BLUE[1]), ("fine", fine, AMBER[1])] if target == "coarse" else \
                [("fine", fine, AMBER[1]), ("coarse", coarse, BLUE[1])]
        for j, (lv, v, c) in enumerate(order):
            x = x0 + j * (bw + 0.06)
            ax.bar(x, v.mean(), width=bw, color=c, alpha=0.95 if j == 0 else 0.55, lw=0, zorder=1)
            ax.scatter(x + rng.uniform(-0.07, 0.07, 5), v, s=4, color="white", edgecolor=DARK, lw=0.35, zorder=3)
            if v.mean() > 2.5:
                ax.text(x, 0.35, f"{v.mean():.1f}", fontsize=6.5, ha="center", va="bottom", color="white",
                        fontweight="bold", zorder=4)
            else:
                ax.text(x + 0.2, max(v.max(), 0) + 0.3, f"{v.mean():.1f}", fontsize=6.5, ha="center", va="bottom",
                        color=DARK, zorder=4)
        tgt, oth = order[0][1].mean(), order[1][1].mean()
        share = max(oth, 0) / tgt
        xa, xb = x0, x0 + bw + 0.06
        top = max(tgt, order[0][1].max(), order[1][1].max())
        arr = FancyArrowPatch((xa, top + 0.6), (xb, max(order[1][1].max(), oth, 0) + 0.5),
                              connectionstyle="arc3,rad=-0.4", arrowstyle="-|>", mutation_scale=7,
                              color=DARK if target == "coarse" else GREY_T, lw=0.9, zorder=4)
        ax.add_patch(arr)
        lab_s = "≈0%" if share < 0.05 else f"{100 * share:.0f}%"
        ax.text((xa + xb) / 2, top + 1.9, lab_s, fontsize=8.5 if target == "coarse" else 7.5,
                ha="center", va="bottom", color=DARK if target == "coarse" else GREY_T, fontweight="bold")
        ax.text((xa + xb) / 2, -2.9, lab.replace(" ", "\n", 1), fontsize=6, ha="center", va="top", color=DARK,
                linespacing=0.95)
        rows += [dict(block=lab, targets=("coarse structure" if target == "coarse" else "fine-grained learning"),
                      seed=s, coarse_lag_epochs=a, fine_lag_epochs=b) for s, (a, b) in zip(SEEDS, M)]
        rows.append(dict(block=lab, targets=target, seed="mean", coarse_lag_epochs=coarse.mean(),
                         fine_lag_epochs=fine.mean(), share_crossing_over=share))
        print(f"c {lab}: target {tgt:.2f}, other {oth:.2f}, crosses {100 * share:.0f}%")
    xm = 2 * step - 0.5 * step + split / 2 + 0.25
    ax.axvline(xm, color="#D1D5DB", lw=0.6, zorder=0)
    tr = ax.get_xaxis_transform()
    ax.text(0.5 * step + 0.2, 1.0, "aimed at the coarse structure", fontsize=6, ha="center", va="bottom",
            color=BLUE_T, transform=tr)
    ax.text(2.5 * step + split + 0.2, 1.0, "aimed at fine-grained learning", fontsize=6, ha="center", va="bottom",
            color=AMBER_T, transform=tr)
    ax.axhline(0, color=GREYS[3], lw=0.6, zorder=0)
    ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
    ax.set_xlim(-0.35, 3 * step + split + bw + 0.35); ax.set_ylim(-4.3, 13.5); ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_ylabel("Epochs behind no block\nat epoch 10")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE[1], label="coarse structure"), Patch(color=AMBER[1], label="fine-class accuracy")],
              fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, handlelength=1.0,
              handleheight=0.8, borderaxespad=0.1, columnspacing=1.5)
    title(ax, "Asymmetry: most of the coarse delay reaches fine-grained learning,\nlittle of the fine delay "
              "reaches the coarse structure")
    ns.source_data(pd.DataFrame(rows), NAME, "c")


# d
def panel_d(ax, P):
    F.panel_d(ax, P)
    for t in list(ax.texts):
        if t.get_text().endswith("%"):
            t.set_fontsize(7)
    W_ = P[P.epoch.between(1, 10)]
    sh = {a: (100 * W_[W_.arm == a].groupby("seed").n_cl_confined.sum()
              / W_[W_.arm == a].groupby("seed").n_cl.sum()).loc[SEEDS] for a in ("gradjoint", "gradsc")}
    lower = int((sh["gradsc"].values < sh["gradjoint"].values).sum())
    bracket(ax, 1, 2, 48.5, f"halved, {lower}/5 seeds", h=1.5, fs=6.5)
    n = {a: W_[W_.arm == a].groupby("seed").n_cl.mean().loc[SEEDS].mean() for a in ("sham", "gradjoint", "gradsc")}
    ax.set_xticklabels([f"no\nblock\n({n['sham']:.0f})", f"matched\nblock\n({n['gradjoint']:.0f})",
                        f"coarse\nblock\n({n['gradsc']:.0f})"], fontsize=6.5, linespacing=0.95)
    ax.set_ylim(0, 55)
    TITLES[ax] = "Nesting: class features still form,\nbut outside their superclass"
    print("d class features per checkpoint (in brackets):", {k: round(v, 1) for k, v in n.items()})


def main():
    ns.apply()
    for f in (ns.FINAL_OUT / "source_data").glob(f"{NAME}_*.csv"):
        f.unlink()
    P = pd.read_csv(F.RESULTS / "RDT_pace" / "long.csv")
    Pl = pd.read_csv(F.RESULTS / "PLACEMENT_curve" / "long.csv")
    R = pd.read_csv(F.RESULTS / "RDT_completion" / "rdt_completion_long.csv")
    fig = plt.figure(figsize=(W * ns.MM, H * ns.MM))
    mm = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
    y1, h1 = 62, 33
    y2, h2 = 13, 31
    a, b = mm(15, y1, 62, h1), mm(118, y1, 58, h1)
    c = mm(15, y2, 95, h2 - 3)
    d = mm(136, y2, 34, h2 - 3)
    panel_a(a, P)
    F.TITLES.clear()
    panel_b(b, Pl)
    panel_c(c, Pl, R)
    F.TITLES.clear()
    panel_d(d, P)
    t1, t2 = H - 1.5, y2 + h2 + 6
    for ax, letter, x, y in ((a, "a", 0.5, t1), (b, "b", 100, t1), (c, "c", 0.5, t2), (d, "d", 118, t2)):
        fig.text(x / W, y / H, letter, fontsize=8, fontweight="bold", va="top", ha="left")
        fig.text((x + 3.8) / W, (y - 0.25) / H, TITLES[ax], fontsize=7, va="top", ha="left", linespacing=1.1)
    # journal limit: text 5-7 pt, panel letters 8 pt
    from matplotlib.text import Text
    for t in fig.findobj(Text):
        txt = t.get_text().strip()
        if t.get_visible() and txt and t.get_fontsize() > 7.01 and not (len(txt) == 1 and t.get_fontsize() <= 8.01):
            t.set_fontsize(7)
    print("saved", ns.save_final(fig, NAME, 180, strict=False))


if __name__ == "__main__":
    main()
