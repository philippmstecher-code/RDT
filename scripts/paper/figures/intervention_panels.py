#!/usr/bin/env python3
"""Corrupting the coarse structure slows fine-grained learning (label noise and acquisition blocking): shared panel
functions and loaders, imported by interventions/ed6.py (Extended Data Fig. 6), interventions/ed7.py (Extended Data
Fig. 7), routing/routing_panels.py, suppression_panels.py and extended/blocking_routing_panels.py.
Run directly, the script draws the eight panels below.

a  label noise: superclass probe (blue) and fine-class accuracy (amber) under within- (solid) and between-superclass
   noise (dashed), clean training thin grey; mean and t-based 95% CI over 5 seeds.
b  epochs the between-noise networks are behind the within-noise control, probe and accuracy (effect_units._lag).
c  RDT under between-superclass noise: class-selective features per run, unbent, bent earlier and newly bent
   (RDT_noise_fit/features.csv, lineage.csv).
d  acquisition blocking: superclass assignment (fresh read-out) and fine-class accuracy with the superclass directions
   blocked (dashed) and the jointly matched block (solid); block window shaded (GRADJOINT/long.csv).
e  epochs behind the jointly matched and the random-direction block (effect_units.blocking_lags).
f  RDT: Ab-E and Di-E events per epoch around the end of the block (RDT_blocking/rdt_blocking_long.csv).
g  per superclass, share of the learning signal blocked against the fine-class change, epochs 1-3
   (GRADBLOCK/superclass_gradblock.csv).
h  superclass vs fine-grained learning lost, one point per manipulation (effect_units.synthesis).
Colour = level measured (blue superclass, amber fine-class); line style = arm.

Usage: python scripts/paper/figures/intervention_panels.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nmi_style as ns  # noqa: E402
import noise_lib as L  # noqa: E402
from fig_common import RESULTS  # noqa: E402
from suppression_panels import suppression_first3  # noqa: E402

NEUTRAL_HUE = "#374151"
PROBE_HUE = ns.INDEX["SSI"]   # #3B82F6, superclass quantity
ACC_HUE = ns.INDEX["CSI"]     # #F59E0B, fine-class quantity
SSI_C, CSI_C = ns.INDEX["SSI"], ns.INDEX["CSI"]
RED = ns.RED_600
LS = {"clean": ns.COND_LS["standard"], "between_sc": ns.COND_LS["between_sc"], "within_sc": ns.COND_LS["within_sc"]}
ORDER = ("clean", "within_sc", "between_sc")
NAME = "fig5"
SD_DIR = RESULTS / "figures" / "final" / "source_data"
R3_Y, R3_H = 9, 52
R2_Y, R2_H = R3_Y + R3_H + 22, 40
W, H = 180, 120
ZERO_TXT = "#6B7280"
CLEAN_LW, CLEAN_C = 0.8, "#9CA3AF"     # clean training: thin grey reference line
SEED_LW, SEED_ALPHA = 0.4, 0.35        # single seeds (within, between)
NOISE_LW, NOISE_MS = 1.4, 2.8


def mm_axes(fig, x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def mstr(s):
    return s.replace("-", L.MINUS)


def f1(x):
    return mstr(f"{x:.1f}")


# Label-noise design schematics
def panel_a(ax, w_mm, h_mm):
    """Schematic in mm units (1 data unit = 1 mm in x and y)."""
    ax.set_xlim(0, w_mm)
    ax.set_ylim(0, h_mm)
    ax.set_axis_off()
    dark, arrow_c = "#374151", "#4B5563"
    ax.text(0, h_mm - 1.5, "30% of training labels replaced by another class, corrupted once and kept fixed",
            fontsize=7, color=dark, va="center")
    sc = [("Aquatic mammals", ("beaver", "dolphin", "otter"), 24, "#EFF6FF", "#BFDBFE"),
          ("Flowers", ("orchid", "rose", "tulip"), 67, "#EFF6FF", "#BFDBFE")]
    bw, bh = 40, 6.2
    xw = {}
    for _, words, x0, _, _ in sc:
        for k, w in enumerate(words):
            xw[w] = x0 + 7 + 13 * k
    rows = [("Within-\nsuperclass", h_mm - 10.5, "within", ["dolphin", "otter"], "beaver → dolphin or otter",
             "superclass label still correct"),
            ("Between-\nsuperclass", h_mm - 20.5, "between", ["orchid", "rose", "tulip"],
             "beaver → orchid, rose or tulip", "superclass label wrong too")]
    for lab, y, kind, targets, d1, d2 in rows:
        ax.text(0, y, lab, fontsize=6.5, color=dark, va="center", ha="left", linespacing=1.0)
        for name, words, x0, fc, ec in sc:
            ax.add_patch(FancyBboxPatch((x0, y - bh / 2), bw, bh, boxstyle="round,pad=0,rounding_size=0.6",
                                        fc=fc, ec=ec, lw=0.5, zorder=1))
            ax.text(x0 + bw / 2, y + 1.65, name, fontsize=6, color=dark, ha="center", va="center")
            for w in words:
                ax.text(xw[w], y - 0.95, w, fontsize=6, color=dark, ha="center", va="center")
            if kind == "within":
                ax.add_patch(FancyBboxPatch((x0 - 0.5, y - bh / 2 - 0.45), bw + 1.0, bh + 0.9,
                                            boxstyle="round,pad=0,rounding_size=0.8", fc="none", ec=SSI_C, lw=0.9,
                                            ls=(0, (3, 1.6)), zorder=2))
        if kind == "between":
            ax.add_patch(FancyBboxPatch((23.2, y - bh / 2 - 0.6), 84.6, bh + 1.2,
                                        boxstyle="round,pad=0,rounding_size=0.8", fc="none", ec=RED, lw=0.9,
                                        ls=(0, (3, 1.6)), zorder=2))
            ax.plot([65.5, 65.5], [y - 2.3, y + 2.3], color=RED, lw=1.0, zorder=3, solid_capstyle="butt")
        for t in targets:
            dx = xw[t] - xw["beaver"]
            rad = 0.45 if dx < 15 else (0.22 if dx < 30 else (0.12 if dx < 50 else 0.09))
            ax.annotate("", xy=(xw[t], y - 1.95), xytext=(xw["beaver"], y - 1.95),
                        arrowprops=dict(arrowstyle="-|>", color=arrow_c, lw=0.6, mutation_scale=4.5, shrinkA=0.5,
                                        shrinkB=0.5, connectionstyle=f"arc3,rad={rad}"), zorder=4)
        ax.text(113, y + 1.1, d1, fontsize=6.5, color=dark, va="center")
        ax.text(113, y - 1.25, d2, fontsize=6.5, color=RED if kind == "between" else SSI_C, va="center")


def panel_a_compact(ax, w_mm, h_mm):
    """Narrow design schematic beside b and c (mm units). Each noise type: its line style (the key of b, c), two
    superclasses with two classes each and the label replacements; clean training at the bottom."""
    ax.set_xlim(0, w_mm)
    ax.set_ylim(0, h_mm)
    ax.set_axis_off()
    dark, arrow_c = "#374151", "#4B5563"
    ax.text(0, h_mm - 1.2, "30% of labels replaced once\nby another class, kept fixed", fontsize=6.5, color=dark,
            va="top", linespacing=1.1)
    bw, bh, gap = 20.5, 6.4, 2.0
    sc = [("Aquatic mammals", ("beaver", "otter"), 1.0, "#EFF6FF", "#BFDBFE"),
          ("Flowers", ("rose", "tulip"), 1.0 + bw + gap, "#EFF6FF", "#BFDBFE")]
    xw = {w: x0 + 5.0 + 10.5 * k for _, words, x0, _, _ in sc for k, w in enumerate(words)}
    rows = [("within_sc", "Within-superclass noise", h_mm - 15.5, ["otter"], "superclass label still correct", SSI_C),
            ("between_sc", "Between-superclass noise", h_mm - 33.0, ["rose", "tulip"], "superclass label wrong too",
             RED)]
    for kind, lab, y, targets, cap, cc in rows:
        ax.plot([1.0, 7.5], [y + 6.2, y + 6.2], color=L.ERR, ls=LS[kind], lw=1.2)
        ax.text(9.0, y + 6.2, lab, fontsize=6.5, color=dark, va="center")
        for name, words, x0, fc, ec in sc:
            ax.add_patch(FancyBboxPatch((x0, y - bh / 2), bw, bh, boxstyle="round,pad=0,rounding_size=0.6",
                                        fc=fc, ec=ec, lw=0.5, zorder=1))
            ax.text(x0 + bw / 2, y + 1.7, name, fontsize=6, color=dark, ha="center", va="center")
            for w in words:
                ax.text(xw[w], y - 0.9, w, fontsize=6, color=dark, ha="center", va="center")
            if kind == "within_sc":
                ax.add_patch(FancyBboxPatch((x0 - 0.4, y - bh / 2 - 0.4), bw + 0.8, bh + 0.8,
                                            boxstyle="round,pad=0,rounding_size=0.8", fc="none", ec=SSI_C, lw=0.9,
                                            ls=(0, (3, 1.6)), zorder=2))
        if kind == "between_sc":
            ax.add_patch(FancyBboxPatch((0.4, y - bh / 2 - 0.5), 2 * bw + gap + 1.2, bh + 1.0,
                                        boxstyle="round,pad=0,rounding_size=0.8", fc="none", ec=RED, lw=0.9,
                                        ls=(0, (3, 1.6)), zorder=2))
            ax.plot([1.0 + bw + gap / 2] * 2, [y - 2.3, y + 2.3], color=RED, lw=1.0, zorder=3, solid_capstyle="butt")
        for t in targets:
            dx = xw[t] - xw["beaver"]
            rad = 0.5 if dx < 12 else (0.25 if dx < 25 else 0.16)
            ax.annotate("", xy=(xw[t], y - 1.9), xytext=(xw["beaver"], y - 1.9),
                        arrowprops=dict(arrowstyle="-|>", color=arrow_c, lw=0.6, mutation_scale=4.5, shrinkA=0.5,
                                        shrinkB=0.5, connectionstyle=f"arc3,rad={rad}"), zorder=4)
        ax.text(1.0, y - bh / 2 - 2.6, cap, fontsize=6.3, color=cc, va="top")
    ax.plot([1.0, 7.5], [h_mm - 49.5] * 2, color=CLEAN_C, ls=LS["clean"], lw=CLEAN_LW + 0.2)
    ax.text(9.0, h_mm - 49.5, "Clean training (reference)", fontsize=6.5, color=dark, va="center")


def key_bc(fig, x_mm, y_mm):
    """Line-style key for b and c (probe blue / accuracy amber are named by the y axes)."""
    h = [Line2D([], [], ls=LS["clean"], color=CLEAN_C, lw=CLEAN_LW + 0.2, label="Clean training"),
         Line2D([], [], ls=LS["within_sc"], color=L.ERR, lw=1.2, label="Within-superclass noise"),
         Line2D([], [], ls=LS["between_sc"], color=L.ERR, lw=1.2, label="Between-superclass noise")]
    fig.legend(handles=h, loc="center left", bbox_to_anchor=(x_mm / W, y_mm / H), ncol=3, fontsize=6.5,
               frameon=False, handlelength=2.8, columnspacing=2.2, borderaxespad=0, borderpad=0)


def load():
    P = L.static_probe()                              # {arm: seeds x epochs (0, 5, ..., 30)}, %
    A = L.static_accuracy()                           # {arm: seeds x 31}, %
    return P, A


def courses(ax, x, Y, hue, markevery=None):
    """Within and between prominent (mean, t-based 95% CI band over seeds, thin per-seed lines); clean training as
    a thin grey reference line (mean only)."""
    ax.plot(x, Y["clean"].mean(0), color=CLEAN_C, ls=LS["clean"], lw=CLEAN_LW, zorder=2)
    for a in ("within_sc", "between_sc"):
        for r in Y[a]:
            ax.plot(x, r, color=hue, ls="-", lw=SEED_LW, alpha=SEED_ALPHA, zorder=3)
        L.traj(ax, x, Y[a], hue, LS[a], band="ci", lw=NOISE_LW, ms=NOISE_MS, markevery=markevery, zorder=5)


def gap_label(ax, x, ymid, r, big=7):
    """Effect text right of the epoch-30 marker: mean, then the paired 95% CI."""
    ax.text(x, ymid, mstr(f"{r['mean']:+.1f} pp"), fontsize=big, color="black", ha="left", va="bottom",
            clip_on=False)
    ax.text(x, ymid - 0.4, f"95% CI\n{f1(r['ci_lo'])} to {f1(r['ci_hi'])}", fontsize=6, color=L.ERR, ha="left",
            va="top", clip_on=False, linespacing=1.1)


def style_course(ax, ylab, ylim, yt, title):
    ax.set_xlim(-1, 30.8)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_ylim(*ylim)
    ax.set_yticks(yt)
    ax.set_ylabel(ylab)
    ax.set_xlabel("Epoch", labelpad=1.5)
    ax.set_title(title, fontsize=7, pad=4)


# Superclass probe
def panel_b(ax, P, eff):
    xp = P["clean"].columns.values.astype(float)
    Y = {a: P[a].values for a in ORDER}
    courses(ax, xp, Y, PROBE_HUE)
    pw, pb = Y["within_sc"][:, -1].mean(), Y["between_sc"][:, -1].mean()
    xa = 32.0
    ax.annotate("", xy=(xa, pb), xytext=(xa, pw), annotation_clip=False,
                arrowprops=dict(arrowstyle="<|-|>", color=RED, lw=1.0, mutation_scale=5, shrinkA=0, shrinkB=0),
                zorder=5)
    r = eff["probe_epoch30"]
    assert abs(r["mean"] - (pb - pw)) < 1e-9
    gap_label(ax, xa + 1.0, (pw + pb) / 2, r)
    style_course(ax, "Linear superclass probe (%)", (20, 72), [20, 40, 60], "Superclass structure is damaged")
    print(f"[5b] epoch 30 probe: clean {Y['clean'][:, -1].mean():.2f}, within {pw:.2f}, between {pb:.2f}; "
          f"between - within {r['mean']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}], {r['n_seeds_below_zero']}/5 < 0")
    rows = [dict(table="superclass probe per seed (%)", seed=s, epoch=int(e),
                 **{f"probe_{a}": P[a].loc[s, e] for a in ORDER}) for s in L.SEEDS for e in xp.astype(int)]
    for e in xp.astype(int):
        rows.append(dict(table="superclass probe, mean, s.e.m. and 95% CI half-width over seeds (%)", seed="mean", epoch=int(e),
                         **{f"probe_{a}": P[a][e].mean() for a in ORDER},
                         **{f"sem_{a}": L.sem(P[a][e].values) for a in ORDER},
                         **{f"ci95_halfwidth_{a}": P[a][e].mean() - L.t_ci(P[a][e].values)[1] for a in ORDER}))
    rows.append(dict(table="epoch 30, between - within (pp, paired 95% CI over seeds)", seed="effect", epoch=30,
                     mean=r["mean"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], n_seeds_below_zero=r["n_seeds_below_zero"],
                     **{f"seed_{s}": r[f"seed_{s}"] for s in L.SEEDS}))
    ns.source_data(pd.DataFrame(rows), NAME, "b")


# Fine-class accuracy
def panel_c(ax, A, eff):
    xa = np.arange(31)
    courses(ax, xa, A, ACC_HUE, markevery=5)
    aw, ab = A["within_sc"][:, 30].mean(), A["between_sc"][:, 30].mean()
    xb = 32.0
    ax.annotate("", xy=(xb, ab), xytext=(xb, aw), annotation_clip=False,
                arrowprops=dict(arrowstyle="|-|,widthA=0.22,widthB=0.22", color=RED, lw=1.0, shrinkA=0, shrinkB=0),
                zorder=5)
    r = eff["fine_acc_epoch30"]
    assert abs(r["mean"] - (ab - aw)) < 1e-9
    gap_label(ax, xb + 1.0, (aw + ab) / 2, r)
    ax.axhline(30, color="#D1D5DB", lw=0.6, ls=":", zorder=0)
    dl = eff["delay_to_30pct_accuracy"]
    ax.text(30.3, 28.2, f"30% accuracy reached\n{f1(dl['mean'])} epochs later with\nbetween-superclass noise\n"
                        f"(95% CI {f1(dl['ci_lo'])}–{f1(dl['ci_hi'])})",
            fontsize=6, ha="right", va="top", color="black", linespacing=1.15)
    style_course(ax, "Fine-class validation accuracy (%)", (0, 62), [0, 20, 40, 60], "Fine-grained learning slows")
    print(f"[5c] epoch 30 accuracy: clean {A['clean'][:, 30].mean():.2f}, within {aw:.2f}, between {ab:.2f}; "
          f"between - within {r['mean']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]; delay to 30% "
          f"{dl['mean']:+.2f} [{dl['ci_lo']:+.2f}, {dl['ci_hi']:+.2f}] epochs")
    rows = [dict(table="fine-class validation accuracy per seed (%)", seed=s, epoch=int(e),
                 **{f"acc_{a}": A[a][i, e] for a in ORDER}) for i, s in enumerate(L.SEEDS) for e in xa]
    for e in xa:
        rows.append(dict(table="fine-class validation accuracy, mean, s.e.m. and 95% CI half-width over seeds (%)", seed="mean",
                         epoch=int(e), **{f"acc_{a}": A[a][:, e].mean() for a in ORDER},
                         **{f"sem_{a}": L.sem(A[a][:, e]) for a in ORDER},
                         **{f"ci95_halfwidth_{a}": A[a][:, e].mean() - L.t_ci(A[a][:, e])[1] for a in ORDER}))
    for key, lab in (("fine_acc_epoch30", "epoch 30, between - within (pp, paired 95% CI over seeds)"),
                     ("delay_to_30pct_accuracy", "delay to 30% accuracy, between - within (epochs, paired 95% CI)")):
        q = eff[key]
        rows.append(dict(table=lab, seed="effect", epoch=30 if key == "fine_acc_epoch30" else None, mean=q["mean"],
                         ci_lo=q["ci_lo"], ci_hi=q["ci_hi"], n_seeds_below_zero=q["n_seeds_below_zero"],
                         **{f"seed_{s}": q[f"seed_{s}"] for s in L.SEEDS}))
    ns.source_data(pd.DataFrame(rows), NAME, "c")


# Label noise: accuracy differences and RDT
def panel_c_acc(ax):
    """Fixed labels, fine-class accuracy, between- minus within-superclass noise: 5-epoch means with 95% CI over seeds
    and, set apart, similarity-matched minus within-superclass noise at epoch 30. Values read from the stored
    label-noise Source Data (source_data_ed/ed6_g.csv, ed6_h.csv)."""
    G = pd.read_csv(SD_DIR.parent / "source_data_ed" / "ed6_g.csv")
    B = G[G.table == "accuracy between - within, 5-epoch bin (95% CI)"].copy()
    Hh = pd.read_csv(SD_DIR.parent / "source_data_ed" / "ed6_h.csv")
    m30 = Hh[(Hh.measure == "acc_30") & (Hh.contrast == "matched - within")].iloc[0]
    assert len(B) == 6 and (B.ci_hi < 0).all()
    lo_e = B.epoch.str.split("-").str[0].astype(int).values
    hi_e = B.epoch.str.split("-").str[1].astype(int).values
    x = (lo_e + hi_e) / 2
    ax.axhline(0, color=ZERO_TXT, ls="--", lw=0.6, zorder=0)
    ax.errorbar(x, B["mean"], yerr=[B["mean"] - B.ci_lo, B.ci_hi - B["mean"]], fmt="o", ms=3.6, color=CSI_C,
                mec=CSI_C, ecolor=CSI_C, elinewidth=0.9, capsize=2.0, capthick=0.9, zorder=3)
    ax.plot(x, B["mean"], color=CSI_C, lw=0.8, ls="--", zorder=2)
    xm = 38.0
    ax.bar(xm, m30["mean"], width=4.2, color="white", edgecolor=CSI_C, hatch="////", lw=0.9, zorder=2)
    ax.errorbar(xm, m30["mean"], yerr=[[m30["mean"] - m30.ci_lo], [m30.ci_hi - m30["mean"]]], fmt="none",
                ecolor=L.ERR, elinewidth=0.9, capsize=2.0, capthick=0.9, zorder=3)
    ax.axvline(33.0, ymax=0.76, color="#D1D5DB", lw=0.6, zorder=0)
    ax.text(15.5, 0.35, "between − within", fontsize=6, color=ZERO_TXT, ha="center", va="bottom")
    ax.text(xm, 0.35, "matched\n− within", fontsize=6, color=ZERO_TXT, ha="center", va="bottom", linespacing=1.0)
    ax.text(xm, m30.ci_lo - 0.25, mstr(f"{m30['mean']:.1f}"), fontsize=6.5, color=L.ERR, ha="center", va="top")
    ax.set_xlim(-1, 41.5)
    ax.set_xticks([0, 10, 20, 30, xm])
    ax.set_xticklabels(["0", "10", "20", "30", "30"])
    ax.set_ylim(-6.2, 1.6)
    ax.set_yticks([-6, -4, -2, 0])
    ax.set_xlabel("Epoch", labelpad=1.5)
    ax.set_ylabel("Fine-class accuracy,\ndifference (pp)")
    ax.set_title("Lower fine-class\naccuracy", fontsize=7, pad=3)
    rows = [dict(contrast="between - within (fixed labels, p = 0.3)", window=r.epoch, mean=r["mean"], ci_lo=r.ci_lo,
                 ci_hi=r.ci_hi) for _, r in B.iterrows()]
    rows.append(dict(contrast="similarity-matched - within (fixed labels, p = 0.3)", window="epoch 30", mean=m30["mean"],
                     ci_lo=m30.ci_lo, ci_hi=m30.ci_hi, n_seeds_below_zero=m30.n_seeds_below_zero))
    ns.source_data(pd.DataFrame(rows), NAME, "c_accuracy")
    print(f"[5c] accuracy bins {B['mean'].round(2).tolist()}; matched - within {m30['mean']:.2f}")


def panel_c_rdt(ax):
    """Between- vs within-superclass noise, % change per seed (RDT, one common threshold set per seed = the clean
    run's stored thresholds; scripts/paper/noise_rdt_analysis.py): class-selective features (% of alive features,
    census layers, epoch 30) and class-differentiation events (Di-E, all layers, epochs 0-30)."""
    import rdt_side_by_side as RS
    S = L.SEEDS
    items = [("class-\nselective\nfeatures", "n_cls", None), ("Di-E\nevents", None, "di_h")]
    rng = np.random.default_rng(12)
    rows, tops = [], []
    for i, (lab, cat, ev) in enumerate(items):
        if cat:
            b = np.array([RS.census("between", s, cat) for s in S]); w = np.array([RS.census("within", s, cat) for s in S])
        else:
            b = np.array([RS.events("between", s, ev) for s in S], float)
            w = np.array([RS.events("within", s, ev) for s in S], float)
        v = 100 * (b / w - 1)
        m, lo, hi = L.t_ci(v)
        ax.bar(i, m, width=0.6, color=CSI_C, alpha=ns.BAR_ALPHA_CORE, lw=0, zorder=1)
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=9, color=L.DOT, lw=0, zorder=3)
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=L.ERR, elinewidth=0.9, capsize=2.5,
                    capthick=0.9, zorder=4)
        ax.text(i, max(hi, v.max()) + 4, mstr(f"{m:+.0f}%"), fontsize=7, color=L.ERR, ha="center", va="bottom")
        tops.append(max(hi, v.max()))
        full = {"n_cls": "class-selective features", "di_h": "class- differentiation events (Di-E)"}[cat or ev]
        rows.append(dict(measure=full, contrast="between - within (% change)", mean=m, ci_lo=lo,
                         ci_hi=hi, n_seeds_above_zero=int((v > 0).sum()),
                         **{f"seed_{sd}": x for sd, x in zip(S, v)},
                         **{f"between_seed_{sd}": x for sd, x in zip(S, b)},
                         **{f"within_seed_{sd}": x for sd, x in zip(S, w)}))
        print(f"[5c] {lab.replace(chr(10), ' ')}: {m:+.1f}% [{lo:+.1f}, {hi:+.1f}], {(v > 0).sum()}/5 > 0")
    ax.axhline(0, color=ZERO_TXT, ls="--", lw=0.6, zorder=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([it[0] for it in items], fontsize=6, linespacing=1.0)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-10, max(tops) + 22)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Between − within (% change)")
    ax.set_title("More class-selective\nfeatures and events", fontsize=7, pad=3, loc="right")
    ns.source_data(pd.DataFrame(rows), NAME, "c_rdt")


# Routing schematic
def kept_in_superclass():
    """Stored change in fine-class accuracy (pp) on images assigned to the targeted superclass both before and after
    the removal (ROUTING/routing_within.json, 'd_fixed'; mean and 95% CI over seeds per architecture)."""
    J = json.loads((ROUTE / "routing_within.json").read_text())["summary"]
    rows = [dict(arch=a, measure="fine-class accuracy change, images kept in their superclass (pp)", arm=arm,
                 mean=100 * J[a][f"{arm}_d_fixed"][0], ci_lo=100 * J[a][f"{arm}_d_fixed"][1],
                 ci_hi=100 * J[a][f"{arm}_d_fixed"][2], n_seeds=J[a]["n_seeds"])
            for a in ARCHS for arm in ("ab", "ctrl")]
    return pd.DataFrame(rows)


def routing_schematic(ax, w_mm, h_mm, K):
    """Three-step mechanism (mm units): (1) the shared coarse structure is learned first, (2) it routes each image to
    its superclass region (assignment), (3) class distinctions are refined within that region. Cue (red): without the
    directions, images go to the wrong superclass and misrouted images supply more of the learning signal (superclass-blocked
    minus energy-matched, epochs 1-3; GRADBLOCK/routing_summary.json). K (images kept in their superclass) stays in the
    Source Data of e."""
    import json as _json
    ax.set_xlim(0, w_mm)
    ax.set_ylim(0, h_mm)
    ax.set_axis_off()
    dark = "#374151"
    box = dict(boxstyle="round,pad=0.35,rounding_size=0.25", fc="white", ec="#9CA3AF", lw=0.6)
    yr = h_mm - 3.2
    x0, x1, x2 = 12.0, 40.5, 66.5
    ax.text(x0, yr, "Shared coarse structure", fontsize=6.5, color=dark, ha="center", va="center", bbox=box)
    ax.text(x1, yr, "Superclass region", fontsize=6.5, color=dark, ha="center", va="center", bbox={**box, "ec": SSI_C})
    ax.text(x2, yr, "Class within it", fontsize=6.5, color=dark, ha="center", va="center", bbox={**box, "ec": CSI_C})
    arr = dict(arrowstyle="-|>", color=dark, lw=0.7, mutation_scale=5, shrinkA=0, shrinkB=0)
    ax.annotate("", xy=(x1 - 10.2, yr), xytext=(x0 + 13.6, yr), arrowprops=arr)
    ax.annotate("", xy=(x2 - 9.2, yr), xytext=(x1 + 10.2, yr), arrowprops=arr)
    for x, lab in ((x0, "1  learned first"), (x1, "2  routes the image"), (x2, "3  refined within")):
        ax.text(x, yr - 3.3, lab, fontsize=6, color=ZERO_TXT, ha="center", va="center")
    r = _json.load(open(RESULTS / "GRADBLOCK" / "routing_summary.json"))["epochs_1_3|gradsc - gradenergy|energy_misrouted"]
    gain = 100 * r["mean_ci"][0]
    yo = 1.6
    ax.annotate("", xy=(x1 - 4.0, yo + 0.4), xytext=(x1 - 4.0, yr - 4.6),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.8, mutation_scale=5, shrinkA=0, shrinkB=0,
                                ls=(0, (2.5, 1.2))))
    ax.text(x1 - 2.6, yo + 2.6, f"without the directions: wrong superclass;\nmisrouted images supply {gain:.1f} points\n"
            f"more of the learning signal", fontsize=6, color=RED, ha="left", va="center", linespacing=1.05)
    print(f"[5e] mechanism schematic; misrouted share of the learning signal, superclass-blocked - energy-matched, "
          f"epochs 1-3: +{gain:.2f} pp ({r['n_positive']}/{len(r['seeds'])} seeds)")


# Suppression cost
def bars(ax, x, v, color, light, rng, width, dot_s=9):
    """Bar = mean, grey seed dots, dark 95% CI; light = hatched variant."""
    m, lo, hi = L.t_ci(v)
    if light:
        ax.bar(x, m, width=width, color=color, alpha=0.25, lw=0, zorder=1)
        ax.bar(x, m, width=width, facecolor="none", edgecolor=color, lw=0.8, hatch="//////", zorder=1)
    else:
        ax.bar(x, m, width=width, color=color, alpha=ns.BAR_ALPHA_CORE, lw=0, zorder=1)
    ax.scatter(x + rng.uniform(-0.2, 0.2, len(v)) * width, v, s=dot_s, color=L.DOT, lw=0, zorder=3)
    if len(v) >= 3:
        ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=L.ERR, elinewidth=0.9, capsize=2.5,
                    capthick=0.9, zorder=4)
    return m, lo, hi


REFS = (("sham", "vs sham"), ("random", "vs random\ndirections"))


def ref_axis(ax, off, lows, highs, ylab, title, top=1.2):
    """Shared grammar of d and e: x groups = reference (sham, random directions), two bars per group."""
    ax.axhline(0, color=ZERO_TXT, ls="--", lw=0.6, zorder=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r[1] for r in REFS], fontsize=6.5)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(min(lows) - 1.0, max(max(highs), 0) + top)
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=7, pad=3)


def panel_supp_cost(ax, name=NAME, letter="d"):
    """Suppression of the superclass directions in the last block, epochs 1-10: fine-class accuracy over the first three
    epochs against sham and against random directions (suppression_panels.suppression_first3; networks as trained, projection in
    place). The unprojected activations are tested with a fresh read-out (suppression_panels.panel_fresh), since the network's own
    classifier never received input along the removed directions."""
    S = suppression_first3().set_index("window").loc["last_block_1_10"]
    wid = 0.5
    rng = np.random.default_rng(4)
    rows, lows, highs = [], [], []
    for i, (ref, _) in enumerate(REFS):
        v = np.array([S[f"seed{sd}_minus_{ref}"] for sd in L.SEEDS], float)
        m, lo, hi = bars(ax, i, v, ACC_HUE, False, rng, wid)
        assert abs(m - S[f"superclass_minus_{ref}"]) < 1e-9
        y = min(lo, v.min()) - 0.3
        ax.text(i, y, f1(m), fontsize=6.5, color=L.ERR, ha="center", va="top")
        lows.append(y)
        highs.append(max(hi, v.max()))
        rows.append(dict(reference=ref, epochs="1-3", superclass_minus_reference=m, ci95_lo=lo, ci95_hi=hi,
                         n_negative=int((v < 0).sum()), **{f"seed{sd}": val for sd, val in zip(L.SEEDS, v)}))
    ref_axis(ax, 0, lows, highs, "Change in fine-class accuracy,\nfirst 3 epochs (pp)",
             "Suppressing the superclass\ndirections in epochs 1–10", top=1.2)
    ax.set_yticks([-6, -4, -2, 0])
    sd = pd.DataFrame(rows)
    print("[5d] " + "; ".join(f"{r.reference}: {r.superclass_minus_reference:+.2f} [{r.ci95_lo:+.2f}, "
                              f"{r.ci95_hi:+.2f}], {r.n_negative}/5 < 0" for r in sd.itertuples()))
    ns.source_data(sd, name, letter)
    ns.source_data(suppression_first3(), name, f"{letter}_first3_by_window")


def panel_supp_where(ax, name=NAME, letter="e"):
    """Where the suppression cost lies (first three epochs of the epochs 1-10 window): superclass assignment of the
    network's own predictions (blue; suppression_panels.factor_rows) and 5-way accuracy among the classes of each image's true
    superclass, network's own classifier (amber; ORACLE_within oracle_logits), vs sham and vs random directions."""
    import suppression_panels
    R = suppression_panels.factor_rows()
    R = R[R.window == "last_block_1_10"]
    O = pd.read_csv(RESULTS / "ORACLE_within" / "oracle_long.csv")
    O = O[O.epoch.isin([1, 2, 3])].groupby(["arm", "seed"]).oracle_logits.mean()
    off, wid = 0.2, 0.36
    rng = np.random.default_rng(6)
    rows, lows, highs = [], [], []
    for i, (ref, _) in enumerate(REFS):
        r = R[R.reference == ref].sort_values("seed")
        oc = np.array([O[("supp_sc", sd)] - O[("sham" if ref == "sham" else "supp_rand", sd)] for sd in r.seed])
        for v, hue, x, what in ((r.d_superclass_accuracy_pp.values, SSI_C, i - off, "superclass assignment"),
                                (oc, CSI_C, i + off, "class given the true superclass")):
            m, lo, hi = bars(ax, x, v, hue, False, rng, wid)
            y = min(lo, v.min()) - 0.3
            ax.text(x, y, f1(m), fontsize=6.5, color=L.ERR, ha="center", va="top")
            lows.append(y)
            highs.append(max(hi, v.max()))
            rows.append(dict(reference=ref, measure=what, epochs="1-3", superclass_minus_reference=m, ci95_lo=lo,
                             ci95_hi=hi, n_negative=int((v < 0).sum()),
                             **{f"seed{sd}": val for sd, val in zip(r.seed, v)}))
    ref_axis(ax, off, lows, highs, "Change, first\n3 epochs (pp)", "Where the cost lies", top=4.2)
    ax.set_yticks([-10, -5, 0])
    h = [Patch(facecolor=SSI_C, alpha=ns.BAR_ALPHA_CORE, lw=0, label="superclass assignment"),
         Patch(facecolor=CSI_C, alpha=ns.BAR_ALPHA_CORE, lw=0, label="class given the true superclass")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=1, handlelength=1.3,
              handleheight=0.9, borderaxespad=0.1, labelspacing=0.25, fontsize=6.0, frameon=False)
    sd = pd.DataFrame(rows)
    print("[5e] " + "; ".join(f"{r.reference} / {r.measure}: {r.superclass_minus_reference:+.2f} "
                              f"[{r.ci95_lo:+.2f}, {r.ci95_hi:+.2f}], {r.n_negative}/5 < 0" for r in sd.itertuples()))
    ns.source_data(sd, name, letter)
    R = R.copy()
    R["d_class_given_true_superclass_pp"] = [O[("supp_sc", s_)] - O[("sham" if ref == "sham" else "supp_rand", s_)]
                                             for ref, s_ in zip(R.reference, R.seed)]
    ns.source_data(R, name, f"{letter}_factorisation_per_seed")
    summ = []
    for col in ("d_superclass_accuracy_pp", "d_class_given_true_superclass_pp"):
        for ref, _ in REFS:
            v = R[R.reference == ref].sort_values("seed")[col].values
            m, lo, hi = L.t_ci(v)
            summ.append(dict(window="last_block_1_10", epochs="1-3", measure=col, reference=ref, mean=m, ci95_lo=lo,
                             ci95_hi=hi, n_below_zero=int((v < 0).sum()), n=len(v),
                             seeds=" ".join(f"{z:.2f}" for z in v)))
    ns.source_data(pd.DataFrame(summ), name, f"{letter}_summary")


# Removing one superclass's superclass-selective features
ROUTE = RESULTS / "ROUTING"
ARCHS = ("ResNet-18", "ViT-Small", "CCT-7")
TF_FINAL = {"ViT-Small": "8", "CCT-7": "9"}          # checkpoint labels of the final epoch (100 and 50)


def removal_rows():
    """All checkpoints: specificity S and its matched control (pp), per architecture, seed and checkpoint."""
    rows, ngrp = [], {}
    for fn in ("routing_resnet_clean_ep5-10-15-20-25.json", "routing_resnet_clean_ep30.json"):
        for key, v in json.loads((ROUTE / fn).read_text()).items():
            seed, ep = (int(t) for t in key.split("|"))
            r = v["true"]["part2"]
            ngrp[(seed, ep)] = int(sum(n > 0 for n in r["n_ab_per_group"]))
            rows.append(dict(arch="ResNet-18", method="subtraction", seed=seed, checkpoint=str(ep), epoch=ep,
                             final=False, S_pp=100 * r["S"], S_ctrl_pp=100 * r["S_ctrl"],
                             n_superclasses_with_features=ngrp[(seed, ep)],
                             n_superclass_selective_features=int(sum(r["n_ab_per_group"]))))
    for seed, r in json.loads((ROUTE / "routing_resnet_projection.json").read_text()).items():
        rows.append(dict(arch="ResNet-18", method="projection", seed=int(seed), checkpoint="30", epoch=30, final=True,
                         S_pp=100 * r["S"], S_ctrl_pp=100 * r["S_ctrl"],
                         n_superclasses_with_features=ngrp[(int(seed), 30)],
                         n_superclass_selective_features=int(r["n_ab"])))
    epochs = {"CCT-7": {k: v["epoch"] for k, v in json.loads((ROUTE / "routing_cct7.json").read_text())["runs"].items()},
              "ViT-Small": {"1": 1, "4": 5, "8": 100}}
    for key, r in json.loads((ROUTE / "routing_tf.json").read_text()).items():
        lane, ck = key.split("|")
        arch, seed = lane.split("__")[1], int(lane.split("__s")[-1])
        ep = epochs[arch][key] if arch == "CCT-7" else epochs[arch][ck]
        rows.append(dict(arch=arch, method="projection", seed=seed, checkpoint=ck, epoch=ep, final=ck == TF_FINAL[arch],
                         S_pp=100 * r["S"], S_ctrl_pp=100 * r["S_ctrl"],
                         n_superclasses_with_features=int(sum(n > 0 for n in r["n_ab_per_group"])),
                         n_superclass_selective_features=int(r["n_ab"])))
    return pd.DataFrame(rows).sort_values(["arch", "method", "seed", "epoch"]).reset_index(drop=True)


def panel_removal(ax, name=NAME, letter="e"):
    """Removing one superclass's superclass-selective features: specificity S and matched control per architecture,
    final checkpoints (drawn in the Extended Data figures)."""
    R = removal_rows()
    F = R[R.final]
    rng = np.random.default_rng(6)
    off, wid = 0.2, 0.36
    tops = []
    for i, arch in enumerate(ARCHS):
        g = F[F.arch == arch].sort_values("seed")
        col = ns.ARCH[arch]
        for x, v, light in ((i - off, g.S_pp.values, False), (i + off, g.S_ctrl_pp.values, True)):
            m, lo, hi = bars(ax, x, v, col, light, rng, wid)
            top = max(v.max(), hi if len(v) >= 3 else v.max())
            lab = f1(m) if not light else (f1(m) if m < -0.05 else "0.0")
            ax.text(x, max(top, 0) + 0.6, lab, fontsize=7 if not light else 6.5, color=L.ERR, ha="center",
                    va="bottom")
            tops.append(max(top, 0))
    ax.axhline(0, color=ZERO_TXT, ls="--", lw=0.6, zorder=0)
    ax.set_xticks(range(len(ARCHS)))
    ax.set_xticklabels([f"{a}\n(n = {int((F.arch == a).sum())})" for a in ARCHS], fontsize=6.5)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlim(-0.55, len(ARCHS) - 0.45)
    ax.set_ylim(-2, max(tops) + 14)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_ylabel("Extra accuracy loss on the\ntargeted superclass (pp)")
    h = [Patch(facecolor="#9CA3AF", alpha=ns.BAR_ALPHA_CORE, lw=0, label="Superclass-selective features (S)"),
         Patch(facecolor="white", edgecolor="#6B7280", hatch="//////", lw=0.8, label="Matched control")]
    ax.legend(handles=h, loc="upper left", handlelength=1.3, handleheight=0.9, borderaxespad=0.1,
              labelspacing=0.3, fontsize=6.5, frameon=False)
    for a in ARCHS:
        g = F[F.arch == a]
        print(f"[5e] {a}: S {g.S_pp.min():.1f}-{g.S_pp.max():.1f} (mean {g.S_pp.mean():.2f}, n = {len(g)}); control "
              f"{g.S_ctrl_pp.min():.2f} to {g.S_ctrl_pp.max():.2f}; superclasses with features "
              f"{g.n_superclasses_with_features.min()}-{g.n_superclasses_with_features.max()}")
    sub = R[(R.arch == "ResNet-18") & (R.method == "subtraction") & (R.epoch == 30)]
    print(f"[5e] ResNet-18 subtraction, epoch 30: S {sub.S_pp.min():.1f}-{sub.S_pp.max():.1f}")
    K = kept_in_superclass()
    sd = pd.concat([R.assign(table="specificity S per checkpoint", shown_in_panel=R.final),
                    K.assign(table="images kept in their superclass (schematic; mean, 95% CI over seeds)")],
                   ignore_index=True)
    for c in ("seed", "epoch", "n_superclasses_with_features", "n_superclass_selective_features", "n_seeds"):
        sd[c] = sd[c].astype("Int64")
    ns.source_data(sd, name, letter)
    return K


# Routing of the learning signal (signal_routing_panels.py)
def routing_panels():
    import signal_routing_panels as SR
    return SR


def panel_sr_e(ax, name=NAME, letter="g"):
    SR = routing_panels()
    SR.panel_i(ax)
    S = SR.D[SR.D.arm == "sham"][["seed", "epoch", "image_share_misrouted", "signal_share_misrouted",
                                  "image_share_routed", "signal_share_routed", "image_share_correct",
                                  "signal_share_correct"]]
    ns.source_data(S.assign(arm="sham (standard training)"), name, letter)


def panel_sr_f(ax, name=NAME, letter="h"):
    SR = routing_panels()
    SR.panel_ii(ax)
    S = SR.D[SR.D.arm == "sham"][["seed", "epoch", "code_share_misrouted", "code_share_routed", "code_share_correct",
                                  "random_share_misrouted", "random_share_routed", "random_share_correct",
                                  "random_share_all"]]
    ns.source_data(S.assign(arm="sham (standard training)"), name, letter)


def panel_sr_i(ax, name=NAME, letter="i"):
    SR = routing_panels()
    SR.panel_iii(ax)
    D = SR.D
    M = D[D.epoch.isin([1, 2, 3])].groupby(["arm", "seed"]).mean(numeric_only=True)
    rows = []
    for arm in ("gradsc", "gradrg"):
        for col in ("image_share_misrouted", "signal_share_misrouted"):
            v = 100 * (M.loc[arm][col] - M.loc["gradenergy"][col]).loc[list(L.SEEDS)].values
            m, lo, hi = L.t_ci(v)
            rows.append(dict(contrast=f"{arm} - gradenergy", window="epochs 1-3", measure=col + " (pp)", mean=m,
                             ci_lo=lo, ci_hi=hi, n_seeds_above_zero=int((v > 0).sum()),
                             **{f"seed_{sd}": x_ for sd, x_ in zip(L.SEEDS, v)}))
    ns.source_data(pd.DataFrame(rows), name, letter)


# Acquisition blocking
GB = RESULTS / "GRADJOINT" / "long.csv"   # GRADBLOCK arms + jointly matched and redrawn-random blocks (gradjoint_analysis.py)
# colour = intervention arm (ns.ARM)
GB_ARMS = [("sham", "Sham", ns.ARM["sham"], LS["clean"], CLEAN_LW, False),
           ("gradrand", "Random directions", ns.ARM["random_directions"], ns.COND_LS["random"], 1.0, True),
           ("gradenergy", "Energy-matched", ns.ARM["energy"], "--", 1.1, True),
           ("gradrg", "Random-grouping directions", ns.ARM["random_grouping"], "-.", 1.0, True),
           ("gradsc", "Superclass directions", ns.ARM["superclass"], "-", NOISE_LW, True),
           ("gradjoint", "Jointly matched", ns.ARM["joint"], "-", 1.1, True),
           ("gradrandfresh", "Random directions, redrawn", ns.ARM["random_redrawn"], "-.", 1.0, True)]
GB_EPOCHS = np.arange(1, 31)  # no validation record at the branch checkpoint (epoch 0)
GB_WINDOWS = [("1–3", range(1, 4))]


def gb_long():
    D = pd.read_csv(GB)
    D = D[D.arm.isin([a[0] for a in GB_ARMS])]
    assert set(D.seed) == set(L.SEEDS) and D.groupby(["arm", "seed"]).size().eq(30).all(), "incomplete lanes"
    return D


def panel_f(ax, D, name=NAME, letter="e", arms=None):
    ax.axvspan(0, 10, color="#F3F4F6", lw=0, zorder=0)
    ax.text(5, 58.5, "learning signal\nwithheld", fontsize=6, color=ZERO_TXT, ha="center", va="top", linespacing=1.05)
    x = GB_EPOCHS
    rows = []
    for arm, lab, col, ls, lw, band in [a for a in GB_ARMS if arms is None or a[0] in arms]:
        Y = D[D.arm == arm].pivot(index="seed", columns="epoch", values="val_accuracy").loc[list(L.SEEDS), x].values
        if band:
            L.traj(ax, x, Y, col, ls, band="ci", lw=lw, ms=0, zorder=5 if arm == "gradsc" else 4, label=lab,
                   band_alpha=0.18 if arm != "gradrand" else 0.10)
        else:
            ax.plot(x, Y.mean(0), color=col, ls=ls, lw=lw, zorder=3, label=lab)
        for i, sd in enumerate(L.SEEDS):
            rows += [dict(table="fine-class validation accuracy per seed (%)", arm=arm, seed=sd, epoch=int(e),
                          value=Y[i, j]) for j, e in enumerate(x)]
        m, lo, hi = L.t_ci(Y)
        rows += [dict(table="fine-class validation accuracy, mean and 95% CI over seeds (%)", arm=arm,
                      seed="mean", epoch=int(e), value=m[j], ci_lo=lo[j], ci_hi=hi[j]) for j, e in enumerate(x)]
    h, lb = ax.get_legend_handles_labels()
    order = [lb.index(n) for n in ("Superclass directions", "Random-grouping directions", "Energy-matched",
                                   "Random directions", "Sham") if n in lb]
    ax.legend([h[i] for i in order], [lb[i] for i in order], loc="lower right", fontsize=6.0, frameon=False,
              handlelength=2.4, borderaxespad=0.2, labelspacing=0.25)
    ax.set_xlim(-0.5, 30.5)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_ylim(3, 60)
    ax.set_yticks([10, 20, 30, 40, 50, 60])
    ax.set_xlabel("Epoch", labelpad=1.5)
    ax.set_ylabel("Fine-class validation\naccuracy (%)")
    ax.set_title("Withholding the learning signal\nfor the directions of a class grouping", fontsize=7, pad=3)
    P = D.pivot_table(index=["seed", "epoch"], columns="arm", values="val_accuracy")
    for a_, b_ in (("gradsc", "gradenergy"), ("gradrg", "gradenergy"), ("gradsc", "gradrg"), ("gradsc", "sham"),
                   ("gradrg", "sham"), ("gradrand", "sham"), ("gradenergy", "sham")):
        gap = (P[a_] - P[b_]).unstack("seed")
        for lab, eps in (("1-3", range(1, 4)), ("11-15", range(11, 16)), ("30", [30])):
            v = gap.loc[list(eps)].mean(0).loc[list(L.SEEDS)].values
            m, lo, hi = L.t_ci(v)
            what = f"epochs {lab}" if "-" in lab else f"epoch {lab}"
            rows.append(dict(table=f"{a_} - {b_}, {what} (pp, paired 95% CI over seeds)",
                             arm=f"{a_} - {b_}", seed="effect", mean=m, ci_lo=lo, ci_hi=hi,
                             n_seeds_below_zero=int((v < 0).sum()), **{f"seed_{sd}": x_ for sd, x_ in zip(L.SEEDS, v)}))
            print(f"[5f] {a_} - {b_}, {what}: {m:+.2f} [{lo:+.2f}, {hi:+.2f}], {(v < 0).sum()}/5 < 0")
    ns.source_data(pd.DataFrame(rows), name, letter)


def panel_g(ax, D, name=NAME, letter="f"):
    rng = np.random.default_rng(7)
    off, wid = 0.27, 0.4
    rows, lows = [], []
    for i, (wl, eps) in enumerate(GB_WINDOWS):
        O = pd.read_csv(RESULTS / "ORACLE_within" / "oracle_long.csv")
        M = D[D.epoch.isin(list(eps))].groupby(["arm", "seed"])[["fresh_assignment"]].mean().join(
            O[O.epoch.isin(list(eps))].groupby(["arm", "seed"])[["oracle_ridge"]].mean())
        for key, lab, col, x in (("fresh_assignment", "superclass assignment", SSI_C, i - off),
                                 ("oracle_ridge", "class given the true superclass", CSI_C, i + off)):
            v = (M.loc["gradsc"][key] - M.loc["gradenergy"][key]).loc[list(L.SEEDS)].values
            m, lo, hi = bars(ax, x, v, col, False, rng, wid)
            y = min(lo, v.min()) - 0.3
            ax.text(x, y, f1(m), fontsize=6.5, color=L.ERR, ha="center", va="top")
            lows.append(y)
            rows.append(dict(window=f"epochs {wl}", measure=lab, contrast="superclass blocked - energy-matched",
                             mean=m, ci_lo=lo, ci_hi=hi, n_seeds_below_zero=int((v < 0).sum()),
                             **{f"seed_{sd}": val for sd, val in zip(L.SEEDS, v)}))
    ax.axhline(0, color=ZERO_TXT, ls="--", lw=0.6, zorder=0)
    ax.set_xticks([-off, off])
    ax.set_xticklabels(["superclass\nassignment", "class given\ntrue superclass"], fontsize=6.0)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(min(lows) - 1.4, 2.0)
    ax.set_yticks([-6, -4, -2, 0])
    ax.set_ylabel("Superclass minus energy-\nmatched, epochs 1–3 (pp)")
    ax.set_title("Where the deficit lies", fontsize=7, pad=3)
    sd = pd.DataFrame(rows)
    print("[5g] " + "; ".join(f"{r.window} {r.measure}: {r.mean:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}], "
                              f"{r.n_seeds_below_zero}/5 < 0" for r in sd.itertuples()))
    ns.source_data(sd, name, letter)


BLOCK_REFS = (("gradenergy", "vs energy-\nmatched"), ("gradrand", "vs random\ndirections"))


def panel_block_where(ax, D, name=NAME, letter="e", arm="gradsc"):
    """Where the blocking cost lies, epochs 1-3: one fresh 100-way ridge read-out of the stored pooled activations
    (forward pass unchanged): superclass assignment (blue; GRADBLOCK final_long fresh_assignment) and 5-way accuracy
    among the classes of each image's true superclass (amber; ORACLE_within oracle_ridge), blocked arm minus
    energy-matched and minus random-direction blocking."""
    O = pd.read_csv(RESULTS / "ORACLE_within" / "oracle_long.csv")
    M = D[D.epoch.isin([1, 2, 3])].groupby(["arm", "seed"])[["fresh_assignment"]].mean().join(
        O[O.epoch.isin([1, 2, 3])].groupby(["arm", "seed"])[["oracle_ridge"]].mean())
    off, wid = 0.2, 0.36
    rng = np.random.default_rng(7)
    rows, lows, highs = [], [], []
    for i, (ref, _) in enumerate(BLOCK_REFS):
        for key, lab, hue, x in (("fresh_assignment", "superclass assignment", SSI_C, i - off),
                                 ("oracle_ridge", "class given the true superclass", CSI_C, i + off)):
            v = (M.loc[arm][key] - M.loc[ref][key]).loc[list(L.SEEDS)].values
            assert np.isfinite(v).all(), (arm, ref, key)
            m, lo, hi = bars(ax, x, v, hue, False, rng, wid)
            y = min(lo, v.min()) - 0.3
            ax.text(x, y, f1(m), fontsize=6.5, color=L.ERR, ha="center", va="top")
            lows.append(y)
            highs.append(max(hi, v.max()))
            rows.append(dict(arm=arm, reference=ref, window="epochs 1-3", measure=lab, mean=m, ci_lo=lo, ci_hi=hi,
                             n_seeds_below_zero=int((v < 0).sum()), **{f"seed_{sd}": val for sd, val in zip(L.SEEDS, v)}))
    ax.axhline(0, color=ZERO_TXT, ls="--", lw=0.6, zorder=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r[1] for r in BLOCK_REFS], fontsize=6.5)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(min(lows) - 1.0, max(max(highs), 0) + 4.2)
    ax.set_yticks([-6, -4, -2, 0])
    ax.set_ylabel("Change, epochs 1–3 (pp)")
    ax.set_title("Where the deficit lies", fontsize=7, pad=3)
    h = [Patch(facecolor=SSI_C, alpha=ns.BAR_ALPHA_CORE, lw=0, label="superclass assignment"),
         Patch(facecolor=CSI_C, alpha=ns.BAR_ALPHA_CORE, lw=0, label="class given the true superclass")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=1, handlelength=1.3,
              handleheight=0.9, borderaxespad=0.1, labelspacing=0.25, fontsize=6.0, frameon=False)
    sd = pd.DataFrame(rows)
    print(f"[{name}{letter}] " + "; ".join(f"{r.reference} / {r.measure}: {r.mean:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}], "
                                       f"{r.n_seeds_below_zero}/5 < 0" for r in sd.itertuples()))
    ns.source_data(sd, name, letter)


FOREST_ROWS = [  # (row in effect_units.forest, label, group)
    ("label noise, between - within", "Between- vs within-superclass noise", "Label noise, epochs 1–10"),
    ("label noise, similarity-matched - within", "Similarity-matched vs within", "Label noise, epochs 1–10"),
    ("label noise, between - within (superclass probe, epoch 5)", "Between vs within: superclass probe, epoch 5",
     "Label noise, epochs 1–10"),
    ("label noise, matched - within (superclass probe, epoch 5)", "Similarity-matched vs within: probe, epoch 5",
     "Label noise, epochs 1–10"),
    ("blocking, superclass - random directions", "Superclass vs random directions", "Withholding the learning signal, epochs 1–3"),
    ("blocking, superclass - energy-matched", "Superclass vs energy-matched", "Withholding the learning signal, epochs 1–3"),
    ("blocking, random grouping - energy-matched", "Random grouping vs energy-matched",
     "Withholding the learning signal, epochs 1–3"),
    ("blocking, stage-matched pair (layer2 to pooled)", "Stage-matched pair (layer2 to pooled)",
     "Withholding the learning signal, epochs 1–3"),
    ("suppression (use), superclass - random directions", "Superclass vs random directions",
     "Suppressing the directions (use), epochs 1–3"),
    ("suppression (use), superclass - sham", "Superclass vs sham", "Suppressing the directions (use), epochs 1–3"),
]


def panel_forest(ax, name=NAME, letter="f"):
    """Every intervention in one unit: % of the control's gain over chance lost (fine-class accuracy, amber; superclass
    probe, blue), mean and t-based 95% CI over 5 seeds, seed dots; effect_units.forest."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import effect_units as V
    F = V.forest(write=False).set_index("row")
    rng = np.random.default_rng(12)
    y, ticks, labels, groups, rows = 0.0, [], [], [], []
    last = None
    for key, lab, grp in FOREST_ROWS:
        if grp != last:
            y -= 0.55 if last is not None else 0
            groups.append((y, grp))
            y -= 1.0
            last = grp
        r = F.loc[key]
        v = np.array(r.per_seed, float)
        hue = SSI_C if r.unit == "probe" else CSI_C
        ax.scatter(v, y + rng.uniform(-0.18, 0.18, len(v)), s=7, color=L.DOT, lw=0, zorder=3)
        ax.errorbar(r["mean"], y, xerr=[[r["mean"] - r.ci[0]], [r.ci[1] - r["mean"]]], fmt="o", ms=3.6, color=hue,
                    mec=hue, ecolor=L.ERR, elinewidth=0.9, capsize=2.0, capthick=0.9, zorder=4)
        ax.text(4.5, y, f"{r['mean']:.0f}% ({r.n_below_zero}/5)".replace("-", "−"), fontsize=6.0, color=L.ERR,
                va="center", ha="left")
        ticks.append(y), labels.append(lab)
        rows.append(dict(group=grp, row=lab, unit=("superclass probe" if r.unit == "probe" else "fine-class accuracy"),
                         mean_pct=r["mean"], ci_lo=r.ci[0], ci_hi=r.ci[1], n_seeds_below_zero=r.n_below_zero,
                         **{f"seed_{sd}": x for sd, x in zip(L.SEEDS, v)}))
        y -= 1.0
    for gy, g in groups:
        ax.text(-41, gy - 0.35, g, fontsize=6.5, color="#111827", ha="left", va="center", clip_on=False)
    ax.axvline(0, color=ZERO_TXT, ls="--", lw=0.6, zorder=0)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=6.3)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(y + 0.4, 0.3)
    ax.set_xlim(-41, 14)
    ax.set_xticks([-40, -30, -20, -10, 0, 10])
    ax.set_xlabel("Change, % of the control's gain over chance", labelpad=1.5)
    ax.set_title("The cost of each manipulation in one unit", fontsize=7, pad=3)
    h = [Line2D([], [], marker="o", ls="none", ms=3.6, color=CSI_C, label="fine-class accuracy"),
         Line2D([], [], marker="o", ls="none", ms=3.6, color=SSI_C, label="superclass probe")]
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6.0, frameon=False, handletextpad=0.3,
              borderaxespad=0.0)
    ns.source_data(pd.DataFrame(rows), name, letter)


GREYBOX, GREYEDGE, DARK = "#F3F4F6", "#D1D5DB", "#374151"


def panel_a2(ax, w_mm, h_mm):
    """Label-noise design: neutral superclass boxes (blue headers), within- and between-superclass replacement, the
    broken superclass label marked with a red cross; line-style key of b and c."""
    ax.set_xlim(0, w_mm)
    ax.set_ylim(0, h_mm)
    ax.set_axis_off()
    ax.text(0, h_mm - 0.5, "30% of labels replaced\nonce, kept fixed", fontsize=6.5, color=DARK, va="top",
            linespacing=1.1)
    bw, bh, gap = 17.0, 7.6, 1.6
    boxes = [("Trees", ("oak", "maple"), 0.5), ("Flowers", ("rose", "tulip"), 0.5 + bw + gap)]
    xw = {w: x0 + 4.4 + 8.4 * k for _, words, x0 in boxes for k, w in enumerate(words)}
    rows = [("within_sc", "Within-superclass noise", h_mm - 19.5, "maple", "superclass label still correct", DARK),
            ("between_sc", "Between-superclass noise", h_mm - 41.0, "rose", "superclass label wrong too", RED)]
    for kind, lab, y, target, cap, cc in rows:
        ax.plot([0.5, 6.5], [y + 7.4] * 2, color=DARK, ls="-" if kind == "within_sc" else "--", lw=1.2)
        ax.text(7.8, y + 7.4, lab, fontsize=6.2, color=DARK, va="center")
        for name, words, x0 in boxes:
            ax.add_patch(FancyBboxPatch((x0, y - bh / 2), bw, bh, boxstyle="round,pad=0,rounding_size=0.6",
                                        fc=GREYBOX, ec=GREYEDGE, lw=0.5, zorder=1))
            ax.text(x0 + bw / 2, y + 2.0, name, fontsize=6.2, color=SSI_C, ha="center", va="center")
            for w in words:
                ax.text(xw[w], y - 1.3, w, fontsize=6, color=DARK, ha="center", va="center")
        dx = xw[target] - xw["oak"]
        ax.annotate("", xy=(xw[target], y - 2.6), xytext=(xw["oak"], y - 2.6),
                    arrowprops=dict(arrowstyle="-|>", color="#4B5563", lw=0.6, mutation_scale=4.5, shrinkA=0.8,
                                    shrinkB=0.8, connectionstyle=f"arc3,rad={0.55 if dx < 12 else 0.3}"), zorder=4)
        if kind == "between_sc":
            ax.text(0.5 + bw + gap / 2, y - 5.3, "×", fontsize=9, color=RED, ha="center", va="center", zorder=5)
        ax.text(0.5, y - bh / 2 - 3.6, cap, fontsize=6.2, color=cc, va="top")



def panel_b2(ax, P, eff):
    """Superclass probe; horizontal arrow: the between-noise epoch-30 value is where within-superclass noise stood
    about 16 epochs earlier (effect_units.label_noise, per-seed interpolation between 5-epoch checkpoints)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import effect_units as V
    xp = P["clean"].columns.values.astype(float)
    Y = {a: P[a].values for a in ORDER}
    courses(ax, xp, Y, PROBE_HUE)
    lost = V.label_noise()["probe_epochs_lost_vs_within"]
    pb = Y["between_sc"][:, -1].mean()
    ax.annotate("", xy=(30 - lost["mean"], pb), xytext=(30, pb),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.1, mutation_scale=6, shrinkA=0, shrinkB=1.5),
                zorder=6)
    ax.plot([30 - lost["mean"]] * 2, [pb - 1.6, pb + 1.6], color=RED, lw=0.8, zorder=6)
    ax.text(29.5, 44.0, f"{lost['mean']:.0f} epochs\nbehind", fontsize=7, color=RED, ha="right", va="center",
            fontweight="bold", linespacing=1.0)
    ends = {"within_sc": Y["within_sc"][:, -1].mean(), "clean": Y["clean"][:, -1].mean(), "between_sc": pb}
    labs = {"within_sc": "within", "clean": "clean", "between_sc": "between"}
    offs = {"within_sc": 1.6, "clean": -1.2, "between_sc": 0.0}
    for k, v in ends.items():
        ax.text(31.0, v + offs[k], labs[k], fontsize=6.2, color=CLEAN_C if k == "clean" else DARK, va="center",
                ha="left", clip_on=False)
    style_course(ax, "Linear superclass probe (%)", (20, 72), [20, 40, 60],
                 "Between-superclass noise holds the\nsuperclass probe 16 epochs back")
    r = eff["probe_epoch30"]
    rows = [dict(table="superclass probe per seed (%)", seed=s, epoch=int(e),
                 **{f"probe_{a}": P[a].loc[s, e] for a in ORDER}) for s in L.SEEDS for e in xp.astype(int)]
    rows.append(dict(table="epochs behind: 30 - epoch at which within-superclass noise reached the between-noise "
                           "epoch-30 probe (per seed, linear interpolation)", seed="effect", epoch=30,
                     mean=lost["mean"], ci_lo=lost["ci"][0], ci_hi=lost["ci"][1],
                     **{f"seed_{s}": v for s, v in zip(L.SEEDS, lost["per_seed"])}))
    rows.append(dict(table="epoch 30, between - within (pp)", seed="effect", epoch=30, mean=r["mean"], ci_lo=r["ci_lo"],
                     ci_hi=r["ci_hi"], n_seeds_below_zero=r["n_seeds_below_zero"]))
    ns.source_data(pd.DataFrame(rows), NAME, "b")
    print(f"[5b] epochs behind {lost['mean']:.2f} [{lost['ci'][0]:.2f}, {lost['ci'][1]:.2f}]")


def panel_c2(ax, A, eff):
    """Fine-class accuracy, between - within superclass noise per epoch (mean, t-based 95% CI band over seeds);
    bracket over epochs 0-10 with the mean difference and its share of the within-noise run's gain over chance."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import effect_units as V
    x = np.arange(31)
    Dd = A["between_sc"] - A["within_sc"]
    m, lo, hi = L.t_ci(Dd)
    ax.fill_between(x, lo, hi, color=CSI_C, alpha=0.22, lw=0, zorder=1)
    ax.plot(x, m, color=CSI_C, lw=1.4, zorder=3)
    ax.axhline(0, color="black", lw=1.0, zorder=2)
    r010 = eff["fine_acc_epochs0_10"]
    e010 = np.array([r010[f"seed_{s}"] for s in L.SEEDS])
    em = r010["mean"]
    pct = -V.forest(write=False).set_index("row").loc["label noise, between - within", "mean"]
    yb = -6.4
    ax.plot([0, 0, 10, 10], [yb + 0.5, yb, yb, yb + 0.5], color=DARK, lw=0.8)
    ax.text(0, yb - 0.3, mstr(f"{em:.1f} pp = {pct:.0f}% of early\ngain, {int((e010 < 0).sum())}/5 seeds"),
            fontsize=6.3, color=DARK, ha="left", va="top", linespacing=1.05)
    ax.set_xlim(-1, 30.8)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_ylim(-9.5, 2.0)
    ax.set_yticks([-6, -4, -2, 0])
    ax.set_xlabel("Epoch", labelpad=1.5)
    ax.set_ylabel("Fine-class accuracy,\nbetween − within (pp)")
    ax.set_title("Fine-class accuracy stays\nlower throughout", fontsize=7, pad=4)
    rows = [dict(table="between - within fine-class accuracy (pp)", epoch=int(e), mean=m[e], ci_lo=lo[e], ci_hi=hi[e],
                 **{f"seed_{s}": Dd[i, e] for i, s in enumerate(L.SEEDS)}) for e in x]
    rows.append(dict(table="epochs 0-10 mean (pp); % of the within run's gain over chance (epochs 1-10)", epoch=None,
                     mean=em, pct_of_gain=pct, n_seeds_below_zero=int((e010 < 0).sum())))
    ns.source_data(pd.DataFrame(rows), NAME, "c")
    print(f"[5c] epochs 0-10 {em:+.2f}, {pct:.1f}% of gain, {(e010 < 0).sum()}/5")


def panel_d_schematic(ax, w_mm, h_mm):
    """Blocking design and arm key: the forward pass is intact; the learning signal along the superclass directions
    is withheld in epochs 1-10."""
    ax.set_xlim(0, w_mm)
    ax.set_ylim(0, h_mm)
    ax.set_axis_off()
    y0 = h_mm - 13
    ax.add_patch(FancyBboxPatch((6, y0 - 4), 22, 8, boxstyle="round,pad=0,rounding_size=0.8", fc=GREYBOX,
                                ec=GREYEDGE, lw=0.5))
    ax.text(17, y0, "network", fontsize=6.2, color=DARK, ha="center", va="center")
    ax.annotate("", xy=(34, y0 + 2), xytext=(0.5, y0 + 2), arrowprops=dict(arrowstyle="-|>", color=DARK, lw=0.8,
                                                                         mutation_scale=5), zorder=0)
    ax.text(0.5, y0 + 6.2, "forward pass intact", fontsize=6.2, color=DARK, va="center")
    ax.annotate("", xy=(0.5, y0 - 2), xytext=(34, y0 - 2), arrowprops=dict(arrowstyle="-|>", color=ns.ARM["superclass"],
                                                                         lw=0.8, mutation_scale=5), zorder=0)
    ax.text(31, y0 - 2, "×", fontsize=10, color=RED, ha="center", va="center", zorder=5)
    ax.text(0.5, y0 - 6.6, "learning signal along the superclass\ndirections withheld, epochs 1–10", fontsize=6.2,
            color=ns.ARM["superclass"], va="top", linespacing=1.1)
    keys = [("Superclass directions", ns.ARM["superclass"], "-", NOISE_LW),
            ("Energy-matched", ns.ARM["energy"], "--", 1.1),
            ("Random directions", ns.ARM["random_directions"], ns.COND_LS["random"], 1.0),
            ("Sham", ns.ARM["sham"], LS["clean"], CLEAN_LW)]
    for i, (lab, c, ls, lw) in enumerate(keys):
        yy = 14.5 - 4.2 * i
        ax.plot([0.5, 6.5], [yy, yy], color=c, ls=ls, lw=lw)
        ax.text(7.8, yy, lab, fontsize=6.2, color=DARK, va="center")


def panel_d2(ax, D):
    """Blocking time course (panel_f, four arms) with the epoch-3 bracket (share of the controls' early gain) and the
    epoch-equivalent arrow (effect_units.blocking)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import effect_units as V
    panel_f(ax, D, name=NAME, letter="d", arms=("sham", "gradrand", "gradenergy", "gradsc"))
    ax.get_legend().remove()
    B = V.blocking()
    M = D[D.epoch == 3].groupby("arm").val_accuracy.mean()
    yb, yr = M["gradsc"], M["gradrand"]
    ax.plot([3.6, 4.1, 4.1, 3.6], [yb, yb, yr, yr], color=DARK, lw=0.8, zorder=7)
    g1 = B["gradsc_vs_gradrand"]["pct_of_control_gain_lost_epochs_1_3"]["mean"]
    g2 = B["gradsc_vs_gradenergy"]["pct_of_control_gain_lost_epochs_1_3"]["mean"]
    d1 = M["gradsc"] - M["gradrand"]
    ax.annotate(f"epochs 1–3: −3.4 / −3.1 pp\n= {min(g1, g2):.0f}–{max(g1, g2):.0f}% of early gain",
                xy=(4.1, (yb + yr) / 2), xytext=(9.5, 9.0), fontsize=6.3, color=DARK, va="center", ha="left",
                arrowprops=dict(arrowstyle="-", color=DARK, lw=0.5), zorder=7)
    eq = B["gradsc_vs_gradrand"]["epochs_behind_at_epoch_3"]["mean"]
    ax.annotate("", xy=(3 - eq, yb), xytext=(3, yb), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.0,
                                                                     mutation_scale=5, shrinkA=0, shrinkB=0), zorder=8)
    ax.annotate("one epoch behind", xy=(3 - eq / 2, yb - 0.3), xytext=(0.6, 5.0), fontsize=6.3, color=RED,
                ha="left", va="center", fontweight="bold", zorder=8,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.5))
    ax.set_title("Withholding the coarse learning signal costs a fifth of early learning", fontsize=7, pad=3)
    print(f"[5d] epoch 3 superclass - random {d1:+.2f}; gain lost {g1:.1f}/{g2:.1f}%; {eq:.2f} epochs behind")


def panel_e_schematic(ax, w_mm, h_mm):
    """Two-step routing strip above the bars; doubles as the colour key (blue = step 1, amber = step 2)."""
    ax.set_xlim(0, w_mm)
    ax.set_ylim(0, h_mm)
    ax.set_axis_off()
    y = h_mm / 2
    items = [(1.0, "image", DARK, GREYBOX), (13.5, "superclass", SSI_C, "#EFF6FF"), (29.5, "class", "#B45309", "#FEF3C7")]
    for x0, t, c, fc in items:
        wdt = 9.5 if t != "superclass" else 12.0
        ax.add_patch(FancyBboxPatch((x0, y - 2.6), wdt, 5.2, boxstyle="round,pad=0,rounding_size=0.6", fc=fc,
                                    ec=c, lw=0.6))
        ax.text(x0 + wdt / 2, y, t, fontsize=6.2, color=c, ha="center", va="center")
    for xa, xb, t, c in ((10.5, 13.5, "1", SSI_C), (25.5, 29.5, "2", "#B45309")):
        ax.annotate("", xy=(xb, y), xytext=(xa, y), arrowprops=dict(arrowstyle="-|>", color=c, lw=0.7,
                                                                   mutation_scale=4.5, shrinkA=0, shrinkB=0))


def panel_e2(ax, D):
    """Where the blocking deficit lies (fresh read-out; panel_block_where) with the routing share written on it."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import effect_units as V
    panel_block_where(ax, D, name=NAME, letter="e")
    ax.get_legend().remove()
    B = V.blocking()
    s1 = B["routing_share_vs_gradrand"]["pct_of_dlog_fine_in_assignment"]["mean"]
    s2 = B["routing_share_vs_gradenergy"]["pct_of_dlog_fine_in_assignment"]["mean"]
    ax.text(0.5, 0.97, f"{min(s1, s2):.0f}–{max(s1, s2):.0f}% of the deficit lies\nin superclass assignment",
            transform=ax.transAxes, fontsize=6.3, color=SSI_C, ha="center", va="top", fontweight="bold",
            linespacing=1.05)
    ax.set_ylim(ax.get_ylim()[0], 3.2)
    ax.set_title("")


def panel_e_mechanism(ax):
    """RDT mechanism: change vs sham in the share of class-selective features' activity that comes from images of
    other superclasses (pooled-layer SAE, sham thresholds; seeds 7, 42, 137), during the block (filled) and after
    release (hatched), per arm (supp5_a.csv)."""
    S = pd.read_csv(SD_DIR / "supp5_a.csv")
    arms = [("gradsc", "superclass", ns.ARM["superclass"]), ("gradrg", "random\ngrouping", ns.ARM["random_grouping"]),
            ("gradrand", "random\ndirections", ns.ARM["random_directions"]), ("gradenergy", "energy-\nmatched", ns.ARM["energy"])]
    rng = np.random.default_rng(14)
    wins = [("Block (epochs 1–10)", -0.19, False), ("After release (epochs 11–15)", 0.19, True)]
    rows = []
    for i, (arm, lab, col) in enumerate(arms):
        for win, dx, light in wins:
            r = S[(S.arm == arm) & (S.window == win)].iloc[0]
            v = np.array([r[f"off_superclass_change_seed_{sd}"] for sd in (7, 42, 137)])
            x = i + dx
            if light:
                ax.bar(x, r.off_superclass_change_mean, width=0.34, color=col, alpha=0.25, lw=0, zorder=1)
                ax.bar(x, r.off_superclass_change_mean, width=0.34, facecolor="none", edgecolor=col, lw=0.7,
                       hatch="//////", zorder=1)
            else:
                ax.bar(x, r.off_superclass_change_mean, width=0.34, color=col, alpha=ns.BAR_ALPHA_CORE, lw=0,
                       zorder=1)
            ax.scatter(x + rng.uniform(-0.06, 0.06, 3), v, s=6, color=L.DOT, lw=0, zorder=3)
            rows.append(dict(arm=arm, window=win, mean=r.off_superclass_change_mean, ci_lo=r.off_superclass_ci_lo,
                             ci_hi=r.off_superclass_ci_hi, **{f"seed_{sd}": x_ for sd, x_ in zip((7, 42, 137), v)}))
    ax.axhline(0, color=ZERO_TXT, ls="--", lw=0.6, zorder=0)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a[1].replace("-\n", "-").replace("\n", " ") for a in arms], fontsize=6, rotation=35, ha="right",
                       rotation_mode="anchor")
    ax.tick_params(axis="x", length=0, pad=2)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(-0.55, len(arms) - 0.45)
    ax.set_ylim(-4, 13)
    ax.set_yticks([0, 5, 10])
    ax.set_ylabel("Class features' activity\nfrom other superclasses,\nvs sham (pp)")
    ax.set_title("Class features respond\nto the wrong superclass", fontsize=7, pad=4)
    h = [Patch(facecolor="#9CA3AF", alpha=ns.BAR_ALPHA_CORE, lw=0, label="block"),
         Patch(facecolor="white", edgecolor="#6B7280", hatch="//////", lw=0.7, label="after release")]
    ax.legend(handles=h, loc="lower left", fontsize=6, frameon=False, handlelength=1.2, handleheight=0.8,
              borderaxespad=0.1, labelspacing=0.2)
    ns.rdt_tag(ax)
    ns.source_data(pd.DataFrame(rows), NAME, "e_mechanism")


FOREST2 = [  # (row key in effect_units.forest, label, colour, filled)
    ("label noise, between - within", "Label noise: between vs within", ns.NOISE["between"], True),
    ("blocking, superclass - random directions", "Blocking: vs random directions", ns.ARM["superclass"], True),
    ("blocking, superclass - energy-matched", "Blocking: vs energy-matched", ns.ARM["superclass"], True),
    ("blocking, random grouping - energy-matched", "Random-grouping block", ns.ARM["random_grouping"], True),
    ("suppression (use), superclass - random directions", "Suppression: vs random dir.", ns.ARM["superclass"], True),
    ("suppression (use), superclass - sham", "Suppression: vs sham", ns.ARM["superclass"], True),
    ("label noise, similarity-matched - within", "Similarity-matched noise", ns.NOISE["between"], False),
    ("blocking, stage-matched pair (layer2 to pooled)", "Stage-matched block", ns.ARM["superclass"], False),
]


def panel_f2(ax):
    """Every manipulation in one unit: % of the control's gain over chance (fine-class accuracy) lost; main tests
    filled on a pale band, matched controls open below."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import effect_units as V
    F = V.forest(write=False).set_index("row")
    rng = np.random.default_rng(15)
    rows, ys, labs = [], [], []
    y = 0.0
    for k, (key, lab, col, filled) in enumerate(FOREST2):
        if k == 6:
            y -= 0.7
        r = F.loc[key]
        v = np.array(r.per_seed, float)
        ax.scatter(v, y + rng.uniform(-0.17, 0.17, len(v)), s=6, color=L.DOT, lw=0, zorder=3)
        ax.errorbar(r["mean"], y, xerr=[[r["mean"] - r.ci[0]], [r.ci[1] - r["mean"]]], fmt="o", ms=4.0,
                    mfc=col if filled else "white", mec=col, mew=1.0, ecolor=L.ERR, elinewidth=0.9, capsize=2.0,
                    capthick=0.9, zorder=4)
        ys.append(y), labs.append(lab)
        rows.append(dict(row=lab, kind="main test" if filled else "matched control", mean_pct=r["mean"],
                         ci_lo=r.ci[0], ci_hi=r.ci[1], n_seeds_below_zero=r.n_below_zero,
                         **{f"seed_{sd}": x for sd, x in zip(L.SEEDS, v)}))
        y -= 1.0
    ax.axhspan(ys[5] - 0.5, ys[0] + 0.5, color="#FEF2F2", lw=0, zorder=0)
    ax.axvline(0, color=ZERO_TXT, ls="--", lw=0.6, zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels(labs, fontsize=6.2)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(ys[-1] - 0.6, ys[0] + 0.6)
    ax.set_xlim(-32, 10)
    ax.set_xticks([-30, -20, -10, 0])
    ax.set_xlabel("Change, % of the control's early gain", labelpad=1.5)
    main = [-F.loc[k, "mean"] for k, *_ in FOREST2[:6]]
    ctl = [-F.loc[k, "mean"] for k, *_ in FOREST2[6:]]
    xb = 3.0
    ax.plot([xb - 0.6, xb, xb, xb - 0.6], [ys[6] + 0.2, ys[6] + 0.2, ys[7] - 0.2, ys[7] - 0.2], color=DARK, lw=0.7)
    ax.text(xb + 0.6, (ys[6] + ys[7]) / 2, f"{min(ctl):.0f}–{max(ctl):.0f}%", fontsize=6.3, color=DARK, va="center")
    ax.set_title(f"Corrupting coarse structure costs\n{min(main):.0f}–{max(main):.0f}% of the early gain",
                 fontsize=7, pad=3)
    ns.source_data(pd.DataFrame(rows), NAME, "f")


# blue = superclass learning, amber = fine-grained learning
CTRL_LS, MANIP_LS = "-", "--"          # control solid, manipulation dashed (every row)


def _course(ax, x, Y, col, ls, lw, label=None, z=4):
    m, lo, hi = L.t_ci(Y)
    ax.fill_between(x, lo, hi, color=col, alpha=0.16 if ls == CTRL_LS else 0.22, lw=0, zorder=z - 1)
    ax.plot(x, m, color=col, ls=ls, lw=lw, zorder=z, label=label)
    return m


def panel_b3(ax, P):
    """Blue: superclass probe under within- (control, solid) and between-superclass noise (dashed); clean thin grey;
    arrow: 16 epochs behind."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import effect_units as V
    x = P["clean"].columns.values.astype(float)
    ax.plot(x, P["clean"].values.mean(0), color=CLEAN_C, lw=CLEAN_LW, zorder=2)
    _course(ax, x, P["within_sc"].values, SSI_C, CTRL_LS, 1.5)
    pb = _course(ax, x, P["between_sc"].values, SSI_C, MANIP_LS, 1.5)[-1]
    lost = V.label_noise()["probe_epochs_lost_vs_within"]
    F = V.forest(write=False).set_index("row")
    g5 = F.loc["label noise, between - within (superclass probe, epoch 5)"]
    i5 = list(x).index(5.0)
    yw, yb = P["within_sc"].values[:, i5].mean(), P["between_sc"].values[:, i5].mean()
    ax.plot([5.6, 6.2, 6.2, 5.6], [yw, yw, yb, yb], color=SSI_C, lw=0.8, zorder=7)
    ax.text(29.5, 33, f"−{-g5['mean']:.0f}% of the\ncontrol's gain\n(epoch 5)", fontsize=7.5, color=SSI_C, ha="right",
            va="center", fontweight="bold", linespacing=1.0)
    ax.text(30.8, P["within_sc"].values[:, -1].mean() + 1.6, "within", fontsize=6.2, color=SSI_C, va="center",
            ha="left", clip_on=False)
    ax.text(24.5, pb - 3.2, "between", fontsize=6.2, color=SSI_C, va="center", ha="center")
    ax.text(30.8, P["clean"].values[:, -1].mean() - 1.4, "clean", fontsize=6.2, color=CLEAN_C, va="center",
            ha="left", clip_on=False)
    style_course(ax, "Superclass probe (%)", (20, 72), [20, 40, 60], "Superclass learning falls behind")
    rows = [dict(table="superclass probe per seed (%)", seed=s, epoch=int(e),
                 **{f"probe_{a}": P[a].loc[s, e] for a in ORDER}) for s in L.SEEDS for e in x.astype(int)]
    rows.append(dict(table="% of the within-noise control's gain over chance (5%) lost, epoch 5", seed="effect",
                     mean=-g5["mean"], ci_lo=-g5["ci"][1], ci_hi=-g5["ci"][0],
                     **{f"seed_{s}": -v for s, v in zip(L.SEEDS, g5["per_seed"])}))
    rows.append(dict(table="epochs behind (per seed, interpolated)", seed="effect", mean=lost["mean"],
                     ci_lo=lost["ci"][0], ci_hi=lost["ci"][1], **{f"seed_{s}": v for s, v in zip(L.SEEDS, lost["per_seed"])}))
    ns.source_data(pd.DataFrame(rows), NAME, "b")


def panel_c3(ax, A):
    """Amber: fine-class accuracy under within- (solid) and between-superclass noise (dashed); arrows at 30% and 40%."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import effect_units as V
    x = np.arange(31)
    ax.plot(x, A["clean"].mean(0), color=CLEAN_C, lw=CLEAN_LW, zorder=2)
    _course(ax, x, A["within_sc"], CSI_C, CTRL_LS, 1.5)
    _course(ax, x, A["between_sc"], CSI_C, MANIP_LS, 1.5)
    Dl = pd.read_csv(V.OUT / "label_noise_delays.csv")
    rows = []
    for t in (30, 40):
        g = Dl[Dl.threshold == t]
        rows.append(dict(table="epochs to threshold", threshold_pct=t, epoch_within=g.epoch_within.mean(),
                         epoch_between=g.epoch_between.mean(), delay=(g.epoch_between - g.epoch_within).mean()))
    gf = V.forest(write=False).set_index("row").loc["label noise, between - within"]
    ax.axvspan(1, 10, color="#FEF3C7", alpha=0.6, lw=0, zorder=0)
    ax.text(19.5, 22, f"−{-gf['mean']:.0f}% of the\ncontrol's gain\n(epochs 1–10)", fontsize=7.5, color="#B45309",
            ha="center", va="top", fontweight="bold", linespacing=1.0)
    rows.append(dict(table="% of the within-noise control's gain over chance (1%) lost, epochs 1-10", mean=-gf["mean"],
                     ci_lo=-gf["ci"][1], ci_hi=-gf["ci"][0], **{f"seed_{s}": -v for s, v in zip(L.SEEDS, gf["per_seed"])}))
    ax.text(30.8, A["within_sc"][:, -1].mean() + 1.2, "within", fontsize=6.2, color="#B45309", va="center", clip_on=False)
    ax.text(30.8, A["between_sc"][:, -1].mean() - 1.0, "between", fontsize=6.2, color="#B45309", va="center", clip_on=False)
    ax.text(30.8, A["clean"][:, -1].mean(), "clean", fontsize=6.2, color=CLEAN_C, va="center", clip_on=False)
    style_course(ax, "Fine-class accuracy (%)", (0, 62), [0, 20, 40, 60], "Fine-grained learning arrives later")
    ns.source_data(pd.DataFrame(rows), NAME, "c")


def _gb(D, col):
    return {a: D[D.arm == a].pivot(index="seed", columns="epoch", values=col).loc[list(L.SEEDS)] for a in
            ("sham", "gradrand", "gradenergy", "gradsc")}


BLOCK_STYLE = {"gradenergy": ("-", 1.3), "gradrand": (":", 1.3), "gradsc": ("--", 1.6)}


def panel_e3(ax, D):
    """Blue: superclass assignment (fresh ridge read-out of the pooled activations) under blocking vs controls."""
    G = _gb(D, "fresh_assignment")
    x = G["sham"].columns.values
    ax.axvspan(0, 10, color="#F3F4F6", lw=0, zorder=0)
    ax.plot(x, G["sham"].values.mean(0), color=CLEAN_C, lw=CLEAN_LW, zorder=2)
    for a, (ls, lw) in BLOCK_STYLE.items():
        _course(ax, x, G[a].values, SSI_C, ls, lw)
    d = (G["gradsc"][[1, 2, 3]].mean(1) - G["gradenergy"][[1, 2, 3]].mean(1)).values
    yb, ye = G["gradsc"].values[:, 2].mean(), G["gradenergy"].values[:, 2].mean()
    ax.plot([3.6, 4.1, 4.1, 3.6], [yb, yb, ye, ye], color=SSI_C, lw=0.8, zorder=7)
    ax.annotate(mstr(f"{d.mean():.1f} points, epochs 1–3"), xy=(4.1, (yb + ye) / 2), xytext=(11.0, 22.0),
                fontsize=8, color=SSI_C, ha="left", va="center", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=SSI_C, lw=0.5, relpos=(1.0, 0.5)))
    ax.set_xlim(0, 30.5)
    ax.set_xticks([1, 10, 20, 30])
    ax.set_ylim(15, 80)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_xlabel("Epoch", labelpad=1.5)
    ax.set_ylabel("Superclass\nassignment (%)")
    ax.set_title("Superclass learning falls behind", fontsize=7, pad=4)
    ns.source_data(pd.concat({a: g for a, g in G.items()}, names=["arm"]).reset_index(), NAME, "e")


def panel_f3(ax, D):
    """Amber: fine-class accuracy under blocking vs controls; one epoch behind; share of early gain."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import effect_units as V
    G = _gb(D, "val_accuracy")
    x = G["sham"].columns.values
    ax.axvspan(0, 10, color="#F3F4F6", lw=0, zorder=0)
    ax.plot(x, G["sham"].values.mean(0), color=CLEAN_C, lw=CLEAN_LW, zorder=2)
    for a, (ls, lw) in BLOCK_STYLE.items():
        _course(ax, x, G[a].values, CSI_C, ls, lw)
    B = V.blocking()
    g1 = B["gradsc_vs_gradrand"]["pct_of_control_gain_lost_epochs_1_3"]["mean"]
    g2 = B["gradsc_vs_gradenergy"]["pct_of_control_gain_lost_epochs_1_3"]["mean"]
    eq = B["gradsc_vs_gradrand"]["epochs_behind_at_epoch_3"]["mean"]
    yb = G["gradsc"].values[:, 2].mean()
    ax.annotate("", xy=(3 - eq, yb), xytext=(3, yb), arrowprops=dict(arrowstyle="-|>", color="#B45309", lw=1.1,
                                                                     mutation_scale=6, shrinkA=0, shrinkB=0), zorder=8)
    ax.text(5.0, yb - 2.5, f"one epoch behind,\n{min(g1, g2):.0f}–{max(g1, g2):.0f}% of early gain", fontsize=8,
            color="#B45309", ha="left", va="top", fontweight="bold", linespacing=1.0)
    ax.set_xlim(0, 30.5)
    ax.set_xticks([1, 10, 20, 30])
    ax.set_ylim(5, 62)
    ax.set_yticks([10, 30, 50])
    ax.set_xlabel("Epoch", labelpad=1.5)
    ax.set_ylabel("Fine-class accuracy (%)")
    ax.set_title("Fine-grained learning falls behind", fontsize=7, pad=4)
    sd = pd.concat({a: g for a, g in G.items()}, names=["arm"]).reset_index()
    eff = pd.DataFrame([dict(arm="gradsc", table="share of the control's gain over chance lost, epochs 1-3 (%)",
                             vs_random_directions=g1, vs_energy_matched=g2, epochs_behind_at_epoch_3=eq)])
    ns.source_data(pd.concat([sd, eff], ignore_index=True), NAME, "f")


def panel_d3(ax, w_mm, h_mm):
    """Blocking design and line-style key (colour = measure, line = arm)."""
    panel_d_schematic(ax, w_mm, h_mm)
    for t in list(ax.texts)[-4:]:
        t.remove()
    for ln in list(ax.lines)[-4:]:
        ln.remove()
    ax.text(0.5, 12.5, "Controls: an equal share of other\nlearning signal withheld (energy-\nmatched), or the signal along as\n"
            "many random directions; sham", fontsize=6.2, color=DARK, va="top", linespacing=1.1)


SYN_MARK = {"label noise": "o", "blocking": "s", "suppression": "^"}


def panel_g3(ax):
    """Synthesis: superclass learning lost vs fine-grained learning lost (% of the control's gain over chance), every
    manipulation one point (mean, 95% CI over 5 seeds in both directions); filled = targeted manipulation vs a matched
    control, open = partial or untargeted damage; dashed = identity; solid = least-squares line."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import effect_units as V
    S, st = V.synthesis()
    ax.plot([0, 40], [0, 40], color="#D1D5DB", ls="--", lw=0.8, zorder=0)
    xx = np.array([0, 36])
    ax.plot(xx, st["intercept"] + st["slope"] * xx, color=DARK, lw=0.9, zorder=1)
    short = {"Label noise, between vs within": "label noise", "Label noise, similarity-matched vs within": "similarity-\nmatched noise",
             "Blocking vs random directions": "block vs random", "Blocking vs jointly matched": "block vs jointly\nmatched",
             "Random-grouping block vs jointly matched": "random-grouping\nblock", "Upper-stage block, fully matched": "upper-stage\nblock",
             "Random-direction block vs sham": "random-direction\nblock", "Jointly matched block vs sham": "jointly matched\nblock",
             "Suppression vs random directions": "suppression\nvs random", "Suppression vs sham": "suppression\nvs sham"}
    # label anchor (absolute data coordinates), alignment, leader line to the point
    lab_at = {"Label noise, between vs within": (19.3, 13.4, "left", False),
              "Blocking vs random directions": (18.3, 21.9, "left", False),
              "Blocking vs jointly matched": (20.0, 10.4, "left", True),
              "Random-grouping block vs jointly matched": (13.6, 5.9, "left", False),
              "Suppression vs random directions": (33.4, 28.9, "center", False),
              "Suppression vs sham": (31.9, 17.4, "center", False),
              "Label noise, similarity-matched vs within": (3.9, 10.4, "center", True),
              "Upper-stage block, fully matched": (10.6, 1.9, "left", True),
              "Random-direction block vs sham": (8.6, 8.4, "left", True),
              "Jointly matched block vs sham": (8.2, 16.2, "right", False)}
    for r in S.itertuples():
        m = SYN_MARK[r.family]
        ax.errorbar(r.superclass_lost_pct, r.fine_lost_pct,
                    xerr=[[r.superclass_lost_pct - r.superclass_ci[0]], [r.superclass_ci[1] - r.superclass_lost_pct]],
                    yerr=[[r.fine_lost_pct - r.fine_ci[0]], [r.fine_ci[1] - r.fine_lost_pct]], fmt="none",
                    ecolor="#D1D5DB", elinewidth=0.7, capsize=0, zorder=2)
        ax.plot(r.superclass_lost_pct, r.fine_lost_pct, m, ms=6.5, mfc=DARK if r.targeted else "white", mec=DARK,
                mew=1.0, zorder=4)
        if not r.targeted:
            continue
        tx, ty, ha, lead = lab_at[r.manipulation]
        if lead:
            ax.plot([r.superclass_lost_pct, tx], [r.fine_lost_pct, ty - (0.9 if ty > r.fine_lost_pct else -0.5)],
                    color="#9CA3AF", lw=0.5, zorder=1)
        ax.text(tx, ty, short[r.manipulation], fontsize=6, color=DARK, ha=ha, va="center", linespacing=1.0, zorder=5)
    ax.text(0.03, 0.97, f"r = {st['pearson_r']:.2f}", transform=ax.transAxes, fontsize=9, fontweight="bold",
            color=DARK, ha="left", va="top")
    ax.text(36.5, 37.5, "equal loss", fontsize=6, color="#9CA3AF", ha="right", va="bottom", rotation=0)
    ax.text(30.5, 5.0, f"slope {st['slope']:.2f}", fontsize=7, fontweight="bold", color=DARK,
            ha="center", va="center")
    ax.set_xlim(0, 39)
    ax.set_ylim(0, 39)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_yticks([0, 10, 20, 30])
    ax.set_xlabel("Superclass learning lost (% of the control's gain)", color=SSI_C)
    ax.set_ylabel("Fine-grained learning lost\n(% of the control's gain)", color="#B45309")
    ax.set_title("Fine-grained learning loses two thirds of the coarse loss", fontsize=7, pad=4)
    h = [Line2D([], [], marker=m, ls="none", ms=5, mfc=DARK, mec=DARK, label=f) for f, m in SYN_MARK.items()]
    h.append(Line2D([], [], marker="o", ls="none", ms=5, mfc="white", mec=DARK, label="partial or untargeted"))
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(0.0, 0.72), fontsize=6.2, frameon=False, handletextpad=0.2,
              borderaxespad=0.2, labelspacing=0.25)
    out = S.copy()
    out["superclass_ci_lo"], out["superclass_ci_hi"] = zip(*S.superclass_ci)
    out["fine_ci_lo"], out["fine_ci_hi"] = zip(*S.fine_ci)
    ns.source_data(out.drop(columns=["superclass_ci", "fine_ci"]), NAME, "h")
    ns.source_data(pd.DataFrame([st]), NAME, "h_fit")


SHARE_ARMS = [("sham", "#9CA3AF", "-", 0.8, "sham"), ("gradenergy", None, "-", 1.1, "energy-matched"),
              ("gradrand", None, ":", 1.1, "random directions"), ("gradsc", None, "--", 1.5, "superclass blocked")]


def panel_f_sharing(ax, name=NAME, letter="f"):
    """Learning signal shared within superclasses (analysis_outputs/paper/SHARED_signal/T2_blocking/per_epoch.csv,
    producer scripts/paper/shared_signal.py): per class, cos^2 of its mean learning signal (pooled layer, the
    network's own classifier, 10,000 test images) with the mean of the other four classes of its superclass, minus the
    same for 20 random groupings; mean over classes; mean and t-based 95% CI over seeds (n = 5). Arms and line styles
    as in SHARE_ARMS; grey shading = block window."""
    P = pd.read_csv(RESULTS / "SHARED_signal/T2_blocking/per_epoch.csv")
    ax.axvspan(0.5, 10.5, color=GREYBOX, lw=0, zorder=0)
    rows = []
    for arm, c_, ls_, lw_, lab in SHARE_ARMS:
        g = P[P.arm == arm].sort_values("epoch")
        col = c_ or SSI_C
        if arm in ("sham", "gradsc"):
            ax.fill_between(g.epoch, g.excess_mean_ci_lo, g.excess_mean_ci_hi, color=col, alpha=0.18, lw=0, zorder=2)
        ax.plot(g.epoch, g.excess_mean_mean, color=col, ls=ls_, lw=lw_, zorder=3, label=lab)
        rows += [dict(arm=arm, epoch=int(r.epoch), mean=r.excess_mean_mean, ci_lo=r.excess_mean_ci_lo,
                      ci_hi=r.excess_mean_ci_hi) for r in g.itertuples()]
    ax.axhline(0, color=ZERO_TXT, lw=0.6, ls="--", zorder=0)
    ax.set_xlim(0.5, 15.5)
    ax.set_xticks([1, 5, 10, 15])
    ax.set_ylim(-0.005, 0.135)
    ax.set_yticks([0, 0.03, 0.06, 0.09, 0.12])
    ax.set_xlabel("Epoch", labelpad=1.5)
    ax.set_ylabel("Learning signal shared\nwithin superclass", color=SSI_C)
    ax.set_title("Blocking keeps the\nlearning signal shared", fontsize=7, pad=4)
    ax.legend(loc="upper right", fontsize=6, frameon=False, handlelength=1.8, borderaxespad=0.1, labelspacing=0.15)
    ns.source_data(pd.DataFrame(rows), name, letter)


LAG_CTL = (("gradjoint", "-", 1.6, "vs jointly matched"), ("gradrand", "--", 0.9, "vs random"))


def panel_e_lag(ax):
    """Epochs the superclass-blocked networks are behind each control in superclass assignment (blue) and
    fine-class accuracy (amber); solid, against the jointly matched block (as many directions carrying as much of the
    learning signal); dashed, against the random-direction block (as many random directions); mean and t-based 95% CI
    over 5 seeds; shaded, block window (effect_units.blocking_lags on GRADJOINT/long.csv)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import effect_units as V
    G = V.blocking_lags()
    ax.axvspan(0.5, 10.5, color="#F3F4F6", lw=0, zorder=0)
    rows = []
    for ctl, ls, lw, _ in LAG_CTL:
        g = G[G.control == ctl]
        for k, col in (("superclass", SSI_C), ("fine", CSI_C)):
            P = g.pivot(index="seed", columns="epoch", values=f"lag_{k}")
            P = P.loc[:, P.columns <= 20]
            x = P.columns.values
            m, lo, hi = L.t_ci(P.values)
            if ctl == "gradjoint":
                ax.fill_between(x, lo, hi, color=col, alpha=0.13, lw=0, zorder=1)
            ax.plot(x, m, color=col, ls=ls, lw=lw, zorder=3, alpha=1.0 if ctl == "gradjoint" else 0.85)
            rows += [dict(control=ctl, measure=k, epoch=int(e), mean_lag=m[i], ci_lo=lo[i], ci_hi=hi[i],
                          **{f"seed_{sd}": P.loc[sd, e] for sd in P.index}) for i, e in enumerate(x)]
    e10 = G[(G.control == "gradjoint") & (G.epoch == 10)][["lag_superclass", "lag_fine"]].mean()
    ax.annotate(f"{e10.lag_superclass:.1f} epochs behind", xy=(10, e10.lag_superclass), xytext=(0.9, 10.2),
                fontsize=7, color=SSI_C, fontweight="bold", va="center",
                arrowprops=dict(arrowstyle="-", color=SSI_C, lw=0.5, relpos=(1.0, 0.5)))
    ax.annotate(f"{e10.lag_fine:.1f} epochs behind", xy=(10, e10.lag_fine), xytext=(0.9, 9.0),
                fontsize=7, color="#B45309", fontweight="bold", va="center",
                arrowprops=dict(arrowstyle="-", color="#B45309", lw=0.5, relpos=(1.0, 0.5)))
    ax.axhline(0, color=ZERO_TXT, lw=0.6, ls="--", zorder=0)
    ax.text(5.5, 0.25, "block", fontsize=6, color=ZERO_TXT, ha="center", va="bottom")
    h = [Line2D([], [], color=SSI_C, lw=1.5, label="superclass"),
         Line2D([], [], color=CSI_C, lw=1.5, label="fine-grained")]
    h += [Line2D([], [], color=DARK, lw=lw, ls=ls, label=lab) for _, ls, lw, lab in LAG_CTL]
    ax.legend(handles=h, loc="upper right", bbox_to_anchor=(1.03, 1.02), fontsize=6, frameon=False, handlelength=1.0,
              borderaxespad=0.1, labelspacing=0.12, handletextpad=0.4)
    ax.set_xlim(0.5, 20.3)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(-1, 13.2)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel("Epoch", labelpad=1.5)
    ax.set_ylabel("Epochs behind the control")
    ns.source_data(pd.DataFrame(rows), NAME, "e")
    print(f"[5b] epoch 10 lags vs jointly matched {e10.lag_superclass:.2f} / {e10.lag_fine:.2f}")
    return G


def panel_lag_proportion(ax, G):
    """During the block (epochs 1-10), fine-class lag against superclass lag, mean over seeds per epoch;
    filled, against the jointly matched block; open, against the random-direction block; lines, per-seed slopes through
    the origin averaged (effect_units.lag_proportion); dashed, equal lag."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import effect_units as V
    ax.plot([0, 8], [0, 8], color="#D1D5DB", ls="--", lw=0.8, zorder=0)
    rows, txt = [], []
    for ctl, mfc, lab in (("gradjoint", DARK, "jointly matched"), ("gradrand", "white", "random directions")):
        g = G[(G.control == ctl) & G.epoch.between(1, 10)]
        m = g.groupby("epoch")[["lag_superclass", "lag_fine"]].mean()
        pr = V.lag_proportion(G, ctl=ctl)
        b = pr["slope_fine_on_superclass"]["mean"]
        ax.plot([0, 8], [0, 8 * b], color=DARK, lw=1.0 if ctl == "gradjoint" else 0.7,
                ls="-" if ctl == "gradjoint" else "--", zorder=1)
        ax.plot(m.lag_superclass, m.lag_fine, "o", ms=3.4, mfc=mfc, mec=DARK, mew=0.7, ls="none", zorder=3)
        rows.append(dict(control=ctl, slope=b, slope_ci_lo=pr["slope_fine_on_superclass"]["ci"][0],
                         slope_ci_hi=pr["slope_fine_on_superclass"]["ci"][1],
                         slope_per_seed=pr["slope_fine_on_superclass"]["per_seed"],
                         lag_superclass_epoch10=pr["lag_superclass_end"]["mean"], lag_fine_epoch10=pr["lag_fine_end"]["mean"]))
        rows += [dict(control=ctl, epoch=int(e), mean_lag_superclass=r.lag_superclass, mean_lag_fine=r.lag_fine)
                 for e, r in m.iterrows()]
        txt.append(b)
    ax.text(7.9, 2.0, f"× {txt[0]:.2f}", fontsize=8, fontweight="bold", color=DARK, ha="right", va="center")
    ax.text(7.9, 0.9, "jointly\nmatched", fontsize=6, color=DARK, ha="right", va="center", linespacing=1.0)
    ax.text(0.3, 7.6, f"× {txt[1]:.2f}", fontsize=7, color=DARK, ha="left", va="top")
    ax.text(0.3, 6.5, "random\ndirections", fontsize=6, color=DARK, ha="left", va="top", linespacing=1.0)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_xticks([0, 4, 8])
    ax.set_yticks([0, 4, 8])
    ax.set_xlabel("Superclass lag\n(epochs)", color=SSI_C, labelpad=1.5, linespacing=1.0)
    ax.set_ylabel("Fine-grained lag (epochs)", color="#B45309", labelpad=1)
    ax.set_title("Fine lags two\nthirds as far", fontsize=7, pad=4, linespacing=1.0)
    ns.source_data(pd.DataFrame(rows), NAME, "e_prop")
    print(f"[5b] fine/superclass lag slope: jointly matched {txt[0]:.3f}, random {txt[1]:.3f}")


TIMING_ARMS = [("sham", "sham", "o", "white"), ("gradenergy", "energy-matched", "s", "white"),
               ("gradenergy_l2", "energy-matched, layer2+", "D", "white"), ("gradrand", "random directions", "^", "white"),
               ("gradrg", "random grouping", "^", DARK), ("gradsc_l2", "superclass, layer2+", "D", DARK),
               ("gradsc", "superclass directions", "s", DARK)]


def panel_f_timing(ax):
    """Every blocking network (7 arms x 5 seeds): epoch at which superclass assignment reaches 45% (x) and fine-class
    accuracy reaches 30% (y); filled, blocks of coarse directions; open, controls; Spearman over the 35 networks
    (robust over thresholds 40-50% x 25-35%: 0.84-0.97)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import effect_units as V
    M = V.milestones(45, 30)
    for a, lab, mk, fc in TIMING_ARMS:
        g = M[M.arm == a]
        ax.plot(g.epoch_superclass, g.epoch_fine, mk, ms=4.2, mfc=fc, mec=DARK, mew=0.8, ls="none", label=lab, zorder=3)
    ok = M.dropna()
    rho, p = stats.spearmanr(ok.epoch_superclass, ok.epoch_fine) if False else (None, None)
    from scipy import stats as _st
    rho, p = _st.spearmanr(ok.epoch_superclass, ok.epoch_fine)
    ax.plot([2, 16], [2, 16], color="#D1D5DB", ls="--", lw=0.8, zorder=0)
    ax.text(0.04, 0.97, f"ρ = {rho:.2f}", transform=ax.transAxes, fontsize=9, fontweight="bold", color=DARK, va="top")
    ax.text(0.04, 0.85, f"{len(ok)} networks", transform=ax.transAxes, fontsize=6.2, color=DARK, va="top")
    ax.set_xlim(2, 16)
    ax.set_ylim(2, 16)
    ax.set_xticks([4, 8, 12, 16])
    ax.set_yticks([4, 8, 12, 16])
    ax.set_xlabel("Epoch superclass assignment\nreaches 45%", color=SSI_C, labelpad=1.5)
    ax.set_ylabel("Epoch fine accuracy\nreaches 30%", color="#B45309")
    ax.set_title("Later coarse, later fine", fontsize=7, pad=4)
    for x_, y_, t_, ha_ in ((4.2, 3.0, "sham", "center"), (4.6, 10.8, "energy-\nmatched", "center"),
                            (10.2, 4.1, "random\ndirections", "center"), (11.0, 14.6, "superclass and\nrandom-grouping\nblocks", "center")):
        ax.text(x_, y_, t_, fontsize=6, color=DARK, ha=ha_, va="center", linespacing=1.0)
    ns.source_data(M, NAME, "f")
    print(f"[5f] Spearman {rho:.3f}, p {p:.1e}, n {len(ok)}")


def panel_g_routing(ax):
    """Where the blocking deficit lies: the fine-class deficit of the superclass-blocked networks over epochs 1-3 split
    exactly into superclass assignment and the choice among the classes of the right superclass (fresh read-out;
    log factorisation per epoch: dlog fine = dlog assignment + dlog within; effect_units.blocking)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import effect_units as V
    B = V.blocking()
    rows = []
    for i, (ctl, lab) in enumerate((("gradjoint", "vs jointly\nmatched block"), ("gradrand", "vs random-\ndirection block"))):
        r = B[f"routing_share_vs_{ctl}"]
        f, a_, w = r["dlog_fine"]["mean"], r["dlog_assignment"]["mean"], r["dlog_within"]["mean"]
        sa = r["pct_of_dlog_fine_in_assignment"]["mean"]       # mean of the per-seed shares
        sw = 100 - sa
        y = 1 - i
        ax.barh(y, sa, height=0.55, color=SSI_C, alpha=ns.BAR_ALPHA_CORE, lw=0)
        ax.barh(y, sw, left=sa, height=0.55, color=CSI_C, alpha=ns.BAR_ALPHA_CORE, lw=0)
        ax.text(sa / 2, y, f"{sa:.0f}%", fontsize=8, fontweight="bold", color="white", ha="center", va="center")
        rows.append(dict(control=ctl, share_assignment_pct=sa, share_within_pct=sw, dlog_fine=f, dlog_assignment=a_,
                         dlog_within=w, share_ci=r["pct_of_dlog_fine_in_assignment"]["ci"],
                         share_per_seed=r["pct_of_dlog_fine_in_assignment"]["per_seed"]))
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["vs jointly\nmatched", "vs random\ndirections"], fontsize=6.3, linespacing=1.0)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("Share of the fine-class deficit (%)", labelpad=1.5)
    ax.text(0, 1.52, "superclass assignment", fontsize=6.3, color=SSI_C, ha="left", va="bottom")
    ax.text(100, 1.52, "class within superclass", fontsize=6.3, color="#B45309", ha="right", va="bottom")
    ax.set_title("The deficit lies in routing", fontsize=7, pad=10)
    ns.source_data(pd.DataFrame(rows), NAME, "g")


def panel_g_dose(ax):
    """Per superclass, share of its learning signal blocked against its fine-class change vs sham, epochs 1-3 (fresh
    read-out; mean over seeds and epochs). Filled, the superclass block; open, the random-direction block; line,
    least-squares fit to the superclass block."""
    from scipy import stats as _st
    d = pd.read_csv(RESULTS / "GRADBLOCK" / "superclass_gradblock.csv")
    d = d[d.epoch.isin([1, 2, 3])]
    m = d.groupby("superclass")[["blocked_share", "fresh_fine_gradsc_minus_sham",
                                 "fresh_fine_gradrand_minus_sham"]].mean()
    x = 100 * m.blocked_share
    rs, ps = _st.spearmanr(m.blocked_share, m.fresh_fine_gradsc_minus_sham)
    rr, pr = _st.spearmanr(m.blocked_share, m.fresh_fine_gradrand_minus_sham)
    ax.axhline(0, color="#9CA3AF", lw=0.6, ls="--", zorder=0)
    ax.scatter(x, m.fresh_fine_gradrand_minus_sham, s=16, facecolor="white", edgecolor=DARK, lw=0.8, zorder=2)
    ax.scatter(x, m.fresh_fine_gradsc_minus_sham, s=18, color=DARK, lw=0, zorder=3)
    b = np.polyfit(x, m.fresh_fine_gradsc_minus_sham, 1)
    xx = np.linspace(x.min(), x.max(), 10)
    ax.plot(xx, np.polyval(b, xx), color=DARK, lw=1.0, zorder=2)
    ax.text(0.03, 0.04, f"ρ = {rs:.2f}".replace("-", "−"), transform=ax.transAxes, fontsize=9, fontweight="bold",
            color=DARK, ha="left", va="bottom")
    ax.text(39.8, -10.3, "superclass\nblock", fontsize=6.2, color=DARK, ha="right", va="center", linespacing=1.0)
    ax.text(39.8, 1.3, "random-direction block", fontsize=6.2, color=DARK, ha="right", va="bottom")
    ax.set_xlim(19.5, 40.5)
    ax.set_ylim(-12.6, 3.6)
    ax.set_xticks([20, 25, 30, 35, 40])
    ax.set_yticks([-10, -5, 0])
    ax.set_xlabel("Learning signal blocked (% of the superclass’s)", color=SSI_C, labelpad=1.5)
    ax.set_ylabel("Fine-class change vs\nno block, epochs 1–3 (pp)", color="#B45309")
    ax.set_title("Superclasses that lose more learning signal fall further behind", fontsize=7, pad=4)
    rows = m.reset_index().rename(columns={"fresh_fine_gradsc_minus_sham": "superclass_block_minus_sham_pp",
                                           "fresh_fine_gradrand_minus_sham": "random_block_minus_sham_pp"})
    rows["spearman_superclass"], rows["p_superclass"], rows["spearman_random"], rows["p_random"] = rs, ps, rr, pr
    ns.source_data(rows, NAME, "g")
    print(f"[5g] superclass rho {rs:.3f} P {ps:.4f}; random rho {rr:.3f} P {pr:.3f}")


def panel_a_noise(ax, P, A):
    """Label noise on one axis: superclass probe (blue) and fine-class accuracy (amber) under within- (control, solid)
    and between-superclass noise (dashed); clean training thin grey. Effects from effect_units.forest."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import effect_units as V
    F = V.forest(write=False).set_index("row")
    g5 = F.loc["label noise, between - within (superclass probe, epoch 5)"]
    gf = F.loc["label noise, between - within"]
    xp = P["clean"].columns.values.astype(float)
    xa = np.arange(31)
    ax.plot(xp, P["clean"].values.mean(0), color=CLEAN_C, lw=CLEAN_LW, zorder=2)
    ax.plot(xa, A["clean"].mean(0), color=CLEAN_C, lw=CLEAN_LW, zorder=2)
    _course(ax, xp, P["within_sc"].values, SSI_C, CTRL_LS, 1.4)
    _course(ax, xp, P["between_sc"].values, SSI_C, MANIP_LS, 1.4)
    _course(ax, xa, A["within_sc"], CSI_C, CTRL_LS, 1.4)
    _course(ax, xa, A["between_sc"], CSI_C, MANIP_LS, 1.4)
    ax.text(0.5, 95, f"superclass probe −{-g5['mean']:.0f}%", fontsize=6.5, color=SSI_C, ha="left",
            va="center")
    ax.text(0.5, 88.5, f"fine-class accuracy −{-gf['mean']:.0f}%", fontsize=6.5, color="#B45309",
            ha="left", va="center")
    h = [Line2D([], [], color=DARK, ls=CTRL_LS, lw=1.1, label="within (oak → maple)"),
         Line2D([], [], color=DARK, ls=MANIP_LS, lw=1.1, label="between (oak → rose)"),
         Line2D([], [], color=CLEAN_C, lw=CLEAN_LW, label="clean")]
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(-0.01, 0.85), ncol=2, fontsize=6, frameon=False,
              handlelength=1.6, handletextpad=0.3, columnspacing=0.8, labelspacing=0.15, borderaxespad=0)
    ax.set_xlim(-0.5, 30.5)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_ylim(0, 99)
    ax.set_yticks([0, 20, 40, 60])
    ax.spines["left"].set_bounds(0, 70)
    ax.set_xlabel("Epoch", labelpad=1.5)
    ax.set_ylabel("Probe or accuracy (%)")
    ax.set_title("Corrupting superclass labels slows both", fontsize=7, pad=4)


def panel_b_noise_lag(ax, P, A):
    """Epochs the between-superclass noise networks are behind the within-superclass control in the superclass probe
    (blue; checkpoints every 5 epochs) and in fine-class accuracy (amber; every epoch); labels at epoch 10. Per seed,
    t minus the epoch at which the control first reached the same value (linear interpolation, effect_units._lag); mean
    and t-based 95% CI over 5 seeds."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import effect_units as V
    rows, ends, lags = [], {}, {}
    for k, (x, Yb, Yc), col, mk in (("superclass", (P["clean"].columns.values.astype(float), P["between_sc"].values,
                                                    P["within_sc"].values), SSI_C, "o"),
                                    ("fine", (np.arange(31.0), np.asarray(A["between_sc"]), np.asarray(A["within_sc"])),
                                     CSI_C, None)):
        lag = np.array([[V._lag(e, Yb[i, j], x, Yc[i]) for j, e in enumerate(x)] for i in range(Yb.shape[0])])
        m, lo, hi = L.t_ci(lag)
        ax.fill_between(x, lo, hi, color=col, alpha=0.16, lw=0, zorder=1)
        ax.plot(x, m, color=col, lw=1.6, zorder=3, marker=mk, ms=2.6 if mk else 0)
        ends[k] = m[list(x).index(10.0)]
        lags[k] = (x, lag)
        rows += [dict(measure=k, epoch=int(e), mean_lag=m[j], ci_lo=lo[j], ci_hi=hi[j],
                      **{f"seed_{sd}": lag[i, j] for i, sd in enumerate(L.SEEDS)}) for j, e in enumerate(x)]
    # epoch at which each lag first reaches 2 epochs (per seed, linear interpolation), mean over seeds
    def first(x, y, thr=2.0):
        for k_ in range(1, len(x)):
            if y[k_] >= thr > y[k_ - 1]:
                return x[k_ - 1] + (thr - y[k_ - 1]) / (y[k_] - y[k_ - 1]) * (x[k_] - x[k_ - 1])
        return np.nan
    cross = {k: np.array([first(x, lag[i]) for i in range(lag.shape[0])]) for k, (x, lag) in lags.items()}
    ax.plot([0, np.nanmean(cross["fine"])], [2, 2], color="#9CA3AF", lw=0.6, ls=":", zorder=0)
    for k, col in (("superclass", SSI_C), ("fine", CSI_C)):
        ax.plot(np.nanmean(cross[k]), 2, marker="v", ms=4.5, color=col, mec="white", mew=0.4, zorder=6)
    d_ = np.nanmean(cross["fine"] - cross["superclass"])
    ax.text(np.nanmean(cross["fine"]) + 0.6, 0.7, f"{d_:.1f} epochs later", fontsize=6, color="#B45309", ha="left",
            va="center", zorder=6)
    rows += [dict(measure=f"{k}: epoch at which the lag first reaches 2 epochs", mean_lag=np.nanmean(v),
                  **{f"seed_{sd}": v[i] for i, sd in enumerate(L.SEEDS)}) for k, v in cross.items()]
    print(f"[5b] lag reaches 2 epochs: superclass {np.nanmean(cross['superclass']):.2f}, fine {np.nanmean(cross['fine']):.2f}; "
          f"fine later in {(cross['fine'] > cross['superclass']).sum()}/5 seeds")
    ax.text(0.5, 18.0, f"superclass {ends['superclass']:.1f} epochs behind (epoch 10)", fontsize=6.5,
            color=SSI_C, ha="left", va="center")
    ax.text(0.5, 16.4, f"fine-class {ends['fine']:.1f} epochs behind", fontsize=6.5, color="#B45309",
            ha="left", va="center")
    ax.axhline(0, color=ZERO_TXT, lw=0.6, ls="--", zorder=0)
    ax.set_xlim(-0.5, 30.5)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_ylim(-1.5, 19.5)
    ax.set_yticks([0, 5, 10, 15])
    ax.set_xlabel("Epoch", labelpad=1.5)
    ax.set_ylabel("Epochs behind the control\n(within-superclass noise)")
    ax.set_title("Fine-grained learning falls behind later and less", fontsize=7, pad=4)
    ns.source_data(pd.DataFrame(rows), NAME, "nl")
    print(f"[5b] label noise, epoch 10 lags: probe {ends['superclass']:.2f}, accuracy {ends['fine']:.2f}")


BLOCK_ARMS = (("gradjoint", CTRL_LS), ("gradsc", MANIP_LS))   # control solid, superclass block dashed


def panel_b_courses(ax):
    """Superclass assignment (fresh linear read-out, blue) and fine-class accuracy (amber) with the superclass
    directions blocked (dashed) and with the jointly matched block (control, solid; as many directions carrying as much
    of the learning signal); no block (sham) thin light grey; shaded, block (epochs 1-10). Mean and t-based 95% CI
    over 5 seeds (GRADJOINT/long.csv; epochs with a fresh read-out)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import effect_units as V
    D = pd.read_csv(V.GBL).dropna(subset=["fresh_assignment"])
    ax.axvspan(0.5, 10.5, ymax=70 / 99, color="#F3F4F6", lw=0, zorder=0)   # shading stops below the legend
    rows = []
    for col_name, col in (("fresh_assignment", SSI_C), ("val_accuracy", CSI_C)):
        S = D[D.arm == "sham"].pivot(index="seed", columns="epoch", values=col_name).loc[L.SEEDS]
        ax.plot(S.columns.values, S.values.mean(0), color=CLEAN_C, lw=CLEAN_LW, alpha=0.55, zorder=2)
        rows += [dict(arm="sham", measure=col_name, epoch=int(e), mean=v) for e, v in zip(S.columns, S.values.mean(0))]
        for arm, ls in BLOCK_ARMS:
            P = D[D.arm == arm].pivot(index="seed", columns="epoch", values=col_name).loc[L.SEEDS]
            P = P.loc[:, P.columns.isin(S.columns)]          # the epochs with a read-out in every arm
            m = _course(ax, P.columns.values, P.values, col, ls, 1.4)
            _, lo, hi = L.t_ci(P.values)
            rows += [dict(arm=arm, measure=col_name, epoch=int(e), mean=m[i], ci_lo=lo[i], ci_hi=hi[i],
                          **{f"seed_{sd}": P.loc[sd, e] for sd in P.index}) for i, e in enumerate(P.columns)]
    G = V.blocking_lags()
    e10 = G[(G.control == "gradjoint") & (G.epoch == 10)][["lag_superclass", "lag_fine"]].mean()
    ax.text(5.5, 1.5, "block", fontsize=6, color=ZERO_TXT, ha="center", va="bottom")
    h = [Line2D([], [], color=DARK, ls=CTRL_LS, lw=1.1, label="control (jointly matched)"),
         Line2D([], [], color=DARK, ls=MANIP_LS, lw=1.1, label="superclass block"),
         Line2D([], [], color=CLEAN_C, lw=CLEAN_LW, label="no block")]
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(-0.01, 0.85), ncol=2, fontsize=6, frameon=False,
              handlelength=1.6, handletextpad=0.3, columnspacing=0.8, labelspacing=0.15, borderaxespad=0)
    ax.set_xlim(0.5, 30.5)
    ax.set_xticks([1, 10, 20, 30])
    ax.set_ylim(0, 99)
    ax.set_yticks([0, 20, 40, 60])
    ax.spines["left"].set_bounds(0, 70)
    ax.set_xlabel("Epoch", labelpad=1.5)
    ax.set_ylabel("Assignment or accuracy (%)")
    ns.source_data(pd.DataFrame(rows), NAME, "bc")
    print(f"[5b] epoch 10 lags vs jointly matched {e10.lag_superclass:.2f} / {e10.lag_fine:.2f}")



RDT_NOISE = RESULTS / "RDT_noise"
RDT_BLOCK = RESULTS / "RDT_blocking"
NOISE_ARMS = (("clean", "clean"), ("within", "within"), ("between", "between"))


def panel_noise_rdt(ax):
    """RDT + forward pass, fixed-label noise (five seeds; RDT_noise_fit, scripts/paper/noise_fit_features.py and
    noise_fit_lineage.py): pooled-layer class-selective features per run under between-superclass noise, split into
    features that fire as much on mislabelled training images of their own class as on correctly labelled ones
    (unbent, filled), features bent in the interval (index below the 5th percentile of clean training with the
    between-superclass flip pattern as a dummy; at epoch 5 all bent features count as newly bent) and features already bent at the previous
    checkpoint (lineage by RDT matching). Grey line, all class-selective features in clean training; dotted, bent
    features under within-superclass noise. Mean over seeds."""
    D = pd.read_csv(RESULTS / "RDT_noise_fit" / "lineage.csv")
    F_ = pd.read_csv(RESULTS / "RDT_noise_fit" / "features.csv")
    lo = F_[F_.arm == "clean"].groupby("epoch").nfi.quantile(0.05)
    F_["bent"] = F_.nfi < F_.epoch.map(lo)
    ep = [5, 10, 15, 20, 25, 30]

    def per_run(df, mask):
        return df[mask].groupby(["seed", "epoch"]).size().unstack("epoch").reindex(columns=ep).fillna(0).mean()
    b5 = F_[(F_.arm == "between") & (F_.epoch == 5)]
    unb = per_run(F_, (F_.arm == "between") & ~F_.bent)
    new = per_run(D, (D.arm == "between") & D.bent & (D.origin != "from bent class feature"))
    old = per_run(D, (D.arm == "between") & D.bent & (D.origin == "from bent class feature"))
    new[5] = b5.groupby("seed").bent.sum().mean()
    old[5] = 0.0
    clean_tot = F_[F_.arm == "clean"].groupby(["seed", "epoch"]).size().unstack("epoch").mean()
    within_bent = per_run(F_, (F_.arm == "within") & F_.bent)
    x = np.array(ep, float)
    w = 3.4
    ax.bar(x, unb.values, width=w, color=CSI_C, alpha=ns.BAR_ALPHA_CORE, edgecolor="white", lw=0.4,
           label="unbent", zorder=2)
    ax.bar(x, old.values, width=w, bottom=unb.values, color="#B45309", alpha=0.85, edgecolor="white", lw=0.4,
           label="bent before", zorder=2)
    ax.bar(x, new.values, width=w, bottom=unb.values + old.values, facecolor="white", edgecolor="#B45309",
           hatch="//////", lw=0.6, label="newly bent", zorder=2)
    ax.plot(x, clean_tot.values, color="#6B7280", lw=1.0, marker="o", ms=2.2, zorder=4)
    ax.plot(x, within_bent.values, color="#7C2D12", lw=1.0, ls=(0, (1, 1.2)), marker="o", ms=2.0, zorder=4)
    S = F_.groupby(["arm", "seed"]).apply(lambda g: g[g.epoch == 30].bent.mean(), include_groups=False).unstack("arm")
    born = D[(D.arm == "between") & D.bent & (D.epoch == 30)]
    from_existing = 100 * (born.origin != "born").mean()
    tot30 = (unb + old + new)[30]
    ax.text(30, tot30 + 5, f"{100 * S.between.mean():.0f}%\nbent", fontsize=6.5, color="#B45309", ha="center",
            va="bottom", linespacing=1.0, fontweight="bold")
    ax.text(32.4, clean_tot[30], "clean,\nall", fontsize=6, color="#6B7280", ha="left", va="center",
            linespacing=1.0, clip_on=False)
    ax.text(32.4, within_bent[30], "within,\nbent", fontsize=6, color="#7C2D12", ha="left", va="center",
            linespacing=1.0, clip_on=False)
    leg = ax.legend(loc="upper left", fontsize=6, frameon=False, handlelength=1.0, handleheight=0.8, borderaxespad=0.1,
                    labelspacing=0.2, title="between-superclass noise", title_fontsize=6, alignment="left")
    leg.get_title().set_color("#374151")
    ax.set_xlim(2, 33)
    ax.set_xticks(ep)
    ax.set_ylim(0, 265)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Class-selective features")
    ax.set_title("Class features bent\nto fit the wrong labels", fontsize=7, pad=3)
    ax.spines["left"].set_bounds(0, 250)
    ax.set_yticks([0, 50, 100, 150, 200, 250])
    rows = [dict(series=n_, epoch=int(e_), mean_per_run=float(v_)) for n_, ser in
            (("between, unbent", unb), ("between, bent earlier", old), ("between, bent in this interval", new),
             ("clean, all class-selective", clean_tot), ("within, bent", within_bent)) for e_, v_ in ser.items()]
    # per seed (all three arms): class-selective and bent features; between-superclass noise also split into newly bent
    # and bent before (the plotted segments); bent share = bent / class_selective
    per = F_.groupby(["arm", "seed", "epoch"]).agg(class_selective=("feature", "size"), bent=("bent", "sum")).reset_index()
    bb = D[(D.arm == "between") & D.bent].assign(before=lambda d: d.origin == "from bent class feature")
    split = bb.groupby(["seed", "epoch"]).before.agg(bent_before="sum", n="size").reset_index()
    split["newly_bent"] = split.n - split.bent_before
    per = per.merge(split[["seed", "epoch", "bent_before", "newly_bent"]].assign(arm="between"),
                    on=["arm", "seed", "epoch"], how="left")
    m5 = (per.arm == "between") & (per.epoch == 5)
    per.loc[m5, "bent_before"], per.loc[m5, "newly_bent"] = 0, per.loc[m5, "bent"]
    rows += [dict(series=f"per seed: {r.arm}", seed=int(r.seed), epoch=int(r.epoch), class_selective=int(r.class_selective),
                  bent=int(r.bent), **({"unbent": int(r.class_selective - r.bent), "bent_before": int(r.bent_before),
                                         "newly_bent": int(r.newly_bent)} if r.arm == "between" else {}))
             for r in per.itertuples()]
    ns.source_data(pd.DataFrame(rows), NAME, "noise_rdt")
    print(f"[noise fit] bent share at epoch 30: between {np.round(100 * S.between.values, 0).tolist()}, within "
          f"{np.round(100 * S.within.values, 0).tolist()}, clean {np.round(100 * S.clean.values, 0).tolist()}; "
          f"bent at 30 descending from existing features {from_existing:.0f}%")


def panel_release(ax):
    """RDT, acquisition blocking (per-epoch SAEs, seeds 7, 42, 137; one threshold set per seed, all five layers;
    RDT_blocking/rdt_blocking_long.csv): superclass (Ab-E, blue) and class-differentiation events (Di-E, amber) per
    epoch around the end of the block, after blocking the superclass directions (solid) or the energy-matched
    directions outside them (dashed, Di-E only). Mean and range over seeds."""
    d = pd.read_csv(RDT_BLOCK / "rdt_blocking_long.csv")
    e = d[(d.tset == "common") & (d.kind == "events") & (d.layer == "all") & d.epoch.between(6, 15)]
    ax.axvspan(5.5, 10.5, color="#F3F4F6", lw=0, zorder=0)
    rows = []
    for arm, m, col, ls, lw in (("gradsc", "Ab-E", SSI_C, "-", 1.4), ("gradsc", "Di-E", CSI_C, "-", 1.4),
                                ("gradenergy", "Di-E", CSI_C, "--", 1.0), ("gradenergy", "Ab-E", SSI_C, "--", 1.0)):
        P = e[(e.arm == arm) & (e.measure == m)].pivot(index="seed", columns="epoch", values="value") / 1e3
        x = P.columns.values
        ax.plot(x, P.mean(), color=col, ls=ls, lw=lw, marker="o" if ls == "-" else None, ms=2.4, zorder=3)
        ax.fill_between(x, P.min(), P.max(), color=col, alpha=0.15 if ls == "-" else 0.08, lw=0, zorder=2)
        rows += [dict(arm=arm, measure=m, seed=s, epoch=int(ep), events_k=v) for s, r in P.iterrows()
                 for ep, v in r.items()]
    ax.axvline(10.5, color=ns.NEUTRAL, lw=0.6, zorder=1)
    ax.annotate("superclass first", xy=(10.95, 2.9), xytext=(5.8, 2.6), fontsize=6.2, color=SSI_C, ha="left",
                va="center", arrowprops=dict(arrowstyle="-", color=SSI_C, lw=0.5, shrinkB=1))
    ax.annotate("differentiation\none epoch later", xy=(12, 12.5), xytext=(12.3, 20.0), fontsize=6.2, color=ACC_HUE,
                ha="left", va="center", linespacing=1.0, arrowprops=dict(arrowstyle="-", color=ACC_HUE, lw=0.5))
    ax.annotate("fine-class block\nreleased: at once", xy=(10.9, 22.4), xytext=(5.8, 22.4), fontsize=6.2,
                color=ACC_HUE, ha="left", va="center", linespacing=1.0,
                arrowprops=dict(arrowstyle="-", color=ACC_HUE, lw=0.5, ls="--", shrinkB=1))
    ax.set_xlim(5.5, 15.5)
    ax.set_xticks([6, 8, 10, 11, 12, 14])
    ax.set_ylim(0, 30)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Events per epoch (×10³)")
    ax.set_title("Release replays the order", fontsize=7, pad=4)
    ns.rdt_tag(ax, x=0.02, ha="left", y=0.86)
    ns.source_data(pd.DataFrame(rows), NAME, "release")
    g = e.pivot_table(index=["arm", "measure", "seed"], columns="epoch", values="value")
    print("[release] gradsc Ab-E 10->11 x", (g.loc[("gradsc", "Ab-E")][11] / g.loc[("gradsc", "Ab-E")][10]).round(1).tolist(),
          "| gradsc Di-E 10,11,12", g.loc[("gradsc", "Di-E")][[10, 11, 12]].values.tolist(),
          "| energy Di-E 10->11 x", (g.loc[("gradenergy", "Di-E")][11] / g.loc[("gradenergy", "Di-E")][10]).round(1).tolist())


SD_MAP = {"b": "a_probe", "c": "a_accuracy", "c_rdt": "a_rdt", "nl": "b", "noise_rdt": "c", "bc": "d", "e": "e",
          "e_prop": "e_proportion", "release": "f", "g": "g", "h": "h", "h_fit": "h_fit"}


def main():
    """Three rows: a-c label noise (courses, lag, RDT); d-f blocking (courses, lag, RDT at release); g per-superclass
    dose-response, h all manipulations. Source Data files are renamed through SD_MAP."""
    ns.apply()
    P, A = load()
    _sd = ns.source_data
    ns.source_data = lambda df, name, letter: _sd(df, name, SD_MAP.get(letter, letter) if name == NAME else letter)
    for f in (ns.FINAL_OUT / "source_data").glob(f"{NAME}_*.csv"):
        f.unlink()
    fig = plt.figure(figsize=(W * ns.MM, H * ns.MM))
    # Source Data of the probe and accuracy courses and effects, written via hidden axes
    for fn, data in ((panel_b3, P), (panel_c3, A)):
        tmp = fig.add_axes([0, 0, 0.01, 0.01])
        fn(tmp, data)
        tmp.remove()
    # one row per experiment: courses with the control left, epochs behind middle, RDT right
    SUP = 178
    fig.set_size_inches(W * ns.MM, SUP * ns.MM)
    mm = lambda x, y, w, h: fig.add_axes([x / W, y / SUP, w / W, h / SUP])
    ax_a, ax_b = mm(11, 128, 45, 38), mm(71, 128, 45, 38)
    ax_c1 = mm(136, 128, 36, 34)
    ax_d, ax_e, ax_f = mm(11, 70, 45, 38), mm(71, 70, 45, 38), mm(136, 70, 40, 38)
    ax_g, ax_h = mm(17, 9, 61, 44), mm(108, 9, 64, 44)
    panel_a_noise(ax_a, P, A)
    panel_b_noise_lag(ax_b, P, A)
    panel_noise_rdt(ax_c1)
    panel_b_courses(ax_d)
    ax_d.set_title("Blocking slows both; both recover", fontsize=7, pad=4)
    panel_e_lag(ax_e)
    ax_e.set_title("Both fall behind in step", fontsize=7, pad=4)
    panel_release(ax_f)
    panel_g_dose(ax_g)
    panel_g3(ax_h)
    for ax, letter, xmm, ymm in ((ax_a, "a", 0.5, SUP - 1), (ax_b, "b", 61, SUP - 1), (ax_c1, "c", 123, SUP - 1),
                                 (ax_d, "d", 0.5, SUP - 59), (ax_e, "e", 61, SUP - 59), (ax_f, "f", 123, SUP - 59),
                                 (ax_g, "g", 0.5, 60), (ax_h, "h", 88, 60)):
        b = ax.get_position()
        ns.panel(ax, letter, x=(xmm / W - b.x0) / b.width, y=(ymm / SUP - b.y0) / b.height)
    ns.source_data = _sd
    print("saved", ns.save_final(fig, NAME, 180))


if __name__ == "__main__":
    main()
