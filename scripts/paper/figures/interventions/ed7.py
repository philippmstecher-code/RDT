#!/usr/bin/env python3
"""Extended Data Fig. 7: blocking the learning signal along the superclass directions.

  a  superclass assignment (fresh read-out, top) and fine-class accuracy (bottom), epochs 1-30
  b  coarse block and random-grouping block, minus energy-matched, epochs 1-3
  c  blocks from the third stage to the pooled layer, fully matched: epochs behind at the end of the block
  d  coarse block minus matched block, network's own classifier: fine-class accuracy, superclass accuracy, and the class
     within the true superclass
  e  fine-class lag against superclass lag during the block; per-seed proportion against the matched block and the
     random-direction block (effect_units.blocking_lags / lag_proportion)
  f  per superclass, share of its learning signal blocked against its fine-class change, epochs 1-3
  g  ten manipulations, superclass against fine-grained learning lost (effect_units.synthesis), numbered 1-10 with the
     key beside the axes; the three overlapping points near (7, 4) are repeated in an inset
  h  release: each level's RDT events per epoch relative to its own mean in epochs 6-10 (log2 axis), coarse and matched
     block; onset = first epoch after release with events above that level; triangles, median onset; strip, one dot
     per seed (kind events, layer all, common thresholds)

Colour = level measured (superclass blue, class amber; cross-level scatter neutral); within a level, coarse block
darkest solid, matched block mid dashed, random directions light dotted, no block (sham) thin grey.

Data (analysis_outputs/paper): GRADJOINT/long.csv (a, c, e), GRADBLOCK/final_long.csv and
RANDGROUP_control/specificity_arms.csv (b), ORACLE_within/oracle_summary.json (d),
GRADBLOCK/superclass_gradblock.csv (f), RDT_completion/rdt_completion_long.csv (h).
Writes ed7.* and Source Data ed7_<letter>.csv.

Usage: [ES_FIG_OUT=<dir>] python scripts/paper/figures/interventions/ed7.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))           # scripts/paper/figures
sys.path.insert(0, str(HERE.parent / "routing"))
sys.path.insert(0, str(HERE.parents[1]))       # scripts/paper
import nmi_style as ns  # noqa: E402
from fig_common import DOT, ERR, REF, RESULTS, SEEDS, WINDOW  # noqa: E402

FIG = "ed7"
W_MM, H_MM = 180, 175
BLUE = getattr(ns, "BLUE", ["#1E40AF", "#3B82F6", "#60A5FA", "#93C5FD"])
AMBER = getattr(ns, "AMBER", ["#92400E", "#D97706", "#F59E0B", "#FCD34D"])
GREYS = getattr(ns, "GREYS", ["#374151", "#6B7280", "#9CA3AF", "#E5E7EB"])
BLUE_T, AMBER_T, GREY_T = "#1E40AF", "#92400E", "#6B7280"
SC_TXT, CL_TXT = "#2563EB", "#B45309"          # axis-label colours (text-safe)
DARK = GREYS[0]
BAR_A = 0.85
# arm -> (label, ramp index or None = grey reference, line style, line width)
ARMS = {"gradsc": ("coarse block", 0, "-", 1.4),
        "gradjoint": ("matched block", 1, "--", 1.1),
        "gradrand": ("random directions", 2, ":", 1.3),
        "sham": ("no block", None, "-", 0.8)}


def arm_col(arm, ramp):
    k = ARMS[arm][1]
    return GREYS[2] if k is None else ramp[k]


# helpers
def tci(v):
    v = np.asarray(v, float)
    m = v.mean()
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return m, m - h, m + h


def minus(s):
    return s.replace("-", "−")


class Canvas:
    """Axes and text placed in mm from the top-left corner of a W x H mm canvas."""

    def __init__(self, w_mm, h_mm):
        self.w, self.h = w_mm, h_mm
        self.fig = plt.figure(figsize=(w_mm * ns.MM, h_mm * ns.MM))

    def ax(self, x, y_top, w, h):
        return self.fig.add_axes([x / self.w, 1 - (y_top + h) / self.h, w / self.w, h / self.h])

    def letter(self, k, x, y):
        self.fig.text(x / self.w, 1 - y / self.h, k, fontsize=8, fontweight="bold", va="top", ha="left")

    def title(self, text, x, y, w=None, ha="left"):
        """Panel title (claim) at an absolute position; top-aligned."""
        self.fig.text(x / self.w, 1 - y / self.h, text, fontsize=7, va="top", ha=ha, linespacing=1.1)


def bar(ax, x, v, color, rng, width, style="filled", dot_s=6, alpha=None):
    """Bar = mean, grey seed dots, dark 95% CI. style: filled | hatched (light fill + hatch) | open."""
    alpha = BAR_A if alpha is None else alpha
    m, lo, hi = tci(v)
    if style == "hatched":
        ax.bar(x, m, width=width, color=color, alpha=0.25, lw=0, zorder=1)
        ax.bar(x, m, width=width, facecolor="none", edgecolor=color, lw=0.8, hatch="//////", zorder=1)
    else:
        ax.bar(x, m, width=width, color=color, alpha=alpha, lw=0, zorder=1)
    ax.scatter(x + rng.uniform(-0.2, 0.2, len(v)) * width, v, s=dot_s, color=DOT, lw=0, zorder=3)
    ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=ERR, elinewidth=0.9, capsize=2.2,
                capthick=0.9, zorder=4)
    return m, lo, hi


def cat_axis(ax, labels, fs=6.3):
    ax.axhline(0, color=REF, ls="--", lw=0.6, zorder=0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=fs)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(-0.55, len(labels) - 0.45)


def key_strip(cv, rows, y_mm, x0_mm=14, gap_mm=4.0, fs=6.5):
    """Shared key: rows of (kind, label, style dict); kind 'line' | 'patch' | 'hatch' | 'marker'.
    Entries are laid out left to right with their text widths measured on the canvas."""
    fig = cv.fig
    r = fig.canvas.get_renderer()
    for k, row in enumerate(rows):
        x = x0_mm
        y = y_mm + k * 3.6
        yf = 1 - y / cv.h
        for kind, lab, st in row:
            hx = x / cv.w
            if kind == "line":
                fig.add_artist(Line2D([hx, (x + 2.75) / cv.w, (x + 5.5) / cv.w], [yf] * 3, transform=fig.transFigure,
                                      color=st["c"], ls=st.get("ls", "-"), lw=st.get("lw", 1.1),
                                      marker=st.get("marker"), ms=2.4, markevery=[1] if st.get("marker") else None))
                tx = x + 6.5
            elif kind == "marker":
                fig.add_artist(Line2D([(x + 1.2) / cv.w], [yf], transform=fig.transFigure, ls="none",
                                      marker=st.get("marker", "o"), ms=3.2, mfc=st.get("mfc", st["c"]),
                                      mec=st["c"], mew=0.8))
                tx = x + 3.4
            elif kind in ("patch", "hatch", "shade"):
                rect = Rectangle((hx, yf - 1.1 / cv.h), 3.2 / cv.w, 2.2 / cv.h, transform=fig.transFigure,
                                 facecolor=(st["c"] if kind != "hatch" else "white"),
                                 alpha=(st.get("alpha", BAR_A) if kind == "patch" else 1.0),
                                 edgecolor=(st.get("ec", "none") if kind != "hatch" else st["c"]),
                                 hatch=("//////" if kind == "hatch" else None), lw=0.7)
                fig.add_artist(rect)
                tx = x + 4.2
            else:
                tx = x
            t = fig.text(tx / cv.w, yf, lab, fontsize=fs, va="center", ha="left", color=st.get("tc", "black"))
            wpx = t.get_window_extent(r).width
            x = tx + wpx / fig.dpi * 25.4 + gap_mm


# a: curves
def panel_a(cv, x, y, w, h_each, gap):
    import intervention_panels
    D = intervention_panels.gb_long()
    fa = cv.ax(x, y, w, h_each)
    fb = cv.ax(x, y + h_each + gap, w, h_each)
    order = ("sham", "gradrand", "gradjoint", "gradsc")
    rows = []
    for ax_, col, ramp, lab, yl, yt in ((fa, "fresh_assignment", BLUE, "Superclass\nassignment (%)", (15, 72), [20, 40, 60]),
                                        (fb, "val_accuracy", AMBER, "Fine-class\naccuracy (%)", (5, 62), [20, 40, 60])):
        ax_.axvspan(0.5, 10.5, color=WINDOW, lw=0, zorder=0)
        for arm in order:
            lab_, _, ls_, lw_ = ARMS[arm]
            P = D[D.arm == arm].pivot(index="seed", columns="epoch", values=col).dropna(axis=1)
            ax_.plot(P.columns, P.values.mean(0), color=arm_col(arm, ramp), ls=ls_, lw=lw_,
                     zorder=4 if arm == "gradsc" else 3, dash_capstyle="round" if ls_ == ":" else "butt")
            rows += [dict(measure=col, arm=arm, arm_label=lab_, epoch=int(e), mean=v,
                          **{f"seed_{s}": P.loc[s, e] for s in P.index}) for e, v in zip(P.columns, P.values.mean(0))]
        ax_.set_xlim(0.5, 30.5)
        ax_.set_ylim(*yl)
        ax_.set_yticks(yt)
        ax_.set_xticks([1, 10, 20, 30])
        ax_.set_ylabel(lab, labelpad=1.5)
    fa.yaxis.label.set_color(SC_TXT)
    fb.yaxis.label.set_color(CL_TXT)
    fa.tick_params(labelbottom=False)
    fb.set_xlabel("Epoch", labelpad=1.5)
    fa.text(5.5, 70, "block", fontsize=6.3, color=GREY_T, ha="center", va="top")
    ns.source_data(pd.DataFrame(rows), FIG, "a")
    return fa, fb


# b: blocking a random grouping
def panel_b(ax):
    D = pd.read_csv(RESULTS / "GRADBLOCK" / "final_long.csv")
    M = D[D.epoch.isin([1, 2, 3])].groupby(["arm", "seed"])[["fresh_assignment", "val_accuracy"]].mean()
    share = pd.read_csv(RESULTS / "RANDGROUP_control" / "specificity_arms.csv").groupby("arm").coarse_removed_share.mean()
    rng = np.random.default_rng(11)
    out, lows = [], []
    groups = (("gradsc", "coarse\nblock"), ("gradrg", "random\ngrouping"))
    meas = (("fresh_assignment", "superclass assignment (fresh read-out)", BLUE[0], -0.19),
            ("val_accuracy", "fine-class accuracy (network)", AMBER[0], 0.19))
    for i, (arm, lab) in enumerate(groups):
        for col, mlab, hue, dx in meas:
            v = (M.loc[arm][col] - M.loc["gradenergy"][col]).loc[SEEDS].values
            m, lo, hi = bar(ax, i + dx, v, hue, rng, 0.34)
            lows.append(min(lo, v.min()))
            out.append(dict(arm=arm, arm_label=lab.replace("\n", " "), contrast=f"{arm} - gradenergy", epochs="1-3",
                            measure=mlab, mean=m, ci95_lo=lo, ci95_hi=hi, n_below_zero=int((v < 0).sum()), n=len(v),
                            coarse_structure_share=float(share["superclass" if arm == "gradsc" else "randgroup"]),
                            **{f"seed_{s}": x_ for s, x_ in zip(SEEDS, v)}))
    cat_axis(ax, [g[1] for g in groups])
    ax.set_ylim(min(lows) - 0.8, 0.6)
    ax.set_yticks([-6, -4, -2, 0])
    ax.set_ylabel("Minus energy-matched,\nepochs 1–3 (pp)")
    rg = share["randgroup"]
    sd = pd.DataFrame(out)
    ns.source_data(sd, FIG, "b")
    print("[b] " + "; ".join(f"{r.arm} {r.measure[:22]}: {r.mean:+.2f} [{r.ci95_lo:+.2f}, {r.ci95_hi:+.2f}] "
                             f"{r.n_below_zero}/5" for r in sd.itertuples()) + f"; share {rg:.3f}")
    return rg


# c: fully matched doses
def panel_c(ax):
    import effect_units as V
    G = V.blocking_lags(pairs=(("gradsc_l3", "gradjoint_l3"),))
    pr = V.lag_proportion(G, blk="gradsc_l3", ctl="gradjoint_l3")
    rng = np.random.default_rng(10)
    out, npos = [], []
    con = "coarse block - matched block, third stage to the pooled layer"
    for i, (key, hue, lab) in enumerate((("lag_superclass_end", BLUE[0], "superclass assignment"),
                                         ("lag_fine_end", AMBER[0], "fine-class accuracy"))):
        r = pr[key]
        v = np.array(r["per_seed"], float)
        bar(ax, i, v, hue, rng, 0.6)
        m, (lo, hi) = r["mean"], r["ci"]
        assert abs(m - v.mean()) < 1e-3
        ax.text(i, max(hi, v.max()) + 0.25, f"{m:.1f}", fontsize=6.5, color=ERR, ha="center", va="bottom")
        npos.append(int((v > 0).sum()))
        out.append(dict(contrast=con, epoch=10, measure=f"{lab}, epochs behind", mean=m, ci95_lo=lo, ci95_hi=hi,
                        n_above_zero=npos[-1], **{f"seed_{s}": x_ for s, x_ in zip(SEEDS, v)}))
    sl = pr["slope_fine_on_superclass"]
    out.append(dict(contrast=con, epoch="1-10", measure="fine-class lag per epoch of superclass lag", mean=sl["mean"],
                    ci95_lo=sl["ci"][0], ci95_hi=sl["ci"][1], **{f"seed_{s}": x_ for s, x_ in zip(SEEDS, sl["per_seed"])}))
    cat_axis(ax, ["super-\nclass", "fine\nclass"], fs=6.3)
    for t, col in zip(ax.get_xticklabels(), (SC_TXT, CL_TXT)):
        t.set_color(col)
    ax.set_ylim(0, 7.4)
    ax.set_yticks([0, 2, 4, 6])
    ax.set_ylabel("Epochs behind, end of block")
    n_all = min(npos)
    ax.text(0.5, 7.4, f"every seed ({n_all}/5)" if n_all == 5 else f"{n_all}/5 seeds", fontsize=6.3, color=ERR,
            ha="center", va="top")
    sd = pd.DataFrame(out)
    ns.source_data(sd, FIG, "c")
    print("[c] " + "; ".join(f"{r.measure}: {r.mean:.2f} [{r.ci95_lo:.2f}, {r.ci95_hi:.2f}]" for r in sd.itertuples()))
    return pr


# d: where the fine-class deficit lies
def panel_d(ax):
    """Coarse block minus matched block, network's own classifier, all 10,000 test images
    (ORACLE_within/oracle_summary.json; scripts/paper/oracle_within.py)."""
    S = json.load(open(RESULTS / "ORACLE_within" / "oracle_summary.json"))
    wins = [("epochs 1-3", "epochs\n1–3"), ("epochs 8-10", "epochs\n8–10"),
            ("epochs 11-15", "epochs\n11–15"), ("epoch 30", "epoch\n30")]
    meas = [("net_fine", "fine-class accuracy", AMBER[0], "filled"),
            ("net_assignment", "superclass accuracy", BLUE[0], "filled"),
            ("oracle_logits", "class within the true superclass", AMBER[0], "hatched")]
    rng = np.random.default_rng(8)
    wid, rows = 0.26, []
    for i, (w, wl) in enumerate(wins):
        for j, (k, lab, c, st) in enumerate(meas):
            r = S[f"jointly matched|{w}|{k}"]
            v = np.array(r["per_seed"], float)
            m, lo, hi = bar(ax, i + (j - 1) * wid * 1.1, v, c, rng, wid, style=st, dot_s=4)
            rows.append(dict(contrast="coarse block - matched block", window=w, measure=k, label=lab, mean=m,
                             ci_lo=lo, ci_hi=hi, n_lower=r["n_lower"], **{f"seed_{s}": x for s, x in zip(SEEDS, v)}))
    cat_axis(ax, [wl for _, wl in wins])
    ax.set_ylabel("Coarse block minus\nmatched block (pp)")
    ax.set_ylim(-13, 1.5)
    ax.set_yticks([-12, -8, -4, 0])
    ns.source_data(pd.DataFrame(rows), FIG, "d")


# e: fine-class lag against superclass lag
def panel_e(ax):
    """During the block (epochs 1-10), fine-class lag against superclass lag, mean over seeds per epoch; filled, against
    the matched block; open, against the random-direction block; lines, per-seed slopes through the origin averaged."""
    import effect_units as V
    G = V.blocking_lags()
    ax.plot([0, 8], [0, 8], color="#D1D5DB", ls="--", lw=0.8, zorder=0)
    rows, txt = [], []
    for ctl, mfc, lab in (("gradjoint", DARK, "matched block"), ("gradrand", "white", "random directions")):
        g = G[(G.control == ctl) & G.epoch.between(1, 10)]
        m = g.groupby("epoch")[["lag_superclass", "lag_fine"]].mean()
        pr = V.lag_proportion(G, ctl=ctl)
        b = pr["slope_fine_on_superclass"]["mean"]
        ax.plot([0, 8], [0, 8 * b], color=DARK, lw=1.0 if ctl == "gradjoint" else 0.7,
                ls="-" if ctl == "gradjoint" else "--", zorder=1)
        ax.plot(m.lag_superclass, m.lag_fine, "o", ms=3.4, mfc=mfc, mec=DARK, mew=0.7, ls="none", zorder=3)
        rows.append(dict(control=ctl, control_label=lab, slope=b, slope_ci_lo=pr["slope_fine_on_superclass"]["ci"][0],
                         slope_ci_hi=pr["slope_fine_on_superclass"]["ci"][1],
                         slope_per_seed=pr["slope_fine_on_superclass"]["per_seed"],
                         lag_superclass_epoch10=pr["lag_superclass_end"]["mean"],
                         lag_fine_epoch10=pr["lag_fine_end"]["mean"]))
        rows += [dict(control=ctl, control_label=lab, epoch=int(e), mean_lag_superclass=r.lag_superclass,
                      mean_lag_fine=r.lag_fine) for e, r in m.iterrows()]
        txt.append((b, pr["slope_fine_on_superclass"]["ci"]))
    ax.text(7.9, 2.0, f"× {txt[0][0]:.2f}", fontsize=7, color=DARK, ha="right", va="center")
    ax.text(7.9, 0.9, "matched\nblock", fontsize=6, color=DARK, ha="right", va="center", linespacing=1.0)
    ax.text(0.3, 7.6, f"× {txt[1][0]:.2f}", fontsize=7, color=DARK, ha="left", va="top")
    ax.text(0.3, 6.6, "random\ndirections", fontsize=6, color=DARK, ha="left", va="top", linespacing=1.0)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_xticks([0, 4, 8])
    ax.set_yticks([0, 4, 8])
    ax.set_xlabel("Superclass lag (epochs)", color=SC_TXT, labelpad=1.5)
    ax.set_ylabel("Fine-class lag (epochs)", color=CL_TXT, labelpad=1)
    ns.source_data(pd.DataFrame(rows), FIG, "e")
    print(f"[e] slopes: matched {txt[0][0]:.3f} {txt[0][1]}, random {txt[1][0]:.3f} {txt[1][1]}")


# f: per-superclass dose
def panel_f(ax):
    d = pd.read_csv(RESULTS / "GRADBLOCK" / "superclass_gradblock.csv")
    d = d[d.epoch.isin([1, 2, 3])]
    m = d.groupby("superclass")[["blocked_share", "fresh_fine_gradsc_minus_sham",
                                 "fresh_fine_gradrand_minus_sham"]].mean()
    x = 100 * m.blocked_share
    rs, ps = stats.spearmanr(m.blocked_share, m.fresh_fine_gradsc_minus_sham)
    rr, pr = stats.spearmanr(m.blocked_share, m.fresh_fine_gradrand_minus_sham)
    ax.axhline(0, color=GREYS[2], lw=0.6, ls="--", zorder=0)
    ax.scatter(x, m.fresh_fine_gradrand_minus_sham, s=10, facecolor="white", edgecolor=AMBER[2], lw=0.8, zorder=2)
    ax.scatter(x, m.fresh_fine_gradsc_minus_sham, s=12, color=AMBER[0], lw=0, zorder=3)
    b = np.polyfit(x, m.fresh_fine_gradsc_minus_sham, 1)
    xx = np.linspace(x.min(), x.max(), 10)
    ax.plot(xx, np.polyval(b, xx), color=AMBER[0], lw=0.9, zorder=2)
    ax.text(0.03, 0.04, f"ρ = {rs:.2f}".replace("-", "−"), transform=ax.transAxes, fontsize=7, color=DARK,
            ha="left", va="bottom")
    ax.text(36.6, -11.6, "coarse block", fontsize=6, color=AMBER_T, ha="right", va="center")
    ax.text(40.3, 1.4, "random directions", fontsize=6, color=CL_TXT, ha="right", va="bottom")
    ax.set_xlim(19.5, 40.5)
    ax.set_ylim(-12.6, 3.6)
    ax.set_xticks([20, 30, 40])
    ax.set_yticks([-10, -5, 0])
    ax.set_xlabel("Learning signal blocked (%)", color=SC_TXT, labelpad=1.5)
    ax.set_ylabel("Fine-class change vs no\nblock, epochs 1–3 (pp)", color=CL_TXT)
    rows = m.reset_index().rename(columns={"blocked_share": "blocked_share_of_superclass_learning_signal",
                                           "fresh_fine_gradsc_minus_sham": "coarse_block_minus_no_block_pp",
                                           "fresh_fine_gradrand_minus_sham": "random_direction_block_minus_no_block_pp"})
    rows["spearman_coarse_block"], rows["p_coarse_block"], rows["spearman_random"], rows["p_random"] = rs, ps, rr, pr
    ns.source_data(rows, FIG, "f")
    print(f"[f] coarse block rho {rs:.3f} P {ps:.4f}; random rho {rr:.3f} P {pr:.3f}")
    return rs, ps


# g: ten manipulations, numbered
SYN_MARK = {"label noise": "o", "blocking": "s", "suppression": "^"}
SYN_NAME = {"Label noise, between vs within": "noise across vs within superclasses",
            "Label noise, similarity-matched vs within": "noise at matched similarity vs within",
            "Blocking vs random directions": "coarse block vs random directions",
            "Blocking vs jointly matched": "coarse block vs matched block",
            "Random-grouping block vs jointly matched": "random grouping vs matched block",
            "Upper-stage block, fully matched": "third-stage block, fully matched",
            "Random-direction block vs sham": "random directions vs no block",
            "Jointly matched block vs sham": "matched block vs no block",
            "Suppression vs random directions": "suppression vs random directions",
            "Suppression vs sham": "suppression vs no block"}
# number label offset (data units) and alignment for each point in the main axes; None = drawn in the inset only
SYN_LAB = {1: (1.3, 0.0, "left"), 2: None, 3: (0.0, 1.8, "center"), 4: (-1.3, 0.0, "right"),
           5: (1.3, -0.3, "left"), 6: None, 7: None, 8: (-1.3, 0.2, "right"), 9: (0.0, 1.8, "center"),
           10: (0.0, -1.9, "center")}
ZOOM = (5.6, 8.5, 3.0, 5.6)      # x0, x1, y0, y1 of the inset


def panel_g(ax, key_x, key_y, cv):
    import effect_units as V
    S, st = V.synthesis()
    ax.plot([0, 40], [0, 40], color="#D1D5DB", ls="--", lw=0.8, zorder=0)
    xx = np.array([0, 36])
    ax.plot(xx, st["intercept"] + st["slope"] * xx, color=DARK, lw=0.9, zorder=1)
    ins = ax.inset_axes([0.62, 0.05, 0.34, 0.30])
    for i, r in enumerate(S.itertuples(), start=1):
        mk, mfc = SYN_MARK[r.family], (DARK if r.targeted else "white")
        ax.errorbar(r.superclass_lost_pct, r.fine_lost_pct,
                    xerr=[[r.superclass_lost_pct - r.superclass_ci[0]], [r.superclass_ci[1] - r.superclass_lost_pct]],
                    yerr=[[r.fine_lost_pct - r.fine_ci[0]], [r.fine_ci[1] - r.fine_lost_pct]], fmt="none",
                    ecolor="#E5E7EB", elinewidth=0.7, capsize=0, zorder=2)
        ax.plot(r.superclass_lost_pct, r.fine_lost_pct, mk, ms=4.6, mfc=mfc, mec=DARK, mew=0.9, zorder=4)
        if SYN_LAB[i] is None:
            ins.plot(r.superclass_lost_pct, r.fine_lost_pct, mk, ms=5, mfc=mfc, mec=DARK, mew=0.9)
            ins.text(r.superclass_lost_pct + 0.28, r.fine_lost_pct, str(i), fontsize=6, color=DARK, ha="left",
                     va="center")
        else:
            dx, dy, ha = SYN_LAB[i]
            ax.text(r.superclass_lost_pct + dx, r.fine_lost_pct + dy, str(i), fontsize=6, color=DARK, ha=ha,
                    va="center", zorder=5)
        # numbered key beside the axes (marker + number + name)
        yk = key_y + (i - 1) * 3.1
        fig = cv.fig
        fig.add_artist(Line2D([(key_x + 1.0) / cv.w], [1 - yk / cv.h], transform=fig.transFigure, ls="none",
                              marker=mk, ms=3.6, mfc=mfc, mec=DARK, mew=0.8))
        fig.text((key_x + 2.6) / cv.w, 1 - yk / cv.h, f"{i}", fontsize=6, ha="left", va="center", color=DARK)
        fig.text((key_x + 6.0) / cv.w, 1 - yk / cv.h, SYN_NAME[r.manipulation], fontsize=6, ha="left", va="center",
                 color=DARK)
    x0, x1, y0, y1 = ZOOM
    ins.set_xlim(x0, x1); ins.set_ylim(y0, y1)
    ins.set_xticks([]); ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_visible(True); sp.set_color(GREYS[2]); sp.set_linewidth(0.5)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=GREYS[2], lw=0.5, zorder=6))
    ax.text(0.03, 0.97, f"r = {st['pearson_r']:.2f}", transform=ax.transAxes, fontsize=7, color=DARK, ha="left",
            va="top")
    ax.text(0.03, 0.88, f"slope {st['slope']:.2f}", transform=ax.transAxes, fontsize=7, color=DARK, ha="left",
            va="top")
    ax.text(29.5, 33.5, "equal loss", fontsize=6, color="#9CA3AF", ha="right", va="center")
    ax.set_xlim(0, 39)
    ax.set_ylim(0, 39)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_yticks([0, 10, 20, 30])
    ax.set_xlabel("Superclass learning lost (%)", color=SC_TXT, labelpad=1.5)
    ax.set_ylabel("Fine-grained learning lost (%)", color=CL_TXT)
    out = S.copy()
    out.insert(0, "number", range(1, len(S) + 1))
    out.insert(2, "label", [SYN_NAME[m] for m in S.manipulation])
    out["superclass_ci_lo"], out["superclass_ci_hi"] = zip(*S.superclass_ci)
    out["fine_ci_lo"], out["fine_ci_hi"] = zip(*S.fine_ci)
    ns.source_data(out.drop(columns=["superclass_ci", "fine_ci"]), FIG, "g")
    ns.source_data(pd.DataFrame([st]), FIG, "g_fit")
    print(f"[g] r {st['pearson_r']:.3f} slope {st['slope']:.3f} rho {st['spearman_rho']:.3f} P {st['spearman_p']:.4f}")


# h: release
REL_EPOCHS = list(range(6, 16))
REL_BASE = list(range(6, 11))


def release_data():
    R = pd.read_csv(RESULTS / "RDT_completion" / "rdt_completion_long.csv")
    q = R[(R.kind == "events") & (R.layer == "all") & (R.tset == "common") & R.measure.isin(["Ab-E", "Di-E"])
          & R.arm.isin(["gradsc", "gradjoint"]) & R.epoch.between(6, 15)]
    rows = []
    for arm in ("gradsc", "gradjoint"):
        for m, lev in (("Ab-E", "superclass"), ("Di-E", "class")):
            for s in SEEDS:
                v = q[(q.arm == arm) & (q.measure == m) & (q.seed == s)].set_index("epoch").value.reindex(REL_EPOCHS)
                base = v.loc[REL_BASE].mean()
                post = [e for e in range(11, 16) if v.loc[e] > base]
                onset = post[0] if post else np.nan
                for e in REL_EPOCHS:
                    rows.append(dict(arm=arm, arm_label=ARMS[arm][0], level=lev, process=m, seed=s, epoch=e,
                                     events=v.loc[e], block_level_epochs_6_10=base, ratio_to_block_level=v.loc[e] / base,
                                     ratio_to_previous_epoch=(v.loc[e] / v.loc[e - 1]) if e > 6 else np.nan,
                                     onset_epoch=onset))
    return pd.DataFrame(rows)


def panel_h(cv, x, y, w_each, gap, h_main, h_strip):
    D = release_data()
    LO, HI = 0.15, 14
    axes = []
    for k, (arm, show) in enumerate((("gradsc", ("superclass", "class")), ("gradjoint", ("class",)))):
        xa = x + k * (w_each + gap)
        ax = cv.ax(xa, y, w_each, h_main)
        st = cv.ax(xa, y + h_main + 0.8, w_each, h_strip)
        axes.append(ax)
        ri, ls = ARMS[arm][1], ARMS[arm][2]
        ax.axvspan(7.5, 10.5, color=WINDOW, lw=0, zorder=0)
        ax.axvline(10.5, color=GREYS[1], lw=0.6, zorder=1)
        ax.axhline(1, color=GREYS[2], lw=0.5, ls=(0, (1.5, 1.5)), zorder=1)
        for lev, ramp in (("superclass", BLUE), ("class", AMBER)):
            g = D[(D.arm == arm) & (D.level == lev)]
            P = g.pivot(index="seed", columns="epoch", values="ratio_to_block_level").loc[SEEDS]
            c = ramp[ri]
            for s in SEEDS:
                ax.plot(P.columns, P.loc[s].values, color=c, lw=0.45, alpha=0.45, zorder=2)
            gm = np.exp(np.log(P.values).mean(0))
            ax.plot(P.columns, gm, color=c, lw=1.4, ls=ls, zorder=3)
        ax.set_yscale("log", base=2)
        ax.set_ylim(LO, HI)
        ax.set_yticks([0.5, 1, 2, 4, 8])
        ax.set_yticklabels(["½", "1", "2", "4", "8"])
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlim(7.6, 14.4)
        ax.set_xticks([8, 10, 12, 14])
        ax.tick_params(labelbottom=False)
        ax.text(10.6, HI * 0.93, "release", fontsize=6, color=GREY_T, ha="left", va="top")
        ax.set_title(ARMS[arm][0], fontsize=6.5, pad=2, color=DARK)
        # onsets: triangles on the x-axis at the median seed onset; strip, one dot per seed
        st.set_xlim(7.6, 14.4)
        st.set_ylim(-0.6, 1.6)
        st.set_yticks([])
        st.spines["left"].set_visible(False)
        st.set_xticks([8, 10, 12, 14])
        st.set_xlabel("Epoch", labelpad=1.0)
        on_med = {}
        for row, (lev, ramp) in zip((1, 0), (("superclass", BLUE), ("class", AMBER))):
            if lev not in show:
                continue
            on = D[(D.arm == arm) & (D.level == lev)].groupby("seed").onset_epoch.first().loc[SEEDS]
            med = float(np.median(on.values))
            on_med[lev] = med
            c = ramp[ri]
            ax.plot([med], [LO], marker="^", ms=4.4, color=c, mec="white", mew=0.4, clip_on=False, zorder=6)
            for e_, grp in on.groupby(on):
                n = len(grp)
                offs = (np.arange(n) - (n - 1) / 2) * 0.17
                st.scatter(e_ + offs, np.full(n, row), s=5, color=c, lw=0, zorder=3, clip_on=False)
                if n == 1 and len(on) > 1:
                    st.text(e_ + 0.2, row, f"{grp.index[0]}", fontsize=6, color=GREY_T, ha="left", va="center", clip_on=False)
        if arm == "gradsc":
            a_, b_ = on_med["superclass"], on_med["class"]
            yb = 0.205
            ax.plot([a_, a_, b_, b_], [yb * 0.9, yb, yb, yb * 0.9], color=DARK, lw=0.6, zorder=5)
            ax.text(b_ + 0.15, yb, "1 epoch", fontsize=6, color=DARK, ha="left", va="center")
        if k == 0:
            ax.set_ylabel("Events per epoch\n(× epochs 6–10)", labelpad=1.5)
            st.text(7.45, 0.5, "onset", fontsize=6, color=GREY_T, ha="right", va="center")
        else:
            ax.tick_params(labelleft=False)
    ns.source_data(D, FIG, "h")
    on = D.groupby(["arm", "level", "seed"]).onset_epoch.first().unstack("seed")
    print("[h] onsets:\n" + on.to_string())
    r = D[D.epoch.isin([11, 12])].pivot_table(index=["arm", "level", "epoch"], columns="seed",
                                                 values="ratio_to_previous_epoch")
    print("[h] ratio to previous epoch:\n" + r.round(2).to_string())
    return axes


def main():
    ns.apply()
    sd_dir = ns.FINAL_OUT / "source_data"
    for f in sd_dir.glob(f"{FIG}_*.csv"):
        f.unlink()
    cv = Canvas(W_MM, H_MM)
    L = lambda arm: ("line", ARMS[arm][0], dict(c=(GREYS[2] if ARMS[arm][1] is None else DARK), ls=ARMS[arm][2],
                                               lw=ARMS[arm][3]))
    key_strip(cv, [
        [L("gradsc"), L("gradjoint"), L("gradrand"), L("sham"),
         ("shade", "block, epochs 1–10", dict(c=WINDOW, ec="#D1D5DB"))],
        [("patch", "superclass measures", dict(c=BLUE[0], alpha=BAR_A)),
         ("patch", "class measures", dict(c=AMBER[0], alpha=BAR_A)),
         ("hatch", "class within the true superclass (d)", dict(c=AMBER[0]))],
    ], y_mm=2.5, x0_mm=3)
    # row 1: a | b | c
    T1, H1 = 21, 36
    panel_a(cv, 15, T1, 50, 16.5, 3.0)
    cv.letter("a", 1, T1 - 8.5)
    cv.title("Both fall behind during the\nblock and largely recover", 1 + 3.5, T1 - 8)
    b = cv.ax(89, T1, 40, H1)
    rg = panel_b(b)
    cv.letter("b", 73, T1 - 8.5)
    cv.title(f"A random grouping holding {100 * rg:.0f}% of the\ncoarse structure costs about as much", 73 + 3.5,
             T1 - 8)
    c = cv.ax(150, T1, 28, H1)
    panel_c(c)
    cv.letter("c", 136, T1 - 8.5)
    cv.title("With doses fully matched,\nboth still lag", 136 + 3.5, T1 - 8)
    # row 2: d | e | f
    T2, H2 = 76, 30
    d = cv.ax(17, T2, 68, H2)
    panel_d(d)
    cv.letter("d", 1, T2 - 8.5)
    cv.title("Superclass accuracy carries most of the fine-class deficit,\nand the class within the true superclass "
             "falls behind as well", 1 + 3.5, T2 - 8)
    e = cv.ax(102, T2, 27, 27)
    panel_e(e)
    cv.letter("e", 91, T2 - 8.5)
    cv.title("Fine-class lag is two thirds\nof the superclass lag", 91 + 3.5, T2 - 8)
    f = cv.ax(149, T2, 30, 27)
    panel_f(f)
    cv.letter("f", 134, T2 - 8.5)
    cv.title("Superclasses that lose more\nsignal fall further behind", 134 + 3.5, T2 - 8)
    # row 3: g (+ numbered key) | h
    T3 = 125
    g = cv.ax(13, T3, 40, 38)
    panel_g(g, 57, T3 + 3, cv)
    cv.letter("g", 1, T3 - 8.5)
    cv.title("Across ten manipulations, fine-grained learning\nloses two thirds of the coarse loss", 1 + 3.5, T3 - 8)
    panel_h(cv, 128, T3 + 2, 23, 4, 26, 4)
    cv.letter("h", 112, T3 - 8.5)
    cv.title("After release, superclass events surge first;\nclass events rise one epoch later", 112 + 3.5, T3 - 8)
    out = ns.save_final(cv.fig, FIG, W_MM, ed=True)
    print("saved", out)


if __name__ == "__main__":
    main()
