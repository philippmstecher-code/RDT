#!/usr/bin/env python3
"""Blocking the coarse learning signal slows fine-grained learning through misrouted images: panels of the blocking
experiment, and the drawing helpers (tci, Canvas, bar, cat_axis, key_strip) that routing/routing_panels.py uses for
Extended Data Fig. 6. Run directly, the script draws the panels below on one canvas.

Learning signal along the superclass directions blocked in epochs 1-10, forward pass intact.
  a  superclass assignment (fresh read-out, top) and fine-class accuracy (bottom), epochs 1-30, five arms
  b  superclass vs random-grouping directions blocked, minus energy-matched, epochs 1-3
  c  fully matched dose (layer3 to the pooled layer, vs the jointly matched block): epochs behind at epoch 10
  d  blocking deficit split into superclass assignment and class within superclass, epochs 1-3
     (intervention_panels.panel_g_routing)
  e  misrouted share of the learning signal (filled) and of the images (hatched), minus energy-matched
  f  learning signal shared within superclasses under each block, epochs 1-15
  g  fine-class lag against superclass lag during the block, per epoch; slopes per seed (intervention_panels.panel_lag_proportion)
Colours: intervention arms from ns.ARM; superclass dashed, matched controls solid, random directions dotted, redrawn
random directions dash-dot; outcome bars blue (superclass assignment) / amber (fine-class accuracy).
Data (read-only): GRADJOINT/long.csv (a, c via effect_units), GRADBLOCK/final_long.csv (b),
SIGNAL_routing/signal_routing_long.csv (e), SHARED_signal/T2_blocking/per_epoch.csv (f).
Usage: python scripts/paper/figures/extended/blocking_routing_panels.py  (ES_FIG_OUT=<dir> redirects the output)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))           # scripts/paper/figures
sys.path.insert(0, str(HERE.parents[1]))       # scripts/paper
import nmi_style as ns  # noqa: E402
from fig_common import DOT, ERR, REF, RESULTS, SEEDS, WINDOW  # noqa: E402

FIG = "ed5"
W_MM, H_MM = 180, 178
SSI, CSI = ns.INDEX["SSI"], ns.INDEX["CSI"]   # superclass assignment blue, fine-class amber
CSI_TXT = "#B45309"                            # amber text (darker for legibility)
BAR_A = ns.BAR_ALPHA_CORE
A = ns.ARM
# arm -> (label, colour, line style, line width)
ARMS = {"gradsc": ("superclass directions", A["superclass"], "--", 1.4),
        "gradjoint": ("jointly matched", A["joint"], "-", 1.0),
        "gradenergy": ("energy-matched", A["energy"], "-", 1.0),
        "gradrand": ("random directions", A["random_directions"], ":", 1.1),
        "gradrandfresh": ("random directions, redrawn", A["random_redrawn"], "-.", 0.9),
        "sham": ("sham", A["sham"], "-", 0.8)}


# helpers (shared with routing/routing_panels.py)
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
        """Panel title at an absolute position; top-aligned."""
        self.fig.text(x / self.w, 1 - y / self.h, text, fontsize=7, va="top", ha=ha, linespacing=1.1)


def bar(ax, x, v, color, rng, width, style="filled", dot_s=6, alpha=BAR_A):
    """Bar = mean, grey seed dots, dark 95% CI. style: filled | hatched (light fill + hatch) | open."""
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


# a: superclass assignment and fine-class accuracy over training
def panel_a(cv, x, y, w, h_each, gap):
    import intervention_panels
    D = intervention_panels.gb_long()
    fa = cv.ax(x, y, w, h_each)
    fb = cv.ax(x, y + h_each + gap, w, h_each)
    order = ("sham", "gradrandfresh", "gradrand", "gradjoint", "gradsc")
    rows = []
    for ax_, col, lab, yl, yt in ((fa, "fresh_assignment", "Superclass\nassignment (%)", (15, 72), [20, 40, 60]),
                                  (fb, "val_accuracy", "Fine-class\naccuracy (%)", (5, 62), [20, 40, 60])):
        ax_.axvspan(0.5, 10.5, color=WINDOW, lw=0, zorder=0)
        for arm in order:
            lab_, c_, ls_, lw_ = ARMS[arm]
            P = D[D.arm == arm].pivot(index="seed", columns="epoch", values=col).dropna(axis=1)
            ax_.plot(P.columns, P.values.mean(0), color=c_, ls=ls_, lw=lw_, zorder=4 if arm == "gradsc" else 3)
            rows += [dict(measure=col, arm=arm, arm_label=lab_, epoch=int(e), mean=v,
                          **{f"seed_{s}": P.loc[s, e] for s in P.index}) for e, v in zip(P.columns, P.values.mean(0))]
        ax_.set_xlim(0.5, 30.5)
        ax_.set_ylim(*yl)
        ax_.set_yticks(yt)
        ax_.set_xticks([1, 10, 20, 30])
        ax_.set_ylabel(lab, labelpad=1.5)
    fa.yaxis.label.set_color(SSI)
    fb.yaxis.label.set_color(CSI_TXT)
    fa.tick_params(labelbottom=False)
    fb.set_xlabel("Epoch", labelpad=1.5)
    fa.text(5.5, 70, "block", fontsize=6.3, color="#6B7280", ha="center", va="top")
    ns.source_data(pd.DataFrame(rows), FIG, "a")
    return fa, fb


# b: blocking a random grouping
def panel_b(ax):
    D = pd.read_csv(RESULTS / "GRADBLOCK" / "final_long.csv")
    M = D[D.epoch.isin([1, 2, 3])].groupby(["arm", "seed"])[["fresh_assignment", "val_accuracy"]].mean()
    share = pd.read_csv(RESULTS / "RANDGROUP_control" / "specificity_arms.csv").groupby("arm").coarse_removed_share.mean()
    rng = np.random.default_rng(11)
    out, lows = [], []
    groups = (("gradsc", "superclass\ndirections"), ("gradrg", "random-grouping\ndirections"))
    meas = (("fresh_assignment", "superclass assignment (fresh read-out)", SSI, -0.19),
            ("val_accuracy", "fine-class accuracy (network)", CSI, 0.19))
    for i, (arm, lab) in enumerate(groups):
        for col, mlab, hue, dx in meas:
            v = (M.loc[arm][col] - M.loc["gradenergy"][col]).loc[SEEDS].values
            m, lo, hi = bar(ax, i + dx, v, hue, rng, 0.34)
            lows.append(min(lo, v.min()))
            out.append(dict(arm=arm, contrast=f"{arm} - gradenergy", epochs="1-3", measure=mlab, mean=m, ci95_lo=lo,
                            ci95_hi=hi, n_below_zero=int((v < 0).sum()), n=len(v),
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
    for i, (key, hue, lab) in enumerate((("lag_superclass_end", SSI, "superclass assignment"),
                                         ("lag_fine_end", CSI, "fine-class accuracy"))):
        r = pr[key]
        v = np.array(r["per_seed"], float)
        bar(ax, i, v, hue, rng, 0.6)   # per-seed values are stored rounded to 1e-4; the label uses the unrounded mean
        m, (lo, hi) = r["mean"], r["ci"]
        assert abs(m - v.mean()) < 1e-3
        ax.text(i, max(hi, v.max()) + 0.25, f"{m:.1f}", fontsize=6.5, color=ERR, ha="center", va="bottom")
        npos.append(int((v > 0).sum()))
        out.append(dict(contrast="superclass - jointly matched, layer3 to pooled layer", epoch=10,
                        measure=f"{lab}, epochs behind", mean=m, ci95_lo=lo, ci95_hi=hi, n_above_zero=npos[-1],
                        **{f"seed_{s}": x_ for s, x_ in zip(SEEDS, v)}))
    sl = pr["slope_fine_on_superclass"]
    out.append(dict(contrast="superclass - jointly matched, layer3 to pooled layer", epoch="1-10",
                    measure="fine-class lag per epoch of superclass lag", mean=sl["mean"], ci95_lo=sl["ci"][0],
                    ci95_hi=sl["ci"][1], **{f"seed_{s}": x_ for s, x_ in zip(SEEDS, sl["per_seed"])}))
    cat_axis(ax, ["super-\nclass", "fine\nclass"], fs=6.3)
    for t, col in zip(ax.get_xticklabels(), (SSI, CSI_TXT)):
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


# per-superclass dose-response (not placed in this figure)
def panel_d(ax):
    d = pd.read_csv(RESULTS / "GRADBLOCK" / "superclass_gradblock.csv")
    d = d[d.epoch.isin([1, 2, 3])]
    m = d.groupby("superclass")[["blocked_share", "fresh_fine_gradsc_minus_sham",
                                 "fresh_fine_gradrand_minus_sham"]].mean()
    x = 100 * m.blocked_share
    rs, ps = stats.spearmanr(m.blocked_share, m.fresh_fine_gradsc_minus_sham)
    rr, pr = stats.spearmanr(m.blocked_share, m.fresh_fine_gradrand_minus_sham)
    ax.axhline(0, color=REF, lw=0.6, ls="--", zorder=0)
    ax.scatter(x, m.fresh_fine_gradrand_minus_sham, s=11, facecolor="white", edgecolor=A["random_directions"],
               lw=0.7, zorder=2)
    ax.scatter(x, m.fresh_fine_gradsc_minus_sham, s=13, color=A["superclass"], lw=0, zorder=3)
    b = np.polyfit(x, m.fresh_fine_gradsc_minus_sham, 1)
    xx = np.linspace(x.min(), x.max(), 10)
    ax.plot(xx, np.polyval(b, xx), color=A["superclass"], lw=0.9, zorder=2)
    ax.set_ylim(-12.6, 2.8)
    ax.text(0.98, 0.99, minus(f"ρ = {rs:.2f}, P = {ps:.3f}"), transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, color=ERR)
    ax.set_xlabel("Learning signal blocked (% of the superclass’s)", labelpad=1.5)
    ax.set_ylabel("Fine-class change vs\nsham, epochs 1–3 (pp)")
    ax.set_xticks([20, 25, 30, 35, 40])
    ax.set_yticks([-10, -5, 0])
    rows = m.reset_index().rename(columns={"fresh_fine_gradsc_minus_sham": "superclass_block_minus_sham_pp",
                                           "fresh_fine_gradrand_minus_sham": "random_block_minus_sham_pp"})
    rows["spearman_superclass"], rows["p_superclass"], rows["spearman_random"], rows["p_random"] = rs, ps, rr, pr
    ns.source_data(rows, FIG, "d")
    print(f"[d] superclass rho {rs:.3f} P {ps:.4f}; random rho {rr:.3f} P {pr:.3f}")
    return rs, ps


# e: misrouted images keep the learning signal
def panel_e(ax):
    D = pd.read_csv(RESULTS / "SIGNAL_routing" / "signal_routing_long.csv")
    M = D[D.epoch.isin([1, 2, 3])].groupby(["arm", "seed"]).mean(numeric_only=True)
    rng = np.random.default_rng(3)
    arms = (("gradsc", "superclass\ndirections", A["superclass"]),
            ("gradrg", "random-grouping\ndirections", A["random_grouping"]))
    meas = (("signal_share_misrouted", "learning signal", -0.2, "filled"),
            ("image_share_misrouted", "images", 0.2, "hatched"))
    rows, vals = [], {}
    for i, (arm, lab, col_arm) in enumerate(arms):
        for col, mlab, dx, style in meas:
            v = 100 * (M.loc[arm][col] - M.loc["gradenergy"][col]).loc[SEEDS].values
            m, lo, hi = bar(ax, i + dx, v, col_arm, rng, 0.36, style=style)
            vals[(arm, col)] = (m, max(hi, v.max()))
            rows.append(dict(arm=arm, contrast=f"{arm} - gradenergy", epochs="1-3", measure=f"{col} (pp)", mean=m,
                             ci95_lo=lo, ci95_hi=hi, n_seeds_above_zero=int((v > 0).sum()),
                             **{f"seed_{s}": x_ for s, x_ in zip(SEEDS, v)}))
    ms, ts = vals[("gradsc", "signal_share_misrouted")]
    mi, ti = vals[("gradsc", "image_share_misrouted")]
    ax.text(0, max(ts, ti) + 0.5, f"+{ms:.1f} / +{mi:.1f}", fontsize=6.5, color=ERR, ha="center", va="bottom")
    cat_axis(ax, [a[1] for a in arms])
    ax.set_ylim(-0.8, 9.5)
    ax.set_yticks([0, 4, 8])
    ax.set_ylabel("Misrouted share, minus energy-\nmatched, epochs 1–3 (pp)")
    sd = pd.DataFrame(rows)
    ns.source_data(sd, FIG, "e")
    print("[e] " + "; ".join(f"{r.arm} {r.measure}: {r.mean:+.2f} [{r.ci95_lo:+.2f}, {r.ci95_hi:+.2f}] "
                             f"{r.n_seeds_above_zero}/5" for r in sd.itertuples()))


# f: learning signal shared within superclasses
def panel_f(ax):
    P = pd.read_csv(RESULTS / "SHARED_signal/T2_blocking/per_epoch.csv")
    ax.axvspan(0.5, 10.5, color=WINDOW, lw=0, zorder=0)
    rows = []
    for arm in ("sham", "gradenergy", "gradrand", "gradsc"):
        lab, c_, ls_, lw_ = ARMS[arm]
        g = P[P.arm == arm].sort_values("epoch")
        if arm in ("sham", "gradsc"):
            ax.fill_between(g.epoch, g.excess_mean_ci_lo, g.excess_mean_ci_hi, color=c_, alpha=0.18, lw=0, zorder=2)
        ax.plot(g.epoch, g.excess_mean_mean, color=c_, ls=ls_, lw=lw_, zorder=4 if arm == "gradsc" else 3)
        rows += [dict(arm=arm, arm_label=lab, epoch=int(r.epoch), mean=r.excess_mean_mean, ci_lo=r.excess_mean_ci_lo,
                      ci_hi=r.excess_mean_ci_hi) for r in g.itertuples()]
    ax.axhline(0, color=REF, lw=0.6, ls="--", zorder=0)
    ax.set_xlim(0.5, 15.5)
    ax.set_xticks([1, 5, 10, 15])
    ax.set_ylim(-0.005, 0.105)
    ax.set_yticks([0, 0.05, 0.1])
    ax.set_xlabel("Epoch", labelpad=1.5)
    ax.set_ylabel("Learning signal shared\nwithin superclass")
    ns.source_data(pd.DataFrame(rows), FIG, "f")


def main():
    ns.apply()
    sd_dir = ns.FINAL_OUT / "source_data"
    for f in sd_dir.glob(f"{FIG}_*.csv"):
        f.unlink()
    cv = Canvas(W_MM, H_MM)
    L = lambda arm, **kw: ("line", ARMS[arm][0], dict(c=ARMS[arm][1], ls=ARMS[arm][2], lw=ARMS[arm][3], **kw))
    key_strip(cv, [
        [L("gradsc"), L("gradjoint"), L("gradenergy"), L("gradrand"), L("gradrandfresh"), L("sham")],
        [("patch", "random-grouping directions", dict(c=A["random_grouping"])),
         ("shade", "block, epochs 1\u201310", dict(c=WINDOW, ec="#D1D5DB"))],
        [("patch", "superclass assignment", dict(c=SSI)), ("patch", "fine-class accuracy", dict(c=CSI)),
         ("patch", "learning signal", dict(c="#9CA3AF")), ("hatch", "images", dict(c="#6B7280"))],
    ], y_mm=2.5, x0_mm=3)
    # row 1: a | b | c
    T1, H1 = 26, 40
    xa, wa = 15, 50
    panel_a(cv, xa, T1, wa, 18.5, 3.0)
    cv.letter("a", 1, T1 - 9)
    cv.title("Both fall behind during the\nblock and recover", 1 + 3.5, T1 - 8.5)
    xb, wb = 89, 40
    b = cv.ax(xb, T1, wb, H1)
    rg = panel_b(b)
    cv.letter("b", 73, T1 - 9)
    cv.title(f"A random grouping holding {100 * rg:.0f}% of the\ncoarse structure costs about as much", 73 + 3.5,
             T1 - 8.5)
    xc, wc = 150, 28
    c = cv.ax(xc, T1, wc, H1)
    panel_c(c)
    cv.letter("c", 136, T1 - 9)
    cv.title("With doses fully matched,\nboth still lag", 136 + 3.5, T1 - 8.5)
    # row 2: d | e | f
    T2, H2 = 85, 34
    xd, wd = 15, 50
    # d: routing decomposition drawn by intervention_panels.panel_g_routing, source data redirected to this figure
    import intervention_panels as F5d
    d = cv.ax(xd + 6, T2 + 5, wd - 6, H2 - 5)
    _sdd = ns.source_data
    ns.source_data = lambda df, name, letter: _sdd(df, FIG, "d") if name == F5d.NAME else _sdd(df, name, letter)
    F5d.panel_g_routing(d)
    ns.source_data = _sdd
    d.set_title("")
    for t in d.texts:                       # narrow panel: break the right-hand header
        if t.get_text() == "class within superclass":
            t.set_text("class within\nsuperclass")
    cv.letter("d", 1, T2 - 9)
    cv.title("Superclass assignment carries most\nof the fine-class deficit", 1 + 3.5, T2 - 8.5)
    xe, we = 89, 40
    e = cv.ax(xe, T2, we, H2)
    panel_e(e)
    cv.letter("e", 73, T2 - 9)
    cv.title("Misrouted images keep\nsupplying learning signal", 73 + 3.5, T2 - 8.5)
    xf, wf = 150, 28
    f = cv.ax(xf, T2, wf, H2)
    panel_f(f)
    cv.letter("f", 136, T2 - 9)
    cv.title("Blocking keeps the learning\nsignal shared longer", 136 + 3.5, T2 - 8.5)
    # row 3: g (fine-class lag against superclass lag; drawn by intervention_panels)
    import intervention_panels as F5
    _sd = ns.source_data
    ns.source_data = lambda df, name, letter: _sd(df, FIG, "g") if name == F5.NAME else _sd(df, name, letter)
    tmp = cv.fig.add_axes([0, 0, 0.01, 0.01])
    G = F5.panel_e_lag(tmp)
    tmp.remove()
    T3, H3 = 138, 26
    g = cv.ax(15, T3, 30, H3)
    F5.panel_lag_proportion(g, G)
    g.set_title("")
    ns.source_data = _sd
    cv.letter("g", 1, T3 - 9)
    cv.title("Fine-class lag is two thirds\nof the superclass lag", 1 + 3.5, T3 - 8.5)
    out = ns.save_final(cv.fig, FIG, W_MM, ed=True)
    print("saved", out)


if __name__ == "__main__":
    main()
