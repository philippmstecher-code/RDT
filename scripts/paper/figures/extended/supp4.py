#!/usr/bin/env python3
"""Supplementary Fig. 4: analysis checks.

a-c   observed vs permutation-null SSI; adaptive vs floor thresholds; null-baseline p values (9 standard runs;
      sae_results.json null_baseline / metadata.adaptive_thresholds, via validation_panels.load_core)
d-f   selectivity vs activation magnitude (magnitude_panels.py; cached per-feature table)
g, h  ResNet-18 seed 256: Ab-E / Di-E counts and churn, 4x vs 8x SAE
i, j  ResNet-18 seed 42: mean SSI / CSI and churn, 50 vs 200 epochs
Line styles as in Extended Data Fig. 1c: solid circles = 4x SAE, 50 epochs; dotted open diamonds = 8x SAE;
dash-dot open triangles = 200 epochs.
Usage: python scripts/paper/figures/extended/supp4.py  (ES_FIG_OUT=<dir> redirects the output)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import nmi_style as ns  # noqa: E402
import core_runs_lib as L4  # noqa: E402
import validation_panels  # noqa: E402
import magnitude_panels as m7  # noqa: E402
from checks_lib import ARCHS, MMCanvas  # noqa: E402

FIG = "supp4"
W_MM, H_MM = 180, 160
RB = ns.ARCH["ResNet-18"]
DARK = ns.REF_DARK
TS, FS = 6.6, 6.2
VAR = {"base": dict(ls="-", marker="o", filled=True), "8x": dict(ls=":", marker="D", filled=False),
       "200ep": dict(ls="-.", marker="^", filled=False)}


def style(ax):
    ax.tick_params(labelsize=FS, pad=1.5)
    ax.xaxis.label.set_size(6.5)
    ax.yaxis.label.set_size(6.5)


def title(ax, t):
    ax.set_title(t, fontsize=TS, pad=3, linespacing=1.1)


def line(ax, y, color, var, alpha=1.0):
    v = VAR[var]
    ax.plot(range(len(y)), y, color=color, ls=v["ls"], marker=v["marker"], ms=2.5, lw=0.9, alpha=alpha,
            mfc=color if v["filled"] else "white", mec=color, mew=0.7)


# a-c
def panels_abc(ax_a, ax_b, ax_c, N, T):
    validation_panels.panel_d(ax_a, N)
    validation_panels.panel_e(ax_b, T)
    validation_panels.panel_f(ax_c, N)
    for ax in (ax_a, ax_b, ax_c):
        ax.get_legend().remove()
        style(ax)
    ax_c.set_ylim(5e-5, 1.5)
    ax_c.set_ylabel("p value (permutation)")
    title(ax_a, "Observed SSI exceeds the null\nat every checkpoint")
    title(ax_b, "Adaptive thresholds lie well\nabove the fixed floors")
    title(ax_c, f"p ≤ {N.p_value.max():.3f} at every checkpoint\nof every run")
    assert (N.observed_ssi > N.null_ssi).all()


# d-f
def panels_def(ax_d, ax_e, ax_f, fig):
    df = m7.collect()
    cats = m7.categories(df)
    n = {k: int(v.sum()) for k, v in cats.items()}
    n["total"] = len(df)
    assert n == m7.CORE_N, n
    st = m7.statistics(df, cats)
    present = [k for k in m7.CAT_ORDER if n[k]]
    colors, labels = [m7.CAT_COLOR[k] for k in present], [m7.CAT_LABEL[k] for k in present]
    m7.violin_box(ax_d, [df.cond_mag[cats[k]].values for k in present], colors, labels,
                  "Conditional magnitude\n(mean |act.| when active)")
    m7.violin_box(ax_e, [df.sparsity[cats[k]].values for k in present], colors, labels,
                  "Activation sparsity\n(frac. samples active)")
    alive = df[(df.ssi > 0) | (df.csi > 0)]
    rng = np.random.default_rng(42)
    idx = rng.choice(len(alive), 6000, replace=False) if len(alive) > 6000 else np.arange(len(alive))
    sub = alive.iloc[idx]
    log_mag = np.log10(sub.cond_mag.values + 1e-8)
    sc = ax_f.scatter(sub.ssi, sub.csi, c=log_mag, s=1.2, alpha=0.45, cmap="viridis", rasterized=True, linewidths=0)
    x0, y0, w, h = ax_f.get_position().bounds
    cax = fig.add_axes([x0 + w + 2.0 / W_MM, y0 + 0.1 * h, 1.6 / W_MM, 0.8 * h])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("log10 conditional magnitude", fontsize=6)
    cb.ax.tick_params(labelsize=6, pad=1)
    cb.outline.set_linewidth(0.5)
    t0 = df.iloc[0]
    ax_f.axvline(t0.ssi_t, color="#9CA3AF", ls="--", lw=0.6, zorder=1)
    ax_f.axhline(t0.csi_t, color="#9CA3AF", ls="--", lw=0.6, zorder=1)
    ax_f.set_xlabel("SSI (superclass selectivity)")
    ax_f.set_ylabel("CSI (class selectivity)")
    ax_f.set_xlim(-0.02, 1.02)
    ax_f.set_ylim(-0.02, 1.02)
    for ax in (ax_d, ax_e, ax_f):
        style(ax)
    ax_d.tick_params(axis="x", labelsize=6)
    ax_e.tick_params(axis="x", labelsize=6)
    mw = st[(st.test.str.startswith("Mann")) & (st.group == "High-SSI vs High-CSI")].effect_size_r.iloc[0]
    rho = st[st.test.str.startswith("Spearman")].statistic.abs().max()
    title(ax_d, "High-SSI and high-CSI features\nfire with similar magnitude")
    title(ax_e, "High-SSI and high-CSI features\nare comparably sparse")
    title(ax_f, f"No magnitude gradient in\nSSI–CSI space (|ρ| < {np.ceil(rho * 100) / 100:.2f})")
    cat_cols = pd.DataFrame({f"is_{k.replace(' ', '_').replace('-', '_')}": v for k, v in cats.items()})
    return (pd.concat([df, cat_cols], axis=1), sub.assign(log10_cond_mag=log_mag), st,
            dict(mann_whitney_r_ssi_csi=round(float(mw), 3), max_abs_rho=round(float(rho), 4)))


# g-j
def tr_style(ax, n, xlabel="Training transition"):
    ax.set_xticks(range(0, n, 2))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xlabel(xlabel)
    style(ax)


def panels_ghij(ax_g, ax_h, ax_i, ax_j):
    p = {k: L4.process_intensity(*v) for k, v in L4.ED4_LANES.items()}
    s = {k: L4.selectivity_evolution(*v) for k, v in L4.ED4_LANES.items()}
    for k, col in (("ab_h", ns.PROC["Ab-E"]), ("di_h", ns.PROC["Di-E"])):
        line(ax_g, [t[k] / 1e3 for t in p["4x_s256"]], col, "base")
        line(ax_g, [t[k] / 1e3 for t in p["8x"]], col, "8x")
    tr_style(ax_g, 10)
    ax_g.set_ylim(0, 17.5)
    ax_g.set_ylabel("Events (thousands)")
    title(ax_g, "8×: more Di-E events,\nthe same rise")
    line(ax_h, [100 * t["churn"] for t in p["4x_s256"]], RB, "base")
    line(ax_h, [100 * t["churn"] for t in p["8x"]], RB, "8x")
    tr_style(ax_h, 10)
    ax_h.set_ylim(10, 85)
    ax_h.set_ylabel("Churn rate (%)")
    title(ax_h, "Churn is highest first\nat both SAE sizes")
    for k, col in (("mean_ssi", ns.INDEX["SSI"]), ("mean_csi", ns.INDEX["CSI"])):
        line(ax_i, [e[k] for e in s["4x"]], col, "base")
        line(ax_i, [e[k] for e in s["200ep"]], col, "200ep")
    tr_style(ax_i, 11, "Checkpoint")
    ax_i.set_ylim(0.10, 0.40)
    ax_i.set_ylabel("Mean SSI or CSI")
    title(ax_i, "Superclass selectivity\nsaturates first")
    line(ax_j, [100 * t["churn"] for t in p["4x"]], RB, "base")
    line(ax_j, [100 * t["churn"] for t in p["200ep"]], RB, "200ep")
    tr_style(ax_j, 10)
    ax_j.set_ylim(10, 85)
    ax_j.set_ylabel("Churn rate (%)")
    title(ax_j, "Churn is highest first\nunder both durations")

    def tr_df(keys, names):
        return pd.DataFrame([{"run": names[r], "transition_index": i, "transition": t["transition"], "ab_h": t["ab_h"],
                              "di_h": t["di_h"], "churn_pct": 100 * t["churn"]}
                             for r in keys for i, t in enumerate(p[r])])
    gh = tr_df(["4x_s256", "8x"], {"4x_s256": "4x_seed256", "8x": "8x_seed256"})
    ij = tr_df(["4x", "200ep"], {"4x": "50ep_seed42", "200ep": "200ep_seed42"})
    sel = pd.DataFrame([{"run": "50ep_seed42" if r == "4x" else "200ep_seed42", "checkpoint": e["checkpoint"],
                         "mean_ssi": e["mean_ssi"], "mean_csi": e["mean_csi"]} for r in ("4x", "200ep") for e in s[r]])
    return (gh[["run", "transition_index", "transition", "ab_h", "di_h"]],
            gh[["run", "transition_index", "transition", "churn_pct"]], sel,
            ij[["run", "transition_index", "transition", "churn_pct"]])


# legend strip
def legend_strip(fig):
    grey = "#9CA3AF"

    def var(v, label):
        s = VAR[v]
        return Line2D([], [], color=DARK, ls=s["ls"], marker=s["marker"], ms=2.5, lw=0.9,
                      mfc=DARK if s["filled"] else "white", mec=DARK, mew=0.7, label=label)
    row1 = [Line2D([], [], color=ns.INDEX["SSI"], marker="o", ms=2.4, lw=0.9, label="Observed SSI"),
            Line2D([], [], color=ns.SLATE, ls="--", marker="s", ms=1.8, lw=0.8, label="Null SSI"),
            Patch(facecolor=grey, alpha=0.8, label="Adaptive threshold"),
            Patch(facecolor=grey, alpha=0.3, edgecolor="grey", lw=0.3, label="Floor")]
    row1 += [Line2D([], [], color=ns.ARCH[a], marker="o", ms=1.8, lw=0.5, alpha=0.7, label=a) for a in ARCHS]
    row1 += [Line2D([], [], color="black", ls="--", lw=0.7, label="p = 0.05"),
             Line2D([], [], color="black", ls=":", lw=0.6, label="p = 0.001")]
    row2 = [var("base", "4× SAE, 50 epochs"), var("8x", "8× SAE"), var("200ep", "200 epochs"),
            Patch(color=ns.PROC["Ab-E"], alpha=0.85, label="Ab-E, SSI"),
            Patch(color=ns.PROC["Di-E"], alpha=0.85, label="Di-E, CSI"),
            Patch(color=ns.PROC["Tg-E"], alpha=0.85, label="SAI"),
            Patch(color=m7.UNSPEC_COLOR, label="Below threshold"),
            Line2D([], [], color="#9CA3AF", ls="--", lw=0.6, label="SSI, CSI thresholds")]
    kw = dict(frameon=False, fontsize=6.5, handlelength=2.0, handletextpad=0.35, columnspacing=1.0,
              borderaxespad=0, borderpad=0)
    fig.legend(handles=row1, loc="upper center", bbox_to_anchor=(0.5, 1 - 1.0 / H_MM), ncol=len(row1), **kw)
    fig.legend(handles=row2, loc="upper center", bbox_to_anchor=(0.5, 1 - 5.0 / H_MM), ncol=len(row2), **kw)


def main():
    L4.apply_style()
    fig = plt.figure(figsize=(W_MM * ns.MM, H_MM * ns.MM))
    C = MMCanvas(fig, W_MM, H_MM)
    r1, r2, r3 = 20, 69, 122
    cols3 = (14, 74, 131)
    ax_a, ax_b, ax_c = (C.ax(x, r1, 43, 27) for x in cols3)
    ax_d, ax_e = (C.ax(x, r2, 43, 33) for x in cols3[:2])
    ax_f = C.ax(131, r2, 33, 33)
    ax_g, ax_h, ax_i, ax_j = (C.ax(x, r3, 31, 26) for x in (13, 58, 103, 148))

    R, N, T = validation_panels.load_core()
    panels_abc(ax_a, ax_b, ax_c, N, T)
    mag, mag_c, mag_st, mag_vals = panels_def(ax_d, ax_e, ax_f, fig)
    g, h, i, j = panels_ghij(ax_g, ax_h, ax_i, ax_j)
    legend_strip(fig)

    letters = [("a", ax_a, 1), ("b", ax_b, 61), ("c", ax_c, 118), ("d", ax_d, 1), ("e", ax_e, 61), ("f", ax_f, 118),
               ("g", ax_g, 1), ("h", ax_h, 46), ("i", ax_i, 91), ("j", ax_j, 136)]
    for L, ax, xmm in letters:
        x0, y0, w, hh = ax.get_position().bounds
        ns.panel(ax, L, x=(xmm - x0 * W_MM) / (w * W_MM), y=1 + 7.0 / (hh * H_MM))

    ns.source_data(N, FIG, "a_c")
    ns.source_data(T, FIG, "b")
    ns.source_data(mag, FIG, "d_e")
    ns.source_data(mag_c, FIG, "f")
    ns.source_data(mag_st, FIG, "d_f_stats")
    ns.source_data(g, FIG, "g")
    ns.source_data(h, FIG, "h")
    ns.source_data(i, FIG, "i")
    ns.source_data(j, FIG, "j")
    print(ns.save_final(fig, FIG, W_MM))
    print("max p", N.p_value.max(), mag_vals)


if __name__ == "__main__":
    main()
