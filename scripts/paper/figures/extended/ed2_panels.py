#!/usr/bin/env python3
"""Extended Data Fig. 2, panels a-d (saved by interventions/ed2.py): Granger analysis with the training run as the unit.

a  forest plot of the run-level Granger coefficients (Tg-E -> Ab-E, Ab-E -> Di-E and the reverse Di-E -> Ab-E):
   beta with t-based 95% CI (cluster-robust SE x t_0.975,8) and run-level P (granger_panels.network_level(); 9 CIFAR-100
   runs, 72 run-transitions).
b  pooled Granger F by last transition included (log y); descriptive, no P value.
c  Ab-E(t) vs Di-E(t + 1) per superclass-transition, fit line; pooled beta.
d  Ab-E vs Di-E centre of mass per superclass x run, identity line.
Event shares, Tg-E dose and per-superclass coefficients are written to Source Data only. Statistics are recomputed
with the figures/granger_panels.py code path and checked against the reference Source Data in figures/final/source_data_ed.
Usage: python scripts/paper/figures/extended/ed2_panels.py  (ES_FIG_OUT=<dir> redirects the output)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import nmi_style as ns  # noqa: E402
import granger_stats as G  # noqa: E402
from granger_panels import network_level, run_level_com, run_level_order  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
NAME = "ed2"
FINAL_SD = ns.OUT / "final" / "source_data_ed"          # read-only reference Source Data
TG, AB, DI = ns.PROC["Tg-E"], ns.PROC["Ab-E"], ns.PROC["Di-E"]
TG_DI = "#64748B"                                     # Tg-E -> Di-E line
CIFAR, TINY = ns.REF_DARK, ns.TINY
TITLE_KW = dict(loc="left", fontsize=7, pad=4)
ARROW = "→"


def p_txt(p):
    """Run-level P: scientific below 0.001, two significant figures below 0.05, two decimals above."""
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return rf"$P$ = {m} $\times$ 10$^{{{int(e)}}}$"
    if p < 0.05:
        return rf"$P$ = {p:.2g}"
    return rf"$P$ = {p:.2f}"


def check_against_final(S, net, runs_f):
    """Assert that the recomputed statistics equal the reference Source Data."""
    ref = pd.read_csv(FINAL_SD / "ed2_network_level.csv")
    assert np.allclose(ref[["beta", "se_cluster", "p_cluster"]].values,
                       net[["beta", "se_cluster", "p_cluster"]].values, rtol=1e-9), "network-level drift"
    sw_ref = pd.read_csv(FINAL_SD / "ed2_b.csv")
    assert np.allclose(sw_ref.F.values, S["sweep"].F.values, rtol=1e-9), "sweep drift"
    com_ref = pd.read_csv(FINAL_SD / "ed2_d.csv")
    assert np.allclose(com_ref[["ab_com", "di_com"]].values, S["com_df"][["ab_com", "di_com"]].values), "com drift"
    st = pd.read_csv(FINAL_SD / "ed2_statistics.csv")
    st = st[st.version.eq("final transition removed")].set_index("statistic")["value"]
    assert np.isclose(st["d: beta"], S["fwd"]["beta"])
    assert np.isclose(st["d: Ab-E(t) -> Di-E(t+1) Granger F"], S["fwd"]["F"])
    assert int(st["f: points above identity"]) == S["com"]["n_above_identity"]
    fr = pd.read_csv(FINAL_SD / "ed2_d_runs.csv")
    assert fr.order_tg_ab_di.sum() == runs_f.order_tg_ab_di.sum()


def main():
    ns.apply()
    plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                         "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial"})
    S = G.all_stats(drop_terminal=True)
    df, pw, com = S["df"], S["panel"], S["com_df"]
    net = network_level()
    assert len(net) == 3 and (net.n_run_transitions == 72).all() and (net.n_runs == 9).all()
    runs_f = run_level_com(com).merge(run_level_order(df), on=["dataset", "lane"])
    check_against_final(S, net, runs_f)

    tcrit = stats.t.ppf(0.975, net.n_runs.iloc[0] - 1)
    net = net.assign(t_crit=tcrit, ci_low=net.beta - tcrit * net.se_cluster, ci_high=net.beta + tcrit * net.se_cluster)

    fig = plt.figure(figsize=(180 * ns.MM, 118 * ns.MM))
    gs = fig.add_gridspec(2, 2, left=0.135, right=0.975, bottom=0.085, top=0.88, wspace=0.30, hspace=0.52)

    # a: run-level forest plot
    ax = fig.add_subplot(gs[0, 0])
    rows = [("Tg->Ab", f"Tg-E {ARROW} Ab-E", TG, "o", TG),
            ("Ab->Di", f"Ab-E {ARROW} Di-E", AB, "^", AB),
            ("Di->Ab (reverse)", f"Di-E {ARROW} Ab-E\n(reverse)", DI, "o", "white")]
    for i, (key, lab, col, mk, mfc) in enumerate(rows):
        r = net.set_index("test").loc[key]
        ax.plot([r.ci_low, r.ci_high], [i, i], "-", color=col, lw=1.2, solid_capstyle="butt")
        ax.plot(r.beta, i, mk, color=col, mfc=mfc, mec=col, mew=1.0, ms=4.5, zorder=3)
        ax.text(1.0, i, p_txt(r.p_cluster), transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=6.5)
    ax.axvline(0, color="black", lw=0.5, zorder=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[1] for r in rows])
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.set_xlim(-0.4, 1.75)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel(r"Run-level coefficient $\beta$ (95% CI)")
    ax.set_title("Each stage predicts the next across runs", **TITLE_KW)
    fig.text(0.008, ax.get_position().y1 + 0.016, "a", fontsize=8, fontweight="bold", va="bottom")

    # b: F by last transition (descriptive)
    ax = fig.add_subplot(gs[0, 1])
    sw = S["sweep"]
    tmax = int(df.transition.max())
    for test, col, mk, ls, lw in [("Tg->Ab", TG, "o", "-", 1.0), ("Ab->Di", AB, "^", "-", 1.0),
                                  ("Tg->Di", TG_DI, "s", "--", 0.8)]:
        d = sw[sw.test == test]
        ax.plot(d.max_transition_included, d.F, marker=mk, ls=ls, color=col, ms=2.8, lw=lw,
                alpha=0.8 if test == "Tg->Di" else 1)
    f2 = sw[(sw.test == "Tg->Ab") & (sw.max_transition_included == 2)].F.item()
    ax.plot([2, 2], [0.36, f2 / 1.3], "-", color="#6B7280", lw=0.5, zorder=1)
    ax.text(0.75, 0.2, rf"Tg-E {ARROW} Ab-E at $t$ $\leq$ 2: $F$ = {f2:.1f}", fontsize=6.5, ha="left", va="center")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    ax.yaxis.set_major_formatter(FixedFormatter(["1", "10", "100", "1,000"]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0.12, 1200)
    ax.set_xlim(0.6, tmax + 0.4)
    ax.set_xticks(range(1, tmax + 1, 2))
    ax.set_xlabel("Last transition included")
    ax.set_ylabel(r"Granger $F$ (pooled)")
    ax.set_title("Task-general events predict early; abstraction predicts later", **TITLE_KW)
    fig.text(ax.get_position().x0 - 0.075, ax.get_position().y1 + 0.016, "b", fontsize=8, fontweight="bold", va="bottom")

    # c: Ab-E(t) vs Di-E(t + 1)
    ax = fig.add_subplot(gs[1, 0])
    is_c = pw.dataset == "cifar100"
    x, y = np.log1p(pw.ab_h_t), np.log1p(pw.di_h_t1)
    ax.scatter(x[is_c], y[is_c], c=CIFAR, s=7, alpha=0.12, marker="o", linewidths=0, rasterized=True)
    ax.scatter(x[~is_c], y[~is_c], c=TINY, s=12, alpha=0.6, marker="x", linewidths=0.6, rasterized=True)
    z = np.polyfit(x.values, y.values, 1)
    xl = np.linspace(x.min(), x.max(), 100)
    ax.plot(xl, np.poly1d(z)(xl), "-", color="black", lw=0.8, alpha=0.6)
    ax.text(0.04, 0.96, rf"$\beta$ = {S['fwd']['beta']:.2f}", transform=ax.transAxes, fontsize=6.5, va="top")
    ax.set_xlabel(r"log(Ab-E at $t$)")
    ax.set_ylabel(r"log(Di-E at $t$ + 1)")
    ax.set_title("Abstraction goes with differentiation at the next transition", **TITLE_KW)
    fig.text(0.008, ax.get_position().y1 + 0.016, "c", fontsize=8, fontweight="bold", va="bottom")

    # d: centres of mass
    ax = fig.add_subplot(gs[1, 1])
    is_c = com.dataset == "cifar100"
    ax.scatter(com.ab_com[is_c], com.di_com[is_c], c=CIFAR, s=9, alpha=0.4, marker="o", linewidths=0)
    ax.scatter(com.ab_com[~is_c], com.di_com[~is_c], c=TINY, s=14, alpha=0.7, marker="x", linewidths=0.6)
    lo = min(com.ab_com.min(), com.di_com.min()) - 0.3
    hi = max(com.ab_com.max(), com.di_com.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], "--", color="grey", lw=0.6, alpha=0.6)
    n_above, n_tot = S["com"]["n_above_identity"], len(com)
    ax.text(0.96, 0.06, f"Di-E later in {n_above} of {n_tot}\nsuperclass × runs", transform=ax.transAxes,
            fontsize=6.5, ha="right", va="bottom")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("auto")
    ax.set_xlabel("Ab-E centre of mass (transition)")
    ax.set_ylabel("Di-E centre of mass (transition)")
    ax.set_title("Differentiation follows abstraction in every run", **TITLE_KW)
    fig.text(ax.get_position().x0 - 0.075, ax.get_position().y1 + 0.016, "d", fontsize=8, fontweight="bold", va="bottom")

    # shared legend strip
    H = [Line2D([], [], color=TG, marker="o", ms=3.5, lw=1.0, label=f"Tg-E {ARROW} Ab-E"),
         Line2D([], [], color=AB, marker="^", ms=3.5, lw=1.0, label=f"Ab-E {ARROW} Di-E"),
         Line2D([], [], color=DI, marker="o", mfc="white", ms=3.5, lw=1.0, label=f"Di-E {ARROW} Ab-E (reverse)"),
         Line2D([], [], color=TG_DI, marker="s", ms=3, lw=0.8, ls="--", label=f"Tg-E {ARROW} Di-E"),
         Line2D([], [], color=CIFAR, marker="o", ms=3.5, lw=0, alpha=0.6, label="CIFAR-100"),
         Line2D([], [], color=TINY, marker="x", ms=4, mew=0.8, lw=0, label="Tiny ImageNet"),
         Line2D([], [], color="grey", ls="--", lw=0.6, label="identity")]
    fig.legend(handles=H, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=7, fontsize=6.5, frameon=False,
               handlelength=2.0, columnspacing=1.3, handletextpad=0.5)

    # Source Data
    ns.source_data(net[["test", "beta", "se_cluster", "t_crit", "ci_low", "ci_high", "p_cluster", "reference",
                        "n_run_transitions", "n_runs"]], NAME, "a")
    ns.source_data(sw, NAME, "b")
    ns.source_data(pw, NAME, "c")
    ns.source_data(com, NAME, "d")
    ns.source_data(runs_f, NAME, "d_runs")
    ns.source_data(S["a"], NAME, "event_shares")                 # same data as Fig. 1b
    ns.source_data(S["dose_df"], NAME, "tg_dose")
    ns.source_data(S["per_sc"], NAME, "per_superclass")
    out = ns.save_final(fig, NAME, 180, ed=True)
    print("saved", out)
    print(net[["test", "beta", "ci_low", "ci_high", "p_cluster"]].to_string())
    print("Tg->Ab F(t<=2) =", f2, "| beta =", S["fwd"]["beta"], "| above identity", n_above, "/", n_tot,
          "| runs in order", int(runs_f.order_tg_ab_di.sum()), "/", len(runs_f),
          "| per-sc positive", int((S["per_sc"].beta > 0).sum()), "/", len(S["per_sc"]),
          "| reverse F", S["rev"]["F"], "fwd F", S["fwd"]["F"])


if __name__ == "__main__":
    main()
