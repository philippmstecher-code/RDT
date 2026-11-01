#!/usr/bin/env python3
"""Granger analysis of task-general, superclass-abstraction and differentiation events across superclasses,
without the duplicated final transition. Provides network_level, run_level_com and run_level_order to
extended/ed2_panels.py (Extended Data Fig. 2a-d); run directly, it draws the six-panel overview below.

Process colours Tg-E purple, Ab-E blue, Di-E amber; datasets CIFAR-100 slate-grey circles, Tiny ImageNet rose
crosses. Data: per-superclass per-transition event counts (superclass_series.py). With the final
transition kept, check_core() asserts the reference statistics (F = 625.5, beta = 1.35, dR2 = 0.098, 1,836
observations, reverse F = 52.6, Tg->Ab F = 21.0 at t <= 2, rho = 0.58, incremental F = 12.6, beta 0.82-3.24,
rho = 0.60); the figure drops each run's ``k_to_terminal`` transition (last checkpoint vs itself).

a  % of all events per transition (CIFAR-100, mean over runs and superclasses); transitions 8 and 9
   exist only in 6 and 3 of the 9 runs (open markers).
b  Granger F of the pooled panel (superclass, run and dataset fixed effects) as a function of the last
   transition included.
c  cumulative Tg-E (t <= 2) vs later Di-E (t > 2) per superclass and run.
d  Ab-E(t) vs Di-E(t + 1), all superclass-transition observations.
e  per-superclass Ab-E -> Di-E coefficients (CIFAR-100, run fixed effects), 95% CI; Bonferroni x 20.
f  Ab-E vs Di-E centre of mass (event-weighted mean transition).
Usage: ES_FIG_OUT=<dir> python scripts/paper/figures/granger_panels.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nmi_style as ns  # noqa: E402
import granger_stats as G  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
NAME = "ed5"
TG, AB, DI = "#8B5CF6", "#3B82F6", "#F59E0B"
GREY, DARK_GREY = "#D1D5DB", "#64748B"
TINY_COLOR = ns.TINY
CIFAR_COLOR = ns.REF_DARK
PHASE_ALPHA = 0.06
SHORT = {'aquatic_mammals': 'aquatic mammals', 'fish': 'fish', 'flowers': 'flowers',
         'food_containers': 'food containers', 'fruit_and_vegetables': 'fruit & veg.',
         'household_electrical_devices': 'electrical dev.', 'household_furniture': 'furniture',
         'insects': 'insects', 'large_carnivores': 'lg. carnivores',
         'large_man-made_outdoor_things': 'lg. outdoor things', 'large_natural_outdoor_scenes': 'outdoor scenes',
         'large_omnivores_and_herbivores': 'omniv. & herbiv.', 'medium_mammals': 'med. mammals',
         'non-insect_invertebrates': 'invertebrates', 'people': 'people', 'reptiles': 'reptiles',
         'small_mammals': 'sm. mammals', 'trees': 'trees', 'vehicles_1': 'vehicles 1', 'vehicles_2': 'vehicles 2'}


def leg(ax, loc, **kw):
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.7, fontsize=6, loc=loc,
              handlelength=1.6, labelspacing=0.3, **kw)


def ptxt(p):
    return r"$P$ < 0.0001" if p < 1e-4 else rf"$P$ = {p:.2g}"


GRANGER_EXCLUSIVE = Path(__file__).resolve().parents[3] / "analysis_outputs" / "paper" / "GRANGER_exclusive"


def network_level():
    """Network-level Granger tests with the training run as the unit (GRANGER_exclusive, 'Network level'): events with a
    single label (exclusive Tg/Ab/Di) summed over the network per transition and normalised per run (share of the
    run's events of that process); outcome(t+1) ~ outcome(t) + predictor(t) + run fixed effects, cluster-robust SEs
    by run; 9 CIFAR-100 runs, 72 run-transitions."""
    import statsmodels.formula.api as smf
    s = pd.read_csv(GRANGER_EXCLUSIVE / "series_exclusive.csv")
    g = s.groupby(["lane", "transition"])[["tg_ex", "ab_ex", "di_ex"]].sum().reset_index()
    g.columns = ["lane", "t", "tg", "ab", "di"]
    for c in ("tg", "ab", "di"):
        g[c] = g[c] / g.groupby("lane")[c].transform("sum")
    g = g.sort_values(["lane", "t"])
    for c in ("tg", "ab", "di"):
        g[c + "1"] = g.groupby("lane")[c].shift(-1)
    g = g.dropna()
    rows = []
    for pred, out, lab in (("tg", "ab", "Tg->Ab"), ("ab", "di", "Ab->Di"), ("di", "ab", "Di->Ab (reverse)")):
        m = smf.ols(f"{out}1 ~ {out} + {pred} + C(lane)", g).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(g.lane)[0]}, use_t=True)
        G = g.lane.nunique()
        t_stat = m.params[pred] / m.bse[pred]
        rows.append({"test": lab, "beta": m.params[pred], "se_cluster": m.bse[pred],
                     "p_cluster": float(2 * stats.t.sf(abs(t_stat), G - 1)), "reference": f"t, df = {G - 1}",
                     "p_cluster_normal": float(2 * stats.norm.sf(abs(t_stat))),
                     "n_run_transitions": int(m.nobs), "n_runs": G})
    return pd.DataFrame(rows)


def run_level_com(com):
    com = com.assign(di_minus_ab=com.di_com - com.ab_com)
    return pd.DataFrame([dict(dataset=ds, lane=lane, n_superclasses=len(g), n_di_later=int((g.di_minus_ab > 0).sum()),
                              median_di_minus_ab=float(g.di_minus_ab.median()))
                         for (ds, lane), g in com.groupby(["dataset", "lane"])])


def run_level_order(df):
    """Per run: event-weighted centre of mass (transition) of Tg-E, Ab-E, Di-E summed over superclasses."""
    rows = []
    for (ds, lane), g in df.groupby(["dataset", "lane"]):
        s = g.groupby("transition")[["tg_h", "ab_h", "di_h"]].sum()
        t = s.index.values.astype(float)
        r = dict(dataset=ds, lane=lane, **{f"com_{k}": float(np.average(t, weights=s[k])) for k in s})
        r["order_tg_ab_di"] = r["com_tg_h"] < r["com_ab_h"] < r["com_di_h"]
        rows.append(r)
    return pd.DataFrame(rows)


def check_core():
    S = G.all_stats(drop_terminal=False)
    f, r, d, c, sc = S["fwd"], S["rev"], S["dose"], S["com"], S["per_sc"]
    sw = S["sweep"].set_index(["max_transition_included", "test"])["F"]
    assert round(f["F"], 1) == 625.5 and round(f["beta"], 2) == 1.35 and round(f["delta_r2"], 3) == 0.098
    assert f["n_obs"] == 1836 and round(r["F"], 1) == 52.6 and round(r["beta"], 3) == -0.048
    assert round(sw[(2, "Tg->Ab")], 1) == 21.0 and round(d["rho"], 2) == 0.58 and d["N"] == 207
    assert round(d["F_inc"], 1) == 12.6 and round(c["rho"], 2) == 0.60
    assert round(sc.beta.min(), 2) == 0.82 and round(sc.beta.max(), 2) == 3.24
    return S


def main():
    ns.apply()
    plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                         "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial"})   # math symbols in Arial, not DejaVu
    sub = check_core()
    S = G.all_stats(drop_terminal=True)
    df, panel = S["df"], S["panel"]
    tmax = int(df["transition"].max())                      # 9 (ResNet-18 9->10)
    n_all = df[df.dataset == "cifar100"].lane.nunique()

    fig = plt.figure(figsize=(180 * ns.MM, 150 * ns.MM))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1, 1], wspace=0.45, hspace=0.36, left=0.075, right=0.975, bottom=0.075, top=0.955)
    # phase boundaries and names as in fig1.py draw_b: I to transition 2, II 2-5, III from 5
    phases = [(-0.5, 2, TG, "Phase I\nTask-general"), (2, 5, AB, "Phase II\nAbstraction"),
              (5, tmax + 0.5, DI, "Phase III\nDifferentiation")]

    # a
    ax = fig.add_subplot(gs[0, 0])
    A = S["a"]
    for lab, col in [("Tg-E", TG), ("Ab-E", AB), ("Di-E", DI)]:
        d = A[A.process == lab].sort_values("transition")
        full = d[d.n_runs == n_all]
        ax.plot(d.transition, d.pct_of_total_events, "-", color=col, lw=1.0, zorder=2)
        ax.plot(full.transition, full.pct_of_total_events, "o", color=col, ms=2.5, zorder=3)
        ax.plot([], [], "o-", color=col, ms=2.5, lw=1.0, label=lab)
        part = d[d.n_runs < n_all]
        ax.plot(part.transition, part.pct_of_total_events, "o", mfc="white", mec=col, mew=0.8, ms=2.8, zorder=3)
    for x0, x1, c, lab in phases:
        ax.axvspan(x0, x1, alpha=PHASE_ALPHA, color=c, zorder=0, lw=0)
        ax.text((x0 + x1) / 2, 35.5, lab, ha="center", va="center", fontsize=6, color="#111111", linespacing=1.05)
    for xb, c in ((2, TG), (5, AB)):
        ax.axvline(xb, color=c, alpha=0.12, lw=0.8, zorder=0)
    ax.set_ylim(0, 39)
    ax.set_yticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_xlim(-0.5, tmax + 0.5)
    ax.set_xticks(range(0, tmax + 1, 2))
    ax.set_xlabel("Transition")
    ax.set_ylabel("% of total events")
    leg(ax, "upper left", bbox_to_anchor=(0.0, 0.85))
    ns.panel(ax, "a", x=-0.24, y=1.08)

    # b
    ax = fig.add_subplot(gs[0, 1])
    sw = S["sweep"]
    for test, col, mk, ls, lw, al, lab in [("Tg->Ab", TG, "o", "-", 1.0, 1, r"Tg$\to$Ab"),
                                           ("Ab->Di", AB, "^", "-", 1.0, 1, r"Ab$\to$Di"),
                                           ("Tg->Di", DARK_GREY, "s", "--", 0.8, 0.6, r"Tg$\to$Di")]:
        d = sw[sw.test == test]
        ax.plot(d.max_transition_included, d.F, marker=mk, ls=ls, color=col, ms=2.8, lw=lw, alpha=al, label=lab)
    ax.axhline(3.84, color="grey", ls=":", lw=0.5, alpha=0.6)
    ax.text(tmax - 0.1, 2.6, r"$P$ = 0.05", fontsize=6, color="grey", style="italic", ha="right", va="top")
    for x0, x1, c, _ in phases:            # same boundaries as a, on the last transition included
        ax.axvspan(max(x0, 0.6), x1, alpha=PHASE_ALPHA, color=c, zorder=0, lw=0)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    ax.yaxis.set_major_formatter(FixedFormatter(["1", "10", "100", "1,000"]))   # plain labels, no exponents
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0.2, 1200)
    ax.set_xlim(0.6, tmax + 0.4)
    ax.set_xticks(range(1, tmax + 1, 2))
    ax.set_xlabel("Last transition included")
    ax.set_ylabel(r"Granger $F$")
    leg(ax, "upper left")
    ns.panel(ax, "b", x=-0.24, y=1.08)

    # c
    ax = fig.add_subplot(gs[0, 2])
    dd, dst = S["dose_df"], S["dose"]
    is_c = dd.dataset == "cifar100"
    x, y = np.log1p(dd.tg_dose), np.log1p(dd.post_di)
    ax.scatter(x[is_c], y[is_c], c=CIFAR_COLOR, s=8, alpha=0.35, marker="o", linewidths=0, label="CIFAR-100")
    ax.scatter(x[~is_c], y[~is_c], c=TINY_COLOR, s=16, alpha=0.7, marker="x", linewidths=0.6, label="Tiny ImageNet")
    z = np.polyfit(x.values, y.values, 1)
    xl = np.linspace(x.min(), x.max(), 100)
    ax.plot(xl, np.poly1d(z)(xl), "-", color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"log(Tg-E dose, $t \leq 2$)")
    ax.set_ylabel(r"log(later Di-E, $t > 2$)")
    ax.text(0.05, 0.95, rf"$\rho$ = {dst['rho']:.2f}, {ptxt(dst['p'])}", transform=ax.transAxes, fontsize=6, va="top")
    leg(ax, "lower right")
    ns.panel(ax, "c", x=-0.24, y=1.08)

    # d
    ax = fig.add_subplot(gs[1, 0])
    fwd = S["fwd"]
    is_c = panel.dataset == "cifar100"
    x, y = np.log1p(panel.ab_h_t), np.log1p(panel.di_h_t1)
    ax.scatter(x[is_c], y[is_c], c=CIFAR_COLOR, s=8, alpha=0.12, marker="o", linewidths=0, label="CIFAR-100")
    ax.scatter(x[~is_c], y[~is_c], c=TINY_COLOR, s=16, alpha=0.6, marker="x", linewidths=0.6, label="Tiny ImageNet")
    z = np.polyfit(x.values, y.values, 1)
    xl = np.linspace(x.min(), x.max(), 100)
    ax.plot(xl, np.poly1d(z)(xl), "-", color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("log(Ab-E at t)")
    ax.set_ylabel("log(Di-E at t + 1)")
    ax.text(0.04, 0.97, rf"$F$ = {fwd['F']:.1f}, $\beta$ = {fwd['beta']:.2f}" + "\n"
            + rf"ΔR² = {fwd['delta_r2']:.3f}, {ptxt(fwd['p'])}", transform=ax.transAxes, fontsize=6,
            va="top", bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5))
    leg(ax, "lower right")
    for lh in ax.get_legend().legend_handles:      # keys readable despite the low point alpha
        lh.set_alpha(0.6)
    ns.panel(ax, "d", x=-0.24, y=1.08)

    # e
    ax = fig.add_subplot(gs[1, 1])
    sc = S["per_sc"].sort_values("beta").reset_index(drop=True)
    for i, r in sc.iterrows():
        col = AB if r.significant_bonferroni else GREY
        ax.plot([r.ci_low, r.ci_high], [i, i], "-", color=col, lw=0.8, solid_capstyle="round")
        ax.plot(r.beta, i, "o", color=col, ms=3.2, mec="white", mew=0.3, zorder=3)
    ax.set_yticks(range(len(sc)))
    ax.set_yticklabels([SHORT.get(s, s) for s in sc.superclass], fontsize=6)
    ax.tick_params(axis="y", length=1.5, pad=1.5)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel(r"Ab-E(t) coefficient $\beta$ (95% CI)")
    ax.set_ylim(len(sc) - 0.4, -0.6)
    ns.panel(ax, "e", x=-0.62, y=1.08)

    # f
    ax = fig.add_subplot(gs[1, 2])
    com, cst = S["com_df"], S["com"]
    is_c = com.dataset == "cifar100"
    ax.scatter(com.ab_com[is_c], com.di_com[is_c], c=CIFAR_COLOR, s=10, alpha=0.4, marker="o", linewidths=0, label="CIFAR-100")
    ax.scatter(com.ab_com[~is_c], com.di_com[~is_c], c=TINY_COLOR, s=18, alpha=0.7, marker="x", linewidths=0.6,
               label="Tiny ImageNet")
    lims = [min(com.ab_com.min(), com.di_com.min()) - 0.3, max(com.ab_com.max(), com.di_com.max()) + 0.3]
    ax.plot(lims, lims, "--", color="grey", lw=0.5, alpha=0.4)
    ax.set_xlabel("Ab-E centre of mass (transition)")
    ax.set_ylabel("Di-E centre of mass (transition)")
    ax.text(0.05, 0.95, rf"$\rho$ = {cst['rho']:.2f}, {ptxt(cst['p'])}", transform=ax.transAxes, fontsize=6, va="top")
    leg(ax, "lower right")
    ns.panel(ax, "f", x=-0.24, y=1.08)

    # source data and statistics (final transition kept vs removed)
    ns.source_data(A, NAME, "a")
    ns.source_data(sw, NAME, "b")
    ns.source_data(dd, NAME, "c")
    ns.source_data(panel, NAME, "d")
    ns.source_data(S["per_sc"], NAME, "e")
    ns.source_data(com, NAME, "f")
    net = network_level()
    assert len(net) == 3 and (net.n_run_transitions == 72).all() and (net.n_runs == 9).all()
    ns.source_data(net, NAME, "network_level")
    runs_f = run_level_com(com).merge(run_level_order(df), on=["dataset", "lane"])
    ns.source_data(runs_f, NAME, "f_runs")
    print(net.to_string())
    print(runs_f.to_string())
    rows = []
    for tag, R in (("final transition kept", sub), ("final transition removed", S)):
        swr = R["sweep"].set_index(["max_transition_included", "test"])
        mt = int(R["sweep"].max_transition_included.max())
        rows += [
            {"version": tag, "statistic": "d: Ab-E(t) -> Di-E(t+1) Granger F", "value": R["fwd"]["F"]},
            {"version": tag, "statistic": "d: beta", "value": R["fwd"]["beta"]},
            {"version": tag, "statistic": "d: delta R2", "value": R["fwd"]["delta_r2"]},
            {"version": tag, "statistic": "d: P", "value": R["fwd"]["p"]},
            {"version": tag, "statistic": "d: observations", "value": R["fwd"]["n_obs"]},
            {"version": tag, "statistic": "d: reverse Di-E(t) -> Ab-E(t+1) F", "value": R["rev"]["F"]},
            {"version": tag, "statistic": "d: reverse beta", "value": R["rev"]["beta"]},
            {"version": tag, "statistic": "d: forward/reverse F ratio", "value": R["fwd"]["F"] / R["rev"]["F"]},
            {"version": tag, "statistic": "b: Tg->Ab F at t <= 2", "value": swr.loc[(2, "Tg->Ab"), "F"]},
            {"version": tag, "statistic": "b: Tg->Ab P at t <= 2", "value": swr.loc[(2, "Tg->Ab"), "p"]},
            {"version": tag, "statistic": "b: Tg->Di F at t <= 2", "value": swr.loc[(2, "Tg->Di"), "F"]},
            {"version": tag, "statistic": "b: last transition included", "value": mt},
            {"version": tag, "statistic": "b: Ab->Di F at full span", "value": swr.loc[(mt, "Ab->Di"), "F"]},
            {"version": tag, "statistic": "b: Tg->Ab F at full span", "value": swr.loc[(mt, "Tg->Ab"), "F"]},
            {"version": tag, "statistic": "b: Tg->Ab P at full span", "value": swr.loc[(mt, "Tg->Ab"), "p"]},
            {"version": tag, "statistic": "b: Tg->Di F at full span", "value": swr.loc[(mt, "Tg->Di"), "F"]},
            {"version": tag, "statistic": "b: Tg->Di P at full span", "value": swr.loc[(mt, "Tg->Di"), "p"]},
            {"version": tag, "statistic": "c: Spearman rho (Tg dose vs later Di-E)", "value": R["dose"]["rho"]},
            {"version": tag, "statistic": "c: N superclass x runs", "value": R["dose"]["N"]},
            {"version": tag, "statistic": "c: incremental F (Tg dose beyond Ab dose)", "value": R["dose"]["F_inc"]},
            {"version": tag, "statistic": "c: incremental P", "value": R["dose"]["p_inc"]},
            {"version": tag, "statistic": "e: beta min", "value": R["per_sc"].beta.min()},
            {"version": tag, "statistic": "e: beta max", "value": R["per_sc"].beta.max()},
            {"version": tag, "statistic": "e: positive and Bonferroni-significant (of 20)",
             "value": int(((R["per_sc"].beta > 0) & R["per_sc"].significant_bonferroni).sum())},
            {"version": tag, "statistic": "f: Spearman rho (centres of mass)", "value": R["com"]["rho"]},
            {"version": tag, "statistic": "f: points above identity", "value": R["com"]["n_above_identity"]},
        ]
    st = pd.DataFrame(rows)
    ns.source_data(st, NAME, "statistics")
    out = ns.save_final(fig, NAME, 180, ed=True)
    print("saved", out)
    print(st.pivot(index="statistic", columns="version", values="value").to_string())


if __name__ == "__main__":
    import os as _os
    if not _os.environ.get("ES_FIG_OUT"):
        raise SystemExit("granger_panels.py draws the six-panel Granger overview; Extended Data Fig. 2a-d is drawn by "
                         "extended/ed2_panels.py. Set ES_FIG_OUT to an output folder to draw the overview.")
    main()
