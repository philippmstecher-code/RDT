#!/usr/bin/env python3
"""Loaders and panels for the methodology validation and the robustness of the developmental ordering, reused by
extended/ed1.py (Extended Data Fig. 1), supp5.py (Supplementary Fig. 5) and extended/supp4.py (Supplementary Fig. 4).

a, b, d, e, f  false-positive rates, within-checkpoint stability, observed vs null SSI, adaptive thresholds and null
               p-values of the 9 standard runs (<lane>/sae_analysis/sae_results.json).
c              construction/refinement ratio (Tg-E + Ab-E)/Di-E, ResNet-18 seed 256, shared vs independent SAE
               initialisation, stored classification per SAE; the recount (checks_lib.a2_lane_pi) and a check of the
               core shared line are in the Source Data.
g, h, i        census per checkpoint, exclusive categories (CENSUS_exclusive/census_exclusive_per_checkpoint.csv).
j              superclass share of between-class variance in the pooled-layer activations, 16 networks
               (DICTFREE/population: 9; DATA_order/transient_stage: 7 held out), against 100 random 20 x 5 groupings.
k              share at initialisation, peak and end per network; dashed, SimCLR without labels
               (SIMCLR/simclr_measures.csv).
l              robustness ledger (MATCHING/process_ordering_by_rule.csv; TRANSFORMED/transformed_results.json).
main() draws all panels as one stand-alone figure, saved under the name ed1, and runs only when ES_FIG_OUT is set.

Usage: ES_FIG_OUT=<dir> python scripts/paper/figures/validation_panels.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nmi_style as ns  # noqa: E402
from checks_lib import (ARCHS, CORE, CORE_EXP, INDEP_EXP, INDEP_LANE, RESULTS, MMCanvas, a2_lane_pi, cr,  # noqa: E402
                     sae_results)

FIG = "ed1"
W_MM, H_MM = 180, 169   # height <= 170 mm
ARCH, ARCH_C = ARCHS, ns.ARCH
BLUE, AMBER, PURPLE, RED500 = ns.INDEX["SSI"], ns.INDEX["CSI"], ns.INDEX["SAI"], ns.PROC["De-E"]
RED = ns.RED_600            # independent initialisation
SLATE = ns.SLATE            # null / random-grouping reference
DARK = ns.REF_DARK
GRID = "#9CA3AF"
SMALL = 6.0
STORE = {}


def ax_style(ax):
    ax.tick_params(labelsize=6.2, pad=1.5)
    ax.xaxis.label.set_size(6.5)
    ax.yaxis.label.set_size(6.5)


def plain_log_ticks(ax, ticks):
    """Plain decimal tick labels on a log axis (no mathtext exponents, which print below 6 pt)."""
    from matplotlib.ticker import FixedLocator, NullFormatter
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.4f}".rstrip("0").rstrip(".")))
    ax.yaxis.set_minor_formatter(NullFormatter())


def title(ax, t):
    ax.set_title(t, fontsize=6.8, pad=3)


# a, b, d, e, f
def load_core():
    rows, nulls, thr = [], [], []
    for arch, seed, pre in CORE:
        d = sae_results(CORE_EXP, pre)
        w = d["within_checkpoint_control"]["per_checkpoint"]
        lay = [l for e in w for l in e["per_layer"]]
        rows.append(dict(arch=arch, seed=seed, false_birth=100 * np.mean([l["false_birth_rate"] for l in lay]),
                         false_death=100 * np.mean([l["false_death_rate"] for l in lay]),
                         stable=100 * np.mean([l["stable_rate"] for l in lay]),
                         checkpoints="/".join(e["checkpoint"] for e in w)))
        nb = {e["checkpoint"]: e for e in d["null_baseline"]["per_checkpoint"]}
        for ck in map(str, range(9)):
            nulls.append(dict(arch=arch, seed=seed, checkpoint=int(ck), observed_ssi=nb[ck]["observed_ssi"],
                              null_ssi=nb[ck]["null_ssi"], p_value=nb[ck]["p_value"],
                              n_permutations=d["null_baseline"]["n_permutations"]))
        at = d["metadata"]["adaptive_thresholds"]
        thr.append(dict(arch=arch, seed=seed, SSI=at["ssi_adaptive_thresh"], CSI=at["csi_adaptive_thresh"],
                        SAI=at["sai_adaptive_thresh"], SSI_floor=at["ssi_floor"], CSI_floor=at["csi_floor"],
                        SAI_floor=at["sai_floor"]))
    return pd.DataFrame(rows), pd.DataFrame(nulls), pd.DataFrame(thr)


def panel_a(ax, R):
    x, w = np.arange(3), 0.35
    g = R.groupby("arch")
    for off, col, c, lab in ((-w / 2, "false_birth", BLUE, "False birth"), (w / 2, "false_death", RED500, "False death")):
        m = [g[col].mean()[a] for a in ARCH]
        s = [g[col].std(ddof=0)[a] for a in ARCH]
        ax.bar(x + off, m, w, yerr=s, color=c, alpha=0.8, edgecolor="white", lw=0.3, label=lab,
               error_kw=dict(ecolor="black", elinewidth=0.7, capsize=1.6, capthick=0.7))
    for yv in (2, 10):
        ax.axhline(yv, color="grey", ls="--", lw=0.5, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ARCH)
    ax.set_ylabel("False-positive rate (%)")
    ax.set_xlabel("Architecture")
    ax.set_ylim(0, 22)
    ax.legend(fontsize=SMALL, loc="upper left", handlelength=1.2, borderaxespad=0.2)
    title(ax, "False-positive rates (N = 9)")
    out = R.groupby("arch")[["false_birth", "false_death", "stable"]].agg(["mean", lambda v: np.std(v)])
    STORE["a_false_death_range_by_arch"] = {a: [round(R[R.arch == a].false_death.min(), 2),
                                                round(R[R.arch == a].false_death.max(), 2)] for a in ARCH}
    return out


def panel_b(ax, R):
    y = np.arange(3)
    g = R.groupby("arch")["stable"]
    m = [g.mean()[a] for a in ARCH]
    s = [g.std(ddof=0)[a] for a in ARCH]
    ax.barh(y, m, xerr=s, color=[ARCH_C[a] for a in ARCH], alpha=0.8, height=0.5, edgecolor="white", lw=0.3,
            error_kw=dict(ecolor="black", elinewidth=0.7, capsize=1.6, capthick=0.7))
    for i, a in enumerate(ARCH):
        ax.plot(R[R.arch == a].stable, [i] * 3, "o", color="black", ms=2.2, alpha=0.5, mew=0, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(ARCH)
    ax.set_xlabel("Mean stable rate (%)")
    ax.set_xlim(0, 65)
    title(ax, "Within-checkpoint stability (N = 9)")


def panel_d(ax, N):
    g = N.groupby("checkpoint")
    x = np.arange(9)
    om, osd = g.observed_ssi.mean().values, g.observed_ssi.std(ddof=0).values
    nm, nsd = g.null_ssi.mean().values, g.null_ssi.std(ddof=0).values
    ax.fill_between(x, nm - nsd, nm + nsd, color="#94A3B8", alpha=0.2, lw=0)
    ax.fill_between(x, om - osd, om + osd, color=BLUE, alpha=0.15, lw=0)
    ax.plot(x, om, "o-", color=BLUE, ms=2.4, lw=0.9, label="Observed SSI")
    ax.plot(x, nm, "s--", color=SLATE, ms=1.8, lw=0.8, label="Null SSI")
    ax.set_xticks(x)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Mean SSI")
    ax.legend(fontsize=SMALL, loc="lower center", handlelength=2.0, borderaxespad=0.2)
    title(ax, "Observed vs null SSI (N = 9)")


def panel_e(ax, T):
    names = ["SSI", "CSI", "SAI"]
    x, w = np.arange(3), 0.35
    m = [T[n].mean() for n in names]
    s = [T[n].std(ddof=0) for n in names]
    floors = [T[f"{n}_floor"].iloc[0] for n in names]
    cols = [ns.INDEX[n] for n in names]
    ax.bar(x - w / 2, m, w, yerr=s, color=cols, alpha=0.8, edgecolor="white", lw=0.3,
           error_kw=dict(ecolor="black", elinewidth=0.7, capsize=1.6, capthick=0.7))
    ax.bar(x + w / 2, floors, w, color=cols, alpha=0.3, edgecolor="grey", lw=0.3)
    for i, n in enumerate(names):
        ax.plot([i - w / 2] * len(T), T[n], "o", color="black", ms=1.6, alpha=0.4, mew=0, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Threshold value")
    ax.set_ylim(0, 1.08)
    h = [Patch(facecolor=GRID, alpha=0.8, label="Adaptive (p95 null)"),
         Patch(facecolor=GRID, alpha=0.3, edgecolor="grey", lw=0.3, label="Floor")]
    ax.legend(handles=h, fontsize=SMALL, loc="upper left", handlelength=1.2, borderaxespad=0.2)
    title(ax, "Adaptive vs floor thresholds (N = 9)")


def panel_f(ax, N):
    for (arch, seed), g in N.groupby(["arch", "seed"]):
        g = g.sort_values("checkpoint")
        ax.semilogy(g.checkpoint, np.maximum(g.p_value, 1e-4), "o-", color=ARCH_C[arch], ms=1.8, lw=0.5, alpha=0.5)
    ax.axhline(0.05, color="black", ls="--", lw=0.7, alpha=0.6)
    ax.axhline(0.001, color="black", ls=":", lw=0.5, alpha=0.4)
    ax.set_xticks(range(9))
    ax.set_ylim(5e-5, 30)   # head room so the legend sits above the p = 0.05 line
    plain_log_ticks(ax, [1e-4, 1e-3, 1e-2, 1e-1, 1])
    from matplotlib.ticker import FixedLocator
    ax.yaxis.set_minor_locator(FixedLocator([m * 10.0 ** e for e in range(-5, 0) for m in range(2, 10)
                                             if 5e-5 <= m * 10.0 ** e <= 1]))
    ax.spines["left"].set_bounds(5e-5, 1.5)
    ax.set_ylim(5e-5, 30)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("p-value (permutation)")
    h = [Patch(facecolor=ARCH_C[a], alpha=0.7, label=a) for a in ARCH]
    h.append(Line2D([], [], color="black", ls="--", lw=0.7, label="p = 0.05"))
    ax.legend(handles=h, fontsize=SMALL, loc="upper right", handlelength=1.2, borderaxespad=0.1, labelspacing=0.25)
    title(ax, "Null baseline p-values (N = 9)")
    STORE["f_max_p"] = float(N.p_value.max())


# c: shared vs independent SAE initialisation
SHARED_256 = "935c13ae-a026-4eae-a2b2-0306879b2e8c"
# Stored reference values for panel c (Ab-E/Di-E per transition, shared and independent initialisation), as plot
# coordinates: log10 ratio = (y - SUB_Y1) / SUB_DEC; 11 points incl. k->terminal. core_c_check reproduces them.
SUB_Y1, SUB_DEC = 6512.77, 970.6
SUB_SHARED_Y = [6517.57, 6471.48, 6297.17, 6089.72, 5835.09, 6068.78, 6041.55, 6127.72, 5598.31, 5490.63, 6004.61]
SUB_INDEP_Y = [6272.19, 6446.36, 6026.21, 6484.61, 5909.12, 6058.96, 6054.5, 6124.63, 5833.77, 5622.13, 5751.51]


def core_c_check():
    sub_sh = 10 ** ((np.array(SUB_SHARED_Y) - SUB_Y1) / SUB_DEC)
    sub_in = 10 ** ((np.array(SUB_INDEP_Y) - SUB_Y1) / SUB_DEC)
    rows = []
    cand = {"ResNet-18 s42 shared": ("abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb", "7db28a9a"),
            "ResNet-18 s256 shared": ("abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb", "935c13ae"),
            "ResNet-18 s256 independent (stored, 1000 perm.)": (INDEP_EXP, "independent_sae_seed256")}
    ser = {}
    for k, (e, p) in cand.items():
        pi = sae_results(e, p)["process_intensity"]
        ser[k] = [min(t["ab_h"] / max(t["di_h"], 1), 300) for t in pi]
    for i in range(11):
        r = dict(point=i, reference_shared_abE_over_diE=round(sub_sh[i], 3),
                 reference_independent_abE_over_diE=round(sub_in[i], 3))
        for k, v in ser.items():
            r[k] = round(v[i], 3)
        rows.append(r)
    df = pd.DataFrame(rows)
    err42 = float(np.max(np.abs(np.log10(df["ResNet-18 s42 shared"]) - np.log10(df.reference_shared_abE_over_diE))))
    STORE["c_core_shared_line_is_seed42_maxabs_log10_err"] = round(err42, 4)
    STORE["c_core_independent_first_point"] = round(float(sub_in[0]), 3)
    STORE["c_reference_first_point_ratio"] = round(1501 / 2656, 3)
    assert err42 < 0.01, err42
    return df


def panel_c(ax):
    sh, ind = a2_lane_pi(SHARED_256), a2_lane_pi(INDEP_LANE)
    sh_st = sae_results(CORE_EXP, "935c13ae")["process_intensity"][:-1]
    in_st = sae_results(INDEP_EXP, "independent_sae_seed256")["process_intensity"][:-1]
    assert len(sh) == len(ind) == 10
    x = np.arange(10)
    # Both lines use each SAE's stored classification (sae_results.json process_intensity). The recount of
    # checks_lib.a2_lane_pi (Source Data) is identical except at the last transition of the independent SAE.
    ysh, yin = np.array([cr(t) for t in sh_st]), np.array([cr(t) for t in in_st])
    ysh_a2, yin_a2 = np.array([cr(t) for t in sh]), np.array([cr(t) for t in ind])
    assert np.allclose(ysh, ysh_a2) and np.allclose(yin[:-1], yin_a2[:-1])
    ax.axhline(1, color="grey", ls="--", lw=0.5)
    ax.plot(x, ysh, "o-", color=ARCH_C["ResNet-18"], ms=2.4, lw=0.9, label="Shared initialisation")
    ax.plot(x, yin, "s--", color=RED, ms=2.4, lw=0.9, label="Independent initialisation")
    ax.set_yscale("log")
    ax.set_ylim(0.06, 4e5)
    plain_log_ticks(ax, [0.1, 1, 10, 100, 1e3, 1e4, 1e5])
    ax.set_xticks(range(0, 10, 2))
    ax.set_xlim(-0.5, 9.5)
    ax.set_xlabel("Training transition")
    ax.set_ylabel("C/R ratio (log)")
    ax.legend(fontsize=SMALL, loc="upper right", handlelength=2.0, borderaxespad=0.1, labelspacing=0.25)
    ax.text(0.97, 0.70, f"Independent {yin[0]:.0f}:1 → {yin[-1]:.2f}:1", transform=ax.transAxes, fontsize=SMALL,
            color=RED, ha="right", va="top")
    title(ax, "Shared vs independent SAE initialisation")
    rows = [dict(transition=sh[i]["transition"], shared_tg=sh[i]["tg_h"], shared_ab=sh[i]["ab_h"], shared_di=sh[i]["di_h"],
                 shared_CR_A2_recount=ysh_a2[i], indep_tg=ind[i]["tg_h"], indep_ab=ind[i]["ab_h"],
                 indep_di=ind[i]["di_h"], indep_CR_A2_recount=yin_a2[i],
                 shared_tg_stored=sh_st[i]["tg_h"], shared_ab_stored=sh_st[i]["ab_h"], shared_di_stored=sh_st[i]["di_h"],
                 indep_tg_stored=in_st[i]["tg_h"], indep_ab_stored=in_st[i]["ab_h"], indep_di_stored=in_st[i]["di_h"],
                 shared_CR_plotted_stored=ysh[i], indep_CR_plotted_stored=yin[i])
            for i in range(10)]
    df = pd.DataFrame(rows)
    hc = pd.read_csv(RESULTS / "NOISE_null" / "tables" / "headline_cr.csv").set_index("lane")
    ref = hc.loc["ResNet18-100class-10ms-seed256-indepSAE"]
    assert abs(yin[0] - ref.CR_first) < 1e-6 and abs(yin_a2[-1] - ref.CR_last_real) < 1e-6
    STORE["c_indep_CR_first_last"] = [round(yin[0], 1), round(yin[-1], 3)]
    STORE["c_indep_CR_last_A2_recount"] = round(yin_a2[-1], 3)
    STORE["c_shared_s256_CR_last"] = round(ysh[-1], 3)
    STORE["c_n_transitions_both_below_1_last"] = bool(ysh[-1] < 1 and yin[-1] < 1)
    return df


# g-i: census, each run's stored pipeline thresholds
CENSUS_VARIANT = "dedup_stored"
CENSUS_COL = [("frac_tg", "Task-general"), ("frac_ab_ex", "Superclass-selective"), ("frac_di_ex", "Class-selective")]


def census_style(group, arch):
    if group == "tinyimagenet":
        return dict(color=ns.TINY, ls="-", marker="o")
    if group == "indep_init":
        return dict(color=RED, ls="--", marker="s")
    return dict(color=ARCH_C[arch], ls="-", marker="o")


def com(v):
    """Centre of mass over normalised training position (census_analyze.com)."""
    v = np.clip(np.asarray(v, float), 0, None)
    tau = np.linspace(0, 1, len(v))
    return float((tau * v).sum() / v.sum())


def panels_ghi(axes):
    d = pd.read_csv(RESULTS / "CENSUS_exclusive" / "census_exclusive_per_checkpoint.csv")
    d = d[(d.variant == CENSUS_VARIANT) & (d.checkpoint != "terminal")].copy()
    d["checkpoint"] = d.checkpoint.astype(int)
    d["progress"] = d.checkpoint / d.groupby("lane").checkpoint.transform("max")
    lanes = list(d.groupby(["group", "arch", "lane"]))
    assert len(lanes) == 11
    order = {"core": 0, "tinyimagenet": 1, "indep_init": 2}
    lanes.sort(key=lambda z: order[z[0][0]])
    coms = {}
    for (grp, arch, lane), g in lanes:
        g = g.sort_values("checkpoint")
        coms[lane] = [com(g[c].values) for c, _ in CENSUS_COL]
    n_order = sum(c[0] < c[1] < c[2] for c in coms.values())
    for k, (ax, (col, ttl)) in enumerate(zip(axes, CENSUS_COL)):
        top = d[col].max() * (1.34 if k == 2 else 1.12)
        for (grp, arch, lane), g in lanes:
            g = g.sort_values("checkpoint")
            st = census_style(grp, arch)
            ax.plot(g.progress, g[col], lw=0.8, ms=1.8, alpha=0.9, zorder=3 if grp != "core" else 2, **st)
            ax.plot([coms[lane][k]] * 2, [top * 0.955, top], color=st["color"], lw=0.9, zorder=4, solid_capstyle="butt")
        med = float(np.median([c[k] for c in coms.values()]))
        ax.axvline(med, color=DARK, ls=(0, (2, 1.5)), lw=1.3, zorder=1)
        if k == 0:
            ax.text(med + 0.02, 0.935, "median centre of mass", fontsize=SMALL, color=DARK, va="top",
                    ha="left", transform=ax.get_xaxis_transform())
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(0, top)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xlabel("Training progress (checkpoints)")
        title(ax, ttl)
    axes[0].set_ylabel("Fraction of alive features")
    h = [Line2D([], [], color=ARCH_C[a], lw=0.8, marker="o", ms=1.8, label=a) for a in ARCH]
    h += [Line2D([], [], color=ns.TINY, lw=0.8, marker="o", ms=1.8, label="ResNet-18, Tiny ImageNet"),
          Line2D([], [], color=RED, ls="--", lw=0.8, marker="s", ms=1.8, label="ResNet-18 s256, indep. init.")]
    # legend right of the median line, below the one-line median label (nothing crosses it)
    axes[0].legend(handles=h, fontsize=SMALL, loc="upper right", bbox_to_anchor=(1.0, 0.85), handlelength=1.2,
                   handletextpad=0.4, labelspacing=0.2, borderpad=0.2, borderaxespad=0.0)
    axes[2].text(0.03, 0.95, f"Ordering holds:\n{n_order} of 11 runs", transform=axes[2].transAxes,
                 fontsize=7, fontweight="bold", color=DARK, va="top", ha="left", linespacing=1.15)
    oc = pd.read_csv(RESULTS / "CENSUS_exclusive" / "ordering_counts.csv")
    r = oc[(oc.variant == CENSUS_VARIANT) & (oc.checkpoints == "no_terminal") & (oc.definition == "exclusive")].iloc[0]
    assert n_order == int(r.n_order_com) == 11
    STORE["g_ordering_n_of_11"] = int(r.n_order_com)
    STORE["g_superclass_rise_fall_n_of_11"] = int(r.n_ab_rise_fall)
    STORE["g_median_centre_of_mass_tg_ab_di"] = [round(float(np.median([c[k] for c in coms.values()])), 3)
                                                 for k in range(3)]
    cols = ["group", "arch", "lane", "checkpoint", "progress", "n_alive", "frac_tg", "frac_ab_ex", "frac_di_ex",
            "thr_sai", "thr_ssi", "thr_csi"]
    com_df = pd.DataFrame([dict(lane=l, com_task_general=c[0], com_superclass=c[1], com_class=c[2],
                                order_holds=c[0] < c[1] < c[2]) for l, c in coms.items()])
    return d[cols].sort_values(["group", "arch", "lane", "checkpoint"]), com_df


# j: superclass share without SAEs
def load_nosae():
    a = pd.read_csv(RESULTS / "DICTFREE" / "population" / "tables" / "population_per_checkpoint.csv")
    a = a[a.is_pooled].assign(set="open")
    b = pd.read_csv(RESULTS / "DATA_order" / "transient_stage" / "share_sc_per_checkpoint.csv")
    b = b[b.layer.isin(["avgpool", "norm", "encoder.ln"])].assign(set="held out")
    cols = ["set", "lane", "group", "arch", "seed", "layer", "checkpoint", "idx", "epoch", "val_acc", "raw_share_sc",
            "raw_share_sc_null_mean", "raw_share_sc_null_p95"]
    d = pd.concat([a[cols], b[cols]], ignore_index=True)
    assert d.lane.nunique() == 16 and (d.groupby("set").lane.nunique() == pd.Series({"held out": 7, "open": 9})).all()
    return d


def panel_j(ax, d):
    lo, hi = d.raw_share_sc_null_mean.mean(), d.raw_share_sc_null_p95.max()
    ax.axhspan(d.raw_share_sc_null_mean.min(), hi, color=SLATE, alpha=0.2, lw=0)
    ax.axhline(lo, color=SLATE, ls="--", lw=0.8)
    peaks = {}
    rows = []
    for (st, lane), g in sorted(d.groupby(["set", "lane"]), key=lambda kv: (kv[0][0] != "open", kv[0][1])):
        g = g.sort_values("idx")
        arch = g.arch.iloc[0]
        x = g.val_acc.values / g.val_acc.values[-1]
        y = g.raw_share_sc.values
        held = st == "held out"
        ax.plot(x, y, color=ARCH_C[arch], lw=0.7, alpha=0.85, zorder=2)
        ax.plot(x, y, "o", ms=2.0, color=ARCH_C[arch], mfc="white" if held else ARCH_C[arch], mew=0.6, zorder=3)
        k = int(np.argmax(y))
        peaks[lane] = 0 < k < len(y) - 1
        for xi, yi, (_, r) in zip(x, y, g.iterrows()):
            rows.append(dict(set=st, lane=lane, arch=arch, seed=r.seed, checkpoint=r.checkpoint, epoch=r.epoch,
                             val_acc=r.val_acc, progress=xi, superclass_share=yi,
                             random_grouping_mean=r.raw_share_sc_null_mean, random_grouping_p95=r.raw_share_sc_null_p95))
    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(0.15, 0.74)
    ax.set_xlabel("Validation accuracy (fraction of final)")
    ax.set_ylabel("Superclass share of\nbetween-class variance")
    h = [Line2D([], [], color=DARK, marker="o", ls="none", ms=2.4, label="Original networks (9)"),
         Line2D([], [], color=DARK, marker="o", mfc="white", ls="none", ms=2.4, mew=0.6, label="Held-out networks (7)")]
    ax.legend(handles=h, fontsize=SMALL, loc="center right", bbox_to_anchor=(1.0, 0.43), handletextpad=0.2,
              labelspacing=0.25, borderaxespad=0.1)
    ax.text(0.5, (lo + hi) / 2 + 0.005, "100 random groupings of the classes", fontsize=SMALL, color=SLATE,
            ha="center", va="center")
    title(ax, "Activations, no SAEs: superclass share over training")
    STORE["j_interior_peak"] = f"{sum(peaks.values())}/{len(peaks)}"
    STORE["j_random_grouping_mean_and_max_p95"] = [round(lo, 4), round(hi, 4)]
    first = d.sort_values("idx").groupby("lane").raw_share_sc.first()
    STORE["j_initial_share_range"] = [round(first.min(), 3), round(first.max(), 3)]
    return pd.DataFrame(rows), peaks


def interior_peaks(d):
    out = {}
    for lane, g in d.groupby("lane"):
        y = g.sort_values("idx").raw_share_sc.values
        k = int(np.argmax(y))
        out[lane] = 0 < k < len(y) - 1
    return out


def panel_k_rise_fall(ax, d):
    """Share at initialisation, at its peak and at the end of training, one line per network."""
    lo, hi = d.raw_share_sc_null_mean.mean(), d.raw_share_sc_null_p95.max()
    ax.axhspan(d.raw_share_sc_null_mean.min(), hi, color=SLATE, alpha=0.2, lw=0)
    ax.axhline(lo, color=SLATE, ls="--", lw=0.8)
    rows = []
    rng = np.random.default_rng(1)
    for (st, lane), g in sorted(d.groupby(["set", "lane"]), key=lambda kv: (kv[0][0] != "open", kv[0][1])):
        g = g.sort_values("idx")
        arch = g.arch.iloc[0]
        y = g.raw_share_sc.values
        k = int(np.argmax(y))
        v = [y[0], y[k], y[-1]]
        jx = rng.uniform(-0.06, 0.06)
        xs = np.array([0, 1, 2]) + jx
        held = st == "held out"
        ax.plot(xs, v, color=ARCH_C[arch], lw=0.8, alpha=0.8, zorder=2)
        ax.plot(xs, v, "o", ms=2.8, color=ARCH_C[arch], mfc="white" if held else ARCH_C[arch], mew=0.7, zorder=3)
        rows.append(dict(set=st, lane=lane, arch=arch, share_initial=v[0], share_peak=v[1], share_final=v[2],
                         peak_checkpoint_index=k, n_checkpoints=len(y), interior_peak=0 < k < len(y) - 1,
                         rise=v[1] - v[0], decline=v[1] - v[2]))
    df = pd.DataFrame(rows)
    n_int = int(df.interior_peak.sum())
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Initialisation", "Peak", "End of\ntraining"])
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(0.15, 0.74)
    ax.set_ylabel("Superclass share of\nbetween-class variance")
    ax.text(1.0, 0.36, f"Rise, then fall:\n{n_int} of {len(df)} networks", fontsize=7, fontweight="bold", color=DARK,
            ha="center", va="center", linespacing=1.15)
    ax.text(1.0, (lo + hi) / 2 + 0.005, "random groupings", fontsize=SMALL, color=SLATE, ha="center", va="center")
    title(ax, "Rise and fall per network")
    STORE["k_interior_peak"] = f"{n_int}/{len(df)}"
    STORE["k_rise_range"] = [round(df.rise.min(), 3), round(df.rise.max(), 3)]
    STORE["k_decline_range"] = [round(df.decline.min(), 3), round(df.decline.max(), 3)]
    return df


# SimCLR without labels (simclr_train.py; ResNet-18, seeds 7/42/137, epochs 0-100). ed1g_share_sc (simclr_analysis.py)
# uses the same estimator, layer, test images and permutation null as panels j and k.
SIMCLR_C = DARK
SIMCLR_EPOCHS = [0, 1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
SIMCLR_DX = 0.26           # SimCLR group sits right of each supervised cluster


def load_simclr():
    m = pd.read_csv(RESULTS / "SIMCLR" / "simclr_measures.csv")
    m = m[(m.model == "simclr") & m.measure.isin(["ed1g_share_sc", "ed1g_null_mean", "ed1g_null_p95"])]
    w = m.pivot_table(index=["seed", "epoch"], columns="measure", values="value").reset_index()
    assert sorted(w.seed.unique()) == [7, 42, 137]
    assert all(sorted(g.epoch) == SIMCLR_EPOCHS for _, g in w.groupby("seed"))
    summ = json.loads((RESULTS / "SIMCLR" / "simclr_summary.json").read_text())
    for r in (r for r in summ["by_model_epoch"] if r["model"] == "simclr"):
        assert abs(w[w.epoch == r["epoch"]].ed1g_share_sc.mean() - r["ed1g_share_sc"]) < 1e-4
    return w.sort_values(["seed", "epoch"])


def panel_k_simclr(ax, w):
    rows = []
    for i, (seed, g) in enumerate(w.groupby("seed")):
        y = g.ed1g_share_sc.values
        k = int(np.argmax(y))
        xs = np.array([0, 1, 2]) + SIMCLR_DX + (i - 1) * 0.03
        v = [y[0], y[k], y[-1]]
        ax.plot(xs, v, color=SIMCLR_C, ls=(0, (3, 1.6)), lw=0.8, alpha=0.9, zorder=4)
        ax.plot(xs, v, "s", ms=2.4, color=SIMCLR_C, mew=0, zorder=5)
        for j, (_, r) in enumerate(g.iterrows()):
            rows.append(dict(set="SimCLR", lane=f"simclr_ResNet-18_s{seed}", arch="ResNet-18", seed=seed,
                             epoch=int(r.epoch), superclass_share=r.ed1g_share_sc,
                             random_grouping_mean=r.ed1g_null_mean, random_grouping_p95=r.ed1g_null_p95,
                             is_initial=j == 0, is_peak=j == k, is_final=j == len(y) - 1))
    h = [Line2D([], [], color=SIMCLR_C, ls=(0, (3, 1.6)), lw=0.8, marker="s", ms=2.4, mew=0,
                label="SimCLR,\nno labels (3)")]
    ax.legend(handles=h, fontsize=SMALL, loc="upper right", handlelength=2.2, handletextpad=0.4,
              borderaxespad=0.1, frameon=False)
    return pd.DataFrame(rows)


# Supplementary Fig. 5a: same-SAE noise floor
def panel_noise_floor(ax):
    d = pd.read_csv(RESULTS / "NOISE_null" / "tables" / "resnet_dup_null_per_transition.csv")
    d = d[d.group == "core"].copy()
    d["k"] = d.transition.str.split("->").str[0].astype(int)
    # sums over the two duplicate-SAE layers (layer4, avgpool) -> per layer
    d["real"] = (d.ab_h_real + d.di_h_real) / 2
    d["null"] = (d.ab_h_null + d.di_h_null) / 2
    g = d.groupby("k")
    x = np.arange(10)
    for col, c, ls, mk, lab in (("real", ARCH_C["ResNet-18"], "-", "o", "Consecutive checkpoints"),
                                ("null", SLATE, "--", "s", "Same checkpoint, two SAEs")):
        m, lo, hi = g[col].mean() / 1e3, g[col].min() / 1e3, g[col].max() / 1e3
        ax.fill_between(x, lo, hi, color=c, alpha=0.2, lw=0)
        ax.plot(x, m, color=c, ls=ls, marker=mk, ms=2.2, lw=0.9, label=lab)
    ax.set_xticks(range(0, 10, 2))
    ax.set_xlim(-0.4, 9.4)
    ax.set_ylim(0, None)
    ax.set_xlabel("Training transition")
    ax.set_ylabel("Ab-E + Di-E events per layer\n(thousands)")
    ax.legend(fontsize=SMALL, loc="upper left", handlelength=2.0, borderaxespad=0.1, labelspacing=0.25)
    title(ax, "Same-SAE noise floor (ResNet-18)")
    ratio = g["null"].sum() / g["real"].sum()
    STORE["k_null_over_real_first2_last3"] = [round(float(g["null"].sum()[:2].sum() / g["real"].sum()[:2].sum()), 3),
                                               round(float(g["null"].sum()[-3:].sum() / g["real"].sum()[-3:].sum()), 3)]
    out = d[["lane", "transition", "ab_h_real", "di_h_real", "ab_h_null", "di_h_null", "real", "null"]].copy()
    out["ratio_null_over_real_pooled_by_transition"] = out.transition.map(
        {f"{k}->{k + 1}": v for k, v in ratio.items()})
    return out.rename(columns={"real": "abE_plus_diE_real_per_layer", "null": "abE_plus_diE_same_sae_null_per_layer"})


# l: robustness ledger
def ledger_data(peaks):
    runs = [(a, s) for a in ARCH for s in (42, 137, 256)]
    A3 = pd.read_csv(RESULTS / "MATCHING" / "process_ordering_by_rule.csv")
    A3 = A3[A3.weighting == "sample_weighted"]
    A5 = json.loads((RESULTS / "TRANSFORMED" / "transformed_results.json").read_text())["processes"]["paper_all_layers"]

    def a3(a, s, rule, col):
        r = A3[(A3.arch == a) & (A3.seed == s) & (A3.rule == rule)]
        assert len(r) == 1
        return bool(r[col].iloc[0])

    def a5(a, s, key, col):
        r = [x for x in A5[key]["per_lane"] if x["arch"] == a and x["seed"] == s]
        assert len(r) == 1
        return bool(r[0][col])

    def declines(x):  # cr_first is None when the first transition has no Di-E (infinite ratio)
        return x["cr_first"] is None or x["cr_last_real"] < x["cr_first"]

    alt = ("mutual_best", "greedy", "confirmed")
    spec = [  # (section, label, fn(a, s) -> True/False/None)
        ("Developmental ordering", "Hungarian (main analysis)",
         lambda a, s: a3(a, s, "hungarian_pipeline_all_layers", "order_tg_ab_di")),
        (None, "Mutual best matches", lambda a, s: a3(a, s, "mutual_best", "order_tg_ab_di")),
        (None, "Greedy matching", lambda a, s: a3(a, s, "greedy", "order_tg_ab_di")),
        (None, "No partner above r = 0.2", lambda a, s: a3(a, s, "confirmed", "order_tg_ab_di")),
        (None, "Transformed as deaths + births", lambda a, s: a5(a, s, "turnover", "com_order_holds")),
        (None, "Transformed as continuations", lambda a, s: a5(a, s, "continuity", "com_order_holds")),
        ("C/R ratio declines", "All three matching rules", lambda a, s: all(a3(a, s, r, "cr_declines") for r in alt)),
        (None, "Both transformed treatments",
         lambda a, s: all(declines(x) for k in ("turnover", "continuity")
                          for x in A5[k]["per_lane"] if x["arch"] == a and x["seed"] == s)),
        ("Without SAEs", "Superclass share rises, falls",
         lambda a, s: peaks.get(f"core_{a}_s{s}") if s != 256 else None),
    ]
    rows = []
    for sec, lab, fn in spec:
        vals = [fn(a, s) for a, s in runs]
        rows.append(dict(section=sec, label=lab, vals=vals))
    n_order_alt = sum(a3(a, s, r, "order_tg_ab_di") for a, s in runs for r in alt)
    n_cr_alt = sum(a3(a, s, r, "cr_declines") for a, s in runs for r in alt)
    n_cr_tr = sum(declines(x) for k in ("turnover", "continuity") for x in A5[k]["per_lane"])
    STORE["l_matching_order_n_of_27"] = int(n_order_alt)
    STORE["l_matching_cr_decline_n_of_27"] = int(n_cr_alt)
    STORE["l_transformed_order"] = {k: int(A5[k]["n_com_order_holds"]) for k in ("excluded", "turnover", "continuity")}
    STORE["l_transformed_cr_decline_n_of_18"] = int(n_cr_tr)
    for k in ("turnover", "continuity"):
        assert sum(declines(x) for x in A5[k]["per_lane"]) == A5[k]["n_decline_first_to_last_real"]
    for r in rows:
        v = [x for x in r["vals"] if x is not None]
        r["count"] = f"{sum(v)}/{len(v)}"
    rows[6]["count"] = f"{n_cr_alt}/27"
    rows[7]["count"] = f"{n_cr_tr}/18"
    return runs, rows


def panel_l(ax, runs, rows):
    lines = []
    for r in rows:
        if r["section"]:
            lines.append(("sec", r["section"]))
        lines.append(("row", r))
    n = len(lines)
    xs = np.arange(len(runs))
    for i, (kind, obj) in enumerate(lines):
        y = n - 1 - i
        if kind == "sec":
            ax.text(-0.7, y, obj, ha="right", va="center", fontsize=SMALL, fontweight="bold", color=DARK)
            continue
        ax.text(-0.7, y, obj["label"], ha="right", va="center", fontsize=SMALL, color=DARK)
        for j, ((a, s), v) in enumerate(zip(runs, obj["vals"])):
            if v is None:
                ax.text(j, y, "–", ha="center", va="center", fontsize=SMALL, color=GRID)
            elif v:
                ax.plot(j, y, "o", ms=3.0, color=DARK, mew=0)
            else:
                ax.plot(j, y, "o", ms=3.0, mfc="white", mec=DARK, mew=0.7)
        ax.text(len(runs) - 0.2, y, obj["count"], ha="left", va="center", fontsize=SMALL, color=DARK)
    for j, (a, s) in enumerate(runs):
        ax.text(j, n - 0.25, str(s), ha="center", va="bottom", fontsize=SMALL, color=ARCH_C[a], rotation=90)
    for k, a in enumerate(ARCH):
        ax.text(3 * k + 1, n + 1.15, a, ha="center", va="bottom", fontsize=SMALL, color=ARCH_C[a])
    for xv in (2.5, 5.5):
        ax.axvline(xv, color="#E5E7EB", lw=0.6, zorder=0)
    ax.set_xlim(-0.6, len(runs) + 0.9)
    ax.set_ylim(-0.6, n + 1.9)
    ax.axis("off")


def main():
    ns.apply()
    fig = plt.figure(figsize=(W_MM * ns.MM, H_MM * ns.MM))
    C = MMCanvas(fig, W_MM, H_MM)
    AW, AH = 43, 28
    cols = [14, 75, 136]
    r1, r2, r3, r4 = 5, 46, 90, 131
    ax_a, ax_b, ax_c = (C.ax(x, r1, AW, AH) for x in cols)
    ax_d, ax_e, ax_f = (C.ax(x, r2, AW, AH) for x in cols)
    ax_g, ax_h, ax_i = (C.ax(x, r3, AW, 27) for x in cols)
    ax_j = C.ax(14, r4, 82, 30)
    ax_k = C.ax(124, r4, 55, 30)

    R, N, T = load_core()
    panel_a(ax_a, R)
    panel_b(ax_b, R)
    cdf = panel_c(ax_c)
    panel_d(ax_d, N)
    panel_e(ax_e, T)
    panel_f(ax_f, N)
    census, com_df = panels_ghi([ax_g, ax_h, ax_i])
    nos = load_nosae()
    jdf, peaks = panel_j(ax_j, nos)
    kdf = panel_k_rise_fall(ax_k, nos)
    assert all(peaks[l] == v for l, v in zip(kdf.lane, kdf.interior_peak))
    k_simclr = panel_k_simclr(ax_k, load_simclr())

    for ax in (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h, ax_i, ax_j, ax_k):
        ax_style(ax)
    letters = dict(a=ax_a, b=ax_b, c=ax_c, d=ax_d, e=ax_e, f=ax_f, g=ax_g, h=ax_h, i=ax_i, j=ax_j, k=ax_k)
    col_left = {14: 1.0, 75: 62.0, 136: 123.0, 124: 106.0}
    for L, ax in letters.items():
        x0, y0, w, h = ax.get_position().bounds
        xmm, wmm, hmm = x0 * W_MM, w * W_MM, h * H_MM
        left = col_left.get(round(xmm), 1.0)
        ns.panel(ax, L, x=(left - xmm) / wmm, y=1 + 5.5 / hmm)

    ns.source_data(R, FIG, "a_b")
    ns.source_data(cdf, FIG, "c")
    ns.source_data(core_c_check(), FIG, "c_core_check")
    ns.source_data(N, FIG, "d_f")
    ns.source_data(T, FIG, "e")
    ns.source_data(census, FIG, "g_i")
    ns.source_data(com_df, FIG, "g_i_centre_of_mass")
    ns.source_data(jdf, FIG, "j")
    ns.source_data(kdf, FIG, "k")
    ns.source_data(k_simclr, FIG, "k_simclr")
    out = ns.save_final(fig, FIG, W_MM, ed=True)
    ns.source_data(pd.DataFrame([dict(key=k, value=json.dumps(v)) for k, v in STORE.items()]), FIG, "stored_values")
    print(out)
    for k, v in STORE.items():
        print(k, v)


if __name__ == "__main__":
    import os as _os
    if not _os.environ.get("ES_FIG_OUT"):
        raise SystemExit("validation_panels.py saves under the name ed1 (Extended Data Fig. 1 is drawn by extended/ed1.py); "
                         "set ES_FIG_OUT to a separate output folder to draw this stand-alone composite.")
    main()
