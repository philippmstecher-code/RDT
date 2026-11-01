#!/usr/bin/env python3
"""Extended Data Fig. 1: reliable tracking, independent SAEs and SAE-free measures reproduce the developmental order.

a    false births / deaths between two SAEs of the same checkpoint (9 standard runs; mean +- s.d.)
b    within-checkpoint stable matches, bars in the order of a, dots = runs
c    construction/refinement ratio (Tg-E + Ab-E)/max(Di-E, 1) (checks_lib.cr, as in Fig. 1c), log axis, two sub-axes
     sharing y: left, ResNet-18 seed 256 with shared SAE init (4x), independent SAE init, 8x SAE; right, ResNet-18
     seed 42 trained for 50 or 200 epochs. Stored pipeline classification (sae_results.json process_intensity),
     duplicated final transition removed.
d-f  feature census (exclusive categories, stored thresholds); median centre of mass in the category colour
g    superclass share of between-class variance without SAEs, 16 networks, vs 100 random groupings
h    share at initialisation, peak and end; SimCLR without labels dashed
Reads stored results only; nothing is recomputed beyond what is plotted.
Usage: python scripts/paper/figures/extended/ed1.py  (ES_FIG_OUT=<dir> redirects the output)
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import nmi_style as ns  # noqa: E402
import core_runs_lib as L4  # noqa: E402
import validation_panels  # noqa: E402
from checks_lib import ARCHS, INDEP_EXP, MMCanvas, cr, sae_results  # noqa: E402

FIG = "ed1"
W_MM, H_MM = 180, 182
ARCH_C = ns.ARCH
RB = ARCH_C["ResNet-18"]
RED = ns.RED_600
SLATE = ns.SLATE
DARK = ns.REF_DARK
FB, FD = ns.PROC["As-E"], ns.PROC["De-E"]   # births cyan, deaths red (process colours, as Fig. 3b)
CAT_C = [ns.PROC["Tg-E"], ns.PROC["Ab-E"], ns.PROC["Di-E"]]
TS = 6.8   # title size
FS = 6.2   # tick / annotation size
STORE = {}

# line encodings of panel c (reference = standard run: shared SAE initialisation, 4x SAE, 50 epochs)
STY = {
    "standard": dict(color=RB, ls="-", marker="o", mfc=RB),
    "independent": dict(color=RED, ls="--", marker="s", mfc=RED),
    "8x": dict(color=RB, ls=":", marker="D", mfc="white"),
    "200ep": dict(color=RB, ls="-.", marker="^", mfc="white"),
}


def style(ax):
    ax.tick_params(labelsize=FS, pad=1.5)
    ax.xaxis.label.set_size(6.5)
    ax.yaxis.label.set_size(6.5)


def title(ax, t, **kw):
    ax.set_title(t, fontsize=TS, pad=3, **kw)


# a, b
def panel_a(ax, R):
    x, w = np.arange(3), 0.36
    g = R.groupby("arch")
    for off, col, c in ((-w / 2, "false_birth", FB), (w / 2, "false_death", FD)):
        m = [g[col].mean()[a] for a in ARCHS]
        s = [g[col].std(ddof=0)[a] for a in ARCHS]
        ax.bar(x + off, m, w, yerr=s, color=c, alpha=0.8, edgecolor="white", lw=0.3,
               error_kw=dict(ecolor="black", elinewidth=0.6, capsize=1.5, capthick=0.6))
    for yv in (2, 10):
        ax.axhline(yv, color="grey", ls="--", lw=0.5, alpha=0.6, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(ARCHS)
    ax.set_ylabel("False-positive rate (%)")
    ax.set_ylim(0, 21)
    title(ax, "False births stay below 2%")
    STORE["a_false_birth_mean_by_arch"] = {a: round(float(g.false_birth.mean()[a]), 2) for a in ARCHS}


def panel_b(ax, R):
    x = np.arange(3)
    g = R.groupby("arch")["stable"]
    m = [g.mean()[a] for a in ARCHS]
    s = [g.std(ddof=0)[a] for a in ARCHS]
    ax.bar(x, m, 0.55, yerr=s, color=[ARCH_C[a] for a in ARCHS], alpha=0.8, edgecolor="white", lw=0.3,
           error_kw=dict(ecolor="black", elinewidth=0.6, capsize=1.5, capthick=0.6))
    rng = np.random.default_rng(3)
    for i, a in enumerate(ARCHS):
        v = R[R.arch == a].stable.values
        ax.plot(i + rng.uniform(-0.12, 0.12, len(v)), v, "o", color="black", ms=2.0, alpha=0.55, mew=0, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(ARCHS)
    ax.set_ylabel("Stable matches (%)")
    ax.set_ylim(0, 70)
    lo, hi = R.stable.min(), R.stable.max()
    title(ax, f"{lo:.0f}–{hi:.0f}% of features match stably")
    STORE["b_stable_range_pct"] = [round(lo, 2), round(hi, 2)]


# c
def c_series():
    lanes = {  # key: (sub-axis, label, seed, (experiment, lane prefix))
        "standard_s256": ("left", "standard", 256, L4.ED4_LANES["4x_s256"]),
        "independent": ("left", "independent", 256, (INDEP_EXP, "independent_sae_seed256")),
        "8x": ("left", "8x", 256, L4.ED4_LANES["8x"]),
        "standard_s42": ("right", "standard", 42, L4.ED4_LANES["4x"]),
        "200ep": ("right", "200ep", 42, L4.ED4_LANES["200ep"]),
    }
    rows = []
    for key, (side, sty, seed, (exp, pre)) in lanes.items():
        pi = sae_results(exp, pre)["process_intensity"]
        assert pi[-1]["transition"].endswith("->terminal")
        for i, t in enumerate(pi[:-1]):
            rows.append(dict(sub_axis=side, condition={"standard": "standard (shared init, 4x SAE, 50 epochs)",
                                                       "independent": "independent SAE initialisation (4x)",
                                                       "8x": "8x SAE (shared init)",
                                                       "200ep": "200 epochs (shared init, 4x SAE)"}[sty],
                             style=sty, arch="ResNet-18", seed=seed, lane_prefix=pre, transition_index=i,
                             transition=t["transition"], tg_h=t["tg_h"], ab_h=t["ab_h"], di_h=t["di_h"],
                             di_floored_to_1=t["di_h"] == 0, construction_refinement_ratio=cr(t)))
    df = pd.DataFrame(rows)
    assert (df.groupby(["sub_axis", "style"]).size() == 10).all()
    # consistency with the stored shared-init / independent-init ratios
    old = pd.read_csv(ns.OUT / "final" / "source_data_ed" / "ed1_c.csv")
    l = df[df.sub_axis == "left"]
    assert np.allclose(l[l["style"] == "standard"].construction_refinement_ratio.values, old.shared_CR_plotted_stored)
    assert np.allclose(l[l["style"] == "independent"].construction_refinement_ratio.values, old.indep_CR_plotted_stored)
    return df


def ratio_fmt(v):
    return f"{v:,.0f}:1" if v >= 10 else f"{v:.2f}:1"


def panel_c(axL, axR, df):
    for ax, side in ((axL, "left"), (axR, "right")):
        ax.axhline(1, color="grey", ls="--", lw=0.5, zorder=0)
        sub = df[df.sub_axis == side]
        order = ["standard", "8x", "200ep", "independent"]
        for st in [s for s in order if s in set(sub["style"])]:
            g = sub[sub["style"] == st].sort_values("transition_index")
            s = STY[st]
            ax.plot(g.transition_index, g.construction_refinement_ratio, color=s["color"], ls=s["ls"], lw=0.9,
                    marker=s["marker"], ms=2.6, mfc=s["mfc"], mec=s["color"], mew=0.7, zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(0.06, 6e5)
        L4.plain_log_ticks(ax, [0.1, 1, 10, 100, 1e3, 1e4, 1e5])
        ax.set_xticks(range(0, 10, 2))
        ax.set_xlim(-0.5, 9.5)
        style(ax)
    axL.set_ylabel("Construction/\nrefinement ratio")
    axL.set_xlabel("Transition (seed 256)")
    axR.set_xlabel("Transition (seed 42)")
    axR.tick_params(labelleft=False)
    axR.spines["left"].set_visible(True)
    last = {}
    for side, st in (("left", "standard"), ("left", "independent"), ("left", "8x"), ("right", "standard"),
                     ("right", "200ep")):
        g = df[(df.sub_axis == side) & (df["style"] == st)].sort_values("transition_index")
        last[f"{side}_{st}"] = (float(g.construction_refinement_ratio.iloc[0]),
                                float(g.construction_refinement_ratio.iloc[-1]))
    i0, i1 = last["left_independent"]
    axL.text(0.97, 0.97, f"independent\n{ratio_fmt(i0)} → {ratio_fmt(i1)}", transform=axL.transAxes, ha="right",
             va="top", fontsize=FS, color=RED, linespacing=1.1)
    e0, e1 = last["right_200ep"]
    axR.text(0.97, 0.97, f"200 epochs\n{ratio_fmt(e0)} → {ratio_fmt(e1)}", transform=axR.transAxes, ha="right",
             va="top", fontsize=FS, color=RB, linespacing=1.1)
    STORE["c_first_last"] = {k: [round(a, 3), round(b, 3)] for k, (a, b) in last.items()}
    return last


# d-f census
CENSUS_TITLES = ["Task-general features peak first", "Superclass features peak mid-training",
                 "Class features rise last"]


def census_style(group, arch):
    if group == "tinyimagenet":
        return dict(color=ns.TINY, ls="-", marker="o", alpha=0.9, z=3)
    if group == "indep_init":
        return dict(color=RED, ls="--", marker="s", alpha=0.95, z=4)
    return dict(color=ARCH_C[arch], ls="-", marker="o", alpha=0.55 if arch == "CCT-7" else 0.85, z=2)


def panels_census(axes):
    d = pd.read_csv(validation_panels.RESULTS / "CENSUS_exclusive" / "census_exclusive_per_checkpoint.csv")
    d = d[(d.variant == validation_panels.CENSUS_VARIANT) & (d.checkpoint != "terminal")].copy()
    d["checkpoint"] = d.checkpoint.astype(int)
    d["progress"] = d.checkpoint / d.groupby("lane").checkpoint.transform("max")
    lanes = list(d.groupby(["group", "arch", "lane"]))
    assert len(lanes) == 11
    order = {"core": 0, "tinyimagenet": 1, "indep_init": 2}
    lanes.sort(key=lambda z: order[z[0][0]])
    coms = {lane: [validation_panels.com(g.sort_values("checkpoint")[c].values) for c, _ in validation_panels.CENSUS_COL]
            for (_, _, lane), g in lanes}
    n_order = sum(c[0] < c[1] < c[2] for c in coms.values())
    meds = []
    for k, (ax, (col, _)) in enumerate(zip(axes, validation_panels.CENSUS_COL)):
        top = d[col].max() * 1.13
        for (grp, arch, lane), g in lanes:
            g = g.sort_values("checkpoint")
            s = census_style(grp, arch)
            ax.plot(g.progress, g[col], color=s["color"], ls=s["ls"], marker=s["marker"], lw=0.7, ms=1.6,
                    alpha=s["alpha"], zorder=s["z"])
            ax.plot([coms[lane][k]] * 2, [top * 0.955, top], color=s["color"], lw=0.8, zorder=5,
                    solid_capstyle="butt")
        med = float(np.median([c[k] for c in coms.values()]))
        meds.append(med)
        ax.axvline(med, color=CAT_C[k], ls=(0, (2.2, 1.4)), lw=1.4, zorder=1)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(0, top)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xlabel("Training progress")
        title(ax, CENSUS_TITLES[k])
        style(ax)
    axes[0].set_ylabel("Fraction of alive features")
    axes[2].text(0.03, 0.86, f"order holds in\n{n_order} of 11", transform=axes[2].transAxes, fontsize=FS,
                 color=DARK, va="top", ha="left", linespacing=1.1)
    oc = pd.read_csv(validation_panels.RESULTS / "CENSUS_exclusive" / "ordering_counts.csv")
    r = oc[(oc.variant == validation_panels.CENSUS_VARIANT) & (oc.checkpoints == "no_terminal") & (oc.definition == "exclusive")]
    assert n_order == int(r.iloc[0].n_order_com) == 11
    STORE["d_f_order_n_of_11"] = n_order
    STORE["d_f_median_com"] = [round(m, 3) for m in meds]
    cols = ["group", "arch", "lane", "checkpoint", "progress", "n_alive", "frac_tg", "frac_ab_ex", "frac_di_ex",
            "thr_sai", "thr_ssi", "thr_csi"]
    com_df = pd.DataFrame([dict(lane=l, com_task_general=c[0], com_superclass=c[1], com_class=c[2],
                                order_holds=c[0] < c[1] < c[2]) for l, c in coms.items()])
    return d[cols].sort_values(["group", "arch", "lane", "checkpoint"]), com_df


# g, h
def band(ax, d):
    lo, hi = d.raw_share_sc_null_mean.mean(), d.raw_share_sc_null_p95.max()
    ax.axhspan(d.raw_share_sc_null_mean.min(), hi, color=SLATE, alpha=0.2, lw=0)
    ax.axhline(lo, color=SLATE, ls="--", lw=0.7)
    return lo, hi


def panel_g(ax, d):
    lo, hi = band(ax, d)
    rows = []
    for (st, lane), g in sorted(d.groupby(["set", "lane"]), key=lambda kv: (kv[0][0] != "open", kv[0][1])):
        g = g.sort_values("idx")
        arch = g.arch.iloc[0]
        x = g.val_acc.values / g.val_acc.values[-1]
        y = g.raw_share_sc.values
        held = st == "held out"
        ax.plot(x, y, color=ARCH_C[arch], lw=0.7, alpha=0.85, zorder=2)
        ax.plot(x, y, "o", ms=1.9, color=ARCH_C[arch], mfc="white" if held else ARCH_C[arch], mew=0.55, zorder=3)
        for xi, yi, (_, r) in zip(x, y, g.iterrows()):
            rows.append(dict(set=st, lane=lane, arch=arch, seed=r.seed, checkpoint=r.checkpoint, epoch=r.epoch,
                             val_acc=r.val_acc, progress=xi, superclass_share=yi,
                             random_grouping_mean=r.raw_share_sc_null_mean, random_grouping_p95=r.raw_share_sc_null_p95))
    ax.text(0.5, (d.raw_share_sc_null_mean.min() + hi) / 2, "100 random groupings", fontsize=FS, color=SLATE,
            ha="center", va="center")
    ax.set_xlim(-0.03, 1.04)
    ax.set_ylim(0.15, 0.74)
    ax.set_xlabel("Validation accuracy (fraction of final)")
    ax.set_ylabel("Superclass share of\nbetween-class variance")
    title(ax, "Superclass structure rises, then falls, without SAEs")
    STORE["g_random_grouping_mean_and_max_p95"] = [round(lo, 4), round(hi, 4)]
    first = d.sort_values("idx").groupby("lane").raw_share_sc.first()
    STORE["g_initial_share_range"] = [round(first.min(), 3), round(first.max(), 3)]
    return pd.DataFrame(rows)


def panel_h(ax, d, w):
    band(ax, d)
    rows = []
    rng = np.random.default_rng(1)
    for (st, lane), g in sorted(d.groupby(["set", "lane"]), key=lambda kv: (kv[0][0] != "open", kv[0][1])):
        g = g.sort_values("idx")
        arch = g.arch.iloc[0]
        y = g.raw_share_sc.values
        k = int(np.argmax(y))
        v = [y[0], y[k], y[-1]]
        xs = np.array([0, 1, 2]) + rng.uniform(-0.06, 0.06)
        held = st == "held out"
        ax.plot(xs, v, color=ARCH_C[arch], lw=0.7, alpha=0.8, zorder=2)
        ax.plot(xs, v, "o", ms=2.4, color=ARCH_C[arch], mfc="white" if held else ARCH_C[arch], mew=0.6, zorder=3)
        rows.append(dict(set=st, lane=lane, arch=arch, share_initial=v[0], share_peak=v[1], share_final=v[2],
                         peak_checkpoint_index=k, n_checkpoints=len(y), interior_peak=0 < k < len(y) - 1,
                         rise=v[1] - v[0], decline=v[1] - v[2]))
    df = pd.DataFrame(rows)
    srows = []
    for i, (seed, g) in enumerate(w.groupby("seed")):
        y = g.ed1g_share_sc.values
        k = int(np.argmax(y))
        xs = np.array([0, 1, 2]) + validation_panels.SIMCLR_DX + (i - 1) * 0.03
        ax.plot(xs, [y[0], y[k], y[-1]], color=DARK, ls=(0, (3, 1.6)), lw=0.8, marker="s", ms=2.2, mew=0, zorder=4)
        for j, (_, r) in enumerate(g.iterrows()):
            srows.append(dict(set="SimCLR", lane=f"simclr_ResNet-18_s{seed}", arch="ResNet-18", seed=seed,
                              epoch=int(r.epoch), superclass_share=r.ed1g_share_sc,
                              random_grouping_mean=r.ed1g_null_mean, random_grouping_p95=r.ed1g_null_p95,
                              is_initial=j == 0, is_peak=j == k, is_final=j == len(y) - 1))
    n_int = int(df.interior_peak.sum())
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Initialisation", "Peak", "End"])
    ax.set_xlim(-0.3, 2.45)
    ax.set_ylim(0.15, 0.74)
    ax.set_ylabel("Superclass share")
    title(ax, f"{n_int} of {len(df)} supervised networks rise, then fall")
    STORE["h_interior_peak"] = f"{n_int}/{len(df)}"
    return df, pd.DataFrame(srows)


# legend strip
def legend_strip(fig):
    def ln(st, label):
        s = STY[st]
        return Line2D([], [], color=s["color"], ls=s["ls"], marker=s["marker"], ms=2.6, mfc=s["mfc"],
                      mec=s["color"], mew=0.7, lw=0.9, label=label)
    row1 = [Patch(color=FB, alpha=0.8, label="false birth"), Patch(color=FD, alpha=0.8, label="false death")]
    row1 += [Line2D([], [], color=ARCH_C[a], marker="o", ms=2.2, lw=0.8, label=a) for a in ARCHS]
    row1 += [Line2D([], [], color=ns.TINY, marker="o", ms=2.2, lw=0.8, label="Tiny ImageNet"),
             Line2D([], [], color=DARK, marker="o", mfc="white", ms=2.4, mew=0.6, ls="none", label="held-out network"),
             Line2D([], [], color=DARK, ls=(0, (3, 1.6)), marker="s", ms=2.2, mew=0, lw=0.8,
                    label="SimCLR (no labels)")]
    # the standard run (solid) is shared init., 4x SAE and 50 epochs at once; each variant changes one of the three,
    # so the key lists each property beside its variant
    row2 = [ln("standard", "shared SAE init."), ln("independent", "independent SAE init."),
            ln("standard", "4× SAE"), ln("8x", "8× SAE"), ln("standard", "50 epochs"), ln("200ep", "200 epochs")]
    med = tuple(Line2D([], [], color=c, ls=(0, (2.2, 1.4)), lw=1.4) for c in CAT_C)
    labels2 = [h.get_label() for h in row2] + ["median centre of mass"]
    kw = dict(frameon=False, fontsize=6.5, handlelength=2.0, handletextpad=0.35, columnspacing=1.0,
              borderaxespad=0, borderpad=0)
    fig.legend(handles=row1, loc="upper center", bbox_to_anchor=(0.5, 1 - 1.0 / H_MM), ncol=len(row1), **kw)
    fig.legend(handles=row2 + [med], labels=labels2, loc="upper center", bbox_to_anchor=(0.5, 1 - 5.0 / H_MM),
               ncol=len(row2) + 1, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.2)}, **kw)


# assemble
def main():
    L4.apply_style()
    fig = plt.figure(figsize=(W_MM * ns.MM, H_MM * ns.MM))
    C = MMCanvas(fig, W_MM, H_MM)
    r1, r2, r3, AH = 18, 75, 132, 37
    ax_a = C.ax(13, r1, 38, AH)
    ax_b = C.ax(62, r1, 35, AH)
    ax_cL = C.ax(111, r1, 32, AH)
    ax_cR = C.ax(146, r1, 32, AH, sharey=ax_cL)
    ax_d, ax_e, ax_f = (C.ax(x, r2, 47, AH) for x in (13, 72, 131))
    ax_g = C.ax(13, r3, 98, AH)
    ax_h = C.ax(131, r3, 47, AH)

    R, _, _ = validation_panels.load_core()
    panel_a(ax_a, R)
    panel_b(ax_b, R)
    cdf = c_series()
    panel_c(ax_cL, ax_cR, cdf)
    fig.text((111 + 178) / 2 / W_MM, 1 - (r1 - 0.95) / H_MM, "Construction gives way to refinement in every variant",
             ha="center", va="bottom", fontsize=TS)
    census, com_df = panels_census([ax_d, ax_e, ax_f])
    nos = validation_panels.load_nosae()
    gdf = panel_g(ax_g, nos)
    hdf, hs = panel_h(ax_h, nos, validation_panels.load_simclr())
    for ax in (ax_a, ax_b, ax_g, ax_h):
        style(ax)
    legend_strip(fig)

    letters = [("a", ax_a, 1), ("b", ax_b, 51), ("c", ax_cL, 97), ("d", ax_d, 1), ("e", ax_e, 60), ("f", ax_f, 119),
               ("g", ax_g, 1), ("h", ax_h, 119)]
    for L, ax, xmm in letters:
        x0, y0, w, h = ax.get_position().bounds
        ns.panel(ax, L, x=(xmm - x0 * W_MM) / (w * W_MM), y=1 + 6.0 / (h * H_MM))

    ns.source_data(R[["arch", "seed", "false_birth", "false_death"]], FIG, "a")
    ns.source_data(R[["arch", "seed", "stable"]], FIG, "b")
    ns.source_data(cdf, FIG, "c")
    ns.source_data(census, FIG, "d_f")
    ns.source_data(com_df, FIG, "d_f_centre_of_mass")
    ns.source_data(gdf, FIG, "g")
    ns.source_data(hdf, FIG, "h")
    ns.source_data(hs, FIG, "h_simclr")
    out = ns.save_final(fig, FIG, W_MM, ed=True)
    print(out)
    for k, v in STORE.items():
        print(k, json.dumps(v))


if __name__ == "__main__":
    main()
