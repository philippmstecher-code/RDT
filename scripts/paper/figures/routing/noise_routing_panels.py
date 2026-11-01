#!/usr/bin/env python3
"""Label-noise panels for Extended Data Fig. 8 (interventions/ed8.py loads this module and draws its panels a and b
from it): fixed between-superclass label noise lowers the coarse structure throughout training, beyond label
similarity. Run alone, it draws the stand-alone figure below.

a  superclass probe, between-superclass noise minus clean, labels redrawn every epoch vs kept fixed
b  between - within over training, fixed labels: probe per checkpoint (top), fine-class accuracy per epoch with
   5-epoch bins (bottom)
c  similarity-matched control (NOISE_matched/noise_matched_per_seed.csv, noise_matched_results.json): paired contrasts, same y-scale as b
d  superclass read-outs, RDT and SAE-free (rdt_side_by_side.measures)
e  nesting ratio of class-selective features (RDT_noise/nesting_per_lane.csv)
f  Di-E fraction per transition, labels redrawn (p = 0.3)
Condition by line style (grey solid, clean training; dashed, between-superclass noise and its differences; dotted,
random noise), outcome by hue (blue, superclass; amber, class). Open markers, labels redrawn every epoch; filled,
fixed.
Usage: python scripts/paper/figures/routing/noise_routing_panels.py  (ES_FIG_OUT=<dir> redirects the output)
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nmi_style as ns  # noqa: E402
import noise_lib as L  # noqa: E402
from fig_common import RESULTS  # noqa: E402

NAME = "ed4"
W, H = 180, 166
SC, CL = ns.INDEX["SSI"], ns.INDEX["CSI"]
CLEAN_C, CLEAN_LW = "#9CA3AF", 0.9        # clean / standard training
LS = {"clean": "-", "within": "-", "between": "--", "random": ":"}
ALPHA = ns.BAR_ALPHA_CORE
ZERO = L.ZERO
TS = 7     # panel title size
AS = 6.5   # annotation size (key number)


def mm_axes(fig, x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def title(ax, txt):
    ax.set_title(txt, fontsize=TS, pad=3, linespacing=1.05, loc="left")


# a: redrawn vs fixed
def panel_a(ax):
    R_ = L.redrawn_metric("probe_sc_avgpool")
    Dr = R_["between_sc_p03"] - R_["standard"]
    P = L.static_probe()
    Df = P["between_sc"] - P["clean"]
    rows, out = [], {}
    for D, mfc, tag in ((Dr, "white", "redrawn"), (Df, SC, "fixed")):
        x = D.columns.values.astype(float)
        for s_, row in D.iterrows():
            ax.plot(x, row.values, color=SC, lw=0.5, alpha=0.35, ls=LS["between"], zorder=2)
            for e, v in zip(x, row.values):
                rows.append(dict(design=tag, seed=s_, epoch=int(e), between_minus_clean_pp=v))
        m = D.values.mean(0)
        ax.plot(x, m, color=SC, ls=LS["between"], lw=1.1, marker="o", ms=3.6, mfc=mfc, mec=SC, mew=0.9, zorder=4)
        m30, lo30, hi30 = L.t_ci(D[30].values)
        out[tag] = m30
        print(f"[a] {tag}: between - clean at epoch 30 {m30:+.2f} [{lo30:+.2f}, {hi30:+.2f}], "
              f"{int((D[30].values < 0).sum())}/5 below zero")
        for e, mm_ in zip(x, m):
            rows.append(dict(design=tag, seed="mean", epoch=int(e), between_minus_clean_pp=mm_))
        rows.append(dict(design=tag, seed="epoch 30 mean (95% CI)", epoch=30, between_minus_clean_pp=m30,
                         ci_lo=lo30, ci_hi=hi30))
    ax.axhline(0, color=ZERO, ls="--", lw=0.6, zorder=1)
    ax.text(29, 1.0, "redrawn", fontsize=6, color=L.ERR, ha="right", va="bottom")
    ax.text(29, -7.6, "fixed", fontsize=6, color=L.ERR, ha="right", va="top")
    ax.text(30.6, -11.6, f"epoch 30: {L.fmt(out['redrawn'])} vs {L.fmt(out['fixed'])} pp", fontsize=AS,
            color="black", ha="right", va="bottom")
    ax.set_xlim(-1, 31)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_ylim(-12, 3)
    ax.set_yticks([-10, -5, 0])
    ax.set_xlabel("Epoch", labelpad=1.5)
    ax.set_ylabel("Superclass probe, between-\nsuperclass − clean (pp)")
    title(ax, "Fixed labels keep the coarse\nstructure lower; redrawn\nlabels let it recover")
    ns.source_data(pd.DataFrame(rows), NAME, "a")


# b: between - within over training
def panel_b(axp, axa):
    P, A = L.static_probe(), L.static_accuracy()
    eff = L.fixed_label_effects(P, A, tag="ED8b")
    rows = []
    xp = P["clean"].columns.values.astype(float)
    Dp = (P["between_sc"] - P["within_sc"]).values
    for r in Dp:
        axp.plot(xp, r, color=SC, lw=0.5, alpha=0.4, zorder=2)
    m, lo, hi = L.t_ci(Dp[:, 1:])
    axp.fill_between(xp[1:], lo, hi, color=SC, alpha=0.2, lw=0, zorder=1)
    axp.plot(xp, Dp.mean(0), color=SC, ls=LS["between"], lw=1.1, marker="o", ms=3, zorder=4)
    axp.axhline(0, color=ZERO, lw=0.6, ls="--", zorder=0)
    lo_abs, hi_abs = np.abs(m).min(), np.abs(m).max()
    axp.text(30.5, -13.6, f"{L.MINUS}{lo_abs:.1f} to {L.MINUS}{hi_abs:.1f} pp at every checkpoint", fontsize=AS,
             ha="right", va="bottom")
    print("[b] probe between - within: " + ", ".join(f"{e}: {a:+.2f} [{b:+.2f}, {c:+.2f}]"
                                                     for e, a, b, c in zip(xp[1:].astype(int), m, lo, hi)))
    for j, e in enumerate(xp[1:].astype(int)):
        rows.append(dict(table="probe between - within per checkpoint (95% CI)", epoch=e, mean=m[j], ci_lo=lo[j],
                         ci_hi=hi[j], n_seeds_positive=int((Dp[:, j + 1] > 0).sum()),
                         **{f"seed_{s}": Dp[i, j + 1] for i, s in enumerate(L.SEEDS)}))
    xa = np.arange(31)
    Dd = A["between_sc"] - A["within_sc"]
    for r in Dd:
        axa.plot(xa, r, color=CL, lw=0.5, alpha=0.4, zorder=2)
    axa.plot(xa, Dd.mean(0), color=CL, ls=LS["between"], lw=1.0, zorder=3)
    bins = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30)]
    B = np.array([[r[a:b + 1].mean() for a, b in bins] for r in Dd])
    bm, blo, bhi = L.t_ci(B)
    xc = np.array([(a + b) / 2 for a, b in bins])
    axa.errorbar(xc, bm, yerr=[bm - blo, bhi - bm], fmt="o", ms=3.0, color=L.ERR, mfc=CL, mec=L.ERR, mew=0.6,
                 elinewidth=0.9, capsize=2, capthick=0.9, zorder=5)
    axa.axhline(0, color=ZERO, lw=0.6, ls="--", zorder=0)
    d = eff["delay_to_30pct_accuracy"]
    axa.text(30.5, -13.6, f"30% accuracy reached {d['mean']:.1f} epochs later", fontsize=AS, ha="right",
             va="bottom")
    pm, plo, phi = L.t_ci(Dd[:, 1:])
    for e in range(1, 31):
        rows.append(dict(table="accuracy between - within per epoch (95% CI)", epoch=e, mean=pm[e - 1],
                         ci_lo=plo[e - 1], ci_hi=phi[e - 1], ci_includes_zero=bool(plo[e - 1] <= 0 <= phi[e - 1]),
                         n_seeds_positive=int((Dd[:, e] > 0).sum()),
                         **{f"seed_{s}": Dd[i, e] for i, s in enumerate(L.SEEDS)}))
    for j, (a, b) in enumerate(bins):
        rows.append(dict(table="accuracy between - within, 5-epoch bin (95% CI)", epoch=f"{a}-{b}", mean=bm[j],
                         ci_lo=blo[j], ci_hi=bhi[j], n_seeds_positive=int((B[:, j] > 0).sum()),
                         **{f"seed_{s}": B[i, j] for i, s in enumerate(L.SEEDS)}))
    print("[b] accuracy 5-epoch bins: " + ", ".join(f"{a}-{b}: {v:+.2f} [{x:+.2f}, {y:+.2f}]"
                                                    for (a, b), v, x, y in zip(bins, bm, blo, bhi)))
    for key in ("probe_epoch30", "fine_acc_epochs0_10", "fine_acc_epoch30", "delay_to_30pct_accuracy"):
        r = eff[key]
        rows.append(dict(table=f"endpoint, between - within: {key} (95% CI; {r['unit']})", epoch=None,
                         mean=r["mean"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                         n_seeds_below_zero=r["n_seeds_below_zero"], **{f"seed_{s}": r[f"seed_{s}"] for s in L.SEEDS}))
    for ax, yl in ((axp, "Probe (pp)"), (axa, "Accuracy (pp)")):
        ax.set_xlim(-1, 31)
        ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
        ax.set_ylim(-14.2, 3.2)
        ax.set_yticks([-10, -5, 0])
        ax.set_ylabel(yl)
    axp.set_xticklabels([])
    axa.set_xlabel("Epoch", labelpad=1.5)
    title(axp, "Crossing superclasses costs probe and accuracy\nthroughout training (between − within, fixed labels)")
    ns.source_data(pd.DataFrame(rows), NAME, "b")
    return eff


# c: similarity-matched control
HNM = RESULTS / "NOISE_matched"
C_CONTRASTS = (("matched", "within", "matched −\nwithin"), ("between", "matched", "between −\nmatched"))
# (measure, row, offset within the contrast group, alpha, direct label)
C_MEASURES = (("probe_30", "probe", 0.0, ALPHA, None), ("acc_30", "accuracy", -0.21, ALPHA, "30"),
              ("acc_0_10", "accuracy", 0.21, 0.35, "0–10"))


def panel_c(axp, axa):
    """Similarity-matched control (fixed labels, p = 0.3): x = paired contrast; probe at epoch 30 (top), fine-class
    accuracy at epoch 30 (amber) and over epochs 0-10 (light amber) (bottom); same y-scale as b."""
    D = pd.read_csv(HNM / "noise_matched_per_seed.csv")
    J = json.load(open(HNM / "noise_matched_results.json"))
    rng = np.random.default_rng(9)
    rows = []
    for meas, row, dx, alpha, lab in C_MEASURES:
        ax = axp if row == "probe" else axa
        hue = SC if row == "probe" else CL
        w = 0.42 if row == "probe" else 0.36
        P = D.pivot(index="seed", columns="arm", values=meas).loc[L.SEEDS]
        for j, (a, b, _) in enumerate(C_CONTRASTS):
            d = (P[a] - P[b]).values
            m, lo, hi = L.t_ci(d)
            ref = J[f"{meas}: {a} - {b}"]
            assert np.allclose([m, lo, hi], ref["mean_ci"], atol=1e-9) and np.allclose(d, ref["per_seed"], atol=6e-3)
            x = j + dx
            ax.bar(x, m, width=w, color=hue, alpha=alpha, lw=0, zorder=2)
            ax.scatter(x + rng.uniform(-0.25, 0.25, len(d)) * w, d, s=4, color=L.DOT, lw=0, zorder=3)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=L.ERR, elinewidth=0.7, capsize=1.6,
                        capthick=0.7, zorder=4)
            if meas == "probe_30" and a == "matched":
                ax.text(x + w / 2 + 0.06, m, L.fmt(m, 1) + " pp", ha="left", va="center", fontsize=AS)
            s = L.paired_summary(d)
            rows.append(dict(measure=meas, contrast=f"{a} - {b}", mean=m, ci_lo=lo, ci_hi=hi, n=len(d),
                             n_seeds_below_zero=s["n_neg"], p_paired_t=s["p"],
                             **{f"seed_{sd}": v for sd, v in zip(L.SEEDS, d)}))
            print(f"[c] {meas} {a} - {b}: {m:+.2f} [{lo:+.2f}, {hi:+.2f}], {s['n_neg']}/5 below zero")
    axa.text(0.5, -13.6, "dark, epoch 30\nlight, epochs 0–10", ha="center", va="bottom", fontsize=6,
             color=L.ERR, linespacing=1.05)
    for ax in (axp, axa):
        ax.axhline(0, color=ZERO, lw=0.6, ls="--", zorder=0)
        ax.set_xlim(-0.55, 1.5)
        ax.set_ylim(-14.2, 3.2)
        ax.set_yticks([-10, -5, 0])
        ax.tick_params(labelleft=False)
        ax.set_xticks([0, 1])
    axp.set_xticklabels([])
    axa.set_xticklabels([c[2] for c in C_CONTRASTS], fontsize=6.5, linespacing=1.0)
    axa.tick_params(axis="x", length=0, pad=2)
    title(axp, "The probe cost remains\nat matched label similarity")
    arms = D.melt(id_vars=["seed", "arm"], var_name="measure", value_name="value").assign(table="per seed and arm")
    ns.source_data(pd.concat([pd.DataFrame(rows).assign(table="paired contrast (95% t-CI over seeds)"), arms],
                             ignore_index=True), NAME, "c")


# misrouted images (not placed in this figure)
def panel_d(ax):
    C = pd.read_csv(RESULTS / "SHARED_signal/T4_label_noise/contrasts_per_epoch.csv")
    g = C[(C.contrast == "between - within") & (C.measure == "misrouted_rate")].sort_values("epoch")
    m, lo, hi = 100 * g["mean"].values, 100 * g.ci_lo.values, 100 * g.ci_hi.values
    ax.fill_between(g.epoch, lo, hi, color=SC, alpha=0.2, lw=0, zorder=1)
    ax.plot(g.epoch, m, color=SC, ls=LS["between"], lw=1.1, marker="o", ms=3, zorder=3)
    ax.axhline(0, color=ZERO, lw=0.6, ls="--", zorder=0)
    ax.text(30.5, 0.6, f"+{m.min():.1f} to +{m.max():.1f} pp", fontsize=AS, ha="right", va="bottom")
    print("[d] misrouted between - within (pp): " + ", ".join(f"{e}: {v:+.2f} [{a:+.2f}, {b:+.2f}]"
                                                              for e, v, a, b in zip(g.epoch, m, lo, hi)))
    ax.set_xlim(3, 31)
    ax.set_xticks([5, 10, 15, 20, 25, 30])
    ax.set_ylim(-1, 11)
    ax.set_yticks([0, 5, 10])
    ax.set_xlabel("Epoch", labelpad=1.5)
    ax.set_ylabel("Images in the wrong superclass,\nbetween − within (pp)")
    title(ax, "Crossing superclasses sends more\nimages to the wrong superclass")
    ns.source_data(g.assign(mean_pp=m, ci_lo_pp=lo, ci_hi_pp=hi)[["epoch", "mean_pp", "ci_lo_pp", "ci_hi_pp",
                                                                   "n_pos", "n_neg", "per_seed"]], NAME, "d")


# d (panel_e): superclass read-outs, RDT and SAE-free
def panel_e(ax):
    import rdt_side_by_side as RS
    M = [m for m in RS.measures() if m[2] == SC or m[1] == "free"]   # drop the class-level RDT bars
    labels = ["superclass-\nselective\nfeatures", "superclass\nevents\n(Ab-E)", "superclass\nprobe",
              "fine-class\naccuracy"]
    xs = [0, 1, 2.4, 3.4]
    rng = np.random.default_rng(5)
    rows = []
    for x, (lab, kind, col, v), tl in zip(xs, M, labels):
        m, lo, hi = L.t_ci(v)
        ax.bar(x, m, width=0.6, color=col, alpha=ALPHA, lw=0, zorder=1)
        ax.scatter(x + rng.uniform(-0.13, 0.13, len(v)), v, s=6, color=L.DOT, lw=0, zorder=3)
        ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=L.ERR, elinewidth=0.9, capsize=2.2,
                    capthick=0.9, zorder=4)
        if x == 0:
            ax.text(x + 0.36, m, f"{L.fmt(m, 0)}%", fontsize=AS, ha="left", va="center")
        rows.append(dict(measure=lab.replace("\n", " "), readout="RDT" if kind == "rdt" else "SAE-free",
                         contrast="between - within (% change)", mean=m, ci_lo=lo, ci_hi=hi,
                         **{f"seed_{s}": z for s, z in zip(L.SEEDS, v)}))
        print(f"[e] {lab.replace(chr(10), ' ')} ({kind}): {m:+.1f}% [{lo:+.1f}, {hi:+.1f}]")
    ax.axhline(0, color=ZERO, ls="--", lw=0.6, zorder=0)
    ax.axvline(1.7, color="#D1D5DB", lw=0.6, zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=6, linespacing=1.0)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(-0.55, 3.9)
    ax.set_ylim(-50, 20)
    ax.set_yticks([-40, -20, 0, 20])
    ax.set_ylabel("Between vs within\n(% change)")
    ax.text(0.5, 19, "RDT", fontsize=6, color=L.ERR, ha="center", va="top")
    ax.text(2.9, 19, "without SAEs", fontsize=6, color=L.ERR, ha="center", va="top")
    title(ax, "SAE-based and SAE-free superclass\nread-outs agree in direction")
    ns.source_data(pd.DataFrame(rows), NAME, "e")


# e (panel_f): nesting ratio
def panel_f(ax):
    N = pd.read_csv(RESULTS / "RDT_noise" / "nesting_per_lane.csv").set_index(["arm", "seed"])
    rng = np.random.default_rng(11)
    arms = (("clean", "clean", CLEAN_C), ("within", "within", CL), ("between", "between", CL))
    rows, mean = [], {}
    for i, (a, lab, col) in enumerate(arms):
        v = N.loc[a].loc[list(L.SEEDS)].nesting_median_ratio.values
        m, lo, hi = L.t_ci(v)
        mean[a] = m
        if a == "between":
            ax.bar(i, m, width=0.6, facecolor="white", edgecolor=col, lw=1.0, ls=(0, (3, 1.6)), zorder=1)
        else:
            ax.bar(i, m, width=0.6, color=col, alpha=ALPHA, lw=0, zorder=1)
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=6, color=L.DOT, lw=0, zorder=3)
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=L.ERR, elinewidth=0.9, capsize=2.2, zorder=4)
        rows.append(dict(arm=a, mean=m, ci_lo=lo, ci_hi=hi, **{f"seed_{s}": x for s, x in zip(L.SEEDS, v)},
                         n_class_selective_mean=float(N.loc[a].loc[list(L.SEEDS)].n_cls_mean.mean())))
        print(f"[f] nesting {a}: {m:.2f} [{lo:.2f}, {hi:.2f}]")
    ax.text(2.0, 5.6, f"{mean['within']:.1f} vs {mean['between']:.1f}", fontsize=AS, ha="center", va="top")
    ax.axhline(1, color=ZERO, ls="--", lw=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels([a[1] for a in arms], fontsize=6.5)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_ylim(0, 5.8)
    ax.set_yticks([0, 2, 4])
    ax.set_ylabel("Nesting ratio")
    title(ax, "Class features lie less within\ntheir superclass directions")
    ns.source_data(pd.DataFrame(rows), NAME, "f")


# f (panel_g): Di-E fraction per transition, labels redrawn
def panel_g(ax):
    T = L.die_trajectories()
    rows, first = [], {}
    for c, n in (("standard", "clean"), ("between_sc_p03", "between"), ("random_p03", "random")):
        P = T[T.condition == c].pivot(index="seed", columns="t_index", values="di_frac").loc[L.SEEDS]
        x = P.columns.values.astype(float)
        col = CLEAN_C if n == "clean" else CL
        m, h = L.traj(ax, x, P.values, col, LS[n], band="ci", lw=1.1 if n != "clean" else 1.0, ms=2.6,
                      band_alpha=0.12, mfc="white")
        first[n] = m[0]
        for e, mm_, hh in zip(x, m, h):
            rows.append(dict(condition=c, transition=f"{int(e)}->{int(e) + 1}", epochs=f"{5 * int(e)}-{5 * int(e) + 5}",
                             di_frac_mean=mm_, ci_lo=mm_ - hh, ci_hi=mm_ + hh, n=P.shape[0]))
    ax.text(0.15, 3, f"epochs 0–5: {first['between']:.0f}% vs {first['clean']:.0f}%", fontsize=AS, ha="left",
            va="bottom")
    ax.set_xticks(range(6))
    ax.set_xticklabels([f"{5 * i}–{5 * i + 5}" for i in range(6)], fontsize=6.5)
    ax.set_xlim(-0.3, 5.3)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("Transition (epochs)", labelpad=1.5)
    ax.set_ylabel("Di-E fraction (%)")
    title(ax, "Redrawn labels delay differentiation, which catches up (p = 0.3)")
    print("[g] Di-E fraction:", "; ".join(f"{r['condition']} {r['transition']} {r['di_frac_mean']:.1f}" for r in rows))
    ns.source_data(pd.DataFrame(rows), NAME, "g")


def legend_strip(fig):
    dk = L.ERR
    h1 = [Line2D([], [], color=CLEAN_C, ls="-", lw=CLEAN_LW, label="clean training"),
          Line2D([], [], color=dk, ls=LS["between"], lw=1.1, label="between-superclass noise (and between − reference)"),
          Line2D([], [], color=dk, ls=LS["random"], lw=1.1, label="random noise"),
          Line2D([], [], color=dk, ls="none", marker="o", ms=3.6, mfc="white", mec=dk, label="labels redrawn every epoch"),
          Line2D([], [], color=dk, ls="none", marker="o", ms=3.6, mfc=dk, mec=dk, label="labels kept fixed"),
          Patch(color=SC, alpha=ALPHA, lw=0, label="superclass (probe, assignment, features, Ab-E)"),
          Patch(color=CL, alpha=ALPHA, lw=0, label="class (accuracy, class features, Di-E)")]
    fig.legend(handles=h1[:3] + h1[5:6], loc="upper center", bbox_to_anchor=(0.5, 1 - 0.3 / H), ncol=4,
               fontsize=6.5, handlelength=2.6, columnspacing=1.6, frameon=False)
    fig.legend(handles=h1[3:5] + h1[6:], loc="upper center", bbox_to_anchor=(0.5, 1 - 4.0 / H), ncol=3,
               fontsize=6.5, handlelength=2.6, columnspacing=1.6, frameon=False)


def main():
    """Draw panels a-f; panel_e, panel_f and panel_g fill figure panels d, e and f, and their Source Data are
    re-lettered to match."""
    ns.apply()
    _sd = ns.source_data
    REL = {"e": "d", "f": "e", "g": "f"}
    ns.source_data = lambda df, name, letter: _sd(df, name, REL.get(letter, letter) if name == NAME else letter)
    for f in (ns.FINAL_OUT / "source_data").glob(f"{NAME}_*.csv"):
        f.unlink()
    fig = plt.figure(figsize=(W * ns.MM, H * ns.MM))
    y1, h1 = 108, 34
    hh = (h1 - 3) / 2
    ax_a = mm_axes(fig, 16, y1, 34, h1)
    ax_bp = mm_axes(fig, 70, y1 + hh + 3, 66, hh)
    ax_ba = mm_axes(fig, 70, y1, 66, hh)
    ax_cp = mm_axes(fig, 140, y1 + hh + 3, 38, hh)
    ax_ca = mm_axes(fig, 140, y1, 38, hh)
    # row 2: d | e
    y2, h2 = 58, 28
    ax_d = mm_axes(fig, 16, y2, 76, h2)
    ax_e = mm_axes(fig, 124, y2, 44, h2)
    # row 3: f
    y3, h3 = 10, 26
    ax_f = mm_axes(fig, 16, y3, 96, h3)
    panel_a(ax_a)
    panel_b(ax_bp, ax_ba)
    panel_c(ax_cp, ax_ca)
    panel_e(ax_d)
    panel_f(ax_e)
    panel_g(ax_f)
    legend_strip(fig)
    t1, t2, t3 = y1 + h1 + 14, y2 + h2 + 11, y3 + h3 + 11
    for ax, letter, xmm, ymm in ((ax_a, "a", 1, t1), (ax_bp, "b", 56, t1), (ax_cp, "c", 137, t1),
                                 (ax_d, "d", 1, t2), (ax_e, "e", 108, t2), (ax_f, "f", 1, t3)):
        b = ax.get_position()
        ns.panel(ax, letter, x=(xmm / W - b.x0) / b.width, y=(ymm / H - b.y0) / b.height)
    ns.source_data = _sd
    print("saved", ns.save_final(fig, NAME, 180, ed=True))


if __name__ == "__main__":
    main()
