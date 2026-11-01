#!/usr/bin/env python3
"""Extended Data Fig. 8: label noise across vs within superclasses.

Panels a and b are drawn by routing/noise_routing_panels.py (loaded as a module; its colours, line styles and labels are overridden
here); c-f follow the panels of noise_routing_panels.py with the colours and names below; g and h are local.

  a  superclass probe, noise across superclasses minus clean, labels redrawn every epoch (open) or fixed (filled)
  b  across minus within, fixed labels: probe (top) and fine-class accuracy (bottom); 5-epoch bin means in Source Data
  c  noise from equally similar classes of other superclasses (matched similarity)
  d  across minus within, % change, RDT and SAE-free read-outs
  e  nesting ratio: across darkest amber, within light amber, clean grey
  f  class-differentiation events (Di-E fraction), labels redrawn every epoch
  g  epochs behind the reference at epoch 10, superclass accuracy (network's own classifier; blue end) and fine-class
     accuracy (amber end); grey connectors
  h  share of class-selective features bent to the wrong labels (index below the 5th percentile of clean training at
     that epoch; RDT_noise_fit/features.csv), across, within and clean

Colour = level measured (superclass blue, class amber, clean grey); within a level, noise across superclasses darkest
solid, noise within superclasses mid dashed, random noise light dotted, clean thin grey solid.

Output: analysis_outputs/paper/figures/interventions/ (ed8.pdf/png/tif, source_data/ed8_a-h.csv), or $ES_FIG_OUT.

Usage: [ES_FIG_OUT=<dir>] python scripts/paper/figures/interventions/ed8.py
"""
import importlib.util as _ilu
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))     # scripts/paper/figures
sys.path.insert(0, str(HERE.parents[1]))  # scripts/paper
import nmi_style as ns  # noqa: E402

if not os.environ.get("ES_FIG_OUT"):
    ns.FINAL_OUT = ns.OUT / "interventions"

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

_spec = _ilu.spec_from_file_location("noise_routing_panels", HERE.parent / "routing" / "noise_routing_panels.py")
E = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(E)
L = E.L

NAME = "ed8"
E.NAME = NAME
W, H = E.W, E.H
RESULTS = ns.OUT.parent
SEEDS = [7, 42, 137, 256, 314]
N7 = [0, 5, 10, 15, 20, 25, 30]
BLUE = getattr(ns, "BLUE", ["#1E40AF", "#3B82F6", "#60A5FA", "#93C5FD"])
AMBER = getattr(ns, "AMBER", ["#92400E", "#D97706", "#F59E0B", "#FCD34D"])
GREYS = getattr(ns, "GREYS", ["#374151", "#6B7280", "#9CA3AF", "#E5E7EB"])
BLUE_T, AMBER_T, GREY_T = "#1E40AF", "#92400E", "#6B7280"
DARK = GREYS[0]
ALPHA = 0.85
SSI0 = E.SC                      # original superclass hue, used only to filter rdt_side_by_side.measures()
SC, CL = BLUE[0], AMBER[0]
# overrides for the noise_routing_panels.py panels drawn as they are (a, b)
E.SC, E.CL, E.CLEAN_C = SC, CL, GREYS[2]
E.LS = {"clean": "-", "within": "--", "between": "-", "random": ":"}
TS, AS = E.TS, E.AS


def title(ax, txt):
    ax.set_title(txt, fontsize=TS, pad=3, linespacing=1.05, loc="left")


def behind(a, c, eps):
    """Epochs behind: epoch minus the epoch at which the reference's running maximum reaches a."""
    c = np.maximum.accumulate(np.asarray(c, float))
    return [e - float(np.interp(v, c, eps)) if v <= c[-1] else np.nan for e, v in zip(eps, a)]


def noise_lags(Pl, arm, ctrl, epoch=10):
    """Per seed (superclass-accuracy lag, fine-class-accuracy lag) of `arm` behind `ctrl` at `epoch`."""
    out = []
    for s in SEEDS:
        a = Pl[(Pl.family == "noise") & (Pl.arm == arm) & (Pl.seed == s)].set_index("epoch").loc[N7]
        b = Pl[(Pl.family == "noise") & (Pl.arm == ctrl) & (Pl.seed == s)].set_index("epoch").loc[N7]
        out.append([behind(a[m].values, b[m].values, N7)[N7.index(epoch)] for m in ("placement", "fine")])
    return np.array(out)


# a, b: noise_routing_panels.py panels, recoloured and relabelled
def panel_a(ax):
    E.panel_a(ax)
    ax.set_ylabel("Superclass probe, noise across\nsuperclasses − clean (pp)")


def panel_b(axp, axa):
    eff = E.panel_b(axp, axa)
    for c in list(axa.containers):          # 5-epoch bin means with CIs: dropped from the artwork (Source Data keeps them)
        for art in c.get_children():
            art.remove()
    title(axp, "Crossing superclasses costs probe and accuracy\nthroughout training (across − within, fixed labels)")
    return eff


# c: similarity-matched control
C_CONTRASTS = (("matched", "within", "matched\n− within"), ("between", "matched", "across\n− matched"))
C_MEASURES = (("probe_30", "probe", 0.0, BLUE[0], None), ("acc_30", "accuracy", -0.21, AMBER[0], "30"),
              ("acc_0_10", "accuracy", 0.21, AMBER[3], "0–10"))


def panel_c(axp, axa):
    D = pd.read_csv(E.HNM / "noise_matched_per_seed.csv")
    J = json.load(open(E.HNM / "noise_matched_results.json"))
    rng = np.random.default_rng(9)
    rows = []
    for meas, row, dx, hue, lab in C_MEASURES:
        ax = axp if row == "probe" else axa
        w = 0.42 if row == "probe" else 0.36
        P = D.pivot(index="seed", columns="arm", values=meas).loc[L.SEEDS]
        for j, (a, b, _) in enumerate(C_CONTRASTS):
            d = (P[a] - P[b]).values
            m, lo, hi = L.t_ci(d)
            ref = J[f"{meas}: {a} - {b}"]
            assert np.allclose([m, lo, hi], ref["mean_ci"], atol=1e-9) and np.allclose(d, ref["per_seed"], atol=6e-3)
            x = j + dx
            ax.bar(x, m, width=w, color=hue, alpha=ALPHA, lw=0, zorder=2)
            ax.scatter(x + rng.uniform(-0.25, 0.25, len(d)) * w, d, s=4, color=L.DOT, lw=0, zorder=3)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=L.ERR, elinewidth=0.7, capsize=1.6,
                        capthick=0.7, zorder=4)
            if meas == "probe_30" and a == "matched":
                ax.text(x + w / 2 + 0.06, m, L.fmt(m, 1) + " pp", ha="left", va="center", fontsize=AS)
            s = L.paired_summary(d)
            name = {"between": "across", "matched": "across at matched similarity", "within": "within"}
            rows.append(dict(measure=meas, contrast=f"{name[a]} - {name[b]}", mean=m, ci_lo=lo, ci_hi=hi, n=len(d),
                             n_seeds_below_zero=s["n_neg"], p_paired_t=s["p"],
                             **{f"seed_{sd}": v for sd, v in zip(L.SEEDS, d)}))
            print(f"[c] {meas} {a} - {b}: {m:+.2f} [{lo:+.2f}, {hi:+.2f}], {s['n_neg']}/5 below zero")
    axa.text(0.5, -13.6, "dark, epoch 30\nlight, epochs 0–10", ha="center", va="bottom", fontsize=6,
             color=L.ERR, linespacing=1.05)
    for ax in (axp, axa):
        ax.axhline(0, color=E.ZERO, lw=0.6, ls="--", zorder=0)
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


# d: superclass read-outs, RDT and SAE-free
def panel_d(ax):
    import rdt_side_by_side as RS
    M = [m for m in RS.measures() if m[2] == SSI0 or m[1] == "free"]
    labels = ["superclass-\nselective\nfeatures", "superclass\nevents\n(Ab-E)", "superclass\nprobe",
              "fine-class\naccuracy"]
    xs = [0, 1, 2.4, 3.4]
    rng = np.random.default_rng(5)
    rows = []
    for x, (lab, kind, col, v), tl in zip(xs, M, labels):
        hue = SC if col == SSI0 else CL
        m, lo, hi = L.t_ci(v)
        ax.bar(x, m, width=0.6, color=hue, alpha=ALPHA, lw=0, zorder=1)
        ax.scatter(x + rng.uniform(-0.13, 0.13, len(v)), v, s=6, color=L.DOT, lw=0, zorder=3)
        ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=L.ERR, elinewidth=0.9, capsize=2.2,
                    capthick=0.9, zorder=4)
        if x == 0:
            ax.text(x + 0.36, m, f"{L.fmt(m, 0)}%", fontsize=AS, ha="left", va="center")
        rows.append(dict(measure=lab.replace("\n", " "), readout="RDT" if kind == "rdt" else "SAE-free",
                         contrast="across - within (% change)", mean=m, ci_lo=lo, ci_hi=hi,
                         **{f"seed_{s}": z for s, z in zip(L.SEEDS, v)}))
        print(f"[d] {lab.replace(chr(10), ' ')} ({kind}): {m:+.1f}% [{lo:+.1f}, {hi:+.1f}]")
    ax.axhline(0, color=E.ZERO, ls="--", lw=0.6, zorder=0)
    ax.axvline(1.7, color="#D1D5DB", lw=0.6, zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=6, linespacing=1.0)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(-0.55, 3.9)
    ax.set_ylim(-50, 20)
    ax.set_yticks([-40, -20, 0, 20])
    ax.set_ylabel("Across vs within\n(% change)")
    ax.text(0.5, 19, "RDT", fontsize=6, color=L.ERR, ha="center", va="top")
    ax.text(2.9, 19, "without SAEs", fontsize=6, color=L.ERR, ha="center", va="top")
    title(ax, "SAE-based and SAE-free superclass\nread-outs agree in direction")
    ns.source_data(pd.DataFrame(rows), NAME, "d")


# e: nesting ratio
def panel_e(ax):
    N = pd.read_csv(RESULTS / "RDT_noise" / "nesting_per_lane.csv").set_index(["arm", "seed"])
    rng = np.random.default_rng(11)
    arms = (("clean", "clean", GREYS[2]), ("within", "within", AMBER[2]), ("between", "across", AMBER[0]))
    rows, mean = [], {}
    for i, (a, lab, col) in enumerate(arms):
        v = N.loc[a].loc[list(L.SEEDS)].nesting_median_ratio.values
        m, lo, hi = L.t_ci(v)
        mean[a] = m
        ax.bar(i, m, width=0.6, color=col, alpha=ALPHA, lw=0, zorder=1)
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=6, color=L.DOT, lw=0, zorder=3)
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=L.ERR, elinewidth=0.9, capsize=2.2, zorder=4)
        rows.append(dict(arm=a, arm_label=lab, mean=m, ci_lo=lo, ci_hi=hi, **{f"seed_{s}": x for s, x in zip(L.SEEDS, v)},
                         n_class_selective_mean=float(N.loc[a].loc[list(L.SEEDS)].n_cls_mean.mean())))
        print(f"[e] nesting {a}: {m:.2f} [{lo:.2f}, {hi:.2f}]")
    ax.text(1.5, 5.6, f"{mean['within']:.1f} vs {mean['between']:.1f}", fontsize=AS, ha="center", va="top")
    ax.axhline(1, color=E.ZERO, ls="--", lw=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels([a[1] for a in arms], fontsize=6.5)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_ylim(0, 5.8)
    ax.set_yticks([0, 2, 4])
    ax.set_ylabel("Nesting ratio")
    title(ax, "Class features lie less within\ntheir superclass directions")
    ns.source_data(pd.DataFrame(rows), NAME, "e")


# f: Di-E fraction per transition, labels redrawn
def panel_f(ax):
    T = L.die_trajectories()
    rows, first = [], {}
    for c, n, col, ls, lw in (("standard", "clean", GREYS[2], "-", 0.9), ("between_sc_p03", "across", AMBER[0], "-", 1.2),
                              ("random_p03", "random", AMBER[2], ":", 1.2)):
        P = T[T.condition == c].pivot(index="seed", columns="t_index", values="di_frac").loc[L.SEEDS]
        x = P.columns.values.astype(float)
        m, h = L.traj(ax, x, P.values, col, ls, band="ci", lw=lw, ms=2.6, band_alpha=0.12, mfc="white")
        first[n] = m[0]
        for e, mm_, hh in zip(x, m, h):
            rows.append(dict(condition=c, condition_label=n, transition=f"{int(e)}->{int(e) + 1}",
                             epochs=f"{5 * int(e)}-{5 * int(e) + 5}", di_frac_mean=mm_, ci_lo=mm_ - hh, ci_hi=mm_ + hh,
                             n=P.shape[0]))
    ax.text(0.15, 3, f"epochs 0–5: {first['across']:.0f}% vs {first['clean']:.0f}%", fontsize=AS, ha="left",
            va="bottom")
    ax.set_xticks(range(6))
    ax.set_xticklabels([f"{5 * i}–{5 * i + 5}" for i in range(6)], fontsize=6.3)
    ax.set_xlim(-0.3, 5.3)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("Transition (epochs)", labelpad=1.5)
    ax.set_ylabel("Di-E fraction (%)")
    title(ax, "Redrawn labels delay differentiation,\nwhich catches up (p = 0.3)")
    ns.source_data(pd.DataFrame(rows), NAME, "f")


# g: which level falls further behind
def panel_g(ax):
    Pl = pd.read_csv(RESULTS / "PLACEMENT_curve" / "long.csv")
    nw, cl = "vs within", "vs clean"
    items = [("noise across\nsuperclasses", noise_lags(Pl, "between", "within"), nw, "between", "within"),
             ("noise across,\nmatched\nsimilarity", noise_lags(Pl, "matched", "within"), nw, "matched", "within"),
             ("noise within\nsuperclasses", noise_lags(Pl, "within", "clean"), cl, "within", "clean")]
    SCD, CLD = BLUE[1], AMBER[1]
    gap, step = 0.5, 1.2
    tr = ax.get_xaxis_transform()
    rows, xs = [], []
    for k, (lab, XY, ref, arm, ctrl) in enumerate(items):
        x0 = k * step
        xs.append(x0)
        for sx, fy in XY:
            ax.plot([x0, x0 + gap], [sx, fy], color=GREYS[2], lw=0.5, alpha=0.8, zorder=2)
        m = XY.mean(0)
        ax.plot([x0, x0 + gap], m, color=GREYS[1], lw=1.6, zorder=3, solid_capstyle="round")
        ax.scatter([x0], [m[0]], s=16, color=SCD, edgecolor="white", lw=0.5, zorder=4)
        ax.scatter([x0 + gap], [m[1]], s=16, color=CLD, edgecolor="white", lw=0.5, zorder=4)
        ax.text(x0 + gap / 2, -0.03, lab, fontsize=6, ha="center", va="top", color=DARK, linespacing=0.95,
                transform=tr)
        ax.text(x0 + gap / 2, -0.36, ref, fontsize=6, ha="center", va="top", color=GREY_T, transform=tr)
        name = lab.replace("\n", " ")
        base = dict(condition=name, noise_arm=arm, reference={"within": "noise within superclasses", "clean": "clean training"}[ctrl], reference_arm=ctrl,
                    coarse_measure="superclass accuracy (network's own classifier)",
                    fine_measure="fine-class accuracy", epoch=10)
        rows += [dict(base, seed=s, coarse_lag_epochs=a, fine_lag_epochs=b) for s, (a, b) in zip(SEEDS, XY)]
        rows.append(dict(base, seed="mean", coarse_lag_epochs=m[0], fine_lag_epochs=m[1]))
        print(f"[g] {name} ({ref}): superclass " + ", ".join(f"{v:.3f}" for v in XY[:, 0]) + f" (mean {m[0]:.3f}); "
              "fine " + ", ".join(f"{v:.3f}" for v in XY[:, 1]) + f" (mean {m[1]:.3f})")
    xm = (xs[1] + gap + xs[2]) / 2
    ax.axvline(xm, color="#D1D5DB", lw=0.6, zorder=0)
    ax.text((xs[0] + xs[1] + gap) / 2, 1.03, "coarse structure\ndamaged", fontsize=6, ha="center", va="bottom",
            color=GREY_T, linespacing=0.95, transform=tr)
    ax.text(xs[2] + gap / 2, 1.03, "coarse structure\nspared", fontsize=6, ha="center", va="bottom",
            color=GREY_T, linespacing=0.95, transform=tr)
    h = [Line2D([], [], marker="o", ls="none", ms=4, color=SCD, label="superclass accuracy"),
         Line2D([], [], marker="o", ls="none", ms=4, color=CLD, label="fine-class accuracy")]
    ax.legend(handles=h, fontsize=6, frameon=False, loc="lower left", bbox_to_anchor=(0.0, -0.02), handletextpad=0.1,
              borderaxespad=0.1, labelspacing=0.3, handlelength=1.0)
    ax.axhline(0, color="#D1D5DB", lw=0.5, zorder=0)
    ax.set_xlim(-0.25, xs[-1] + gap + 0.25)
    ax.set_ylim(-2.9, 6)
    ax.set_yticks([-2, 0, 2, 4, 6])
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("Epochs behind reference,\nat epoch 10")
    ax.set_title("The damaged level\nfalls further behind", fontsize=TS, pad=17, linespacing=1.05)
    ns.source_data(pd.DataFrame(rows), NAME, "g")


# h: class features bent to the wrong labels
def panel_h(ax):
    F_ = pd.read_csv(RESULTS / "RDT_noise_fit" / "features.csv")
    lo = F_[F_.arm == "clean"].groupby("epoch").nfi.quantile(0.05)
    F_["bent"] = F_.nfi < F_.epoch.map(lo)
    ep = [5, 10, 15, 20, 25, 30]
    per = F_.groupby(["arm", "seed", "epoch"]).agg(class_selective=("feature", "size"), bent=("bent", "sum")).reset_index()
    per["bent_pct"] = 100 * per.bent / per.class_selective
    rows = []
    sty = {"between": ("across", AMBER[0], "-", 1.5, AMBER_T), "within": ("within", AMBER[1], "--", 1.3, "#B45309"),
           "clean": ("clean", GREYS[2], "-", 0.9, GREY_T)}
    for arm in ("clean", "within", "between"):
        lab, c, ls, lw, tc = sty[arm]
        Y = per[per.arm == arm].pivot(index="seed", columns="epoch", values="bent_pct").loc[SEEDS, ep].values
        for y in Y:
            ax.plot(ep, y, color=c, lw=0.45, alpha=0.45, zorder=2)
        ax.plot(ep, Y.mean(0), color=c, lw=lw, ls=ls, zorder=3)
        lo_, hi_ = Y[:, -1].min(), Y[:, -1].max()
        ax.text(31.2, Y[:, -1].mean(), f"{lab}\n{lo_:.0f}–{hi_:.0f}%", fontsize=6, color=tc, ha="left", va="center",
                linespacing=0.95, clip_on=False)
        rows += [dict(arm=arm, arm_label={"between": "noise across superclasses", "within": "noise within superclasses",
                                          "clean": "clean training"}[arm], seed=r.seed, epoch=r.epoch,
                      class_selective=r.class_selective, bent=r.bent, bent_pct=r.bent_pct)
                 for r in per[per.arm == arm].itertuples()]
        print(f"[h] {arm}: epoch 30 bent {Y[:, -1].mean():.1f}% ({lo_:.1f}-{hi_:.1f})")
    ax.set_xlim(4, 31); ax.set_xticks([5, 10, 20, 30]); ax.set_ylim(0, 100); ax.set_yticks([0, 50, 100])
    ax.set_xlabel("Epoch", labelpad=1.5); ax.set_ylabel("Class-selective features\nbent to wrong labels (%)")
    title(ax, "Noise across superclasses\nmakes class features fit\nthe wrong labels")
    ns.source_data(pd.DataFrame(rows), NAME, "h")


def legend_strip(fig):
    dk = DARK
    h = [Line2D([], [], color=GREYS[2], ls="-", lw=0.9, label="clean training"),
         Line2D([], [], color=dk, ls="-", lw=1.2, label="noise across superclasses (and across − reference)"),
         Line2D([], [], color=dk, ls="--", lw=1.1, label="noise within superclasses"),
         Line2D([], [], color=dk, ls=":", lw=1.2, label="random noise"),
         Line2D([], [], color=dk, ls="none", marker="o", ms=3.6, mfc="white", mec=dk, label="labels redrawn every epoch"),
         Line2D([], [], color=dk, ls="none", marker="o", ms=3.6, mfc=dk, mec=dk, label="labels kept fixed"),
         Patch(color=SC, alpha=ALPHA, lw=0, label="superclass (probe, accuracy, features, Ab-E)"),
         Patch(color=CL, alpha=ALPHA, lw=0, label="class (accuracy, features, Di-E)")]
    fig.legend(handles=h[:4], loc="upper center", bbox_to_anchor=(0.5, 1 - 0.3 / H), ncol=4,
               fontsize=6.5, handlelength=2.6, columnspacing=1.4, frameon=False)
    fig.legend(handles=h[4:], loc="upper center", bbox_to_anchor=(0.5, 1 - 4.0 / H), ncol=4,
               fontsize=6.5, handlelength=1.6, columnspacing=1.2, handletextpad=0.5, frameon=False)


def main():
    ns.apply()
    for f in (ns.FINAL_OUT / "source_data").glob(f"{NAME}_*.csv"):
        f.unlink()
    fig = plt.figure(figsize=(W * ns.MM, H * ns.MM))
    y1, h1 = 108, 34
    hh = (h1 - 3) / 2
    ax_a = E.mm_axes(fig, 16, y1, 34, h1)
    ax_bp = E.mm_axes(fig, 70, y1 + hh + 3, 66, hh)
    ax_ba = E.mm_axes(fig, 70, y1, 66, hh)
    ax_cp = E.mm_axes(fig, 140, y1 + hh + 3, 38, hh)
    ax_ca = E.mm_axes(fig, 140, y1, 38, hh)
    y2, h2 = 58, 28
    ax_d = E.mm_axes(fig, 16, y2, 76, h2)
    ax_e = E.mm_axes(fig, 124, y2, 44, h2)
    y3, h3 = 10, 26
    ax_f = E.mm_axes(fig, 14, y3, 40, h3)
    ax_g = E.mm_axes(fig, 73, y3 + 3, 48, h3 - 4)
    ax_h = E.mm_axes(fig, 141, y3, 26, h3)
    E.ns.source_data = ns.source_data
    panel_a(ax_a)
    panel_b(ax_bp, ax_ba)
    panel_c(ax_cp, ax_ca)
    panel_d(ax_d)
    panel_e(ax_e)
    panel_f(ax_f)
    panel_g(ax_g)
    panel_h(ax_h)
    legend_strip(fig)
    t1, t2, t3 = y1 + h1 + 14, y2 + h2 + 11, y3 + h3 + 14
    for ax, letter, xmm, ymm in ((ax_a, "a", 1, t1), (ax_bp, "b", 56, t1), (ax_cp, "c", 137, t1),
                                 (ax_d, "d", 1, t2), (ax_e, "e", 108, t2), (ax_f, "f", 1, t3),
                                 (ax_g, "g", 57, t3), (ax_h, "h", 125, t3)):
        b = ax.get_position()
        ns.panel(ax, letter, x=(xmm / W - b.x0) / b.width, y=(ymm / H - b.y0) / b.height)
    print("saved", ns.save_final(fig, NAME, 180, ed=True))


if __name__ == "__main__":
    main()
