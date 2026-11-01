#!/usr/bin/env python3
"""Curriculum sweep and controlled-curriculum loaders and panels. The sweep table, conditions and lane paths are
used through curriculum_lib.py by Fig. 6 (interventions/fig6_panels.py) and Extended Data Fig. 9
(interventions/ed9.py). Run directly, the script draws an overview of the curriculum data:

a  validation accuracy of the 7-condition curriculum sweep (experiment 7885dc97, standard and switch e05-e30):
   solid = fine-class phase, dashed = 20-class superclass phase, dotted = switch transition; grey 57-59% band.
b  train (solid) and validation (dashed) accuracy of the true-superclass arm (sc_true_K20, mean +- SEM) against
   epoch relative to the start of fine-class training; coarse phase shaded, gaps annotated from the data.
c  curriculum: fine-class epochs to first reach 20/30/40/50% validation accuracy (true / random: epochs 21-50;
   scratch: epochs 1-30); bars = mean, dots = seeds, SEM.
d  100-way linear probe on frozen avgpool activations over fine-class epochs (CURRICULUM_probe/probe.csv from
   scripts/paper/curriculum_probe.py), with the head start at fine epoch 0.
e  Di-E per 1,000 iterations, standard vs switch e20 (sweep, thresholds of standard-50), dotted = same-state null.
Also writes curriculum_probe_true_random, curriculum_gap_arms, curriculum_di_share and curriculum_gap_sweep. Only seeds whose
three curriculum lanes carry .lane_done are used.

Usage: python scripts/paper/figures/curriculum_sweep_panels.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nmi_style as ns  # noqa: E402
from fig_common import (REPO, RESULTS, SEEDS, RUNS_EXTENDED, DOT, ERR, REF, WINDOW, mean_sem, traj,  # noqa: E402
                        lane_done, val_curve)

CUR = RUNS_EXTENDED / "curriculum" / "lanes"
SWEEP = REPO / "data" / "experiments" / "7885dc97-1062-4b2f-9999-61d63b93d097" / "lanes"
K_CUR = 20  # superclass epochs of the controlled experiment
ARM = {"sc_true": ("sc_true_K20_seed{}", "True superclasses", "-", "o"),
       "sc_rand": ("sc_rand_K20_seed{}", "Random grouping", "--", "s"),
       "scratch": ("scratch_E30_seed{}", "Scratch", ":", "^")}
ARM_SHORT = {"sc_true": "True", "sc_rand": "Random", "scratch": "Scratch"}
# Sweep-condition palette and markers
COND = ["standard50", "switch_e05", "switch_e10", "switch_e15", "switch_e20", "switch_e25", "switch_e30"]
CCOL = {"standard50": "#3B82F6", "switch_e05": "#6366F1", "switch_e10": "#8B5CF6", "switch_e15": "#A855F7",
        "switch_e20": "#D946EF", "switch_e25": "#F59E0B", "switch_e30": "#DC2626", "standard30": "#3B82F6"}
CMK = {"standard50": "o", "switch_e05": "s", "switch_e10": "D", "switch_e15": "P", "switch_e20": "X",
       "switch_e25": "^", "switch_e30": "v", "standard30": "o"}
CLAB = {"standard50": "Std", "switch_e05": "e05", "switch_e10": "e10", "switch_e15": "e15",
        "switch_e20": "e20", "switch_e25": "e25", "switch_e30": "e30", "standard30": "Std\n30 ep"}
CLEG = {"standard50": "Standard (0 SC)", **{c: f"Switch {CLAB[c]}" for c in COND[1:]}}
KNOM = {c: (0 if c == "standard50" else int(c[-2:])) for c in COND}
ANNOT = dict(fontweight="normal")
CSI = ns.INDEX["CSI"]  # amber: class-level quantities
REF_DARK = ns.REF_DARK
DIE = ns.PROC["Di-E"]  # amber: differentiation events
LIGHT = "#9CA3AF"      # third / reference condition in neutral (accuracy / probe) panels
# Colours mark processes, line styles mark conditions. Panels without an index or process use neutral tones: true
# superclasses solid (dark), random grouping dashed (grey), scratch / standard dotted (light grey). Di-E is amber.
RULE_ARM = {"sc_true": (REF_DARK, "-"), "sc_rand": (ns.NEUTRAL, "--"), "scratch": (LIGHT, ":")}
RULE_E = {"switch_e20": (DIE, "-"), "standard50": (DIE, ":"), "null": (DIE, ":", 0.8, 0.6)}


def sweep_table():
    tc = pd.read_csv(RESULTS / "CURRICULUM_sweep" / "tables" / "A_training_cost.csv")
    return tc, tc[tc.condition.isin(COND)].set_index("condition")


def lane(arm, s):
    return CUR / ARM[arm][0].format(s)


def complete_seeds():
    return [s for s in SEEDS if all(lane_done(lane(a, s)) for a in ARM)]


def fine_curve(arm, s):
    c = val_curve(lane(arm, s))
    off = 0 if arm == "scratch" else K_CUR
    return np.array([c[off + e] for e in range(1, 31)])


def gap(ln, epoch=None):
    tr, va = val_curve(ln, "train_accuracy"), val_curve(ln)
    e = max(va) if epoch is None else epoch
    return tr[e] - va[e]


# a
def panel_a(ax):
    _, sw = sweep_table()
    for c in COND:
        v = val_curve(SWEEP / sw.loc[c, "lane_id"])
        k = KNOM[c]
        col = CCOL[c]
        if k:
            sc = [e for e in [1] + list(range(5, k + 1, 5)) if e in v]
            ax.plot(sc, [v[e] for e in sc], color=col, ls="--", lw=0.9, alpha=0.6, zorder=1)
            ax.plot([k, k + 5], [v[k], v[k + 5]], color=col, ls=":", lw=0.8, alpha=0.6, zorder=1)
            fine = list(range(k + 5, 51, 5))
        else:
            fine = [1] + list(range(5, 51, 5))
        ax.plot(fine, [v[e] for e in fine], color=col, marker=CMK[c], ms=3.6, lw=1.1, label=CLEG[c], zorder=3)
        print(f"[a] {c}: SC epochs (nominal) {k}, first logged epoch {min(v)}, "
              f"val@switch {v[k] if k else float('nan'):.2f}, final val {v[50]:.2f}")
    ax.axhspan(57, 59, color=WINDOW, ec="#D1D5DB", lw=0.5, zorder=0)
    ax.text(3, 58, "57–59%", fontsize=6.5, va="center", ha="left", color=ERR)
    ax.text(5, 49, "dashed = 20-class\n(SC phase)", fontsize=6, color=REF, ha="left", va="center")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_xlim(0, 52)
    ax.set_ylim(10, 67)
    ax.set_xticks(range(0, 51, 10))
    leg = ax.legend(loc="lower right", bbox_to_anchor=(1.02, -0.01), handlelength=2.0, labelspacing=0.18,
                    fontsize=6, markerscale=0.85, title="n = 1 seed per condition", title_fontsize=6,
                    alignment="left")
    leg.get_title().set_color(REF)


# b
def gap_curve(ln):
    tr, va = val_curve(ln, "train_accuracy"), val_curve(ln)
    return np.array([tr[e] - va[e] for e in sorted(va)])


def panel_gap_arms(ax, seeds):
    """Extended Data: train - validation gap vs epoch relative to the start of fine-class training, 3 arms."""
    ax.axvspan(-20.5, 0, color=WINDOW, lw=0, zorder=0)
    ax.axvline(0, color=REF, lw=0.5, ls=":", zorder=0)
    G = {a: np.array([gap_curve(lane(a, s)) for s in seeds]) for a in ARM}
    for a in ARM:
        off = 0 if a == "scratch" else K_CUR
        x = np.arange(1, G[a].shape[1] + 1) - off
        me = [i for i, xx in enumerate(x) if xx in (-19, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30)]
        traj(ax, x, G[a], RULE_ARM[a][0], ls=RULE_ARM[a][1], marker=ARM[a][3], ms=2.6, lw=1.3, label=ARM[a][1],
             markevery=me)
    coarse_end = np.concatenate([G[a][:, K_CUR - 1] for a in ("sc_true", "sc_rand")])
    fine_end = np.concatenate([G[a][:, -1] for a in ARM])
    for a in ARM:
        off = 0 if a == "scratch" else K_CUR
        f = G[a][:, off:off + 30].mean(0)
        b1 = np.polyfit(np.arange(1, 31), f, 1)[0]
        cz = G[a][:, :K_CUR].mean(0) if off else None
        print(f"[b] {a}: n={len(G[a])}; "
              + (f"coarse-phase gap range {cz.min():.2f}-{cz.max():.2f} pp, gap@ep20 {cz[-1]:.2f}; " if off else "")
              + f"fine-phase gap slope {b1:.3f} pp/epoch; final gap {G[a][:, -1].mean():.2f} pp "
              f"(seeds {np.round(G[a][:, -1], 2).tolist()})")
    print(f"[b] end of coarse phase (true+random pooled): {coarse_end.mean():.2f} pp; "
          f"end of fine training (all arms pooled): {fine_end.mean():.2f} pp")
    ax.annotate(f"≈{coarse_end.mean():.0f} pp", xy=(0, coarse_end.mean()), xytext=(-3.5, 12),
                fontsize=7, color=ERR, ha="center", va="bottom", **ANNOT,
                arrowprops=dict(arrowstyle="->", color=REF, lw=0.7, shrinkA=1, shrinkB=3))
    ax.annotate(f"≈{fine_end.mean():.0f} pp", xy=(30, fine_end.mean()), xytext=(24.5, 30.5),
                fontsize=7, color=ERR, ha="center", va="bottom", **ANNOT,
                arrowprops=dict(arrowstyle="->", color=REF, lw=0.7, shrinkA=1, shrinkB=3))
    ax.text(-10.25, 0.97, "coarse labels", transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=6.5, color=ERR)
    ax.text(15, 0.97, "fine labels", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.5,
            color=ERR)
    ax.set_xlim(-20.5, 31)
    lo = min((G[a].mean(0) - G[a].std(0, ddof=1) / np.sqrt(len(seeds)) if len(seeds) > 1 else G[a].mean(0)).min()
             for a in ARM)
    ax.axhline(0, color=REF, lw=0.6, ls="--", zorder=0)
    ax.set_ylim(np.floor(lo) - 1, 36)
    ax.set_xticks([-20, -10, 0, 10, 20, 30])
    ax.set_xlabel("Epoch relative to start of fine-class training")
    ax.set_ylabel("Train − validation gap (pp)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.42, 0.90), handlelength=2.6,
              title=f"n = {len(seeds)} seeds, mean ± SEM", title_fontsize=6, alignment="left")
    ax.get_legend().get_title().set_color(REF)
    return coarse_end.mean(), fine_end.mean()


def panel_b(ax, seeds, col=None):
    """Train and validation accuracy of the true-superclass arm vs epoch relative to the fine-class start.
    col (optional): line / fill colour; omitted -> ARM_COLOR['sc_true']."""
    lw = 1.1 if col is None else 1.3  # dashed validation line >= 1.2 with a custom colour
    col = ns.ARM_COLOR["sc_true"] if col is None else col
    ax.axvspan(-20.5, 0, color=WINDOW, lw=0, zorder=0)
    ax.axvline(0, color=REF, lw=0.5, ls=":", zorder=0)
    lns = [lane("sc_true", s) for s in seeds]
    TR = np.array([[v for _, v in sorted(val_curve(l, "train_accuracy").items())] for l in lns])
    VA = np.array([[v for _, v in sorted(val_curve(l).items())] for l in lns])
    x = np.arange(1, TR.shape[1] + 1) - K_CUR
    tr = traj(ax, x, TR, col, ls="-", marker=None, ms=0, lw=lw, label="Train")
    va = traj(ax, x, VA, col, ls="--", marker=None, ms=0, lw=lw, label="Validation")
    ax.fill_between(x, va, tr, where=tr >= va, color=col, alpha=0.15, lw=0, zorder=0.5, interpolate=True)
    g = TR - VA
    gc, gf = g[:, K_CUR - 1], g[:, -1]
    print(f"[b] sc_true n={len(seeds)}: end of coarse phase (ep {K_CUR}) train {TR[:, K_CUR - 1].mean():.2f} / "
          f"val {VA[:, K_CUR - 1].mean():.2f}, gap {gc.mean():.2f} pp (seeds {np.round(gc, 2).tolist()}); "
          f"end of fine training train {TR[:, -1].mean():.2f} / val {VA[:, -1].mean():.2f}, gap {gf.mean():.2f} pp "
          f"(seeds {np.round(gf, 2).tolist()})")
    ax.annotate(f"gap ≈ {gc.mean():.1f} pp", xy=(0.2, (tr[K_CUR - 1] + va[K_CUR - 1]) / 2), xytext=(3, 72),
                fontsize=7, color=ERR, ha="left", va="bottom", **ANNOT,
                arrowprops=dict(arrowstyle="->", color=REF, lw=0.7, shrinkA=1, shrinkB=2))
    ax.annotate(f"gap ≈ {gf.mean():.1f} pp", xy=(29.5, (tr[-1] + va[-1]) / 2), xytext=(19, 80),
                fontsize=7, color=ERR, ha="center", va="bottom", **ANNOT,
                arrowprops=dict(arrowstyle="->", color=REF, lw=0.7, shrinkA=1, shrinkB=2))
    ax.text(-10.25, 0.98, "coarse labels\n(20-way)", transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=6.5, color=ERR)
    ax.text(15, 0.98, "fine labels (100-way)", transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=6.5, color=ERR)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.84), handlelength=2.4, labelspacing=0.2)
    ax.set_xlim(-20.5, 31)
    ax.set_ylim(0, 100)
    ax.set_xticks([-20, -10, 0, 10, 20, 30])
    ax.set_xlabel("Epoch relative to start of fine-class training")
    ax.set_ylabel("Accuracy (%)")
    return gc.mean(), gf.mean()


# Extended Data: sweep gap
def panel_gap_sweep(axr):
    """Final gap of the sweep vs number of fine-class epochs (n = 1 per condition)."""
    tc, sw = sweep_table()
    fx, fy = [], []
    for c in COND:
        lid = sw.loc[c, "lane_id"]
        g = gap(SWEEP / lid)
        fe = int(sw.loc[c, "fine_epochs"])
        assert abs(g - sw.loc[c, "final_gap_pp"]) < 1e-6
        fx.append(fe)
        fy.append(g)
        axr.plot(fe, g, marker=CMK[c], color=CCOL[c], ms=4, ls="none", zorder=3)
        print(f"[b-right] {c}: fine epochs={fe}, final gap={g:.2f} pp")
    b1, b0 = np.polyfit(fx, fy, 1)
    r = np.corrcoef(fx, fy)[0, 1]
    xx = np.array([17, 53])
    axr.plot(xx, b0 + b1 * xx, color=REF, ls="--", lw=0.8, zorder=1)
    s30 = pd.read_csv(RESULTS / "CURRICULUM_sweep" / "tables" / "A_std30_summary.csv").set_index("metric")
    g30 = s30.loc["final_gap_pp"]
    per = tc[tc.condition == "standard30"].final_gap_pp.values
    axr.scatter(np.full(len(per), 30) + 1.6, per, s=7, color=DOT, lw=0, zorder=3)
    axr.errorbar(30, g30["mean"], yerr=g30["sd"], fmt="o", mfc="white", mec=REF_DARK, mew=0.9, ms=4,
                 ecolor=ERR, elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=4)
    axr.annotate(f"Standard, 30 epochs\n(n = {int(g30['n_reached'])}, mean ± SD)", xy=(31.9, g30["mean"] - 1.0),
                 xytext=(33.0, 21.2), fontsize=6, color=ERR, ha="left", va="top",
                 arrowprops=dict(arrowstyle="-", color=REF, lw=0.6))
    for c, x, y in zip(COND, fx, fy):
        if c in ("standard50", "switch_e30"):
            axr.text(x + (-2.2 if c == "standard50" else 2.2), y, CLAB[c], fontsize=6, color=ERR,
                     ha="right" if c == "standard50" else "left", va="center")
    axr.text(0.98, 0.03, f"r = {r:.3f}\n{b1:.1f} pp per\nfine-class epoch", transform=axr.transAxes,
             ha="right", va="bottom", fontsize=6, color=REF)
    axr.text(0.03, 0.99, "Sweep: n = 1 seed\nper condition", transform=axr.transAxes, ha="left", va="top",
             fontsize=6, color=REF)
    print(f"[b-right] sweep fit gap = {b0:.2f} + {b1:.3f} * fine_epochs (r = {r:.4f}); std30 n="
          f"{int(g30['n_reached'])} gap {g30['mean']:.2f} +- {g30['sd']:.2f} SD; fit at 30 = {b0 + b1 * 30:.2f}")
    axr.set_xlabel("Fine-class epochs")
    axr.set_ylabel("Final gap (pp)")
    axr.set_xlim(16, 54)
    axr.set_ylim(14, 40)
    axr.set_xticks([20, 30, 40, 50])
    return b1, r


# c
TARGETS = [20, 30, 40, 50]


def epochs_to(curve, t):
    """First fine-class epoch (1-based) with validation accuracy >= t; NaN if never reached."""
    hit = np.nonzero(np.asarray(curve) >= t)[0]
    return float(hit[0] + 1) if len(hit) else np.nan


def panel_c(ax, seeds):
    """Fine-class epochs to reach 20/30/40/50% validation accuracy; inset: fine-phase learning curves."""
    curves = {a: np.array([fine_curve(a, s) for s in seeds]) for a in ARM}
    T = {a: np.array([[epochs_to(c, t) for t in TARGETS] for c in curves[a]]) for a in ARM}  # seeds x targets
    w = 0.26
    off = {"sc_true": -w, "sc_rand": 0.0, "scratch": w}
    rng = np.random.default_rng(0)
    for a in ARM:
        col = ns.ARM_COLOR[a]
        for i, t in enumerate(TARGETS):
            v = T[a][:, i]
            v = v[~np.isnan(v)]
            x = i + off[a]
            m, se = mean_sem(v) if len(v) else (np.nan, np.nan)
            ax.bar(x, m, width=w * 0.92, color=col, alpha=ns.FILL_ALPHA, lw=0, zorder=1,
                   label=ARM[a][1] if i == 0 else None)
            ax.scatter(x + rng.uniform(-0.25, 0.25, len(v)) * w, v, s=7, color=DOT, lw=0, zorder=3)
            if len(v) > 1:
                ax.errorbar(x, m, yerr=se, fmt="none", ecolor=ERR, elinewidth=0.8, capsize=2, capthick=0.8,
                            zorder=4)
        for s_, row in zip(seeds, T[a]):
            print(f"[c] {a} seed {s_}: fine epochs to " +
                  ", ".join(f"{t}% = {'never' if np.isnan(e) else int(e)}" for t, e in zip(TARGETS, row)))
        print(f"[c] {a}: n={len(seeds)} mean " + ", ".join(
            f"{t}% = {np.nanmean(T[a][:, i]):.2f} ± {mean_sem(T[a][:, i][~np.isnan(T[a][:, i])])[1]:.2f} SEM"
            for i, t in enumerate(TARGETS)))
    saved = {t: T["scratch"][:, i] - T["sc_true"][:, i] for i, t in enumerate(TARGETS)}
    saved_r = {t: T["scratch"][:, i] - T["sc_rand"][:, i] for i, t in enumerate(TARGETS)}
    for t in TARGETS:
        print(f"[c] epochs saved vs scratch at {t}%: true {np.nanmean(saved[t]):.2f} (per seed "
              f"{saved[t].tolist()}), random {np.nanmean(saved_r[t]):.2f} (per seed {saved_r[t].tolist()})")
    top = np.nanmax(np.concatenate([T[a].ravel() for a in ARM]))
    for t in (30, 40):
        i = TARGETS.index(t)
        hi = np.nanmax(T["scratch"][:, i])
        lo = np.nanmean(T["sc_true"][:, i])
        sv = np.nanmean(saved[t])
        y = hi + 0.09 * top
        ax.plot([i + off["sc_true"], i + off["sc_true"], i + off["scratch"], i + off["scratch"]],
                [lo + 0.03 * top, y, y, hi + 0.03 * top], color=REF, lw=0.6, zorder=2)
        ax.text(i, y + 0.02 * top, f"≈{sv:.1f} epochs\nsaved".replace(".0 ", " "), ha="center", va="bottom",
                fontsize=6.5, color=ERR, **ANNOT)
    ax.set_xticks(range(len(TARGETS)))
    ax.set_xticklabels([f"{t}%" for t in TARGETS])
    ax.set_xlim(-0.55, len(TARGETS) - 0.45)
    ax.set_ylim(0, top * 1.55)
    ax.set_xlabel("Target validation accuracy")
    ax.set_ylabel("Fine-class epochs to reach target")
    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.04), handlelength=1.2, handleheight=0.8,
                    title=f"n = {len(seeds)} seeds, mean ± SEM", title_fontsize=6, alignment="left")
    leg.get_title().set_color(REF)

    return T, saved


# d
PROBE = RESULTS / "CURRICULUM_probe" / "probe.csv"


def load_probe(seeds):
    d = pd.read_csv(PROBE)
    d = d[d.layer == "avgpool"].drop_duplicates(["arm", "seed", "epoch"])  # final milestone is duplicated
    return d[d.seed.isin(seeds)]


def probe_matrix(d, arm, col):
    m = d[d.arm == arm].pivot(index="seed", columns="epoch", values=col) * 100
    return m.dropna(axis=1)


def panel_d(ax, axr, seeds, arm_style=None, bar_color=None, bar_alpha=None):
    """Left: 100-way probe over fine-class epochs (curriculum arms from the switch, scratch from init).
    Right: true - random probe differences at the switch (paired by seed).
    arm_style (optional): {arm: (colour, ls)} for the left panel (e.g. RULE_ARM); omitted -> ARM_COLOR, solid.
    bar_color / bar_alpha (optional): fill of the right-panel bars; omitted -> ARM_COLOR['sc_true'], FILL_ALPHA."""
    sty = arm_style or {a: (ns.ARM_COLOR[a], "-") for a in ARM}
    d = load_probe(seeds)
    M = {a: probe_matrix(d, a, "fine100") for a in ARM}
    fine = {a: (M[a].loc[:, M[a].columns >= K_CUR].rename(columns=lambda e: e - K_CUR) if a != "scratch"
                else M[a]) for a in ARM}
    n_cur = len(set(fine["sc_true"].index) & set(fine["sc_rand"].index))
    n_scr = len(fine["scratch"].index)
    for a in ("sc_rand", "sc_true", "scratch"):
        m = fine[a]
        traj(ax, m.columns.values, m.values, sty[a][0], ls=sty[a][1], marker=None, ms=0, lw=1.4)
        print(f"[d] {a}: seeds {list(m.index)}; fine100 probe by fine epoch "
              + ", ".join(f"{e}:{v:.1f}" for e, v in m.mean().items()))
    scr = fine["scratch"].mean()
    cur0 = np.mean([fine[a][0].mean() for a in ("sc_true", "sc_rand")])
    scr0 = scr[0]
    head = cur0 - scr0
    x_eq = np.interp(cur0, scr.values, scr.index.values)
    eq = {a: np.array([np.interp(v, scr.values, scr.index.values) for v in fine[a][0]]) for a in ("sc_true", "sc_rand")}
    print(f"[d] start: curriculum (true+random) {cur0:.2f}% (true {fine['sc_true'][0].mean():.2f}, random "
          f"{fine['sc_rand'][0].mean():.2f}), scratch {scr0:.2f}% -> head start {head:+.2f} pp; scratch reaches "
          f"{cur0:.2f}% at fine epoch {x_eq:.2f} (per seed true {np.round(eq['sc_true'], 2).tolist()}, "
          f"random {np.round(eq['sc_rand'], 2).tolist()})")
    arrow = dict(arrowstyle="<|-|>", color=ERR, lw=0.9, shrinkA=2.5, shrinkB=2.5, mutation_scale=6)
    ax.annotate("", xy=(0, cur0), xytext=(0, scr0), arrowprops=arrow, zorder=5)
    ax.text(4.5, scr0 + 2.0, f"{head:+.0f} pp before\nany fine label".replace("-", "−"), ha="left",
            va="bottom", fontsize=7, color=ERR)
    ax.annotate("", xy=(x_eq, cur0), xytext=(0, cur0), arrowprops=dict(arrow, arrowstyle="-|>", shrinkA=2.5, shrinkB=0),
                zorder=5)
    ax.text(x_eq + 0.8, cur0 - 0.8, f"≈{x_eq:.0f} epochs", ha="left", va="top", fontsize=7, color=ERR)
    ax.text(-0.3, 44.5, "True /\nrandom", ha="left", va="bottom", fontsize=7, color=sty["sc_true"][0])
    ax.text(13.5, 40.5, "Scratch", ha="left", va="top", fontsize=7, color=sty["scratch"][0])
    ax.text(30, 11, f"n = {n_cur} seeds" + (f" (scratch {n_scr})" if n_scr != n_cur else ""), ha="right", va="bottom", fontsize=6, color=REF)
    ax.set_xlim(-1.5, 31)
    ax.spines["bottom"].set_bounds(0, 30)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_ylim(10, 60)
    ax.set_xlabel("Fine-class epoch")
    ax.set_ylabel("Fine-class linear probe (%)")

    # right: true - random at the switch, paired by seed
    rows = [("Superclass separation\n(20-way probe)", probe_matrix(d, "sc_true", "super20")[K_CUR]
             - probe_matrix(d, "sc_rand", "super20")[K_CUR]),
            ("Within-superclass\ndistinctions (5-way)", probe_matrix(d, "sc_true", "within_sc")[K_CUR]
             - probe_matrix(d, "sc_rand", "within_sc")[K_CUR])]
    wt, wr = probe_matrix(d, "sc_true", "within_sc"), probe_matrix(d, "sc_rand", "within_sc")
    rows.append(("Within-SC gain in\nfirst fine epoch", (wt[K_CUR + 1] - wt[K_CUR]) - (wr[K_CUR + 1] - wr[K_CUR])))
    rng = np.random.default_rng(3)
    ys = np.arange(len(rows))[::-1] * 1.0
    col = ns.ARM_COLOR["sc_true"] if bar_color is None else bar_color
    for y, (lab, v) in zip(ys, rows):
        v = v.dropna().values
        m_, se = mean_sem(v)
        axr.barh(y, m_, height=0.34, color=col, alpha=ns.FILL_ALPHA if bar_alpha is None else bar_alpha, lw=0,
                 zorder=1)
        axr.scatter(v, y + rng.uniform(-0.07, 0.07, len(v)), s=8, color=DOT, lw=0, zorder=3)
        if len(v) > 1:
            axr.errorbar(m_, y, xerr=se, fmt="none", ecolor=ERR, elinewidth=0.8, capsize=2, capthick=0.8, zorder=4)
        end = (max(m_ + se, v.max()) if m_ >= 0 else min(m_ - se, v.min()))
        axr.text(end + (0.5 if m_ >= 0 else -0.5), y, f"{m_:+.1f}".replace("-", "−"), ha="left" if m_ >= 0 else "right",
                 va="center", fontsize=7, color=ERR)
        axr.text(-7.6, y + 0.24, lab, ha="left", va="bottom", fontsize=6.5, color=ERR, linespacing=1.0, zorder=6,
                 bbox=dict(fc="white", ec="none", pad=0.4))
        print(f"[d-right] {lab.replace(chr(10), ' ')}: true - random per seed {np.round(v, 2).tolist()}, "
              f"mean {m_:+.2f} pp (SEM {se:.2f})")
    axr.axvline(0, color=ERR, lw=0.7, zorder=2)
    axr.set_xlim(-7.6, 10)
    axr.set_ylim(-0.45, ys[0] + 0.85)
    axr.set_yticks([])
    axr.spines["left"].set_visible(False)
    axr.set_xticks([-5, 0, 5, 10])
    axr.set_xlabel("True − random (pp)")
    axr.set_title("True vs random grouping,\nat the switch", fontsize=7)
    return dict(head=head, cur0=cur0, scr0=scr0, x_eq=x_eq, rows=rows)


# Extended Data: Di-E share
def panel_di_share(ax, axb):
    bl = pd.read_csv(RESULTS / "CURRICULUM_threshold_free" / "tables" / "B_per_lane.csv")
    b = bl[(bl.measure == "B2b") & bl.cond.isin(COND)].set_index("cond").loc[COND]
    _, sw = sweep_table()
    share = (b.real_all_superclass + b.real_all_switch) / b.real_all_total
    rel = b.real_all_total / b.loc["standard50", "real_all_total"]
    K = np.array([KNOM[c] for c in COND])
    kk = np.array([0, 30])
    ax.plot(kk, kk / 50, color=REF, ls="--", lw=0.8, zorder=1)
    e15 = sw.loc["switch_e15"]
    t15 = e15.superclass_epochs_effective / e15.total_epochs_effective
    ax.plot(15, t15, marker="o", ms=3.4, mfc="white", mec=REF, mew=0.8, ls="none", zorder=2)
    ax.annotate(f"{int(e15.superclass_epochs_effective)}/{int(e15.total_epochs_effective)}†", xy=(15, t15),
                xytext=(17.5, 0.17), fontsize=6, color=REF, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=REF, lw=0.5, shrinkA=1, shrinkB=2))
    ax.plot(K, share.values, color=CSI, lw=1.1, zorder=3)
    for c, k, s in zip(COND, K, share.values):
        open_ = c == "switch_e15"
        ax.plot(k, s, marker="o", ms=4, color=CSI, mfc="white" if open_ else CSI, mew=1.0, ls="none", zorder=4)
        print(f"[d] {c}: K={k} share(SC+switch)={s:.3f} (time share {k / 50:.2f}"
              f"{f', effective {t15:.3f}' if open_ else ''}); total rel. Std50 = {rel[c]:.3f} "
              f"(total {b.loc[c, 'real_all_total']:.0f})")
    for c in ("switch_e10", "switch_e30"):
        ax.text(KNOM[c] + 1.0, share[c] - 0.012, f"{share[c] * 100:.0f}%", fontsize=6, color=ERR,
                ha="left", va="top")
    ax.set_xlim(-2.5, 32.5)
    ax.set_ylim(-0.03, 0.68)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_yticklabels(["0", "20", "40", "60"])
    ax.set_ylabel("Di-E in superclass\nphase + switch (%)")
    ax.tick_params(labelbottom=False)
    ax.legend(handles=[Line2D([], [], color=CSI, marker="o", ms=4, lw=1.1, label="Share of Di-E (calibrated)"),
                       Line2D([], [], color=REF, ls="--", lw=0.8, label="Share of training time (K / 50)")],
              loc="upper left", handlelength=2.2)
    ax.text(0.98, 0.03, "n = 1 seed per sweep condition\n(descriptive)", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=6, color=REF)

    axb.bar(K, rel.values, width=3.2, color=CSI, alpha=0.3, lw=0)
    axb.axhline(1, color=REF_DARK, ls="--", lw=0.6)
    for k, v in zip(K, rel.values):
        axb.text(k, v / 2, f"{v:.2f}", fontsize=6, ha="center", va="center", color=ERR)
    axb.set_ylim(0, 1.45)
    axb.set_yticks([0, 1])
    axb.set_ylabel("Total Di-E\n(rel. Std)", fontsize=6.5)
    axb.set_xlim(*ax.get_xlim())
    axb.set_xticks(K)
    axb.set_xlabel("Superclass epochs K")
    axb.annotate("† switch_e15: resumed run; 11 effective superclass epochs of 46 (open markers)",
                 xy=(0, 0), xycoords="axes fraction", xytext=(-28, -30), textcoords="offset points",
                 fontsize=6, color=REF, ha="left", va="top")
    return share, rel



def panel_e(ax, style=None):
    """Di-E per 1,000 iterations over training (sweep, n = 1 seed per condition; common thresholds = those of
    standard-50, tset 'std50'), standard vs switch_e20; dotted = same-state SAE null of each run.
    style (optional): {'standard50': (colour, ls), 'switch_e20': (colour, ls), 'null': (colour, ls, lw, alpha)}
    (e.g. RULE_E: all amber, curriculum solid, standard dotted, nulls thin dotted); omitted -> default colours."""
    st = style or {}
    ECOL = {"standard50": st.get("standard50", (REF_DARK,))[0], "switch_e20": st.get("switch_e20",
                                                                                     (ns.ARM_COLOR["sc_true"],))[0]}
    ELS = {"standard50": st.get("standard50", (None, "-"))[1], "switch_e20": st.get("switch_e20", (None, "-"))[1]}
    NUL = st.get("null")
    d = pd.read_csv(RESULTS / "CURRICULUM_sweep" / "tables" / "B_events_per_transition.csv")
    d = d[d.tset == "std50"]
    out = {}
    for c, lab in (("standard50", "Standard"), ("switch_e20", "Curriculum (switch e20)")):
        x = d[d.cond == c].sort_values("epoch_b")
        ep = x.epoch_b.values
        rate = x.di_h_real_all.values / x.updates.values      # x 10^3 per 1,000 iterations
        null = x.di_h_null.values / x.updates.values
        ax.plot(ep, rate, color=ECOL[c], ls=ELS[c], marker=CMK[c], ms=3.6, lw=1.2 if style is None else 1.4,
                zorder=3)
        if NUL is None:
            ax.plot(ep, null, color=ECOL[c], ls=":", lw=0.8, zorder=2)
        else:
            ax.plot(ep, null, color=NUL[0], ls=NUL[1], lw=NUL[2], alpha=NUL[3], zorder=2)
        out[c] = (ep, rate, null, x.di_h_real_all.values)
        print(f"[e] {c}: Di-E per 1k it by epoch " + ", ".join(f"{e}:{r:.1f}" for e, r in zip(ep, rate))
              + f"; first transition {x.di_h_real_all.values[0]:.0f} events (null {x.di_h_null.values[0]:.0f})")
    ax.axvspan(0, 20, color="#F3F4F6", zorder=0, lw=0)
    ax.axvline(20, color=REF, ls=":", lw=0.7, zorder=1)
    ax.text(10.5, 48.5, "superclass phase", ha="center", va="top", fontsize=6.5, color=REF)
    s0, c0 = out["standard50"][3][0], out["switch_e20"][3][0]
    ax.annotate(f"early burst\n{s0 / c0:.0f}× smaller", xy=(out["switch_e20"][0][0], out["switch_e20"][1][0]),
                xytext=(9.5, 30), fontsize=7, color=ERR, ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=ERR, lw=0.7, shrinkA=1, shrinkB=3, mutation_scale=6))
    ax.text(out["standard50"][0][0] + 1.4, out["standard50"][1][0] - 3.5, "Standard", fontsize=7, color=ECOL["standard50"],
            ha="left", va="center")
    ax.text(31, 23.5, "Curriculum", fontsize=7, color=ECOL["switch_e20"], ha="left", va="bottom")
    if NUL is None:
        ax.text(50, 0.6, "dotted: same-state null; n = 1 seed", fontsize=6, color=REF, ha="right", va="bottom")
    else:  # longer note: top right, clear of the phase boundary
        ax.text(50, 47.5, "thin dotted: same-state null\nn = 1 seed per condition", fontsize=6, color=REF,
                ha="right", va="top", linespacing=1.1)
    ax.set_xlim(0, 51)
    ax.set_ylim(0, 48)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Di-E per 1,000 iterations (×10³)")
    ax.set_title("Differentiation waits for fine labels")
    return out


def main():
    ns.apply()
    seeds = complete_seeds()
    status = {a: [s for s in SEEDS if lane_done(lane(a, s))] for a in ARM}
    print(f"curriculum lanes done per arm: {status}; complete seeds (all three arms): {seeds} -> n = {len(seeds)}")
    if not seeds:
        raise SystemExit("no complete curriculum seed yet")
    fig = plt.figure(figsize=(ns.FULL_W, 150 * ns.MM))
    gs = fig.add_gridspec(2, 1, hspace=0.5)
    gt = GridSpecFromSubplotSpec(1, 2, gs[0], wspace=0.28)
    gb = GridSpecFromSubplotSpec(1, 3, gs[1], wspace=0.42)
    ax_a, ax_b = fig.add_subplot(gt[0]), fig.add_subplot(gt[1])
    ax_c, ax_d, ax_e = fig.add_subplot(gb[0]), fig.add_subplot(gb[1]), fig.add_subplot(gb[2])
    # true vs random differences at the switch go to Extended Data
    fig_ed, ax_dr = plt.subplots(figsize=(ns.HALF_W * 0.62, 62 * ns.MM))

    panel_a(ax_a)
    panel_b(ax_b, seeds)
    panel_c(ax_c, seeds)
    # the ED true-vs-random bars are probe differences (no index): neutral fill
    dres = panel_d(ax_d, ax_dr, seeds, bar_color=ns.NEUTRAL, bar_alpha=0.4)
    ax_b.set_title("Coarse learning generalises, fine learning memorises")
    ax_d.set_title("Coarse labels pre-build\nfine-class structure")
    panel_e(ax_e)
    for ax, l, x in ((ax_a, "a", -0.14), (ax_b, "b", -0.17), (ax_c, "c", -0.24), (ax_d, "d", -0.26),
                     (ax_e, "e", -0.26)):
        ns.panel(ax, l, x=x, y=1.12)
    print("saved", ns.save(fig, "curriculum_sweep_panels"))
    print("saved", ns.save(fig_ed, "curriculum_probe_true_random"))

    fig, ax = plt.subplots(figsize=(ns.FULL_W * 0.5, 62 * ns.MM))
    panel_gap_arms(ax, seeds)
    ax.set_title("Train − validation gap, all arms")
    print("saved", ns.save(fig, "curriculum_gap_arms"))

    fig = plt.figure(figsize=(ns.FULL_W * 0.5, 72 * ns.MM))
    ge = fig.add_gridspec(2, 1, height_ratios=[2.6, 1], hspace=0.12)
    ax, axb = fig.add_subplot(ge[0]), fig.add_subplot(ge[1])
    panel_di_share(ax, axb)
    ax.set_title("Differentiation is redistributed, not reduced")
    print("saved", ns.save(fig, "curriculum_di_share"))

    fig, ax = plt.subplots(figsize=(ns.HALF_W, 62 * ns.MM))
    panel_gap_sweep(ax)
    ax.set_title("Overfitting gap tracks fine-class epochs")
    print("saved", ns.save(fig, "curriculum_gap_sweep"))


if __name__ == "__main__":
    main()
