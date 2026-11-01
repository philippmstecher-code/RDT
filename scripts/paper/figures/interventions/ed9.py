#!/usr/bin/env python3
"""Extended Data Fig. 9: pre-built coarse structure moves fine-grained learning earlier. Level colour rule
(nmi_style.BLUE / AMBER / PURPLE / GREYS: hue = level measured; true superclasses darkest solid or filled, random
grouping mid dashed or hatched, superclass-blind task light dotted or open, scratch thin grey or open).

a  fine-class accuracy, epochs ahead of the same seed's scratch run at every fine-class epoch (fig6_panels.lead_vs_drawn;
   seeds above scratch's best accuracy enter as lower bounds, open circles); grey area, the lead after equal
   superclass-blind pre-training; amber area, the lead beyond it
b  head start after one fine-class epoch for five pre-training tasks of 20 epochs, by label count (5: superclass-blind
   task, five random groups of 20 classes, five groups of four whole superclasses, RAND5_PRETRAIN/ and
   SUP5_PRETRAIN/summary.json from rand5_pretrain_analysis.py; 20: random grouping, true superclasses)
c  events per process during fine-class training, scratch thresholds in every arm, last block (L.events_common);
   process hues (Tg-E purple, Ab-E blue, Di-E amber); scratch open grey; numbers, change against scratch, range over
   seeds and both pre-training arms
d  cumulative differentiation events (Di-E), aligned at the first fine-class epoch; dots and number, share before the
   first fine-class label (range over seeds and both groupings)
e  the 50-epoch sweep (seed 42, one run per condition): validation accuracy, fine-class training in an amber ramp by
   pre-training length (0 = scratch, grey), the 20-class phase in blue dashed; open circle, 30 fine-class epochs from
   scratch (controlled curriculum, mean +- s.d., 5 seeds)
Fig. 6 (fig6.py) uses panel_dose, lead1 and the sweep helpers defined here.
Writes ed9.{pdf,png,tif} and Source Data ed9_* to ES_FIG_OUT.
Run: ES_FIG_OUT=<repo>/analysis_outputs/paper/figures/interventions .venv/bin/python scripts/paper/figures/interventions/ed9.py
"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import nmi_style as ns  # noqa: E402
import curriculum_lib as L  # noqa: E402
import blind_arm as B  # noqa: E402
import fig6_panels as F6N  # noqa: E402
from fig_common import RESULTS  # noqa: E402

NAME = "ed9"
SRC = {}
SEEDS = L.SEEDS
K = 20
FINAL_SD = ns.OUT / "final" / "source_data_ed"     # read-only: Source Data of an earlier render
SCR = ns.SCRATCH_STYLE
GT = ns.GREY_TXT
SHADE = ns.COND_SHADE
LW = 1.3
AMB_RAMP = LinearSegmentedColormap.from_list("amb", ns.AMBER[::-1])     # light -> dark
BLU_RAMP = LinearSegmentedColormap.from_list("blu", ns.BLUE[::-1])
RESUMED = "switch_e15"
RESUME_OFFSET = 4   # switch_e15: counter epoch - true epoch after the resume (devtrain_batch_log.jsonl)
ARMKEY = {"scratch": "scratch", "sc_true": "true", "sc_rand": "rand", "sc_blind": "blind"}


def style(a, ramp):
    """(colour, line style, line width) of an arm in a ramp (scratch grey)."""
    if a == "scratch":
        return SCR["color"], "-", SCR["lw"]
    i, ls = SHADE[a]
    return ramp[i], ls, LW


def ramp_t(k):
    return 0.2 + 0.8 * k / 30


# ───────────────────────── a: head start over fine-class training (the former Fig. 6d) ─────────────────────────
def panel_lead(ax):
    curves = {a: {s: L.fine_val(a, s) for s in SEEDS} for a in ("scratch", "sc_true", "sc_rand")}
    curves["sc_blind"] = B.fine_curves()
    x = np.arange(1, 31)
    M, rows = {}, []
    for a, lab, c, ls in (("sc_blind", "superclass-blind task", ns.AMBER[2], ":"),
                          ("sc_rand", "random grouping", ns.AMBER[1], "--"), ("sc_true", "superclasses", ns.AMBER[0], "-")):
        Y = np.array([[F6N.lead_vs_drawn(v, curves["scratch"][s]) + 1 - t if v <= max(curves["scratch"][s]) else np.nan
                       for t, v in zip(x, curves[a][s])] for s in SEEDS])
        cens = np.isnan(Y)
        Y = np.where(cens, 30 - x[None, :], Y)        # above scratch's best: lower bound
        M[a] = F6N.band(ax, x, Y, c, ls, lw=F6N.LW, alpha=0.0, z=4 if a == "sc_true" else 3)
        if a in ("sc_true", "sc_blind"):
            ks = [i for i, e in enumerate(x) if e == 1 or e % 5 == 0]
            _, lo, hi = zip(*(L.tci(Y[:, i]) for i in ks))
            ax.vlines(x[ks], lo, hi, color=c, lw=0.7, alpha=0.7, zorder=2)
        if cens.any():
            ax.scatter(x[cens.any(0)], M[a][cens.any(0)], s=9, facecolor="white", edgecolor=c, lw=0.8, zorder=6)
        ax.scatter([1], [M[a][0]], s=16, color=c if a != "sc_blind" else "white", edgecolor=c, lw=1.0, zorder=6)
        rows += [dict(task=lab, seed=s, fine_epoch=int(e), epochs_ahead_of_scratch=v, lower_bound=bool(cb))
                 for s, y, cy in zip(SEEDS, Y, cens) for e, v, cb in zip(x, y, cy)]
    ax.fill_between(x, M["sc_true"], M["sc_blind"], color=ns.AMBER[3], alpha=0.45, lw=0, zorder=1.5)
    ax.fill_between(x, 0, M["sc_blind"], where=M["sc_blind"] > 0, interpolate=True, color=ns.GREYS[3], lw=0,
                    zorder=1.4)                     # the lead that equal (superclass-blind) pre-training gives
    ax.axhline(0, color=SCR["color"], lw=SCR["lw"], zorder=2)
    for v, lab, c, va in ((M["sc_true"][0] + 0.15, "4.0", ns.AMBER_TXT[0], "bottom"),
                          (M["sc_rand"][0] - 0.15, "3.5", ns.AMBER_TXT[1], "top"),
                          (M["sc_blind"][0], "0.8", ns.AMBER_TXT[1], "center")):
        ax.text(0.4, v, lab, fontsize=6, color=c, ha="right", va=va)
    ax.set_xlim(-1.2, 30.8)
    ax.set_xticks([1, 10, 20, 30])
    ax.set_ylim(-4.5, 5.8)
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.set_xlabel("Fine-class epoch")
    ax.set_ylabel("Fine-class accuracy,\nepochs ahead of scratch")
    SRC["a"] = pd.DataFrame(rows)
    print("[a] after one fine-class epoch " + ", ".join(f"{k}: {v[0]:.2f}" for k, v in M.items()))


# ───────────────────────── head start against pre-training length (Fig. 6c) ─────────────────────────
SWEEP_K_TRUE = {"switch_e15": 11}      # the resumed run switched after 11 epochs (counter epoch 15)
SEEDS_RDT = [7, 42, 137]               # open seeds of the RDT census


def lead1(arm, s, k=K):
    """Epochs ahead of the same seed's 30-epoch scratch run after one fine-class epoch (Fig. 6d definition)."""
    v = B.fine_curves()[s][0] if arm == "sc_blind" else L.curve(f"{arm}_K{k}", s)[k + 1]
    return F6N.lead_vs_drawn(v, L.fine_val("scratch", s))


def mean_ci(ax, x, v, c, filled, ms=4.0, z=5):
    m, lo, hi = L.tci(v)
    ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=ms, mfc=c if filled else "white", mec=c, mew=1.0,
                ecolor=c, elinewidth=0.9, capsize=0, zorder=z)
    return m, lo, hi


def panel_dose(ax):
    ref = L.sweep_curve("standard50")
    ref = np.array([ref[e] for e in range(1, 51)])
    rows = [dict(source="sweep, seed 42", condition="standard50", pretrain_epochs=0, head_start=0.0)]
    for c in L.COND[1:]:
        k = L.KNOM[c]
        rows.append(dict(source="sweep, seed 42", condition=c, pretrain_epochs=SWEEP_K_TRUE.get(c, k),
                         head_start=F6N.lead_vs_drawn(L.sweep_curve(c)[k + 1], ref)))
    sw = pd.DataFrame(rows)
    ax.plot(sw.pretrain_epochs, sw.head_start, color=ns.AMBER[0], lw=0.7, zorder=3)
    for _, r in sw.iterrows():
        res = r.condition in SWEEP_K_TRUE
        ax.plot(r.pretrain_epochs, r.head_start, "o", ms=2.2, mfc="white" if res else ns.AMBER[0],
                mec=ns.AMBER[0], mew=0.7, zorder=4)
    ax.text(1.0, 1.4, "sweep,\none seed", fontsize=6, color=ns.AMBER_TXT[0], ha="left", va="bottom",
            linespacing=0.95)
    summ = []
    for a, dx in (("sc_true", 1.3), ("sc_rand", 2.6)):
        c, ls, _ = style(a, ns.AMBER)
        M = []
        for k in (20, 30):
            v = np.array([lead1(a, s, k) for s in SEEDS])
            M.append(mean_ci(ax, k + dx, v, c, filled=True)[0])
            summ.append(dict(source="controlled, 5 seeds", arm=L.ARM_LABEL[a], pretrain_epochs=k, mean=v.mean(),
                             ci_lo=L.tci(v)[1], ci_hi=L.tci(v)[2], **{f"seed_{s}": x for s, x in zip(SEEDS, v)}))
        ax.plot([20 + dx, 30 + dx], M, color=c, ls=ls, lw=0.9, zorder=4)
    vb = np.array([lead1("sc_blind", s) for s in SEEDS])
    c, _, _ = style("sc_blind", ns.AMBER)
    mean_ci(ax, 20 + 3.9, vb, c, filled=False)
    ax.text(25.0, vb.mean(), "blind", fontsize=6, color=ns.AMBER_TXT[1], ha="left", va="center")
    summ.append(dict(source="controlled, 5 seeds", arm="superclass-blind task", pretrain_epochs=20, mean=vb.mean(),
                     ci_lo=L.tci(vb)[1], ci_hi=L.tci(vb)[2], **{f"seed_{s}": x for s, x in zip(SEEDS, vb)}))
    ax.text(19.5, 6.6, "5 seeds\n(mean, 95% CI)", fontsize=6, color=GT, ha="right", va="center", linespacing=0.95)
    ax.axhline(0, color=L.REFLINE, lw=0.5, ls="--", zorder=0)
    ax.set_xlim(-1.5, 35)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_ylim(-0.6, 7.6)
    ax.set_yticks([0, 2, 4, 6])
    ax.set_xlabel("Superclass pre-training (epochs)")
    ax.set_ylabel("Head start after one\nfine-class epoch (epochs)")
    SRC["a"] = pd.concat([sw, pd.DataFrame(summ)], ignore_index=True)
    print("[a] sweep " + ", ".join(f"{int(r.pretrain_epochs)}:{r.head_start:.2f}" for _, r in sw.iterrows()))
    print("[a] " + "; ".join(f"{r['arm']} {r['pretrain_epochs']}: {r['mean']:.2f}" for r in summ))


# ───────────────────────── b: 20 against 30 epochs, true superclasses ─────────────────────────
def census_at_switch(path, lane, k):
    d = pd.read_csv(path)
    d = d[(d.src == "sae") & (d.lane == lane) & (d.epoch == k)].set_index("seed").census_genuine * 100
    return d.loc[SEEDS_RDT].values


def paired(ax, v20, v30, c, fmt):
    for a_, b_ in zip(v20, v30):
        ax.plot([0, 1], [a_, b_], color=ns.GREYS[2], lw=0.6, marker="o", ms=1.8, zorder=2)
    for x, v in ((0, v20), (1, v30)):
        ax.plot([x - 0.22, x + 0.22], [v.mean()] * 2, color=c, lw=2.0, solid_capstyle="butt", zorder=4)
        ax.text(x + (-0.3 if x == 0 else 0.3), v.mean(), fmt(v.mean()), fontsize=6, color=c,
                ha="right" if x == 0 else "left", va="center")
    ax.set_xlim(-1.35, 1.85)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["20", "30"])


def panel_k30(axc, axl):
    cen = RESULTS / "DICTFREE" / "hierarchy" / "D_sae_census_timecourse.csv"
    k30 = RESULTS / "SWEEP_RDT" / "census_K30.csv"
    c20, c30 = census_at_switch(cen, "sc_true_K20", 20), census_at_switch(k30, "sc_true_K30", 30)
    r20, r30 = census_at_switch(cen, "sc_rand_K20", 20), census_at_switch(k30, "sc_rand_K30", 30)
    l20 = np.array([lead1("sc_true", s, 20) for s in SEEDS])
    l30 = np.array([lead1("sc_true", s, 30) for s in SEEDS])
    paired(axc, c20, c30, ns.BLUE[0], lambda m: f"{m:.0f}%")
    paired(axl, l20, l30, ns.AMBER[0], lambda m: f"{m:.1f}")
    axc.set_ylim(18, 36); axc.set_yticks([20, 25, 30, 35])
    axc.set_ylabel("Superclass features\nat the switch (% of alive)")
    axl.set_ylim(0, 8); axl.set_yticks([0, 2, 4, 6, 8])
    axl.set_ylabel("Head start (epochs)")
    for a_ in (axc, axl):
        a_.set_xlabel("Pre-training (epochs)")
    rows = [dict(measure="RDT features selective for the true superclasses at the switch (% of alive)",
                 task=t, pretrain_epochs=k, seed=s, value=x)
            for t, k, v in (("true superclasses", 20, c20), ("true superclasses", 30, c30),
                            ("random grouping", 20, r20), ("random grouping", 30, r30))
            for s, x in zip(SEEDS_RDT, v)]
    rows += [dict(measure="head start after one fine-class epoch (epochs)", task="true superclasses",
                  pretrain_epochs=k, seed=s, value=x) for k, v in ((20, l20), (30, l30)) for s, x in zip(SEEDS, v)]
    SRC["b"] = pd.DataFrame(rows)
    print(f"[b] census true {c20.round(1).tolist()} -> {c30.round(1).tolist()} ({c20.mean():.1f} -> {c30.mean():.1f}); "
          f"random {r20.mean():.1f} -> {r30.mean():.1f} ({int((r30 > r20).sum())}/3 up); "
          f"lead {l20.mean():.2f} -> {l30.mean():.2f} ({int((l30 > l20).sum())}/5 up), diff CI "
          f"{L.tci(l30 - l20)[1]:.2f}-{L.tci(l30 - l20)[2]:.2f}")


# ───────────────────────── c: label count ─────────────────────────
def five_label(arm):
    d = json.load(open(RESULTS / f"{arm.upper()}_PRETRAIN" / "summary.json"))["lead_after_one_fine_epoch"]
    return np.array([d[str(s)] for s in SEEDS])


def panel_labels(ax):
    tasks = [("superclass-blind", 5, np.array([lead1("sc_blind", s) for s in SEEDS]), ns.AMBER[2], "open"),
             ("random groups", 5, five_label("rand5"), ns.AMBER[2], "hatch"),
             ("whole superclasses", 5, five_label("sup5"), ns.AMBER[2], "fill"),
             ("random grouping", 20, np.array([lead1("sc_rand", s) for s in SEEDS]), ns.AMBER[1], "hatch"),
             ("true superclasses", 20, np.array([lead1("sc_true", s) for s in SEEDS]), ns.AMBER[0], "fill")]
    xs = [0, 1, 2, 3.6, 4.6]
    rows, w = [], 0.72
    for x, (lab, n, v, c, st) in zip(xs, tasks):
        m, lo, hi = L.tci(v)
        if st == "open":
            ax.bar(x, m, width=w, facecolor="white", edgecolor=c, lw=0.9, ls=":", zorder=1)
        elif st == "hatch":
            ax.bar(x, m, width=w, facecolor=c, edgecolor=c, alpha=0.45, hatch=L.HATCH, lw=0.9, zorder=1)
        else:
            ax.bar(x, m, width=w, facecolor=c, edgecolor=c, alpha=0.85, lw=0.9, zorder=1)
        ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=ns.GREYS[0], elinewidth=0.8, capsize=0,
                    zorder=4)
        rng = np.random.default_rng(int(10 * x))
        ax.scatter(x + rng.uniform(-0.2, 0.2, len(v)) * w, v, s=5, color=ns.GREYS[0], edgecolors="white",
                   linewidths=0.3, zorder=3)
        ax.text(x, max(hi, v.max()) + 0.25, f"{m:.1f}", fontsize=6, color=GT, ha="center", va="bottom")
        rows += [dict(task=lab, labels=n, seed=s, head_start=y) for s, y in zip(SEEDS, v)]
    for x0, x1, t in ((0, 2, "5 labels"), (3.6, 4.6, "20 labels")):
        ax.plot([x0 - 0.35, x1 + 0.35], [-2.05] * 2, color=GT, lw=0.6, clip_on=False)
        ax.text((x0 + x1) / 2, -2.2, t, fontsize=6, color=GT, ha="center", va="top", clip_on=False)
    ax.set_xticks(xs)
    ax.set_xticklabels(["blind", "ran-\ndom", "super-\nclass\ngroups", "ran-\ndom", "super-\nclasses"], fontsize=6,
                       linespacing=0.9)
    ax.set_xlim(-0.6, 5.2)
    ax.set_ylim(0, 7.6)
    ax.set_yticks([0, 2, 4, 6])
    ax.set_ylabel("Head start (epochs)")
    D = pd.DataFrame(rows)
    P = D.pivot(index="seed", columns="task", values="head_start")
    st = [L.paired(P[a] - P[b], f"{a} - {b}") for a, b in
          (("whole superclasses", "random groups"), ("whole superclasses", "superclass-blind"),
           ("random groups", "superclass-blind"), ("true superclasses", "whole superclasses"),
           ("random grouping", "random groups"))]
    SRC["c"], SRC["c_stats"] = D, pd.DataFrame(st)
    print("[c] " + "; ".join(f"{r['contrast']}: {r['mean']:.2f} ({r['ci_lo']:.2f}-{r['ci_hi']:.2f}) {r['n_pos']}/5"
                             for r in st))


# ───────────────────────── d: the 50-epoch sweep ─────────────────────────
def panel_sweep(ax):
    rows = []
    for c in L.COND:
        v = L.sweep_curve(c)
        k = L.KNOM[c]
        mk = ns.SWEEP[k][1]
        tx = (lambda e: e - RESUME_OFFSET if e >= 10 else e) if c == RESUMED else (lambda e: e)
        k_true = tx(k) if k else 0
        col = AMB_RAMP(ramp_t(k_true)) if k else SCR["color"]
        if k:
            bc = BLU_RAMP(ramp_t(k_true))
            sc = [e for e in [1] + list(range(5, k + 1, 5)) if e in v]
            ax.plot([tx(e) for e in sc], [v[e] for e in sc], color=bc, ls="--", lw=0.9, zorder=1)
            ax.plot([tx(k), tx(k + 5)], [v[k], v[k + 5]], color=col, ls=":", lw=0.8, zorder=1)
            fine = list(range(k + 5, 51, 5))
            rows += [dict(condition=c, pretrain_epochs=k_true, epoch=tx(e), counter_epoch=e,
                          phase="pre-training (20-class)", val_acc=v[e]) for e in sc]
        else:
            fine = [1] + list(range(5, 51, 5))
        lab = "11 (resumed)" if c == RESUMED else ("none" if not k else f"{k}")
        ax.plot([tx(e) for e in fine], [v[e] for e in fine], color=col, marker=mk, ms=3.0, lw=1.0 if k else 0.9,
                mfc="white" if c == RESUMED else col, mec=col, mew=0.9, zorder=3, label=lab)
        rows += [dict(condition=c, pretrain_epochs=k_true, epoch=tx(e), counter_epoch=e, phase="fine-class (100-class)",
                      val_acc=v[e]) for e in fine]
    ax.axhspan(57, 59, color=L.WINDOW, ec="#D1D5DB", lw=0.5, zorder=0)
    ax.text(1.0, 58, "57–59%", fontsize=6, va="center", ha="left", color=GT)
    v30 = np.array([L.fine_val("scratch", s)[-1] for s in SEEDS])
    m30, sd30 = v30.mean(), v30.std(ddof=1)
    ax.errorbar(30, m30, yerr=sd30, fmt="o", ms=4.2, mfc="white", mec=ns.GREYS[0], mew=1.1, ecolor=ns.GREYS[0],
                elinewidth=0.9, capsize=2, capthick=0.9, zorder=6)
    ax.text(1.0, 71.5, "○ scratch, 30 epochs (5 seeds)", fontsize=6, color=ns.GREYS[0], ha="left", va="top")
    ax.annotate("", xy=(50.0, 63.6), xytext=(30.0, 63.6),
                arrowprops=dict(arrowstyle="-|>", color=ns.NEUTRAL, lw=0.9, mutation_scale=7, shrinkA=0, shrinkB=0),
                zorder=6)
    ax.plot([30, 30], [m30 + sd30 + 0.4, 63.6], color=ns.NEUTRAL, lw=0.8, ls=":", zorder=5)
    ax.text(40.0, 64.3, "no time saved", fontsize=6, color=GT, ha="center", va="bottom")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_xlim(0, 52)
    ax.set_ylim(10, 72)
    ax.spines["left"].set_bounds(10, 60)
    ax.set_yticks(range(10, 61, 10))
    ax.set_xticks(range(0, 51, 10))
    h, lab = ax.get_legend_handles_labels()
    h.append(Line2D([], [], color=ns.BLUE[1], ls="--", lw=0.9))
    lab.append("20-class phase")
    leg = ax.legend(h, lab, loc="lower right", bbox_to_anchor=(1.02, -0.01), handlelength=1.8, labelspacing=0.15,
                    columnspacing=0.8, ncol=2, fontsize=6, markerscale=0.9, title="Superclass pre-training (epochs)",
                    title_fontsize=6, alignment="left")
    leg.get_title().set_color(GT)
    rows += [dict(condition="scratch, 30 fine-class epochs (controlled curriculum, own schedule)", pretrain_epochs=0,
                  epoch=30, phase="final", val_acc=x, seed=s) for s, x in zip(SEEDS, v30)]
    rows.append(dict(condition="scratch summary", epoch=30, phase="mean", val_acc=m30, sd=sd30, n=len(v30)))
    SRC["d"] = pd.DataFrame(rows)
    print(f"[d] scratch 30 epochs {m30:.2f} +- {sd30:.2f}; sweep final "
          + ", ".join(f"{c}:{L.sweep_curve(c)[50]:.2f}" for c in L.COND))


# ───────────────────────── e: events per process in fine-class training ─────────────────────────
PROC = (("tg_h", "Tg-E", ns.PURPLE), ("ab_h", "Ab-E", ns.BLUE), ("di_h", "Di-E", ns.AMBER))
LANES3 = (("scratch_E30", "scratch"), ("sc_rand_K20", "sc_rand"), ("sc_true_K20", "sc_true"))


def bar(ax, x, vals, arm, ramp, w, seed):
    m = vals.mean()
    if arm == "scratch":
        ax.bar(x, m, width=w, facecolor="white", edgecolor=ns.GREYS[1], lw=0.9, zorder=1)
    elif arm == "sc_rand":
        ax.bar(x, m, width=w, facecolor=ramp[1], edgecolor=ramp[1], alpha=0.45, hatch=L.HATCH, lw=0.9, zorder=1)
    else:
        ax.bar(x, m, width=w, facecolor=ramp[0], edgecolor=ramp[0], alpha=0.85, lw=0.9, zorder=1)
    rng = np.random.default_rng(seed)
    ax.scatter(x + rng.uniform(-0.2, 0.2, len(vals)) * w, vals, s=6, color=ns.GREYS[0], edgecolors="white",
               linewidths=0.3, zorder=3)


def panel_events(ax):
    d = L.events_common()
    f = d[d.phase == "fine"].groupby(["arm", "seed"])[["tg_h", "ab_h", "di_h"]].sum()
    rows, per, stat = [], [], []
    w = 0.26
    for j, (col, plab, ramp) in enumerate(PROC):
        base = f.loc["scratch_E30"][col]
        for k, (lane, arm) in enumerate(LANES3):
            v = f.loc[lane][col] / 1e3
            bar(ax, j + (k - 1) * w, v.values, arm, ramp, w * 0.9, 10 * j + k)
            rows += [dict(process=plab, arm=L.ARM_LABEL[arm], seed=s, events_k=e) for s, e in v.items()]
            if arm != "scratch":
                vv = f.loc[lane][col]
                c_ = (vv / base.loc[vv.index] - 1) * 100
                per += [dict(process=plab, arm=lane.removesuffix("_K20"), seed=s_, pct_change_vs_scratch=x)
                        for s_, x in c_.items()]
                diff = vv.values - base.loc[vv.index].values
                st = L.paired(diff / 1e3, f"{L.ARM_LABEL[arm]} - scratch, {plab} (k events)")
                st.update(mean_pct_change=c_.mean(), n_lower=int((diff < 0).sum()))
                stat.append(st)
    P = pd.DataFrame(per)
    for j, (col, plab, ramp) in enumerate(PROC):
        ch = np.round(P[P.process == plab].pct_change_vs_scratch.values)
        near, far = ch.max(), ch.min()
        fmt = lambda x: f"+{x:.0f}" if x > 0 else f"−{abs(x):.0f}"
        y_ = {"Tg-E": 205, "Ab-E": 225, "Di-E": 575}[plab]
        ax.plot([j - 0.12, j + 0.38], [y_ - 12, y_ - 12], color=L.REFLINE, lw=0.5)
        ax.text(j + 0.13, y_, f"{fmt(near)} to {fmt(far)}%", ha="center", va="bottom", fontsize=6, color=GT)
    ax.set_xticks(range(3))
    ax.set_xticklabels([p[1] for p in PROC])
    for t, p in zip(ax.get_xticklabels(), PROC):
        t.set_color(p[2][0])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 640)
    ax.set_ylabel("Events in fine-class\ntraining (×10³)")
    SRC["e"] = pd.DataFrame(rows)
    SRC["e_stats"] = pd.DataFrame(stat)
    SRC["e_pct_per_seed"] = P
    print("[e] " + "; ".join(f"{r['contrast']}: {r['mean_pct_change']:+.1f}% ({r['n_lower']}/5 lower)" for r in stat))


# ───────────────────────── f: differentiation events ─────────────────────────
def panel_di(ax):
    ax.set_ylim(0, 1000)
    rows, S = F6N.events_panel(ax, "di", ns.AMBER, ns.AMBER_TXT)
    ax.set_ylabel("Differentiation events\n(×10³, cumulative)")
    ax.set_xlabel("Epoch relative to the\nfirst fine-class epoch", linespacing=1.0)
    ax.set_xticks([-20, 0, 20])
    ax.set_yticks([0, 400, 800])
    SRC["f"] = pd.DataFrame(rows)
    SRC["f_share_before_first_fine_label"] = S


# ───────────────────────── key strip ─────────────────────────
def key(fig, y, H, blind):
    k = ns.GREYS[0]
    arms = [(Line2D([], [], color=k, ls="-", lw=LW), Patch(fc=k, ec=k, alpha=0.85, lw=0.8)),
            (Line2D([], [], color=k, ls="--", lw=LW), Patch(fc=k, ec=k, alpha=0.45, hatch=L.HATCH, lw=0.8)),
            (Line2D([], [], color=SCR["color"], ls="-", lw=SCR["lw"]), Patch(fc="white", ec=SCR["color"], lw=0.8))]
    labels = ["true superclasses", "random grouping", "scratch (no pre-training)"]
    if blind:
        arms.insert(2, (Line2D([], [], color=k, ls=":", lw=LW),))
        labels.insert(2, "superclass-blind task")
    fig.legend(arms, labels, loc="center left", bbox_to_anchor=(14 / 180, 1 - y / H), ncol=len(labels),
               frameon=False, fontsize=6.5, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
               handlelength=3.6, columnspacing=2.2, handletextpad=0.6)


def main():
    ns.apply()
    print(B.status())
    H, W = 122, 180
    fig = plt.figure(figsize=(ns.FULL_W, H * ns.MM))

    def box(x0, x1, top, h):
        return fig.add_axes([x0 / W, 1 - (top + h) / H, (x1 - x0) / W, h / H])

    r1, r2, rh = 20.0, 78.0, 32.0
    ax = {"a": box(16, 92, r1, rh), "b": box(126, 168, r1, rh), "c": box(16, 56, r2, rh), "d": box(72, 89, r2, rh),
          "e": box(126, 176, r2, rh)}
    panel_lead(ax["a"])
    panel_labels(ax["b"])
    panel_events(ax["c"])
    panel_di(ax["d"])
    panel_sweep(ax["e"])
    titles = {"a": "The head start is largest early\nin fine-class training",
              "b": "At five labels, the aligned\ntask leads",
              "c": "Fewer events in fine-class\ntraining after pre-training",
              "d": "Differentiation begins\nunder coarse labels",
              "e": "Every schedule of the sweep\nends at 57–59%"}
    pos = {"a": (2, r1 - 5.0), "b": (112, r1 - 5.0), "c": (2, r2 - 5.0), "d": (58, r2 - 5.0), "e": (112, r2 - 5.0)}
    for k_, (xm, ym) in pos.items():
        va = "top" if "\n" in titles[k_] else "baseline"
        y_ = ym - 2.6 if va == "top" else ym
        fig.text(xm / W, 1 - y_ / H, k_, fontsize=8, fontweight="bold", va=va)
        fig.text((xm + 4.5) / W, 1 - y_ / H, titles[k_], fontsize=7, va=va, linespacing=1.0)
    # Source Data keys follow the panel letters (panel_labels, panel_events and panel_di write c, e and f)
    remap = {"c": "b", "c_stats": "b_stats", "e": "c", "e_stats": "c_stats", "e_pct_per_seed": "c_pct_per_seed",
             "f": "d", "f_share_before_first_fine_label": "d_share_before_first_fine_label", "d": "e"}
    new = {"a": SRC.pop("a")}
    new.update({remap[k]: v for k, v in SRC.items()})
    SRC.clear(); SRC.update(new)
    key(fig, 3.0, H, blind=True)
    for f in (ns.FINAL_OUT / "source_data").glob(f"{NAME}_*.csv"):
        f.unlink()
    for p_, df in SRC.items():
        ns.source_data(df, NAME, p_)
    comp = FINAL_SD / "ed9_event_composition.csv"
    if comp.exists():
        shutil.copyfile(comp, ns.FINAL_OUT / "source_data" / f"{NAME}_event_composition.csv")
    print("saved", ns.save_final(fig, NAME, 180, ed=True))


if __name__ == "__main__":
    main()
