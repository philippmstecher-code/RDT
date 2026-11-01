#!/usr/bin/env python3
"""Fig. 4: task-general and superclass features persist.

a  construction/refinement mean-lifespan ratio per run (bars from zero, dashed parity line)
b  Spearman rho of SSI and CSI with lifespan per run (bootstrap s.e.)
c  developmental persistence of the three exclusive cohorts (mean +- s.e.m. over seeds)
d  task-general, superclass- and class-selective features per checkpoint (0-8), stacked, mean of the three
   standard runs per architecture (CENSUS_exclusive/census_exclusive_per_checkpoint.csv)
Panels a-c read analysis_outputs/paper/figures/fig4_lifespan_data.json (fig4_lifespan.py); the SAI-dependent values
are replaced by the stored-SAI values of THRESHOLD_impact/fig4_stored_sai.json (scripts/paper/fig4_stored_sai.py).
draw_d_nest and draw_d_ablation are imported by interventions/ed6.py.
Usage: python scripts/paper/figures/fig4.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nmi_style as ns  # noqa: E402
from core_runs_lib import MM, add_axes_mm, apply_style  # noqa: E402
import suppression_panels as E9  # noqa: E402  (loaders, bar idiom and per-window suppression deltas)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

NAME = "fig4"
W, H = 180.0, 148.0
CACHE = ns.OUT / "fig4_lifespan_data.json"
# Every run's stored thresholds for all three indices (task-general = stored SAI). The stored-SAI values come from
# scripts/paper/fig4_stored_sai.py (same records, cohort rule, bootstrap and seeds as fig4_lifespan.py).
STORED_SAI = ns.OUT.parent / "THRESHOLD_impact" / "fig4_stored_sai.json"
ARCHS = ("ResNet-18", "ViT-Small", "CCT-7")
SHORT = {"ResNet-18": "R18", "ViT-Small": "ViT", "CCT-7": "CCT"}
ERR = ns.REF_DARK
REF = E9.REF
SSI_C, CSI_C = ns.INDEX["SSI"], ns.INDEX["CSI"]
BIG = 7.0
COHORT_LAB = {"SAI": "Task-general (SAI)", "SSI": "Superclass-selective (SSI)", "CSI": "Class-selective (CSI)"}


def p_stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def sep_lines(ax):
    for x in (2.5, 5.5):
        ax.axvline(x, color="#D1D5DB", lw=0.6, ls="--", zorder=0)


def run_ticks(ax, runs):
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([f"{SHORT[r['arch']]}-{r['seed']}" for r in runs], rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.set_xlim(-0.6, len(runs) - 0.4)


def draw_a(ax, runs):
    """Run-wise ratios as bars from zero in the architecture colours, value on top, dashed parity line."""
    x = np.arange(len(runs))
    ratios = np.array([r["ratio"] for r in runs])
    ax.bar(x, ratios, width=0.72, color=[ns.ARCH[r["arch"]] for r in runs], alpha=ns.FILL_ALPHA, lw=0, zorder=2)
    for xi, v in zip(x, ratios):
        ax.text(xi, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    ax.axhline(1.0, color="black", lw=0.7, ls="--", zorder=3)
    sep_lines(ax)
    run_ticks(ax, runs)
    ax.set_ylim(0, 2.4)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
    ax.set_ylabel("Mean lifespan ratio")
    ax.set_title("Construction / refinement feature lifespan")


def draw_b(ax, runs):
    x = np.arange(len(runs))
    w = 0.36
    lo_all = []
    for k, off, lab in (("ssi", -w / 2, "SSI–lifespan"), ("csi", w / 2, "CSI–lifespan")):
        col = ns.INDEX[k.upper()]
        rr = np.array([r["rho"][k]["rho"] for r in runs])
        se = np.array([r["rho"][k]["se"] for r in runs])
        ax.bar(x + off, rr, width=w, color=col, alpha=ns.FILL_ALPHA, lw=0, label=lab,
               yerr=se, ecolor=ERR, capsize=2.0, error_kw={"linewidth": 0.6, "capthick": 0.7})
        for i, r in enumerate(runs):
            s = p_stars(r["rho"][k]["p"])
            if k == "ssi":
                ax.text(x[i] + off, max(rr[i] + se[i], 0) + 0.008, s, ha="center", va="bottom", fontsize=6)
            else:
                ax.text(x[i] + off, min(rr[i] - se[i], 0) - 0.006, s, ha="center", va="top", fontsize=6)
            lo_all.append(min(rr[i] - se[i], 0))
    ax.axhline(0, color="black", lw=0.5)
    sep_lines(ax)
    run_ticks(ax, runs)
    hi = max(max(r["rho"]["ssi"]["rho"] + r["rho"]["ssi"]["se"] for r in runs), 0.05)
    ax.set_ylim(min(lo_all) - 0.07, hi + 0.04)
    ax.set_ylabel("Spearman rho")
    ax.set_title("Selectivity–lifespan correlation")
    n_pos = sum(r["rho"]["ssi"]["rho"] > 0 for r in runs)
    n_neg = sum(r["rho"]["csi"]["rho"] < 0 for r in runs)
    n_neg_sig = sum(r["rho"]["csi"]["rho"] < 0 and r["rho"]["csi"]["p"] < 0.05 for r in runs)
    ax.legend(loc="lower left", fontsize=6, handlelength=1.0, handleheight=0.7, borderaxespad=0.2, borderpad=0.3,
              labelspacing=0.25, handletextpad=0.4, frameon=True, framealpha=0.9, edgecolor="#D1D5DB")
    print(f"[b] SSI-lifespan positive {n_pos}/{len(runs)}, CSI-lifespan negative {n_neg}/{len(runs)} ({n_neg_sig} sig)")


def draw_c(ax, runs):
    wc = 0.22
    offs = {"SAI": -wc, "SSI": 0.0, "CSI": wc}
    top = 0
    ratios = {}
    for ai, arch in enumerate(ARCHS):
        rs = [r for r in runs if r["arch"] == arch]
        for k in ("SAI", "SSI", "CSI"):
            v = np.array([r["dev"][k] for r in rs])
            m, sem = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
            xc = ai + offs[k]
            ax.bar(xc, m, width=wc, color=ns.INDEX[k], alpha=ns.FILL_ALPHA, lw=0)
            ax.errorbar(xc, m, yerr=sem, color=ERR, capsize=2.0, lw=0.6, capthick=0.7, zorder=4)
            ax.scatter(np.full(len(v), xc), v, s=10, color=ns.INDEX[k], edgecolors="white", linewidths=0.5, zorder=5)
            top = max(top, v.max(), m + sem)
        sai_m = np.mean([r["dev"]["SAI"] for r in rs])
        csi_m = np.mean([r["dev"]["CSI"] for r in rs])
        ratios[arch] = sai_m / csi_m
        gtop = max(max(r["dev"][k] for r in rs) for k in ("SAI", "SSI", "CSI"))
        ax.text(ai, gtop + 0.015, f"SAI/CSI={sai_m / csi_m:.1f}\u00d7", ha="center", va="bottom", fontsize=6,
                color="black")
    ax.set_xticks(range(len(ARCHS)))
    ax.set_xticklabels(ARCHS)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 0.62)
    ax.legend(handles=[Patch(fc=ns.INDEX[k], alpha=ns.FILL_ALPHA, label=k) for k in ("SAI", "SSI", "CSI")],
              loc="upper right", fontsize=6.5, handlelength=1.2, handleheight=0.8, frameon=False)
    ax.set_ylabel("Developmental persistence")
    ax.set_title("Persistence by cohort")
    return ratios


def draw_d_nest(ax):
    """Nesting of class-selective decoders within their own superclass (DEPENDENCY)."""
    t = pd.read_csv((E9.RESULTS / "DEPENDENCY") / "T1_T4_per_seed.csv")
    rng = np.random.default_rng(2)
    rows, vals = [], []
    for test, sae in (("T4_avgpool_r_other", "pooled-layer SAE"), ("T4_layer4_r_other", "independent SAE")):
        d = t[t.test == test].sort_values("seed")
        vals.append(d["median"].values)
        rows += [dict(sae=sae, seed=int(sd), median_projection_ratio=v) for sd, v in zip(d.seed, d["median"])]
    xs = [0, 1.0]
    st = E9.bars(ax, xs[:1], vals[:1], CSI_C, False, rng, width=0.7) + E9.bars(ax, xs[1:], vals[1:], CSI_C, True, rng,
                                                                               width=0.7)
    for x, (m, lo, hi), v in zip(xs, st, vals):
        ax.text(x, max(hi, v.max()) + 0.08, f"{m:.1f}\u00d7", fontsize=BIG, fontweight="bold", color=ERR,
                ha="center", va="bottom")
    ax.axhline(1, color=REF, ls="--", lw=0.6, zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Pooled-\nlayer\nSAE", "Indep-\nendent\nSAE"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 3.6)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_ylabel("Projection ratio, own vs\nother superclasses")
    ax.set_title("Class-selective decoders\nwithin own superclass", fontsize=7)
    sd = pd.DataFrame(rows)
    for sae, (m, lo, hi) in zip(("pooled-layer SAE", "independent SAE"), st):
        sd.loc[sd.sae == sae, ["mean", "ci95_lo", "ci95_hi"]] = m, lo, hi
    return sd, st


def draw_d_ablation(ax):
    """Cross-layer ablation of superclass-selective features (DEPENDENCY)."""
    t = pd.read_csv((E9.RESULTS / "DEPENDENCY") / "T1_ratio_by_target_category.csv")
    seeds = [str(sd) for sd in E9.SEEDS]
    rng = np.random.default_rng(1)
    rows, st = [], {}
    pos = {("SSI", "avgpool"): -0.36, ("SSI", "layer4"): 0.36, ("CSI", "avgpool"): 1.64, ("CSI", "layer4"): 2.36}
    for (cat, layer), x in pos.items():
        r = t[(t.set == "SSIq") & (t.layer == layer) & (t["cat"] == cat)].iloc[0]
        v = r[seeds].astype(float).values
        col = SSI_C if cat == "SSI" else CSI_C
        (m, lo, hi), = E9.bars(ax, [x], [v], col, layer == "layer4", rng, width=0.68)
        ax.text(x, max(hi, v.max()) + 0.035, f"{m:.2f}", fontsize=BIG, fontweight="bold", color=ERR, ha="center",
                va="bottom", bbox=dict(fc="white", ec="none", pad=0.3), zorder=5)
        sae = "pooled-layer SAE" if layer == "avgpool" else "independent SAE"
        tgt = "superclass-selective" if cat == "SSI" else "class-selective"
        st[(tgt, sae)] = (m, lo, hi)
        rows += [dict(target=tgt, sae=sae, seed=int(sd), ratio=float(r[sd]), mean=m, ci95_lo=lo, ci95_hi=hi)
                 for sd in seeds]
    ax.axhline(1, color=REF, ls="--", lw=0.6, zorder=0)
    ax.text(2.8, 1.02, "matched random", fontsize=6, color=REF, ha="right", va="bottom")
    ax.set_xticks([0, 2])
    ax.set_xticklabels(["Superclass-\nselective", "Class-\nselective"])
    ax.set_xlim(-0.8, 2.85)
    ax.set_ylim(0, 1.5)
    ax.set_yticks([0, 0.5, 1.0, 1.5])
    ax.set_xlabel("Targets in the next block", labelpad=2)
    ax.set_ylabel("Ablation effect relative\nto matched random sets")
    ax.set_title("Removing superclass-selective\nfeatures one block earlier", fontsize=7)
    return pd.DataFrame(rows), st


def draw_e(ax, cases):
    """Cost of suppressing the superclass code, first three epochs of each window (suppression_panels deltas)."""
    rng = np.random.default_rng(4)
    rows, lows = [], []
    off, wid = 0.23, 0.4
    heads = ["Last block\nepochs 1\u201310", "Last block\nepochs 11\u201320", "Last block\nepochs 21\u201330",
             "Every layer\nepochs 1\u201310"]
    centres = [0, 1.1, 2.2, 3.5]
    for c, xc, hd in zip(cases, centres, heads):
        f_sham = c["D"]["superclass"][:, :3].mean(1)
        f_rand = f_sham - c["D"]["random"][:, :3].mean(1)
        (ms, los, his), = E9.bars(ax, [xc - off], [f_sham], E9.ACC_HUE, False, rng, width=wid, dot_s=12)
        (mr, lor, hir), = E9.bars(ax, [xc + off], [f_rand], E9.ACC_HUE, True, rng, width=wid, dot_s=12)
        for x, m, lo, v in ((xc - off, ms, los, f_sham), (xc + off, mr, lor, f_rand)):
            y = min(lo, v.min()) - 0.3
            ax.text(x, y, E9.minus(f"{m:.1f}"), fontsize=BIG, fontweight="bold", color=ERR, ha="center", va="top")
            lows.append(y)
        ax.text(xc, 0.55, hd, fontsize=6.5, color=ERR, ha="center", va="bottom", linespacing=1.05)
        rows.append(dict(window=c["tag"], epochs=f"{c['st']}-{c['st'] + 2}", superclass_minus_sham=ms,
                         sham_ci95_lo=los, sham_ci95_hi=his, sham_n_negative=int((f_sham < 0).sum()),
                         superclass_minus_random=mr, random_ci95_lo=lor, random_ci95_hi=hir,
                         random_n_negative=int((f_rand < 0).sum()),
                         **{f"seed{sd}_minus_sham": v for sd, v in zip(c["seeds"], f_sham)},
                         **{f"seed{sd}_minus_random": v for sd, v in zip(c["seeds"], f_rand)},
                         reference=("sham branched at epoch 10" if c["st"] == 21 else "sham, same branch")))
    ax.axhline(0, color=REF, ls="--", lw=0.6, zorder=0)
    ax.axvline(2.85, color="#D1D5DB", lw=0.6, ls="--", zorder=0)
    ax.set_ylim(min(lows) - 1.6, 2.6)
    ax.set_yticks([-15, -10, -5, 0])
    ax.set_xlim(-0.5, 4.0)
    ax.set_xticks([x for xc in centres for x in (xc - off, xc + off)])
    ax.set_xticklabels(["vs\nsham", "vs\nrandom"] * len(centres), fontsize=6.5)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("Change in fine-class accuracy,\nfirst 3 epochs of suppression (pp)")
    ax.set_title("Suppressing the superclass code during training", fontsize=7, pad=4)
    return pd.DataFrame(rows)


# routing (targeted ablation of one superclass's features)
ROUTE = E9.RESULTS / "ROUTING"
TF_FINAL = {"ViT-Small": "8", "CCT-7": "9"}          # checkpoint labels of the final epoch (100 and 50)


def routing_rows():
    """All checkpoints: specificity S and its matched control (pp), per architecture, seed and checkpoint."""
    rows, ngrp = [], {}
    # ResNet-18, conservative variant (exact subtraction of the features' reconstruction contributions), epochs 5-30
    for fn in ("routing_resnet_clean_ep5-10-15-20-25.json", "routing_resnet_clean_ep30.json"):
        for key, v in json.loads((ROUTE / fn).read_text()).items():
            seed, ep = (int(t) for t in key.split("|"))
            r = v["true"]["part2"]
            ngrp[(seed, ep)] = int(sum(n > 0 for n in r["n_ab_per_group"]))
            rows.append(dict(arch="ResNet-18", method="subtraction", seed=seed, checkpoint=str(ep), epoch=ep,
                             final=False, S_pp=100 * r["S"], S_ctrl_pp=100 * r["S_ctrl"],
                             n_superclasses_with_features=ngrp[(seed, ep)],
                             n_superclass_selective_features=int(sum(r["n_ab_per_group"]))))
    # ResNet-18, mean-preserving projection (as in the transformers), epoch 30
    for seed, r in json.loads((ROUTE / "routing_resnet_projection.json").read_text()).items():
        rows.append(dict(arch="ResNet-18", method="projection", seed=int(seed), checkpoint="30", epoch=30, final=True,
                         S_pp=100 * r["S"], S_ctrl_pp=100 * r["S_ctrl"],
                         n_superclasses_with_features=ngrp[(int(seed), 30)],
                         n_superclass_selective_features=int(r["n_ab"])))
    epochs = {"CCT-7": {k: v["epoch"] for k, v in json.loads((ROUTE / "routing_cct7.json").read_text())["runs"].items()},
              "ViT-Small": {"1": 1, "4": 5, "8": 100}}
    for key, r in json.loads((ROUTE / "routing_tf.json").read_text()).items():
        lane, ck = key.split("|")
        arch, seed = lane.split("__")[1], int(lane.split("__s")[-1])
        ep = epochs[arch][key] if arch == "CCT-7" else epochs[arch][ck]
        rows.append(dict(arch=arch, method="projection", seed=seed, checkpoint=ck, epoch=ep, final=ck == TF_FINAL[arch],
                         S_pp=100 * r["S"], S_ctrl_pp=100 * r["S_ctrl"],
                         n_superclasses_with_features=int(sum(n > 0 for n in r["n_ab_per_group"])),
                         n_superclass_selective_features=int(r["n_ab"])))
    return pd.DataFrame(rows).sort_values(["arch", "method", "seed", "epoch"]).reset_index(drop=True)


def draw_f(ax):
    """S at the final checkpoint per architecture (filled) vs the activation-mass-matched control (hatched)."""
    R = routing_rows()
    F = R[R.final]
    rng = np.random.default_rng(6)
    off, wid = 0.2, 0.36
    tops = []
    for i, arch in enumerate(ARCHS):
        g = F[F.arch == arch].sort_values("seed")
        col = ns.ARCH[arch]
        for x, v, light in ((i - off, g.S_pp.values, False), (i + off, g.S_ctrl_pp.values, True)):
            m = v.mean()
            if light:
                ax.bar(x, m, width=wid, color=col, alpha=0.25, lw=0, zorder=1)
                ax.bar(x, m, width=wid, facecolor="none", edgecolor=col, lw=0.8, hatch="//////", zorder=1)
            else:
                ax.bar(x, m, width=wid, color=col, alpha=E9.BAR_A, lw=0, zorder=1)
            ax.scatter(x + rng.uniform(-0.2, 0.2, len(v)) * wid, v, s=9, color=E9.DOT, lw=0, zorder=3)
            hi = v.max()
            if len(v) >= 3:
                mm, lo, hi_ci = E9.tci(v)
                ax.errorbar(x, mm, yerr=[[mm - lo], [hi_ci - mm]], fmt="none", ecolor=ERR, elinewidth=0.9,
                            capsize=2.5, capthick=0.9, zorder=4)
                hi = max(hi, hi_ci)
            lab = f"{m:.1f}" if not light else E9.minus(f"{m:.1f}") if m < -0.05 else "0.0"
            ax.text(x, max(hi, 0) + 0.6, lab, fontsize=BIG if not light else 6.5, color=ERR, ha="center",
                    va="bottom")
            tops.append(max(hi, 0))
    ax.axhline(0, color=REF, ls="--", lw=0.6, zorder=0)
    ax.set_xticks(range(len(ARCHS)))
    ax.set_xticklabels([f"{a}\n(n = {int((F.arch == a).sum())})" for a in ARCHS], fontsize=6.5)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlim(-0.55, len(ARCHS) - 0.45)
    ax.set_ylim(-2, max(tops) + 16)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_ylabel("Specificity S (pp)")
    ax.set_title("Removing one superclass\u2019s features\nmisroutes only that superclass", fontsize=7)
    from matplotlib.patches import Patch as _P
    h = [_P(facecolor="#9CA3AF", alpha=E9.BAR_A, lw=0, label="superclass-selective features"),
         _P(facecolor="white", edgecolor="#6B7280", hatch="//////", lw=0.8, label="matched control")]
    ax.legend(handles=h, loc="upper left", handlelength=1.3, handleheight=0.9, borderaxespad=0.1,
              labelspacing=0.3, fontsize=6.5)
    return R


# durability after 50 and 200 epochs
G_FULL, G_LIGHT = ns.BAR_ALPHA_CORE, 0.35
PARITY = "#6B7280"


def draw_durability(ax, E):
    """Task-general survival advantage and superclass/class lifespan ratio, 50 (full) vs 200 epochs (light)."""
    w = 0.36
    groups = (("tg_surv_adv", ns.INDEX["SAI"], "Task-general vs\nall other features\n(survival)"),
              ("ssi_csi_life_methods", ns.INDEX["SSI"], "Superclass- vs\nclass-selective\n(lifespan)"))
    rows = []
    for gi, (k, col, _) in enumerate(groups):
        for run, off, al in (("50ep", -w / 2, G_FULL), ("200ep", w / 2, G_LIGHT)):
            v, (lo, hi) = E[run][k], E[run][k + "_ci95"]
            xc = gi + off
            ax.bar(xc, v, width=w, color=col, alpha=al, lw=0)
            ax.errorbar(xc, v, yerr=[[v - lo], [hi - v]], color=ERR, capsize=2.0, lw=0.6, capthick=0.7, zorder=4)
            ax.text(xc, hi + 0.05, f"{v:.1f}\u00d7", ha="center", va="bottom", fontsize=BIG, color=ERR)
            ax.text(xc, 0.08, "50 ep" if run == "50ep" else "200 ep", ha="center", va="bottom", fontsize=6,
                    color="white" if run == "50ep" else ns.REF_DARK, rotation=90)
            rows.append({"measure": k, "run": run, "value": v, "ci95_low": lo, "ci95_high": hi,
                         "p_200_minus_50": E["diff_200_minus_50"][k]["p_two_sided"]})
        p = E["diff_200_minus_50"][k]["p_two_sided"]
        yb = max(E["50ep"][k + "_ci95"][1], E["200ep"][k + "_ci95"][1]) + 0.5
        ax.plot([gi - w / 2, gi - w / 2, gi + w / 2, gi + w / 2], [yb - 0.06, yb, yb, yb - 0.06], color=ERR, lw=0.6)
        ax.text(gi, yb + 0.04, f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}", ha="center", va="bottom", fontsize=6)
    ax.axhline(1.0, color=PARITY, ls="--", lw=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([g[2] for g in groups])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 4.6)
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.set_ylabel("Ratio (ResNet-18)")
    ax.set_title("Durability after 50 and 200 epochs")
    ax.text(1.58, 1.0, "parity", ha="left", va="center", fontsize=6, color=PARITY, clip_on=False)
    s50, s200 = E["50ep"]["survival_pct"], E["200ep"]["survival_pct"]
    ax.text(0.03, 0.98, f"Overall survival: {s50:.1f}% (50 ep),\n{s200:.1f}% (200 ep)", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.5, color=ERR, linespacing=1.2)
    for run, sv in (("50ep", s50), ("200ep", s200)):
        rows.append({"measure": "survival_pct", "run": run, "value": sv, "ci95_low": E[run]["survival_pct_ci95"][0],
                     "ci95_high": E[run]["survival_pct_ci95"][1],
                     "p_200_minus_50": E["diff_200_minus_50"]["survival_pct"]["p_two_sided"]})
    return pd.DataFrame(rows)



CENSUS = ns.OUT.parent / "CENSUS_exclusive" / "census_exclusive_per_checkpoint.csv"
CENSUS_LAST = 8   # last checkpoint shown; full runs in Extended Data Fig. 1d-f
CENSUS_LANES = {"ResNet-18": "ResNet18-100class-10ms-seed", "ViT-Small": "VitSmall-100class-10ms-seed",
                "CCT-7": "CCT7-100class-10ms-seed"}


def draw_census(axes):
    """Panel d: stacked counts of task-general (SAI), superclass-selective (SSI) and class-selective
    (CSI) features per checkpoint, mean of the three standard runs per architecture, one small panel per architecture
    stacked in a column; exclusive categories, each run's stored thresholds, duplicate final checkpoint removed (as
    Extended Data Fig. 1d-f)."""
    d = pd.read_csv(CENSUS)
    d = d[(d.variant == "dedup_stored") & (d.checkpoint != "terminal")].copy()
    d["checkpoint"] = d.checkpoint.astype(int)
    rows = []
    for i, (ax, arch) in enumerate(zip(axes, ARCHS)):
        g = d[d.lane.isin([CENSUS_LANES[arch] + s for s in ("42", "137", "256")])]
        assert g.lane.nunique() == 3, arch
        m = g.groupby("checkpoint")[["n_tg", "n_ab_ex", "n_di_ex"]].mean()
        m = m[m.index <= CENSUS_LAST]   # checkpoints 0-8, the range all three architectures share
        ax.stackplot(m.index, m.n_tg, m.n_ab_ex, m.n_di_ex, colors=[ns.INDEX[k] for k in ("SAI", "SSI", "CSI")],
                     alpha=ns.FILL_ALPHA, edgecolor="white", lw=0.6)
        ax.set_title(arch, fontsize=7, pad=2)
        ax.set_xlim(-0.3, CENSUS_LAST + 0.3)
        ax.set_xticks(range(0, CENSUS_LAST + 1))
        ax.set_ylim(0, None)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
        ax.tick_params(labelsize=6, pad=1.5)
        if i == len(axes) - 1:
            ax.set_xlabel("Training checkpoint")
        else:
            ax.set_xticklabels([])
        if i == 1:
            ax.set_ylabel("Features above\nadaptive threshold")
        rows += [dict(arch=arch, checkpoint=int(c), task_general_SAI=r.n_tg, superclass_selective_SSI=r.n_ab_ex,
                      class_selective_CSI=r.n_di_ex, n_runs=3) for c, r in m.iterrows()]
    axes[0].legend(handles=[Patch(fc=ns.INDEX[k], alpha=ns.FILL_ALPHA, label=k) for k in ("SAI", "SSI", "CSI")],
                   loc="lower left", bbox_to_anchor=(-0.02, 1.2), ncol=3, fontsize=6.5, handlelength=1.2,
                   handleheight=0.8, columnspacing=1.0, frameon=False, borderaxespad=0)
    return pd.DataFrame(rows)


def apply_stored_sai(D):
    """Replace the SAI-dependent values (panel a and the 50/200-epoch durability) with the stored-SAI values of
    fig4_stored_sai.json. Panels b (threshold-free) and c (stored thresholds) are unchanged."""
    S = json.loads(STORED_SAI.read_text())
    by = {(r["arch"], r["seed"]): r for r in S["fig4a"]}
    for r in D["runs"]:
        s = by[(r["arch"], r["seed"])]
        assert abs(s["sai_0.9"]["ratio"] - r["ratio"]) < 1e-9, "fig4_stored_sai.json does not match the cache"
        st = s["sai_stored"]
        r.update(ratio=st["ratio"], n_coarse=st["n_coarse"], n_fine=st["n_fine"], thr_sai=s["thr_sai"],
                 hr_birth_stratified=st["hr_birth"],
                 hr_birth_stratified_p=st["hr_birth_p"])
    E, F = D["ep200"], S["fig4d"]
    for run in ("50ep", "200ep"):
        assert abs(E[run]["survival_pct"] - F[run]["survival_pct"]) < 1e-9
        E[run]["thr"] = F[run]["thr"]
        E[run]["tg_surv_adv"], E[run]["tg_surv_adv_ci95"] = F[run]["tg_surv_adv"], F[run]["tg_surv_adv_ci95"]
        E[run]["ssi_csi_life_methods"] = F[run]["ssi_csi_life"]
        E[run]["ssi_csi_life_methods_ci95"] = F[run]["ssi_csi_life_ci95"]
    E["diff_200_minus_50"]["tg_surv_adv"] = F["diff_tg_surv_adv"]
    E["diff_200_minus_50"]["ssi_csi_life_methods"] = F["diff_ssi_csi_life"]
    E["diff_200_minus_50"]["survival_pct"] = F["diff_survival_pct"]
    return D


def main():
    apply_style()
    D = json.loads(CACHE.read_text())
    assert all("ratio_overlap" in r for r in D["runs"]), "stale cache"
    D = apply_stored_sai(D)
    runs = D["runs"]
    fig = plt.figure(figsize=(W * MM, H * MM))
    # row 1: a | b; row 2: c | d (census, one small panel per architecture)
    y1, h1 = 96, 40
    ax_a = add_axes_mm(fig, 16, y1, 62, h1)
    ax_b = add_axes_mm(fig, 104, y1, 72, h1)
    draw_a(ax_a, runs)
    draw_b(ax_b, runs)
    ax_c = add_axes_mm(fig, 16, 13, 52, 51)
    ratios = draw_c(ax_c, runs)
    ax_census = [add_axes_mm(fig, 104, y, 72, 13) for y in (51, 32, 13)]
    sd_census = draw_census(ax_census)
    for ax, k, x in ((ax_a, "a", -0.2), (ax_b, "b", -0.18), (ax_c, "c", -0.25)):
        ns.panel(ax, k, x=x, y=1.17)
    ns.panel(ax_census[0], "d", x=(89 - 104) / 72, y=1 + (68 + 6.8 - 64) / 13)
    for _ax in (ax_a, ax_b, ax_c, ax_census[0]):
        ns.rdt_tag(_ax)
    ns.source_data(pd.DataFrame([{"arch": r["arch"], "seed": r["seed"], "lifespan_ratio": r["ratio"],
                                  "n_task_general_or_superclass": r["n_coarse"], "n_class_selective": r["n_fine"],
                                  "stored_sai_threshold": r["thr_sai"],
                                  "hr_birth_stratified_coarse_vs_fine": r["hr_birth_stratified"],
                                  "hr_birth_stratified_p": r["hr_birth_stratified_p"]}
                                 for r in runs]), NAME, "a")
    ns.source_data(pd.DataFrame([{"arch": r["arch"], "seed": r["seed"], "index": k.upper(), "spearman_rho": r["rho"][k]["rho"],
                                  "bootstrap_se": r["rho"][k]["se"], "p": r["rho"][k]["p"], "n_records": r["n_records"]}
                                 for r in runs for k in ("ssi", "csi")]), NAME, "b")
    ns.source_data(pd.DataFrame([{"arch": r["arch"], "seed": r["seed"], "cohort": k, "cohort_name": COHORT_LAB[k],
                                  "developmental_persistence": r["dev"][k], "n_features": r["dev"]["n_" + k]}
                                 for r in runs for k in ("SAI", "SSI", "CSI")]), NAME, "c")
    ns.source_data(sd_census, NAME, "d")
    rr = [r["ratio"] for r in runs]
    print(f"4a ratio range {min(rr):.3f}-{max(rr):.3f}, >1 in {sum(v > 1 for v in rr)}/9")
    print("4c SAI/CSI:", {k: round(v, 2) for k, v in ratios.items()})
    out = ns.save_final(fig, NAME, 180)
    print("saved", out)


if __name__ == "__main__":
    main()
