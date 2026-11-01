#!/usr/bin/env python3
"""Feature lifespan and scaffold persistence: computes the per-run values that Fig. 4 (fig4.py) reads from
fig4_lifespan_data.json and draws them as an overview figure, panels a-e.

Alive features only (active on >= 1% of samples); the duplicated terminal checkpoint is excluded, so the last real
milestone is the right-censoring point; ResNet-18 layer4/avgpool (identical activations) are counted once. Stored
per-run thresholds; mutually exclusive categories:
  task-general = SAI > t_sai; superclass-selective = SSI > t_ssi and CSI <= t_csi and not task-general;
  class-selective = CSI > t_csi and not task-general.
The overlapping definition (SSI > t_ssi regardless of CSI) is also computed (keys *_overlap, dev_priority) and its
ratio is checked against PERSIST_robust/tables/decomposition.csv (step S3_dedup_avgpool).

a  coarse (task-general with SAI > 0.9, or superclass-selective) / fine (class-selective) mean-lifespan ratio per run
b  Spearman rho of SSI and CSI with lifespan per run; bootstrap SE; stars by p
c  developmental persistence of features alive at the last real checkpoint by category; mean +- SEM over 3 seeds
d  census of alive features per checkpoint in the three categories, stacked, mean over 3 seeds, common 0-10 axis
e  50 vs 200 epochs (ResNet-18 seed 42; persist_200ep.RUNS): task-general survival advantage and superclass- /
   class-selective lifespan ratio; record-level bootstrap 95% CIs (2,000 resamples) and p of the difference

Writes analysis_outputs/paper/figures/fig4_lifespan.{pdf,png}, fig4_lifespan_data.json (cache) and
analysis_outputs/paper/PERSIST_robust/fig4_methods_rule.json.
Usage: python scripts/paper/figures/fig4_lifespan.py [--replot]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import nmi_style  # noqa: E402
from nmi_style import ARCH, FILL_ALPHA, FULL_W, INDEX, NEUTRAL, PROC, panel, save  # noqa: E402
from persist_analyze import C, cohort_masks, cohort_stats, strat_logrank  # noqa: E402
from persist_lib import CORE_LANES, load_lane, records_real, sel_arrays, terminal_persistence  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

TABLES = HERE.parents[2] / "analysis_outputs" / "paper" / "PERSIST_robust" / "tables"
ARCHS = ("ResNet-18", "ViT-Small", "CCT-7")
SHORT = {"ResNet-18": "R18", "ViT-Small": "ViT", "CCT-7": "CCT"}
ERR = "#374151"
N_BOOT = 1000
CACHE = nmi_style.OUT / "fig4_lifespan_data.json"
SUMM_OUT = HERE.parents[2] / "analysis_outputs" / "paper" / "PERSIST_robust" / "fig4_methods_rule.json"


# computation
def p_stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def boot_se_spearman(x, y, rng):
    n = len(x)
    vals = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.integers(0, n, n)
        vals[b] = stats.spearmanr(x[i], y[i])[0]
    return float(np.nanstd(vals, ddof=1))


def census_counts(d, res, thr):
    """Per real checkpoint over alive features of the census layers:
    exclusive (Methods rule) (n_tg, n_ab, n_di) and overlapping (n_tg, n_ab, n_di)."""
    labels = d["labels"]
    real = labels[:-1]
    trans = [f"{a}->{b}" for a, b in zip(real[:-1], real[1:])]
    layers = d["layers"]
    if d["arch"] == "ResNet-18":
        layers = [l for l in layers if l != "layer4"]  # layer4 == avgpool input; keep avgpool
    out, out_ov = [], []
    for b, lab in enumerate(real):
        tg = ab = di = ab_ov = di_ov = 0
        for L in layers:
            per_t = d["layer_data"][L]
            alive = per_t[trans[b]]["alive_a"] if b < len(trans) else per_t[trans[-1]]["alive_b"]
            ssi, csi, sai = sel_arrays(res["selectivity"], lab, L, len(alive))
            al = alive.astype(bool)
            t = al & (sai > thr["sai"])
            s = al & (ssi > thr["ssi"])
            c = al & (csi > thr["csi"])
            tg += int(t.sum())
            ab += int((s & ~c & ~t).sum())
            di += int((c & ~t).sum())
            ab_ov += int((s & ~t).sum())
            di_ov += int(c.sum())
        out.append((tg, ab, di))
        out_ov.append((tg, ab_ov, di_ov))
    return out, out_ov


def dev_persistence(T, thr, rule):
    """Fig. 4c cohorts on features alive at the last real checkpoint.
    rule 'methods': SAI = SAI > t; SSI = SSI > t & CSI <= t & not SAI; CSI = CSI > t & not SAI.
    rule 'priority': SAI > SSI > CSI priority (SSI & CSI both above threshold -> SSI)."""
    ssi, csi, sai, pers = T[:, 2], T[:, 3], T[:, 4], T[:, 1]
    A, S, Cs = sai > thr["sai"], ssi > thr["ssi"], csi > thr["csi"]
    if rule == "methods":
        B, Cc = S & ~Cs & ~A, Cs & ~A
    else:
        B, Cc = S & ~A, Cs & ~A & ~S
    return {"SAI": float(pers[A].mean()), "SSI": float(pers[B].mean()), "CSI": float(pers[Cc].mean()),
            "n_SAI": int(A.sum()), "n_SSI": int(B.sum()), "n_CSI": int(Cc.sum())}


def compute():
    rng = np.random.default_rng(0)
    runs = []
    for lane, arch, seed in CORE_LANES:
        print(f"== {arch} s{seed}", flush=True)
        d, res = load_lane(lane)
        at = res["metadata"]["analysis_thresholds"]
        stored = {"sai": at["sai_task_general_thresh"], "ssi": at["ssi_abstraction_thresh"],
                  "csi": at["csi_differentiation_thresh"]}
        R = records_real(d, res, "paper", dedup=True)
        life = R[:, C["life"]]
        thr4a = {"sai": 0.9, "ssi": stored["ssi"], "csi": stored["csi"]}
        # (a) Methods rule (exclusive) and overlapping cohorts (comparison / validation)
        st = cohort_stats(R, thr4a, "methods")
        st_ov = cohort_stats(R, thr4a, "overlap")
        # birth-only stratified log-rank with the stored thresholds (comparison only)
        mc = (R[:, C["sai"]] > 0.9) | (R[:, C["ssi"]] > stored["ssi"])
        mf = R[:, C["csi"]] > stored["csi"]
        _, hr_p_ov, hr_ov = strat_logrank(life, R[:, C["cens"]], mc & ~mf, mf & ~mc, R[:, C["birth"]])
        mm = cohort_masks(R, thr4a, "methods")
        _, hr_p, hr = strat_logrank(life, R[:, C["cens"]], mm["coarse"], mm["fine"], R[:, C["birth"]])
        # (b) Spearman
        rho = {}
        for k in ("ssi", "csi"):
            r, p = stats.spearmanr(R[:, C[k]], life)
            rho[k] = {"rho": float(r), "p": float(p), "se": boot_se_spearman(R[:, C[k]], life, rng)}
        # (c) developmental persistence, exclusive cohorts (stored thresholds)
        T = terminal_persistence(d, res, "paper", dedup=True)
        dev = dev_persistence(T, stored, "methods")
        dev_ov = dev_persistence(T, stored, "priority")
        # (d) census
        cen, cen_ov = census_counts(d, res, stored)
        runs.append({"arch": arch, "seed": seed, "n_records": int(len(R)), "ratio": float(st["ratio"]),
                     "n_coarse": st["n_coarse"], "n_fine": st["n_fine"],
                     "ratio_overlap": float(st_ov["ratio"]), "n_coarse_overlap": st_ov["n_coarse"],
                     "n_fine_overlap": st_ov["n_fine"],
                     "ratio_birth_layer_matched": float(st["ratio_birth_layer_matched"]),
                     "lr_hr_layer_birth": float(st["lr_hr"]), "lr_p_layer_birth": float(st["lr_p"]),
                     "rho": rho, "dev": dev, "dev_priority": dev_ov, "census": cen, "census_overlap": cen_ov,
                     "hr_birth_stored": float(hr), "hr_birth_stored_p": float(hr_p),
                     "hr_birth_stored_overlap": float(hr_ov), "hr_birth_stored_overlap_p": float(hr_p_ov),
                     "labels": d["labels"][:-1], "thr": stored})
    return {"runs": runs}


EP_BOOT = 2000


def _ep200_vec(R, thr):
    """(e) statistics of one record set (records_real, 'paper'; t_sai = 0.9, stored t_ssi / t_csi):
    overall survival %, task-general survival advantage (survival-to-final rate of SAI > 0.9 records / that of
    all other records), superclass-/class-selective mean-lifespan ratio under the Methods rule (exclusive) and
    under the SAI > SSI > CSI priority rule (the 'excl' keys of ep200.json)."""
    life, cens = R[:, C["life"]], R[:, C["cens"]]
    ssi, csi, sai = R[:, C["ssi"]], R[:, C["csi"]], R[:, C["sai"]]
    A, S, Cs = sai > 0.9, ssi > thr["ssi"], csi > thr["csi"]
    ratio = lambda x, a, b: x[a].mean() / x[b].mean() if a.any() and b.any() and x[b].mean() > 0 else np.nan  # noqa: E731
    return np.array([100 * cens.mean(), ratio(cens, A, ~A),
                     ratio(life, S & ~Cs & ~A, Cs & ~A),       # Methods rule (exclusive)
                     ratio(life, S & ~A, Cs & ~A & ~S)])      # SAI > SSI > CSI priority ('excl' in ep200.json)


def compute_ep200():
    """Panel e: ResNet-18 seed 42, 50 vs 200 epochs (persist_200ep.RUNS); record-level bootstrap with the
    same draws as persist_200ep.bootstrap (2000 resamples, seed 0 per run)."""
    from persist_200ep import RUNS, load, stored_thr
    names = ("survival_pct", "tg_surv_adv", "ssi_csi_life_methods", "ssi_csi_life_priority")
    out, boots = {}, {}
    for run, (exp, lane) in RUNS.items():
        d, res = load(exp, lane)
        thr = stored_thr(res)
        R = records_real(d, res, "paper")
        est = _ep200_vec(R, thr)
        rng = np.random.default_rng(0)
        B = np.array([_ep200_vec(R[rng.integers(0, len(R), len(R))], thr) for _ in range(EP_BOOT)])
        boots[run] = B
        ssi, csi, sai = R[:, C["ssi"]], R[:, C["csi"]], R[:, C["sai"]]
        A, S, Cs = sai > 0.9, ssi > thr["ssi"], csi > thr["csi"]
        out[run] = {"n_records": int(len(R)), "thr": thr,
                    "n_tg_other": [int(A.sum()), int((~A).sum())],
                    "n_ssi_csi_methods": [int((S & ~Cs & ~A).sum()), int((Cs & ~A).sum())],
                    "n_ssi_csi_priority": [int((S & ~A).sum()), int((Cs & ~A & ~S).sum())]}
        for i, nm in enumerate(names):
            out[run][nm] = float(est[i])
            out[run][nm + "_ci95"] = [float(x) for x in np.nanpercentile(B[:, i], [2.5, 97.5])]
    out["diff_200_minus_50"] = {}
    for i, nm in enumerate(names):
        diff = boots["200ep"][:, i] - boots["50ep"][:, i]
        out["diff_200_minus_50"][nm] = {
            "ci95": [float(x) for x in np.nanpercentile(diff, [2.5, 97.5])],
            "p_two_sided": float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean())))}
    out["note"] = ("records_real(d, res, 'paper') (alive, terminal duplicate removed, avgpool dropped); "
                   "task-general = SAI > 0.9; stored t_ssi / t_csi; record-level bootstrap (records of one "
                   f"feature chain are not independent -> CIs optimistic), {EP_BOOT} resamples, seed 0")
    return out


# plotting
def sep_lines(ax):
    for x in (2.5, 5.5):
        ax.axvline(x, color="#D1D5DB", lw=0.6, ls="--", zorder=0)


def run_ticks(ax, runs):
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([f"{SHORT[r['arch']]}-{r['seed']}" for r in runs], rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.set_xlim(-0.6, len(runs) - 0.4)


def plot(D):
    nmi_style.apply()
    runs = D["runs"]
    x = np.arange(len(runs))
    fig = plt.figure(figsize=(FULL_W, FULL_W))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1, 1.3], hspace=0.42,
                     left=0.10, right=0.97, top=0.95, bottom=0.07)
    gtop = outer[0].subgridspec(1, 2, wspace=0.38)
    gbot = outer[1].subgridspec(1, 3, width_ratios=BOT_W, wspace=0.42)

    # (a) lifespan ratio
    ax = fig.add_subplot(gtop[0])
    ratios = [r["ratio"] for r in runs]
    bars = ax.bar(x, ratios, width=0.7, color=[ARCH[r["arch"]] for r in runs], alpha=FILL_ALPHA, lw=0)
    for b, v in zip(bars, ratios):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    ax.axhline(1.0, color="black", ls="--", lw=0.6, alpha=0.6)
    sep_lines(ax)
    run_ticks(ax, runs)
    ax.set_ylim(0, max(ratios) * 1.19)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
    ax.set_ylabel("Mean lifespan ratio")
    ax.set_title("Construction / refinement feature lifespan")
    panel(ax, "a")

    # (b) Spearman rho
    ax = fig.add_subplot(gtop[1])
    w = 0.36
    lo_all = []
    for k, off, lab in (("ssi", -w / 2, "SSI-lifespan"), ("csi", w / 2, "CSI-lifespan")):
        col = INDEX[k.upper()]
        rr = np.array([r["rho"][k]["rho"] for r in runs])
        se = np.array([r["rho"][k]["se"] for r in runs])
        ax.bar(x + off, rr, width=w, color=col, alpha=FILL_ALPHA, lw=0, label=lab,
               yerr=se, ecolor=ERR, capsize=2.5, error_kw={"linewidth": 0.6, "capthick": 0.8})
        for i, r in enumerate(runs):
            s = p_stars(r["rho"][k]["p"])
            if k == "ssi":
                y = max(rr[i] + se[i], 0) + 0.012
                ax.text(x[i] + off, y, s, ha="center", va="bottom", fontsize=6)
            else:
                y = min(rr[i] - se[i], 0) - 0.006
                ax.text(x[i] + off, y, s, ha="center", va="top", fontsize=6)
            lo_all.append(min(rr[i] - se[i], 0))
    ax.axhline(0, color="black", lw=0.5)
    sep_lines(ax)
    run_ticks(ax, runs)
    hi = max(max(r["rho"]["ssi"]["rho"] + r["rho"]["ssi"]["se"] for r in runs), 0.05)
    ax.set_ylim(min(lo_all) - 0.04, hi + 0.08)
    ax.set_ylabel("Spearman rho")
    ax.legend(loc="upper left", handlelength=2.6, borderaxespad=0.2)
    panel(ax, "b")

    # (c) developmental persistence
    ax = fig.add_subplot(gbot[0])
    wc = 0.22
    offs = {"SAI": -wc, "SSI": 0.0, "CSI": wc}
    top = 0
    for ai, arch in enumerate(ARCHS):
        rs = [r for r in runs if r["arch"] == arch]
        for k in ("SAI", "SSI", "CSI"):
            v = np.array([r["dev"][k] for r in rs])
            m, sem = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
            xc = ai + offs[k]
            ax.bar(xc, m, width=wc, color=INDEX[k], alpha=FILL_ALPHA, lw=0)
            ax.errorbar(xc, m, yerr=sem, color=ERR, capsize=2.5, lw=0.6, capthick=0.8, zorder=4)
            ax.scatter(np.full(len(v), xc), v, s=14, color=INDEX[k], edgecolors="white", linewidths=0.5,
                       zorder=5)
            top = max(top, v.max(), m + sem)
        sai_m = np.mean([r["dev"]["SAI"] for r in rs])
        csi_m = np.mean([r["dev"]["CSI"] for r in rs])
        gtop = max(max(r["dev"][k] for r in rs) for k in ("SAI", "SSI", "CSI"))
        ax.text(ai, gtop + 0.015, f"SAI/CSI={sai_m / csi_m:.1f}×", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(range(len(ARCHS)))
    ax.set_xticklabels(ARCHS)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 0.6 if top > 0.45 else 0.5)
    ax.set_ylabel("Developmental persistence")
    ax.legend(handles=[Patch(color=INDEX[k], alpha=FILL_ALPHA, lw=0, label=k) for k in ("SAI", "SSI", "CSI")],
              loc="upper right", handlelength=2.6, labelspacing=0.8)
    panel(ax, "c")

    # (d) scaffold census
    sub = gbot[1].subgridspec(3, 1, hspace=0.62)
    axes = [fig.add_subplot(sub[i]) for i in range(3)]
    for i, (ax, arch) in enumerate(zip(axes, ARCHS)):
        rs = [r for r in runs if r["arch"] == arch]
        cnt = np.mean([np.array(r["census"], dtype=float) for r in rs], axis=0)  # (n_ckpt, 3)
        xs = np.linspace(0, 10, len(cnt))
        ax.stackplot(xs, cnt[:, 0], cnt[:, 1], cnt[:, 2], colors=[INDEX["SAI"], INDEX["SSI"], INDEX["CSI"]],
                     alpha=FILL_ALPHA, edgecolor="white", linewidth=0.8, labels=["SAI", "SSI", "CSI"])
        ax.set_xlim(-0.5, 10.5)
        ax.set_xticks(range(11))
        ax.set_ylim(0, cnt.sum(1).max() * 1.05)
        ax.set_title(arch, pad=3)
        if i < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Training checkpoint")
        if i == 1:
            ax.set_ylabel("Features above\nadaptive threshold")
        ax.yaxis.set_major_locator(plt.MaxNLocator(3, integer=True))
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.22), ncol=3, handlelength=2.6, columnspacing=0.6,
                   handletextpad=0.5, borderaxespad=0)
    panel(axes[0], "d", x=-0.17, y=1.12)

    # (e) 50 vs 200 epochs (ResNet-18 seed 42)
    plot_e(fig.add_subplot(gbot[2]), D["ep200"])
    return save(fig, "fig4_lifespan")


BOT_W = (1.0, 1.0, 0.72)  # c | d | e
E_LIGHT = 0.32            # alpha of the 50-epoch bars (200 epochs: FILL_ALPHA)


def plot_e(ax, E):
    w = 0.38
    groups = (("tg_surv_adv", PROC["Tg-E"], "Task-general\nvs all other\n(survival)"),
              ("ssi_csi_life_methods", PROC["Ab-E"], "Superclass- vs\nclass-selective\n(lifespan)"))
    tops = []
    for gi, (k, col, _) in enumerate(groups):
        for run, off, al in (("50ep", -w / 2, E_LIGHT), ("200ep", w / 2, FILL_ALPHA)):
            v, (lo, hi) = E[run][k], E[run][k + "_ci95"]
            xc = gi + off
            ax.bar(xc, v, width=w, color=col, alpha=al, lw=0)
            ax.errorbar(xc, v, yerr=[[v - lo], [hi - v]], color=ERR, capsize=2.5, lw=0.6, capthick=0.8, zorder=4)
            ax.text(xc, hi + 0.06, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
            tops.append(hi)
        p = E["diff_200_minus_50"][k]["p_two_sided"]
        yb = max(E["50ep"][k + "_ci95"][1], E["200ep"][k + "_ci95"][1]) + 0.36
        ax.plot([gi - w / 2, gi - w / 2, gi + w / 2, gi + w / 2], [yb - 0.06, yb, yb, yb - 0.06], color=ERR, lw=0.6)
        ax.text(gi, yb + 0.04, f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.axhline(1.0, color="black", ls="--", lw=0.6, alpha=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([g[2] for g in groups])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 3.6)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.set_ylabel("Ratio (durability advantage)")
    ax.set_title("General features become more\ndurable with longer training", pad=4)
    ax.legend(handles=[Patch(color=NEUTRAL, alpha=E_LIGHT, lw=0, label="50 epochs"),
                       Patch(color=NEUTRAL, alpha=FILL_ALPHA, lw=0, label="200 epochs"),
                       Line2D([], [], color="black", ls="--", lw=0.6, alpha=0.6, label="no advantage")],
              loc="upper left", handlelength=1.6, borderaxespad=0.2)
    s50, s200 = E["50ep"]["survival_pct"], E["200ep"]["survival_pct"]
    ax.text(0.03, 0.80, f"overall survival\n{s50:.1f}% $\\rightarrow$ {s200:.1f}%", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.5)
    ax.text(0.0, -0.165, "ResNet-18, one run per duration;\n95% CI: record-level bootstrap",
            transform=ax.transAxes, ha="left", va="top", fontsize=6, color=NEUTRAL)
    panel(ax, "e", x=-0.30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replot", action="store_true", help="re-use cached numbers")
    a = ap.parse_args()
    if a.replot and CACHE.exists():
        D = json.loads(CACHE.read_text())
    else:
        D = compute()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(D, indent=1))
    if "ep200" not in D:  # panel e; cheap (cached matching + record-level bootstrap)
        D["ep200"] = compute_ep200()
        CACHE.write_text(json.dumps(D, indent=1))
    runs = D["runs"]

    assert all("ratio_overlap" in r for r in runs), "cache lacks the exclusive-rule keys: rerun without --replot"

    # validation of the overlap ratio against decomposition.csv (S3_dedup_avgpool)
    dec = pd.read_csv(TABLES / "decomposition.csv")
    dec = dec[dec.step == "S3_dedup_avgpool"].set_index(["arch", "seed"])
    print("\n(a) construction/refinement ratio: Methods rule (plotted) | overlap (vs decomposition S3)")
    for r in runs:
        ref = dec.loc[(r["arch"], r["seed"]), "ratio"]
        print(f"  {r['arch']:10s} {r['seed']:4d}  methods {r['ratio']:.4f} (n {r['n_coarse']}/{r['n_fine']}; "
              f"layer x birth-matched {r['ratio_birth_layer_matched']:.3f}; HR {r['lr_hr_layer_birth']:.3f} "
              f"p {r['lr_p_layer_birth']:.2g})  | overlap {r['ratio_overlap']:.4f} (n {r['n_coarse_overlap']}/"
              f"{r['n_fine_overlap']})  ref {ref:.4f} {'OK' if abs(r['ratio_overlap'] - ref) < 1e-9 else 'MISMATCH'}")
    assert all(abs(r["ratio_overlap"] - dec.loc[(r["arch"], r["seed"]), "ratio"]) < 1e-9 for r in runs)
    for key in ("ratio", "ratio_overlap"):
        v = [r[key] for r in runs]
        print(f"  {key}: range {min(v):.2f}-{max(v):.2f}; >1 in {sum(x > 1 for x in v)}/9")

    print("\n(b) Spearman rho (bootstrap SE, stars)")
    for r in runs:
        s = " ".join(f"{k.upper()} {r['rho'][k]['rho']:+.3f}±{r['rho'][k]['se']:.3f} {p_stars(r['rho'][k]['p'])}"
                     for k in ("ssi", "csi"))
        print(f"  {r['arch']:10s} {r['seed']:4d}  {s}")
    print(f"  SSI>0 & p<.05: {sum(r['rho']['ssi']['rho'] > 0 and r['rho']['ssi']['p'] < .05 for r in runs)}/9; "
          f"CSI<0 & p<.05: {sum(r['rho']['csi']['rho'] < 0 and r['rho']['csi']['p'] < .05 for r in runs)}/9")

    summ = {"rule": "task-general = SAI > t_sai; superclass-selective = SSI > t_ssi & CSI <= t_csi & not task-general; "
                    "class-selective = CSI > t_csi & not task-general. Lifespan ratio (fig4a): t_sai = 0.9, stored t_ssi/t_csi; "
                    "coarse = task-general OR superclass-selective, fine = class-selective. Persistence and census (fig4c/d): stored t_sai/t_ssi/t_csi.",
            "fig4a": {}, "fig4c": {}, "fig4d": {}}
    for key, nm in (("ratio", "methods"), ("ratio_overlap", "overlap_core")):
        v = [r[key] for r in runs]
        summ["fig4a"][nm] = {"per_run": {f"{r['arch']}|{r['seed']}": round(r[key], 4) for r in runs},
                             "range": [round(min(v), 3), round(max(v), 3)], "n_gt1": int(sum(x > 1 for x in v))}
    summ["fig4a"]["methods_layer_birth_matched_range"] = [round(min(r["ratio_birth_layer_matched"] for r in runs), 3),
                                                          round(max(r["ratio_birth_layer_matched"] for r in runs), 3)]
    summ["fig4a"]["methods_n_layer_birth_logrank_HR_lt1_sig"] = int(sum(r["lr_hr_layer_birth"] < 1 and r["lr_p_layer_birth"] < .05 for r in runs))
    summ["fig4a"]["methods_n_birth_logrank_HR_lt1_sig"] = int(sum(r["hr_birth_stored"] < 1 and r["hr_birth_stored_p"] < .05 for r in runs))
    summ["fig4a"]["overlap_n_birth_logrank_HR_lt1_sig"] = int(sum(r["hr_birth_stored_overlap"] < 1 and r["hr_birth_stored_overlap_p"] < .05 for r in runs))

    for dk, nm in (("dev", "methods"), ("dev_priority", "priority_core")):
        print(f"\n(c) developmental persistence (mean ± SEM over seeds), cohorts: {nm}")
        summ["fig4c"][nm] = {}
        for arch in ARCHS:
            rs = [r for r in runs if r["arch"] == arch]
            line = []
            for k in ("SAI", "SSI", "CSI"):
                v = np.array([r[dk][k] for r in rs])
                line.append(f"{k} {v.mean():.3f}±{v.std(ddof=1) / np.sqrt(3):.3f} [{', '.join(f'{x:.3f}' for x in v)}]")
            m = {k: np.mean([r[dk][k] for r in rs]) for k in ("SAI", "SSI", "CSI")}
            print(f"  {arch:10s} " + "; ".join(line) + f"; SAI/CSI={m['SAI'] / m['CSI']:.2f}x SSI/CSI={m['SSI'] / m['CSI']:.2f}x"
                  f"; n(SAI,SSI,CSI)={[(r[dk]['n_SAI'], r[dk]['n_SSI'], r[dk]['n_CSI']) for r in rs]}")
            summ["fig4c"][nm][arch] = {"SAI_over_CSI": round(m["SAI"] / m["CSI"], 3), "SSI_over_CSI": round(m["SSI"] / m["CSI"], 3),
                                       "mean": {k: round(v, 4) for k, v in m.items()},
                                       "n_runs_SAI_gt_CSI": int(sum(r[dk]["SAI"] > r[dk]["CSI"] for r in rs)),
                                       "n_runs_SSI_gt_CSI": int(sum(r[dk]["SSI"] > r[dk]["CSI"] for r in rs)),
                                       "n_per_run": [[r[dk]["n_SAI"], r[dk]["n_SSI"], r[dk]["n_CSI"]] for r in rs]}

    for ck, nm in (("census", "methods"), ("census_overlap", "overlap_core")):
        print(f"\n(d) census {nm} (mean over seeds): first -> last real checkpoint (SAI, SSI, CSI, total)")
        summ["fig4d"][nm] = {}
        for arch in ARCHS:
            rs = [r for r in runs if r["arch"] == arch]
            cnt = np.mean([np.array(r[ck], dtype=float) for r in rs], axis=0)
            f, l = cnt[0], cnt[-1]
            pk = int(np.argmax(cnt.sum(1)))
            pa = int(np.argmax(cnt[:, 1]))
            print(f"  {arch:10s} ckpts {rs[0]['labels'][0]}..{rs[0]['labels'][-1]}: "
                  f"{f.round(0).astype(int).tolist()} tot {f.sum():.0f} -> {l.round(0).astype(int).tolist()} tot {l.sum():.0f}; "
                  f"peak total {cnt.sum(1)[pk]:.0f} at ckpt {rs[0]['labels'][pk]}; SSI peak {cnt[pa, 1]:.0f} at ckpt "
                  f"{rs[0]['labels'][pa]}")
            summ["fig4d"][nm][arch] = {"first": f.round(1).tolist(), "last": l.round(1).tolist(),
                                       "ssi_peak": round(float(cnt[pa, 1]), 1), "ssi_peak_ckpt": rs[0]["labels"][pa]}
    E = D["ep200"]
    print("\n(e) ResNet-18 s42, 50 -> 200 epochs (record-level bootstrap 95% CI; p of 200-50 difference)")
    for k in ("survival_pct", "tg_surv_adv", "ssi_csi_life_methods", "ssi_csi_life_priority"):
        a, b, df = E["50ep"], E["200ep"], E["diff_200_minus_50"][k]
        print(f"  {k:24s} {a[k]:.3f} [{a[k + '_ci95'][0]:.3f}, {a[k + '_ci95'][1]:.3f}] -> {b[k]:.3f} "
              f"[{b[k + '_ci95'][0]:.3f}, {b[k + '_ci95'][1]:.3f}]; diff CI [{df['ci95'][0]:.3f}, {df['ci95'][1]:.3f}] "
              f"p {df['p_two_sided']:.3f}")
    summ["fig4e"] = {**E, "plotted": ["tg_surv_adv", "ssi_csi_life_methods"],
                     "ep200_json_keys": {"tg_surv_adv": "tg_vs_nontg|surv|SAI>0.9/SAI<=0.9",
                                         "ssi_csi_life_priority": "stored_sai0.9|excl|life|SSI/CSI",
                                         "survival_pct": "survival_pct_records"}}
    SUMM_OUT.write_text(json.dumps(summ, indent=1))
    print(f"\nwrote {SUMM_OUT}")

    nl = pd.read_csv(TABLES / "no_layer_matching.csv")
    nl = nl[nl.rule == "paper"]
    sig = nl[(nl.logrank_birth_HR < 1) & (nl.logrank_birth_p < 0.05)]
    print("\nBirth-stratified log-rank HR (no_layer_matching.csv, rule 'paper'):")
    for row in nl.itertuples():
        print(f"  {row.arch:10s} {row.seed:4d}  HR {row.logrank_birth_HR:.3f}  p {row.logrank_birth_p:.2g}")
    print(f"  all runs {nl.logrank_birth_HR.min():.2f}-{nl.logrank_birth_HR.max():.2f}; HR<1 & p<.05 in {len(sig)}/9 "
          f"(range {sig.logrank_birth_HR.min():.2f}-{sig.logrank_birth_HR.max():.2f}); "
          f"reversed sig {int(((nl.logrank_birth_HR > 1) & (nl.logrank_birth_p < .05)).sum())}/9")
    for k, nm in (("hr_birth_stored", "Methods rule"), ("hr_birth_stored_overlap", "overlap")):
        print(f"  same test, stored thresholds, {nm} (this script): " +
              ", ".join(f"{SHORT[r['arch']]}-{r['seed']} {r[k]:.2f} (p={r[k + '_p']:.1g})" for r in runs))

    out = plot(D)
    print(f"\nsaved {out} (+ .pdf)")


if __name__ == "__main__":
    main()
