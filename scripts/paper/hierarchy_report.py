#!/usr/bin/env python3
"""Class hierarchy, step 5: tables + figures from the caches of hierarchy_geometry.py and hierarchy_sae.py and the
stored CKA/RSA and SAE top-k caches (open seeds only). Usage: python scripts/paper/hierarchy_report.py"""
from __future__ import annotations

import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from hierarchy_lib import OUT, SEEDS  # noqa: E402

C = OUT / "cache"
COL = {"sc_true_K20": "#2a78d6", "sc_rand_K20": "#eb6834", "scratch_E30": "#52514e", "taskirr": "#4a3aa7",
       "sc_true_K30": "#2a78d6", "sc_rand_K30": "#eb6834"}
NAME = {"sc_true_K20": "true superclasses → fine", "sc_rand_K20": "random grouping → fine",
        "scratch_E30": "fine only (scratch)", "taskirr": "task-irrelevant SC (5-way)",
        "sc_true_K30": "true SC (K=30)", "sc_rand_K30": "random (K=30)"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})


def load_geo():
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(C / "geo_*_s*.csv")))], ignore_index=True)
    assert set(d.seed) <= set(SEEDS)
    d["true_sh_exc"] = d.true_share - d.null20_share_mean
    d["imp_sh_exc"] = d.imposed_share - d.null20_share_mean
    d["tl_sh_exc"] = d.tlabel_share - d.null5_share_mean
    d["true_wg_rel"] = d.true_wg_sep / d.null20_wg_sep_mean
    d["imp_wg_rel"] = d.imposed_wg_sep / d.null20_wg_sep_mean
    d["true_within_rel"] = d.p_true_within / d.p_null_within
    return d


def paired(df, a, b, metric):
    x = df[a] - df[b]
    return {"metric": metric, "contrast": f"{a} - {b}", "mean": x.mean(), "min": x.min(), "max": x.max(),
            "n_pos": int((x > 0).sum()), "n": len(x), **{f"s{s}": v for s, v in zip(df.seed, x)}}


def main():
    g = load_geo()
    g.to_csv(OUT / "D_geometry_all.csv.gz", index=False)
    ap = g[g.layer == "avgpool"]
    M = ["p_fine", "p_true_hoct", "p_imposed_hoct", "p_null_hoct", "true_auc", "imposed_auc", "true_sh_exc",
         "imp_sh_exc", "true_share_w", "true_wg_rel", "imp_wg_rel", "p_true_within", "p_null_within",
         "true_within_rel", "centroid_pr", "fisher_w", "tl_sh_exc", "tlabel_auc", "p_tlabel_hoct", "val_acc"]
    tc = ap.groupby(["lane", "epoch"])[M].agg(["mean", "min", "max"])
    tc.columns = [f"{a}_{b}" for a, b in tc.columns]
    tc.reset_index().to_csv(OUT / "D_timecourse_avgpool.csv", index=False)
    l3 = g[g.layer == "layer3"].groupby(["lane", "epoch"])[M].mean().reset_index()
    l3.to_csv(OUT / "D_timecourse_layer3.csv", index=False)
    lay = g[g.epoch.isin([0, 20])].groupby(["lane", "epoch", "layer"])[
        ["true_sh_exc", "imp_sh_exc", "true_auc", "imposed_auc", "true_wg_rel", "tl_sh_exc", "tlabel_auc"]].mean()
    lay.reset_index().to_csv(OUT / "D_by_layer_epoch0_20.csv", index=False)

    # switch contrasts (epoch 20), per seed
    def at(lane, ep, cols, layer="avgpool"):
        x = g[(g.lane == lane) & (g.epoch == ep) & (g.layer == layer)].set_index("seed")[cols]
        return x
    cols = ["p_true_hoct", "p_imposed_hoct", "true_auc", "imposed_auc", "true_sh_exc", "imp_sh_exc",
            "true_wg_rel", "imp_wg_rel", "p_fine", "p_true_within", "true_within_rel", "centroid_pr"]
    rows = []
    for layer in ("avgpool", "layer3"):
        T, R, S0, S20 = (at("sc_true_K20", 20, cols, layer), at("sc_rand_K20", 20, cols, layer),
                         at("scratch_E30", 0, cols, layer), at("scratch_E30", 20, cols, layer))
        S5 = at("scratch_E30", 5, cols, layer)
        for m in cols:
            df = pd.DataFrame({"seed": T.index, "sc_true@20": T[m].values, "sc_rand@20": R[m].values,
                               "init": S0[m].values, "scratch@5": S5[m].values, "scratch@20": S20[m].values})
            for a, b in (("sc_true@20", "sc_rand@20"), ("sc_rand@20", "init"), ("sc_rand@20", "scratch@20"),
                         ("sc_rand@20", "scratch@5"), ("sc_true@20", "scratch@20")):
                rows.append({"layer": layer, **paired(df, a, b, m)})
        # gain of true vs imposed structure within sc_rand
        gt = R.p_true_hoct - S0.p_true_hoct
        gi = R.p_imposed_hoct - S0.p_imposed_hoct
        x = gt - gi
        rows.append({"layer": layer, "metric": "hoct_gain_true_minus_imposed", "contrast": "sc_rand@20 vs init",
                     "mean": x.mean(), "min": x.min(), "max": x.max(), "n_pos": int((x > 0).sum()), "n": len(x),
                     **{f"s{s}": v for s, v in x.items()}})
    pd.DataFrame(rows).to_csv(OUT / "D_switch_contrasts.csv", index=False)

    # taskirr contrasts vs init and vs scratch at matched epochs
    rows = []
    for ep in (5, 10, 20, 30):
        for m in ("true_sh_exc", "true_auc", "p_true_hoct", "p_fine", "tl_sh_exc", "tlabel_auc", "true_wg_rel"):
            ti = at("taskirr", ep, [m]).rename(columns={m: "taskirr"})
            ti["init"] = at("taskirr", 0, [m])[m]
            ti["scratch"] = at("scratch_E30", ep, [m])[m]
            ti = ti.reset_index()
            for a, b in (("taskirr", "init"), ("taskirr", "scratch")):
                rows.append({"epoch": ep, **paired(ti, a, b, m)})
    pd.DataFrame(rows).to_csv(OUT / "D_taskirr_contrasts.csv", index=False)

    # does true-SC structure track class discrimination? residual vs scratch curve
    sc = ap[ap.lane == "scratch_E30"].sort_values("p_fine")
    fit = np.polyfit(sc.p_fine, sc.p_true_hoct, 2)
    ap = ap.assign(hoct_resid=ap.p_true_hoct - np.polyval(fit, ap.p_fine))
    ap.groupby(["lane", "epoch"])[["p_fine", "p_true_hoct", "hoct_resid"]].mean().reset_index().to_csv(
        OUT / "D_hoct_vs_fine_residual.csv", index=False)

    # CKA / RSA convergence
    ck = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(C / "cka_s*.csv")))], ignore_index=True)
    assert set(ck.seed) <= set(SEEDS)
    rows = []
    for (s, L), d in ck.groupby(["seed", "layer"]):
        def get(a, ea, b, eb):
            q = d[((d.a_lane == a) & (d.a_epoch == ea) & (d.b_lane == b) & (d.b_epoch == eb)) |
                  ((d.a_lane == b) & (d.a_epoch == eb) & (d.b_lane == a) & (d.b_epoch == ea))]
            return (np.nan, np.nan) if q.empty else (float(q.cka.iloc[0]), float(q.rsa.iloc[0]))
        for k in (0, 1, 5, 10, 15, 20, 25, 30):
            r = {"seed": s, "layer": L, "fine_epoch": k}
            for arm in ("sc_true_K20", "sc_rand_K20"):
                if k > 0:
                    r[f"cka_{arm}_vs_scratch_matched"], r[f"rsa_{arm}_vs_scratch_matched"] = get(arm, 20 + k, "scratch_E30", k if k != 1 else 5)
                r[f"cka_{arm}_vs_scratch_end"], r[f"rsa_{arm}_vs_scratch_end"] = get(arm, 20 + k, "scratch_E30", 30)
            r["cka_true_vs_rand"], r["rsa_true_vs_rand"] = get("sc_true_K20", 20 + k, "sc_rand_K20", 20 + k)
            r["cka_scratch_k_vs_scratch_end"], r["rsa_scratch_k_vs_scratch_end"] = get("scratch_E30", max(k, 0), "scratch_E30", 30) if k < 30 else (1.0, 1.0)
            rows.append(r)
        for ep in (5, 10, 15, 20):
            rows.append({"seed": s, "layer": L, "coarse_epoch": ep,
                         "cka_true_vs_rand": get("sc_true_K20", ep, "sc_rand_K20", ep)[0],
                         "rsa_true_vs_rand": get("sc_true_K20", ep, "sc_rand_K20", ep)[1],
                         "cka_true_vs_scratch": get("sc_true_K20", ep, "scratch_E30", ep)[0],
                         "cka_rand_vs_scratch": get("sc_rand_K20", ep, "scratch_E30", ep)[0],
                         "cka_taskirr_vs_scratch": get("taskirr", ep, "scratch_E30", ep)[0],
                         "rsa_true_vs_scratch": get("sc_true_K20", ep, "scratch_E30", ep)[1],
                         "rsa_rand_vs_scratch": get("sc_rand_K20", ep, "scratch_E30", ep)[1],
                         "rsa_taskirr_vs_scratch": get("taskirr", ep, "scratch_E30", ep)[1]})
    cv = pd.DataFrame(rows)
    cv.to_csv(OUT / "D_cka_convergence.csv", index=False)

    # cross-seed RSA ceiling (class RDMs, scratch end) + end-state RSA of curriculum arms across seeds
    rd = {}
    for s in SEEDS:
        z = np.load(C / f"rdm_avgpool_s{s}.npz")
        rd[s] = {it: r for it, r in zip(z["items"], z["rdm"])}
    rows = []
    for i, a in enumerate(SEEDS):
        for b in SEEDS[i + 1:]:
            for la, lb in (("scratch_E30|30", "scratch_E30|30"), ("sc_true_K20|50", "sc_true_K20|50"),
                           ("sc_rand_K20|50", "sc_rand_K20|50"), ("sc_true_K20|50", "scratch_E30|30"),
                           ("sc_rand_K20|50", "scratch_E30|30"), ("sc_true_K20|20", "sc_true_K20|20"),
                           ("sc_rand_K20|20", "sc_rand_K20|20"), ("scratch_E30|0", "scratch_E30|0")):
                ka = [k for k in rd[a] if k.startswith(la + "|")][0]
                kb = [k for k in rd[b] if k.startswith(lb + "|")][0]
                rows.append({"seed_a": a, "seed_b": b, "a": la, "b": lb, "rsa": spearmanr(rd[a][ka], rd[b][kb])[0]})
    pd.DataFrame(rows).to_csv(OUT / "D_cross_seed_rsa.csv", index=False)

    resid_table(ap)
    sae_tables()
    figures(ap, cv)
    sae_figure()


def resid_table(ap):
    """Per seed: true-SC metric minus the value predicted from the fine-probe accuracy by that seed's
    scratch trajectory (quadratic in p_fine through scratch epochs 0..30)."""
    rows = []
    for s in SEEDS:
        d = ap[ap.seed == s]
        sc = d[d.lane == "scratch_E30"]
        for m in ("p_true_hoct", "true_auc", "true_sh_exc"):
            fit = np.polyfit(sc.p_fine, sc[m], 2)
            for _, r in d.iterrows():
                rows.append({"seed": s, "lane": r.lane, "epoch": r.epoch, "metric": m, "p_fine": r.p_fine,
                             "value": r[m], "resid": r[m] - np.polyval(fit, r.p_fine)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "D_resid_vs_scratch_curve_per_seed.csv", index=False)
    w = df.pivot_table(index=["metric", "lane", "epoch"], columns="seed", values="resid")
    w["mean"], w["min"], w["max"] = w[SEEDS].mean(1), w[SEEDS].min(1), w[SEEDS].max(1)
    w.reset_index().to_csv(OUT / "D_resid_vs_scratch_curve_summary.csv", index=False)


def sae_tables():
    thr_all = json.load(open(OUT.parents[1] / "SAE_outcomes" / "thresholds_EC.json"))["common"]
    rows = []
    for src in ("sae", "raw"):
        for s in SEEDS:
            thr = thr_all[str(s)]["thresholds"]["ssi_abstraction_thresh"]
            for lane in ("sc_true_K20", "sc_rand_K20", "scratch_E30"):
                f = pd.read_csv(C / f"sae_feat_{lane}_s{s}_ms4_{src}.csv.gz")
                hi = f.ssi_true > thr
                rows.append({"src": src, "seed": s, "lane": lane, "n_alive": len(f), "census_ssi": hi.mean(),
                             "hi_single_class": (hi & (f.cover == 1)).mean(), "hi_two_class": (hi & (f.cover == 2)).mean(),
                             "hi_multi_class_ge3": (hi & (f.cover >= 3)).mean(),
                             "census_genuine": (hi & (f.ssi_true > f.ssi_null_p95)).mean(),
                             "census_null_partition": (f.ssi_null_mean > thr).mean(),
                             "mean_breadth": f.breadth.mean(), "mean_density": f.density.mean()})
    pd.DataFrame(rows).to_csv(OUT / "D_sae_switch_decomposition.csv", index=False)
    ss = pd.read_csv(C / "sae_summary.csv")
    ss.to_csv(OUT / "D_sae_census_timecourse.csv", index=False)
    pd.read_csv(C / "sae_k_summary.csv").to_csv(OUT / "D_sae_k_sweep_switch.csv", index=False)


def sae_figure():
    dec = pd.read_csv(OUT / "D_sae_switch_decomposition.csv")
    dec = dec[dec.src == "sae"]
    dp = pd.read_csv(OUT / "D_sae_ssi_vs_dprime_switch.csv")
    kk = pd.read_csv(OUT / "D_sae_k_sweep_switch.csv")
    lanes = ["sc_true_K20", "sc_rand_K20", "scratch_E30"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    x = np.arange(3)
    for i, (m, lab) in enumerate((("hi_single_class", "1 class"), ("hi_two_class", "2 classes"),
                                  ("hi_multi_class_ge3", ">=3 of 5 classes"))):
        for j, ln in enumerate(lanes):
            v = dec[dec.lane == ln][m]
            axs[0].bar(i + (j - 1) * 0.27, v.mean(), 0.25, color=COL[ln], label=NAME[ln] if i == 0 else None)
            axs[0].plot([i + (j - 1) * 0.27] * len(v), v, "o", color="#0b0b0b", ms=2.5)
    axs[0].set_xticks(x, ["1 class", "2 classes", ">=3 classes"])
    axs[0].set_ylabel("fraction of alive SAE features")
    axs[0].set_title("SSI-selective features at the switch,\nby sibling classes covered in the best superclass",
                     fontsize=9, loc="left")
    axs[0].legend(frameon=False, fontsize=7)
    for j, ln in enumerate(lanes):
        q = dp[dp.lane == ln]
        for i, m in enumerate(("census_ssi", "census_dprime")):
            axs[1].bar(i + (j - 1) * 0.27, q[m].mean(), 0.25, color=COL[ln])
            axs[1].plot([i + (j - 1) * 0.27] * len(q), q[m], "o", color="#0b0b0b", ms=2.5)
    axs[1].set_xticks([0, 1], ["SSI > threshold\n(mass fraction, curriculum_test)", "best-SC d' > null p95\n(sample discriminability)"])
    axs[1].set_title("Same SAE features, two selectivity criteria", fontsize=9, loc="left")
    for ln in lanes:
        q = kk[kk.lane == ln].groupby("k").census_ssi.agg(["mean", "min", "max"])
        axs[2].fill_between(q.index, q["min"], q["max"], color=COL[ln], alpha=0.15, lw=0)
        axs[2].plot(q.index, q["mean"], color=COL[ln], marker="o", ms=3, lw=2, label=NAME[ln])
    axs[2].set_xscale("log", base=2)
    axs[2].set_xlabel("top-k of retrained SAE")
    axs[2].set_ylabel("SSI census")
    axs[2].set_title("Reversal persists for every k (retrained SAEs)", fontsize=9, loc="left")
    fig.savefig(OUT / "D_fig4_hs1_reversal_mechanism.png", dpi=160)
    plt.close(fig)


def band(ax, d, lane, metric, label=None, ls="-"):
    q = d[d.lane == lane].groupby("epoch")[metric].agg(["mean", "min", "max"])
    ax.fill_between(q.index, q["min"], q["max"], color=COL[lane], alpha=0.15, lw=0)
    ax.plot(q.index, q["mean"], color=COL[lane], lw=2, ls=ls, marker="o", ms=3, label=label or NAME[lane])


def figures(ap, cv):
    lanes = ["sc_true_K20", "sc_rand_K20", "scratch_E30", "taskirr"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for ax, m, t in zip(axs, ["p_true_hoct", "true_auc", "true_sh_exc"],
                        ["True-superclass transfer to held-out classes\n(20-way probe, chance 0.05)",
                         "Centroid nesting AUC, true superclasses\n(0.5 = none)",
                         "Excess between-class variance on true-SC means\n(minus random-partition null)"]):
        for ln in lanes:
            band(ax, ap, ln, m)
        ax.axvline(20, color="#9a9890", lw=1, ls=":")
        ax.set_title(t, fontsize=9, loc="left")
        ax.set_xlabel("training epoch (curriculum switch at 20)")
    axs[0].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "D_fig1_true_superclass_timecourse.png", dpi=160)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for ln in ["sc_rand_K20", "sc_true_K20", "scratch_E30"]:
        band(axs[0], ap, ln, "p_imposed_hoct")
    band(axs[0], ap, "sc_rand_K20", "p_true_hoct", label="random grouping: TRUE superclasses", ls="--")
    axs[0].axvline(20, color="#9a9890", lw=1, ls=":")
    axs[0].set_title("Imposed random partition: held-out-class transfer\n(dashed: true SC in the same lanes)",
                     fontsize=9, loc="left")
    axs[0].set_xlabel("training epoch")
    axs[0].legend(frameon=False, fontsize=8)
    for ln in lanes:
        q = ap[ap.lane == ln].groupby("epoch")[["p_fine", "p_true_hoct"]].mean()
        axs[1].plot(q.p_fine, q.p_true_hoct, color=COL[ln], marker="o", ms=3, lw=1.5, label=NAME[ln])
        for ep in (0, 20):
            if ep in q.index:
                axs[1].annotate(str(ep), (q.p_fine[ep], q.p_true_hoct[ep]), fontsize=7, color="#52514e",
                                xytext=(3, 3), textcoords="offset points")
    axs[1].set_xlabel("fine-class linear probe accuracy (avgpool)")
    axs[1].set_ylabel("true-SC held-out-class transfer")
    axs[1].set_title("Superclass structure vs class discrimination\n(trajectories; labels = epoch)",
                     fontsize=9, loc="left")
    fig.savefig(OUT / "D_fig2_imposed_vs_true_and_vs_fine.png", dpi=160)
    plt.close(fig)

    q = cv[cv.layer == "avgpool"].dropna(subset=["fine_epoch"])
    fig, ax = plt.subplots(figsize=(5, 3.4), constrained_layout=True)
    for arm in ("sc_true_K20", "sc_rand_K20"):
        z = q.groupby("fine_epoch")[f"cka_{arm}_vs_scratch_end"].agg(["mean", "min", "max"])
        ax.fill_between(z.index, z["min"], z["max"], color=COL[arm], alpha=0.15, lw=0)
        ax.plot(z.index, z["mean"], color=COL[arm], marker="o", ms=3, lw=2, label=NAME[arm])
    z = q.groupby("fine_epoch")["cka_scratch_k_vs_scratch_end"].mean()
    ax.plot(z.index, z.values, color=COL["scratch_E30"], marker="o", ms=3, lw=2, label="scratch itself")
    z = q.groupby("fine_epoch")["cka_true_vs_rand"].mean()
    ax.plot(z.index, z.values, color="#1baf7a", marker="s", ms=3, lw=1.5, ls="--", label="true vs random arm")
    ax.set_xlabel("fine-phase epoch")
    ax.set_ylabel("linear CKA (avgpool, same seed)")
    ax.set_title("Convergence to the end state of scratch training", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(OUT / "D_fig3_cka_convergence.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
