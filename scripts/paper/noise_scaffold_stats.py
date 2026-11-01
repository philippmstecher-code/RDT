#!/usr/bin/env python3
"""Statistics and figures for the label-noise scaffold measurements, Di-E reproduction and SAE-noise null.

Inputs (analysis_outputs/paper/NOISE_scaffold/, from noise_scaffold_compute.py and the per-layer,
per-superclass process counts of the stored transition files): raw_behaviour_geometry.csv, raw_probe_alpha_grid.csv,
sae_process_counts_by_layer_superclass.csv, plus the stored sae_results.json of each lane.
Outputs (same folder): per-lane metrics, condition summaries, contrasts, alpha selection, Di-E fractions,
contrasts and trajectories, SAE-null summary, stats_summary.json and figures.

Usage: python scripts/paper/noise_scaffold_stats.py
"""
import json
import os

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from noise_scaffold_compute import LANES_DIR, OUT, lane_registry  # noqa: E402

SEEDS = [7, 42, 137, 256, 314]
EPOCHS = [0, 5, 10, 15, 20, 25, 30]
CONDS = ["standard", "within_sc_p01", "within_sc_p03", "between_sc_p01",
         "between_sc_p03", "random_p01", "random_p03"]
RNG_SEED = 42
N_BOOT = 10_000

CONTRASTS = []
for p in ("p03", "p01"):
    CONTRASTS += [
        (f"between_sc_{p}", f"random_{p}"),
        (f"between_sc_{p}", "standard"),
        (f"random_{p}", "standard"),
        (f"within_sc_{p}", "standard"),
        (f"between_sc_{p}", f"within_sc_{p}"),
    ]

PROBE_LAYERS = ["layer1", "layer2", "layer3", "avgpool"]
BEHAV = ["fine_acc", "sc_acc", "wsc_cond_acc", "err_within_sc_share"]
GEOM = [f"{g}_{l}" for l in ("layer3", "avgpool") for g in
        ("fisher_sc", "fisher_fine_within_sc", "silhouette_sc", "silhouette_sc_centroid_cos")]


# Extended Data Table 1 final fine accuracy (%) of the 35 label-noise runs, seed order 42,137,256,7,314
ED_TABLE1 = {
    "standard": [58.5, 57.8, 58.2, 58.7, 58.2], "within_sc_p01": [57.4, 57.4, 57.7, 57.5, 57.5],
    "within_sc_p03": [55.7, 55.5, 55.8, 55.3, 54.8], "between_sc_p01": [57.0, 57.7, 57.3, 57.4, 57.5],
    "between_sc_p03": [54.0, 54.3, 54.5, 54.4, 55.5], "random_p01": [57.7, 57.2, 57.2, 58.1, 57.3],
    "random_p03": [54.8, 54.4, 52.9, 55.3, 55.0],
}
ED_SEED_ORDER = [42, 137, 256, 7, 314]


def ed_table_check(per):
    rows = []
    fin = per[per.milestone == 6].set_index(["condition", "seed"])
    for cond, vals in ED_TABLE1.items():
        for seed, v in zip(ED_SEED_ORDER, vals):
            r = fin.loc[(cond, seed)]
            rows.append(dict(condition=cond, seed=seed, ed_table1=v, fine_acc_predictions=round(100 * r.fine_acc, 2),
                             val_acc_logged=round(100 * r.val_acc_logged, 2),
                             diff_pred_vs_ed=round(100 * r.fine_acc - v, 2)))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/ed_table1_check.csv", index=False)
    return df


# helpers
def t_ci(x):
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    if n < 2:
        return m, np.nan, np.nan
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return m, m - h, m + h


def boot_pct_ci(x, rng):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (N_BOOT, len(x)))
    bm = x[idx].mean(1)
    return np.percentile(bm, 2.5), np.percentile(bm, 97.5)


def bca_ci(diffs, seed=RNG_SEED, n_boot=N_BOOT):
    """Bias-corrected and accelerated (BCa) bootstrap interval, jackknife acceleration."""
    diffs = np.asarray(diffs, float)
    n = len(diffs)
    rng = np.random.default_rng(seed)
    mean_diff = diffs.mean()
    boot = np.array([diffs[rng.choice(n, n, replace=True)].mean() for _ in range(n_boot)])
    prop = np.mean(boot < mean_diff)
    if prop in (0.0, 1.0):
        return np.nan, np.nan
    z0 = stats.norm.ppf(prop)
    jack = np.array([np.delete(diffs, i).mean() for i in range(n)])
    jd = jack.mean() - jack
    a = np.sum(jd ** 3) / (6.0 * (np.sum(jd ** 2)) ** 1.5 + 1e-15)
    zl, zh = stats.norm.ppf(0.025), stats.norm.ppf(0.975)
    pl = stats.norm.cdf(z0 + (z0 + zl) / (1 - a * (z0 + zl)))
    ph = stats.norm.cdf(z0 + (z0 + zh) / (1 - a * (z0 + zh)))
    return np.percentile(boot, 100 * pl), np.percentile(boot, 100 * ph)


def paired(d, rng):
    d = np.asarray(d, float)
    m, lo, hi = t_ci(d)
    sd = d.std(ddof=1)
    p = stats.ttest_1samp(d, 0).pvalue if sd > 0 else np.nan
    blo, bhi = boot_pct_ci(d, rng)
    return dict(mean_diff=m, t_ci_lo=lo, t_ci_hi=hi, p_paired_t=p, boot_ci_lo=blo, boot_ci_hi=bhi,
                n_neg=int((d < 0).sum()), n_pos=int((d > 0).sum()),
                cohen_dz=m / sd if sd > 0 else np.nan, diffs=";".join(f"{v:.5g}" for v in d))


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    order = np.argsort(pv)
    m = len(pv)
    adj = np.maximum.accumulate((m - np.arange(m)) * pv[order]).clip(max=1)
    tmp = np.empty(m)
    tmp[order] = adj
    out[ok] = tmp
    return out


def bh(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = len(pv)
    order = np.argsort(pv)
    ranked = pv[order] * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    tmp = np.empty(m)
    tmp[order] = adj
    out[ok] = tmp
    return out


def auc(vals_by_epoch, lo, hi):
    e = [x for x in EPOCHS if lo <= x <= hi]
    y = [vals_by_epoch[x] for x in e]
    return np.trapezoid(y, e) / (hi - lo)


# 1. probes / metrics
def behaviour_from_preds(P):
    from noise_scaffold_compute import FINE_TO_COARSE, Y_FINE, Y_SC
    ok = P == Y_FINE
    sc_ok = FINE_TO_COARSE[P] == Y_SC
    return dict(fine_acc=float(ok.mean()), sc_acc=float(sc_ok.mean()),
                wsc_cond_acc=float(ok[sc_ok].mean()),
                err_within_sc_share=float((sc_ok & ~ok).sum() / max((~ok).sum(), 1)))


def build_per_lane():
    beh = pd.read_csv(f"{OUT}/raw_behaviour_geometry.csv")
    # Lanes whose stored sample_predictions disagree with their own checkpoint: use predictions
    # recomputed from checkpoint.pt (recomputed_predictions_inconsistent_lanes.npz) so that
    # behaviour and activations come from the same model state.
    beh["behaviour_source"] = "sample_predictions.pt"
    pf = f"{OUT}/recomputed_predictions_inconsistent_lanes.npz"
    if os.path.exists(pf):
        rec = np.load(pf)
        for key in rec.files:
            lid, m = key.split(":")
            idx = (beh.lane_id == lid) & (beh.milestone == int(m))
            for k, v in behaviour_from_preds(rec[key].astype(int)).items():
                beh.loc[idx, k] = v
            beh.loc[idx, "behaviour_source"] = "recomputed from checkpoint.pt"
    grid = pd.read_csv(f"{OUT}/raw_probe_alpha_grid.csv")
    std = grid[grid.condition == "standard"]
    sel_rows, chosen = [], {}
    for tgt in ("probe_sc", "probe_fine", "probe_wsc"):
        g = std.groupby("alpha")[tgt].mean()
        chosen[tgt] = float(g.idxmax())
        for a, v in g.items():
            sel_rows.append(dict(target=tgt, alpha=a, mean_acc_standard_runs=v, chosen=a == chosen[tgt]))
        # per-layer sensitivity
        for layer, gl in std.groupby("layer"):
            for a, v in gl.groupby("alpha")[tgt].mean().items():
                sel_rows.append(dict(target=tgt, layer=layer, alpha=a, mean_acc_standard_runs=v,
                                     chosen=a == chosen[tgt]))
    pd.DataFrame(sel_rows).to_csv(f"{OUT}/alpha_selection.csv", index=False)

    wide = []
    for tgt, a in chosen.items():
        sub = grid[grid.alpha == a].pivot_table(index=["lane_id", "milestone"], columns="layer", values=tgt)
        sub.columns = [f"{tgt}_{c}" for c in sub.columns]
        wide.append(sub)
    wide = pd.concat(wide, axis=1).reset_index()
    checks = {
        "layer4_equals_avgpool_all": bool(beh.layer4_equals_avgpool.all()),
        "ms7_equals_ms6_all_lanes": bool(beh.loc[beh.milestone == 7, "ms7_equals_ms6_acts"].astype(bool).all()),
        "fine_acc_equals_logged_val_acc_max_absdiff": float((beh.fine_acc - beh.val_acc_logged).abs().max()),
    }
    # check whether runs with the same seed share their initialisation across conditions
    m0 = beh[beh.milestone == 0]
    checks["milestone0_identical_across_conditions_within_seed"] = bool(
        m0.groupby("seed").act_md5.nunique().eq(1).all())
    per = beh[beh.milestone <= 6].merge(wide, on=["lane_id", "milestone"])
    per = per.drop(columns=["ms7_equals_ms6_acts"])
    cols = ["lane_id", "condition", "seed", "milestone", "epoch", "val_acc_logged"] + BEHAV + \
        sorted([c for c in per.columns if c.startswith("probe_")]) + GEOM + ["layer4_equals_avgpool", "act_md5",
                                                                          "behaviour_source"]
    per = per[cols].sort_values(["condition", "seed", "milestone"])
    per.to_csv(f"{OUT}/per_lane_metrics.csv", index=False)
    return per, chosen, checks


def measures(per):
    return BEHAV + sorted([c for c in per.columns if c.startswith("probe_")]) + GEOM


def condition_summary(per, rng):
    rows = []
    for cond in CONDS:
        for ep in EPOCHS:
            sub = per[(per.condition == cond) & (per.epoch == ep)]
            for meas in measures(per):
                x = sub.sort_values("seed")[meas].values
                m, lo, hi = t_ci(x)
                blo, bhi = boot_pct_ci(x, rng)
                rows.append(dict(condition=cond, epoch=ep, measure=meas, n=len(x), mean=m, sd=x.std(ddof=1),
                                 t_ci_lo=lo, t_ci_hi=hi, boot_ci_lo=blo, boot_ci_hi=bhi))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/condition_summary.csv", index=False)
    return df


def contrasts(per, rng):
    rows = []
    piv = {meas: per.pivot_table(index=["condition", "seed"], columns="epoch", values=meas)
           for meas in measures(per)}
    for meas, pv in piv.items():
        for a, b in CONTRASTS:
            A = pv.loc[a].reindex(SEEDS)
            B = pv.loc[b].reindex(SEEDS)
            for ep in EPOCHS:
                d = (A[ep] - B[ep]).values
                rows.append(dict(measure=meas, contrast=f"{a} - {b}", window=f"epoch_{ep}", **paired(d, rng)))
            for lo, hi, name in ((0, 10, "AUC_0-10"), (0, 30, "AUC_0-30")):
                da = np.array([auc(A.loc[s], lo, hi) - auc(B.loc[s], lo, hi) for s in SEEDS])
                rows.append(dict(measure=meas, contrast=f"{a} - {b}", window=name, **paired(da, rng)))
    df = pd.DataFrame(rows)
    df["p_holm_all"] = holm(df.p_paired_t.values)
    df["p_bh_all"] = bh(df.p_paired_t.values)
    fam = df.contrast.str.startswith("between_sc") & df.contrast.str.contains("random")
    df["p_holm_between_vs_random_family"] = np.nan
    df["p_bh_between_vs_random_family"] = np.nan
    df.loc[fam, "p_holm_between_vs_random_family"] = holm(df.loc[fam, "p_paired_t"].values)
    df.loc[fam, "p_bh_between_vs_random_family"] = bh(df.loc[fam, "p_paired_t"].values)
    df.to_csv(f"{OUT}/contrasts.csv", index=False)
    return df


def decomposition(per):
    """fine = sc_acc * wsc_cond_acc exactly (a correct fine prediction implies a
    correct superclass). Log-decompose condition deficits vs standard."""
    rows = []
    pv = {m: per.pivot_table(index=["condition", "seed"], columns="epoch", values=m)
          for m in ("fine_acc", "sc_acc", "wsc_cond_acc")}
    ident = float(np.abs(per.fine_acc - per.sc_acc * per.wsc_cond_acc).max())
    for cond in CONDS[1:]:
        for ep in EPOCHS[1:]:
            dl = {m: np.log(pv[m].loc[cond, ep].reindex(SEEDS).values) -
                  np.log(pv[m].loc["standard", ep].reindex(SEEDS).values) for m in pv}
            tot = dl["fine_acc"].mean()
            rows.append(dict(condition=cond, epoch=ep,
                             dlog_fine=tot, dlog_sc=dl["sc_acc"].mean(), dlog_wsc_cond=dl["wsc_cond_acc"].mean(),
                             share_sc=dl["sc_acc"].mean() / tot if tot != 0 else np.nan,
                             share_wsc_cond=dl["wsc_cond_acc"].mean() / tot if tot != 0 else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/fine_acc_decomposition.csv", index=False)
    return df, ident


# 2. Di-E reproduction & null
def die_analysis(rng):
    reg = {r["lane_id"]: r for r in lane_registry()}
    cnt = pd.read_csv(f"{OUT}/sae_process_counts_by_layer_superclass.csv")
    P5 = ["ab_h", "tg_h", "di_h", "as_h", "de_h"]
    c0 = cnt.groupby(["lane_id", "condition", "seed", "transition", "layer"])[P5].sum().reset_index()
    # Canonicalise: layers in {avgpool, layer4, layers1-3}. Where a transition file was
    # corrupt (1 file: standard seed 42, 0->1, layer4) its counts sit inside the residual
    # 'layers1-3+layer4'. We split that merged residual using the median share of the lost
    # layer in (lost layer + layers1-3) across all intact lanes at the same transition, per
    # process. Totals (and hence the paper definition) stay exact; the cell is flagged.
    intact = c0[c0.layer.isin(["avgpool", "layer4", "layers1-3"])]
    fixed = []
    for (lid, tr), d in c0.groupby(["lane_id", "transition"]):
        d = d.set_index("layer")
        merged = [l for l in d.index if l.startswith("layers1-3+")]
        base = d.iloc[0][["lane_id", "condition", "seed", "transition"]].to_dict()
        if not merged:
            for l in ("avgpool", "layer4", "layers1-3"):
                fixed.append(dict(base, layer=l, approx=False, **d.loc[l, P5].to_dict()))
            continue
        lost = merged[0].split("+")[1:]
        assert len(lost) == 1, (lid, tr, lost)
        lost = lost[0]
        twin = "avgpool" if lost == "layer4" else "layer4"
        ref = intact[(intact.transition == tr) & (intact.lane_id != lid)].pivot_table(
            index="lane_id", columns="layer", values=P5, aggfunc="sum")
        share = {p: float(np.nanmedian(ref[(p, lost)] / (ref[(p, lost)] + ref[(p, "layers1-3")])))
                 for p in P5}
        m = d.loc[merged[0], P5]
        est = {p: m[p] * (share[p] if np.isfinite(share[p]) else 0.5) for p in P5}
        fixed.append(dict(base, layer=twin, approx=False, **d.loc[twin, P5].to_dict()))
        fixed.append(dict(base, layer=lost, approx=True, **est))
        fixed.append(dict(base, layer="layers1-3", approx=True, **{p: m[p] - est[p] for p in P5}))
    c = pd.DataFrame(fixed)
    c[P5] = c[P5].astype(float)
    c["approx"] = c["approx"].astype(bool)
    approx_cells = c[c.approx][["condition", "seed", "transition", "layer"]].drop_duplicates()
    approx_cells.to_csv(f"{OUT}/sae_counts_approximated_cells.csv", index=False)
    real_tr = [f"{i}->{i + 1}" for i in range(6)]

    def frac(sub, num="di_h"):
        s = sub[["ab_h", "tg_h", "di_h"]].sum()
        return s[num] / s.sum()

    variants = {
        "paper (all 7 transitions, 5 layers)": lambda d: d,
        "excl. 6->7 null transition": lambda d: d[d.transition.isin(real_tr)],
        "excl. layer4 duplicate": lambda d: d[d.layer != "layer4"],
        "excl. 6->7 and layer4": lambda d: d[d.transition.isin(real_tr) & (d.layer != "layer4")],
        "avgpool only, real transitions": lambda d: d[d.transition.isin(real_tr) & (d.layer == "avgpool")],
        "layer4 only, real transitions": lambda d: d[d.transition.isin(real_tr) & (d.layer == "layer4")],
        "layers1-3 only, real transitions": lambda d: d[d.transition.isin(real_tr) & (d.layer == "layers1-3")],
        "0->1 only (init -> epoch 5)": lambda d: d[d.transition == "0->1"],
        "1->2..5->6 (excl. 0->1 and 6->7)": lambda d: d[d.transition.isin(real_tr[1:])],
    }
    lane_rows = []
    for lid, d in c.groupby("lane_id"):
        r = reg[lid]
        row = dict(lane_id=lid, condition=r["condition"], seed=r["seed"])
        for vname, f in variants.items():
            sub = f(d)
            row[f"di_frac | {vname}"] = frac(sub, "di_h")
            row[f"ab_frac | {vname}"] = frac(sub, "ab_h")
            row[f"di_count | {vname}"] = int(sub.di_h.sum())
        # noise-floor-corrected: subtract per-layer 6->7 null counts from every real transition
        null = d[d.transition == "6->7"].set_index("layer")[["ab_h", "tg_h", "di_h"]]
        corr = {p: 0.0 for p in ("ab_h", "tg_h", "di_h")}
        for _, x in d[d.transition.isin(real_tr)].iterrows():
            for p in corr:
                corr[p] += max(x[p] - null.loc[x.layer, p], 0)
        row["di_frac | null-subtracted (6->7 per layer), real transitions"] = corr["di_h"] / sum(corr.values())
        row["ab_frac | null-subtracted (6->7 per layer), real transitions"] = corr["ab_h"] / sum(corr.values())
        lane_rows.append(row)
    lanes = pd.DataFrame(lane_rows).sort_values(["condition", "seed"])
    lanes.to_csv(f"{OUT}/die_fraction_per_lane.csv", index=False)
    c.groupby(["condition", "transition", "layer"])[["ab_h", "tg_h", "di_h", "as_h", "de_h"]].agg(["mean", "sem"]).round(1) \
        .to_csv(f"{OUT}/sae_counts_by_condition_layer_transition.csv")

    # cross-check vs sae_results.json process_intensity (paper definition)
    xs = []
    for lid in lanes.lane_id:
        pi = json.load(open(f"{LANES_DIR}/{lid}/sae_analysis/sae_results.json"))["process_intensity"]
        s = {p: sum(t[p] for t in pi) for p in ("ab_h", "tg_h", "di_h")}
        xs.append(s["di_h"] / sum(s.values()))
    max_dev = float(np.max(np.abs(np.array(xs) - lanes["di_frac | paper (all 7 transitions, 5 layers)"].values)))

    crow = []
    vnames = list(variants) + ["null-subtracted (6->7 per layer), real transitions"]
    for vname in vnames:
        for proc in ("di", "ab"):
            col = f"{proc}_frac | {vname}"
            pv = lanes.pivot_table(index="seed", columns="condition", values=col).reindex(SEEDS)
            for cond in CONDS:
                m, lo, hi = t_ci(pv[cond].values)
                crow.append(dict(variant=vname, process=proc, contrast=f"{cond} (level)", mean_diff=m,
                                 t_ci_lo=lo, t_ci_hi=hi, sem=pv[cond].std(ddof=1) / np.sqrt(5)))
            for a, b in CONTRASTS:
                d = (pv[a] - pv[b]).values
                res = paired(d, rng)
                res["bca_lo"], res["bca_hi"] = bca_ci(d)
                res["p_sign_one_sided"] = stats.binomtest(int((d < 0).sum()), 5, 0.5, alternative="greater").pvalue
                crow.append(dict(variant=vname, process=proc, contrast=f"{a} - {b}", **res))
    dc = pd.DataFrame(crow)
    dc.to_csv(f"{OUT}/die_contrasts.csv", index=False)

    # Di-E trajectory (Extended Data Fig. 8f): per-transition fractions and counts (all layers; and dedup)
    trows = []
    for (lid, tr), d in c.groupby(["lane_id", "transition"]):
        r = reg[lid]
        for lay_name, dd in (("all layers", d), ("excl. layer4", d[d.layer != "layer4"])):
            s = dd[["ab_h", "tg_h", "di_h"]].sum()
            trows.append(dict(lane_id=lid, condition=r["condition"], seed=r["seed"], transition=tr, layers=lay_name,
                              di_count=s.di_h, construction_count=s.ab_h + s.tg_h,
                              di_frac=s.di_h / s.sum(), construction_frac=(s.ab_h + s.tg_h) / s.sum()))
    traj = pd.DataFrame(trows)
    traj.to_csv(f"{OUT}/die_trajectory_per_lane.csv", index=False)
    tsum = traj.groupby(["condition", "transition", "layers"]).agg(
        di_frac_mean=("di_frac", "mean"), di_frac_sem=("di_frac", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        construction_frac_mean=("construction_frac", "mean"),
        construction_frac_sem=("construction_frac", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        di_count_mean=("di_count", "mean"), di_count_sem=("di_count", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        construction_count_mean=("construction_count", "mean"),
        construction_count_sem=("construction_count", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
    ).reset_index()
    tsum.to_csv(f"{OUT}/die_trajectory.csv", index=False)

    # SAE-noise null summary
    nrows = []
    for lid, d in c.groupby("lane_id"):
        r = reg[lid]
        tot = d.groupby("transition")[["ab_h", "tg_h", "di_h"]].sum()
        real_mid = tot.loc[[f"{i}->{i + 1}" for i in range(1, 6)]].mean()
        for p in ("ab_h", "tg_h", "di_h"):
            nrows.append(dict(lane_id=lid, condition=r["condition"], seed=r["seed"], process=p, kind="6->7 / mean(1->2..5->6)",
                              value=tot.loc["6->7", p] / real_mid[p] if real_mid[p] > 0 else np.nan))
        l4 = d[d.layer == "layer4"].set_index("transition")
        ap = d[d.layer == "avgpool"].set_index("transition")
        for tr in real_tr:
            if l4.loc[tr, "approx"] or ap.loc[tr, "approx"]:
                continue
            for p in ("ab_h", "tg_h", "di_h"):
                a_, b_ = l4.loc[tr, p], ap.loc[tr, p]
                nrows.append(dict(lane_id=lid, condition=r["condition"], seed=r["seed"], process=p,
                                  kind=f"|layer4-avgpool|/mean @ {tr}",
                                  value=abs(a_ - b_) / ((a_ + b_) / 2) if (a_ + b_) > 0 else np.nan,
                                  layer4=a_, avgpool=b_))
        share = d[d.transition.isin(real_tr)].groupby("layer").di_h.sum()
        nrows.append(dict(lane_id=lid, condition=r["condition"], seed=r["seed"], process="di_h",
                          kind="share of Di-E from layer4+avgpool", value=(share["layer4"] + share["avgpool"]) / share.sum()))
        wc = json.load(open(f"{LANES_DIR}/{lid}/sae_analysis/sae_results.json"))["within_checkpoint_control"]
        for k, v in wc["summary"].items():
            nrows.append(dict(lane_id=lid, condition=r["condition"], seed=r["seed"], process="matching",
                              kind=f"within_checkpoint_control {k}", value=v))
    null = pd.DataFrame(nrows)
    null.to_csv(f"{OUT}/sae_null_per_lane.csv", index=False)
    nsum = null.groupby(["kind", "process"]).value.agg(["mean", "median", "min", "max", "count"]).reset_index()
    nsum.to_csv(f"{OUT}/sae_null_summary.csv", index=False)
    return lanes, dc, tsum, null, nsum, max_dev, c, approx_cells


# 3. figures
COL = {"standard": "#333333", "within_sc": "#2a78d6", "between_sc": "#eb6834", "random": "#1baf7a"}
LAB = {"standard": "Standard", "within_sc": "Within-SC", "between_sc": "Between-SC", "random": "Random"}


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 400, "pdf.fonttype": 42,
    })


def ctype(cond):
    return cond.rsplit("_p", 1)[0] if cond != "standard" else "standard"


def fig_trajectories(cs):
    panels = [("probe_sc_avgpool", "SC probe, avgpool", 0.05),
              ("probe_sc_layer3", "SC probe, layer3", 0.05),
              ("sc_acc", "Network SC accuracy", 0.05),
              ("wsc_cond_acc", "Fine acc. | SC correct", 0.20),
              ("fine_acc", "Network fine accuracy", 0.01)]
    fig, axes = plt.subplots(2, len(panels), figsize=(7.2, 3.4), sharex=True)
    for r, dose in enumerate(("p03", "p01")):
        conds = ["standard", f"within_sc_{dose}", f"between_sc_{dose}", f"random_{dose}"]
        for j, (meas, title, chance) in enumerate(panels):
            ax = axes[r, j]
            for cond in conds:
                s = cs[(cs.condition == cond) & (cs.measure == meas)].sort_values("epoch")
                col = COL[ctype(cond)]
                ax.fill_between(s.epoch, 100 * s.t_ci_lo, 100 * s.t_ci_hi, color=col, alpha=0.18, lw=0)
                ax.plot(s.epoch, 100 * s["mean"], color=col, lw=1.2, label=LAB[ctype(cond)],
                        ls=(0, (3, 1.5)) if ctype(cond) == "random" else "-")
            ax.axhline(100 * chance, color="#9a9a9a", lw=0.5, ls=":")
            ax.set_xticks([0, 10, 20, 30])
            ax.grid(axis="y", color="#e6e6e6", lw=0.4)
            if r == 0:
                ax.set_title(title, pad=3)
            if r == 1:
                ax.set_xlabel("Epoch")
            if j == 0:
                ax.set_ylabel(f"p = 0.{dose[-1]}\n% correct")
    for i, ax in enumerate(axes.flat):
        ax.text(-0.3, 1.1, chr(97 + i), transform=ax.transAxes, fontsize=8, fontweight="bold")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=0.6, h_pad=0.8)
    fig.savefig(f"{OUT}/fig_scaffold_trajectories.png")
    fig.savefig(f"{OUT}/fig_scaffold_trajectories.pdf")
    plt.close(fig)


def fig_contrasts(ct):
    panels = [("probe_sc_avgpool", "SC probe, avgpool"), ("probe_sc_layer3", "SC probe, layer3"),
              ("probe_wsc_avgpool", "Within-SC probe, avgpool"), ("sc_acc", "Network SC accuracy"),
              ("wsc_cond_acc", "Fine acc. | SC correct"), ("fine_acc", "Network fine accuracy"),
              ("fisher_sc_avgpool", "SC Fisher ratio, avgpool"), ("silhouette_sc_centroid_cos_avgpool", "SC centroid silhouette")]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.3))
    for ax, (meas, title) in zip(axes.flat, panels):
        scale = 100 if not meas.startswith(("fisher", "silhouette")) else 1
        for k, (dose, mk, off) in enumerate((("p03", "o", -0.9), ("p01", "s", 0.9))):
            s = ct[(ct.measure == meas) & (ct.contrast == f"between_sc_{dose} - random_{dose}") &
                   ct.window.str.startswith("epoch_")].copy()
            s["ep"] = s.window.str.replace("epoch_", "").astype(int)
            s = s[s.ep > 0].sort_values("ep")
            col = "#eb6834" if dose == "p03" else "#2a78d6"
            ax.errorbar(s.ep + off, scale * s.mean_diff,
                        yerr=[scale * (s.mean_diff - s.t_ci_lo), scale * (s.t_ci_hi - s.mean_diff)],
                        fmt=mk, ms=3, color=col, mfc=col if dose == "p03" else "white", lw=0.8, capsize=0,
                        label=f"p = 0.{dose[-1]}")
        ax.axhline(0, color="#555555", lw=0.6)
        ax.set_title(title, pad=3)
        ax.set_xticks([5, 10, 15, 20, 25, 30])
        ax.grid(axis="y", color="#e6e6e6", lw=0.4)
    for ax in axes[1]:
        ax.set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Between-SC − random\n(pp, paired by seed)")
    axes[1, 0].set_ylabel("Between-SC − random\n(pp, paired by seed)")
    axes[1, 2].set_ylabel("Δ (ratio)")
    for i, ax in enumerate(axes.flat):
        ax.text(-0.32, 1.1, chr(97 + i), transform=ax.transAxes, fontsize=8, fontweight="bold")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=0.5, h_pad=0.8)
    fig.savefig(f"{OUT}/fig_contrasts_between_vs_random.png")
    fig.savefig(f"{OUT}/fig_contrasts_between_vs_random.pdf")
    plt.close(fig)


def fig_die(lanes, dc, tsum, null, c):
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4), gridspec_kw=dict(width_ratios=[1, 1.05, 1, 0.9]))
    # a: Di-E fraction by condition, paper vs cleaned definition
    ax = axes[0]
    defs = [("paper (all 7 transitions, 5 layers)", "Paper definition", "o"),
            ("excl. 6->7 and layer4", "Excl. 6→7 null & layer4 duplicate", "s")]
    for k, (v, lab, mk) in enumerate(defs):
        col = f"di_frac | {v}"
        xs = np.arange(len(CONDS)) + (k - 0.5) * 0.3
        g = lanes.groupby("condition")[col]
        m = g.mean().reindex(CONDS)
        se = (g.std(ddof=1) / np.sqrt(5)).reindex(CONDS)
        ax.errorbar(xs, 100 * m, yerr=100 * se, fmt=mk, ms=3, lw=0.8, color="#333333",
                    mfc="#333333" if k == 0 else "white", label=lab)
    ax.set_xticks(range(len(CONDS)))
    ax.set_xticklabels(["Standard", "Within .1", "Within .3", "Between .1", "Between .3", "Random .1", "Random .3"],
                       rotation=55, ha="right", fontsize=6)
    ax.set_ylabel("Di-E fraction (%)\nDi/(Ab+Tg+Di), mean ± s.e.m.")
    ax.set_ylim(20, 70)
    ax.legend(frameon=False, loc="upper center", fontsize=5.5, ncol=1, bbox_to_anchor=(0.5, 1.02))
    ax.set_title("Di-E fraction", pad=3)
    # b: paired between-random contrast across variants (BCa)
    ax = axes[1]
    vs = ["paper (all 7 transitions, 5 layers)", "excl. 6->7 null transition", "excl. layer4 duplicate",
          "excl. 6->7 and layer4", "null-subtracted (6->7 per layer), real transitions",
          "0->1 only (init -> epoch 5)", "1->2..5->6 (excl. 0->1 and 6->7)",
          "avgpool only, real transitions", "layer4 only, real transitions", "layers1-3 only, real transitions"]
    short = ["Paper", "−6→7", "−layer4", "−6→7 −layer4", "Null-subtracted", "0→1 only", "1→2 … 5→6",
             "avgpool only", "layer4 only", "layers 1–3 only"]
    for i, v in enumerate(vs):
        s = dc[(dc.variant == v) & (dc.process == "di") & (dc.contrast == "between_sc_p03 - random_p03")].iloc[0]
        ax.errorbar(100 * s.mean_diff, i, xerr=[[100 * (s.mean_diff - s.bca_lo)], [100 * (s.bca_hi - s.mean_diff)]],
                    fmt="o", ms=3, color="#eb6834", lw=0.8)
    ax.axvline(0, color="#555555", lw=0.6)
    ax.set_yticks(range(len(vs)))
    ax.set_yticklabels(short)
    ax.invert_yaxis()
    ax.set_xlabel("Between-SC − random, p=0.3\n(pp Di-E fraction, BCa 95% CI)")
    ax.set_title("Paired contrast", pad=3)
    # c: Di-E counts per transition (all layers), as the trajectory of Extended Data Fig. 8f
    ax = axes[2]
    trs = [f"{i}->{i + 1}" for i in range(7)]
    for cond in ("standard", "between_sc_p03", "random_p03"):
        s = tsum[(tsum.condition == cond) & (tsum.layers == "all layers")].set_index("transition").reindex(trs)
        x = np.arange(7)
        col = COL[ctype(cond)]
        ax.plot(x, s.di_count_mean / 1e3, color=col, lw=1.2, label=LAB[ctype(cond)],
                ls=(0, (3, 1.5)) if ctype(cond) == "random" else "-")
        ax.fill_between(x, (s.di_count_mean - s.di_count_sem) / 1e3, (s.di_count_mean + s.di_count_sem) / 1e3,
                        color=col, alpha=0.2, lw=0)
    ax.set_ylim(0, 85)
    ax.axvspan(5.6, 6.4, color="#e3e3e3", lw=0, zorder=0)
    ax.text(6, 83, "null:\nsame\nmodel", ha="center", va="top", fontsize=5.5)
    ax.set_xticks(range(7))
    ax.set_xticklabels(["0→1", "", "2→3", "", "4→5", "", "6→7"])
    ax.set_xlabel("Checkpoint transition")
    ax.set_ylabel("Di-E events (×10³), mean ± s.e.m.")
    ax.legend(frameon=False, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.55, 1.0))
    ax.set_title("Di-E per transition", pad=3)
    # d: layer4 vs avgpool Di-E counts (identical inputs)
    ax = axes[3]
    d = c[c.transition.isin([f"{i}->{i + 1}" for i in range(1, 6)])]
    bad = d[d.approx].set_index(["lane_id", "transition"]).index
    d = d[~d.set_index(["lane_id", "transition"]).index.isin(bad)]
    l4 = d[d.layer == "layer4"].set_index(["lane_id", "transition"]).di_h
    ap = d[d.layer == "avgpool"].set_index(["lane_id", "transition"]).di_h
    ap = ap.reindex(l4.index)
    ax.scatter(ap / 1e3, l4 / 1e3, s=4, color="#333333", alpha=0.6, lw=0)
    mx = max(l4.max(), ap.max()) / 1e3 * 1.05
    ax.plot([0, mx], [0, mx], color="#9a9a9a", lw=0.6, ls="--")
    ax.set_xlim(0, mx)
    ax.set_ylim(0, mx)
    ax.set_xlabel("Di-E, avgpool SAE (×10³)")
    ax.set_ylabel("Di-E, layer4 SAE (×10³)")
    r = np.corrcoef(ap, l4)[0, 1]
    ax.set_title("layer4 vs avgpool SAE", pad=3)
    ax.text(0.04, 0.96, f"identical inputs\nr = {r:.2f}", transform=ax.transAxes, va="top", fontsize=6)
    for i, ax in enumerate(axes):
        ax.text(-0.3, 1.08, chr(97 + i), transform=ax.transAxes, fontsize=8, fontweight="bold")
    fig.tight_layout(w_pad=0.8)
    fig.savefig(f"{OUT}/fig_die_reproduction_null.png")
    fig.savefig(f"{OUT}/fig_die_reproduction_null.pdf")
    plt.close(fig)


def main():
    style()
    rng = np.random.default_rng(RNG_SEED)
    per, chosen, checks = build_per_lane()
    cs = condition_summary(per, rng)
    ct = contrasts(per, rng)
    dec, ident = decomposition(per)
    checks["fine_eq_sc_times_cond_max_absdiff"] = ident
    ed = ed_table_check(per)
    checks["ed_table1_max_absdiff_pp"] = float(ed.diff_pred_vs_ed.abs().max())
    checks["ed_table1_mismatches_gt_0.06pp"] = ed[ed.diff_pred_vs_ed.abs() > 0.06].to_dict("records")
    bad = per[(per.fine_acc - per.val_acc_logged).abs() > 1e-4][["condition", "seed", "epoch", "fine_acc", "val_acc_logged"]]
    checks["logged_vs_prediction_acc_mismatches"] = bad.to_dict("records")
    lanes, dc, tsum, null, nsum, max_dev, c, approx_cells = die_analysis(rng)
    checks["die_frac_vs_process_intensity_max_absdiff"] = max_dev
    checks["approximated_count_cells"] = approx_cells.to_dict("records")
    fig_trajectories(cs)
    fig_contrasts(ct)
    fig_die(lanes, dc, tsum, null, c)
    json.dump({"chosen_alpha": chosen, "checks": checks}, open(f"{OUT}/stats_summary.json", "w"), indent=2)
    print(json.dumps({"chosen_alpha": chosen, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
