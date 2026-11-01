#!/usr/bin/env python3
"""Statistics for the dependency tests T1-T4 (per seed, mean, 95% CI, pass/fail) and a summary figure.

Criteria
  T1  median over (CSI feature x milestone 1..6) of drop_scaffold / mean matched-random drop
      > 1.2 with bootstrap 95% CI excluding 1, in >= 4/5 seeds (primary target: own avgpool
      SAE; robustness: own layer4 SAE).
  T1b (SAE-free) median drop_SCsub / drop_WSCsub > 1.2, CI excluding 1, >= 4/5 seeds.
  T2  SSI-first > CSI-first (sign test p < .05) in >= 4/5 seeds AND SSI-first rate above the
      random-hierarchy control (p_vs_fake < .05); dependence-first likewise (sign test).
  T3  median paired slope difference (SSI - matched control) > 0 with Wilcoxon p < .05,
      in >= 4/5 seeds.
  T4  median e_same / e_other > 1.2 with CI excluding 1 in >= 4/5 seeds (also vs random).
Reads DEPENDENCY/raw/ and T2_precedence_per_seed.csv; writes per-seed tables, dependency_summary.json and
dependency_summary.png to analysis_outputs/paper/DEPENDENCY/.
Usage: python scripts/paper/dependency_stats.py
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from dependency_lib import OUT, boot_ci, t_ci

RAW = OUT / "raw"
SEEDS = [7, 42, 137, 256, 314]


def seed_summary(df, val, crit=1.2):
    rows = []
    for s in SEEDS:
        x = df.loc[df.seed == s, val].replace([np.inf, -np.inf], np.nan).dropna().values
        med = float(np.median(x)) if len(x) else np.nan
        lo, hi = boot_ci(x)
        rows.append(dict(seed=s, n=len(x), median=med, ci_lo=lo, ci_hi=hi, pass_seed=bool(med > crit and lo > 1)))
    out = pd.DataFrame(rows)
    m, lo, hi = t_ci(out["median"])
    return out, dict(mean_of_seed_medians=m, ci95_lo=lo, ci95_hi=hi, n_seeds_pass=int(out.pass_seed.sum()),
                     PASS=bool(out.pass_seed.sum() >= 4))


def t1():
    fa = pd.concat([pd.read_csv(p) for p in sorted(RAW.glob("t1_features_seed*.csv.gz"))])
    fa = fa[(fa.milestone >= 1) & (fa.which == "own")].copy()
    fa["ratio"] = fa.drop_scaf / fa.drop_rand_mean
    # specificity: same ratio for avgpool SSI-category targets (not class-specific)
    spec = fa[fa.set.isin(["SSIq", "SAIq", "CSIq"])].groupby(["set", "layer", "cat", "seed"]).ratio.median().unstack("seed")
    spec.to_csv(OUT / "T1_ratio_by_target_category.csv", float_format="%.4g")
    f = fa[fa.cat == "CSI"].copy()
    f["ratio"] = f.drop_scaf / f.drop_rand_mean
    f["ratio_varnorm"] = f.dropn_scaf / f.dropn_rand_mean
    res, tabs = {}, []
    for st in ("SSIq", "SAIq", "SCAFq", "CSIq"):
        for ly in ("avgpool", "layer4"):
            for val in ("ratio", "ratio_varnorm"):
                d = f[(f.set == st) & (f.layer == ly)]
                tab, summ = seed_summary(d, val)
                tab.insert(0, "test", f"T1_{st}_{ly}_{val}")
                tabs.append(tab)
                res[f"T1_{st}_{ly}_{val}"] = summ
    # per-milestone medians (for figure / temporal profile)
    pm = (f.groupby(["set", "layer", "seed", "milestone"]).ratio.median().reset_index())
    pm.to_csv(OUT / "T1_ratio_by_milestone.csv", index=False, float_format="%.5g")
    # T1b SAE-free subspace contrast (one row per feature x milestone; take set-independent rows once)
    g = f.drop_duplicates(["seed", "milestone", "layer", "fid"]).copy()
    g["ratio_sub"] = g.drop_SCsub / g.drop_WSCsub
    g["ratio_sub_varnorm"] = (g.drop_SCsub / g.var_SCsub) / (g.drop_WSCsub / g.var_WSCsub)
    for ly in ("avgpool", "layer4"):
        for val in ("ratio_sub", "ratio_sub_varnorm"):
            tab, summ = seed_summary(g[g.layer == ly], val)
            tab.insert(0, "test", f"T1b_SCsub_vs_WSCsub_{ly}_{val}")
            tabs.append(tab)
            res[f"T1b_SCsub_vs_WSCsub_{ly}_{val}"] = summ
    g.groupby(["layer", "seed", "milestone"]).ratio_sub.median().reset_index().to_csv(
        OUT / "T1b_subspace_ratio_by_milestone.csv", index=False, float_format="%.5g")
    sl = pd.concat([pd.read_csv(p) for p in sorted(RAW.glob("t1_setlevel_seed*.csv"))])
    sl.to_csv(OUT / "T1_setlevel_all.csv", index=False, float_format="%.5g")
    sets = pd.concat([pd.read_csv(p) for p in sorted(RAW.glob("t1_sets_seed*.csv"))])
    sets.to_csv(OUT / "T1_layer3_sets.csv", index=False, float_format="%.5g")
    return res, pd.concat(tabs)


def t2():
    d = pd.read_csv(OUT / "T2_precedence_per_seed.csv")
    res = {}
    for ly in ("avgpool", "layer4"):
        x = d[d.layer == ly]
        ok_sel = (x.ssi_first > x.csi_first) & (x.p_sign < 0.05) & (x.p_vs_fake < 0.05)
        m, lo, hi = t_ci(x.frac_ssi_first)
        mf, lof, hif = t_ci(x.fake_frac_mean)
        res[f"T2_selectivity_{ly}"] = dict(frac_ssi_first_mean=m, ci95=[lo, hi], fake_frac_mean=mf, fake_ci95=[lof, hif],
                                           n_seeds_pass=int(ok_sel.sum()), PASS=bool(ok_sel.sum() >= 4))
        for k in ("SSIq", "SAIq", "ANY"):
            if f"dep{k}_first" not in x:
                continue
            b, a = x[f"dep{k}_first"], x[f"csi_first_vs_dep{k}"]
            ok = (b > a) & (x[f"p_dep{k}"] < 0.05)
            fr = b / (a + b).replace(0, np.nan)
            m, lo, hi = t_ci(fr)
            res[f"T2_dependence_{k}_{ly}"] = dict(frac_dep_first_mean=m, ci95=[lo, hi], n_seeds_pass=int(ok.sum()),
                                                PASS=bool(ok.sum() >= 4))
    return res


def t3():
    d = pd.concat([pd.read_csv(p) for p in sorted(RAW.glob("t3_lineages_seed*.csv"))])
    res, rows, pairs_all = {}, [], []
    rng = np.random.default_rng(0)
    for ly in ("avgpool", "layer4"):
        ok = []
        for s in SEEDS:
            x = d[(d.seed == s) & (d.layer == ly) & d.slope.notna()]
            ssi = x[x.group == "SSI"]
            ctl = x[x.group == "nonSSI"].copy()
            used, diffs = set(), []
            for _, r in ssi.sample(frac=1, random_state=s).iterrows():
                c = ctl[(ctl.start == r.start) & ~ctl.lineage.isin(used)]
                if c.empty:
                    c = ctl[~ctl.lineage.isin(used)]
                if c.empty:
                    continue
                j = (np.log(c.fire0) - np.log(r.fire0)).abs().idxmin() if c.index.is_unique else None
                cr = c.loc[j]
                used.add(cr.lineage)
                diffs.append(r.slope - cr.slope)
                pairs_all.append(dict(seed=s, layer=ly, ssi_slope=r.slope, ctl_slope=cr.slope, ssi_fire=r.fire0, ctl_fire=cr.fire0,
                                      ssi_eta0=r.eta_start, ctl_eta0=cr.eta_start))
            diffs = np.array(diffs)
            p = float(wilcoxon(diffs, alternative="greater").pvalue) if len(diffs) >= 5 else np.nan
            med = float(np.median(diffs)) if len(diffs) else np.nan
            lo, hi = boot_ci(diffs)
            passed = bool(len(diffs) and med > 0 and p < 0.05)
            ok.append(passed)
            rows.append(dict(test=f"T3_{ly}", seed=s, n_pairs=len(diffs), n_ssi=len(ssi), n_ctl=len(ctl),
                             median_slope_diff_per_epoch=med, ci_lo=lo, ci_hi=hi, p_wilcoxon=p,
                             median_ssi_slope=float(ssi.slope.median()), median_ctl_slope=float(ctl.slope.median()),
                             pass_seed=passed))
        pa = pd.DataFrame([p for p in pairs_all if p["layer"] == ly])
        pooled_p = float(wilcoxon(pa.ssi_slope - pa.ctl_slope, alternative="greater").pvalue)
        m, lo, hi = t_ci([r["median_slope_diff_per_epoch"] for r in rows if r["test"] == f"T3_{ly}"])
        res[f"T3_{ly}"] = dict(mean_of_seed_median_diff=m, ci95=[lo, hi], n_seeds_pass=int(sum(ok)), PASS=bool(sum(ok) >= 4),
                               pooled_pairs=len(pa), pooled_wilcoxon_p=pooled_p,
                               pooled_median_diff=float(np.median(pa.ssi_slope - pa.ctl_slope)))
    pd.DataFrame(pairs_all).to_csv(OUT / "T3_matched_pairs.csv", index=False, float_format="%.5g")
    return res, pd.DataFrame(rows)


def t4():
    d = pd.concat([pd.read_csv(p) for p in sorted(RAW.glob("t4_projection_seed*.csv"))])
    d["r_other"] = d.e_same / d.e_other
    d["r_rand"] = d.e_same / d.e_rand
    res, tabs = {}, []
    for ly in ("avgpool", "layer4"):
        for val in ("r_other", "r_rand"):
            tab, summ = seed_summary(d[d.layer == ly], val)
            tab.insert(0, "test", f"T4_{ly}_{val}")
            tabs.append(tab)
            res[f"T4_{ly}_{val}"] = summ
    d.groupby(["layer", "seed", "milestone"])[["e_same", "e_other", "e_rand", "e_null", "r_other", "r_rand"]].median() \
        .reset_index().to_csv(OUT / "T4_by_milestone.csv", index=False, float_format="%.5g")
    return res, pd.concat(tabs)


def figure(r1, r4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = {"SSIq": "#2a78d6", "SAIq": "#eb6834", "CSIq": "#1baf7a"}
    pm = pd.read_csv(OUT / "T1_ratio_by_milestone.csv")
    sub = pd.read_csv(OUT / "T1b_subspace_ratio_by_milestone.csv")
    t4m = pd.read_csv(OUT / "T4_by_milestone.csv")
    t3p = pd.read_csv(OUT / "T3_matched_pairs.csv")
    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    for st, c in C.items():
        x = pm[(pm.set == st) & (pm.layer == "avgpool")]
        for s in SEEDS:
            y = x[x.seed == s]
            ax[0].plot(y.milestone * 5, y.ratio, color=c, alpha=0.25, lw=1)
        g = x.groupby("milestone").ratio.median()
        ax[0].plot(g.index * 5, g.values, color=c, lw=2, marker="o", ms=5, label=st)
    ax[0].axhline(1, color="#888", lw=1, ls="--"); ax[0].axhline(1.2, color="#bbb", lw=1, ls=":")
    ax[0].set(title="T1 ablation of layer3 set\n(avgpool CSI features)", xlabel="epoch", ylabel="drop / matched-random drop")
    ax[0].legend(frameon=False)
    for ly, c in (("avgpool", "#2a78d6"), ("layer4", "#eb6834")):
        g = sub[sub.layer == ly].groupby("milestone").ratio_sub.median()
        ax[1].plot(g.index * 5, g.values, color=c, lw=2, marker="o", ms=5, label=f"{ly} SAE")
    ax[1].axhline(1, color="#888", lw=1, ls="--")
    ax[1].set(title="T1b SAE-free: superclass vs within-SC\nlayer3 subspace ablation", xlabel="epoch", ylabel="drop(SC) / drop(within-SC)")
    ax[1].legend(frameon=False)
    for ly, c in (("avgpool", "#2a78d6"), ("layer4", "#eb6834")):
        x = t3p[t3p.layer == ly]
        ax[2].scatter(x.ctl_slope, x.ssi_slope, s=14, color=c, alpha=0.6, label=f"{ly} SAE")
    lim = np.nanpercentile(np.abs(np.r_[t3p.ctl_slope, t3p.ssi_slope]), 98)
    ax[2].plot([-lim, lim], [-lim, lim], color="#888", lw=1, ls="--")
    ax[2].set(title="T3 eta² slope, SSI lineage vs\nfiring-matched control", xlabel="control slope (/epoch)", ylabel="SSI slope (/epoch)")
    ax[2].legend(frameon=False)
    for ly, c in (("avgpool", "#2a78d6"), ("layer4", "#eb6834")):
        g = t4m[t4m.layer == ly].groupby("milestone")[["r_other", "r_rand"]].median()
        ax[3].plot(g.index * 5, g.r_other, color=c, lw=2, marker="o", ms=5, label=f"{ly}: same/other SC")
        ax[3].plot(g.index * 5, g.r_rand, color=c, lw=2, marker="s", ms=5, ls="--", label=f"{ly}: same/random")
    ax[3].axhline(1, color="#888", lw=1, ls="--")
    ax[3].set(title="T4 CSI decoder in SSI subspace\n(same superclass vs controls)", xlabel="epoch", ylabel="projection ratio")
    ax[3].legend(frameon=False, fontsize=8)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "dependency_summary.png", dpi=150)


def main():
    r1, tab1 = t1()
    r2 = t2()
    r3, tab3 = t3()
    r4, tab4 = t4()
    pd.concat([tab1, tab4]).to_csv(OUT / "T1_T4_per_seed.csv", index=False, float_format="%.5g")
    tab3.to_csv(OUT / "T3_per_seed.csv", index=False, float_format="%.5g")
    allres = {**r1, **r2, **r3, **r4}
    with open(OUT / "dependency_summary.json", "w") as f:
        json.dump(allres, f, indent=1, default=float)
    for k, v in allres.items():
        print(k, json.dumps(v, default=float))
    figure(r1, r4)


if __name__ == "__main__":
    main()
