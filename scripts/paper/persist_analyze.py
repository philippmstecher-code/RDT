#!/usr/bin/env python3
"""Robustness of the feature-lifespan result that coarse (SAI/SSI) features outlive class (CSI) features
(Fig. 4, Methods 'Feature survival analysis').

Steps (9 core CIFAR-100 runs; records from scripts/paper/persist_lib.py):
 1. Validation: reproduce the stored Spearman rho (feature_survival_all_lanes.json), the
    Fig. 4a construction/refinement lifespan ratio (1.53-3.70x) and the Fig. 4c developmental
    persistence SAI/CSI (2.3-3.7x) with the reference construction (persist_lib.records_reference)
    and stored thresholds.
    Definitions (documented in summary.json):
      Fig. 4a: coarse = SAI > 0.9 OR SSI > t_ssi(stored); fine = CSI > t_csi(stored)
               (overlapping cohorts), pooled record mean lifespan ratio.
      Fig. 4c: records whose chain reaches 'terminal' (incl. terminal-born, lifespan 0),
               lifespan/11, exclusive cohorts SAI > t_sai(stored) > SSI > CSI, ratio of
               seed-mean per architecture.
 2. Real-transition construction (persist_lib.records_real; identical '<k>->terminal' transition excluded, ResNet avgpool
    dropped, pipeline alive mask) under chain rules 'paper' and 'continuity', with
    (a) stored thresholds, (b) alive-only pooled thresholds (CENSUS_robust), (c) a common
    threshold sweep, (d) threshold-free statistics (pooled and layer x birth-stratified
    Spearman; top-vs-bottom quartile mean-lifespan ratio and stratified log-rank).
 3. Counts of runs satisfying 'coarse outlives fine' per variant and sweep fractions.
 4. Same-state refit-stability reference (ResNet layer4<->avgpool per checkpoint; all runs
    '<k>->terminal'), compared with cohort one-step survival and lifespans.

Writes analysis_outputs/paper/PERSIST_robust/{summary.json, tables/*.csv}.
Usage: python scripts/paper/persist_analyze.py
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from persist_lib import (CENSUS_THR, CORE_LANES, DATA, OUT, PAPER_LABEL, RUN_NAME, load_lane,  # noqa: E402
                         records_real, records_reference, terminal_persistence)

TAB = OUT / "tables"
RULES = ("paper", "continuity")
SWEEP_SSI = (0.2, 0.3, 0.4, 0.5)
SWEEP_CSI = (0.2, 0.3, 0.4, 0.5, 0.6)
SWEEP_SAI = (0.9, 0.95)
C = {k: i for i, k in enumerate(("layer_i", "birth", "fid", "life", "maxp", "cens", "ssi", "csi", "sai",
                                  "chain_start"))}


def fnum(x, nd=4):
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, nd)


# statistics
def strat_logrank(life, cens, g1, g2, strata):
    """Discrete-time log-rank stratified by `strata`; group 1 vs group 2.
    Returns (chi2, p, hr_peto) with HR < 1 meaning group 1 lives longer."""
    life = life.astype(int)
    oe = var = 0.0
    for s in np.unique(strata[g1 | g2]):
        ms = strata == s
        a, b = g1 & ms, g2 & ms
        if not a.any() or not b.any():
            continue
        T = int(life[a | b].max()) + 1
        ev = lambda m: np.bincount(life[m & (cens == 0)], minlength=T)[:T].astype(float)  # noqa: E731
        risk = lambda m: np.cumsum(np.bincount(life[m], minlength=T)[:T][::-1])[::-1].astype(float)  # noqa: E731
        d1, d2, n1, n2 = ev(a), ev(b), risk(a), risk(b)
        n, dd = n1 + n2, d1 + d2
        ok = (n > 1) & (dd > 0)
        oe += float((d1[ok] - dd[ok] * n1[ok] / n[ok]).sum())
        var += float((dd[ok] * (n[ok] - dd[ok]) * n1[ok] * n2[ok] / (n[ok] ** 2 * (n[ok] - 1))).sum())
    if var <= 0:
        return np.nan, np.nan, np.nan
    chi2 = oe ** 2 / var
    return chi2, float(stats.chi2.sf(chi2, 1)), float(np.exp(oe / var))


def strat_spearman(x, y, strata):
    """Spearman correlation after rank-centering x and y within strata."""
    xs = np.empty(len(x)); ys = np.empty(len(y))
    keep = np.zeros(len(x), bool)
    for s in np.unique(strata):
        m = strata == s
        if m.sum() < 3:
            continue
        rx = stats.rankdata(x[m]) / m.sum(); ry = stats.rankdata(y[m]) / m.sum()
        xs[m] = rx - rx.mean(); ys[m] = ry - ry.mean(); keep |= m
    xs, ys = xs[keep], ys[keep]
    if xs.std() == 0 or ys.std() == 0:
        return np.nan, np.nan
    r = float(np.corrcoef(xs, ys)[0, 1])
    n = len(xs)
    t = r * np.sqrt((n - 2) / max(1e-12, 1 - r * r))
    return r, float(2 * stats.t.sf(abs(t), n - 2))


def within_quartiles(x, strata):
    """1 = bottom quartile, 4 = top quartile, computed within each stratum (layer)."""
    q = np.zeros(len(x), dtype=np.int8)
    for s in np.unique(strata):
        m = strata == s
        if m.sum() < 8:
            continue
        r = stats.rankdata(x[m]) / m.sum()
        q[m] = np.clip(np.ceil(r * 4), 1, 4).astype(np.int8)
    return q


def cohort_masks(R, thr, definition):
    ssi, csi, sai = R[:, C["ssi"]], R[:, C["csi"]], R[:, C["sai"]]
    A, B, Cc = sai > thr["sai"], ssi > thr["ssi"], csi > thr["csi"]
    if definition == "overlap":           # overlapping cohorts (reference Fig. 4a construction)
        return {"SAI": A, "SSI": B, "coarse": A | B, "fine": Cc}
    if definition == "methods":           # Methods rule (mutually exclusive categories)
        # task-general = SAI > t_sai; superclass-selective = SSI > t_ssi & CSI <= t_csi & not task-general;
        # class-selective = CSI > t_csi & not task-general (whatever SSI: a class lies in one superclass,
        # so SSI is also high for class-selective features). coarse = task-general OR superclass-selective.
        B3 = B & ~Cc & ~A; C3 = Cc & ~A
        return {"SAI": A, "SSI": B3, "coarse": A | B3, "fine": C3}
    B2 = B & ~A; C2 = Cc & ~A & ~B        # exclusive SAI > SSI > CSI
    return {"SAI": A, "SSI": B2, "coarse": A | B2, "fine": C2}


def cohort_stats(R, thr, definition="overlap"):
    m = cohort_masks(R, thr, definition)
    life, cens = R[:, C["life"]], R[:, C["cens"]]
    strata = R[:, C["layer_i"]] * 100 + R[:, C["birth"]]
    mc, mf = m["coarse"], m["fine"]
    out = {"n_coarse": int(mc.sum()), "n_fine": int(mf.sum()), "n_SAI": int(m["SAI"].sum()),
           "n_SSI": int(m["SSI"].sum())}
    mean = lambda mm: float(life[mm].mean()) if mm.any() else np.nan  # noqa: E731
    out.update(mean_coarse=mean(mc), mean_fine=mean(mf), mean_SAI=mean(m["SAI"]), mean_SSI=mean(m["SSI"]))
    out["ratio"] = out["mean_coarse"] / out["mean_fine"] if out["mean_fine"] and mf.any() else np.nan
    # coarse-only vs fine-only when cohorts overlap
    g1, g2 = mc & ~mf, mf & ~mc
    if g1.any() and g2.any():
        chi2, p, hr = strat_logrank(life, cens, g1, g2, strata)
    else:
        chi2 = p = hr = np.nan
    out.update(lr_chi2=chi2, lr_p=p, lr_hr=hr)
    # layer-matched ratio: per-layer ratio of means, averaged with fine-cohort weights
    num = den = 0.0
    for li in np.unique(R[:, C["layer_i"]]):
        ml = R[:, C["layer_i"]] == li
        if (mc & ml).any() and (mf & ml).any():
            w = (mf & ml).sum()
            num += w * life[mc & ml].mean(); den += w * life[mf & ml].mean()
    out["ratio_layer_matched"] = num / den if den > 0 else np.nan
    # layer x birth-matched (direct standardisation to the fine cohort's strata)
    num = den = 0.0
    for s in np.unique(strata[mf]):
        ms = strata == s
        if (mc & ms).any():
            w = (mf & ms).sum()
            num += w * life[mc & ms].mean(); den += w * life[mf & ms].mean()
    out["ratio_birth_layer_matched"] = num / den if den > 0 else np.nan
    birth = R[:, C["birth"]]
    out["mean_birth_coarse"] = float(birth[mc].mean()) if mc.any() else np.nan
    out["mean_birth_fine"] = float(birth[mf].mean()) if mf.any() else np.nan
    return out


def holds(st, sig=False):
    if not np.isfinite(st["ratio"]):
        return False
    ok = st["ratio"] > 1
    if sig:
        ok = ok and np.isfinite(st["lr_p"]) and st["lr_p"] < 0.05 and st["lr_hr"] < 1
    return bool(ok)


def threshold_free(R):
    life, cens = R[:, C["life"]], R[:, C["cens"]]
    layer = R[:, C["layer_i"]]
    strata = layer * 100 + R[:, C["birth"]]
    out = {}
    for k in ("ssi", "csi", "sai"):
        x = R[:, C[k]]
        rho, p = stats.spearmanr(x, life)
        srho, sp = strat_spearman(x, life, strata)
        q = within_quartiles(x, layer)
        top, bot = q == 4, q == 1
        chi2, lp, hr = strat_logrank(life, cens, top, bot, strata)
        out.update({f"rho_{k}": rho, f"p_{k}": p, f"srho_{k}": srho, f"sp_{k}": sp,
                    f"q_ratio_{k}": life[top].mean() / life[bot].mean() if life[bot].mean() > 0 else np.nan,
                    f"q_hr_{k}": hr, f"q_p_{k}": lp})
    return out


def dev_persistence_real(T, thr):
    """T rows: (layer_i, age/n_t, ssi, csi, sai) for features alive at the last real ckpt;
    exclusive cohorts (as Fig. 4c)."""
    ssi, csi, sai, p = T[:, 2], T[:, 3], T[:, 4], T[:, 1]
    A = sai > thr["sai"]; B = (ssi > thr["ssi"]) & ~A; Cc = (csi > thr["csi"]) & ~A & ~(ssi > thr["ssi"])
    m = lambda mm: float(p[mm].mean()) if mm.any() else np.nan  # noqa: E731
    return {"pers_SAI": m(A), "pers_SSI": m(B), "pers_CSI": m(Cc), "n_pers_CSI": int(Cc.sum()),
            "pers_SAI_over_CSI": m(A) / m(Cc) if Cc.any() and m(Cc) > 0 else np.nan,
            "pers_SSI_over_CSI": m(B) / m(Cc) if Cc.any() and m(Cc) > 0 else np.nan}


# main
def main():
    TAB.mkdir(parents=True, exist_ok=True)
    thr_tab = pd.read_csv(CENSUS_THR).set_index("lane")
    ref = json.loads((DATA / "feature_survival_all_lanes.json").read_text())

    val_rows, var_rows, sweep_rows, tf_rows, life_rows = [], [], [], [], []
    persist_orig = {}
    recs = {}
    for lane, arch, seed in CORE_LANES:
        print(f"== {arch} s{seed}", flush=True)
        d, res = load_lane(lane)
        at = res["metadata"]["analysis_thresholds"]
        stored = {"sai": at["sai_task_general_thresh"], "ssi": at["ssi_abstraction_thresh"],
                  "csi": at["csi_differentiation_thresh"]}
        tr = thr_tab.loc[RUN_NAME[arch].format(seed)]
        alive = {"sai": float(tr["alive_thr_sai"]), "ssi": float(tr["alive_thr_ssi"]),
                 "csi": float(tr["alive_thr_csi"])}
        fig4a_stored = {"sai": 0.9, "ssi": stored["ssi"], "csi": stored["csi"]}
        fig4a_alive = {"sai": 0.9, "ssi": alive["ssi"], "csi": alive["csi"]}

        # 1. validation (reference construction)
        Ro = records_reference(d, res)
        rp = ref[PAPER_LABEL[arch].format(seed)]
        st = cohort_stats(Ro, fig4a_stored, "overlap")
        life, cens = Ro[:, C["life"]], Ro[:, C["cens"]]
        A = Ro[:, C["sai"]] > stored["sai"]; B = (Ro[:, C["ssi"]] > stored["ssi"]) & ~A
        Cc = (Ro[:, C["csi"]] > stored["csi"]) & ~A & ~(Ro[:, C["ssi"]] > stored["ssi"])
        tm = cens == 1
        persist_orig.setdefault(arch, []).append([life[A & tm].mean() / 11, life[B & tm].mean() / 11,
                                                  life[Cc & tm].mean() / 11])
        val_rows.append({"arch": arch, "seed": seed, "n_records_repro": len(Ro), "n_records_stored": rp["n_total_tracked"],
                         "rho_ssi_repro": fnum(stats.spearmanr(Ro[:, C["ssi"]], life)[0]),
                         "rho_ssi_stored": rp["ssi_survival_corr"],
                         "rho_csi_repro": fnum(stats.spearmanr(Ro[:, C["csi"]], life)[0]),
                         "rho_csi_stored": rp["csi_survival_corr"],
                         "fig4a_ratio_repro": fnum(st["ratio"], 3)})
        var_rows.append({"variant": "reference", "rule": "reference", "arch": arch, "seed": seed,
                         "thr_sai": 0.9, "thr_ssi": stored["ssi"], "thr_csi": stored["csi"], **st})
        tf_rows.append({"variant": "reference", "rule": "reference", "unit": "all", "arch": arch,
                        "seed": seed, "n": len(Ro), **threshold_free(Ro)})

        # 2. real-transition construction
        for rule in RULES:
            R = records_real(d, res, rule)
            recs[(lane, rule)] = R
            T = terminal_persistence(d, res, rule)
            for vname, thr in (("stored", fig4a_stored), ("alive", fig4a_alive)):
                st = cohort_stats(R, thr, "overlap")
                ste = cohort_stats(R, {**thr, "sai": stored["sai"] if vname == "stored" else alive["sai"]},
                                   "exclusive")
                pthr = stored if vname == "stored" else alive
                var_rows.append({"variant": vname, "rule": rule, "arch": arch, "seed": seed,
                                 "thr_sai": thr["sai"], "thr_ssi": thr["ssi"], "thr_csi": thr["csi"], **st,
                                 "ratio_exclusive_adaptiveSAI": ste["ratio"],
                                 "lr_hr_exclusive": ste["lr_hr"], "lr_p_exclusive": ste["lr_p"],
                                 **dev_persistence_real(T, pthr)})
            for ts, tc, ta in itertools.product(SWEEP_SSI, SWEEP_CSI, SWEEP_SAI):
                st = cohort_stats(R, {"sai": ta, "ssi": ts, "csi": tc}, "overlap")
                sweep_rows.append({"rule": rule, "arch": arch, "seed": seed, "thr_ssi": ts, "thr_csi": tc,
                                   "thr_sai": ta, "ratio": st["ratio"], "ratio_layer_matched": st["ratio_layer_matched"],
                                   "ratio_birth_layer_matched": st["ratio_birth_layer_matched"],
                                   "lr_hr": st["lr_hr"], "lr_p": st["lr_p"], "n_coarse": st["n_coarse"],
                                   "n_fine": st["n_fine"]})
            for unit in ("all", "chain_start"):
                RR = R if unit == "all" else R[R[:, C["chain_start"]] == 1]
                tf_rows.append({"variant": "corrected", "rule": rule, "unit": unit, "arch": arch, "seed": seed,
                                "n": len(RR), **threshold_free(RR)})
            # per-layer / per-cohort one-step survival (for refit comparison)
            for li in np.unique(R[:, C["layer_i"]]):
                ml = R[:, C["layer_i"]] == li
                for vname, thr in (("stored", fig4a_stored), ("alive", fig4a_alive)):
                    m = cohort_masks(R, thr, "overlap")
                    for coh in ("coarse", "fine", "SAI", "SSI"):
                        mm = m[coh] & ml
                        if mm.any():
                            life_rows.append({"rule": rule, "thr": vname, "arch": arch, "seed": seed,
                                              "layer": d["layers"][int(li)], "cohort": coh, "n": int(mm.sum()),
                                              "mean_life": R[mm, C["life"]].mean(),
                                              "p_step": float((R[mm, C["life"]] >= 1).mean()),
                                              "mean_maxp": R[mm, C["maxp"]].mean()})
                    mm = ml
                    life_rows.append({"rule": rule, "thr": vname, "arch": arch, "seed": seed,
                                      "layer": d["layers"][int(li)], "cohort": "all", "n": int(mm.sum()),
                                      "mean_life": R[mm, C["life"]].mean(),
                                      "p_step": float((R[mm, C["life"]] >= 1).mean()),
                                      "mean_maxp": R[mm, C["maxp"]].mean()})

    V = pd.DataFrame(var_rows); S = pd.DataFrame(sweep_rows); TF = pd.DataFrame(tf_rows)
    LF = pd.DataFrame(life_rows); VAL = pd.DataFrame(val_rows)
    VAL.to_csv(TAB / "validation.csv", index=False)
    V.to_csv(TAB / "variants_per_run.csv", index=False)
    S.to_csv(TAB / "sweep_per_run.csv", index=False)
    TF.to_csv(TAB / "threshold_free_per_run.csv", index=False)
    LF.to_csv(TAB / "cohort_life_per_layer.csv", index=False)

    # sweep aggregation
    S["holds"] = S["ratio"] > 1
    S["holds_sig"] = (S["ratio"] > 1) & (S["lr_hr"] < 1) & (S["lr_p"] < 0.05)
    S["holds_lm"] = S["ratio_layer_matched"] > 1
    S["holds_blm"] = S["ratio_birth_layer_matched"] > 1
    S["reversed_sig"] = (S["lr_hr"] > 1) & (S["lr_p"] < 0.05)
    combo = S.groupby(["rule", "thr_ssi", "thr_csi", "thr_sai"]).agg(
        n_runs=("ratio", "size"), n_defined=("ratio", lambda x: int(np.isfinite(x).sum())),
        n_holds=("holds", "sum"), n_holds_sig=("holds_sig", "sum"), n_holds_layer_matched=("holds_lm", "sum"),
        n_holds_birth_layer_matched=("holds_blm", "sum"), n_reversed_sig=("reversed_sig", "sum"),
        median_ratio=("ratio", "median"), min_ratio=("ratio", "min"), max_ratio=("ratio", "max")).reset_index()
    combo.to_csv(TAB / "sweep_per_combo.csv", index=False)

    # summary
    summ = {"validation": {}, "variants": {}, "sweep": {}, "threshold_free": {}}
    pers = {a: np.array(v) for a, v in persist_orig.items()}
    summ["validation"] = {
        "rho_exact_match": bool(((VAL.rho_ssi_repro - VAL.rho_ssi_stored).abs().max() < 1e-3)
                                and ((VAL.rho_csi_repro - VAL.rho_csi_stored).abs().max() < 1e-3)),
        "max_abs_rho_diff": fnum(max((VAL.rho_ssi_repro - VAL.rho_ssi_stored).abs().max(),
                                     (VAL.rho_csi_repro - VAL.rho_csi_stored).abs().max())),
        "n_records_match": bool((VAL.n_records_repro == VAL.n_records_stored).all()),
        "fig4a_ratio_range": [fnum(VAL.fig4a_ratio_repro.min(), 2), fnum(VAL.fig4a_ratio_repro.max(), 2)],
        "fig4c_sai_over_csi_per_arch": {a: fnum(v.mean(0)[0] / v.mean(0)[2], 2) for a, v in pers.items()},
        "fig4c_ssi_over_csi_per_arch": {a: fnum(v.mean(0)[1] / v.mean(0)[2], 2) for a, v in pers.items()},
        "definitions": {
            "fig4a": "coarse = SAI>0.9 OR SSI>stored t_ssi; fine = CSI>stored t_csi (overlapping); pooled mean lifespan of all records (every alive feature at every checkpoint incl. terminal); forward chain incl. identical last->terminal transition; all layers",
            "fig4c": "records whose chain reaches 'terminal' (incl. terminal-born), lifespan/11; exclusive cohorts SAI>stored t_sai, then SSI, then CSI; ratio of 3-seed means"}}
    for (vn, rule), g in V.groupby(["variant", "rule"]):
        e = {"n_runs_ratio_gt1": int((g.ratio > 1).sum()),
             "n_runs_ratio_gt1_and_logrank_sig": int(((g.ratio > 1) & (g.lr_hr < 1) & (g.lr_p < 0.05)).sum()),
             "n_runs_layer_matched_gt1": int((g.ratio_layer_matched > 1).sum()),
             "n_runs_birth_layer_matched_gt1": int((g.ratio_birth_layer_matched > 1).sum()),
             "n_runs_logrank_reversed_sig": int(((g.lr_hr > 1) & (g.lr_p < 0.05)).sum()),
             "birth_layer_matched_range": [fnum(g.ratio_birth_layer_matched.min(), 2), fnum(g.ratio_birth_layer_matched.max(), 2)],
             "mean_birth_coarse_vs_fine": [fnum(g.mean_birth_coarse.mean(), 2), fnum(g.mean_birth_fine.mean(), 2)],
             "ratio_range": [fnum(g.ratio.min(), 2), fnum(g.ratio.max(), 2)],
             "per_arch_mean_ratio": {a: fnum(x.ratio.mean(), 2) for a, x in g.groupby("arch")},
             "per_arch_mean_ratio_layer_matched": {a: fnum(x.ratio_layer_matched.mean(), 2) for a, x in g.groupby("arch")},
             "per_run": [{"arch": r.arch, "seed": int(r.seed), "ratio": fnum(r.ratio, 3),
                          "ratio_layer_matched": fnum(r.ratio_layer_matched, 3),
                          "ratio_birth_layer_matched": fnum(r.ratio_birth_layer_matched, 3), "lr_hr": fnum(r.lr_hr, 3),
                          "lr_p": fnum(r.lr_p, 6), "n_coarse": int(r.n_coarse), "n_fine": int(r.n_fine)}
                         for r in g.itertuples()]}
        if "pers_SAI_over_CSI" in g and g.pers_SAI_over_CSI.notna().any():
            e["dev_persistence_SAI_over_CSI_per_arch"] = {a: fnum(x.pers_SAI.mean() / x.pers_CSI.mean(), 2)
                                                         for a, x in g.groupby("arch")}
            e["dev_persistence_SSI_over_CSI_per_arch"] = {a: fnum(x.pers_SSI.mean() / x.pers_CSI.mean(), 2)
                                                         for a, x in g.groupby("arch")}
            e["n_runs_dev_pers_SAI_gt_CSI"] = int((g.pers_SAI_over_CSI > 1).sum())
            e["ratio_exclusive_adaptiveSAI_range"] = [fnum(g.ratio_exclusive_adaptiveSAI.min(), 2),
                                                      fnum(g.ratio_exclusive_adaptiveSAI.max(), 2)]
            e["n_runs_ratio_exclusive_gt1"] = int((g.ratio_exclusive_adaptiveSAI > 1).sum())
        summ["variants"][f"{vn}|{rule}"] = e
    for rule, g in combo.groupby("rule"):
        summ["sweep"][rule] = {
            "n_combos": len(g),
            "frac_combos_all9_ratio_gt1": fnum((g.n_holds == 9).mean(), 3),
            "frac_combos_all9_sig": fnum((g.n_holds_sig == 9).mean(), 3),
            "frac_combos_all9_layer_matched_gt1": fnum((g.n_holds_layer_matched == 9).mean(), 3),
            "frac_combos_all9_birth_layer_matched_gt1": fnum((g.n_holds_birth_layer_matched == 9).mean(), 3),
            "frac_run_combos_ratio_gt1": fnum(S[S.rule == rule].holds.mean(), 3),
            "frac_run_combos_birth_layer_matched_gt1": fnum(S[S.rule == rule].holds_blm.mean(), 3),
            "frac_run_combos_reversed_sig": fnum(S[S.rule == rule].reversed_sig.mean(), 3),
            "per_arch_frac_run_combos_birth_layer_matched_gt1": {a: fnum(x.holds_blm.mean(), 3) for a, x in S[S.rule == rule].groupby("arch")},
            "frac_run_combos_sig": fnum(S[S.rule == rule].holds_sig.mean(), 3),
            "median_n_runs_holding": float(g.n_holds.median()),
            "min_n_runs_holding": int(g.n_holds.min()),
            "per_arch_frac_run_combos_ratio_gt1": {a: fnum(x.holds.mean(), 3) for a, x in S[S.rule == rule].groupby("arch")},
            "per_arch_frac_run_combos_sig": {a: fnum(x.holds_sig.mean(), 3) for a, x in S[S.rule == rule].groupby("arch")},
            "ratio_range": [fnum(S[S.rule == rule].ratio.min(), 2), fnum(S[S.rule == rule].ratio.max(), 2)]}
    for (vn, rule, unit), g in TF.groupby(["variant", "rule", "unit"]):
        e = {}
        for k in ("ssi", "csi", "sai"):
            e[k] = {"n_rho_pos": int((g[f"rho_{k}"] > 0).sum()),
                    "n_rho_pos_sig": int(((g[f"rho_{k}"] > 0) & (g[f"p_{k}"] < 0.05)).sum()),
                    "n_rho_neg_sig": int(((g[f"rho_{k}"] < 0) & (g[f"p_{k}"] < 0.05)).sum()),
                    "n_srho_pos_sig": int(((g[f"srho_{k}"] > 0) & (g[f"sp_{k}"] < 0.05)).sum()),
                    "n_srho_neg_sig": int(((g[f"srho_{k}"] < 0) & (g[f"sp_{k}"] < 0.05)).sum()),
                    "n_top_q_longer_sig": int(((g[f"q_hr_{k}"] < 1) & (g[f"q_p_{k}"] < 0.05)).sum()),
                    "n_top_q_shorter_sig": int(((g[f"q_hr_{k}"] > 1) & (g[f"q_p_{k}"] < 0.05)).sum()),
                    "rho_range": [fnum(g[f"rho_{k}"].min(), 3), fnum(g[f"rho_{k}"].max(), 3)],
                    "srho_range": [fnum(g[f"srho_{k}"].min(), 3), fnum(g[f"srho_{k}"].max(), 3)],
                    "q_ratio_range": [fnum(g[f"q_ratio_{k}"].min(), 2), fnum(g[f"q_ratio_{k}"].max(), 2)],
                    "per_arch_mean_srho": {a: fnum(x[f"srho_{k}"].mean(), 3) for a, x in g.groupby("arch")}}
        summ["threshold_free"][f"{vn}|{rule}|{unit}"] = e
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: summ[k] for k in ("validation", "sweep")}, indent=1))
    for k, v in summ["variants"].items():
        print(k, {kk: vv for kk, vv in v.items() if kk != "per_run"})
    for k, v in summ["threshold_free"].items():
        print(k, {kk: {x: v[kk][x] for x in ("n_rho_pos_sig", "n_rho_neg_sig", "n_srho_pos_sig", "n_srho_neg_sig",
                                             "n_top_q_longer_sig", "n_top_q_shorter_sig", "srho_range")} for kk in v})


if __name__ == "__main__":
    main()
