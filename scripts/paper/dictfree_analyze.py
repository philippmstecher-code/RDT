#!/usr/bin/env python3
"""Dictionary-free developmental program, analysis step: reads the dictfree_compute.py caches and builds unit-level
censuses under all variants, population curves, timing (same criteria as census_analyze.timing) and comparisons with
the SAE census. Writes analysis_outputs/paper/DICTFREE/population/tables/*.csv.

Usage: dictfree_analyze.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dictfree_lib import OUT, REPO, open_lanes  # noqa: E402

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from census_analyze import timing  # noqa: E402

TAB = OUT / "tables"
FLOOR = {"ssi": 0.1, "csi": 0.15}
CSI_CAP = 0.7
SAE_RUN = {"core_ResNet-18_s42": "ResNet18-100class-10ms-seed42", "core_ViT-Small_s42": "VitSmall-100class-10ms-seed42",
           "core_CCT-7_s42": "CCT7-100class-10ms-seed42"}


def load_npz(key):
    z = np.load(OUT / "cache" / f"{key}.npz")
    d = {}
    for k in z.files:
        lab, L, b, nm = k.split("|")
        d.setdefault((lab, L, b), {})[nm] = z[k]
    return d


def pooled_thresholds(D, basis):
    """p95 of pooled null draws (alive units, all checkpoints & layers of the lane) per index."""
    out = {}
    for nm in ("ssi_abs", "csi_abs", "ssi_shift", "csi_shift", "fsc", "fwsc", "fcls"):
        vals = []
        for (lab, L, b), e in D.items():
            if b != basis:
                continue
            a = e["alive"]
            nul = e[f"{nm}_null"][:, a].astype(np.float64).ravel()
            if nm.startswith("csi"):
                nul = nul[nul > 1e-10]
            vals.append(nul)
        v = np.concatenate(vals)
        t = float(np.percentile(v, 95))
        if nm.startswith("ssi"):
            t = max(t, FLOOR["ssi"])
        if nm.startswith("csi"):
            t = max(min(t, CSI_CAP), FLOOR["csi"])
        out[nm] = t
    return out


def census_rows(ln, D, sae_thr):
    rows = []
    thr = {b: pooled_thresholds(D, b) for b in {k[2] for k in D}}
    for (lab, L, b), e in D.items():
        a = e["alive"]
        na = int(a.sum())
        base = {"lane": ln["key"], "arch": ln["arch"], "group": ln["group"], "seed": ln["seed"], "checkpoint": lab,
                "layer": L, "basis": b, "is_pooled": L == ln["pooled"], "n_units": int(a.size), "n_alive": na}
        if na == 0:
            continue
        for var in ("abs", "shift"):
            ssi, csi, sai = e[f"ssi_{var}"][a], e[f"csi_{var}"][a], e[f"sai_{var}"][a]
            csi = np.nan_to_num(csi, nan=0.0)
            schemes = {"fl": (e[f"ssi_{var}_p95"][a], e[f"csi_{var}_p95"][a]),
                       "pooled": (thr[b][f"ssi_{var}"], thr[b][f"csi_{var}"]),
                       "sae": (sae_thr["ssi"], sae_thr["csi"])}
            for sch, (ts, tc) in schemes.items():
                for tgdef, tg in (("tg09", sai > 0.9), ("tgnull", sai >= e[f"sai_{var}_p05"][a])):
                    ab_incl = ssi > ts
                    rows.append({**base, "variant": var, "scheme": sch, "tgdef": tgdef,
                                 "frac_tg": tg.mean(), "frac_ab": (ab_incl & ~tg).mean(), "frac_ab_incl": ab_incl.mean(),
                                 "frac_di": (csi > tc).mean(), "mean_ssi": ssi.mean(), "mean_csi": csi.mean(),
                                 "mean_sai": sai.mean(),
                                 "thr_ssi": float(np.mean(ts)), "thr_csi": float(np.mean(tc))})
        T = e["T"][a]
        fsc, fwsc, fcls = e["bsc"][a] / T, (e["bcls"][a] - e["bsc"][a]) / T, e["bcls"][a] / T
        for sch, (t1, t2, t3) in {"fl": (e["fcls_p95"][a], e["fsc_p95"][a], e["fwsc_p95"][a]),
                                  "pooled": (thr[b]["fcls"], thr[b]["fsc"], thr[b]["fwsc"])}.items():
            rows.append({**base, "variant": "var", "scheme": sch, "tgdef": "noclass",
                         "frac_tg": (fcls <= t1).mean(), "frac_ab": (fsc > t2).mean(), "frac_ab_incl": (fsc > t2).mean(),
                         "frac_di": (fwsc > t3).mean(), "mean_ssi": fsc.mean(), "mean_csi": fwsc.mean(),
                         "mean_sai": 1 - fcls.mean(), "thr_ssi": float(np.mean(t2)), "thr_csi": float(np.mean(t3))})
    return rows, thr


def sae_curves():
    c = pd.read_csv(REPO / "analysis_outputs/paper/CENSUS_robust/tables/census_per_checkpoint.csv")
    c = c[(c.variant == "a_alive_pooled") & (c.checkpoint != "terminal")]
    out = {}
    for k, run in SAE_RUN.items():
        x = c[c.lane == run].copy()
        x["ck"] = x.checkpoint.astype(int)
        x = x.sort_values("ck")
        out[k] = {"tg": x.frac_tg.values, "ab": x.frac_ab.values, "di": x.frac_di.values, "ck": x.ck.values}
    e = pd.read_csv(REPO / "analysis_outputs/paper/SAE_outcomes/census_EN.csv")
    for s in (7, 42, 137):
        x = e[(e.lane == f"clean__s{s}") & (e.tset == "common") & (e.layer == "avgpool")].sort_values("checkpoint")
        out[f"clean_ResNet-18_s{s}"] = {"tg": x.sai.values, "ab": x.ssi.values, "di": x.csi.values,
                                        "ck": x.checkpoint.values}
    return out


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    lanes = open_lanes()
    lj = {l["key"]: l for l in json.load(open(REPO / "analysis_outputs/paper/CORE_lanes/lanes.json"))["lanes"]}
    pop = pd.concat([pd.read_csv(OUT / "cache" / f"{ln['key']}_pop.csv") for ln in lanes], ignore_index=True)
    pop["checkpoint"] = pop.checkpoint.astype(str)
    pop.to_csv(TAB / "population_per_checkpoint.csv", index=False)
    rows, thr_rows = [], []
    for ln in lanes:
        D = load_npz(ln["key"])
        sae_thr = lj[ln["key"]]["thr_null"] if ln["key"] in lj else lj[f"clean_ResNet-18_s{ln['seed']}"]["thr_null"]
        r, thr = census_rows(ln, D, sae_thr)
        rows += r
        for b, t in thr.items():
            thr_rows.append({"lane": ln["key"], "basis": b, **t})
        print("census", ln["key"], flush=True)
    cen = pd.DataFrame(rows)
    order = {ln["key"]: {c["label"]: c["idx"] for c in ln["ckpts"]} for ln in lanes}
    cen["idx"] = [order[l][c] for l, c in zip(cen.lane, cen.checkpoint)]
    cen = cen.sort_values(["lane", "variant", "scheme", "tgdef", "basis", "layer", "idx"])
    cen.to_csv(TAB / "unit_census_per_checkpoint.csv", index=False)
    pd.DataFrame(thr_rows).to_csv(TAB / "unit_pooled_thresholds.csv", index=False)

    # timing (census criteria) for every candidate triple
    tim = []
    for keys, g in cen.groupby(["lane", "variant", "scheme", "tgdef", "basis", "layer"]):
        g = g.sort_values("idx")
        for kind, cols, bsub in (("frac", ("frac_tg", "frac_ab", "frac_di"), False),
                                 ("mean", ("mean_sai", "mean_ssi", "mean_csi"), True)):
            r = timing({"tg": g[cols[0]].values, "ab": g[cols[1]].values, "di": g[cols[2]].values}, baseline_sub=bsub)
            tim.append({**dict(zip(["lane", "variant", "scheme", "tgdef", "basis", "layer"], keys)), "kind": kind,
                        "is_pooled": bool(g.is_pooled.iloc[0]), **r})
    for (lane, layer), g in pop.groupby(["lane", "layer"]):
        g = g.sort_values("idx")
        for sp in ("raw", "wht"):
            f = {"tg": 1 - g[f"{sp}_fcls"].values, "ab": g[f"{sp}_fsc"].values, "di": g[f"{sp}_fwsc"].values}
            for kind, bsub in (("pop_frac", False), ("pop_frac_minsub", True)):
                r = timing(f, baseline_sub=bsub)
                tim.append({"lane": lane, "variant": f"pop_{sp}", "scheme": "-", "tgdef": "-", "basis": sp,
                            "layer": layer, "kind": kind, "is_pooled": bool(g.is_pooled.iloc[0]), **r})
    tim = pd.DataFrame(tim)
    tim.to_csv(TAB / "timing_all_variants.csv", index=False)
    pooled = tim[tim.is_pooled]
    summ = (pooled.groupby(["variant", "scheme", "tgdef", "basis", "kind"])
            .agg(n=("lane", "size"), n_order_com=("order_com", "sum"), n_order_argmax=("order_argmax", "sum"),
                 n_ab_rise_fall=("ab_rise_fall", "sum"), n_tg_peak_init=("tg_peak_init", "sum"),
                 n_all3=("all3", "sum"), mean_com_tg=("com_tg", "mean"), mean_com_ab=("com_ab", "mean"),
                 mean_com_di=("com_di", "mean"))
            .reset_index().sort_values(["n_all3", "n_order_com"], ascending=False))
    summ.to_csv(TAB / "timing_summary_pooled_layer.csv", index=False)
    print(summ.to_string())

    # comparison with SAE census Ab curve (pooled layer)
    sae = sae_curves()
    comp = []
    cand = {"pop_raw_share_sc": ("pop", "raw_share_sc"), "pop_wht_share_sc": ("pop", "wht_share_sc"),
            "pop_raw_fsc": ("pop", "raw_fsc"), "pop_wht_fsc": ("pop", "wht_fsc"),
            "pop_raw_share_sc_z": ("pop", "raw_share_sc_z"),
            "ncc_sc20_minus_wsc5_norm": ("pop", None), "net_err_insc": ("pop", "net_err_insc"),
            "raw_ncc_err_insc": ("pop", "raw_ncc_err_insc")}
    for key, s in sae.items():
        g = pop[(pop.lane == key) & pop.is_pooled].sort_values("idx")
        n = min(len(g), len(s["ab"]))
        for nm, (_, col) in cand.items():
            if col is None:
                v = (g.raw_ncc_sc20.values - 0.05) / 0.95 - (g.raw_ncc_wsc5.values - 0.2) / 0.8
            else:
                v = g[col].values
            v, a = v[:n], s["ab"][:n]
            rho = stats.spearmanr(v, a).correlation
            comp.append({"lane": key, "measure": nm, "n_ck": n, "spearman_vs_sae_ab": rho,
                         "argmax_idx": int(np.nanargmax(v)), "sae_ab_argmax_idx": int(np.argmax(a)),
                         "last_over_max": float(v[-1] / np.nanmax(v)), "sae_ab_last_over_max": float(a[-1] / a.max())})
        for (var, sch, tgd, b) in (("abs", "pooled", "tg09", "raw"), ("abs", "sae", "tg09", "raw"),
                                   ("abs", "fl", "tg09", "raw"), ("shift", "pooled", "tg09", "raw"),
                                   ("var", "fl", "noclass", "raw"), ("var", "pooled", "noclass", "raw"),
                                   ("abs", "pooled", "tg09", "ica"), ("var", "fl", "noclass", "ica"),
                                   ("abs", "pooled", "tg09", "pca"), ("var", "fl", "noclass", "pca")):
            g = cen[(cen.lane == key) & cen.is_pooled & (cen.variant == var) & (cen.scheme == sch) &
                    (cen.tgdef == tgd) & (cen.basis == b)].sort_values("idx")
            v = g.frac_ab.values[:n]
            a = s["ab"][:n]
            comp.append({"lane": key, "measure": f"unit_{var}_{sch}_{tgd}_{b}_frac_ab", "n_ck": n,
                         "spearman_vs_sae_ab": stats.spearmanr(v, a).correlation,
                         "argmax_idx": int(np.argmax(v)), "sae_ab_argmax_idx": int(np.argmax(a)),
                         "last_over_max": float(v[-1] / v.max()) if v.max() > 0 else np.nan,
                         "sae_ab_last_over_max": float(a[-1] / a.max())})
    pd.DataFrame(comp).to_csv(TAB / "compare_with_sae_ab.csv", index=False)
    pd.DataFrame([{"lane": k, "ck": c, "sae_tg": t, "sae_ab": a, "sae_di": d} for k, s in sae.items()
                  for c, t, a, d in zip(s["ck"], s["tg"], s["ab"], s["di"])]).to_csv(TAB / "sae_census_reference.csv",
                                                                                  index=False)


if __name__ == "__main__":
    main()
