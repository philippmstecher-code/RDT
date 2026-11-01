#!/usr/bin/env python3
"""Noise-null comparison of SAE process counts, headline construction/refinement ratio C/R = (Tg-E + Ab-E) / Di-E
without the terminal duplicate transition, and the selective-feature census per checkpoint.

Reads analysis_outputs/paper/NOISE_null/per_lane/*.json (written by noise_null_lane.py) and writes CSV tables,
summary.json and figures to analysis_outputs/paper/NOISE_null/. Lane names are read from the job databases.

Usage: python scripts/paper/noise_null_analyze.py <job database> [<further job database> ...]
"""
from __future__ import annotations

import csv
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noise_null_lib import OUT  # noqa: E402

PROCS = ["tg_h", "ab_h", "di_h", "as_h", "de_h"]
PN = {"tg_h": "Tg-E", "ab_h": "Ab-E", "di_h": "Di-E", "as_h": "As-E", "de_h": "De-E"}
DUP = ("layer4", "avgpool")
CORE_EXP = "abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb"
TIN_EXP = "821eaadb-0160-48f8-9669-ec3d9185c428"
IND_EXP = "independent_sae"
LN_EXP = "0c28799c-ec9d-4e45-ac30-302dff810a8d"
CUR_EXP = "7885dc97-1062-4b2f-9999-61d63b93d097"
DBS = [Path(p) for p in sys.argv[1:]]  # job databases (SQLite, table 'lanes': id, name)


def lane_names() -> Dict[str, str]:
    names: Dict[str, str] = {}
    for db in DBS:
        c = sqlite3.connect(str(db))
        for lid, nm in c.execute("select id, name from lanes"):
            names.setdefault(lid, nm)
    return names


def cr(d: Dict[str, float]) -> float:
    num = d["tg_h"] + d["ab_h"]
    if d["di_h"] <= 0:
        return math.inf if num > 0 else math.nan
    return num / d["di_h"]


def fmt(x: float, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return f"{x:.{nd}f}"


def group_of(exp_dir: str) -> str:
    e = Path(exp_dir).name
    return {CORE_EXP: "core", TIN_EXP: "tinyimagenet", IND_EXP: "indep_init",
            LN_EXP: "label_noise", CUR_EXP: "curriculum"}.get(e, e)


def arch_of(name: str) -> str:
    n = name.lower()
    if "vit" in n:
        return "ViT-Small"
    if "cct" in n:
        return "CCT-7"
    return "ResNet-18"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def com(vals: List[float]) -> float:
    """Centre of mass over normalised transition position (0 = first, 1 = last)."""
    v = np.clip(np.asarray(vals, dtype=float), 0, None)
    if v.sum() <= 0:
        return math.nan
    tau = np.linspace(0, 1, len(v)) if len(v) > 1 else np.zeros(1)
    return float((tau * v).sum() / v.sum())


def main() -> None:
    names = lane_names()
    tdir = OUT / "tables"
    fdir = OUT / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    lanes = []
    for p in sorted((OUT / "per_lane").glob("*.json")):
        d = json.load(open(p))
        d["name"] = names.get(d["lane_id"], d["lane_id"][:8])
        d["group"] = group_of(d["exp_dir"])
        d["arch"] = arch_of(d["name"])
        lanes.append(d)
    order = {"core": 0, "tinyimagenet": 1, "indep_init": 2, "curriculum": 3, "label_noise": 4}
    lanes.sort(key=lambda d: (order.get(d["group"], 9), d["arch"], d["name"]))
    summary: Dict[str, Any] = {}

    # 1. validation
    val_rows = []
    for d in lanes:
        v = d["validation"]
        max_abs = 0
        for x in v["pi"].values():
            for p in PROCS:
                max_abs = max(max_abs, abs(x["stored"][p] - x["recomputed"][p]))
        n_fm = sum(len(d["real"][k]) for k in d["real"])
        val_rows.append({"group": d["group"], "lane": d["name"], "lane_id": d["lane_id"],
                         "layers_run": "+".join(d["layers_run"]),
                         "transitions_pi_exact": f"{v.get('n_pi_exact', 0)}/{v.get('n_pi', 0)}",
                         "pi_max_abs_diff": max_abs,
                         "fm_layer_transitions_checked": n_fm, "fm_mismatches": len(v["fm_mismatch"]),
                         "selectivity_max_abs_diff": v["selectivity_maxabs"]})
    write_csv(tdir / "validation.csv", val_rows)
    summary["validation"] = val_rows

    # helpers per lane
    def keys_of(d):
        ks = list(d["real"].keys())
        term = [k for k in ks if k.endswith("->terminal")]
        real = [k for k in ks if not k.endswith("->terminal")]
        return real, (term[0] if term else None)

    def agg(d, key, layers):
        return {p: sum(d["real"][key][L][p] for L in layers) for p in PROCS}

    # 2. headline C/R (all layers)
    head_rows = []
    for d in lanes:
        if d["layers_run"] != d["layers_all"]:
            continue
        real, term = keys_of(d)
        first = agg(d, real[0], d["layers_all"])
        lastr = agg(d, real[-1], d["layers_all"])
        row = {"group": d["group"], "arch": d["arch"], "lane": d["name"],
               "first_transition": real[0], "CR_first": cr(first),
               "last_real_transition": real[-1], "CR_last_real": cr(lastr),
               "di_first": first["di_h"]}
        if term:
            t = agg(d, term, d["layers_all"])
            row.update({"terminal_transition": term, "CR_terminal_dup": cr(t)})
        # pooled last three real transitions (robustness)
        l3 = {p: sum(agg(d, k, d["layers_all"])[p] for k in real[-3:]) for p in PROCS}
        row["CR_last3_real_pooled"] = cr(l3)
        head_rows.append(row)
    write_csv(tdir / "headline_cr.csv", head_rows)

    def decl(f, l):
        return (l < f) or (math.isinf(f) and not math.isinf(l))

    core = [r for r in head_rows if r["group"] == "core"]
    hs: Dict[str, Any] = {"n_core": len(core)}
    for lab, col in [("with_terminal", "CR_terminal_dup"),
                     ("last_real (terminal removed)", "CR_last_real"),
                     ("last3_real_pooled", "CR_last3_real_pooled")]:
        lasts = [r[col] for r in core]
        nd = sum(decl(r["CR_first"], r[col]) for r in core)
        p = stats.binomtest(nd, len(core), 0.5, alternative="greater").pvalue if core else math.nan
        hs[lab] = {"n_declining": nd, "sign_test_p": p,
                   "n_last_below_1": sum(1 for x in lasts if x < 1),
                   "median_last": float(np.median(lasts)) if lasts else math.nan,
                   "range_last": [float(min(lasts)), float(max(lasts))] if lasts else None,
                   "per_lane": {r["lane"]: r[col] for r in core}}
    hs["CR_first_per_lane"] = {r["lane"]: r["CR_first"] for r in core}
    hs["n_first_di_zero"] = sum(1 for r in core if r["di_first"] == 0)
    for r in head_rows:
        if r["group"] in ("tinyimagenet", "indep_init"):
            hs[r["group"]] = {k: r.get(k) for k in ("lane", "CR_first", "CR_terminal_dup", "CR_last_real",
                                                     "CR_last3_real_pooled")}
    summary["headline"] = hs

    # 3. all-layer terminal-duplicate null (all lanes with a terminal)
    term_rows = []
    for d in lanes:
        if d["layers_run"] != d["layers_all"]:
            continue
        real, term = keys_of(d)
        if not term:
            continue
        T = agg(d, term, d["layers_all"])
        n = len(real)
        for i, k in enumerate(real):
            R = agg(d, k, d["layers_all"])
            row = {"group": d["group"], "arch": d["arch"], "lane": d["name"], "transition": k,
                   "pos": i / (n - 1) if n > 1 else 0.0, "CR_real": cr(R), "CR_null_terminal": cr(T)}
            for p in PROCS:
                row[f"{p}_real"] = R[p]
                row[f"{p}_null"] = T[p]
                row[f"{p}_null_over_real"] = (T[p] / R[p]) if R[p] > 0 else (math.inf if T[p] > 0 else math.nan)
            term_rows.append(row)
    write_csv(tdir / "all_layers_terminal_null.csv", term_rows)

    # summary: null/real at last real transition and first transition, core lanes
    tsum: Dict[str, Any] = {}
    for which, sel in [("first", lambda r: r["pos"] == 0.0), ("last_real", lambda r: r["pos"] == 1.0)]:
        rows = [r for r in term_rows if r["group"] == "core" and sel(r)]
        tsum[which] = {}
        for p in PROCS:
            vals = [r[f"{p}_null_over_real"] for r in rows]
            fin = [v for v in vals if not (math.isnan(v) or math.isinf(v))]
            tsum[which][p] = {"median": float(np.median(fin)) if fin else math.nan,
                              "min": float(min(fin)) if fin else math.nan,
                              "max": float(max(fin)) if fin else math.nan,
                              "n_finite": len(fin), "n": len(vals),
                              "per_lane": {r["lane"]: r[f"{p}_null_over_real"] for r in rows}}
        tsum[which]["CR_real"] = {r["lane"]: r["CR_real"] for r in rows}
        tsum[which]["CR_null_terminal"] = {r["lane"]: r["CR_null_terminal"] for r in rows}
    summary["terminal_null_all_layers"] = tsum

    # 4. ResNet state-matched duplicate null (layer4 vs avgpool)
    dup_rows = []
    lane_dup: Dict[str, Any] = {}
    for d in lanes:
        if not d["is_resnet"] or not d["null_dup"]:
            continue
        real, term = keys_of(d)
        nd_ = d["null_dup"]

        def nmean(lab):
            return {p: 0.5 * (nd_[lab]["layer4->avgpool"][p] + nd_[lab]["avgpool->layer4"][p]) for p in PROCS}

        rows_l = []
        n = len(real)
        for i, k in enumerate(real):
            a, b = k.split("->")
            Na, Nb = nmean(a), nmean(b)
            N4 = [nd_[x][y] for x in (a, b) for y in ("layer4->avgpool", "avgpool->layer4")]
            R = {p: sum(d["real"][k][L][p] for L in DUP) for p in PROCS}  # pooled 2 layers
            row = {"group": d["group"], "lane": d["name"], "transition": k, "pos": i / (n - 1) if n > 1 else 0.0}
            for p in PROCS:
                nm = Na[p] + Nb[p]          # = 2 layers x mean(N_a, N_b)
                nmax = 2 * max(x[p] for x in N4)
                row[f"{p}_real"] = R[p]
                row[f"{p}_null"] = nm
                row[f"{p}_null_max"] = nmax
                row[f"{p}_excess"] = R[p] - nm
                row[f"{p}_excess_conservative"] = R[p] - nmax
                row[f"{p}_null_over_real"] = nm / R[p] if R[p] > 0 else (math.inf if nm > 0 else math.nan)
                if term:
                    Tt = sum(d["real"][term][L][p] for L in DUP)
                    row[f"{p}_null_terminal_sameinit"] = Tt
            Rn = {p: row[f"{p}_null"] for p in PROCS}
            Ex = {p: max(row[f"{p}_excess"], 0.0) for p in PROCS}
            row["CR_real"] = cr(R)
            row["CR_null_state"] = cr(Rn)
            row["CR_excess_pos"] = cr(Ex)
            cb = {x: sum(d["census"][b][L][x] for L in DUP) for x in ("w_tg", "w_ab", "w_di", "n_tg", "n_ab", "n_di", "n_alive")}
            ca = {x: sum(d["census"][a][L][x] for L in DUP) for x in ("w_tg", "w_ab", "w_di")}
            row["CR_census_b_weighted"] = (cb["w_tg"] + cb["w_ab"]) / cb["w_di"] if cb["w_di"] > 0 else math.inf
            row["CR_census_ab_mean_weighted"] = ((ca["w_tg"] + ca["w_ab"] + cb["w_tg"] + cb["w_ab"]) /
                                                 (ca["w_di"] + cb["w_di"])) if (ca["w_di"] + cb["w_di"]) > 0 else math.inf
            rows_l.append(row)
        dup_rows.extend(rows_l)
        # per-lane summary
        def fin_log(x):
            return [math.log(v) for v in x if v > 0 and not math.isinf(v) and not math.isnan(v)]
        crr = [r["CR_real"] for r in rows_l]
        crn = [r["CR_null_state"] for r in rows_l]
        crc = [r["CR_census_ab_mean_weighted"] for r in rows_l]
        ok = [i for i in range(len(rows_l)) if all(0 < v < math.inf for v in (crr[i], crn[i], crc[i]))]
        sp_rn = stats.spearmanr([crr[i] for i in ok], [crn[i] for i in ok]).correlation if len(ok) >= 4 else math.nan
        sp_rc = stats.spearmanr([crr[i] for i in ok], [crc[i] for i in ok]).correlation if len(ok) >= 4 else math.nan
        ls = {"group": d["group"], "n_transitions": len(rows_l), "spearman_CRreal_CRnull": sp_rn,
              "spearman_CRreal_CRcensus": sp_rc, "n_used_corr": len(ok)}
        for p in PROCS:
            ls[f"com_real_{p}"] = com([r[f"{p}_real"] for r in rows_l])
            ls[f"com_excess_{p}"] = com([r[f"{p}_excess"] for r in rows_l])
            ls[f"sum_real_{p}"] = float(sum(r[f"{p}_real"] for r in rows_l))
            ls[f"sum_null_{p}"] = float(sum(r[f"{p}_null"] for r in rows_l))
            ls[f"sum_excess_{p}"] = float(sum(r[f"{p}_excess"] for r in rows_l))
            first, last = rows_l[0], rows_l[-1]
            ls[f"first_null_over_real_{p}"] = first[f"{p}_null_over_real"]
            ls[f"last_null_over_real_{p}"] = last[f"{p}_null_over_real"]
            late = rows_l[-3:]
            sr, sn = sum(r[f"{p}_real"] for r in late), sum(r[f"{p}_null"] for r in late)
            ls[f"late3_null_over_real_{p}"] = sn / sr if sr > 0 else math.nan
            early = rows_l[:2]
            sr, sn = sum(r[f"{p}_real"] for r in early), sum(r[f"{p}_null"] for r in early)
            ls[f"early2_null_over_real_{p}"] = sn / sr if sr > 0 else math.nan
            mid = rows_l[2:-3]
            sr, sn = sum(r[f"{p}_real"] for r in mid), sum(r[f"{p}_null"] for r in mid)
            ls[f"mid_null_over_real_{p}"] = sn / sr if sr > 0 else math.nan
            if term:
                for w, sub in (("early2", rows_l[:2]), ("late3", rows_l[-3:])):
                    sr = sum(r[f"{p}_real"] for r in sub)
                    sn = sum(r[f"{p}_null_terminal_sameinit"] for r in sub)
                    ls[f"{w}_termnull_over_real_{p}"] = sn / sr if sr > 0 else math.nan
            # number of transitions where real exceeds the conservative (max) null
            ls[f"n_trans_real_gt_nullmax_{p}"] = sum(1 for r in rows_l if r[f"{p}_excess_conservative"] > 0)
        ls["CR_real_first"], ls["CR_real_last"] = rows_l[0]["CR_real"], rows_l[-1]["CR_real"]
        ls["CR_null_first"], ls["CR_null_last"] = rows_l[0]["CR_null_state"], rows_l[-1]["CR_null_state"]
        ls["CR_excess_first"], ls["CR_excess_last"] = rows_l[0]["CR_excess_pos"], rows_l[-1]["CR_excess_pos"]
        ls["CR_census_first"], ls["CR_census_last"] = rows_l[0]["CR_census_ab_mean_weighted"], rows_l[-1]["CR_census_ab_mean_weighted"]
        lane_dup[d["name"]] = ls
    write_csv(tdir / "resnet_dup_null_per_transition.csv", dup_rows)
    write_csv(tdir / "resnet_dup_null_per_lane.csv", [{"lane": k, **v} for k, v in lane_dup.items()])

    def grp_stats(groups, key):
        vals = [v.get(key, math.nan) for v in lane_dup.values() if v["group"] in groups]
        vals = [x for x in vals if not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
        if not vals:
            return {"n": 0}
        return {"n": len(vals), "median": float(np.median(vals)), "min": float(min(vals)), "max": float(max(vals))}

    dsum: Dict[str, Any] = {}
    for gname, groups in [("core_resnet", ["core"]), ("tin+indep", ["tinyimagenet", "indep_init"]),
                          ("label_noise", ["label_noise"]), ("curriculum", ["curriculum"]),
                          ("all_resnet", ["core", "tinyimagenet", "indep_init", "label_noise", "curriculum"])]:
        g: Dict[str, Any] = {}
        for p in PROCS:
            for w in ("early2", "mid", "late3"):
                g[f"{w}_null_over_real_{p}"] = grp_stats(groups, f"{w}_null_over_real_{p}")
            for w in ("early2", "late3"):
                g[f"{w}_termnull_over_real_{p}"] = grp_stats(groups, f"{w}_termnull_over_real_{p}")
            g[f"com_real_{p}"] = grp_stats(groups, f"com_real_{p}")
            g[f"com_excess_{p}"] = grp_stats(groups, f"com_excess_{p}")
        for k in ("spearman_CRreal_CRnull", "spearman_CRreal_CRcensus", "CR_real_first", "CR_real_last",
                  "CR_null_first", "CR_null_last", "CR_excess_first", "CR_excess_last",
                  "CR_census_first", "CR_census_last"):
            g[k] = grp_stats(groups, k)
        sel = [v for v in lane_dup.values() if v["group"] in groups]
        g["n_lanes"] = len(sel)
        g["n_com_excess_ab_lt_di"] = sum(1 for v in sel if v["com_excess_ab_h"] < v["com_excess_di_h"])
        g["n_com_real_ab_lt_di"] = sum(1 for v in sel if v["com_real_ab_h"] < v["com_real_di_h"])
        g["n_CRexcess_decline"] = sum(1 for v in sel if decl(v["CR_excess_first"], v["CR_excess_last"]))
        g["n_CRreal_decline"] = sum(1 for v in sel if decl(v["CR_real_first"], v["CR_real_last"]))
        g["n_CRnull_decline"] = sum(1 for v in sel if decl(v["CR_null_first"], v["CR_null_last"]))
        g["n_CRexcess_last_lt1"] = sum(1 for v in sel if v["CR_excess_last"] < 1)
        g["n_CRreal_last_lt1"] = sum(1 for v in sel if v["CR_real_last"] < 1)
        g["n_CRnull_last_lt1"] = sum(1 for v in sel if v["CR_null_last"] < 1)
        g["sum_excess_share"] = {p: (sum(v[f"sum_excess_{p}"] for v in sel) / sum(v[f"sum_real_{p}"] for v in sel))
                                 if sum(v[f"sum_real_{p}"] for v in sel) > 0 else math.nan for p in PROCS}
        dsum[gname] = g
    summary["resnet_dup_null"] = dsum

    # 5. census per checkpoint (all lanes, all layers run)
    cen_rows = []
    for d in lanes:
        for lab, layers in d["census"].items():
            tot = {x: sum(v[x] for v in layers.values()) for x in next(iter(layers.values()))}
            r = {"group": d["group"], "arch": d["arch"], "lane": d["name"], "checkpoint": lab,
                 "layers": "+".join(layers.keys()), **tot}
            r["frac_tg"] = tot["n_tg"] / tot["n_alive"] if tot["n_alive"] else math.nan
            r["frac_ab"] = tot["n_ab"] / tot["n_alive"] if tot["n_alive"] else math.nan
            r["frac_di"] = tot["n_di"] / tot["n_alive"] if tot["n_alive"] else math.nan
            r["CR_census_count"] = (tot["n_tg"] + tot["n_ab"]) / tot["n_di"] if tot["n_di"] else math.inf
            r["CR_census_weighted"] = (tot["w_tg"] + tot["w_ab"]) / tot["w_di"] if tot["w_di"] else math.inf
            cen_rows.append(r)
            for L, v in layers.items():
                pass
    write_csv(tdir / "census_per_checkpoint.csv", cen_rows)

    # census timing: centre of mass and argmax (over checkpoint position) of the selective fractions
    ct_rows = []
    byl: Dict[str, List[Dict[str, Any]]] = {}
    for r in cen_rows:
        byl.setdefault((r["group"], r["lane"]), []).append(r)
    for (g, ln), rr in byl.items():
        row = {"group": g, "lane": ln, "n_checkpoints": len(rr)}
        for x in ("tg", "ab", "di"):
            fr = [r[f"frac_{x}"] for r in rr]
            row[f"com_frac_{x}"] = com(fr)
            row[f"argmax_frac_{x}"] = int(np.nanargmax(fr)) / (len(fr) - 1)
            row[f"frac_{x}_first"], row[f"frac_{x}_last"] = fr[0], fr[-1]
            row[f"frac_{x}_max"] = float(np.nanmax(fr))
        row["order_com_tg_lt_ab_lt_di"] = row["com_frac_tg"] < row["com_frac_ab"] < row["com_frac_di"]
        row["order_argmax_tg_le_ab_le_di"] = row["argmax_frac_tg"] <= row["argmax_frac_ab"] <= row["argmax_frac_di"]
        row["CR_census_first"] = rr[0]["CR_census_weighted"]
        row["CR_census_last"] = rr[-1]["CR_census_weighted"]
        ct_rows.append(row)
    write_csv(tdir / "census_timing.csv", ct_rows)
    summary["census_timing"] = ct_rows

    # census vs C/R real (all layers, core lanes): spearman per lane over real transitions
    cc = {}
    for d in lanes:
        if d["layers_run"] != d["layers_all"]:
            continue
        real, term = keys_of(d)
        xs, ys = [], []
        for k in real:
            b = k.split("->")[1]
            a = k.split("->")[0]
            R = agg(d, k, d["layers_all"])
            ca = {x: sum(d["census"][a][L][x] for L in d["layers_all"]) for x in ("w_tg", "w_ab", "w_di")}
            cb = {x: sum(d["census"][b][L][x] for L in d["layers_all"]) for x in ("w_tg", "w_ab", "w_di")}
            c = (ca["w_tg"] + ca["w_ab"] + cb["w_tg"] + cb["w_ab"]) / max(1e-9, ca["w_di"] + cb["w_di"])
            x, y = cr(R), c
            if 0 < x < math.inf and 0 < y < 1e8:
                xs.append(x)
                ys.append(y)
        cc[d["name"]] = {"group": d["group"], "n": len(xs),
                         "spearman": stats.spearmanr(xs, ys).correlation if len(xs) >= 4 else math.nan}
    summary["census_vs_CR_all_layers"] = cc

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=lambda o: None if (isinstance(o, float) and math.isnan(o)) else str(o))
    print(json.dumps({"headline": hs, "dup": {k: {kk: vv for kk, vv in v.items() if kk.startswith("n_")} for k, v in dsum.items()}},
                     indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
