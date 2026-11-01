#!/usr/bin/env python3
"""Census check: does the Tg < Ab < Di ordering survive stricter definitions of a superclass-selective feature?

Runs: the 11 census runs (9 core, Tiny ImageNet, independent-init SAE; lane list and metadata from
analysis_outputs/paper/CENSUS_robust/per_lane/*_meta.json, written by census_compute.py) and, descriptively,
the 5 clean 30-epoch ResNet-18 runs (static_noise_s{seed}/lanes/standard_resnet18, thresholds
SAE_outcomes/thresholds_EN.json "common").

Census (logic of census_analyze.census_fracs): alive features (>= 1% of samples active) of the census layers pooled
(ResNet: layer4 dropped, avgpool kept), stored Phase-4 selectivity from sae_results.json:
  Tg = SAI > t_sai;  Ab = SSI > t_ssi & ~Tg;  Di = CSI > t_csi.
Threshold sets: dedup_stored (stored per-run thresholds; Fig. 4d) and a_alive_pooled (CENSUS_robust
thresholds_per_run.csv alive_thr_*). Clean runs: EN common thresholds (SAI fixed 0.9).
Timing / ordering: census_analyze.timing (centre of mass over normalised checkpoint position).

Per feature (hierarchy_sae.feature_table and best-group d' definitions, on |codes| of alive features):
  cover  within the best true superclass, number of its classes carrying >= 50% of the best class's mass;
  d'     best-true-superclass d' (mean_s - mean_rest) / sqrt(0.5 (var_s + var_rest)) + 1e-8, census iff
         above the p95 of best-group d' pooled over all alive features of the same matrix under 10 random
         partitions null_partitions(10, template, 80000).
Variants of the superclass census:
  (a) Ab_a = Ab & cover != 1               (cover = 1 features excluded from the SSI census)
  (b) Ab_b = d' > cut & ~Tg                (d' census replaces the SSI census; ~Tg kept as in Ab)
  (b0) d' > cut                            (descriptive: without the ~Tg exclusion)
Outcome: number of runs in which the COM ordering Tg < Ab < Di holds under each variant. Criterion: the ordering
holds in >= 10 of 11 runs, evaluated under (a) and under (b); the d' census and the dictionary-free transient
(transient_stage) are reported alongside. Both threshold sets are evaluated.
Method notes:
  * d' is computed from per-class first/second moments (algebraically identical to best_dprime, population variance
    clipped at 0) and validated against best_dprime on one matrix per run.
  * The d' census is applied per checkpoint x census layer (the d' null cut is per matrix) and pooled over census
    layers like the SSI census; (b) keeps the census '& ~Tg' exclusion of Ab; (b0) without it is descriptive.
  * Census layers are pooled (census_analyze definition, needed to reproduce CENSUS_robust); the final/pooled layer
    alone (ResNet avgpool, ViT encoder.ln, CCT norm) is reported as a sensitivity.
  * Tiny ImageNet: the lane's Phase-4 superclass map is the template for cover and d' nulls
    (null_partitions(10, template, 80000)); CIFAR-100: hierarchy_lib.true_vec (asserted identical partition).

Outputs (analysis_outputs/paper/DATA_order/census_check/): counts_per_ckpt_layer.csv, census_per_ckpt.csv,
timing.csv, reproduction.csv, validation_ssi_dprime.csv, results.json.
Usage: census_strict.py [--workers 2]
"""
from __future__ import annotations

import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List  # noqa: E402

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from noise_null_lib import Lane  # noqa: E402
from sae_outcomes import checkpoint_info, real_ckpts  # noqa: E402
from census_analyze import ORDER, arch_of, timing  # noqa: E402
from census_compute import ACT_EPS, ALIVE_FRAC  # noqa: E402
from hierarchy_lib import classes, null_partitions, true_vec  # noqa: E402
from hierarchy_sae import masks, ssi  # noqa: E402

REPO = HERE.parents[1]
ROBUST = REPO / "analysis_outputs" / "paper" / "CENSUS_robust"
SAE_outcomes = REPO / "analysis_outputs" / "paper" / "SAE_outcomes"
OUT = REPO / "analysis_outputs" / "paper" / "DATA_order" / "census_check"
RUNS_EXTENDED = Path(RDT_DATA + "/extended/experiments")
CLEAN_SEEDS = [7, 42, 137, 256, 314]
N_NULL_DP = 10       # d' null: null_partitions(10, tv, 80000)
DP_BASE = 80000
FINAL_LAYER = {"ResNet-18": "avgpool", "ViT-Small": "encoder.ln", "CCT-7": "norm"}
VARIANTS = ("orig", "a", "b", "b0")


# d'

def best_dprime(H, gy):
    """Best-group d' per feature: max over groups k of (mean_k - mean_rest) / sqrt(0.5 (var_k + var_rest))."""
    out = np.full(H.shape[1], -np.inf)
    for k in np.unique(gy):
        a, b = H[gy == k], H[gy != k]
        dp = (a.mean(0) - b.mean(0)) / (np.sqrt(0.5 * (a.var(0) + b.var(0))) + 1e-8)
        out = np.maximum(out, dp)
    return out


def best_dprime_moments(S1, S2, nc, g):
    """best_dprime from per-class sums (S1 = sum x, S2 = sum x^2, nc = class sizes); g = group of each class.
    Algebraically identical to best_dprime (population variance); validated against it per run."""
    N = nc.sum()
    T1, T2 = S1.sum(0), S2.sum(0)
    out = np.full(S1.shape[1], -np.inf)
    for k in np.unique(g):
        m = g == k
        na = nc[m].sum()
        nb = N - na
        a1, a2 = S1[m].sum(0), S2[m].sum(0)
        ma, mb = a1 / na, (T1 - a1) / nb
        va = np.maximum(a2 / na - ma ** 2, 0.0)
        vb = np.maximum((T2 - a2) / nb - mb ** 2, 0.0)
        out = np.maximum(out, (ma - mb) / (np.sqrt(0.5 * (va + vb)) + 1e-8))
    return out


# runs

def census_runs() -> List[Dict[str, Any]]:
    thr = {r["lane"]: r for r in csv.DictReader(open(ROBUST / "tables" / "thresholds_per_run.csv"))}
    runs = []
    for p in sorted((ROBUST / "per_lane").glob("*_meta.json")):
        m = json.load(open(p))
        t = thr[m["name"]]
        layers = m["layers"]
        runs.append({
            "run": m["name"], "group": m["group"], "arch": arch_of(m["name"]), "dataset": m["dataset"],
            "dir": m["lane_dir"], "labels": m["labels"], "layers": layers,
            "census_layers": [L for L in layers if not ("layer4" in layers and "avgpool" in layers and L == "layer4")],
            "thr": {"dedup_stored": {"sai": float(t["stored_thr_sai"]), "ssi": float(t["stored_thr_ssi"]),
                                  "csi": float(t["stored_thr_csi"])},
                    "a_alive_pooled": {"sai": float(t["alive_thr_sai"]), "ssi": float(t["alive_thr_ssi"]),
                                       "csi": float(t["alive_thr_csi"])}}})
    runs.sort(key=lambda r: (ORDER[r["group"]], r["arch"], r["run"]))
    return runs


def clean_runs() -> List[Dict[str, Any]]:
    en = json.load(open(SAE_outcomes / "thresholds_EN.json"))["common"]
    out = []
    for s in CLEAN_SEEDS:
        d = RUNS_EXTENDED / f"static_noise_s{s}" / "lanes" / "standard_resnet18"
        lane = Lane(d)
        labels = [c["label"] for c in real_ckpts(checkpoint_info(d, lane.labels))]
        t = en[str(s)]["thresholds"]
        layers = lane.layers
        out.append({"run": f"clean30_resnet18_s{s}", "group": "clean30", "arch": "ResNet-18", "dataset": "cifar100",
                    "dir": str(d), "labels": labels, "layers": layers,
                    "census_layers": [L for L in layers if L != "layer4"],
                    "thr": {"EN_common": {"sai": t["sai_task_general_thresh"], "ssi": t["ssi_abstraction_thresh"],
                                          "csi": t["csi_differentiation_thresh"]}}})
    return out


def group_template(lane: Lane, dataset: str):
    """Superclass of each class index (lane's class order) and the 10 d' null partitions in the same order.
    CIFAR-100: hierarchy_lib.true_vec / null_partitions(10, true_vec, 80000) (canonical class order) re-indexed to the
    lane's class order by class name, and checked to be the same partition as the lane's Phase-4 superclass map.
    Tiny ImageNet: the lane's own superclass map (as the census code) and null_partitions(10, it, 80000)."""
    names = sorted(lane.superclass_groups.keys())
    gv = np.full(lane.n_classes, -1)
    for i, sc in enumerate(names):
        for c in lane.superclass_groups[sc]:
            if c < lane.n_classes:
                gv[c] = i
    assert (gv >= 0).all()
    if dataset == "cifar100":
        canon = classes()
        idx = np.array([canon.index(c.removeprefix("cifar100_")) for c in lane.selected_classes])
        assert len(idx) == lane.n_classes == len(canon) and len(set(idx.tolist())) == len(idx)
        tv = true_vec()[idx]
        pairs = set(zip(tv.tolist(), gv.tolist()))
        assert len(pairs) == len(set(tv.tolist())) == len(set(gv.tolist())), "superclass partitions differ"
        return tv, [p[idx] for p in null_partitions(N_NULL_DP, true_vec(), DP_BASE)]
    return gv, null_partitions(N_NULL_DP, gv, DP_BASE)


def process(run: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    lane = Lane(Path(run["dir"]), dataset_id=run["dataset"])
    n_cls, n_per = lane.n_classes, lane.n_per_class
    n = n_cls * n_per
    alive_min = max(1, int(np.ceil(n * ALIVE_FRAC)))
    y = np.repeat(np.arange(n_cls), n_per)
    tv, nulls = group_template(lane, run["dataset"])
    mk_true = masks(tv)
    mt = mk_true.astype(bool)
    nc = np.full(n_cls, n_per, dtype=np.float64)
    rows, val = [], {"ssi_maxabs_vs_stored": 0.0, "dp_maxabs_vs_verbatim": None, "dp_cut_absdiff": None,
                     "dp_census_mismatch": None}
    did_val = False
    for lab in run["labels"]:
        for L in run["layers"]:
            H = np.abs(lane.load_codes(lab, L)).astype(np.float64)
            assert H.shape[0] == n
            alive = (H > ACT_EPS).sum(0) >= alive_min
            Ha = H[:, alive]
            del H
            st = lane.results["selectivity"][lab][L]
            s_ssi = np.asarray(st["feature_ssi"], dtype=np.float64)[alive]
            s_csi = np.asarray(st["feature_csi"], dtype=np.float64)[alive]
            s_sai = np.asarray(st["feature_sai"], dtype=np.float64)[alive]
            # cover (hierarchy_sae.feature_table)
            M = np.stack([Ha[y == c].mean(0) for c in range(n_cls)], 1)  # f x n_cls
            st_true, best = ssi(M, mk_true)
            if s_ssi.size:
                val["ssi_maxabs_vs_stored"] = max(val["ssi_maxabs_vs_stored"], float(np.max(np.abs(st_true - s_ssi))))
            cover = np.array([(M[i, mt[b]] >= 0.5 * M[i, mt[b]].max()).sum() for i, b in enumerate(best)], dtype=int)
            # d'
            S1 = np.stack([Ha[y == c].sum(0) for c in range(n_cls)])
            S2 = np.stack([(Ha[y == c] ** 2).sum(0) for c in range(n_cls)])
            bt = best_dprime_moments(S1, S2, nc, tv)
            bn = np.stack([best_dprime_moments(S1, S2, nc, p) for p in nulls])
            cut = float(np.percentile(bn, 95)) if bn.size else float("nan")
            if not did_val and Ha.shape[1] > 50 and L == FINAL_LAYER[run["arch"]] and lab == run["labels"][len(run["labels"]) // 2]:
                bt_v = best_dprime(Ha, tv[y])
                bn_v = np.stack([best_dprime(Ha, p[y]) for p in nulls])
                cut_v = float(np.percentile(bn_v, 95))
                val.update({"dp_ckpt": lab, "dp_layer": L, "dp_n_alive": int(Ha.shape[1]),
                            "dp_maxabs_vs_verbatim": float(max(np.max(np.abs(bt - bt_v)), np.max(np.abs(bn - bn_v)))),
                            "dp_cut_absdiff": abs(cut - cut_v),
                            "dp_census_mismatch": int(((bt > cut) != (bt_v > cut_v)).sum())})
                did_val = True
            dp_hi = bt > cut
            for tset, T in run["thr"].items():
                tg = s_sai > T["sai"]
                ssi_hi = s_ssi > T["ssi"]
                ab = ssi_hi & ~tg
                di = s_csi > T["csi"]
                rows.append({
                    "run": run["run"], "group": run["group"], "arch": run["arch"], "tset": tset,
                    "checkpoint": lab, "layer": L, "census_layer": L in run["census_layers"],
                    "n_alive": int(alive.sum()), "n_tg": int(tg.sum()), "n_di": int(di.sum()),
                    "n_ab": int(ab.sum()),
                    "n_ab_c1": int((ab & (cover == 1)).sum()), "n_ab_c2": int((ab & (cover == 2)).sum()),
                    "n_ab_c3p": int((ab & (cover >= 3)).sum()),
                    "n_ssi": int(ssi_hi.sum()), "n_ssi_c1": int((ssi_hi & (cover == 1)).sum()),
                    "n_a": int((ab & (cover != 1)).sum()),
                    "n_b": int((dp_hi & ~tg).sum()), "n_b0": int(dp_hi.sum()),
                    "n_b_and_ab": int((dp_hi & ab).sum()),
                    "n_ab_c1_dp": int((ab & (cover == 1) & dp_hi).sum()),
                    "dprime_cut": cut})
    return {"run": run["run"], "rows": rows, "val": val, "seconds": time.time() - t0}


# analysis

def aggregate(cnt: pd.DataFrame, layer_sel: str) -> pd.DataFrame:
    d = cnt[cnt.census_layer] if layer_sel == "census_layers" else cnt[cnt.layer == cnt.arch.map(FINAL_LAYER)]
    keys = ["run", "group", "arch", "tset", "checkpoint"]
    num = [c for c in d.columns if c.startswith("n_")]
    g = d.groupby(keys, sort=False)[num].sum().reset_index()
    g["layers"] = layer_sel
    for c in ("tg", "di", "ab", "a", "b", "b0"):
        g[f"frac_{c}"] = g[f"n_{c}"] / g["n_alive"]
    for c in ("c1", "c2", "c3p"):
        g[f"ab_share_{c}"] = g[f"n_ab_{c}"] / g["n_ab"].where(g["n_ab"] > 0)
    g["ssi_share_c1"] = g["n_ssi_c1"] / g["n_ssi"].where(g["n_ssi"] > 0)
    return g


def timings(cen: pd.DataFrame, labels_of: Dict[str, List[str]]) -> pd.DataFrame:
    out = []
    for (run, tset, layers), d in cen.groupby(["run", "tset", "layers"], sort=False):
        d = d.set_index("checkpoint").loc[labels_of[run]]
        for v, col in (("orig", "frac_ab"), ("a", "frac_a"), ("b", "frac_b"), ("b0", "frac_b0")):
            t = timing({"tg": d.frac_tg.tolist(), "ab": d[col].tolist(), "di": d.frac_di.tolist()})
            ab = d[col].to_numpy()
            pk = int(np.argmax(d.frac_ab.to_numpy()))
            out.append({"run": run, "group": d.group.iloc[0], "arch": d.arch.iloc[0], "tset": tset, "layers": layers,
                        "variant": v, "n_checkpoints": len(d), **t,
                        "ab_max_frac": float(ab.max()),
                        "ssi_peak_ckpt": d.index[pk], "ssi_peak_pos": pk / (len(d) - 1),
                        "ab_share_c1_at_ssi_peak": float(d.ab_share_c1.iloc[pk]),
                        "ab_share_c2_at_ssi_peak": float(d.ab_share_c2.iloc[pk]),
                        "ab_share_c3p_at_ssi_peak": float(d.ab_share_c3p.iloc[pk]),
                        "ab_share_c1_mean": float(d.ab_share_c1.mean())})
    return pd.DataFrame(out)


def reproduction(cen: pd.DataFrame, tim: pd.DataFrame) -> pd.DataFrame:
    ref = pd.read_csv(ROBUST / "tables" / "census_per_checkpoint.csv", dtype={"checkpoint": str})
    rt = pd.read_csv(ROBUST / "tables" / "census_timing.csv")
    rows = []
    for tset in ("dedup_stored", "a_alive_pooled"):
        mine = cen[(cen.tset == tset) & (cen.layers == "census_layers")]
        for run, d in mine.groupby("run", sort=False):
            r = ref[(ref.variant == tset) & (ref.lane == run)].set_index("checkpoint")
            d = d.set_index("checkpoint")
            common = d.index.intersection(r.index)
            mt = tim[(tim.run == run) & (tim.tset == tset) & (tim.layers == "census_layers") & (tim.variant == "orig")].iloc[0]
            rr = rt[(rt.variant == tset) & (rt.lane == run)].iloc[0]
            rows.append({"run": run, "tset": tset, "n_ckpt_mine": len(d), "n_ckpt_ref": len(r), "n_common": len(common),
                         "n_alive_exact": bool((d.loc[common, "n_alive"].values == r.loc[common, "n_alive"].values).all()),
                         **{f"maxabs_frac_{x}": float(np.max(np.abs(d.loc[common, f"frac_{x}"].values
                                                                      - r.loc[common, f"frac_{x}"].values)))
                            for x in ("tg", "ab", "di")},
                         **{f"com_{x}_absdiff": abs(float(mt[f"com_{x}"]) - float(rr[f"com_{x}"])) for x in ("tg", "ab", "di")},
                         "order_com_mine": bool(mt.order_com), "order_com_ref": bool(rr.order_com)})
    return pd.DataFrame(rows)


def verdict(tim: pd.DataFrame, tset: str) -> Dict[str, Any]:
    t = tim[(tim.group != "clean30") & (tim.tset == tset) & (tim.layers == "census_layers")]
    n = {v: int(t[t.variant == v].order_com.sum()) for v in VARIANTS}
    nr = int((t.variant == "orig").sum())
    fails = {v: t[(t.variant == v) & ~t.order_com].run.tolist() for v in VARIANTS}
    holds_a, holds_b = n["a"] >= 10, n["b"] >= 10
    if holds_a:
        v = "criterion met under (a): census ordering Tg < Ab < Di holds under (a) in >= 10 of 11 runs."
    elif holds_b:
        v = "criterion met under (b), not met under (a): ordering holds under (b) in >= 10 of 11 runs; d' census reported alongside."
    else:
        v = "criterion not met under (a) or (b); dictionary-free transient (transient_stage) reported alongside."
    return {"tset": tset, "n_runs": nr, "n_order_com": n, "fail_order_com": fails, "holds_a": holds_a,
            "holds_b": holds_b, "verdict": v}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    runs = census_runs() + clean_runs()
    labels_of = {r["run"]: r["labels"] for r in runs}
    res = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(process, runs):
            print(f"{r['run']}: {r['seconds']:.0f}s val={r['val']}", flush=True)
            res.append(r)
    order = {r["run"]: i for i, r in enumerate(runs)}
    res.sort(key=lambda r: order[r["run"]])
    cnt = pd.DataFrame([row for r in res for row in r["rows"]])
    cnt.to_csv(OUT / "counts_per_ckpt_layer.csv", index=False)
    val = pd.DataFrame([{"run": r["run"], **r["val"]} for r in res])
    val.to_csv(OUT / "validation_ssi_dprime.csv", index=False)
    cen = pd.concat([aggregate(cnt, "census_layers"), aggregate(cnt, "final_layer")], ignore_index=True)
    cen.to_csv(OUT / "census_per_ckpt.csv", index=False)
    tim = timings(cen, labels_of)
    tim.to_csv(OUT / "timing.csv", index=False)
    rep = reproduction(cen, tim)
    rep.to_csv(OUT / "reproduction.csv", index=False)

    # descriptive: SSI-census peak timing with vs without cover = 1; clean runs; final-layer-only sensitivity
    def counts(layers, tset, grp_clean):
        t = tim[(tim.layers == layers) & (tim.tset == tset) & ((tim.group == "clean30") == grp_clean)]
        return {v: f"{int(t[t.variant == v].order_com.sum())}/{int((t.variant == v).sum())}" for v in VARIANTS}

    summary = {
        "rule": "criterion: COM order Tg<Ab<Di holds in >= 10/11 runs, evaluated under (a) and under (b) "
                "(d' census); the dictionary-free transient (transient_stage) is reported alongside",
        "verdict_dedup_stored": verdict(tim, "dedup_stored"),
        "verdict_a_alive_pooled": verdict(tim, "a_alive_pooled"),
        "sensitivity_final_layer_only": {ts: counts("final_layer", ts, False) for ts in ("dedup_stored", "a_alive_pooled")},
        "clean30_census_layers": counts("census_layers", "EN_common", True),
        "clean30_final_layer": counts("final_layer", "EN_common", True),
        "reproduction": {"all_exact_n_alive": bool(rep.n_alive_exact.all()),
                         "max_abs_frac_diff": float(rep[[c for c in rep.columns if c.startswith("maxabs_frac")]].max().max()),
                         "max_abs_com_diff": float(rep[[c for c in rep.columns if c.endswith("_absdiff")]].max().max()),
                         "order_com_all_equal": bool((rep.order_com_mine == rep.order_com_ref).all()),
                         "ckpt_sets_equal": bool(((rep.n_ckpt_mine == rep.n_ckpt_ref) & (rep.n_common == rep.n_ckpt_ref)).all())},
        "validation": {"ssi_recomputed_vs_stored_maxabs": float(val.ssi_maxabs_vs_stored.max()),
                       "dprime_moments_vs_verbatim_maxabs": float(val.dp_maxabs_vs_verbatim.max()),
                       "dprime_cut_absdiff_max": float(val.dp_cut_absdiff.max()),
                       "dprime_census_mismatch_total": int(val.dp_census_mismatch.sum())},
    }
    with open(OUT / "results.json", "w") as f:
        json.dump(summary, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: summary[k] for k in ("verdict_dedup_stored", "verdict_a_alive_pooled", "reproduction", "validation",
                                              "sensitivity_final_layer_only", "clean30_census_layers", "clean30_final_layer")},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
