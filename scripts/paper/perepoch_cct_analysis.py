#!/usr/bin/env python3
"""Per-epoch CCT-7 run (run_perepoch_cct.py): developmental program at one-epoch resolution, compared with the
accuracy-spaced checkpoints of the core CCT-7 runs.

Input  (read-only): <lane>/sae_analysis/{sae_results.json, activations/<epoch>/<layer>.npy},
                    <lane>/dev_snapshots/milestone_<epoch>/individual_multilayer_activations.pt, devtrain_snapshots.json
                    core CCT-7 lanes (abf16d20; seeds 42/137/256): devtrain_snapshots.json, devtrain_metrics.json,
                    sae_results.json (process_intensity).
Thresholds: the run's stored per-run pipeline thresholds (metadata.analysis_thresholds) everywhere.

1  Process events per transition (sae_results.json process_intensity, all 6 SAE layers, raw counts): Tg-E, Ab-E,
   Di-E counts and shares of Tg+Ab+Di; C/R = (Tg+Ab)/max(Di,1); centre of mass (COM, transition index t = epoch
   t -> t+1, weighted by count); strict order Tg < Ab < Di with both gaps >= 0.05 transitions (smaller gaps = tie,
   i.e. not ordered); first transition with Di-E > Ab-E and first transition from which Di-E > Ab-E at every later
   transition. Windows: all transitions, epochs 0-10, epochs 0-15.
2  Feature-type census per epoch (census_exclusive.counts; alive = active on >= 1 % of images; all 6 CCT layers):
   overlapping (key 'overlapping' in the outputs) Tg = SAI>t, Ab = SSI>t & ~Tg, Di = CSI>t, and exclusive variants; census_analyze.timing
   (COM order over normalised checkpoint position, argmax order, Ab rise-and-fall).
3  SAE-free superclass share (Extended Data Fig. 1g,h measure; producers dictfree_compute.population /
   transient_stage): superclass share of split-half between-class variance of the unwhitened pooled layer ('norm'),
   100 random 20 x 5 groupings as null, 20-replicate image bootstrap SD, transient_stage.transient rule.
   The core seed-42 values (DICTFREE/population population table) are listed for comparison.
4  Sparse vs dense: the per-epoch checkpoints subsampled at each core CCT-7 run's checkpoint epochs
   (round(weight_updates / 391), duplicates dropped; seed-42 grid primary, seeds 137/256 secondary). Per grid
   interval a -> b and layer: direct pipeline match a -> b on the per-epoch run's own SAE codes (matching_lib.pipeline_match
   = saeanalysis._phase5_feature_matching; Phase-6 labels matching_lib.Labeler with the stored thresholds) against the sum
   of the per-epoch counts inside the interval: births/deaths/stable/transformed (feature level), Tg/Ab/Di labels
   (feature level), sample-weighted As/De/Tg/Ab/Di (== process_intensity). Missed fraction = 1 - direct/dense.
   Chains (matching_lib.chain_compare): features directly stable a -> b vs stable at every epoch step. The core run's
   own stored counts per interval are listed for context (different run and thresholds). Also the process COM / order
   computed on the coarse (direct) grid counts.
5  Flicker (present -> absent -> present): a feature alive at epoch k that the pipeline counts as
   died at k -> k+1 but that is directly stable (r >= 0.5) at k -> k+2; rates per alive feature and per death,
   also among "consensus" features (stable at k-1 -> k); blip births (born at k+1, died at k+1 -> k+2).
Per-step recomputation is validated against the stored feature_matching counts and process_intensity.
0  Reproduction check: per-epoch validation accuracy vs the core CCT-7 runs (devtrain_metrics.json).

Usage: python scripts/paper/perepoch_cct_analysis.py [--lane DIR] [--out DIR] [--skip-matching]
Writes <out>/{results.json, process_per_transition.csv, census_per_epoch.csv, sae_free_share_per_epoch.csv,
steps_per_layer.csv, sparse_vs_dense_per_layer.csv, sparse_vs_dense.csv, flicker_per_layer.csv, flicker_per_epoch.csv}.
"""
from __future__ import annotations

import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from matching_lib import (EPS_ACT, Labeler, Lane, chain_compare, feature_counts, outcome_from_pipeline,  # noqa: E402
                    pipeline_match)
from transient_stage import transient  # noqa: E402
from census_analyze import timing  # noqa: E402
from census_exclusive import counts as census_counts  # noqa: E402
from dictfree_compute import Ctx, population  # noqa: E402
from dictfree_lib import superclass_of_class  # noqa: E402
from dictfree_share_splits import N_SPLIT, share, splits  # noqa: E402

REPO = HERE.parents[1]
LANE_DEFAULT = Path(RDT_DATA + "/extended/experiments/perepoch_cct/lanes/cct7_seed42")
OUT_DEFAULT = REPO / "analysis_outputs" / "paper" / "PEREPOCH_cct"
CORE = Path(RDT_DATA + "/core/experiments/abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb/lanes")
CORE_LANES = {42: "4bebc329-c26a-4eb4-b18c-7586e513f49e", 137: "edf8275e-c99f-4436-a13d-4ed50f072a66",
             256: "445dd4d4-770a-4f33-b82b-c0bc7baed9a7"}
DICTFREE_POP = REPO / "analysis_outputs" / "paper" / "DICTFREE" / "population" / "tables" / "population_per_checkpoint.csv"
BPE = 391
P = ("tg_h", "ab_h", "di_h")
TIE = 0.05
POOLED = "norm"
WINDOWS = {"all": None, "epochs_0_10": 10, "epochs_0_15": 15}
W_KEYS = ("w_tg_h", "w_ab_h", "w_di_h", "w_as_h", "w_de_h")
F_KEYS = ("n_born", "n_died", "n_stable", "n_transformed", "f_tg", "f_ab", "f_di", "f_tg_born", "f_ab_born", "f_di_born")


def jdump(o, path):
    def conv(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return None if not np.isfinite(x) else float(x)
        if isinstance(x, np.bool_):
            return bool(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        return str(x)
    with open(path, "w") as f:
        json.dump(o, f, indent=1, default=conv)


# 0 reproduction
def load_snaps(lane_dir: Path):
    s = json.load(open(lane_dir / "dev_snapshots" / "metrics" / "devtrain_snapshots.json"))
    s = [x for x in s if x["snapshot_type"] != "final" or x["weight_updates"] not in
         {y["weight_updates"] for y in s if y["snapshot_type"] != "final"}]  # drop a final that duplicates a milestone
    for x in s:
        assert x["weight_updates"] == BPE * x["milestone_index"], x
    return s


def reproduction(snaps):
    mine = {x["milestone_index"]: x["val_accuracy"] for x in snaps if x["milestone_index"] > 0}
    out = {"per_epoch_val_acc": mine}
    for seed, lid in CORE_LANES.items():
        p = CORE / lid / "dev_snapshots" / "metrics" / "devtrain_metrics.json"
        if not p.exists():
            continue
        sub = {m["epoch"]: m["val_accuracy"] for m in json.load(open(p))}
        d = [mine[e] - sub[e] for e in mine if e in sub]
        out[f"core_seed{seed}"] = {"val_acc": sub, "n_epochs_compared": len(d),
                                        "mean_diff": float(np.mean(d)) if d else None,
                                        "max_abs_diff": float(np.max(np.abs(d))) if d else None}
    subs = [out[f"core_seed{s}"]["val_acc"] for s in CORE_LANES if f"core_seed{s}" in out]
    if len(subs) > 1:
        sp = [max(v[e] for v in subs) - min(v[e] for v in subs) for e in mine if all(e in v for v in subs)]
        out["core_seed_range_max"] = float(max(sp)) if sp else None
    return out


# 1 process events
def com(X):
    t = np.arange(len(X), dtype=float)
    return [float((t * X[:, j]).sum() / X[:, j].sum()) if X[:, j].sum() > 0 else math.nan for j in range(X.shape[1])]


def summarise(X, labels):
    """X: (n_transitions, 3) Tg, Ab, Di; labels: transition names."""
    tot = X.sum(1, keepdims=True)
    sh = 100 * X / np.maximum(tot, 1)
    cr = (X[:, 0] + X[:, 1]) / np.maximum(X[:, 2], 1)
    c = com(X)
    finite = all(np.isfinite(c))
    gaps = [c[1] - c[0], c[2] - c[1]] if finite else [math.nan, math.nan]
    di_gt = X[:, 2] > X[:, 1]
    first = int(np.flatnonzero(di_gt)[0]) if di_gt.any() else None
    from_ = None
    for t in range(len(X)):
        if di_gt[t:].all():
            from_ = t
            break
    return {"n_transitions": len(X), "shares_tg_ab_di": sh.round(2).tolist(), "cr": cr.round(3).tolist(),
            "com_tg_ab_di": [round(v, 3) for v in c], "com_gaps_ab_tg_di_ab": [round(v, 3) for v in gaps],
            "order_strict": bool(finite and gaps[0] >= TIE and gaps[1] >= TIE),
            "order_any_gap": bool(finite and c[0] < c[1] < c[2]),
            "ties": [bool(finite and abs(g) < TIE) for g in gaps],
            "tg_largest_first_transition": bool(X[0, 0] > max(X[0, 1], X[0, 2])),
            "peak_share_transition": [labels[int(np.argmax(sh[:, j]))] for j in range(3)],
            "first_transition_di_gt_ab": labels[first] if first is not None else None,
            "di_gt_ab_from_transition": labels[from_] if from_ is not None else None,
            "cr_first": float(cr[0]), "cr_last": float(cr[-1]),
            "first_transition_cr_below_1": labels[int(np.flatnonzero(cr < 1)[0])] if (cr < 1).any() else None,
            "totals_tg_ab_di": X.sum(0).astype(int).tolist()}


def process_section(res):
    pi = res["process_intensity"]
    labels = [t["transition"] for t in pi]
    for i, t in enumerate(labels):
        assert t == f"{i}->{i + 1}", labels
    X = np.array([[t[p] for p in P] for t in pi], float)
    rows = [{"transition": t["transition"], "epoch_a": i, "epoch_b": i + 1,
             **{k: t[k] for k in ("tg_h", "ab_h", "di_h", "as_h", "de_h", "total")},
             "share_tg": 100 * X[i, 0] / max(X[i].sum(), 1), "share_ab": 100 * X[i, 1] / max(X[i].sum(), 1),
             "share_di": 100 * X[i, 2] / max(X[i].sum(), 1), "cr": (X[i, 0] + X[i, 1]) / max(X[i, 2], 1)}
            for i, t in enumerate(pi)]
    out = {}
    for w, e in WINDOWS.items():
        n = len(X) if e is None else min(e, len(X))
        if e is not None and len(X) < e:
            out[w] = {"skipped": f"only {len(X)} transitions"}
            continue
        out[w] = summarise(X[:n], labels[:n])
    return pd.DataFrame(rows), out


# 2 census
def census_section(lane: Lane, epochs, snaps):
    thr = {"sai": lane.thr["sai_task_general_thresh"], "ssi": lane.thr["ssi_abstraction_thresh"],
           "csi": lane.thr["csi_differentiation_thresh"]}
    va = {x["milestone_index"]: x["val_accuracy"] for x in snaps}
    rows = []
    for e in epochs:
        lab = str(e)
        S = {"ssi": [], "csi": [], "sai": []}
        for L in lane.layers:
            H = np.load(lane.act_path(lab, L), mmap_mode="r")
            n = H.shape[0]
            alive = (np.abs(H) > EPS_ACT).sum(axis=0) >= max(1, int(np.ceil(n * 0.01)))
            sel = lane.sel(lab, L)
            S["ssi"].append(sel["feature_ssi"][alive])
            S["csi"].append(sel["feature_csi"][alive])
            S["sai"].append(sel["feature_sai"][alive])
        S = {k: np.concatenate(v).astype(np.float64) for k, v in S.items()}
        c = census_counts(S, thr)
        n = c["n_alive"]
        rows.append({"epoch": e, "val_acc": va.get(e), **c,
                     **{f"frac_{k[2:]}": c[k] / n if n else math.nan for k in
                        ("n_tg", "n_ab_overlap", "n_di_overlap", "n_ab_ex", "n_di_ex", "n_both")}})
    d = pd.DataFrame(rows)
    tim = {}
    if len(d) >= 3:
        for w, emax in WINDOWS.items():
            g = d if emax is None else d[d.epoch <= emax]
            if emax is not None and d.epoch.max() < emax:
                continue
            for defn, ab, di in (("overlapping", "frac_ab_overlap", "frac_di_overlap"), ("exclusive", "frac_ab_ex", "frac_di_ex")):
                tim[f"{w}|{defn}"] = timing({"tg": list(g.frac_tg), "ab": list(g[ab]), "di": list(g[di])})
    return d, {"thresholds": thr, "timing": tim}


# 3 SAE-free superclass share
def load_pooled(lane_dir: Path, e: int) -> np.ndarray:
    import torch
    a = torch.load(lane_dir / "dev_snapshots" / f"milestone_{e}" / "individual_multilayer_activations.pt",
                   map_location="cpu", weights_only=False)
    v = a[POOLED]
    assert sorted(v.keys()) == list(range(100))
    X = torch.cat([v[c].float() for c in range(100)], 0).numpy().astype(np.float64)
    assert X.shape[0] == 10000
    return X


def share_section(lane_dir: Path, epochs, snaps):
    ctx = Ctx(superclass_of_class(str(lane_dir)))
    va = {x["milestone_index"]: x["val_accuracy"] for x in snaps}
    rows = []
    for i, e in enumerate(epochs):
        X = load_pooled(lane_dir, e)
        p = population(X, ctx, "raw")
        r = {"lane": "perepoch_CCT-7_s42", "layer": POOLED, "checkpoint": str(e), "idx": i, "epoch": e,
             "val_acc": va.get(e), **{k: p[k] for k in ("raw_share_sc", "raw_share_sc_null_mean", "raw_share_sc_null_sd",
                                                       "raw_share_sc_null_p95", "raw_share_sc_z", "raw_fsc", "raw_fcls")}}
        rng = np.random.RandomState(11)
        sp = [splits(rng) for _ in range(N_SPLIT)]
        v = np.array([share(X, a, b, ctx.masks)[0] for a, b in sp])
        r["raw_share_mean"], r["raw_share_sd"] = v.mean(), v.std(ddof=1)
        rb = np.random.RandomState(12)
        r["raw_share_boot_sd"] = np.array([share(X, *splits(rb, boot=True), ctx.masks)[0] for _ in range(N_SPLIT)]).std(ddof=1)
        rows.append(r)
    d = pd.DataFrame(rows)
    out = {"transient_rule": transient(d, "raw") if len(d) >= 2 else None,
           "random_grouping_null_mean_range": [float(d.raw_share_sc_null_mean.min()), float(d.raw_share_sc_null_mean.max())]}
    if DICTFREE_POP.exists():
        s = pd.read_csv(DICTFREE_POP)
        s = s[(s.lane == "core_CCT-7_s42") & s.is_pooled]
        out["core_seed42"] = [{"checkpoint": int(r.checkpoint), "epoch_field": r.epoch, "val_acc": r.val_acc,
                                    "raw_share_sc": r.raw_share_sc} for r in s.itertuples()]
    return d, out


# 4/5 matching: steps, sparse vs dense, flicker
def core_grids():
    g = {}
    for seed, lid in CORE_LANES.items():
        p = CORE / lid / "dev_snapshots" / "metrics" / "devtrain_snapshots.json"
        if not p.exists():
            continue
        s = json.load(open(p))
        ep = [int(round(x["weight_updates"] / BPE)) for x in s]
        uniq = []
        for e in ep:
            if e not in uniq:
                uniq.append(e)
        g[seed] = {"weight_updates": [x["weight_updates"] for x in s], "epochs_exact": [x["weight_updates"] / BPE for x in s],
                   "epochs_nearest": ep, "grid": uniq}
    return g


def core_interval_counts(seed, grid_info):
    """Core run's stored process_intensity summed into the grid intervals (terminal transition excluded)."""
    p = CORE / CORE_LANES[seed] / "sae_analysis" / "sae_results.json"
    if not p.exists():
        return {}
    pi = json.load(open(p))["process_intensity"]
    ep, grid = grid_info["epochs_nearest"], grid_info["grid"]
    out = {}
    for t in pi:
        a, b = t["transition"].split("->")
        if b == "terminal":
            continue
        ea, eb = ep[int(a)], ep[int(b)]
        for ga, gb in zip(grid[:-1], grid[1:]):
            if ga <= ea and eb <= gb:
                o = out.setdefault(f"{ga}->{gb}", {k: 0 for k in ("tg_h", "ab_h", "di_h", "as_h", "de_h")})
                for k in o:
                    o[k] += t[k]
                break
    return out


def matching_section(lane: Lane, epochs):
    lab = Labeler(lane)
    stored_pi = {t["transition"]: t for t in lane.results["process_intensity"]}
    emax = max(epochs)
    grids = core_grids()
    steps_rows, win_rows, flick_rows = [], [], []
    t0 = time.time()
    for L in lane.layers:
        H = {e: np.load(lane.act_path(str(e), L), mmap_mode="r") for e in epochs}
        act, sel = {}, {e: lane.sel(str(e), L) for e in epochs}

        def A(e):
            if e not in act:
                act[e] = np.abs(H[e]) > EPS_ACT
            return act[e]

        step = {}
        for e in epochs[:-1]:
            md = pipeline_match(H[e], H[e + 1])
            o = outcome_from_pipeline(md)
            step[e] = (md, o)
            labs = lab.labels(o, sel[e], sel[e + 1])
            w = lab.weighted(o, labs, A(e), A(e + 1))
            st = lane.stored_counts(f"{e}->{e + 1}", L)
            steps_rows.append({"layer": L, "epoch_a": e, "epoch_b": e + 1, "n_alive_a": int(md["alive_a"].sum()),
                               "n_alive_b": int(md["alive_b"].sum()),
                               **{k: int(md[k]) for k in ("n_stable", "n_born", "n_died", "n_transformed")},
                               "counts_match_stored": bool(st) and all(int(st[k]) == int(md[k]) for k in
                                                                       ("n_stable", "n_born", "n_died", "n_transformed")),
                               **feature_counts(labs), **{f"w_{k}": v for k, v in w.items()}})
        sr = {r["epoch_a"]: r for r in steps_rows if r["layer"] == L}
        # 4: sparse grids
        for seed, gi in grids.items():
            grid = [e for e in gi["grid"] if e <= emax]
            for a, b in zip(grid[:-1], grid[1:]):
                if b - a == 1:
                    md, direct = step[a]
                    chain = {"n_alive": int(md["alive_a"].sum())}
                else:
                    md = pipeline_match(H[a], H[b])
                    direct = outcome_from_pipeline(md)
                    ia = np.where(md["alive_a"])[0]
                    chain = chain_compare([step[e][1] for e in range(a, b)], direct, ia,
                                          prev=step[a - 1][1] if a > 0 else None)
                labs = lab.labels(direct, sel[a], sel[b])
                w = lab.weighted(direct, labs, A(a), A(b))
                r = {"grid_seed": seed, "layer": L, "epoch_a": a, "epoch_b": b, "length_epochs": b - a,
                     **{f"direct_{k}": int(md[k]) for k in ("n_stable", "n_born", "n_died", "n_transformed")},
                     **{f"direct_{k}": v for k, v in feature_counts(labs).items()},
                     **{f"direct_w_{k}": v for k, v in w.items()},
                     **{f"chain_{k}": v for k, v in chain.items()}}
                dense = [sr[e] for e in range(a, b)]
                for k in F_KEYS + W_KEYS + ("w_stable",):
                    r[f"dense_sum_{k}"] = int(sum(x[k] for x in dense))
                # dense births that survive (stable links) to b
                surv = 0
                for e in range(a, b):
                    for j in step[e][1]["born"]:
                        cur, ok = j, True
                        for e2 in range(e + 1, b):
                            if cur in step[e2][1]["stable"]:
                                cur = step[e2][1]["stable"][cur][0]
                            else:
                                ok = False
                                break
                        surv += ok
                r["dense_births_surviving_to_b"] = surv
                win_rows.append(r)
        # 5: flicker (skip-one)
        for k in epochs[:-2]:
            md = pipeline_match(H[k], H[k + 2])
            direct = outcome_from_pipeline(md)
            ia = np.where(md["alive_a"])[0]
            res = chain_compare([step[k][1], step[k + 1][1]], direct, ia, prev=step[k - 1][1] if k > 0 else None)
            res.update({"layer": L, "k": k, "n_born_step1": len(step[k][1]["born"]),
                        "n_blip_births": len(step[k][1]["born"] & step[k + 1][1]["died"])})
            flick_rows.append(res)
        del H, act
        print(f"  matching {L} done ({time.time() - t0:.0f}s)", flush=True)
    S = pd.DataFrame(steps_rows)
    # validation: sample-weighted sums over layers == stored process_intensity
    val = []
    for (a, b), g in S.groupby(["epoch_a", "epoch_b"]):
        st = stored_pi[f"{a}->{b}"]
        val.append(all(int(g[f"w_{k}"].sum()) == int(st[k]) for k in ("tg_h", "ab_h", "di_h", "as_h", "de_h")))
    validation = {"feature_matching_counts_match": f"{int(S.counts_match_stored.sum())}/{len(S)}",
                  "process_intensity_match": f"{sum(val)}/{len(val)}"}
    return S, pd.DataFrame(win_rows), pd.DataFrame(flick_rows), grids, validation


def pool_windows(W, grids):
    num = [c for c in W.columns if c.startswith(("direct_", "dense_", "chain_n_"))]
    P_ = W.groupby(["grid_seed", "epoch_a", "epoch_b", "length_epochs"], as_index=False)[num].sum()
    pairs = [("births", "n_born"), ("deaths", "n_died"), ("transformed", "n_transformed"), ("tg_feat", "f_tg"),
             ("ab_feat", "f_ab"), ("di_feat", "f_di"), ("as_events", "w_as_h"), ("de_events", "w_de_h"),
             ("tg_events", "w_tg_h"), ("ab_events", "w_ab_h"), ("di_events", "w_di_h")]
    for nm, k in pairs:
        d, s = P_[f"direct_{k}"].astype(float), P_[f"dense_sum_{k}"].astype(float)
        P_[f"ratio_dense_over_direct_{nm}"] = s / d.where(d > 0)
        P_[f"missed_frac_{nm}"] = 1 - d / s.where(s > 0)
    P_["births_surviving_frac"] = P_.dense_births_surviving_to_b / P_.dense_sum_n_born.where(P_.dense_sum_n_born > 0)
    rows = []
    for seed in P_.grid_seed.unique():
        sub = core_interval_counts(int(seed), grids[int(seed)])
        for r in P_[P_.grid_seed == seed].itertuples():
            s = sub.get(f"{r.epoch_a}->{r.epoch_b}", {})
            rows.append({k: s.get(k) for k in ("tg_h", "ab_h", "di_h", "as_h", "de_h")})
    for k in ("tg_h", "ab_h", "di_h", "as_h", "de_h"):
        P_[f"pipeline_stored_{k}"] = [r[k] for r in rows]
    # process COM / order on the coarse grid (direct counts) vs per-epoch
    order = {}
    for seed, g in P_.groupby("grid_seed"):
        g = g.sort_values("epoch_a")
        X = g[["direct_w_tg_h", "direct_w_ab_h", "direct_w_di_h"]].to_numpy(float)
        if len(X) >= 1:
            order[int(seed)] = summarise(X, [f"{a}->{b}" for a, b in zip(g.epoch_a, g.epoch_b)])
    return P_, order


def flicker_summary(F):
    num = ["n_alive", "n_died_step1", "n_flicker_step1", "n_blip_births", "n_born_step1"] + \
        [c for c in ("n_consensus", "n_died_step1_consensus", "n_flicker_step1_consensus") if c in F.columns]
    E = F.groupby("k", as_index=False)[num].sum()
    E["flicker_pct_alive"] = 100 * E.n_flicker_step1 / E.n_alive
    E["flicker_pct_of_deaths"] = 100 * E.n_flicker_step1 / E.n_died_step1.where(E.n_died_step1 > 0)
    E["blip_pct_of_births"] = 100 * E.n_blip_births / E.n_born_step1.where(E.n_born_step1 > 0)
    out = {}
    for nm, sel in (("all", E.k >= 0), ("k_0_14", E.k <= 14), ("k_15_plus", E.k >= 15)):
        g = E[sel]
        if len(g) == 0:
            continue
        out[nm] = {"n_k": len(g), "flicker_pct_alive_pooled": 100 * g.n_flicker_step1.sum() / g.n_alive.sum(),
                   "flicker_pct_of_deaths_pooled": 100 * g.n_flicker_step1.sum() / max(g.n_died_step1.sum(), 1),
                   "flicker_pct_alive_range": [float(g.flicker_pct_alive.min()), float(g.flicker_pct_alive.max())],
                   "blip_pct_of_births_pooled": 100 * g.n_blip_births.sum() / max(g.n_born_step1.sum(), 1)}
        if "n_flicker_step1_consensus" in g:
            out[nm]["flicker_pct_of_consensus_deaths"] = (100 * g.n_flicker_step1_consensus.sum()
                                                          / max(g.n_died_step1_consensus.sum(), 1))
    return E, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", type=Path, default=LANE_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--skip-matching", action="store_true", help="sections 0-3 only")
    a = ap.parse_args()
    assert not str(a.out.resolve()).startswith(RDT_DATA + "/core")
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    snaps = load_snaps(a.lane)
    lane = Lane(a.lane)
    epochs = [int(x) for x in lane.labels]
    assert epochs == list(range(len(epochs))), lane.labels
    assert epochs == [x["milestone_index"] for x in snaps], ([x["milestone_index"] for x in snaps], epochs)
    R = {"lane": str(a.lane), "epochs": epochs, "layers": lane.layers,
         "thresholds": {k: lane.thr[k] for k in ("sai_task_general_thresh", "ssi_abstraction_thresh",
                                                 "csi_differentiation_thresh")}}
    R["reproduction"] = reproduction(snaps)
    d1, R["process"] = process_section(lane.results)
    d1.to_csv(a.out / "process_per_transition.csv", index=False)
    print("process:", json.dumps({w: {k: v.get(k) for k in ("com_tg_ab_di", "order_strict", "ties",
                                                            "di_gt_ab_from_transition")} for w, v in R["process"].items()}),
          flush=True)
    d2, R["census"] = census_section(lane, epochs, snaps)
    d2.to_csv(a.out / "census_per_epoch.csv", index=False)
    d3, R["sae_free_share"] = share_section(a.lane, epochs, snaps)
    d3.to_csv(a.out / "sae_free_share_per_epoch.csv", index=False)
    print("census + SAE-free share done", f"({time.time() - t0:.0f}s)", flush=True)
    if not a.skip_matching:
        S, W, F, grids, val = matching_section(lane, epochs)
        S.to_csv(a.out / "steps_per_layer.csv", index=False)
        W.to_csv(a.out / "sparse_vs_dense_per_layer.csv", index=False)
        R["grids"], R["validation"] = grids, val
        if len(W):
            Pw, R["coarse_grid_process_order"] = pool_windows(W, grids)
            Pw.to_csv(a.out / "sparse_vs_dense.csv", index=False)
            R["sparse_vs_dense"] = Pw.to_dict(orient="records")
        if len(F):
            F.to_csv(a.out / "flicker_per_layer.csv", index=False)
            E, R["flicker"] = flicker_summary(F)
            E.to_csv(a.out / "flicker_per_epoch.csv", index=False)
        print("validation:", val, flush=True)
    R["runtime_s"] = time.time() - t0
    jdump(R, a.out / "results.json")
    print("written", a.out, f"({R['runtime_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
