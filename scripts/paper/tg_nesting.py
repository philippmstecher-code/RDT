#!/usr/bin/env python3
"""Do superclass-selective features form within the task-general features? The nesting statistic of Extended Data
Fig. 6 one level down (b1, same layer) and a cross-layer version (b2). No training.

Networks (stored SAEs, codes and pooled channel activations; read only):
  per-epoch  sham_e00 seeds 7, 42, 137 (weights critical_sham, SAEs sham_perepoch), epochs 1-15
  standard   the five standard ResNet-18 lanes of Figs 1-4 (dependency_lib.standard_lanes), milestones 1-6 = epochs 5-30
Categories per network and layer: stored per-run thresholds, 'methods' rule (task-general TG: alive and SAI > t;
superclass-selective SS: SSI > t, CSI <= t, not TG; class-selective CS: CSI > t, not TG; alive = active on >= 1% of
images). Sensitivity: TG at the run's 99th percentile of SAI (all alive features, epochs and layers of the run) instead
of the stored threshold (95th percentile of the observed SAI values).

b1 (same layer; layer1-layer3, where TG features exist). T = span of the unit decoder directions of the layer's TG
features (rank k). Per SS feature f: e = squared norm of the projection of its unit decoder direction onto T. Nulls:
  chance      k / C (a uniformly random subspace of rank k)
  variance    random rank-k subspaces matched to T in captured activation variance: spans of k Gaussian vectors with
              covariance S^beta (S = covariance of the layer's pooled activations), beta set by bisection so that the
              mean captured variance share equals that of T (32 draws)
  dictionary  spans of k decoder directions of the layer's other alive features (neither TG, SS nor CS; 32 draws)
  grouping    the null of that nesting statistic: SS features recomputed under 100 random partitions of the 100 classes into 20 groups of
              five (same thresholds; TG unchanged, as SAI does not depend on the grouping); statistic = median e of the
              random-group SS features
Ratio per network-layer: median over SS features of e / e_null (variance, dictionary, chance); grouping: median e of
the true SS features / mean over partitions of the median e of the random-group SS features (and p).
b2 (cross layer; layer1->layer2, layer2->layer3, layer3->layer4). Predictors: the layer-L pooled activations projected
onto T_L (k dims); targets: the codes of the SS features of layer L+1. Ridge (alpha 1 on standardised predictors),
5-fold cross-validation (folds by image index within class), R^2 pooled over targets (variance-weighted). Nulls: the
variance-matched and dictionary-matched rank-k subspaces of layer L (16 draws each). Ratio = R^2(T) / mean R^2(null).
Summaries: per seed median ratio over the networks of a window (per-epoch 1-5, 6-15; standard 5-30), across seeds
geometric mean with a t-based 95% CI on the log scale, seeds > 1.
Writes analysis_outputs/paper/TG_nesting/{b1_rows.csv, b2_rows.csv, summary.json}.
Usage: python scripts/paper/tg_nesting.py
"""
import json
import os
import sys
from pathlib import Path
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dependency_lib import COARSE_TO_FINE, LaneData, standard_lanes  # noqa: E402

E = Path(RDT_DATA + "/extended/experiments")
OUT = Path(__file__).resolve().parents[2] / "analysis_outputs/paper/TG_nesting"
LAYERS = ["layer1", "layer2", "layer3", "layer4"]
B1_LAYERS = ["layer1", "layer2", "layer3"]
NDRAW = 32
NPERM = 100
EPS = 1e-10


def sel(H, grp=None):
    """selectivity of dependency_lib.selectivity, under a grouping grp (class -> group of 20); grp None = true superclasses."""
    M = np.abs(H).reshape(100, 100, -1).mean(1).T
    tot = M.sum(1)
    groups = [COARSE_TO_FINE[c] for c in range(20)] if grp is None else [np.where(grp == g)[0] for g in range(20)]
    S = np.stack([M[:, g].sum(1) for g in groups], 1)
    frac = S / (tot[:, None] + EPS)
    best = frac.argmax(1)
    ssi = np.maximum((frac.max(1) - 1 / 20) / (1 - 1 / 20 + EPS), 0)
    p = M / (tot[:, None] + EPS)
    with np.errstate(divide="ignore", invalid="ignore"):
        sai = -np.where(p > 0, p * np.log(p + EPS), 0).sum(1) / np.log(100)
    W = np.take_along_axis(M, np.stack([groups[b] for b in best]), 1)
    csi = np.maximum((W.max(1) / (W.sum(1) + EPS) - 0.2) / (0.8 + EPS), 0)
    act = tot > EPS
    return np.where(act, ssi, 0), np.where(act, csi, 0), np.where(act, sai, 0)


def cats(H, t, sai_t=None):
    ssi, csi, sai = sel(H)
    alive = (np.abs(H) > 1e-6).mean(0) >= 0.01
    tg = alive & (sai > (t["sai"] if sai_t is None else sai_t))
    ss = alive & (ssi > t["ssi"]) & (csi <= t["csi"]) & ~tg
    cs = alive & (csi > t["csi"]) & ~tg
    return alive, tg, ss, cs, sai


def orth(B):
    return np.linalg.qr(B)[0]


def var_share(Q, S):
    return float(np.trace(Q.T @ S @ Q) / np.trace(S))


def var_matched(S, k, target, rng, n):
    """n random rank-k orthonormal bases with mean captured variance share ~ target (Gaussian, covariance S^beta)."""
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 0, None) / w.max()
    C = S.shape[0]

    def draws(beta, m, r):
        A = V @ (np.sqrt(w ** beta)[:, None] * r.standard_normal((C, k * m)))
        return [orth(A[:, i * k:(i + 1) * k]) for i in range(m)]

    lo, hi = 0.0, 4.0
    for _ in range(14):
        mid = (lo + hi) / 2
        v = np.mean([var_share(Q, S) for Q in draws(mid, 8, np.random.default_rng(1))])
        lo, hi = (mid, hi) if v < target else (lo, mid)
    Qs = draws((lo + hi) / 2, n, rng)
    return Qs, float(np.mean([var_share(Q, S) for Q in Qs]))


def networks():
    out = []
    for s in (7, 42, 137):
        sd = E / f"sham_perepoch/lanes/sham_e00_seed{s}/sae_analysis"
        w = E / f"critical_sham/lanes/sham_e00_seed{s}"
        snaps = json.load(open(w / "dev_snapshots/metrics/devtrain_snapshots.json"))
        ms = {int(x["epoch"]): x["milestone_index"] for x in snaps if x.get("snapshot_type") == "intermediate"}
        thr = json.load(open(sd / "sae_results.json"))["metadata"]["analysis_thresholds"]
        t = {"sai": thr["sai_task_general_thresh"], "ssi": thr["ssi_abstraction_thresh"], "csi": thr["csi_differentiation_thresh"]}
        for e in range(1, 16):
            out.append({"set": "per-epoch", "seed": s, "epoch": e, "t": t, "sae": sd / "saes" / str(e),
                        "codes": sd / "activations" / str(e), "acts": w / "dev_snapshots" / f"milestone_{ms[e]}"})
    for ln in standard_lanes():
        ld = LaneData(ln["dir"])
        t = {"sai": ld.sai_t, "ssi": ld.ssi_t, "csi": ld.csi_t}
        for k in range(1, 7):
            out.append({"set": "standard", "seed": ln["seed"], "epoch": 5 * k, "t": t,
                        "sae": ln["dir"] / "sae_analysis/saes" / str(k), "codes": ln["dir"] / "sae_analysis/activations" / str(k),
                        "acts": ln["dir"] / "dev_snapshots" / f"milestone_{k}"})
    return out


def load(n):
    A = torch.load(n["acts"] / "individual_multilayer_activations.pt", map_location="cpu", weights_only=False)
    acts, H, D = {}, {}, {}
    for L in LAYERS:
        acts[L] = torch.cat([A[L][c] for c in sorted(A[L])]).double().numpy()
        H[L] = np.load(n["codes"] / f"{L}.npy").astype(np.float64)
        d = torch.load(n["sae"] / f"{L}.pt", map_location="cpu", weights_only=True)["decoder.weight"].double().numpy()
        D[L] = d / np.maximum(np.linalg.norm(d, axis=0, keepdims=True), 1e-12)
    return acts, H, D


def ridge_r2(Z, Y, folds, alpha=1.0):
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    pred = np.zeros_like(Y)
    for f in range(5):
        tr, te = folds != f, folds == f
        zm, ym = Z[tr].mean(0), Y[tr].mean(0)
        Zt, Yt = Z[tr] - zm, Y[tr] - ym
        W = np.linalg.solve(Zt.T @ Zt + alpha * np.eye(Z.shape[1]), Zt.T @ Yt)
        pred[te] = (Z[te] - zm) @ W + ym
    return float(1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nets = networks()
    # run-level 99th percentile of SAI (sensitivity)
    p99 = {}
    folds = np.tile(np.arange(100) % 5, 100)
    b1, b2 = [], []
    for n in nets:
        acts, H, D = load(n)
        key = (n["set"], n["seed"])
        if key not in p99:
            vals = []
            for m in [x for x in nets if (x["set"], x["seed"]) == key]:
                for L in LAYERS:
                    Hm = np.load(m["codes"] / f"{L}.npy").astype(np.float64)
                    a = (np.abs(Hm) > 1e-6).mean(0) >= 0.01
                    vals.append(sel(Hm)[2][a])
            p99[key] = float(np.percentile(np.concatenate(vals), 99))
        rng = np.random.default_rng(n["seed"] * 100 + n["epoch"])
        C_ = {L: cats(H[L], n["t"]) for L in LAYERS}
        for variant, sai_t in (("stored", None), ("p99", p99[key])):
            C = C_ if variant == "stored" else {L: cats(H[L], n["t"], sai_t) for L in LAYERS}
            for L in B1_LAYERS:
                alive, tg, ss, cs, _ = C[L]
                k, Cd = int(tg.sum()), D[L].shape[0]
                if k == 0 or ss.sum() == 0 or k >= Cd:
                    continue
                Q = orth(D[L][:, tg])
                X = acts[L] - acts[L].mean(0)
                S = X.T @ X / len(X)
                vt = var_share(Q, S)
                e = ((Q.T @ D[L][:, ss]) ** 2).sum(0)
                Qv, vv = var_matched(S, Q.shape[1], vt, rng, NDRAW)
                ev = np.mean([((q.T @ D[L][:, ss]) ** 2).sum(0) for q in Qv], 0)
                ot = np.where(alive & ~tg & ~ss & ~cs)[0]
                ed = (np.mean([((orth(D[L][:, rng.choice(ot, k, replace=False)]).T @ D[L][:, ss]) ** 2).sum(0)
                               for _ in range(NDRAW)], 0) if len(ot) >= k else np.full(len(e), np.nan))
                ecs = ((Q.T @ D[L][:, cs]) ** 2).sum(0) if cs.any() else np.array([np.nan])
                # grouping null
                gm = []
                for p in range(NPERM):
                    grp = np.empty(100, int)
                    grp[np.random.RandomState(p + 1).permutation(100)] = np.repeat(np.arange(20), 5)
                    ssi_p, csi_p, _ = sel(H[L], grp)
                    ssp = alive & (ssi_p > n["t"]["ssi"]) & (csi_p <= n["t"]["csi"]) & ~tg
                    if ssp.any():
                        gm.append(float(np.median(((Q.T @ D[L][:, ssp]) ** 2).sum(0))))
                b1.append({"set": n["set"], "seed": n["seed"], "epoch": n["epoch"], "layer": L, "variant": variant,
                           "n_tg": k, "rank_T": Q.shape[1], "C": Cd, "n_ss": int(ss.sum()), "n_cs": int(cs.sum()),
                           "var_share_T": vt, "var_share_null": vv, "e_ss_median": float(np.median(e)),
                           "e_cs_median": float(np.nanmedian(ecs)), "chance": Q.shape[1] / Cd,
                           "ratio_variance": float(np.median(e / ev)), "ratio_dictionary": float(np.nanmedian(e / ed)),
                           "ratio_chance": float(np.median(e) / (Q.shape[1] / Cd)),
                           "ratio_grouping": float(np.median(e) / np.mean(gm)) if gm else np.nan,
                           "p_grouping": (1 + sum(g >= np.median(e) for g in gm)) / (1 + len(gm)) if gm else np.nan,
                           "n_partitions_with_ss": len(gm)})
            for L, L2 in zip(LAYERS[:-1], LAYERS[1:]):
                alive, tg, ss, cs, _ = C[L]
                ss2 = C[L2][2]
                k = int(tg.sum())
                if k == 0 or ss2.sum() == 0 or k >= D[L].shape[0]:
                    continue
                Q = orth(D[L][:, tg])
                X = acts[L] - acts[L].mean(0)
                S = X.T @ X / len(X)
                Y = H[L2][:, ss2]
                r_t = ridge_r2(X @ Q, Y, folds)
                Qv, _ = var_matched(S, Q.shape[1], var_share(Q, S), rng, 16)
                r_v = float(np.mean([ridge_r2(X @ q, Y, folds) for q in Qv]))
                ot = np.where(alive & ~tg & ~ss & ~cs)[0]
                r_d = (float(np.mean([ridge_r2(X @ orth(D[L][:, rng.choice(ot, k, replace=False)]), Y, folds)
                                      for _ in range(16)])) if len(ot) >= k else np.nan)
                b2.append({"set": n["set"], "seed": n["seed"], "epoch": n["epoch"], "pair": f"{L}->{L2}",
                           "variant": variant, "n_tg": k, "n_ss_target": int(ss2.sum()), "r2_tg": r_t,
                           "r2_variance": r_v, "r2_dictionary": r_d, "ratio_variance": r_t / r_v if r_v > 0 else np.nan,
                           "ratio_dictionary": r_t / r_d if r_d and r_d > 0 else np.nan})
        print(n["set"], n["seed"], n["epoch"], flush=True)
    B1, B2 = pd.DataFrame(b1), pd.DataFrame(b2)
    B1.to_csv(OUT / "b1_rows.csv", index=False)
    B2.to_csv(OUT / "b2_rows.csv", index=False)
    wins = {("per-epoch", "1-5"): (1, 5), ("per-epoch", "6-15"): (6, 15), ("standard", "5-30"): (5, 30)}
    res = {"p99_sai_by_run": {f"{a}|{b}": v for (a, b), v in p99.items()}}

    def summarise(D_, unit, cols):
        out = {}
        for variant in ("stored", "p99"):
            for (st, wn), (a, b) in wins.items():
                for u, g in D_[(D_.variant == variant) & (D_.set == st) & D_.epoch.between(a, b)].groupby(unit):
                    for c in cols:
                        v = np.log(g.groupby("seed")[c].median().dropna())
                        if len(v) < 2:
                            continue
                        m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
                        h = stats.t.ppf(0.975, len(v) - 1) * se
                        out[f"{variant}|{st} {wn}|{u}|{c}"] = {
                            "geo_mean": float(np.exp(m)), "ci": [float(np.exp(m - h)), float(np.exp(m + h))],
                            "seeds_gt1": int((v > 0).sum()), "n_seeds": len(v),
                            "per_seed": {int(i): round(float(np.exp(x)), 3) for i, x in v.items()},
                            "n_networks": int(len(g)), "median_n_ss": float(g[[x for x in g.columns if x.startswith("n_ss")][0]].median()),
                            "median_n_tg": float(g.n_tg.median())}
        return out

    res["b1"] = summarise(B1, "layer", ["ratio_variance", "ratio_dictionary", "ratio_chance", "ratio_grouping"])
    res["b2"] = summarise(B2, "pair", ["ratio_variance", "ratio_dictionary"])
    json.dump(res, open(OUT / "summary.json", "w"), indent=1)
    for part in ("b1", "b2"):
        for k_, v in res[part].items():
            if k_.startswith("stored"):
                print(part, k_, round(v["geo_mean"], 2), [round(x, 2) for x in v["ci"]], f"{v['seeds_gt1']}/{v['n_seeds']}",
                      "nets", v["n_networks"], "n_ss", v["median_n_ss"], "n_tg", v["median_n_tg"])


if __name__ == "__main__":
    main()
