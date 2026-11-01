#!/usr/bin/env python3
"""Is the three-phase developmental program visible without sparse autoencoders?

Open networks only (dictfree_lib). For every open lane x checkpoint x stored layer, on the
raw activations (10,000 test images, 100 per class, class-major rows):

Unit level (basis = raw units; pooled layer additionally pca / rot / ica):
  abs   : pipeline Phase-4 SSI / CSI_local / SAI on class means of |x| (identical formula to the SAE census)
  shift : the same indices on class means of x minus the unit's minimum class mean (baseline-subtracted tuning)
  var   : split-half unbiased variance partition of each unit: T (total), b_cls (between-class),
          b_sc (between-superclass part of b_cls), b_wsc = b_cls - b_sc (within-superclass, between-class)
  nulls : SSI  - class->superclass permutation (100 perms, RandomState(42) as the pipeline);
          CSI  - sample->class row shuffle (50 perms), CSI in best shuffled superclass;
          SAI  - row-shuffle null p05 (diagnostic); f_sc - class->superclass permutation;
          f_wsc- rows shuffled within superclass (keeps superclass signal, destroys class-within-superclass);
          f_cls- full row shuffle.  Per unit: own null p95 (p05 for SAI) + 20 null draws (pooled thresholds).
Population level (rotation invariant; raw space and within-class-whitened space):
  traces of T, B_cls, B_sc, B_wsc (split-half unbiased), share_sc = B_sc / B_cls with permutation null,
  split-half nearest-class-centroid accuracies (100-way class, 20-way superclass, 5-way within superclass,
  error-in-true-superclass share), participation ratios of superclass-centroid and within-superclass residual
  covariance.
Network outputs: superclass consistency of the network's own test predictions.

Writes analysis_outputs/paper/DICTFREE/population/cache/{key}.npz and {key}_pop.csv. Read-only on data.
Usage: dictfree_compute.py [--only KEYPREFIX]
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dictfree_lib import OUT, open_lanes, load_acts, load_preds, superclass_of_class  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from census_compute import selectivity_from_M  # noqa: E402

N_CLS, N_PER, N = 100, 100, 10000
Y = np.repeat(np.arange(N_CLS), N_PER)
EPS = 1e-10
ACT_EPS = 1e-6
N_SCPERM, N_ROWPERM, N_KEEP = 100, 50, 20
SHRINK = 0.1


class Ctx:
    def __init__(self, sc: np.ndarray):
        self.sc = sc
        self.ysc = sc[Y]
        self.masks = np.stack([sc == s for s in range(20)])  # (20, 100) bool
        rng = np.random.RandomState(0)
        idxA, idxB = [], []
        for c in range(N_CLS):
            p = rng.permutation(N_PER) + c * N_PER
            idxA.append(np.sort(p[:50]))
            idxB.append(np.sort(p[50:]))
        self.idxA, self.idxB = np.array(idxA), np.array(idxB)  # (100, 50)
        # class -> superclass permutations (pipeline RNG structure)
        rng = np.random.RandomState(42)
        self.perm_masks = []
        for _ in range(N_SCPERM):
            perm = rng.permutation(N_CLS)
            self.perm_masks.append(np.stack([np.isin(np.arange(N_CLS), perm[s * 5:(s + 1) * 5]) for s in range(20)]))
        rng = np.random.RandomState(7)
        self.row_perms = [rng.permutation(N) for _ in range(N_ROWPERM)]
        self.wsc_perms = []
        for _ in range(N_ROWPERM):
            p = np.arange(N)
            for s in range(20):
                ix = np.where(self.ysc == s)[0]
                p[ix] = ix[rng.permutation(ix.size)]
            self.wsc_perms.append(p)
        self.keep = np.random.RandomState(3).choice(N_ROWPERM, N_KEEP, replace=False)
        self.keep_sc = np.random.RandomState(4).choice(N_SCPERM, N_KEEP, replace=False)


def class_means(X: np.ndarray) -> np.ndarray:
    return X.reshape(N_CLS, N_PER, -1).mean(axis=1)


def half_dev(X: np.ndarray, ctx: Ctx):
    MA = X[ctx.idxA].mean(axis=1)
    MB = X[ctx.idxB].mean(axis=1)
    return MA - MA.mean(0), MB - MB.mean(0)


def var_parts(DA, DB, masks):
    """Per-unit split-half unbiased between-class and between-superclass variance."""
    bcls = (DA * DB).mean(0)
    G = masks.astype(np.float64) / masks.sum(1, keepdims=True)
    SA, SB = G @ DA, G @ DB
    bsc = (SA * SB * masks.sum(1, keepdims=True)).sum(0) / N_CLS
    return bcls, bsc


def census_indices(M: np.ndarray, ctx: Ctx):
    """M (units x classes) non-negative -> ssi, csi, sai (pipeline formulas)."""
    return selectivity_from_M(M, ctx.masks, ctx.masks, 20)


def ssi_null(M: np.ndarray, ctx: Ctx) -> np.ndarray:
    tot = M.sum(1)
    out = np.empty((N_SCPERM, M.shape[0]))
    for p, pm in enumerate(ctx.perm_masks):
        fr = (M @ pm.T.astype(np.float64)) / (tot[:, None] + EPS)
        out[p] = np.clip((fr.max(1) - 0.05) / (0.95 + EPS), 0, 1)
    return out


def unit_census(X: np.ndarray, ctx: Ctx) -> dict:
    """All unit-level quantities for one representation X (n x units)."""
    d = X.shape[1]
    res = {}
    T = X.var(0)
    cnt = (np.abs(X) > ACT_EPS).sum(0)
    alive = (cnt >= 100) & (T > 1e-12)
    res["alive"], res["T"] = alive, T
    # var variant
    DA, DB = half_dev(X, ctx)
    bcls, bsc = var_parts(DA, DB, ctx.masks)
    res["bcls"], res["bsc"] = bcls, bsc
    fsc_null = np.empty((N_SCPERM, d))
    for p, pm in enumerate(ctx.perm_masks):
        fsc_null[p] = var_parts(DA, DB, pm)[1] / (T + EPS)
    fwsc_null = np.empty((N_ROWPERM, d))
    fcls_null = np.empty((N_ROWPERM, d))
    # abs / shift variants: real values
    Mabs = class_means(np.abs(X)).T
    Mmean = class_means(X).T
    Mshift = Mmean - Mmean.min(1, keepdims=True)
    for nm, M in (("abs", Mabs), ("shift", Mshift)):
        ssi, csi, sai = census_indices(M, ctx)
        res[f"ssi_{nm}"], res[f"csi_{nm}"], res[f"sai_{nm}"] = ssi, csi, sai
        sn = ssi_null(M, ctx)
        res[f"ssi_{nm}_p95"] = np.percentile(sn, 95, axis=0)
        res[f"ssi_{nm}_null"] = sn[ctx.keep_sc].astype(np.float16)
    csi_null = {nm: np.zeros((N_ROWPERM, d)) for nm in ("abs", "shift")}
    sai_null = {nm: np.zeros((N_ROWPERM, d)) for nm in ("abs", "shift")}
    Xabs = np.abs(X)
    for p in range(N_ROWPERM):
        rp = ctx.row_perms[p]
        Xp = X[rp]
        # f_cls null
        DAp, DBp = half_dev(Xp, ctx)
        fcls_null[p] = (DAp * DBp).mean(0) / (T + EPS)
        # f_wsc null (rows shuffled within superclass)
        Xw = X[ctx.wsc_perms[p]]
        DAw, DBw = half_dev(Xw, ctx)
        bc, bs = var_parts(DAw, DBw, ctx.masks)
        fwsc_null[p] = (bc - bs) / (T + EPS)
        # CSI / SAI sample-shuffle nulls
        Mp_mean = class_means(Xp).T
        for nm, Mp in (("abs", class_means(Xabs[rp]).T), ("shift", Mp_mean - Mp_mean.min(1, keepdims=True))):
            tp = Mp.sum(1)
            fr = (Mp @ ctx.masks.T.astype(np.float64)) / (tp[:, None] + EPS)
            b = fr.argmax(1)
            bm = ctx.masks[b]
            Mw = np.where(bm, Mp, 0.0)
            r = np.where(Mw.sum(1) > EPS, Mw.max(1) / (Mw.sum(1) + EPS), 0.0)
            csi_null[nm][p] = np.where(tp > EPS, np.maximum((r - 0.2) / (0.8 + EPS), 0), 0)
            pr = Mp / (tp[:, None] + EPS)
            with np.errstate(divide="ignore", invalid="ignore"):
                ent = -np.sum(np.where(pr > 0, pr * np.log(pr + EPS), 0.0), axis=1)
            sai_null[nm][p] = ent / math.log(N_CLS)
    for nm in ("abs", "shift"):
        res[f"csi_{nm}_p95"] = np.percentile(csi_null[nm], 95, axis=0)
        res[f"csi_{nm}_null"] = csi_null[nm][ctx.keep].astype(np.float16)
        res[f"sai_{nm}_p05"] = np.percentile(sai_null[nm], 5, axis=0)
    for nm, arr in (("fsc", fsc_null), ("fwsc", fwsc_null), ("fcls", fcls_null)):
        res[f"{nm}_p95"] = np.percentile(arr, 95, axis=0)
        k = ctx.keep_sc if nm == "fsc" else ctx.keep
        res[f"{nm}_null"] = arr[k].astype(np.float16)
    return res


def whiten(X: np.ndarray) -> np.ndarray:
    Xc = X - class_means(X)[Y]
    W = Xc.T @ Xc / (N - N_CLS)
    d = W.shape[0]
    W = (1 - SHRINK) * W + SHRINK * np.trace(W) / d * np.eye(d)
    ev, U = np.linalg.eigh(W)
    ev = np.maximum(ev, 1e-12 * ev.max())
    return (X - X.mean(0)) @ (U / np.sqrt(ev)) @ U.T


def pr(M: np.ndarray) -> float:
    M = M - M.mean(0)
    s = np.linalg.svd(M, compute_uv=False) ** 2
    return float(s.sum() ** 2 / (s ** 2).sum()) if s.sum() > 0 else float("nan")


def population(X: np.ndarray, ctx: Ctx, tag: str) -> dict:
    T = float(X.var(0).sum())
    DA, DB = half_dev(X, ctx)
    bcls_u, bsc_u = var_parts(DA, DB, ctx.masks)
    Bc, Bs = float(bcls_u.sum()), float(bsc_u.sum())
    null = np.array([var_parts(DA, DB, pm)[1].sum() for pm in ctx.perm_masks]) / (Bc + EPS)
    o = {f"{tag}_T": T, f"{tag}_Bcls": Bc, f"{tag}_Bsc": Bs, f"{tag}_Bwsc": Bc - Bs,
         f"{tag}_fcls": Bc / T, f"{tag}_fsc": Bs / T, f"{tag}_fwsc": (Bc - Bs) / T,
         f"{tag}_share_sc": Bs / (Bc + EPS), f"{tag}_share_sc_null_mean": float(null.mean()),
         f"{tag}_share_sc_null_sd": float(null.std()), f"{tag}_share_sc_null_p95": float(np.percentile(null, 95))}
    o[f"{tag}_share_sc_z"] = (o[f"{tag}_share_sc"] - null.mean()) / (null.std() + EPS)
    # nearest class centroid, split-half both directions
    acc = {k: [] for k in ("cls", "sc20", "sc_from_cls", "wsc5", "err_insc")}
    for tr, te in ((ctx.idxA, ctx.idxB), (ctx.idxB, ctx.idxA)):
        C = X[tr].mean(1)  # (100, d)
        S = (ctx.masks.astype(float) / 5) @ C
        Xt = X[te.ravel()]
        yt = np.repeat(np.arange(N_CLS), te.shape[1])
        st = ctx.sc[yt]
        D = (Xt ** 2).sum(1)[:, None] - 2 * Xt @ C.T + (C ** 2).sum(1)[None]
        pc = D.argmin(1)
        DS = (Xt ** 2).sum(1)[:, None] - 2 * Xt @ S.T + (S ** 2).sum(1)[None]
        Dw = np.where(ctx.masks[st], D, np.inf)
        acc["cls"].append((pc == yt).mean())
        acc["sc20"].append((DS.argmin(1) == st).mean())
        acc["sc_from_cls"].append((ctx.sc[pc] == st).mean())
        acc["wsc5"].append((Dw.argmin(1) == yt).mean())
        wrong = pc != yt
        acc["err_insc"].append((ctx.sc[pc[wrong]] == st[wrong]).mean() if wrong.any() else np.nan)
    for k, v in acc.items():
        o[f"{tag}_ncc_{k}"] = float(np.mean(v))
    Mf = class_means(X)
    Sf = (ctx.masks.astype(float) / 5) @ Mf
    o[f"{tag}_pr_sc"] = pr(Sf)
    o[f"{tag}_pr_wsc"] = pr(Mf - Sf[ctx.sc])
    o[f"{tag}_pr_all"] = pr(Mf)
    return o


def rotated_bases(X: np.ndarray):
    Xc = X - X.mean(0)
    C = Xc.T @ Xc / N
    ev, U = np.linalg.eigh(C)
    ev, U = ev[::-1], U[:, ::-1]
    k = int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.99) + 1)
    k = max(2, min(k, X.shape[1]))
    yield "pca", Xc @ U[:, :k]
    Q, _ = np.linalg.qr(np.random.RandomState(0).randn(X.shape[1], X.shape[1]))
    yield "rot", X @ Q
    from sklearn.decomposition import FastICA
    kk = min(64, k)
    ica = FastICA(n_components=kk, whiten="unit-variance", random_state=0, max_iter=400, tol=1e-3)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Sx = ica.fit_transform(Xc @ U[:, :k])
    sk = ((Sx - Sx.mean(0)) ** 3).mean(0)
    yield "ica", Sx * np.sign(sk + 1e-12)


def preds_metrics(pred: np.ndarray, ctx: Ctx) -> dict:
    st = ctx.ysc
    ps = ctx.sc[pred]
    wrong = pred != Y
    corr_sc = ps == st
    return {"net_acc": float((pred == Y).mean()), "net_sc_acc": float(corr_sc.mean()),
            "net_err_insc": float(corr_sc[wrong].mean()) if wrong.any() else np.nan,
            "net_wsc_cond": float((pred == Y)[corr_sc].mean()) if corr_sc.any() else np.nan}


def run_lane(ln: dict) -> None:
    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    f_npz, f_pop = cache / f"{ln['key']}.npz", cache / f"{ln['key']}_pop.csv"
    if f_npz.exists() and f_pop.exists():
        print("skip", ln["key"], flush=True)
        return
    t0 = time.time()
    ctx = Ctx(superclass_of_class(ln["dir"]))
    arrays, rows = {}, []
    for c in ln["ckpts"]:
        acts = load_acts(ln, c["label"])
        if "layer4" in acts and ln["arch"] == "ResNet-18":
            assert np.array_equal(acts["layer4"], acts["avgpool"]), (ln["key"], c["label"])
        try:
            pm = preds_metrics(load_preds(ln, c["label"]), ctx)
        except FileNotFoundError:
            pm = {}
        for L in ln["layers"]:
            X = acts[L]
            row = {"lane": ln["key"], "group": ln["group"], "arch": ln["arch"], "seed": ln["seed"],
                   "checkpoint": c["label"], "idx": c["idx"], "tau": c["tau"], "epoch": c["epoch"],
                   "val_acc": c["val_acc"], "layer": L, "is_pooled": L == ln["pooled"], "d": X.shape[1], **pm}
            row.update(population(X, ctx, "raw"))
            row.update(population(whiten(X), ctx, "wht"))
            rows.append(row)
            bases = [("raw", X)]
            if L == ln["pooled"]:
                bases += list(rotated_bases(X))
            for bname, Xb in bases:
                for k, v in unit_census(Xb, ctx).items():
                    arrays[f"{c['label']}|{L}|{bname}|{k}"] = v
            print(f"  {ln['key']} ck {c['label']} {L} {time.time() - t0:.0f}s", flush=True)
        del acts
    np.savez_compressed(f_npz, **arrays)
    pd.DataFrame(rows).to_csv(f_pop, index=False)
    print("done", ln["key"], f"{time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    for ln in open_lanes():
        if a.only and not ln["key"].startswith(a.only):
            continue
        run_lane(ln)


if __name__ == "__main__":
    main()
