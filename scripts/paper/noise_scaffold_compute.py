#!/usr/bin/env python3
"""Scaffold-quality measurements for the 35-run CIFAR-100 ResNet-18 label-noise experiment.

For every lane and milestone:
  * linear-probe accuracies (5-fold class-stratified CV, train-fold standardisation, ridge one-vs-rest
    least-squares classifier over a grid of penalties) on layer1/layer2/layer3/avgpool for three targets:
    sc (20-way superclass), fine (100-way class), wsc (5-way class within each true superclass, averaged);
  * behavioural metrics from the network's clean-test predictions (sample_predictions.pt, 100 images per class):
    fine accuracy, superclass accuracy, within-superclass accuracy given a correct superclass;
  * geometry on avgpool and layer3: superclass Fisher ratio tr(S_B)/tr(S_W), within-superclass fine Fisher ratio,
    sample-level silhouette (euclidean) and class-centroid silhouette (cosine), both on superclass labels.
layer4 equals avgpool bit for bit in this ResNet-18 (1x1 spatial map) and milestone_7 re-captures milestone_6;
both are verified per lane. Outputs go to analysis_outputs/paper/NOISE_scaffold/.

Usage: python scripts/paper/noise_scaffold_compute.py [lane_id or condition ...]
"""
import hashlib
import io
import json
import os
import pickle
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

EXT = RDT_DATA + "/core"
EXP = "0c28799c-ec9d-4e45-ac30-302dff810a8d"
LANES_DIR = f"{EXT}/experiments/{EXP}/lanes"
DB = f"{EXT}/rcx.db"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "analysis_outputs", "paper", "NOISE_scaffold")

PROBE_LAYERS = ["layer1", "layer2", "layer3", "avgpool"]
GEOM_LAYERS = ["layer3", "avgpool"]
MILESTONES = list(range(8))
MILESTONE_EPOCH = {0: 0, 1: 5, 2: 10, 3: 15, 4: 20, 5: 25, 6: 30, 7: 30}
ALPHAS = [1.0, 10.0, 100.0, 1e3, 1e4, 1e5]
N_FOLDS = 5

# Standard CIFAR-100 coarse mapping (verified against the test-set
# coarse_labels of the run's own dataset copy).
COARSE_TO_FINE = {
    0: [4, 30, 55, 72, 95], 1: [1, 32, 67, 73, 91], 2: [54, 62, 70, 82, 92],
    3: [9, 10, 16, 28, 61], 4: [0, 51, 53, 57, 83], 5: [22, 39, 40, 86, 87],
    6: [5, 20, 25, 84, 94], 7: [6, 7, 14, 18, 24], 8: [3, 42, 43, 88, 97],
    9: [12, 17, 37, 68, 76], 10: [23, 33, 49, 60, 71], 11: [15, 19, 21, 31, 38],
    12: [34, 63, 64, 66, 75], 13: [26, 45, 77, 79, 99], 14: [2, 11, 35, 46, 98],
    15: [27, 29, 44, 78, 93], 16: [36, 50, 65, 74, 80], 17: [47, 52, 56, 59, 96],
    18: [8, 13, 48, 58, 90], 19: [41, 69, 81, 85, 89],
}
FINE_TO_COARSE = np.zeros(100, dtype=int)
for _c, _fs in COARSE_TO_FINE.items():
    FINE_TO_COARSE[_fs] = _c


def lane_registry():
    con = sqlite3.connect(f"file:{DB}?immutable=1", uri=True)
    rows = con.execute(
        "select id, name from lanes where experiment_id=? order by order_index", (EXP,)
    ).fetchall()
    con.close()
    reg = []
    for lid, name in rows:
        # ResNet18-noise-<condition>-seed<k>
        body = name.replace("ResNet18-noise-", "")
        cond, seed = body.rsplit("-seed", 1)
        reg.append({"lane_id": lid, "condition": cond, "seed": int(seed)})
    return reg


def fold_ids():
    """Class-stratified fold id for each of the 10k samples (class-major order)."""
    rng = np.random.default_rng(0)
    f = np.empty(10000, dtype=int)
    for c in range(100):
        f[c * 100:(c + 1) * 100] = rng.permutation(100) % N_FOLDS
    return f


FOLDS = fold_ids()
Y_FINE = np.repeat(np.arange(100), 100)
Y_SC = FINE_TO_COARSE[Y_FINE]


def ridge_cv(X, ys, folds, alphas):
    """Ridge one-vs-rest classifier, k-fold CV. X (n,d) float64 tensor.
    ys: dict name -> int labels (n,). Returns {name: {alpha: acc}}."""
    n, d = X.shape
    res = {k: {a: 0 for a in alphas} for k in ys}
    for k in range(N_FOLDS):
        tr = torch.from_numpy(folds != k)
        te = ~tr
        Xtr, Xte = X[tr], X[te]
        mu = Xtr.mean(0)
        sd = Xtr.std(0).clamp_min(1e-8)
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
        G = Xtr.T @ Xtr
        lam, V = torch.linalg.eigh(G)
        lam = lam.clamp_min(0)
        XteV = Xte @ V
        for name, y in ys.items():
            y = torch.from_numpy(y)
            ytr, yte = y[tr], y[te]
            classes = torch.unique(ytr)
            Y = (ytr[:, None] == classes[None, :]).double()
            ybar = Y.mean(0)
            B = V.T @ (Xtr.T @ (Y - ybar))
            for a in alphas:
                S = XteV @ (B / (lam + a)[:, None]) + ybar
                pred = classes[S.argmax(1)]
                res[name][a] += int((pred == yte).sum())
    return {name: {a: v / n for a, v in d_.items()} for name, d_ in res.items()}


def within_sc_probe(X, alphas):
    accs = {a: [] for a in alphas}
    for c, fs in COARSE_TO_FINE.items():
        idx = np.concatenate([np.arange(f * 100, (f + 1) * 100) for f in fs])
        r = ridge_cv(X[torch.from_numpy(idx)], {"w": Y_FINE[idx]}, FOLDS[idx], alphas)["w"]
        for a in alphas:
            accs[a].append(r[a])
    return {a: float(np.mean(v)) for a, v in accs.items()}


def silhouette(X, labels, metric="euclidean", chunk=2000):
    X = X.double()
    if metric == "cosine":
        X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    n = X.shape[0]
    labs = torch.from_numpy(labels)
    K = int(labs.max()) + 1
    onehot = torch.nn.functional.one_hot(labs, K).double()
    counts = onehot.sum(0)
    s = torch.empty(n, dtype=torch.float64)
    sq = (X * X).sum(1)
    for i in range(0, n, chunk):
        xi = X[i:i + chunk]
        if metric == "cosine":
            D = (1 - xi @ X.T).clamp_min(0)
        else:
            D = (sq[i:i + chunk, None] + sq[None, :] - 2 * xi @ X.T).clamp_min(0).sqrt()
        sums = D @ onehot  # (chunk, K)
        li = labs[i:i + chunk]
        own_cnt = counts[li] - 1
        a = sums[torch.arange(len(li)), li] / own_cnt.clamp_min(1)
        mean_other = sums / counts[None, :]
        mean_other[torch.arange(len(li)), li] = float("inf")
        b = mean_other.min(1).values
        s[i:i + chunk] = (b - a) / torch.maximum(a, b).clamp_min(1e-12)
    return float(s.mean())


def geometry(X):
    X = X.double()
    mu = X.mean(0)
    sc = torch.from_numpy(Y_SC)
    fine = torch.from_numpy(Y_FINE)
    mu_c = torch.stack([X[fine == c].mean(0) for c in range(100)])
    mu_k = torch.stack([X[sc == k].mean(0) for k in range(20)])
    sw_sc = float(((X - mu_k[sc]) ** 2).sum())
    sb_sc = float((((mu_k - mu) ** 2).sum(1) * 500).sum())
    sw_fine = float(((X - mu_c[fine]) ** 2).sum())
    sb_fine_within = float((((mu_c - mu_k[torch.from_numpy(FINE_TO_COARSE)]) ** 2).sum(1) * 100).sum())
    return {
        "fisher_sc": sb_sc / sw_sc,
        "fisher_fine_within_sc": sb_fine_within / sw_fine,
        "silhouette_sc": silhouette(X, Y_SC, "euclidean"),
        "silhouette_sc_centroid_cos": silhouette(mu_c, FINE_TO_COARSE.copy(), "cosine"),
    }


def tensor_md5(d):
    h = hashlib.md5()
    for layer in sorted(d):
        for c in sorted(d[layer]):
            h.update(d[layer][c].numpy().tobytes())
    return h.hexdigest()


CACHE = os.environ.get("NOISE_SCAFFOLD_CACHE", "/tmp/noise_scaffold_cache")


def tload(path):
    """Sequential whole-file read, then parse from memory (2-3x faster than
    torch.load's seek-heavy zip access on a hard disk)."""
    with open(path, "rb") as fh:
        return torch.load(io.BytesIO(fh.read()), map_location="cpu", weights_only=False)


def process_lane(info):
    os.makedirs(CACHE, exist_ok=True)
    cpath = f"{CACHE}/{info['lane_id']}.pkl"
    if os.path.exists(cpath):
        return pickle.load(open(cpath, "rb"))
    out = _process_lane(info)
    pickle.dump(out, open(cpath, "wb"))
    return out


def _process_lane(info):
    torch.set_num_threads(2)
    lid = info["lane_id"]
    base = f"{LANES_DIR}/{lid}/dev_snapshots"
    snaps = {s["milestone_index"]: s for s in json.load(open(f"{base}/metrics/devtrain_snapshots.json"))}
    probe_rows, rows = [], []
    md5 = {}
    for m in MILESTONES:
        mdir = f"{base}/milestone_{m}"
        acts = tload(f"{mdir}/individual_multilayer_activations.pt")
        preds = tload(f"{mdir}/sample_predictions.pt")
        md5[m] = tensor_md5(acts)
        l4_eq_pool = all(torch.equal(acts["layer4"][c], acts["avgpool"][c]) for c in range(100))
        # Behavioural
        P = np.concatenate([preds[c]["predictions"].numpy() for c in range(100)])
        corr_flag = np.concatenate([preds[c]["correct"].numpy() for c in range(100)])
        assert np.array_equal(corr_flag, P == Y_FINE)
        fine_acc = float((P == Y_FINE).mean())
        sc_ok = FINE_TO_COARSE[P] == Y_SC
        sc_acc = float(sc_ok.mean())
        row = dict(info, milestone=m, epoch=MILESTONE_EPOCH[m],
                   val_acc_logged=snaps[m]["val_accuracy"] / 100.0,
                   fine_acc=fine_acc, sc_acc=sc_acc,
                   wsc_cond_acc=float((P == Y_FINE)[sc_ok].mean()) if sc_ok.any() else np.nan,
                   err_within_sc_share=float((sc_ok & (P != Y_FINE)).sum() / max((P != Y_FINE).sum(), 1)),
                   layer4_equals_avgpool=l4_eq_pool, act_md5=md5[m])
        if m == 7:
            row["ms7_equals_ms6_acts"] = md5[7] == md5[6]
            rows.append(row)
            continue  # identical model: no need to re-probe
        for layer in PROBE_LAYERS:
            X = torch.cat([acts[layer][c] for c in range(100)]).double()
            r = ridge_cv(X, {"sc": Y_SC, "fine": Y_FINE}, FOLDS, ALPHAS)
            w = within_sc_probe(X, ALPHAS)
            for a in ALPHAS:
                probe_rows.append(dict(lane_id=lid, condition=info["condition"], seed=info["seed"],
                                       milestone=m, epoch=MILESTONE_EPOCH[m], layer=layer, alpha=a,
                                       probe_sc=r["sc"][a], probe_fine=r["fine"][a], probe_wsc=w[a]))
            if layer in GEOM_LAYERS:
                for k, v in geometry(X).items():
                    row[f"{k}_{layer}"] = v
        rows.append(row)
    return rows, probe_rows


def main():
    reg = lane_registry()
    assert len(reg) == 35
    only = sys.argv[1:]  # optional subset of lane ids for testing
    if only:
        reg = [r for r in reg if r["lane_id"] in only or r["condition"] in only]
    all_rows, all_probe = [], []
    with ProcessPoolExecutor(max_workers=int(os.environ.get("NOISE_SCAFFOLD_WORKERS", 5))) as ex:
        futs = {ex.submit(process_lane, r): r for r in reg}
        for fu in as_completed(futs):
            r = futs[fu]
            rows, prow = fu.result()
            all_rows += rows
            all_probe += prow
            print(f"done {r['condition']:>15s} seed {r['seed']:>3d}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    suffix = "_subset" if only else ""
    pd.DataFrame(all_rows).sort_values(["condition", "seed", "milestone"]).to_csv(
        f"{OUT}/raw_behaviour_geometry{suffix}.csv", index=False)
    pd.DataFrame(all_probe).sort_values(["condition", "seed", "milestone", "layer", "alpha"]).to_csv(
        f"{OUT}/raw_probe_alpha_grid{suffix}.csv", index=False)


if __name__ == "__main__":
    main()
