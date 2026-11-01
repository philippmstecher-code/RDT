#!/usr/bin/env python3
"""One mechanism for blocking and pre-building: until the coarse structure forms, each class's learning signal is
shared with its superclass siblings. The same three class-level routing measures on every relevant set of runs.

Measures per checkpoint, from the stored pooled-layer activations of the 10,000 CIFAR-100 test images (100 per class,
true labels) and the checkpoint's own classifier (W, b), unchanged from the existing producers:
  learning signal   g = W'(softmax(Wx + b) - onehot(y)) per image (gradient_routing.py, class_routing_signal.py)
  excess_mean       mean over classes of class_routing_signal.excess_of(G_c, G_c): cos^2(class-mean g, mean g of the
                    class's 4 true superclass siblings) minus the same for its group-mates in 20 random groupings
                    (RandomState(0..19)) (the sibling sharing of class_routing.py / class_routing_signal.py)
  frac_excess_pos   share of classes with excess > 0, i.e. sharing above the random-grouping mean (class_routing.py
                    'frac_classes_excess_pos'; the reported 57% at epoch 1)
  frac_above_p95    share of classes whose sibling sharing exceeds the 95th percentile of the 20 random groupings
                    (class_routing.py 'frac_above_null_p95')
  misrouted_rate    share of images predicted in a class of another true superclass (signal_routing.py
                    'image_share_misrouted')
  signal_share_mis  sum ||g||^2 over misrouted images / sum ||g||^2 over all images (signal_routing.py
                    'signal_share_misrouted'; Supplementary Methods 'Routing of the learning signal')
  acc               fine-class accuracy of the head on the stored images (equals the snapshot validation accuracy)

Runs (ResNet-18, CIFAR-100, seeds 7, 42, 137, 256, 314):
  T1 sham        critical_sham sham_e00_seed{s}, epochs 0-15, 20, 25, 30 (epoch 0 = branch checkpoint)
  T2 blocking    gradsc / gradrand / gradenergy / gradrg (gradblock, gradenergy, gradrg; block in epochs 1-10,
                 branched at epoch 0), epochs 1-15; sham as T1
  T3 curriculum  curriculum sc_true_K20, sc_rand_K20 (fine epochs 1, 5, ..., 30), scratch_E30 (5, ..., 30); fine
                 epoch e = global epoch 20 + e (curriculum_routing.fine_checkpoints); fine epoch 1 has no scratch
                 checkpoint, so sham epoch 1 serves as an unpaired proxy (different initialisation), reported apart;
                 contrasts at matched fine epoch (paired with scratch_E30) and at matched accuracy (scratch's measure
                 interpolated over its own checkpoints at the pre-trained accuracy; also on the dense sham curve)
  T4 label noise static_noise_s{s}: clean, within, between, matched (p = 0.3, fixed labels), epochs 0, 5, ..., 30
Statistics: per-seed values; paired differences over seeds; mean with t-based 95% CI; seeds in each direction.

Outputs: analysis_outputs/paper/SHARED_signal/
  checkpoints_long.csv          every lane x seed x checkpoint (cache; --recompute rebuilds)
  verification.json             agreement with the existing outputs (max |difference| per measure)
  T1_baseline/   per_seed.csv, per_epoch.csv, summary.json
  T2_blocking/   per_seed.csv, per_epoch.csv, contrasts.csv, contrasts_per_epoch.csv, summary.json
  T3_curriculum/ per_seed.csv, per_epoch.csv, contrasts.csv, summary.json
  T4_label_noise/ per_seed.csv, per_epoch.csv, contrasts.csv, contrasts_per_epoch.csv, summary.json,
                 error_type_long.csv, error_type_per_epoch.csv (diagnostic: class_routing_signal.py decomposition of the
                 sharing into the misrouted and the within-superclass component, and the within-superclass error rate)
Source Data: analysis_outputs/paper/figures/final/source_data/shared_signal_{t1,t2,t3,t4}.csv
Usage: python scripts/paper/shared_signal.py [--recompute] [--workers 4]
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True
from class_routing import loo_share  # noqa: E402
from class_routing_signal import RG, excess_of  # noqa: E402
from coarse_overlap import milestones  # noqa: E402
from curriculum_routing import fine_checkpoints  # noqa: E402
from gradient_routing import classifier  # noqa: E402
from curriculum_probe import SUPER_OF, load  # noqa: E402
from signal_routing import LANES as BLOCK_LANES  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "analysis_outputs/paper"
OUT = RESULTS / "SHARED_signal"
SD = RESULTS / "figures/final/source_data"
E = Path(RDT_DATA + "/extended/experiments")
SEEDS = [7, 42, 137, 256, 314]
NOISE = {"clean": "standard_resnet18", "within": "within_sc_resnet18_throughout",
         "between": "between_sc_resnet18_throughout", "matched": "between_sc_matched_resnet18_throughout"}
CURR = {"true": "sc_true_K20", "rand": "sc_rand_K20", "scratch": "scratch_E30"}
KEYS = ["excess_mean", "frac_excess_pos", "frac_above_p95", "misrouted_rate", "signal_share_mis", "acc"]


# per-checkpoint measures
def snap_val(ld):
    """{milestone index: snapshot validation accuracy (%)}."""
    return {int(x["milestone_index"]): float(x["val_accuracy"])
            for x in json.load(open(ld / "dev_snapshots/metrics/devtrain_snapshots.json"))}


def measure(job):
    ms = Path(job["ms"])
    X, y = load(ms, "avgpool")
    X = X.astype(np.float64)
    assert np.bincount(y, minlength=100).tolist() == [100] * 100
    W, b = classifier(ms)
    Z = X @ W.T + b
    pred = Z.argmax(1)
    P = np.exp(Z - Z.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    P[np.arange(len(y)), y] -= 1.0
    G = P @ W
    e = (G ** 2).sum(1)
    mis = SUPER_OF[pred] != SUPER_OF[y]
    Gc = np.stack([G[y == c].mean(0) for c in range(100)])
    ex = excess_of(Gc, Gc)
    sh = loo_share(Gc, SUPER_OF)
    nulls = np.stack([loo_share(Gc, g) for g in RG])
    assert np.allclose(ex, sh - nulls.mean(0), equal_nan=True)
    acc = float((pred == y).mean())
    if job.get("snap_val") is not None:   # head + stored activations reproduce the snapshot's validation accuracy
        assert abs(100 * acc - job["snap_val"]) < 0.5, (job, acc)
    return {k: v for k, v in job.items() if k != "ms"} | {
        "excess_mean": float(np.nanmean(ex)), "frac_excess_pos": float((ex > 0).mean()),
        "frac_above_p95": float((sh > np.percentile(nulls, 95, axis=0)).mean()),
        "misrouted_rate": float(mis.mean()), "signal_share_mis": float(e[mis].sum() / e.sum()), "acc": acc}


def error_type(job):
    """class_routing_signal.py decomposition, unchanged: excess of the misrouted and of the within-superclass component
    of each class-mean learning signal (sums over the image subset / 100), mean over classes; within-superclass error
    rate. Diagnostic for T4 (where the sibling sharing comes from)."""
    ms = Path(job["ms"])
    X, y = load(ms, "avgpool")
    X = X.astype(np.float64)
    W, b = classifier(ms)
    Z = X @ W.T + b
    pred = Z.argmax(1)
    P = np.exp(Z - Z.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    P[np.arange(len(y)), y] -= 1.0
    G = P @ W
    kind = np.where(pred == y, "correct", np.where(SUPER_OF[pred] == SUPER_OF[y], "within", "misrouted"))
    Gc = np.stack([G[y == c].mean(0) for c in range(100)])
    comp = {k: np.stack([G[(y == c) & (kind == k)].sum(0) / (y == c).sum() for c in range(100)])
            for k in ("misrouted", "within")}
    return {k: v for k, v in job.items() if k != "ms"} | {
        "excess_misrouted": float(np.nanmean(excess_of(comp["misrouted"], Gc))),
        "excess_within": float(np.nanmean(excess_of(comp["within"], Gc))),
        "within_rate": float((kind == "within").mean())}


def jobs():
    out = []
    for s in SEEDS:
        # T1 / T2: sham and blocking arms (lanes exactly as signal_routing.py / gradblock_analysis.py)
        for arm, lf in BLOCK_LANES.items():
            ld = lf(s)
            ms = milestones(ld)
            ms[0] = ld / "dev_snapshots/milestone_0"
            sv = snap_val(ld)
            eps = sorted(ms) if arm == "sham" else [e for e in sorted(ms) if 1 <= e <= 15]
            for e in eps:
                out.append({"family": "block", "arm": arm, "seed": s, "epoch": e, "ms": str(ms[e]),
                            "snap_val": sv[int(ms[e].name.split("_")[1])]})
        # T3: curriculum (fine epochs)
        for arm, name in CURR.items():
            for fe, (ms, val, ge) in sorted(fine_checkpoints(name, s).items()):
                out.append({"family": "curriculum", "arm": arm, "seed": s, "epoch": fe, "global_epoch": ge,
                            "ms": str(ms), "snap_val": val})
        # T4: fixed-label noise
        for arm, name in NOISE.items():
            ld = E / f"static_noise_s{s}/lanes" / name
            ms = milestones(ld)
            ms[0] = ld / "dev_snapshots/milestone_0"
            sv = snap_val(ld)
            for e in sorted(ms):
                out.append({"family": "noise", "arm": arm, "seed": s, "epoch": e, "ms": str(ms[e]),
                            "snap_val": sv[int(ms[e].name.split("_")[1])]})
    return out


def build(recompute, workers):
    f = OUT / "checkpoints_long.csv"
    done = pd.read_csv(f) if f.exists() and not recompute else pd.DataFrame()
    have = set(zip(done.family, done.arm, done.seed, done.epoch)) if len(done) else set()
    todo = [j for j in jobs() if (j["family"], j["arm"], j["seed"], j["epoch"]) not in have]
    print(f"{len(have)} cached, {len(todo)} to compute", flush=True)
    rows = done.to_dict("records") if len(done) else []
    with Pool(workers) as p:
        for i, r in enumerate(p.imap_unordered(measure, todo)):
            rows.append(r)
            if i % 25 == 0 or i == len(todo) - 1:
                print(i + 1, "/", len(todo), r["family"], r["arm"], r["seed"], r["epoch"], flush=True)
                pd.DataFrame(rows).to_csv(f, index=False)
    D = pd.DataFrame(rows).sort_values(["family", "arm", "seed", "epoch"]).reset_index(drop=True)
    D.to_csv(f, index=False)
    return D


# agreement with existing outputs
def verify(D):
    res = {}

    def cmp(tag, mine, ref, pairs):
        m = mine.merge(ref, on=["arm", "seed", "epoch"], suffixes=("", "_ref"))
        res[tag] = {"n_checkpoints": len(m)}
        for a, b in pairs:
            res[tag][f"{a} vs {b}"] = float(np.abs(m[a] - m[b + ("_ref" if b in mine.columns else "")]).max())

    B = D[D.family == "block"]
    sr = pd.read_csv(RESULTS / "SIGNAL_routing/signal_routing_long.csv")
    cmp("SIGNAL_routing/signal_routing_long.csv", B, sr, [("misrouted_rate", "image_share_misrouted"),
                                                          ("signal_share_mis", "signal_share_misrouted")])
    gb = pd.read_csv(RESULTS / "GRADBLOCK/routing_long.csv")
    cmp("GRADBLOCK/routing_long.csv", B, gb.rename(columns={"misrouted_rate": "mr_ref"}),
        [("excess_mean", "excess_all"), ("misrouted_rate", "mr_ref")])
    cr = pd.read_csv(RESULTS / "RANDGROUP_control/class_routing_long.csv").groupby(["seed", "epoch"]).agg(
        excess_ref=("excess", "mean"), pos_ref=("excess", lambda v: (v > 0).mean()), acc_ref=("acc", "mean"),
        p95_ref=("shared", "size")).reset_index()
    L = pd.read_csv(RESULTS / "RANDGROUP_control/class_routing_long.csv")
    cr["p95_ref"] = L.assign(a=L.shared > L.null_p95).groupby(["seed", "epoch"]).a.mean().values
    cmp("RANDGROUP_control/class_routing_long.csv (sham)", B, cr.assign(arm="sham"),
        [("excess_mean", "excess_ref"), ("frac_excess_pos", "pos_ref"), ("frac_above_p95", "p95_ref"),
         ("acc", "acc_ref")])
    cu = pd.read_csv(RESULTS / "CURRICULUM_routing/signal_per_checkpoint.csv")
    cu = cu[cu.design == "K20"].rename(columns={"fine_epoch": "epoch", "excess_all": "ex_ref",
                                                "misrouted_rate": "mr_ref", "acc_act": "acc_ref"})
    cmp("CURRICULUM_routing/signal_per_checkpoint.csv (K20)", D[D.family == "curriculum"], cu,
        [("excess_mean", "ex_ref"), ("misrouted_rate", "mr_ref"), ("acc", "acc_ref")])
    nz = pd.read_csv(RESULTS / "RDT_noise/routing_nesting_long.csv")
    cmp("RDT_noise/routing_nesting_long.csv", D[D.family == "noise"], nz,
        [("misrouted_rate", "image_share_misrouted"), ("signal_share_mis", "signal_share_misrouted")])
    return res


# statistics
def tci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return [float(x.mean()) if len(x) else np.nan, np.nan, np.nan]
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean()), float(x.mean() - h), float(x.mean() + h)]


def summ(d, **tags):
    d = np.asarray(d, float)
    ok = d[np.isfinite(d)]
    m, lo, hi = tci(d)
    return tags | {"n": len(ok), "mean": m, "ci_lo": lo, "ci_hi": hi, "n_pos": int((ok > 0).sum()),
                   "n_neg": int((ok < 0).sum()), "per_seed": json.dumps([round(float(v), 5) for v in d])}


def per_epoch(P, arms):
    rows = []
    for (arm, e), g in P[P.arm.isin(arms)].groupby(["arm", "epoch"]):
        r = {"arm": arm, "epoch": e, "n_seeds": len(g)}
        for k in KEYS:
            m, lo, hi = tci(g[k].values)
            r |= {f"{k}_mean": m, f"{k}_ci_lo": lo, f"{k}_ci_hi": hi}
        rows.append(r)
    return pd.DataFrame(rows)


def contrasts(P, pairs, windows):
    rows = []
    for win, eps in windows.items():
        M = P[P.epoch.isin(eps)].groupby(["arm", "seed"])[KEYS].mean()
        for a, b in pairs:
            for k in KEYS:
                d = np.array([M.loc[(a, s), k] - M.loc[(b, s), k] for s in SEEDS])
                rows.append(summ(d, window=win, contrast=f"{a} - {b}", measure=k))
    return pd.DataFrame(rows)


def contrasts_per_epoch(P, pairs):
    rows = []
    for e in sorted(P.epoch.unique()):
        M = P[P.epoch == e].set_index(["arm", "seed"])
        for a, b in pairs:
            if not all((x, s) in M.index for x in (a, b) for s in SEEDS):
                continue
            for k in KEYS:
                rows.append(summ([M.loc[(a, s), k] - M.loc[(b, s), k] for s in SEEDS], epoch=e,
                                 contrast=f"{a} - {b}", measure=k))
    return pd.DataFrame(rows)


def pick(C, **kw):
    m = np.ones(len(C), bool)
    for k, v in kw.items():
        m &= (C[k] == v).values
    out = {}
    for r in C[m].itertuples():
        key = " | ".join(str(getattr(r, k)) for k in ("window", "epoch", "contrast", "measure") if hasattr(r, k))
        out[key] = [round(r.mean, 5), round(r.ci_lo, 5), round(r.ci_hi, 5), f"{r.n_pos}+/{r.n_neg}-", r.n]
    return out


def save(task, files, summary):
    d = OUT / task
    d.mkdir(parents=True, exist_ok=True)
    for name, df in files.items():
        df.to_csv(d / f"{name}.csv", index=False)
    json.dump(summary, open(d / "summary.json", "w"), indent=1, default=float)


# tasks
def t1(D):
    P = D[(D.family == "block") & (D.arm == "sham")][["arm", "seed", "epoch"] + KEYS]
    pe = per_epoch(P, ["sham"])
    # sharing above the null: one-sample test of the seed means of excess against 0, per epoch
    above = [summ(P[P.epoch == e].set_index("seed").loc[SEEDS, "excess_mean"].values, epoch=e, contrast="sham - null",
                  measure="excess_mean") for e in sorted(P.epoch.unique())]
    A = pd.DataFrame(above)
    pe = pe.merge(A[["epoch", "n_pos", "n_neg"]].rename(columns={"n_pos": "excess_n_seeds_pos",
                                                                 "n_neg": "excess_n_seeds_neg"}), on="epoch")
    win = {"epoch_1": [1], "epochs_2_4": [2, 3, 4], "epochs_1_3": [1, 2, 3], "epochs_1_10": list(range(1, 11)),
           "epochs_11_15": list(range(11, 16)), "epoch_30": [30]}
    W = {w: {k: tci(P[P.epoch.isin(eps)].groupby("seed")[k].mean().loc[SEEDS].values) for k in KEYS}
         for w, eps in win.items()}
    peak = pe.loc[pe.excess_mean_mean.idxmax()]
    summary = {"definition": "sham networks; excess = class_routing_signal.excess_of; signal_share_mis = "
                             "signal_routing.py signal_share_misrouted", "n_seeds": len(SEEDS),
               "by_epoch_mean": pe[["epoch"] + [f"{k}_mean" for k in KEYS] + ["excess_n_seeds_pos"]].round(4)
               .to_dict("records"),
               "windows_mean_ci_over_seeds": W, "excess_vs_null_per_epoch": pick(A),
               "peak_excess_epoch": int(peak.epoch)}
    save("T1_baseline", {"per_seed": P, "per_epoch": pe}, summary)
    pe.to_csv(SD / "shared_signal_t1.csv", index=False)
    return summary


def t2(D):
    P = D[D.family == "block"][["arm", "seed", "epoch"] + KEYS]
    P = P[P.epoch.between(1, 15)]
    arms = ["sham", "gradsc", "gradenergy", "gradrand", "gradrg"]
    pairs = [("gradsc", "gradenergy"), ("gradsc", "gradrand"), ("gradsc", "gradrg"), ("gradsc", "sham"),
             ("gradenergy", "sham"), ("gradrand", "sham"), ("gradrg", "sham"), ("gradrg", "gradenergy")]
    win = {"epochs_1_3": [1, 2, 3], "epochs_1_10": list(range(1, 11)), "epochs_11_15": list(range(11, 16))}
    C = contrasts(P, pairs, win)
    # equal accuracy: each blocked network's measure minus the sham network's (same seed) at the same accuracy,
    # interpolated over sham's checkpoints (epochs 1-15, 20, 25, 30); per seed mean over the window
    Sh = D[(D.family == "block") & (D.arm == "sham") & (D.epoch >= 1)].set_index(["seed", "epoch"])
    rows = []
    for a in ("gradsc", "gradenergy", "gradrand", "gradrg"):
        A = P[P.arm == a].set_index(["seed", "epoch"])
        for wname, eps in (("epochs_1_3", [1, 2, 3]), ("epochs_1_10", list(range(1, 11)))):
            for k in ("excess_mean", "signal_share_mis", "misrouted_rate"):
                d = [np.nanmean([A.loc[(s, e), k] - interp_at(A.loc[(s, e), "acc"], Sh.loc[s].acc.values,
                                                              Sh.loc[s][k].values) for e in eps]) for s in SEEDS]
                rows.append(summ(d, window=f"equal_accuracy_{wname}", contrast=f"{a} - sham", measure=k))
        # peak sharing during the block against the sham peak (epochs 1-15)
        d = [A.loc[s].loc[1:10, "excess_mean"].max() - Sh.loc[s].loc[1:15, "excess_mean"].max() for s in SEEDS]
        rows.append(summ(d, window="peak_block_vs_sham_peak", contrast=f"{a} - sham", measure="excess_mean"))
    C = pd.concat([C, pd.DataFrame(rows)], ignore_index=True)
    CE = contrasts_per_epoch(P, pairs[:4])
    pe = per_epoch(P, arms)
    summary = {"arms": {"gradsc": "superclass-direction block", "gradenergy": "energy-matched",
                        "gradrand": "random directions", "gradrg": "random-grouping directions", "sham": "no block"},
               "block_window": "epochs 1-10; release 11-15",
               "contrasts": {k: pick(C, contrast=k) for k in sorted(C.contrast.unique())}}
    save("T2_blocking", {"per_seed": P, "per_epoch": pe, "contrasts": C, "contrasts_per_epoch": CE}, summary)
    pe.to_csv(SD / "shared_signal_t2.csv", index=False)
    return summary


def misplaced(ms):
    """Per stored test image (class order, 100 per class): True if its stored prediction lies in another superclass."""
    import torch
    sp = torch.load(Path(ms) / "sample_predictions.pt", map_location="cpu", weights_only=False)
    return np.concatenate([SUPER_OF[sp[c]["predictions"].numpy()] != SUPER_OF[c] for c in range(100)])


def rerouting():
    """Re-routing between consecutive checkpoints (sham and blocking arms, epochs 1-15), from the stored per-image
    test predictions: reach = share of images in a wrong superclass at epoch e that are in their own superclass at the
    next checkpoint; slip = share of images in their own superclass at e that are in a wrong one at the next."""
    rows = []
    for arm, lf in BLOCK_LANES.items():
        for s in SEEDS:
            ms = milestones(lf(s))
            eps = [e for e in sorted(ms) if 1 <= e <= 15]
            M = {e: misplaced(ms[e]) for e in eps}
            for a, b in zip(eps[:-1], eps[1:]):
                rows.append({"arm": arm, "seed": s, "from": a, "to": b, "misplaced": M[a].mean(),
                             "reach": (~M[b][M[a]]).mean(), "slip": M[b][~M[a]].mean()})
    R = pd.DataFrame(rows)
    pe = []
    for (arm, a), g in R.groupby(["arm", "from"]):
        r = {"arm": arm, "from": a, "to": int(g.to.iloc[0]), "n_seeds": len(g)}
        for k in ("reach", "slip", "misplaced"):
            m, lo, hi = tci(g[k].values)
            r |= {f"{k}_mean": m, f"{k}_ci_lo": lo, f"{k}_ci_hi": hi}
        pe.append(r)
    pe = pd.DataFrame(pe)
    C = []
    for wname, frm in (("block_1_to_10", range(1, 10)), ("release_10_to_15", range(10, 15)),
                       ("early_1_to_4", range(1, 4))):
        M = R[R["from"].isin(list(frm))].groupby(["arm", "seed"])[["reach", "slip"]].mean()
        for arm in BLOCK_LANES:
            for k in ("reach", "slip"):
                m, lo, hi = tci(M.loc[arm][k].loc[SEEDS].values)
                C.append({"window": wname, "contrast": arm, "measure": k, "n": len(SEEDS), "mean": m, "ci_lo": lo,
                          "ci_hi": hi, "n_pos": len(SEEDS), "n_neg": 0,
                          "per_seed": json.dumps([round(float(v), 5) for v in M.loc[arm][k].loc[SEEDS]])})
        for a, b in (("gradsc", "sham"), ("gradsc", "gradenergy"), ("gradsc", "gradrand"), ("gradsc", "gradrg")):
            for k in ("reach", "slip"):
                C.append(summ((M.loc[a][k] - M.loc[b][k]).loc[SEEDS].values, window=wname, contrast=f"{a} - {b}",
                              measure=k))
    C = pd.DataFrame(C)
    save("T2_blocking", {"rerouting_long": R, "rerouting_per_epoch": pe, "rerouting_contrasts": C},
         json.load(open(OUT / "T2_blocking/summary.json")) | {"rerouting": {
             f"{w}|{c}|{k}": [round(m, 5), round(lo, 5), round(hi, 5)] for w, c, k, m, lo, hi in
             C[["window", "contrast", "measure", "mean", "ci_lo", "ci_hi"]].itertuples(index=False)}})
    pe.to_csv(SD / "shared_signal_rerouting.csv", index=False)
    return R


def interp_at(xq, x, y):
    o = np.argsort(x)
    x, y = np.asarray(x)[o], np.asarray(y)[o]
    return float(np.interp(xq, x, y)) if x[0] <= xq <= x[-1] else np.nan


def t3(D):
    P = D[D.family == "curriculum"][["arm", "seed", "epoch", "global_epoch"] + KEYS]
    shamc = D[(D.family == "block") & (D.arm == "sham") & (D.epoch >= 1)]
    sham1 = shamc[shamc.epoch == 1].assign(arm="scratch_proxy_sham")
    pe = pd.concat([per_epoch(P, list(CURR)), per_epoch(sham1, ["scratch_proxy_sham"])], ignore_index=True)
    rows = []
    for a in ("true", "rand"):
        A = P[P.arm == a].set_index(["seed", "epoch"])
        R = P[P.arm == "scratch"].set_index(["seed", "epoch"])
        for e in sorted(set(A.index.get_level_values(1))):
            if all((s, e) in R.index for s in SEEDS):        # matched fine epoch, paired by initialisation
                for k in KEYS:
                    rows.append(summ([A.loc[(s, e), k] - R.loc[(s, e), k] for s in SEEDS], match="epoch", epoch=e,
                                     contrast=f"{a} - scratch", measure=k))
            # matched accuracy: scratch's measure interpolated over its own checkpoints at the pre-trained accuracy
            for k in KEYS[:-1]:
                d = [A.loc[(s, e), k] - interp_at(A.loc[(s, e), "acc"], R.loc[s].acc.values, R.loc[s][k].values)
                     for s in SEEDS]
                rows.append(summ(d, match="accuracy", epoch=e, contrast=f"{a} - scratch", measure=k)
                            | {"pre_acc": json.dumps([round(100 * A.loc[(s, e), "acc"], 2) for s in SEEDS])})
        # unpaired proxy (sham, different initialisation): fine epoch 1 at matched epoch, and every fine epoch at
        # matched accuracy on the dense sham curve (epochs 1-15, 20, 25, 30), as curriculum_routing.py
        S1 = sham1.set_index("seed")
        for k in KEYS:
            rows.append(summ([A.loc[(s, 1), k] - S1.loc[s, k] for s in SEEDS], match="epoch_proxy_unpaired",
                             epoch=1, contrast=f"{a} - scratch_proxy_sham", measure=k))
        Sh = shamc.set_index(["seed", "epoch"])
        for e in sorted(set(A.index.get_level_values(1))):
            for k in KEYS[:-1]:
                d = [A.loc[(s, e), k] - interp_at(A.loc[(s, e), "acc"], Sh.loc[s].acc.values, Sh.loc[s][k].values)
                     for s in SEEDS]
                rows.append(summ(d, match="accuracy_proxy_unpaired", epoch=e, contrast=f"{a} - scratch_proxy_sham",
                                 measure=k))
        # window: fine epochs 5-10 and 15-30 (matched epoch)
        for win, eps in (("fine_epochs_5_10", [5, 10]), ("fine_epochs_15_30", [15, 20, 25, 30])):
            Ma = P[(P.arm == a) & P.epoch.isin(eps)].groupby("seed")[KEYS].mean()
            Mr = P[(P.arm == "scratch") & P.epoch.isin(eps)].groupby("seed")[KEYS].mean()
            for k in KEYS:
                rows.append(summ((Ma.loc[SEEDS, k] - Mr.loc[SEEDS, k]).values, match=win, epoch=-1,
                                 contrast=f"{a} - scratch", measure=k))
    T = P[P.arm == "true"].set_index(["seed", "epoch"])
    Rr = P[P.arm == "rand"].set_index(["seed", "epoch"])
    for e in sorted(set(T.index.get_level_values(1))):
        for k in KEYS:
            rows.append(summ([T.loc[(s, e), k] - Rr.loc[(s, e), k] for s in SEEDS], match="epoch", epoch=e,
                             contrast="true - rand", measure=k))
    C = pd.DataFrame(rows)
    summary = {"design": "K20: 20 epochs on the 20 true superclasses (true) or the seed's random 20x5 grouping "
                         "(rand), then 30 fine-class epochs (warm restart, fresh classifier); scratch_E30: 30 fine "
                         "epochs, same initialisation; fine epoch e = global epoch 20 + e",
               "checkpoints": {a: sorted(P[(P.arm == a) & (P.seed == 7)].epoch.tolist()) for a in CURR},
               "contrasts": {f"{m}|{c}": pick(C, match=m, contrast=c) for m in C.match.unique()
                             for c in C[C.match == m].contrast.unique()}}
    save("T3_curriculum", {"per_seed": P, "per_epoch": pe, "contrasts": C}, summary)
    pe.to_csv(SD / "shared_signal_t3.csv", index=False)
    return summary


def t4(D, ET):
    P = D[D.family == "noise"][["arm", "seed", "epoch"] + KEYS]
    pairs = [("between", "within"), ("between", "matched"), ("between", "clean"), ("within", "clean"),
             ("matched", "clean"), ("matched", "within")]
    win = {"epochs_5_10": [5, 10], "epochs_15_30": [15, 20, 25, 30], "epochs_5_30": [5, 10, 15, 20, 25, 30]}
    C = contrasts(P, pairs, win)
    CE = contrasts_per_epoch(P[P.epoch > 0], pairs)
    pe = per_epoch(P, list(NOISE))
    # diagnostic: error-type decomposition of the sharing (class_routing_signal.py), per arm and epoch
    X = ET[ET.epoch > 0]
    et = X.groupby(["arm", "epoch"])[["excess_misrouted", "excess_within", "within_rate"]].agg(["mean", "std"])
    et.columns = [f"{a}_{b}" for a, b in et.columns]
    et = et.reset_index()
    M = X[X.epoch.isin(win["epochs_5_10"])].groupby(["arm", "seed"])[["excess_misrouted", "excess_within",
                                                                      "within_rate"]].mean()
    etc = {f"{a} - {b} | {k}": summ([M.loc[(a, s), k] - M.loc[(b, s), k] for s in SEEDS], contrast=f"{a} - {b}",
                                    measure=k) for a, b in pairs for k in M.columns}
    # share of the fine-class deficit carried by superclass assignment (exact factorisation of the network's own
    # predictions: P(fine) = P(right superclass) x P(fine | right superclass)); per seed mean over the window of
    # dlog P(right superclass) / dlog P(fine), between minus each reference
    share = []
    for ref in ("within", "clean", "matched"):
        for wname, eps in (("epochs_5_10", [5, 10]), ("epochs_5_30", [5, 10, 15, 20, 25, 30])):
            d = []
            for s in SEEDS:
                a = P[(P.arm == "between") & (P.seed == s) & P.epoch.isin(eps)].set_index("epoch")
                b = P[(P.arm == ref) & (P.seed == s) & P.epoch.isin(eps)].set_index("epoch")
                d.append(np.log((1 - a.misrouted_rate) / (1 - b.misrouted_rate)).mean()
                         / np.log(a.acc / b.acc).mean())
            share.append(summ(d, window=wname, contrast=f"between - {ref}", measure="assignment_share_of_fine_deficit"))
    share = pd.DataFrame(share)
    summary = {"design": "fixed-label noise p = 0.3 drawn once; measures on the clean test set (TRUE labels) with "
                         "each checkpoint's own classifier; epoch 0 is the shared initialisation",
               "assignment_share": pick(share),
               "contrasts": {k: pick(C, contrast=k) for k in sorted(C.contrast.unique())},
               "per_epoch_contrasts": {k: pick(CE, contrast=k) for k in ("between - within", "between - matched")},
               "error_type_epochs_5_10": {k: [round(v["mean"], 5), round(v["ci_lo"], 5), round(v["ci_hi"], 5),
                                              f"{v['n_pos']}+/{v['n_neg']}-"] for k, v in etc.items()}}
    save("T4_label_noise", {"per_seed": P, "per_epoch": pe, "contrasts": C, "contrasts_per_epoch": CE,
                            "error_type_long": ET, "error_type_per_epoch": et, "assignment_share": share}, summary)
    pe.to_csv(SD / "shared_signal_t4.csv", index=False)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    assert E.exists(), f"experiment data not found: {E}"
    OUT.mkdir(parents=True, exist_ok=True)
    SD.mkdir(parents=True, exist_ok=True)
    D = build(a.recompute, a.workers)
    v = verify(D)
    json.dump(v, open(OUT / "verification.json", "w"), indent=1)
    print(json.dumps(v, indent=1))
    for fn in (t1, t2, t3):
        fn(D)
    f = OUT / "T4_label_noise/error_type_long.csv"
    if f.exists() and not a.recompute:
        ET = pd.read_csv(f)
    else:
        with Pool(a.workers) as p:
            ET = pd.DataFrame(p.map(error_type, [j for j in jobs() if j["family"] == "noise"]))
        ET = ET.drop(columns=["family", "snap_val"]).sort_values(["arm", "seed", "epoch"])
    t4(D, ET)
    rerouting()
    print("done")


if __name__ == "__main__":
    main()
