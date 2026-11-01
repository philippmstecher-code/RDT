#!/usr/bin/env python3
"""Pseudo-superclass subspace control for the suppression results. Analysis only, CPU only (2 threads), read-only on
all data; reuses the machinery of mean_shift.py.

Networks: critical_sham sham_e00_seed{S} at epochs 1,2,3,5,10 (window 1-10) and 11,12,13,15,20 (window 11-20;
--sham-11-20 sham_e00 by default, since the epoch-0 sham trains to epoch 30, or sham_e10), S in {7,42,137,256,314}.
Subspaces per stage, estimated on the pooled activations of CIFAR-100 test images 0-19 of every class:
  true   : superclass directions, iterative (between-group scatter in float64, evr floor 0.02, q_max 19 per pass,
           accumulate, numerical-zero guard 1e-10 x initial between-group variance), no rank cap: stops when the
           floor or the guard stops it -> r_true per stage;
  pseudo : the same procedure on 20 fixed partitions RandomState(p).permutation(100).reshape(20, 5), p = 0..19
           (row = group; class indices in CIFAR fine-label order), truncated to r_true per stage (achieved ranks
           reported);
  random : random orthonormal directions of rank r_true (same generator seeds as 'random_ach' in mean_shift).
Removal: centred x - QQ^T(x - mu) (main) and uncentred x - QQ^T x, in the last block (layer4 + avgpool, main) and at
all stages (layer1-4 + avgpool). Outcome: fine-class accuracy of the network's own head on the 10,000 test images;
cost = A_network - A_removed.

Usage:  pseudo_superclass.py run --config lastblock|allstages [--seeds ...] [--windows ...]
        pseudo_superclass.py analyze
Writes analysis_outputs/paper/PSEUDO_superclass/.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mean_shift import (  # noqa: E402
    ALL, CLS2SC, LAST, MAX_PASS, SEEDS, WINDOWS, acc, between_sc, cifar_test, epoch_label, estimate_coarse,
    forward, lastblock_logits, layer4_raw, load_net, proj_apply, random_basis, sham_dir, stored_predictions, tci)

torch.set_num_threads(2)
REPO = HERE.parents[1]
OUT = REPO / "analysis_outputs" / "paper" / "PSEUDO_superclass"
OUT.mkdir(parents=True, exist_ok=True)
MEANSHIFT_DIR = REPO / "analysis_outputs" / "paper" / "MEAN_shift"

N_PART = 20
PARTS = [np.random.RandomState(p).permutation(100).reshape(20, 5) for p in range(N_PART)]


def part_labels(P):
    lab = np.empty(100, dtype=np.int64)
    for g, row in enumerate(P):
        lab[row] = g
    return lab


PLAB = [part_labels(P) for P in PARTS]
NO_CAP = 10 ** 6  # 'true' subspace: no rank target, floor / guard only


def partition_overlap():
    """Same-superclass class pairs (of 200) that are also same-group pairs in each pseudo partition."""
    same_true = CLS2SC[:, None] == CLS2SC[None, :]
    iu = np.triu_indices(100, 1)
    out = []
    for p, lab in enumerate(PLAB):
        same_p = lab[:, None] == lab[None, :]
        out.append({"partition": p, "pairs_shared": int((same_true & same_p)[iu].sum()),
                    "max_classes_from_one_true_sc_in_a_group": int(max(
                        np.bincount(CLS2SC[row], minlength=20).max() for row in PARTS[p]))})
    return out


def between_fine_frac(A, yc, Q):
    """Fraction of the (fine) between-class variance of pooled activations A lying in span(Q)."""
    if Q is None or Q.shape[1] == 0:
        return 0.0
    A = A.double()
    grand = A.mean(0)
    M = torch.stack([A[yc == c].mean(0) - grand for c in range(100)])  # equal class sizes
    tot = float((M ** 2).sum())
    return float(((M @ Q.double()) ** 2).sum()) / tot


def total_var_frac(A, Q):
    if Q is None or Q.shape[1] == 0:
        return 0.0
    A = A.double() - A.double().mean(0)
    return float(((A @ Q.double()) ** 2).sum()) / float((A ** 2).sum())


# one network
def run_network(net, X, y, idx, stages, rseed, lastblock, log):
    """Returns (rows, meta). rows: one per subspace (true, pseudo_p, random) x version."""
    hold = np.setdiff1d(np.arange(len(y)), idx)
    if lastblock:
        base_logits, h4 = layer4_raw(net, X)
        h4e = h4[idx]
        C = {"layer4": h4.shape[1], "avgpool": h4.shape[1]}
        p4 = h4e.mean(dim=(2, 3))
        mus = {"layer4": p4.mean(0), "avgpool": p4.mean(0)}
        pooled_all = h4.mean(dim=(2, 3))
        pooled = {"layer4": pooled_all, "avgpool": pooled_all}  # unprojected, all 10,000 images

        def get_acts(Q):
            z4 = proj_apply(h4e, Q["layer4"], None, False)
            za = proj_apply(z4.mean(dim=(2, 3)), Q["avgpool"], None, False)
            return {"layer4": z4.mean(dim=(2, 3)), "avgpool": za}

        def logits_of(spec, centred):
            return lastblock_logits(net, h4, spec, centred, mus)
    else:
        base_logits, pooled = forward(net, X, capture=stages)  # unprojected pooled activations, all images
        C = {s: pooled[s].shape[1] for s in stages}
        mus = {s: pooled[s][idx].mean(0) for s in stages}

        def get_acts(Q):
            return forward(net, X, spec=Q, capture=stages, capture_idx=idx)[1]

        def logits_of(spec, centred):
            return forward(net, X, spec=spec, centred=centred, mus=mus)[0]

    A_net = acc(base_logits, y)
    t0 = time.time()
    # true
    ysc = torch.as_tensor(CLS2SC[y[idx]])
    Qt, trace_t = estimate_coarse(get_acts, stages, {s: NO_CAP for s in stages}, ysc, C)
    r_true = {s: int(Qt[s].shape[1]) for s in stages}
    subs = {"true": (Qt, {"passes": len(trace_t), "hit_max_pass": len(trace_t) >= MAX_PASS})}
    log(f"   true r={r_true} passes={len(trace_t)} {time.time() - t0:.0f}s")
    # pseudo
    for p in range(N_PART):
        yp = torch.as_tensor(PLAB[p][y[idx]])
        Qp, trace_p = estimate_coarse(get_acts, stages, r_true, yp, C)
        subs[f"pseudo{p:02d}"] = (Qp, {"passes": len(trace_p), "hit_max_pass": len(trace_p) >= MAX_PASS})
    log(f"   pseudo done {time.time() - t0:.0f}s; ranks "
        f"{sorted(set(tuple(int(subs[f'pseudo{p:02d}'][0][s].shape[1]) for s in stages) for p in range(N_PART)))}")
    # random (seeds as mean_shift 'random_ach': rseed*100 + 50 + k)
    Qr = {s: random_basis(C[s], r_true[s], rseed * 100 + 50 + k) for k, s in enumerate(stages)}
    subs["random"] = (Qr, {"passes": 0, "hit_max_pass": False})

    yt = torch.as_tensor(y)
    rows = []
    for name, (Q, info) in subs.items():
        rec = {"subspace": name, "kind": name.rstrip("0123456789"), "partition": int(name[6:]) if name.startswith(
            "pseudo") else -1, "A_net": A_net, **info}
        for s in stages:
            rec[f"rank_{s}"] = int(Q[s].shape[1])
            for tag, ii in (("est", idx), ("hold", hold)):
                rec[f"bcfrac_{tag}_{s}"] = between_fine_frac(pooled[s][ii], yt[ii], Q[s])
            rec[f"totfrac_hold_{s}"] = total_var_frac(pooled[s][hold], Q[s])
        for ver, cen in (("unc", False), ("cen", True)):
            rec[f"A_{ver}"] = acc(logits_of(Q, cen), y)
            rec[f"cost_{ver}"] = A_net - rec[f"A_{ver}"]
        rows.append(rec)
    log(f"   eval done {time.time() - t0:.0f}s")
    return rows, {"r_true": r_true, "trace_true": trace_t, "A_net": A_net, "base_logits": base_logits}


def main_run(a):
    X, y = cifar_test()
    idx = np.concatenate([np.flatnonzero(y == c)[:20] for c in range(100)])
    lastblock = a.config == "lastblock"
    stages = LAST if lastblock else ALL
    logf = open(OUT / f"run_log_{a.config}.txt", "a")

    def log(*m):
        s = " ".join(str(x) for x in m)
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    log(f"=== run {time.strftime('%Y-%m-%d %H:%M:%S')} config={a.config} seeds={a.seeds} windows={a.windows}")
    json.dump(partition_overlap(), open(OUT / "partitions_overlap.json", "w"), indent=1)
    json.dump({p: PARTS[p].tolist() for p in range(N_PART)}, open(OUT / "partitions.json", "w"))
    rdir = OUT / f"raw_{a.config}"
    rdir.mkdir(exist_ok=True)
    for win in a.windows:
        for e in (a.epochs or WINDOWS[win]["epochs"]):
            if e not in WINDOWS[win]["epochs"]:
                continue
            for s in a.seeds:
                fn = rdir / f"{win}_s{s}_e{e}.csv"
                if fn.exists():
                    log("skip (exists)", fn.name)
                    continue
                t0 = time.time()
                sl = sham_dir(win, s)
                lab, va = epoch_label(sl, e)
                net = load_net(sl, lab)
                log(f"{a.config} {win} seed {s} epoch {e} (milestone {lab})")
                rseed = s * 1000 + e + (0 if lastblock else 500)
                rows, meta = run_network(net, X, y, idx, stages, rseed, lastblock, log)
                agree = float((meta["base_logits"].argmax(1).numpy() == stored_predictions(sl, lab)).mean())
                if agree < 0.999:
                    log("ABORT: normalisation check failed", win, s, e, agree)
                    return
                df = pd.DataFrame(rows)
                df.insert(0, "config", a.config)
                df.insert(1, "window", win)
                df.insert(2, "seed", s)
                df.insert(3, "epoch", e)
                df.insert(4, "sham_label", lab)
                df["pred_agreement"] = agree
                df["val_acc_logged"] = va
                df.to_csv(fn, index=False)
                json.dump(meta["trace_true"], open(rdir / f"trace_true_{win}_s{s}_e{e}.json", "w"))
                t = df[df.kind == "true"].iloc[0]
                ps = df[df.kind == "pseudo"]
                rr = df[df.kind == "random"].iloc[0]
                log(f"   A_net {meta['A_net']:.2f} agree {agree:.4f} r_true {meta['r_true']} | cost cen true "
                    f"{t.cost_cen:.2f} pseudo mean {ps.cost_cen.mean():.2f} [{ps.cost_cen.min():.2f},"
                    f"{ps.cost_cen.max():.2f}] random {rr.cost_cen:.2f} | unc true {t.cost_unc:.2f} pseudo "
                    f"{ps.cost_unc.mean():.2f} random {rr.cost_unc:.2f} | {time.time() - t0:.0f}s")


# analysis
def load_raw(config):
    fs = sorted((OUT / f"raw_{config}").glob("w*_s*_e*.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None


def per_network(df):
    """One row per network: true, pseudo mean / p95 / max, random costs; D; ratios."""
    out = []
    for (win, s, e), g in df.groupby(["window", "seed", "epoch"]):
        t = g[g.kind == "true"].iloc[0]
        ps = g[g.kind == "pseudo"]
        r = g[g.kind == "random"].iloc[0]
        rec = {"window": win, "seed": s, "epoch": e, "A_net": t.A_net}
        for v in ("cen", "unc"):
            rec[f"true_{v}"] = t[f"cost_{v}"]
            rec[f"pseudo_mean_{v}"] = ps[f"cost_{v}"].mean()
            rec[f"pseudo_p95_{v}"] = np.percentile(ps[f"cost_{v}"], 95)
            rec[f"pseudo_max_{v}"] = ps[f"cost_{v}"].max()
            rec[f"random_{v}"] = r[f"cost_{v}"]
            rec[f"D_{v}"] = rec[f"true_{v}"] - rec[f"pseudo_mean_{v}"]
            rec[f"rank_true_of21_{v}"] = int(1 + (ps[f"cost_{v}"] > t[f"cost_{v}"]).sum())
        out.append(rec)
    return pd.DataFrame(out)


def contrast(df, win, epochs, v):
    """Contrast quantities for one window / version / epoch set (per-seed means over the epochs)."""
    w = df[(df.window == win) & (df.epoch.isin(epochs))]
    seeds = sorted(w.seed.unique())
    per_seed = {}
    for s in seeds:
        g = w[w.seed == s]
        t = g[g.kind == "true"][f"cost_{v}"].mean()
        pc = g[g.kind == "pseudo"].groupby("partition")[f"cost_{v}"].mean()  # each partition: mean over epochs
        rc = g[g.kind == "random"][f"cost_{v}"].mean()
        per_seed[int(s)] = {"true": t, "pseudo_mean": float(pc.mean()), "pseudo_p95": float(np.percentile(pc, 95)),
                            "pseudo_min": float(pc.min()), "pseudo_max": float(pc.max()), "random": rc,
                            "D": t - float(pc.mean()), "true_gt_p95": bool(t > np.percentile(pc, 95)),
                            "n_pseudo_ge_true": int((pc >= t).sum()), "n_epochs": int(g.epoch.nunique())}
    ps = pd.DataFrame(per_seed).T
    D = tci(ps.D.astype(float).values)
    res = {"epochs": [int(e) for e in epochs], "per_seed": per_seed, "D": D,
           "n_seeds_D_pos": int((ps.D.astype(float) > 0).sum()),
           "n_seeds_true_gt_p95": int(ps.true_gt_p95.astype(bool).sum()),
           "true": tci(ps.true.astype(float).values), "pseudo_mean": tci(ps.pseudo_mean.astype(float).values),
           "random": tci(ps.random.astype(float).values),
           "true_minus_random": tci((ps.true - ps.random).astype(float).values),
           "pseudo_minus_random": tci((ps.pseudo_mean - ps.random).astype(float).values)}
    tm, pm, rm = res["true"]["mean"], res["pseudo_mean"]["mean"], res["random"]["mean"]
    res["ratio_true_pseudo_random"] = [1.0, pm / tm if tm else np.nan, rm / tm if tm else np.nan]
    res["ratio_true_over_pseudo"] = tm / pm if pm else np.nan
    res["ratio_true_over_random"] = tm / rm if rm else np.nan
    n = len(seeds)
    res["n_seeds"] = n
    res["SUPPORTED_rule"] = bool(n == 5 and D["mean"] > 0 and D["lo"] > 0 and res["n_seeds_D_pos"] >= 4
                                 and res["n_seeds_true_gt_p95"] >= 4)
    return res


def variance_summary(df, stages):
    """Fraction of fine between-class variance (held-out images 20-99, and estimation images) per subspace kind."""
    out = {}
    for (win, e), g in df.groupby(["window", "epoch"]):
        rec = {}
        for kind in ("true", "pseudo", "random"):
            k = g[g.kind == kind]
            for s in stages:
                for tag in ("hold", "est"):
                    col = f"bcfrac_{tag}_{s}"
                    rec[f"{kind}_{tag}_{s}"] = float(k.groupby("seed")[col].mean().mean())
                rec[f"{kind}_rank_{s}"] = float(k.groupby("seed")[f"rank_{s}"].mean().mean())
                rec[f"{kind}_totfrac_hold_{s}"] = float(k.groupby("seed")[f"totfrac_hold_{s}"].mean().mean())
        out[f"{win}_e{e}"] = rec
    return out


def analyze():
    summ = {"partition_overlap": json.load(open(OUT / "partitions_overlap.json"))}
    for config, stages in (("lastblock", LAST), ("allstages", ALL)):
        df = load_raw(config)
        if df is None:
            summ[config] = "NOT RUN"
            continue
        df.to_csv(OUT / f"{config}_raw.csv", index=False)
        pn = per_network(df)
        pn.to_csv(OUT / f"{config}_per_network.csv", index=False)
        S = {"n_networks": int(len(pn)), "ranks": {}, "pseudo_rank_shortfalls": []}
        for s in stages:
            tr = df[df.kind == "true"]
            S["ranks"][s] = {"r_true_values": sorted(int(x) for x in tr[f"rank_{s}"].unique())}
        ps = df[df.kind == "pseudo"].merge(df[df.kind == "true"][["window", "seed", "epoch"] + [f"rank_{s}" for s in
                                                                                                  stages]],
                                           on=["window", "seed", "epoch"], suffixes=("", "_true"))
        for s in stages:
            short = ps[ps[f"rank_{s}"] < ps[f"rank_{s}_true"]]
            for _, r in short.iterrows():
                S["pseudo_rank_shortfalls"].append({"window": r.window, "seed": int(r.seed), "epoch": int(r.epoch),
                                                    "partition": int(r.partition), "stage": s,
                                                    "rank": int(r[f"rank_{s}"]), "r_true": int(r[f"rank_{s}_true"])})
        S["n_pseudo_rank_shortfalls"] = len(S["pseudo_rank_shortfalls"])
        S["any_hit_max_pass"] = bool(df.hit_max_pass.any())
        for win in WINDOWS:
            if win not in set(df.window):
                continue
            eps = sorted(df[df.window == win].epoch.unique())
            for v in ("cen", "unc"):
                S[f"{win}_{v}_first3"] = contrast(df, win, eps[:3], v)
                S[f"{win}_{v}_all5"] = contrast(df, win, eps, v)
                for e in eps:
                    S[f"{win}_{v}_e{e}"] = contrast(df, win, [e], v)
        S["variance"] = variance_summary(df, stages)
        summ[config] = S
    # cross-check against mean_shift (last block): true == coarse, random == random_ach
    lb = load_raw("lastblock")
    if lb is not None:
        b = pd.read_csv(MEANSHIFT_DIR / "lastblock_raw.csv")
        m = lb[lb.kind.isin(["true", "random"])].merge(b, on=["window", "seed", "epoch"])
        chk = []
        for _, r in m.iterrows():
            ref = "coarse" if r.kind == "true" else "random_ach"
            for v in ("unc", "cen"):
                chk.append({"window": r.window, "seed": r.seed, "epoch": r.epoch, "subspace": r.kind, "version": v,
                            "A_pseudo": r[f"A_{v}"], "A_meanshift": r[f"A_{ref}_{v}"], "A_net_pseudo": r.A_net,
                            "A_sham_meanshift": r.A_sham})
        chk = pd.DataFrame(chk)
        chk["absdiff"] = (chk.A_pseudo - chk.A_meanshift).abs()
        chk.to_csv(OUT / "crosscheck_mean_shift.csv", index=False)
        summ["crosscheck_mean_shift"] = {"n": int(len(chk)), "max_absdiff_pp": float(chk.absdiff.max()),
                                  "max_absdiff_true_pp": float(chk[chk.subspace == "true"].absdiff.max()),
                                  "pass_0.1pp": bool(chk[chk.subspace == "true"].absdiff.max() <= 0.1)}
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1, default=float)
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--config", choices=["lastblock", "allstages"], default="lastblock")
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--windows", nargs="*", default=list(WINDOWS))
    ap.add_argument("--epochs", type=int, nargs="*", default=None)
    ap.add_argument("--sham-11-20", choices=["sham_e00", "sham_e10"], default="sham_e00")
    a = ap.parse_args()
    WINDOWS["w11_20"]["sham"] = a.sham_11_20  # sham_dir() reads this mapping
    if a.cmd == "run":
        main_run(a)
    else:
        analyze()
