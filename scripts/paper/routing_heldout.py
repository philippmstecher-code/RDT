#!/usr/bin/env python3
"""Routing specificity S with feature selection and scoring on DISJOINT images (ResNet-18, 5 clean seeds, ep 30).

Main analysis (routing.py): SAE categories (Ab = superclass-selective, best superclass k) are computed from
all 10,000 CIFAR-100 test images, and the removal effect is scored on the same images. Here, per class, the test images
are split into two halves (by their order within the class). Categories are computed on one half (selection) and S is
scored on the other (held-out); both directions (A->B, B->A) are run and averaged. Identical code otherwise
(routing.categories / r2; lane's stored adaptive thresholds; projection-free exact subtraction as in routing.py for
ResNet-18 'S'; matched-mass control). Also reported: the in-sample S on each half, and the full-sample S of routing.py.
Writes analysis_outputs/paper/ROUTING/routing_heldout.json.
Usage: python scripts/paper/routing_heldout.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import routing as R  # noqa: E402
T, E = R.T, R.E


def proj_S(x, D, W, b, y, cat, mu, c2c):
    """S with the cross-architecture ablation for ResNet-18: mean-preserving projection onto the complement of the
    decoder span (routing_transformers.analyse), matched-mass control, same spec() as routing.py."""
    def sc_acc(xx):
        pred = c2c[(xx @ W.T + b).argmax(1)]
        return np.array([(R.TRUE[pred[R.TRUE[y] == j]] == j).mean() for j in range(20)])
    base = sc_acc(x)
    used, pool = set(), np.flatnonzero(cat["other"] | cat["di"])
    lm = np.log(cat["mass"] + 1e-12)
    Mt, Mc = np.full((20, 20), np.nan), np.full((20, 20), np.nan)
    for k in range(20):
        S = np.flatnonzero(cat["ab"] & (cat["best"] == k))
        if len(S) == 0:
            continue
        ctrl = []
        for f in S:
            cand = [q for q in pool[np.argsort(np.abs(lm[pool] - lm[f]))] if q not in used]
            if cand:
                ctrl.append(int(cand[0])); used.add(cand[0])
        for idx, M in ((S, Mt), (np.array(ctrl), Mc)):
            if len(idx) == 0:
                continue
            Q, _ = np.linalg.qr(D[:, idx])
            M[k] = base - sc_acc(x - ((x - mu) @ Q) @ Q.T)

    def spec(M):
        rows = np.flatnonzero(~np.isnan(M).all(1))
        diag = np.array([M[k, k] for k in rows])
        return float(diag.mean() - ((np.nansum(M[rows], 1) - diag) / 19).mean())
    return {"S": spec(Mt), "S_ctrl": spec(Mc)}


def tci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean()), float(x.mean() - h), float(x.mean() + h)]


def main():
    cf = T.cifar_test()
    y = cf["y"]
    rank = np.zeros(len(y), int)
    for c in range(100):
        idx = np.flatnonzero(y == c)
        rank[idx] = np.arange(len(idx))
    halves = {"A": rank < 50, "B": rank >= 50}
    res = {}
    for s in T.SEEDS:
        d = T.RUNS_EXTENDED / f"static_noise_s{s}" / "lanes" / "standard_resnet18"
        lab = {c["epoch"]: c["label"] for c in T.real_ckpts(T.checkpoint_info(d, T._snap_labels(d)))}[30]
        lane = {"dir": str(d), "arch": "ResNet-18", "group": "extended"}
        net = T.load_net(lane, lab)
        F, logits = E.forward(net, cf["X"])
        x = F["layer4"].double().numpy()
        sae = E.load_sae(lane, lab, "avgpool")
        with torch.no_grad():
            H = sae.encode(torch.from_numpy(x).float()).numpy().astype(np.float64)
        D = sae.decoder.weight.detach().numpy().astype(np.float64)
        W = net.fc.weight.detach().numpy().astype(np.float64)
        b = net.fc.bias.detach().numpy().astype(np.float64)
        md = json.load(open(d / "sae_analysis" / "sae_results.json"))["metadata"]["adaptive_thresholds"]
        th = {"sai": md["sai_adaptive_thresh"], "ssi": md["ssi_adaptive_thresh"], "csi": md["csi_adaptive_thresh"]}
        out = {}
        c2c = np.asarray(T.lane_class_to_canonical(lane))
        mu = x.mean(0)
        cat_full = R.categories(H, y, R.TRUE, th)
        out["proj_full"] = proj_S(x, D, W, b, y, cat_full, mu, c2c)
        out["full_in_sample"] = R.r2(x, H, D, W, b, y, R.TRUE, cat_full, np.random.default_rng(s))
        for sel, test in (("A", "B"), ("B", "A")):
            ms, mt = halves[sel], halves[test]
            cat = R.categories(H[ms], y[ms], R.TRUE, th)
            cat["mass"] = H[ms].sum(0)
            out[f"in_{sel}"] = R.r2(x[ms], H[ms], D, W, b, y[ms], R.TRUE, cat, np.random.default_rng(s))
            out[f"heldout_{sel}->{test}"] = R.r2(x[mt], H[mt], D, W, b, y[mt], R.TRUE, cat, np.random.default_rng(s))
            out[f"n_ab_{sel}"] = int(cat["ab"].sum())
            out[f"proj_in_{sel}"] = proj_S(x[ms], D, W, b, y[ms], cat, x[ms].mean(0), c2c)
            out[f"proj_heldout_{sel}->{test}"] = proj_S(x[mt], D, W, b, y[mt], cat, x[ms].mean(0), c2c)
        out["S_heldout"] = float(np.mean([out["heldout_A->B"]["S"], out["heldout_B->A"]["S"]]))
        out["S_ctrl_heldout"] = float(np.mean([out["heldout_A->B"]["S_ctrl"], out["heldout_B->A"]["S_ctrl"]]))
        out["S_insample_half"] = float(np.mean([out["in_A"]["S"], out["in_B"]["S"]]))
        out["P_full"] = out["proj_full"]["S"]
        out["P_insample_half"] = float(np.mean([out["proj_in_A"]["S"], out["proj_in_B"]["S"]]))
        out["P_heldout"] = float(np.mean([out["proj_heldout_A->B"]["S"], out["proj_heldout_B->A"]["S"]]))
        out["P_ctrl_heldout"] = float(np.mean([out["proj_heldout_A->B"]["S_ctrl"], out["proj_heldout_B->A"]["S_ctrl"]]))
        res[s] = {k: v for k, v in out.items() if not isinstance(v, dict)} | {
            k: {kk: vv for kk, vv in v.items() if kk in ("S", "S_ctrl", "n_ab_per_group")} for k, v in out.items() if isinstance(v, dict)}
        print(s, f"PROJECTION full {out['P_full']:.3f} | in-sample half {out['P_insample_half']:.3f} | held-out "
                 f"{out['P_heldout']:.3f} (ctrl {out['P_ctrl_heldout']:.3f}) || SUBTRACTION full {out['full_in_sample']['S']:.3f} | in-sample half {out['S_insample_half']:.3f} | held-out {out['S_heldout']:.3f} "
              f"(ctrl {out['S_ctrl_heldout']:.3f}) | n_ab {out['n_ab_A']}/{out['n_ab_B']} vs full {int(cat_full['ab'].sum())}", flush=True)
    summ = {k: tci([res[s][k] for s in T.SEEDS]) for k in ("S_heldout", "S_ctrl_heldout", "S_insample_half",
                                                          "P_full", "P_insample_half", "P_heldout", "P_ctrl_heldout")}
    summ["P_heldout_minus_ctrl"] = tci([res[s]["P_heldout"] - res[s]["P_ctrl_heldout"] for s in T.SEEDS])
    summ["S_full"] = tci([res[s]["full_in_sample"]["S"] for s in T.SEEDS])
    summ["heldout_minus_ctrl"] = tci([res[s]["S_heldout"] - res[s]["S_ctrl_heldout"] for s in T.SEEDS])
    summ["n_heldout_gt_ctrl"] = int(sum(res[s]["S_heldout"] > res[s]["S_ctrl_heldout"] for s in T.SEEDS))
    res["_summary"] = summ
    json.dump(res, open(R.OUT / "routing_heldout.json", "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
