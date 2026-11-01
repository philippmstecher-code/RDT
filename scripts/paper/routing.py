#!/usr/bin/env python3
"""Routing in ResNet-18: do superclass-selective (Ab) SAE features carry the between-superclass decision and
class-selective (Di) features the within-superclass decision? Existing SAEs and checkpoints only (CPU).

Classifier input = avgpool vector x; logits = W x + b; SAE: x = D h + b_dec + resid.
Feature categories per checkpoint from the SAE codes of all 10,000 test images with the lane's adaptive thresholds and
the pipeline formulas (SAI, SSI = normalised best-group fraction, CSI = normalised max within the best group; alive
>= 1%): Tg = SAI > t_sai; Ab = SSI > t_ssi & CSI <= t_csi & not Tg; Di = CSI > t_csi & not Tg; other = remaining alive.
The pseudo control recomputes SSI / CSI / best group with partition 0 (RandomState(0), 20 groups of five).
Routing, part 1 (key 'R1', function r1): exact margin attribution over correctly classified images
    (competitor fixed from the full logits):
    m_b = mean true-group logit - mean logit of the strongest other group;  m_w = true logit - strongest same-group logit;
    share_G(m) = mean contribution of group G / mean m;  Delta = (Ab(m_b) - Ab(m_w)) + (Di(m_w) - Di(m_b)).
Routing, part 2 (key 'R2', function r2): for each group k: subtract the reconstruction contributions of the
    Ab features whose best group is k; drop in group accuracy of the head for the images of every group j (20 x 20);
    S = mean diagonal - mean off-diagonal.
    Control: per k, the same number of alive non-Ab non-Tg features nearest in log activation mass (no repetition).
Writes analysis_outputs/paper/ROUTING/.

Usage: routing.py [--epochs 5,30]
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import transfer_lib as T  # noqa: E402
import tg_dominant as E  # noqa: E402  (forward hooks, SAE loader)

OUT = T.REPO / "analysis_outputs" / "paper" / "ROUTING"
OUT.mkdir(parents=True, exist_ok=True)
_d = pickle.load(open(T.CIFAR / "test", "rb"), encoding="latin1")
TRUE = np.zeros(100, int)
for f, c in zip(_d["fine_labels"], _d["coarse_labels"]):
    TRUE[f] = c
PSEUDO = np.empty(100, int)
for g, row in enumerate(np.random.RandomState(0).permutation(100).reshape(20, 5)):
    PSEUDO[row] = g


def categories(H, y, grp, th):
    alive = (H > 1e-6).mean(0) >= 0.01
    M = np.stack([H[y == c].mean(0) for c in range(100)], 1)  # F x 100
    tot = M.sum(1) + 1e-12
    p = M / tot[:, None]
    sai = np.where(alive, -(np.where(p > 0, p * np.log(p + 1e-12), 0)).sum(1) / np.log(100), 0)
    G = np.stack([M[:, grp == k].sum(1) for k in range(20)], 1) / tot[:, None]
    best = G.argmax(1)
    ssi = np.clip((G.max(1) - 1 / 20) / (1 - 1 / 20), 0, None)
    csi = np.zeros(len(M))
    for f in range(len(M)):
        w = M[f, grp == best[f]]
        if w.sum() > 1e-12:
            csi[f] = max((w.max() / w.sum() - 1 / len(w)) / (1 - 1 / len(w)), 0)
    tg = alive & (sai > th["sai"])
    ab = alive & (ssi > th["ssi"]) & (csi <= th["csi"]) & ~tg
    di = alive & (csi > th["csi"]) & ~tg
    other = alive & ~tg & ~ab & ~di
    mass = H.sum(0)
    return {"tg": tg, "ab": ab, "di": di, "other": other, "best": best, "mass": mass, "alive": alive}


def r1(x, H, D, bdec, W, b, y, grp, cat):
    logits = x @ W.T + b
    pred = logits.argmax(1)
    ok = pred == y
    contrib = {}
    for g in ("tg", "ab", "di", "other"):
        idx = np.flatnonzero(cat[g])
        contrib[g] = (H[:, idx] @ D[:, idx].T) @ W.T if len(idx) else np.zeros_like(logits)
    xhat = H @ D.T + bdec
    contrib["resid"] = (x - xhat) @ W.T
    contrib["bias"] = np.tile(bdec @ W.T + b, (len(x), 1))
    n = len(x)
    gl = np.stack([logits[:, grp == k].mean(1) for k in range(20)], 1)
    tg_ = grp[y]
    gl2 = gl.copy()
    gl2[np.arange(n), tg_] = -np.inf
    kstar = gl2.argmax(1)
    sib = np.where(grp[None, :] == tg_[:, None], logits, -np.inf)
    sib[np.arange(n), y] = -np.inf
    cstar = sib.argmax(1)
    m_b = gl[np.arange(n), tg_] - gl[np.arange(n), kstar]
    m_w = logits[np.arange(n), y] - logits[np.arange(n), cstar]
    sh = {}
    for g, C in contrib.items():
        cg = np.stack([C[:, grp == k].mean(1) for k in range(20)], 1)
        cb = cg[np.arange(n), tg_] - cg[np.arange(n), kstar]
        cw = C[np.arange(n), y] - C[np.arange(n), cstar]
        sh[g] = {"m_b": float(cb[ok].mean() / m_b[ok].mean()), "m_w": float(cw[ok].mean() / m_w[ok].mean())}
    delta = (sh["ab"]["m_b"] - sh["ab"]["m_w"]) + (sh["di"]["m_w"] - sh["di"]["m_b"])
    return {"shares": sh, "delta": float(delta), "n_correct": int(ok.sum()),
            "n_feat": {g: int(cat[g].sum()) for g in ("tg", "ab", "di", "other")}}


def r2(x, H, D, W, b, y, grp, cat, rng):
    base_pred = (x @ W.T + b).argmax(1)
    base_acc = np.array([(grp[base_pred[grp[y] == j]] == j).mean() for j in range(20)])
    used = set()
    Mt, Mc, nk = np.full((20, 20), np.nan), np.full((20, 20), np.nan), []
    pool = np.flatnonzero(cat["other"] | cat["di"])
    lm = np.log(cat["mass"] + 1e-12)
    for k in range(20):
        S = np.flatnonzero(cat["ab"] & (cat["best"] == k))
        nk.append(len(S))
        if len(S) == 0:
            continue
        ctrl = []
        for f in S:
            cand = [q for q in pool[np.argsort(np.abs(lm[pool] - lm[f]))] if q not in used]
            if cand:
                ctrl.append(cand[0])
                used.add(cand[0])
        for name, idx, M in (("t", S, Mt), ("c", np.array(ctrl), Mc)):
            xa = x - H[:, idx] @ D[:, idx].T
            pa = (xa @ W.T + b).argmax(1)
            acc = np.array([(grp[pa[grp[y] == j]] == j).mean() for j in range(20)])
            M[k] = base_acc - acc
    def spec(M):
        rows = ~np.isnan(M).all(1)
        A = M[rows]
        diag = np.array([A[i, k] for i, k in enumerate(np.flatnonzero(rows))])
        off = (A.sum(1) - diag) / 19
        return float(diag.mean() - off.mean())
    return {"S": spec(Mt), "S_ctrl": spec(Mc), "n_ab_per_group": nk, "matrix": Mt.tolist(), "matrix_ctrl": Mc.tolist()}


def analyse(lane_dir, label, cf, rng):
    lane = {"dir": str(lane_dir), "arch": "ResNet-18", "group": "extended"}
    net = T.load_net(lane, label)
    F, logits = E.forward(net, cf["X"])
    x = F["layer4"].double().numpy()  # avgpool == spatial mean of layer4 (ResNet-18, 1x1)
    sae = E.load_sae(lane, label, "avgpool")
    with torch.no_grad():
        H = sae.encode(torch.from_numpy(x).float()).numpy().astype(np.float64)
    D = sae.decoder.weight.detach().numpy().astype(np.float64)
    bdec = sae.decoder.bias.detach().numpy().astype(np.float64)
    W = net.fc.weight.detach().numpy().astype(np.float64)
    b = net.fc.bias.detach().numpy().astype(np.float64)
    assert np.allclose(x @ W.T + b, logits.numpy(), atol=1e-3), "avgpool x does not reproduce the logits"
    md = json.load(open(Path(lane_dir) / "sae_analysis" / "sae_results.json"))["metadata"]["adaptive_thresholds"]
    th = {"sai": md["sai_adaptive_thresh"], "ssi": md["ssi_adaptive_thresh"], "csi": md["csi_adaptive_thresh"]}
    y = cf["y"]
    out = {}
    for name, grp in (("true", TRUE), ("pseudo", PSEUDO)):
        cat = categories(H, y, grp, th)
        out[name] = {"part1": r1(x, H, D, bdec, W, b, y, grp, cat), "part2": r2(x, H, D, W, b, y, grp, cat, rng)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", default="30")
    a = ap.parse_args()
    eps = [int(e) for e in a.epochs.split(",")]
    cf = T.cifar_test()
    res = {}
    for s in T.SEEDS:
        d = T.RUNS_EXTENDED / f"static_noise_s{s}" / "lanes" / "standard_resnet18"
        lab_of = {c["epoch"]: c["label"] for c in T.real_ckpts(T.checkpoint_info(d, T._snap_labels(d)))}
        for e in eps:
            r = analyse(d, lab_of[e], cf, np.random.default_rng(s))
            res[f"{s}|{e}"] = r
            t, p = r["true"], r["pseudo"]
            print(f"seed {s} ep {e}: Delta true {t['R1']['delta']:+.3f} pseudo {p['R1']['delta']:+.3f} | "
                  f"Ab share m_b {t['R1']['shares']['ab']['m_b']:.2f} m_w {t['R1']['shares']['ab']['m_w']:.2f} | "
                  f"Di share m_b {t['R1']['shares']['di']['m_b']:.2f} m_w {t['R1']['shares']['di']['m_w']:.2f} | "
                  f"S true {t['R2']['S']:.3f} ctrl {t['R2']['S_ctrl']:.3f} | S pseudo {p['R2']['S']:.3f} | nfeat {t['R1']['n_feat']}",
                  flush=True)
    json.dump(res, open(OUT / f"routing_resnet_clean_ep{a.epochs.replace(',', '-')}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
