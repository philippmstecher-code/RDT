#!/usr/bin/env python3
"""Targeted vs variance-matched random removal of superclass information (complements redundancy.py).

Clean ResNet-18 lanes (static_noise standard_resnet18), 5 seeds, epochs 30 (primary) and 5 (descriptive).
Classifier input x = avgpool vector (= layer4, 1x1). SAE categories exactly as in routing.py
(routing.categories; avgpool SAE; lane's stored adaptive thresholds; true superclasses).
Removal: mean-preserving projection x' = x - (x - mu) Q Q^T with Q = orthonormal basis of
  ab   decoder directions of all superclass-selective (Ab) features
  di   decoder directions of all class-selective (Di) features
  rand_var   random orthonormal directions removing the same fraction of centred variance as the set (20 draws)
  rand_rank  random orthonormal directions of the same number as the set's rank (20 draws)
Read-out-free information (decodability): r_sc = S_sc'/S_sc, r_wsc = S_wsc'/S_wsc.
Primary: D = r_sc(rand_var) - r_sc(ab) at epoch 30; criterion: mean > 0, 95% t-CI excludes 0 and >= 4 of 5 seeds
positive. Secondary: specificity of ab, (1 - r_sc) - (1 - r_wsc) > 0; of di, the reverse.
Writes analysis_outputs/paper/REDUNDANCY/targeted.json.
Usage: python scripts/paper/redundancy_targeted.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import routing as R  # noqa: E402
from decodability import measures_from_acts  # noqa: E402
T, E = R.T, R.E
OUT = T.REPO / "analysis_outputs" / "paper" / "REDUNDANCY"
NDRAW = 20


def tci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean()), float(x.mean() - h), float(x.mean() + h)]


def info(x, y):
    return measures_from_acts({c: torch.from_numpy(x[y == c]).float() for c in range(100)})


def project(xc, mu, Q):
    return xc - (xc @ Q) @ Q.T + mu


def main():
    cf = T.cifar_test()
    y = cf["y"]
    res = {}
    for s in T.SEEDS:
        d = T.RUNS_EXTENDED / f"static_noise_s{s}" / "lanes" / "standard_resnet18"
        lab_of = {c["epoch"]: c["label"] for c in T.real_ckpts(T.checkpoint_info(d, T._snap_labels(d)))}
        for ep in (30, 5):
            lane = {"dir": str(d), "arch": "ResNet-18", "group": "extended"}
            net = T.load_net(lane, lab_of[ep])
            F, _ = E.forward(net, cf["X"])
            x = F["layer4"].double().numpy()
            sae = E.load_sae(lane, lab_of[ep], "avgpool")
            with torch.no_grad():
                H = sae.encode(torch.from_numpy(x).float()).numpy().astype(np.float64)
            D = sae.decoder.weight.detach().numpy().astype(np.float64)
            md = json.load(open(d / "sae_analysis" / "sae_results.json"))["metadata"]["adaptive_thresholds"]
            th = {"sai": md["sai_adaptive_thresh"], "ssi": md["ssi_adaptive_thresh"], "csi": md["csi_adaptive_thresh"]}
            cat = R.categories(H, y, R.TRUE, th)
            mu = x.mean(0)
            xc = x - mu
            tot = (xc ** 2).sum()
            base = info(x, y)
            rng = np.random.default_rng(1000 * s + ep)
            row = {"base": base, "n_ab": int(cat["ab"].sum()), "n_di": int(cat["di"].sum()), "sets": {}}
            for nm in ("ab", "di"):
                idx = np.flatnonzero(cat[nm])
                Q, _ = np.linalg.qr(D[:, idx])
                v = float(((xc @ Q) ** 2).sum() / tot)
                m = info(project(xc, mu, Q), y)
                out = {"n": int(len(idx)), "rank": int(Q.shape[1]), "var_frac": v,
                       "r_sc": m["S_sc"] / base["S_sc"], "r_wsc": m["S_wsc"] / base["S_wsc"]}
                for ctrl in ("rand_var", "rand_rank"):
                    rs, rw, vv = [], [], []
                    for _ in range(NDRAW):
                        Qr, _ = np.linalg.qr(rng.standard_normal((x.shape[1], x.shape[1])))
                        if ctrl == "rand_var":
                            cv = np.cumsum(((xc @ Qr) ** 2).sum(0) / tot)
                            k = int((cv < v).sum()) + 1
                        else:
                            k = Q.shape[1]
                        Qk = Qr[:, :k]
                        vv.append(float(((xc @ Qk) ** 2).sum() / tot))
                        mr = info(project(xc, mu, Qk), y)
                        rs.append(mr["S_sc"] / base["S_sc"])
                        rw.append(mr["S_wsc"] / base["S_wsc"])
                    out[ctrl] = {"r_sc": float(np.mean(rs)), "r_wsc": float(np.mean(rw)), "var_frac": float(np.mean(vv))}
                row["sets"][nm] = out
                print(s, ep, nm, {k: (round(v2, 3) if isinstance(v2, float) else v2) for k, v2 in out.items()
                                  if not isinstance(v2, dict)},
                      {c: {k: round(v2, 3) for k, v2 in out[c].items()} for c in ("rand_var", "rand_rank")}, flush=True)
            res[f"{s}|{ep}"] = row
            json.dump(res, open(OUT / "targeted.json", "w"), indent=1, default=float)
    summ = {}
    for ep in (30, 5):
        g = lambda nm, k, c=None: np.array([res[f"{s}|{ep}"]["sets"][nm][c][k] if c else res[f"{s}|{ep}"]["sets"][nm][k]
                                            for s in T.SEEDS])
        Dp = g("ab", "r_sc", "rand_var") - g("ab", "r_sc")
        summ[f"ep{ep}"] = {
            "D_ab_sc_vs_randvar": {"mean_ci": tci(Dp), "n_pos": int((Dp > 0).sum()), "per_seed": Dp.round(4).tolist()},
            "D_ab_sc_vs_randrank": {"mean_ci": tci(g("ab", "r_sc", "rand_rank") - g("ab", "r_sc"))},
            "D_ab_wsc_vs_randvar": {"mean_ci": tci(g("ab", "r_wsc", "rand_var") - g("ab", "r_wsc"))},
            "spec_ab": {"mean_ci": tci((1 - g("ab", "r_sc")) - (1 - g("ab", "r_wsc")))},
            "spec_di": {"mean_ci": tci((1 - g("di", "r_wsc")) - (1 - g("di", "r_sc")))},
            "D_di_wsc_vs_randvar": {"mean_ci": tci(g("di", "r_wsc", "rand_var") - g("di", "r_wsc"))},
            "means": {nm: {"var_frac": float(g(nm, "var_frac").mean()), "r_sc": float(g(nm, "r_sc").mean()),
                           "r_wsc": float(g(nm, "r_wsc").mean()), "rand_var_r_sc": float(g(nm, "r_sc", "rand_var").mean()),
                           "rand_var_r_wsc": float(g(nm, "r_wsc", "rand_var").mean())} for nm in ("ab", "di")}}
    summ["PRIMARY_supported"] = bool(summ["ep30"]["D_ab_sc_vs_randvar"]["mean_ci"][1] > 0
                                     and summ["ep30"]["D_ab_sc_vs_randvar"]["n_pos"] >= 4)
    res["_summary"] = summ
    json.dump(res, open(OUT / "targeted.json", "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
