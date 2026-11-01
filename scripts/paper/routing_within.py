#!/usr/bin/env python3
"""Both halves of routing in one analysis (existing checkpoints and SAEs).
For each architecture, seed and superclass k with superclass-selective (Ab) features, remove those features
(mean-preserving projection at the classifier input; CCT: every token) and, separately, the matched control
features; on k's test images (network's own head, argmax):
  (i)   drop in P(superclass correct)                                   = base - ablated
  (ii)  change in P(fine correct | superclass correct)                  = ablated - base (each among its own assigned set)
  (iii) change in fine accuracy on images assigned to k in both intact and ablated networks (fixed set; no selection)
Per seed: mean over superclasses with Ab features; across seeds: mean, 95% t-CI, seed counts.
ResNet-18 clean seeds at epoch 30 (5); CCT-7 and ViT-Small core seeds 42, 137 at the final checkpoint (2).
Writes analysis_outputs/paper/ROUTING/routing_within.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import routing_transformers as R  # noqa: E402
A, T = R.A, R.T
sys.path.insert(0, str(T.REPO / "packages" / "ml" / "src"))
from sae import SparseAutoencoder  # noqa: E402


@torch.no_grad()
def per_lane(lane, label):
    arch = lane["arch"]
    net = T.load_net(lane, label)
    cf = T.cifar_test()
    c2c = T.lane_class_to_canonical(lane) if lane.get("group") == "core" else np.arange(100)
    y = cf["y"]
    tok = R.site_tokens(net, arch, cf["X"])
    xin = tok.mean(1)
    d = Path(lane["dir"])
    site = {"CCT-7": "norm", "ViT-Small": "encoder_ln", "ResNet-18": "avgpool"}[arch]
    sd = torch.load(d / "sae_analysis" / "saes" / label / f"{site}.pt", map_location="cpu", weights_only=False)
    din = sd["decoder.weight"].shape[0]
    sae = SparseAutoencoder(din, sd["encoder.weight"].shape[0] // din, 32)
    sae.load_state_dict(sd)
    sae.eval()
    H = sae.encode(xin).numpy().astype(np.float64)
    D = sae.decoder.weight.detach().double()
    md = json.load(open(d / "sae_analysis" / "sae_results.json"))["metadata"]["adaptive_thresholds"]
    th = {"sai": md["sai_adaptive_thresh"], "ssi": md["ssi_adaptive_thresh"], "csi": md["csi_adaptive_thresh"]}
    grp = A.TRUE
    cat = A.categories(H, y, grp, th)
    mu = tok.reshape(-1, tok.shape[-1]).mean(0).double()

    def preds(idx=None):
        t = tok
        if idx is not None:
            Q, _ = torch.linalg.qr(D[:, idx])
            tc = tok.double() - mu
            t = (tok.double() - (tc @ Q) @ Q.T).float()
        return c2c[R.head(net, arch, t).argmax(1).numpy()]

    base = preds()
    used = set()
    pool = np.flatnonzero(cat["other"] | cat["di"])
    lm = np.log(cat["mass"] + 1e-12)
    rows = []
    for k in range(20):
        S = np.flatnonzero(cat["ab"] & (cat["best"] == k))
        if len(S) == 0:
            continue
        ctrl = []
        for f in S:
            cand = [q for q in pool[np.argsort(np.abs(lm[pool] - lm[f]))] if q not in used]
            if cand:
                ctrl.append(int(cand[0]))
                used.add(cand[0])
        img = grp[y] == k
        yb, pb = y[img], base[img]
        b_sc = grp[pb] == k
        rec = {"k": k, "n_ab": int(len(S))}
        for name, idx in (("ab", S), ("ctrl", np.array(ctrl))):
            pa = preds(idx)[img]
            a_sc = grp[pa] == k
            both = b_sc & a_sc
            rec[name] = {
                "drop_sc": float(b_sc.mean() - a_sc.mean()),
                "d_cond": float((pa[a_sc] == yb[a_sc]).mean() - (pb[b_sc] == yb[b_sc]).mean()) if a_sc.any() and b_sc.any() else np.nan,
                "d_fixed": float((pa[both] == yb[both]).mean() - (pb[both] == yb[both]).mean()) if both.any() else np.nan,
                "n_both": int(both.sum())}
        rows.append(rec)
    return rows


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return [float(x.mean()) if len(x) else np.nan, np.nan, np.nan]
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean()), float(x.mean() - h), float(x.mean() + h)]


def main():
    lanes = []
    for s in T.SEEDS:
        d = T.RUNS_EXTENDED / f"static_noise_s{s}" / "lanes" / "standard_resnet18"
        lab = {c["epoch"]: c["label"] for c in T.real_ckpts(T.checkpoint_info(d, T._snap_labels(d)))}[30]
        lanes.append(("ResNet-18", s, {"dir": str(d), "arch": "ResNet-18", "group": "extended"}, lab))
    for key, lab in (("core__CCT-7__s42", "9"), ("core__CCT-7__s137", "9"),
                     ("core__ViT-Small__s42", "8"), ("core__ViT-Small__s137", "8")):
        ln = [l for l in T.registry() if l["key"] == key][0]
        lanes.append((ln["arch"], ln["seed"], ln, lab))
    out, summ = {}, {}
    for arch, s, lane, lab in lanes:
        rows = per_lane({**lane, "label": lab}, lab)
        seed_mean = {f"{arm}_{m}": float(np.nanmean([r[arm][m] for r in rows]))
                     for arm in ("ab", "ctrl") for m in ("drop_sc", "d_cond", "d_fixed")}
        out[f"{arch}|{s}"] = {"per_superclass": rows, "seed_mean": seed_mean, "n_superclasses": len(rows)}
        print(arch, s, {k: round(v, 3) for k, v in seed_mean.items()}, "n_k", len(rows), flush=True)
    for arch in ("ResNet-18", "CCT-7", "ViT-Small"):
        sm = [v["seed_mean"] for k, v in out.items() if k.startswith(arch + "|")]
        summ[arch] = {m: ci([x[m] for x in sm]) for m in sm[0]}
        summ[arch]["n_seeds"] = len(sm)
        summ[arch]["seeds_drop_sc_gt_ctrl"] = int(sum(x["ab_drop_sc"] > x["ctrl_drop_sc"] for x in sm))
        summ[arch]["seeds_d_fixed_not_below_ctrl"] = int(sum(x["ab_d_fixed"] >= x["ctrl_d_fixed"] for x in sm))
        summ[arch]["seeds_d_cond_not_below_ctrl"] = int(sum(x["ab_d_cond"] >= x["ctrl_d_cond"] for x in sm))
    json.dump({"summary": summ, "runs": out}, open(A.OUT / "routing_within.json", "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
