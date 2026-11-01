#!/usr/bin/env python3
"""Dose of the acquisition-blocking arms: what share of the learning signal did each arm block, per stage?

Per arm (gradsc, gradrand, gradenergy; sham for reference), seed and epoch 1-10, on the stored checkpoint of that
epoch: the second moment M (C x C) of the per-position gradient of the fine-class cross-entropy at the output of each
stage (layer1-4, pooled layer), measured with no block (the network as stored; BatchNorm in eval mode), on 20 test
images per class (2,000; the size used by the energy-matching rule), and the spatially averaged activations of the
same images. Shares are tr(Q' M Q) / tr(M):
  code_share      share in the current superclass code of that network (all between-superclass directions with
                  eigenvalue > 0, at most 19; lesion.superclass_code_directions). For gradsc a lower bound on what the
                  arm blocks (its accumulated subspace contains earlier codes as well).
  init_share      share in the arm's saved initial basis (rank 8; gradblock_basis_<stage>.npy, never evicted), and
                  init_ratio = init_share / (8 / C): about 1 if the learning signal does not concentrate in blocked
                  directions. For gradrand this tests whether rank / C estimates its dose.
  rank            total blocked rank at the end of that epoch (last re-estimation in the lane's lesion_log.jsonl).
  rank_share      rank / C (expected share of a random subspace of that rank).
The energy arm's achieved share and the superclass arm's share estimated in the same network (shadow basis) come from
the energy arm's lesion_log.jsonl (achieved_grad_share, target_grad_share, measured in-run on the same kind of M).
Writes analysis_outputs/paper/GRADBLOCK/dose_long.csv and dose_summary.json.
Usage: python scripts/paper/gradblock_dose.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, RDT_DATA + "/checkouts/random_grouping/packages/ml/src")
import transfer_lib as T  # noqa: E402
from lesion import superclass_code_directions  # noqa: E402
from curriculum_probe import SUPER_OF  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis_outputs/paper/GRADBLOCK"
E = Path(RDT_DATA + "/extended/experiments")
L5 = "Llayer1-layer2-layer3-layer4-avgpool"
SEEDS = [7, 42, 137, 256, 314]
EPOCHS = range(1, 11)
STAGES = ["layer1", "layer2", "layer3", "layer4", "avgpool"]
LANES = {"sham": lambda s: E / "critical_sham/lanes" / f"sham_e00_seed{s}",
         "gradsc": lambda s: E / "gradblock/lanes" / f"lesion_gradsc_e00_seed{s}_rp65_{L5}",
         "gradrand": lambda s: E / "gradblock/lanes" / f"lesion_gradrand_e00_seed{s}_rp65_{L5}",
         "gradenergy": lambda s: E / "gradenergy/lanes" / f"lesion_gradenergy_e00_seed{s}_rp65_{L5}"}
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def images():
    cf = T.cifar_test()
    idx = np.concatenate([np.where(cf["y"] == c)[0][:20] for c in range(100)])
    return T.cifar_batch(cf["X"][idx]), torch.from_numpy(cf["y"][idx]).long()


def moments(net, X, y):
    net = net.to(DEV).eval()
    outs = {}
    hs = [getattr(net, l).register_forward_hook(lambda m, i, o, _l=l: outs.__setitem__(_l, o)) for l in STAGES]
    M = {l: 0.0 for l in STAGES}
    A = {l: [] for l in STAGES}
    for s in range(0, len(X), 250):
        xb, yb = X[s:s + 250].to(DEV), y[s:s + 250].to(DEV)
        outs.clear()
        loss = nn.functional.cross_entropy(net(xb), yb, reduction="sum")
        gs = torch.autograd.grad(loss, [outs[l] for l in STAGES])
        for l, g in zip(STAGES, gs):
            g = g.detach().float()
            g2 = g.permute(0, 2, 3, 1).reshape(-1, g.shape[1]) if g.dim() == 4 else g.reshape(g.shape[0], -1)
            M[l] = M[l] + (g2.T @ g2).cpu().double()
            o = outs[l].detach().float()
            A[l].append((o.mean((2, 3)) if o.dim() == 4 else o.reshape(o.shape[0], -1)).cpu())
    for h in hs:
        h.remove()
    return M, {l: torch.cat(v) for l, v in A.items()}


def share(M, Q):
    Q = Q.double()
    return float(torch.trace(Q.T @ M @ Q) / torch.trace(M))


def ranks(lane_dir):
    """{epoch: {stage: total rank at the end of that epoch}} from the lesion log."""
    f = lane_dir / "dev_snapshots/metrics/lesion_log.jsonl"
    if not f.exists():
        return {}
    evs = [e for e in map(json.loads, open(f)) if e["event"] == "reproject"]
    out = {}
    for ep in EPOCHS:
        last = [e for e in evs if e["weight_updates"] <= ep * 391]
        if last:
            out[ep] = {l: r["total_rank"] for l, r in last[-1]["report"].items()}
    return out


def main():
    X, y = images()
    ysc = torch.from_numpy(SUPER_OF[y.numpy()]).long()
    rows = []
    for arm, lf in LANES.items():
        for s in SEEDS:
            d = lf(s)
            rk = ranks(d)
            init = {l: torch.from_numpy(np.load(d / f"gradblock_basis_{l}.npy")).float()
                    for l in STAGES if (d / f"gradblock_basis_{l}.npy").exists()}
            for ep in EPOCHS:
                net = T.load_net({"dir": str(d), "arch": "ResNet-18", "group": "extended"}, str(ep))
                M, A = moments(net, X, y)
                for l in STAGES:
                    C = M[l].shape[0]
                    Q, _ = superclass_code_directions(A[l], ysc, q_max=19, evr_floor=1e-6)
                    r = {"arm": arm, "seed": s, "epoch": ep, "stage": l, "channels": C,
                         "code_share": share(M[l], Q), "code_rank": int(Q.shape[1])}
                    if l in init:
                        Qi, _ = torch.linalg.qr(init[l])
                        r["init_share"] = share(M[l], Qi)
                        r["init_ratio"] = r["init_share"] / (Qi.shape[1] / C)
                    if ep in rk:
                        r["rank"] = rk[ep][l]
                        r["rank_share"] = rk[ep][l] / C
                    rows.append(r)
                print(arm, s, ep, flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "dose_long.csv", index=False)

    # energy arm, in-run: achieved and the superclass arm's share in the same network (shadow), per re-estimation
    ev = []
    for s in SEEDS:
        for e in map(json.loads, open(LANES["gradenergy"](s) / "dev_snapshots/metrics/lesion_log.jsonl")):
            if e["event"] == "reproject":
                for l, r in e["report"].items():
                    ev.append({"seed": s, "epoch": e["epoch"], "stage": l, "achieved": r["achieved_grad_share"],
                               "shadow_target": r["target_grad_share"], "energy_rank": r["total_rank"]})
    EV = pd.DataFrame(ev)
    res = {}
    for win, eps in (("epochs_1_3", range(1, 4)), ("epochs_1_10", range(1, 11))):
        P = D[D.epoch.isin(list(eps))].groupby(["arm", "stage", "seed"]).mean(numeric_only=True)
        Q = EV[EV.epoch.isin(list(eps))].groupby(["stage", "seed"]).mean(numeric_only=True)
        out = {}
        for l in STAGES:
            ent = {}
            for arm in LANES:
                g = P.loc[(arm, l)]
                ent[arm] = {k: [round(float(g[k].mean()), 3), [round(float(x), 3) for x in g[k]]]
                            for k in ("code_share", "init_share", "init_ratio", "rank", "rank_share") if k in g}
            q = Q.loc[l]
            ent["gradenergy_inrun"] = {k: [round(float(q[k].mean()), 3), [round(float(x), 3) for x in q[k]]]
                                       for k in ("achieved", "shadow_target", "energy_rank")}
            out[l] = ent
        res[win] = out
    json.dump(res, open(OUT / "dose_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
