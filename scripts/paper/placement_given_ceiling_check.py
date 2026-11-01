#!/usr/bin/env python3
"""Placement given: diagnosis of the pooled-layer shortfall of the jointly matched block (seed 42).

At the pooled layer the learning signal is g = W_fc^T dL/dz, so its second moment M has rank <= 99 (standard loss) or
<= 80 (masked loss: dL/dz is zero outside the true superclass's five logits and sums to zero within them). The jointly
matched block is confined to the orthogonal complement of the current full superclass code S (<= 19 directions). Once
its rank exceeds rank(M) it already blocks ALL of the learning signal outside S, so its achieved share is capped at
1 - share(S), whatever the rank, window or re-estimation frequency. This script measures share(S) from the milestone
checkpoints (20 test images per class, eval mode, no hooks; the trainer uses 20 validation images per class, so values
differ slightly from lesion_log.jsonl) and prints it beside the logged target and achieved shares.
Also printed: share of M in the superclass directions above the 2% EVR floor (the floor the superclass arm adds with).
Read-only. Usage: python scripts/paper/placement_given_ceiling_check.py
"""
import json
import os
import sys
from pathlib import Path
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, RDT_DATA + "/checkouts/placement/packages/ml/src")
import transfer_lib as T  # noqa: E402
from lesion import superclass_code_directions  # noqa: E402
from curriculum_probe import SUPER_OF  # noqa: E402

E = RDT_DATA + "/extended/experiments/"
L5 = "Llayer1-layer2-layer3-layer4-avgpool"
LANES = {"masked": E + "placement_given/lanes/lesion_gradjoint_e00_seed42_rp65_" + L5,
         "standard": E + "gradjoint/lanes/lesion_gradjoint_e00_seed42_rp65_" + L5}


def last_log(ld, ep):
    rr = [json.loads(line) for line in open(ld + "/dev_snapshots/metrics/lesion_log.jsonl")]
    r = [d for d in rr if d.get("event") == "reproject" and d["epoch"] == ep][-1]["report"]["avgpool"]
    return r["shadow_rank"], r["target_grad_share"], r["achieved_grad_share"]


def main():
    cf = T.cifar_test()
    idx = np.concatenate([np.where(cf["y"] == c)[0][:20] for c in range(100)])
    x = T.cifar_batch(cf["X"][idx])
    y = torch.from_numpy(cf["y"][idx])
    ysc = torch.from_numpy(SUPER_OF[cf["y"][idx]])
    member = torch.from_numpy(SUPER_OF[:, None] == SUPER_OF[None, :])
    for reg, ld in LANES.items():
        for ep in (3, 5, 8, 10):
            net = T.load_net({"dir": ld, "arch": "ResNet-18"}, str(ep))
            with torch.no_grad():
                h = net.maxpool(net.relu(net.bn1(net.conv1(x))))
                f = torch.flatten(net.avgpool(net.layer4(net.layer3(net.layer2(net.layer1(h))))), 1)
            f = f.clone().requires_grad_(True)
            z = net.fc(f)
            if reg == "masked":
                z = z.masked_fill(~member[y], float("-inf"))
            g, = torch.autograd.grad(nn.functional.cross_entropy(z, y, reduction="sum"), [f])
            g = g.double()
            M = g.T @ g
            tot = float(torch.trace(M))
            share = {}
            for name, floor in (("S", 1e-6), ("S_floor2pct", 0.02)):
                S, _ = superclass_code_directions(f.detach().float(), ysc, q_max=19, evr_floor=floor)
                S = S.double()
                share[name] = (S.shape[1], float(torch.trace(S.T @ M @ S)) / tot)
            rank_m = int((torch.linalg.eigvalsh(M) > 1e-9 * torch.linalg.eigvalsh(M).max()).sum())
            r, t, a = last_log(ld, ep)
            print(f"{reg:8s} epoch {ep:2d}  rank(M)={rank_m:3d}  share(S, {share['S'][0]} dirs)={share['S'][1]:.3f}  "
                  f"ceiling 1-share(S)={1 - share['S'][1]:.3f}  share(S above 2% floor, {share['S_floor2pct'][0]} dirs)="
                  f"{share['S_floor2pct'][1]:.3f}  | log (last re-estimation): shadow rank {r}, target {t:.3f}, "
                  f"achieved {a:.3f}", flush=True)


if __name__ == "__main__":
    main()
