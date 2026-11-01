#!/usr/bin/env python3
"""ResNet-18 routing with the same ablation as the transformers (mean-preserving projection onto the complement of
the decoder span), 5 clean seeds, final checkpoint, for a like-for-like comparison across architectures.
Writes analysis_outputs/paper/ROUTING/routing_resnet_projection.json."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import routing_transformers as R  # noqa: E402
T = R.T
res = {}
for s in T.SEEDS:
    d = T.RUNS_EXTENDED / f"static_noise_s{s}" / "lanes" / "standard_resnet18"
    lab = {c["epoch"]: c["label"] for c in T.real_ckpts(T.checkpoint_info(d, T._snap_labels(d)))}[30]
    r = R.analyse({"dir": str(d), "arch": "ResNet-18", "group": "extended", "label": lab}, lab)
    res[s] = {k: r[k] for k in ("S", "S_ctrl", "n_ab", "n_di", "base_sc_acc")}
    print(s, round(r["S"], 3), round(r["S_ctrl"], 3), r["n_ab"], flush=True)
json.dump(res, open(R.A.OUT / "routing_resnet_projection.json", "w"), indent=1)
