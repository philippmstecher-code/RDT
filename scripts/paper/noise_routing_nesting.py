#!/usr/bin/env python3
"""Fixed-label noise (Extended Data Fig. 8): routing of the learning signal and nesting of class-selective features in the
clean, within- and between-superclass noise runs.

Arms (static_noise_s{S}/lanes): clean (standard_resnet18), within (within_sc_resnet18_throughout), between
(between_sc_resnet18_throughout); seeds 7, 42, 137, 256, 314; checkpoints epochs 0 (milestone_0), 5, 10, ..., 30.
A  Routing of the learning signal (signal_routing.measure): per checkpoint,
   on the 10,000 test images with their TRUE labels and the network's own classifier, the share of images predicted in
   the wrong superclass (misrouted), the share of the learning signal (squared gradient of the loss at the pooled
   layer) they supply, and the share of each group's signal inside the current rank-19 superclass code.
B  Nesting of class-selective features (the pooled-layer statistic of Extended Data Fig. 6a, nesting_partition_null.ratio_stat):
   pooled-layer SAE, checkpoints epochs 5-30; per class-selective feature, the squared projection of its
   unit decoder direction onto the span of the superclass-selective decoders of its own superclass, over the mean for
   equally sized sets of other superclasses; per seed the median over features and checkpoints. Thresholds = one
   common set per seed (the clean run's stored analysis_thresholds; as noise_rdt_analysis.py), exclusive rule
   (class-selective: CSI > t, alive, SAI <= t; superclass-selective: SSI > t, CSI <= t, alive, SAI <= t).
Contrasts paired over seeds (t-based 95% CI): between - within, between - clean, within - clean.
Writes analysis_outputs/paper/RDT_noise/routing_nesting_long.csv, nesting_per_lane.csv and
routing_nesting_summary.json.
CPU only, read-only on data.
Usage: python scripts/paper/noise_routing_nesting.py
"""
import json
import os
import sys
from pathlib import Path
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coarse_overlap import milestones  # noqa: E402
from nesting_partition_null import groups_true, ratio_stat  # noqa: E402
from dependency_lib import selectivity  # noqa: E402
from signal_routing import measure  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis_outputs/paper/RDT_noise"
E = Path(RDT_DATA + "/extended/experiments")
LANES = {"clean": "standard_resnet18", "within": "within_sc_resnet18_throughout",
         "between": "between_sc_resnet18_throughout"}
SEEDS = [7, 42, 137, 256, 314]
CONTRASTS = [("between", "within"), ("between", "clean"), ("within", "clean")]


def lane_dir(arm, s):
    return E / f"static_noise_s{s}" / "lanes" / LANES[arm]


def thresholds(s):
    r = json.load(open(lane_dir("clean", s) / "sae_analysis" / "sae_results.json"))["metadata"]["analysis_thresholds"]
    r = r if isinstance(r, dict) else json.loads(r)
    return float(r["ssi_abstraction_thresh"]), float(r["csi_differentiation_thresh"]), float(r["sai_task_general_thresh"])


def nesting(arm, s, rng):
    ssi_t, csi_t, sai_t = thresholds(s)
    d = lane_dir(arm, s) / "sae_analysis"
    obs, n_cls, n_sc = [], [], []
    for m in range(1, 7):
        H = np.load(d / "activations" / str(m) / "avgpool.npy")
        sel = selectivity(H)
        D = torch.nn.functional.normalize(
            torch.load(d / "saes" / str(m) / "avgpool.pt", map_location="cpu", weights_only=True)["decoder.weight"]
            .double(), dim=0).numpy()
        alive = (H != 0).mean(0) >= 0.01
        sai = sel["sai"] > sai_t
        csi_idx = np.where((sel["csi"] > csi_t) & alive & ~sai)[0]
        ssi_idx = np.where((sel["ssi"] > ssi_t) & (sel["csi"] <= csi_t) & alive & ~sai)[0]
        obs += ratio_stat(D, sel["M"], csi_idx, ssi_idx, groups_true(), rng)
        n_cls.append(len(csi_idx))
        n_sc.append(len(ssi_idx))
    return {"nesting_median_ratio": float(np.median(obs)) if obs else np.nan, "n_ratio_features": len(obs),
            "n_cls_mean": float(np.mean(n_cls)), "n_sc_mean": float(np.mean(n_sc))}


def tci(d):
    d = np.asarray(d, float)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return [float(d.mean()), float(d.mean() - h), float(d.mean() + h)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, nest = [], []
    rng = np.random.default_rng(0)
    for arm in LANES:
        for s in SEEDS:
            d = lane_dir(arm, s)
            ms = milestones(d)
            ms[0] = d / "dev_snapshots" / "milestone_0"
            for e in sorted(ms):
                rows.append({"arm": arm, "seed": s, "epoch": e, **measure(ms[e], rng)})
            nest.append({"arm": arm, "seed": s, **nesting(arm, s, np.random.default_rng(s))})
            print(arm, s, {k: round(v, 3) for k, v in rows[-1].items() if k.startswith(("image_share_mis",
                  "signal_share_mis", "code_share_mis"))}, nest[-1], flush=True)
    D = pd.DataFrame(rows)
    N = pd.DataFrame(nest)
    D.to_csv(OUT / "routing_nesting_long.csv", index=False)
    N.to_csv(OUT / "nesting_per_lane.csv", index=False)
    res = {}
    for win, eps in (("epochs 5-10", [5, 10]), ("epochs 5-30", [5, 10, 15, 20, 25, 30]), ("epoch 30", [30])):
        M = D[D.epoch.isin(eps)].groupby(["arm", "seed"]).mean(numeric_only=True)
        for a, b in CONTRASTS:
            for k in ("image_share_misrouted", "signal_share_misrouted", "code_share_misrouted", "code_share_routed"):
                dl = 100 * (M.loc[a][k] - M.loc[b][k]).loc[SEEDS].values
                res[f"{win}|{a} - {b}|{k}"] = {"mean_ci_pp": tci(dl), "n_pos": int((dl > 0).sum())}
    Nn = N.set_index(["arm", "seed"])
    for a, b in CONTRASTS:
        for k in ("nesting_median_ratio", "n_cls_mean", "n_sc_mean"):
            dl = (Nn.loc[a][k] - Nn.loc[b][k]).loc[SEEDS].values
            res[f"nesting|{a} - {b}|{k}"] = {"mean_ci": tci(dl), "n_pos": int((dl > 0).sum())}
    res["nesting_by_arm"] = N.groupby("arm")[["nesting_median_ratio", "n_cls_mean", "n_sc_mean"]].mean().round(3).to_dict()
    json.dump(res, open(OUT / "routing_nesting_summary.json", "w"), indent=1)
    for k, v in res.items():
        if "between - within" in k:
            print(k, [round(x, 3) for x in v.get("mean_ci_pp", v.get("mean_ci"))], "pos", v["n_pos"])
    print(res["nesting_by_arm"])


if __name__ == "__main__":
    main()
