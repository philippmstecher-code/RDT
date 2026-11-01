#!/usr/bin/env python3
"""Recompute Phases 4-6 per layer for one lane from stored SAE codes, validate against the stored
sae_results.json, and compute noise-null pseudo-transitions.

Writes analysis_outputs/paper/NOISE_null/per_lane/<lane_id>.json with per-layer transition counts
(tg_h, ab_h, di_h, as_h, de_h, stable) and matching summaries, validation against the stored process_intensity,
feature matching and selectivity, the ResNet state-matched null (layer4(k) <-> avgpool(k), both directions) and the
selective-feature census per checkpoint and layer.

Usage: noise_null_lane.py <experiment_dir> <lane_id> [--dataset cifar100] [--layers all|dup] [--tag TAG]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noise_null_lib import (  # noqa: E402
    COUNT_KEYS, OUT, PROCS, Lane, census, classify_counts, match, selectivity, transition_counts,
)

DUP = ("layer4", "avgpool")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir")
    ap.add_argument("lane_id")
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--layers", default="all", choices=["all", "dup"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    lane_dir = Path(args.exp_dir) / "lanes" / args.lane_id
    lane = Lane(lane_dir, dataset_id=args.dataset)
    labels, layers_all = lane.labels, lane.layers
    is_resnet = all(l in layers_all for l in DUP)
    layers = layers_all if args.layers == "all" else [l for l in layers_all if l in DUP]
    stored_fm = lane.results.get("feature_matching", {})
    stored_pi = {p["transition"]: p for p in lane.results.get("process_intensity", [])}

    out = {
        "lane_id": args.lane_id, "exp_dir": str(args.exp_dir), "dataset": args.dataset,
        "labels": labels, "layers_all": layers_all, "layers_run": layers, "is_resnet": is_resnet,
        "thresholds": {k: lane.thresholds[k] for k in
                       ("ssi_abstraction_thresh", "sai_task_general_thresh", "csi_differentiation_thresh")},
        "null_stats": {k: lane.ati.get(k) for k in ("ssi_null_mean", "ssi_null_std", "csi_null_mean",
                                                     "csi_null_std", "sai_null_mean", "sai_null_std")},
        "real": {}, "null_dup": {}, "census": {}, "validation": {"selectivity_maxabs": 0.0,
                                                               "fm_mismatch": [], "pi": {}},
    }
    t0 = time.time()
    prev = None  # (label, {layer: (H, sel)})
    for lab in labels:
        cur = {}
        for L in layers:
            H = lane.load_codes(lab, L)
            sel = selectivity(lane, H)
            st = lane.results.get("selectivity", {}).get(lab, {}).get(L)
            if st is not None:
                for k in ("feature_ssi", "feature_csi", "feature_sai"):
                    d = float(np.max(np.abs(np.asarray(st[k], dtype=np.float64) -
                                            np.asarray(sel[k], dtype=np.float64))))
                    out["validation"]["selectivity_maxabs"] = max(out["validation"]["selectivity_maxabs"], d)
            out["census"].setdefault(lab, {})[L] = census(lane, sel, H)
            cur[L] = (H, sel)
        # state-matched duplicate null (ResNet)
        if is_resnet:
            (H4, s4), (Hp, sp) = cur["layer4"], cur["avgpool"]
            c1, m1 = transition_counts(lane, H4, Hp, s4, sp)
            c2, m2 = transition_counts(lane, Hp, H4, sp, s4)
            out["null_dup"][lab] = {"layer4->avgpool": {**c1, **m1}, "avgpool->layer4": {**c2, **m2}}
        if prev is not None:
            plab, pdat = prev
            key = f"{plab}->{lab}"
            out["real"][key] = {}
            for L in layers:
                (Ha, sa_), (Hb, sb_) = pdat[L], cur[L]
                c, m = transition_counts(lane, Ha, Hb, sa_, sb_)
                out["real"][key][L] = {**c, **m}
                sfm = stored_fm.get(key, {}).get(L)
                if sfm is not None:
                    for k in ("n_stable", "n_born", "n_died", "n_transformed"):
                        if sfm.get(k) != m[k]:
                            out["validation"]["fm_mismatch"].append([key, L, k, sfm.get(k), m[k]])
            if args.layers == "all" and key in stored_pi:
                rec = {p: sum(out["real"][key][L][p] for L in layers) for p in PROCS}
                sto = {p: stored_pi[key][p] for p in PROCS}
                out["validation"]["pi"][key] = {"stored": sto, "recomputed": rec,
                                                "exact": all(rec[p] == sto[p] for p in PROCS)}
        prev = (lab, cur)
        print(f"  {args.lane_id[:8]} ckpt {lab} done ({time.time() - t0:.0f}s)", flush=True)

    v = out["validation"]
    v["n_pi_exact"] = sum(1 for x in v["pi"].values() if x["exact"])
    v["n_pi"] = len(v["pi"])
    d = OUT / "per_lane"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{args.lane_id}{args.tag}.json", "w") as f:
        json.dump(out, f)
    print(f"{args.lane_id}: pi exact {v['n_pi_exact']}/{v['n_pi']}, fm mismatches {len(v['fm_mismatch'])}, "
          f"sel maxabs {v['selectivity_maxabs']:.3g}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
