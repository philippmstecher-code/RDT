#!/usr/bin/env python3
"""Re-run Phase 5 matching on the saved SAE activations of the 9 core CIFAR-100 lanes and cache what the
transformed-feature analysis needs.

Per (lane, layer, transition) the cache stores stable and transformed pairs (idx_a, idx_b, r) with the number of
samples in which both features are active (n_co, as counted by Phase 6), died and born indices, alive masks,
per-feature active-sample counts, and the decoder-column cosine of each pair with a random alive-pair baseline.
Matching uses the pipeline's `_phase5_feature_matching`; counts are validated against sae_results.json and any
mismatch is recorded in the cache. Writes analysis_outputs/paper/TRANSFORMED/cache/<lane>.pkl.

Usage: python scripts/paper/transformed_matching_cache.py [--lanes 7db28a9a,...]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "packages" / "ml" / "src"))

from saeanalysis import (  # noqa: E402
    _apply_thresholds, _phase5_feature_matching, _safe_layer_filename,
)

DATA = Path(RDT_DATA + "/core")
CORE_EXP = DATA / "experiments" / "abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb"
OUT = REPO / "analysis_outputs" / "paper" / "TRANSFORMED" / "cache"

CORE_LANES = [
    ("7db28a9a-f436-4d8a-a6b4-37e69fbf54eb", "ResNet-18", 42),
    ("a0bcbcba-94a9-4009-bcb6-d725b47588ee", "ResNet-18", 137),
    ("935c13ae-a026-4eae-a2b2-0306879b2e8c", "ResNet-18", 256),
    ("10d31151-68a2-4491-836d-3ead8e00a7ad", "ViT-Small", 42),
    ("909d7f68-55e1-4cd9-8c7b-68d5793798d3", "ViT-Small", 137),
    ("abad8975-de55-4e7e-b95d-05067f0eed90", "ViT-Small", 256),
    ("4bebc329-c26a-4eb4-b18c-7586e513f49e", "CCT-7", 42),
    ("edf8275e-c99f-4436-a13d-4ed50f072a66", "CCT-7", 137),
    ("445dd4d4-770a-4f33-b82b-c0bc7baed9a7", "CCT-7", 256),
]
EPS = 1e-6
RNG = np.random.default_rng(0)


def decoder_unit_columns(sae_dir: Path, label: str, layer: str):
    p = sae_dir / "saes" / label / f"{_safe_layer_filename(layer)}.pt"
    if not p.exists():
        return None
    sd = torch.load(p, map_location="cpu", weights_only=True)
    W = sd["decoder.weight"].float()  # (d_in, d_hidden)
    W = torch.nn.functional.normalize(W, dim=0)
    return W.numpy()


def pair_cos(Wa, Wb, pairs):
    if Wa is None or Wb is None or not pairs:
        return np.array([], dtype=np.float32)
    ia = np.array([p[0] for p in pairs]); ib = np.array([p[1] for p in pairs])
    return np.einsum("di,di->i", Wa[:, ia], Wb[:, ib]).astype(np.float32)


def process_lane(lane_id: str, arch: str, seed: int):
    sae_dir = CORE_EXP / "lanes" / lane_id / "sae_analysis"
    res = json.loads((sae_dir / "sae_results.json").read_text())
    meta = res["metadata"]
    labels, layers = meta["checkpoint_labels"], meta["layers"]
    _apply_thresholds(meta["analysis_thresholds"])
    out = {"lane_id": lane_id, "arch": arch, "seed": seed, "labels": labels,
           "layers": layers, "layer_data": {}, "mismatches": [], "missing": []}
    for layer in layers:
        t0 = time.time()
        act = {}
        for lab in labels:
            p = sae_dir / "activations" / lab / f"{_safe_layer_filename(layer)}.npy"
            if not p.exists():
                out["missing"].append(str(p))
                act = None
                break
            act[lab] = {layer: np.load(p)}
        if act is None:
            print(f"  {layer}: activations missing, skipped", flush=True)
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            fm = _phase5_feature_matching(act, {}, {}, labels, [layer], None)
        W = {lab: decoder_unit_columns(sae_dir, lab, layer) for lab in labels}
        per_t = {}
        for a, b in zip(labels[:-1], labels[1:]):
            t = f"{a}->{b}"
            m = fm[t][layer]
            stored = res["feature_matching"].get(t, {}).get(layer, {})
            for k in ("n_stable", "n_born", "n_died", "n_transformed"):
                if stored.get(k) != m[k]:
                    out["mismatches"].append((t, layer, k, stored.get(k), m[k]))
            Ha, Hb = act[a][layer], act[b][layer]
            act_a = np.abs(Ha) > EPS
            act_b = np.abs(Hb) > EPS

            def with_co(pairs):
                return [(int(i), int(j), float(r), int((act_a[:, i] & act_b[:, j]).sum()))
                        for i, j, r in pairs]

            alive_a = np.where(m["alive_a"])[0]; alive_b = np.where(m["alive_b"])[0]
            rnd = []
            if len(alive_a) and len(alive_b) and W[a] is not None and W[b] is not None:
                ia = RNG.choice(alive_a, 2000); ib = RNG.choice(alive_b, 2000)
                rnd = np.einsum("di,di->i", W[a][:, ia], W[b][:, ib]).astype(np.float32)
            per_t[t] = {
                "stable": with_co(m["stable"]),
                "transformed": with_co(m["transformed"]),
                "died": [int(x) for x in m["died"]],
                "born": [int(x) for x in m["born"]],
                "alive_a": m["alive_a"].astype(bool),
                "alive_b": m["alive_b"].astype(bool),
                "n_a": act_a.sum(axis=0).astype(np.int32),
                "n_b": act_b.sum(axis=0).astype(np.int32),
                "cos_stable": pair_cos(W[a], W[b], m["stable"]),
                "cos_transformed": pair_cos(W[a], W[b], m["transformed"]),
                "cos_random_alive": np.asarray(rnd, dtype=np.float32),
            }
        out["layer_data"][layer] = per_t
        del act, fm, W
        print(f"  {layer}: {time.time() - t0:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", default="", help="comma-separated lane-id prefixes")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sel = [s for s in args.lanes.split(",") if s]
    for lane_id, arch, seed in CORE_LANES:
        if sel and not any(lane_id.startswith(s) for s in sel):
            continue
        dst = OUT / f"{lane_id}.pkl"
        if dst.exists():
            print(f"== {arch} seed {seed}: cached, skip"); continue
        print(f"== {arch} seed {seed} ({lane_id})", flush=True)
        d = process_lane(lane_id, arch, seed)
        print(f"   mismatches vs sae_results.json: {len(d['mismatches'])}", flush=True)
        with open(dst, "wb") as f:
            pickle.dump(d, f)


if __name__ == "__main__":
    main()
