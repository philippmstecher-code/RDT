#!/usr/bin/env python3
"""Class hierarchy, step 1: dictionary-free geometry + probes per lane x checkpoint x layer, open seeds 7/42/137
only. Writes one CSV per lane into
analysis_outputs/paper/DICTFREE/hierarchy/cache/geo_<lane>.csv.

Usage: hierarchy_geometry.py --lanes sc_true_K20 sc_rand_K20 ... --seeds 7 42 137 [--ms 0 4]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from hierarchy_lib import (LAYERS, OUT, SEEDS, Geo, cv_probe, heldout_class_transfer, imposed_vec, lane_dir,
                       load_acts, null_partitions, snapshots, taskirr_vec, true_vec, within_group_probe)

PROBE_LAYERS = ("layer3", "avgpool")
GEO_LAYERS = [L for L in LAYERS if L != "layer4"]  # layer4 hook is post-pool: identical to avgpool (checked)
N_NULL20, N_NULL5 = 40, 20


def run(kind: str, seed: int, only_ms=None) -> pd.DataFrame:
    d = lane_dir(kind, seed)
    parts = {"true": true_vec(), "imposed": imposed_vec(seed), "tlabel": taskirr_vec()}
    nul20 = null_partitions(N_NULL20, parts["true"], 50000)
    nul5 = null_partitions(N_NULL5, parts["tlabel"], 60000)
    rows = []
    for sn in snapshots(d):
        if only_ms is not None and sn["ms"] not in only_ms:
            continue
        t0 = time.time()
        X, y = load_acts(d, sn["ms"], GEO_LAYERS)
        for L in GEO_LAYERS:
            g = Geo(X[L], y)
            r = {"lane": kind, "seed": seed, "ms": sn["ms"], "epoch": sn["epoch"], "val_acc": sn["val_acc"],
                 "layer": L, **g.class_fisher()}
            for pn, pv in parts.items():
                for k, v in g.part(pv).items():
                    r[f"{pn}_{k}"] = v
            for tag, nl in (("null20", nul20), ("null5", nul5)):
                vals = pd.DataFrame([g.part(p) for p in nl])
                for k in vals.columns:
                    r[f"{tag}_{k}_mean"] = float(vals[k].mean())
                    r[f"{tag}_{k}_sd"] = float(vals[k].std())
            if L in PROBE_LAYERS:
                XL = X[L]
                r["p_fine"] = cv_probe(XL, y)
                for pn in ("true", "imposed"):
                    pv = parts[pn]
                    r[f"p_{pn}20"] = cv_probe(XL, pv[y])
                    r[f"p_{pn}_within"] = within_group_probe(XL, y, pv)
                    r[f"p_{pn}_hoct"] = heldout_class_transfer(XL, y, pv)
                r["p_null_hoct"] = float(np.mean([heldout_class_transfer(XL, y, p) for p in nul20[:2]]))
                r["p_null_within"] = float(np.mean([within_group_probe(XL, y, p) for p in nul20[:2]]))
                r["p_null20"] = float(np.mean([cv_probe(XL, p[y]) for p in nul20[:2]]))
                r["p_tlabel5"] = cv_probe(XL, parts["tlabel"][y])
                r["p_tlabel_hoct"] = heldout_class_transfer(XL, y, parts["tlabel"])
            rows.append(r)
        print(f"{kind} s{seed} ms{sn['ms']} ep{sn['epoch']} {time.time() - t0:.1f}s", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--ms", nargs="*", type=int, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    for kind in a.lanes:
        for s in a.seeds:
            if not lane_dir(kind, s).exists():
                print(f"missing {kind} s{s}", flush=True)
                continue
            df = run(kind, s, a.ms)
            df.to_csv(OUT / "cache" / f"geo_{kind}_s{s}{a.tag}.csv", index=False)


if __name__ == "__main__":
    main()
