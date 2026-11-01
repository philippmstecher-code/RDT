#!/usr/bin/env python3
"""One read pass over the stored SAE codes of a lane -> compact local cache (lineage_lib.CACHE).

Per lane, census layer (avgpool, layer3, layer2, layer1) and epoch 0-15:
  M      class-mean |code| (100 classes x features; codes are class-major, 100 test images per class)
  cnt    number of the 10,000 test images on which each feature is active (|code| > 1e-6)
  st_e / tr_e / born_e / died_e   Phase-5 matching e-1 -> e (noise_null_lib.match;
         stable r >= 0.5, transformed 0.2 <= r < 0.5, death/birth r < 0.2)
avgpool only: decoder weights and bias per epoch (dec, bdec) and class-mean signed code (C).
Validation (written to analyses/extract_validation.json): census counts (alive, superclass-selective,
class-selective at the common thresholds) and matching counts (stable, born, died) against the stored
RDT_completion per-lane recount caches, which were themselves validated against the stored pipeline output.

Never writes to the data drive.
Usage: python scripts/paper/lineage/extract.py --arms sham,gradsc --workers 2
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from noise_null_lib import match  # noqa: E402
from lineage_lib import CACHE, EPOCHS, LAYERS, OUT, PER_LANE, SEEDS, common_thresholds, lane_dir, masks, sel_from_M  # noqa: E402


def one(job):
    arm, s = job
    od = CACHE / f"{arm}_seed{s}"
    done = od / "DONE.json"
    if done.exists():
        return json.load(open(done))
    od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = lane_dir(arm, s)
    t = common_thresholds(s)
    ref = json.load(open(PER_LANE / f"{arm}_seed{s}.json"))
    val = {"census_mismatch": [], "matching_mismatch": [], "n_census": 0, "n_matching": 0}
    for L in LAYERS:
        z = {}
        prev = None
        Ms, cnts, decs, bdecs, Cs = [], [], [], [], []
        for e in EPOCHS:
            H = np.load(d / "sae_analysis" / "activations" / str(e) / f"{L}.npy")
            A = np.abs(H)
            M = A.reshape(100, 100, -1).mean(1)
            cnt = (A > 1e-6).sum(0).astype(np.int32)
            Ms.append(M.astype(np.float32)); cnts.append(cnt)
            sel = sel_from_M(M)
            mk = masks(sel, cnt >= 100, t)
            mine = {"n_alive": int((cnt >= 100).sum()), "n_sc": int(mk["sc"].sum()), "n_cls": int(mk["cl"].sum())}
            stored = ref["census"][str(e)]["common"][L]
            val["n_census"] += 1
            if any(mine[k] != stored[k] for k in mine):
                val["census_mismatch"].append({"layer": L, "epoch": e, "mine": mine,
                                               "stored": {k: stored[k] for k in mine}})
            if L == "avgpool":
                w = torch.load(d / "sae_analysis" / "saes" / str(e) / f"{L}.pt", map_location="cpu", weights_only=True)
                decs.append(w["decoder.weight"].numpy().astype(np.float32))
                bdecs.append(w["decoder.bias"].numpy().astype(np.float32))
                Cs.append(H.reshape(100, 100, -1).mean(1).astype(np.float32))
            if prev is not None:
                md = match(prev, H)
                z[f"st_{e}"] = np.array(md["stable"], dtype=np.float64).reshape(-1, 3)
                z[f"tr_{e}"] = np.array(md.get("transformed", []), dtype=np.float64).reshape(-1, 3)
                z[f"born_{e}"] = np.array(sorted(set(md["born"])), dtype=np.int32)
                z[f"died_{e}"] = np.array(sorted(set(md["died"])), dtype=np.int32)
                mm = {"n_stable": md["n_stable"], "n_born": md["n_born"], "n_died": md["n_died"]}
                sm = ref["matching"][f"{e - 1}->{e}"][L]
                val["n_matching"] += 1
                if any(mm[k] != sm[k] for k in mm):
                    val["matching_mismatch"].append({"layer": L, "transition": f"{e - 1}->{e}", "mine": mm, "stored": sm})
            prev = H
        z["M"] = np.stack(Ms); z["cnt"] = np.stack(cnts)
        if L == "avgpool":
            z["dec"] = np.stack(decs); z["bdec"] = np.stack(bdecs); z["C"] = np.stack(Cs)
        np.savez(od / f"{L}.npz", **z)
    res = {"lane": f"{arm}_seed{s}", "seconds": round(time.time() - t0), **val}
    json.dump(res, open(done, "w"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="sham,gradsc,gradjoint,gradrg")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    jobs = [(arm, int(s)) for arm in a.arms.split(",") for s in a.seeds.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    vf = OUT / "extract_validation.json"
    allv = json.load(open(vf)) if vf.exists() else {}
    with mp.get_context("spawn").Pool(min(2, a.workers)) as pool:
        for r in pool.imap_unordered(one, jobs):
            allv[r["lane"]] = r
            print(r["lane"], r["seconds"], "s; census mismatches", len(r["census_mismatch"]), "/", r["n_census"],
                  "; matching mismatches", len(r["matching_mismatch"]), "/", r["n_matching"], flush=True)
            json.dump(allv, open(vf, "w"), indent=1)


if __name__ == "__main__":
    main()
