#!/usr/bin/env python3
"""Checkpoint-density test on clean per-epoch runs (sham_perepoch/sham_e00_seed{7,42,137}: standard training
with a rank-0 projector, epochs 0-15; SAEs and RDT from run_sham_perepoch_sae.py).

Checkpoints are one epoch apart (391 iterations). The dense per-epoch schedule (0..15) is compared with the same
run subsampled to a 5-epoch grid (0, 5, 10, 15):
  * chain-stable through all five per-epoch Hungarian steps vs directly stable k->k+5,
  * flicker (death at step 1 but directly stable) and gaps (any intermediate death),
  * event counts, dense (summed over five steps) vs coarse (direct), feature-level and sample-weighted, for
    As/De/Tg/Ab/Di,
  * per-epoch skip-one (k->k+2), as in the core runs.

Usage: python scripts/paper/checkpoint_density.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matching_lib import (EPS_ACT, LESION_EXP, OUT, Labeler, Lane, chain_compare, feature_counts,  # noqa: E402
                    outcome_from_pipeline, pipeline_match)

LANES = {f"sham_e00_seed{s}": "clean (sham, rank-0 projector)" for s in (7, 42, 137)}
CLEAN_EXP = Path(RDT_DATA + "/extended/experiments/sham_perepoch")
DENSE = [str(e) for e in range(16)]     # epochs 0..15
WINDOWS = [(0, 5), (5, 10), (10, 15)]


def run(lane_name: str):
    lane = Lane(CLEAN_EXP / "lanes" / lane_name)
    lab = Labeler(lane)
    snaps = json.load(open(lane.dir / "dev_snapshots" / "metrics" / "devtrain_snapshots.json"))
    epochs = {str(s["milestone_index"]): (s["epoch"] if s["milestone_index"] > 0 else 0, s["weight_updates"]) for s in snaps}
    for e in range(16):
        assert epochs[str(e)][1] == 391 * e, epochs[str(e)]
    rows_step, rows_win, rows_skip = [], [], []
    t0 = time.time()
    need = list(DENSE)
    for layer in lane.layers:
        H = {l: lane.load(l, layer) for l in need}
        act = {l: np.abs(H[l]) > EPS_ACT for l in need}
        sel = {l: lane.sel(l, layer) for l in need}
        step = {}
        for la, lb in list(zip(DENSE[:-1], DENSE[1:])):
            pair = f"{la}->{lb}"
            md = pipeline_match(H[la], H[lb])
            o = outcome_from_pipeline(md)
            step[pair] = (md, o)
            st = lane.stored_counts(pair, layer)
            labs = lab.labels(o, sel[la], sel[lb])
            w = lab.weighted(o, labs, act[la], act[lb])
            r = {"lane": lane_name, "layer": layer, "pair": pair, "n_alive_a": int(md["alive_a"].sum()),
                 "n_alive_b": int(md["alive_b"].sum()),
                 **{k: int(md[k]) for k in ("n_stable", "n_born", "n_died", "n_transformed")},
                 "counts_match": bool(st) and all(int(st[k]) == int(md[k]) for k in ("n_stable", "n_born", "n_died", "n_transformed")),
                 **feature_counts(labs), **{f"w_{k}": v for k, v in w.items()}}
            rows_step.append(r)
        # windows: dense chain vs direct
        for a, b in WINDOWS:
            la, lb = str(a), str(b)
            md = pipeline_match(H[la], H[lb])
            direct = outcome_from_pipeline(md)
            steps = [step[f"{e}->{e+1}"][1] for e in range(a, b)]
            prev = step[f"{a-1}->{a}"][1] if a > 0 else None
            ia = np.where(md["alive_a"])[0]
            res = chain_compare(steps, direct, ia, prev=prev)
            # end-feature lineage: persistent (chain endpoint) vs new-in-window
            ends = set()
            for i in ia:
                cur = int(i); ok = True
                for o in steps:
                    if cur in o["stable"]:
                        cur = o["stable"][cur][0]
                    else:
                        ok = False; break
                if ok:
                    ends.add(cur)
            ib = set(int(j) for j in np.where(md["alive_b"])[0])
            d_st_b = {v[0] for v in direct["stable"].values()}
            d_tr_b = {v[0] for v in direct["transformed"].values()}
            d_born = direct["born"]
            res.update({"end_persistent": len(ends), "end_new": len(ib - ends),
                        "end_persistent_coarse_stable": len(ends & d_st_b),
                        "end_persistent_coarse_transformed": len(ends & d_tr_b),
                        "end_persistent_coarse_born": len(ends & d_born),
                        "end_new_coarse_stable": len((ib - ends) & d_st_b),
                        "end_new_coarse_born": len((ib - ends) & d_born)})
            labs = lab.labels(direct, sel[la], sel[lb])
            w = lab.weighted(direct, labs, act[la], act[lb])
            res.update({f"coarse_{k}": v for k, v in feature_counts(labs).items()})
            res.update({f"coarse_w_{k}": v for k, v in w.items()})
            res.update({f"coarse_{k}": int(md[k]) for k in ("n_stable", "n_born", "n_died", "n_transformed")})
            dense = [r for r in rows_step if r["layer"] == layer and r["pair"] in {f"{e}->{e+1}" for e in range(a, b)}]
            for k in ("n_born", "n_died", "n_stable", "n_transformed", "f_tg", "f_ab", "f_di", "f_tg_born", "f_ab_born",
                      "f_di_born", "w_tg_h", "w_ab_h", "w_di_h", "w_as_h", "w_de_h"):
                res[f"dense_sum_{k}"] = int(sum(r[k] for r in dense))
            # dense births that survive (via stable links) to the window end
            surv = 0
            for e in range(a, b):
                for j in steps[e - a]["born"]:
                    cur = j; ok = True
                    for o in steps[e - a + 1:]:
                        if cur in o["stable"]:
                            cur = o["stable"][cur][0]
                        else:
                            ok = False; break
                    surv += ok
            res["dense_births_surviving_to_end"] = surv
            res.update({"lane": lane_name, "layer": layer, "window": f"{la}->{lb}"})
            rows_win.append(res)
        # per-epoch skip-one
        for k in range(len(DENSE) - 2):
            la, lm, lb = DENSE[k], DENSE[k + 1], DENSE[k + 2]
            md = pipeline_match(H[la], H[lb])
            direct = outcome_from_pipeline(md)
            ia = np.where(md["alive_a"])[0]
            prev = step[f"{DENSE[k-1]}->{la}"][1] if k > 0 else None
            res = chain_compare([step[f"{la}->{lm}"][1], step[f"{lm}->{lb}"][1]], direct, ia, prev=prev)
            born1 = step[f"{la}->{lm}"][1]["born"]; died2 = step[f"{lm}->{lb}"][1]["died"]
            res.update({"lane": lane_name, "layer": layer, "window": f"{la}->{lb}", "k": k,
                        "n_born_step1": len(born1), "n_blip_births": len(born1 & died2)})
            rows_skip.append(res)
        del H, act
        print(f"  [{lane_name[:22]}] {layer} done ({time.time()-t0:.0f}s)", flush=True)
    ok = sum(r["counts_match"] for r in rows_step)
    print(f"[{lane_name}] stored-count match {ok}/{len(rows_step)}", flush=True)
    return {"steps": rows_step, "windows": rows_win, "skip_one": rows_skip,
            "epochs": {k: v[0] for k, v in epochs.items()}, "description": LANES[lane_name]}


def main():
    out = {}
    for ln in LANES:
        out[ln] = run(ln)
    (OUT / "per_lane").mkdir(parents=True, exist_ok=True)
    with open(OUT / "per_lane" / "clean_density.json", "w") as f:
        json.dump(out, f, default=float)
    print("written", OUT / "per_lane" / "clean_density.json")


if __name__ == "__main__":
    main()
