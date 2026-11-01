#!/usr/bin/env python3
"""Counts of RDT Phase-6 process events and the selective-feature
census for the curriculum sweep (7 lanes) and the 30-epoch standard ResNet-18 runs (5 seeds)
under (i) each run's own adaptive thresholds (validated against stored process_intensity),
(ii) common threshold sets applied identically to every run, plus the layer4<->avgpool
state-matched re-decomposition null at every checkpoint under every threshold set.

Phase 4 selectivity is taken from the stored sae_results.json (identical to recomputation); Phase 5
matching is recomputed from stored SAE codes (noise_null_lib.match), Phase 6 via noise_null_lib.classify_counts
(exact vectorised re-implementation of the pipeline).

Writes analysis_outputs/paper/CURRICULUM_sweep/per_lane/<lane_id>.json. Reads the stored run data only.

Usage: python scripts/paper/curriculum_recount.py [--workers 3] [--only LANE_ID_PREFIX]
"""
from __future__ import annotations

import argparse
import ast
import json
import sqlite3
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noise_null_lib import EXP, PROCS, REPO, Lane, classify_counts, match  # noqa: E402

OUT = REPO / "analysis_outputs" / "paper" / "CURRICULUM_sweep"
CUR_EXP = EXP / "7885dc97-1062-4b2f-9999-61d63b93d097"
STD30_EXP = EXP / "0c28799c-ec9d-4e45-ac30-302dff810a8d"
STD50_ID = "115664cf-b4de-4078-b330-28a98cd1fd03"   # ResNet18-constlr-standard-seed42
E15_ID = "377bbdec-e4e7-4883-8758-63ec7938056d"     # ResNet18-constlr-switch_e15-seed42 (CSI cap 0.7)
DUP = ("layer4", "avgpool")
TKEYS = ("ssi_abstraction_thresh", "sai_task_general_thresh", "csi_differentiation_thresh")


def lane_names() -> Dict[str, str]:
    c = sqlite3.connect(f"file:{EXP.parent / 'rcx.db'}?immutable=1", uri=True)
    return {r[0]: r[1] for r in c.execute("select id, name from lanes")}


def lanes_to_run() -> Dict[str, Path]:
    names = lane_names()
    out = {}
    for d in sorted((CUR_EXP / "lanes").iterdir()):
        out[d.name] = d
    for d in sorted((STD30_EXP / "lanes").iterdir()):
        if "noise-standard" in names.get(d.name, ""):
            out[d.name] = d
    return out


def _lit(x):
    return ast.literal_eval(x) if isinstance(x, str) else x


def load_lane(d: Path) -> Lane:
    lane = Lane(d)
    lane.ati = _lit(lane.results["metadata"].get("adaptive_thresholds")) or {}
    lane.thresholds = _lit(lane.results["metadata"].get("analysis_thresholds"))
    return lane


def thr_set(lane: Lane) -> Dict[str, Any]:
    return {"thresholds": {k: float(lane.thresholds[k]) for k in TKEYS},
            "ati": {k: float(v) for k, v in lane.ati.items() if isinstance(v, (int, float))}}


def census_t(sel: Dict[str, Any], H: np.ndarray, t: Dict[str, float]) -> Dict[str, float]:
    n = H.shape[0]
    cnt = (np.abs(H) > 1e-6).sum(axis=0)
    alive = cnt >= max(1, n * 0.01)
    ssi = np.asarray(sel["feature_ssi"], dtype=np.float64)
    csi = np.asarray(sel["feature_csi"], dtype=np.float64)
    sai = np.asarray(sel["feature_sai"], dtype=np.float64)
    tg = alive & (sai > t["sai_task_general_thresh"])
    ab = alive & (ssi > t["ssi_abstraction_thresh"]) & ~tg
    di = alive & (csi > t["csi_differentiation_thresh"])
    w = cnt.astype(np.float64)
    return {"n_alive": int(alive.sum()), "n_tg": int(tg.sum()), "n_ab": int(ab.sum()), "n_di": int(di.sum()),
            "w_alive": float(w[alive].sum()), "w_tg": float(w[tg].sum()), "w_ab": float(w[ab].sum()),
            "w_di": float(w[di].sum()),
            "csi_mean_alive": float(csi[alive].mean()) if alive.any() else float("nan"),
            "ssi_mean_alive": float(ssi[alive].mean()) if alive.any() else float("nan")}


def run(args) -> str:
    lane_id, lane_dir, tsets = args
    t0 = time.time()
    lane = load_lane(lane_dir)
    tsets = {"own": thr_set(lane), **tsets}
    labels, layers = lane.labels, lane.layers
    stored_pi = {p["transition"]: p for p in lane.results["process_intensity"]}
    snaps = json.load(open(lane_dir / "dev_snapshots" / "metrics" / "devtrain_snapshots.json"))
    out: Dict[str, Any] = {"lane_id": lane_id, "exp_dir": str(lane_dir.parent.parent), "labels": labels,
                           "layers": layers, "threshold_sets": tsets,
                           "snapshots": [{k: s.get(k) for k in ("milestone_index", "epoch", "weight_updates",
                                                                "train_accuracy", "val_accuracy",
                                                                "snapshot_type")} for s in snaps],
                           "real": {}, "null_dup": {}, "census": {}, "validation": {}}
    sel_all = lane.results["selectivity"]
    prev = None
    for lab in labels:
        cur = {L: lane.load_codes(lab, L) for L in layers}
        out["census"][lab] = {ts: {L: census_t(sel_all[lab][L], cur[L], tsets[ts]["thresholds"]) for L in layers}
                              for ts in tsets}
        # state-matched null at the duplicate site
        H4, Hp = cur["layer4"], cur["avgpool"]
        s4, sp = sel_all[lab]["layer4"], sel_all[lab]["avgpool"]
        m1, m2 = match(H4, Hp), match(Hp, H4)
        out["null_dup"][lab] = {ts: {"layer4->avgpool": classify_counts(H4, Hp, m1, s4, sp, **tsets[ts]),
                                     "avgpool->layer4": classify_counts(Hp, H4, m2, sp, s4, **tsets[ts])}
                                for ts in tsets}
        if prev is not None:
            plab, pdat = prev
            key = f"{plab}->{lab}"
            out["real"][key] = {ts: {} for ts in tsets}
            for L in layers:
                md = match(pdat[L], cur[L])
                for ts in tsets:
                    c = classify_counts(pdat[L], cur[L], md, sel_all[plab][L], sel_all[lab][L], **tsets[ts])
                    c["n_stable"], c["n_born"], c["n_died"] = md["n_stable"], md["n_born"], md["n_died"]
                    out["real"][key][ts][L] = c
            rec = {p: sum(out["real"][key]["own"][L][p] for L in layers) for p in PROCS}
            sto = {p: stored_pi[key][p] for p in PROCS} if key in stored_pi else None
            out["validation"][key] = {"stored": sto, "recomputed": rec,
                                      "exact": sto is not None and all(rec[p] == sto[p] for p in PROCS)}
        prev = (lab, cur)
    d = OUT / "per_lane"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{lane_id}.json", "w") as f:
        json.dump(out, f)
    n_ok = sum(v["exact"] for v in out["validation"].values())
    return f"{lane_id[:8]}: own-threshold counts exact {n_ok}/{len(out['validation'])} ({time.time() - t0:.0f}s)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    lanes = lanes_to_run()
    tsets = {"std50": thr_set(load_lane(lanes[STD50_ID])), "cap07": thr_set(load_lane(lanes[E15_ID]))}
    jobs = [(lid, d, tsets) for lid, d in lanes.items() if not a.only or lid.startswith(a.only)]
    print(f"{len(jobs)} lanes; threshold sets: " +
          json.dumps({k: v["thresholds"] for k, v in tsets.items()}), flush=True)
    with Pool(a.workers) as pool:
        for msg in pool.imap_unordered(run, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
