#!/usr/bin/env python3
"""RDT of the fixed-label noise experiment (Extended Data Fig. 8) under one common threshold set per seed, in the framework of
the blocking RDT (blocking_rdt_analysis.py), so that the two interventions are read out the same way.

Lanes: static_noise_s{S}/lanes/{standard_resnet18 (clean), within_sc_resnet18_throughout (within),
between_sc_resnet18_throughout (between)}, seeds 7, 42, 137, 256, 314 (the similarity-matched arm has no SAEs).
Checkpoints: labels 0-6 = epochs 0, 5, ..., 30 (label 7, the duplicated final checkpoint, is excluded).
Thresholds (primary, 'common'): one set per seed = the clean run's stored pipeline thresholds (sae_results.json
analysis_thresholds) with its stored null mean / s.d. (z-scale of Phase 6), applied to all three arms of that seed;
secondary ('own'): each run's own stored set.
Events: Phase-6 per-image counts recounted from the stored SAE codes (noise_null_lib.match / classify_counts; validated per
transition against the run's stored process_intensity under its own thresholds), raw counts summed over the five SAE
layers. Census: exclusive categories (census_ex of blocking_rdt_analysis), % of alive features pooled over layer1,
layer2, layer3 and the pooled layer.
Windows: early = transitions 0->5 and 5->10; late = 10->15 ... 25->30; census early = epochs 5, 10; all = epochs 5-30.
Contrasts (paired over seeds, t-based 95% CI): between - within (primary: the two noise types corrupt the same
fraction of fine labels), between - clean, within - clean.
Epoch-0 noise floor: the three arms of a seed share the initialisation (identical epoch-0 network), so their epoch-0
census differences measure SAE re-training noise.
Outputs analysis_outputs/paper/RDT_noise/: per_lane/<arm>_seed<S>.json, rdt_noise_long.csv, rdt_noise_contrasts.csv,
rdt_noise_summary.json.
Read-only on all data. CPU only.
Usage: python scripts/paper/noise_rdt_analysis.py [--workers 3] [--force]
"""
from __future__ import annotations

import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noise_null_lib import PROCS, classify_counts, match  # noqa: E402
from blocking_rdt_analysis import CENSUS_LAYERS, EV, census_ex, paired  # noqa: E402
from curriculum_recount import load_lane, thr_set  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
E = Path(RDT_DATA + "/extended/experiments")
OUT = REPO / "analysis_outputs/paper/RDT_noise"
LANES = {"clean": "standard_resnet18", "within": "within_sc_resnet18_throughout",
         "between": "between_sc_resnet18_throughout"}
SEEDS = [7, 42, 137, 256, 314]
LABELS = [str(i) for i in range(7)]          # epochs 0, 5, ..., 30
EPOCH = {str(i): 5 * i for i in range(7)}
CONTRASTS = [("between", "within"), ("between", "clean"), ("within", "clean")]


def lane_dir(arm, s):
    return E / f"static_noise_s{s}" / "lanes" / LANES[arm]


def recount(job):
    arm, s, common, out, force = job["arm"], job["seed"], job["common"], Path(job["out"]), job["force"]
    d = lane_dir(arm, s)
    st = (d / "sae_analysis" / "sae_results.json").stat()
    key = {"mtime": st.st_mtime, "size": st.st_size, "common": common}
    if out.exists() and not force and json.load(open(out)).get("key") == key:
        return f"[recount] {arm}_seed{s}: cached"
    t0 = time.time()
    lane = load_lane(d)
    assert [x for x in lane.labels if x in LABELS] == LABELS, lane.labels
    tsets = {"own": thr_set(lane), "common": common}
    sel = lane.results["selectivity"]
    stored = {p["transition"]: p for p in lane.results["process_intensity"]}
    res = {"arm": arm, "seed": s, "key": key, "layers": lane.layers, "census": {}, "events": {}, "validation": {}}
    prev = None
    for lab in LABELS:
        cur = {L: lane.load_codes(lab, L) for L in lane.layers}
        res["census"][lab] = {ts: {L: census_ex(sel[lab][L], cur[L], t["thresholds"]) for L in lane.layers}
                              for ts, t in tsets.items()}
        if prev is not None:
            plab, pc = prev
            k = f"{plab}->{lab}"
            res["events"][k] = {ts: {} for ts in tsets}
            for L in lane.layers:
                md = match(pc[L], cur[L])
                for ts, t in tsets.items():
                    res["events"][k][ts][L] = classify_counts(pc[L], cur[L], md, sel[plab][L], sel[lab][L], **t)
            rec = {p: sum(res["events"][k]["own"][L][p] for L in lane.layers) for p in PROCS}
            sto = {p: int(stored[k][p]) for p in PROCS} if k in stored else None
            res["validation"][k] = {"stored": sto, "recount_own": rec,
                                    "exact": sto is not None and all(rec[p] == sto[p] for p in PROCS)}
        prev = (lab, cur)
    res["seconds"] = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"))
    n_ok = sum(v["exact"] for v in res["validation"].values())
    return f"[recount] {arm}_seed{s}: own-threshold recount exact {n_ok}/{len(res['validation'])} ({res['seconds']:.0f}s)"


def rows_of(r):
    rows = []
    for k, by_ts in r["events"].items():
        a, b = k.split("->")
        for ts, by_L in by_ts.items():
            for p, name in EV.items():
                rows.append(dict(arm=r["arm"], seed=r["seed"], tset=ts, kind="events", epoch=EPOCH[b],
                                 transition=f"{EPOCH[a]}->{EPOCH[b]}", measure=name,
                                 value=sum(by_L[L][p] for L in r["layers"])))
    for lab, by_ts in r["census"].items():
        for ts, by_L in by_ts.items():
            c = {x: sum(by_L[L][x] for L in CENSUS_LAYERS) for x in ("n_alive", "n_tg", "n_sc", "n_cls")}
            na = max(c["n_alive"], 1)
            for m, v in (("pct_sc", 100 * c["n_sc"] / na), ("pct_cls", 100 * c["n_cls"] / na),
                         ("pct_tg", 100 * c["n_tg"] / na), ("n_sc", c["n_sc"]), ("n_cls", c["n_cls"])):
                rows.append(dict(arm=r["arm"], seed=r["seed"], tset=ts, kind="census", epoch=EPOCH[lab],
                                 transition="", measure=m, value=v))
    return rows


WIN = {"events": {"early (0-10)": [5, 10], "late (10-30)": [15, 20, 25, 30], "all (0-30)": [5, 10, 15, 20, 25, 30]},
       "census": {"early (5-10)": [5, 10], "all (5-30)": [5, 10, 15, 20, 25, 30], "epoch 30": [30]}}


def summarise(D, ts):
    out = []
    q = D[D.tset == ts]
    for kind, wins in WIN.items():
        meas = ("Tg-E", "Ab-E", "Di-E") if kind == "events" else ("pct_sc", "pct_cls", "pct_tg")
        agg = "sum" if kind == "events" else "mean"
        for w, eps in wins.items():
            for m in meas:
                V = q[(q.kind == kind) & (q.measure == m) & q.epoch.isin(eps)].groupby(["arm", "seed"]).value.agg(agg)
                for a, b in CONTRASTS:
                    diff = {s: float(V[(a, s)] - V[(b, s)]) for s in SEEDS}
                    rel = [100 * (V[(a, s)] / V[(b, s)] - 1) for s in SEEDS if V[(b, s)]]
                    out.append({"tset": ts, "kind": kind, "window": w, "measure": m, "contrast": f"{a} - {b}",
                                **paired(diff), "pct_change_mean": float(np.mean(rel)) if rel else None,
                                "mean_a": float(np.mean([V[(a, s)] for s in SEEDS])),
                                "mean_b": float(np.mean([V[(b, s)] for s in SEEDS]))})
    # centre of mass of Ab-E and Di-E over the six transitions (index 0..5)
    for m in ("Ab-E", "Di-E"):
        com = {}
        for (arm, s), g in q[(q.kind == "events") & (q.measure == m)].groupby(["arm", "seed"]):
            g = g.sort_values("epoch")
            n = g.value.to_numpy(float)
            com[(arm, s)] = float((np.arange(len(n)) * n).sum() / n.sum())
        for a, b in CONTRASTS:
            out.append({"tset": ts, "kind": "events", "window": "COM (transition index 0-5)", "measure": m,
                        "contrast": f"{a} - {b}", **paired({s: com[(a, s)] - com[(b, s)] for s in SEEDS})})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    common = {s: thr_set(load_lane(lane_dir("clean", s))) for s in SEEDS}
    jobs = [{"arm": arm, "seed": s, "common": common[s], "out": str(OUT / "per_lane" / f"{arm}_seed{s}.json"),
             "force": a.force} for arm in LANES for s in SEEDS]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for msg in pool.imap_unordered(recount, jobs):
            print(msg, flush=True)
    R = [json.load(open(j["out"])) for j in jobs]
    D = pd.DataFrame([x for r in R for x in rows_of(r)])
    D.to_csv(OUT / "rdt_noise_long.csv", index=False)
    val = {f"{r['arm']}_seed{r['seed']}": f"{sum(v['exact'] for v in r['validation'].values())}/{len(r['validation'])}"
           for r in R}
    c0 = D[(D.tset == "common") & (D.kind == "census") & (D.epoch == 0)].pivot_table(
        index=["seed", "measure"], columns="arm", values="value")
    floor = {m: float((c0.xs(m, level="measure")[["within", "between"]].sub(
        c0.xs(m, level="measure")["clean"], axis=0)).abs().max().max()) for m in ("pct_sc", "pct_cls", "pct_tg")}
    S = summarise(D, "common") + summarise(D, "own")
    pd.DataFrame([{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in x.items()} for x in S]).to_csv(
        OUT / "rdt_noise_contrasts.csv", index=False)
    json.dump({"validation_own_vs_stored": val, "validation_all_exact": all(v.split("/")[0] == v.split("/")[1]
                                                                            for v in val.values()),
               "common_thresholds": {str(s): c["thresholds"] for s, c in common.items()},
               "epoch0_census_noise_floor_max_abs_pct": floor, "contrasts": S},
              open(OUT / "rdt_noise_summary.json", "w"), indent=1, default=str)
    print("validation:", val)
    print("epoch-0 noise floor (max |arm - clean|, % points):", floor)
    for x in S:
        if x["tset"] == "common" and x["contrast"] == "between - within":
            print(f"{x['kind']:6s} {x['window']:26s} {x['measure']:6s} mean {x.get('mean', float('nan')):+10.2f} "
                  f"CI {x.get('ci95')} neg {x.get('n_neg')}/5 pct {x.get('pct_change_mean')}")


if __name__ == "__main__":
    main()
