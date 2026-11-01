#!/usr/bin/env python3
"""RDT of the acquisition-blocking runs: events and feature census per epoch under one common threshold set per seed.

Tested contrasts
  P1  During epochs 1-10, superclass-blocked networks record fewer Ab-E events and hold fewer superclass-selective
      features than energy-matched networks.
  P2  The random-grouping block does the same.
  P3  Di-E shifts later (centre of mass) in the blocked arms; after release, Ab-E recovers before Di-E.
  P4  Energy-matched networks stay close to sham.

Arms      sham (sham_perepoch sham_e00_seed{S}: standard training, same branch checkpoint and schedule),
          gradsc = superclass-blocked, gradenergy = energy-matched, gradrg = random-grouping
          (blocking_rdt/lanes/{arm}_seed{S}, from run_blocking_rdt.py); seeds 7, 42, 137; epochs 0-15.
Windows   block = epochs 1-10, release = epochs 11-15; a transition (e-1)->e belongs to epoch e.
Thresholds 'common': one set per seed = the sham run's stored per-run thresholds (sai_task_general_thresh,
          ssi_abstraction_thresh, csi_differentiation_thresh) with its stored null mean / s.d. (the z-scale of
          Phase 6's 'increase' criterion), applied to all arms of that seed. 'own': each run's stored set (reported).
Events    Phase-6 per-image counts computed from the stored SAE codes and Phase-4 selectivity (noise_null_lib.match: Phase 5
          Hungarian matching, stable r >= 0.5, birth/death r < 0.2; noise_null_lib.classify_counts: vectorised Phase 6;
          curriculum_recount.load_lane / thr_set). Raw counts, no null subtraction; 'all' = sum over the five SAE layers
          (as process_intensity), plus each layer. Each lane's count under its own thresholds must equal its stored
          process_intensity at every transition.
Census    exclusive categories (Methods rule): alive = active on >= 1% of images;
          task-general = SAI > t_sai; superclass-selective = SSI > t_ssi & CSI <= t_csi & not task-general;
          class-selective = CSI > t_csi & not task-general (CSI NaN = not class-selective); % of alive features pooled
          over the census layers (layer1-3, avgpool; layer4 duplicates avgpool), plus counts and each layer.
COM       over transitions t = 0 ... 14 with all-layer counts as weights: COM = sum t * n_t / sum n_t.
Recovery  epoch, for blocked arm X, seed S and process p in {Ab-E, Di-E}: first epoch e in 11 ... 15 at which X's
          all-layer count at (e-1)->e is >= the energy-matched count of the same seed; 16 if none.
Measures  (common thresholds; differences paired by seed)
    P1  (a) Ab-E summed over the block window, gradsc - gradenergy < 0
        (b) superclass-selective % averaged over block checkpoints, gradsc - gradenergy < 0
    P2  (a), (b) as P1 with gradrg - gradenergy
    P3  (a) Di-E COM gradsc - gradenergy > 0   (b) Di-E COM gradrg - gradenergy > 0
        (c) recovery epoch Di-E - recovery epoch Ab-E > 0 for gradsc   (d) the same for gradrg
    P4  (a) |mean_S(Ab-E block, gradenergy - sham)| <= 10% of mean_S(sham Ab-E block)
        (b) |mean_S(superclass-selective % block, gradenergy - sham)| <= 10% of the sham mean
        (c) |mean_S(Di-E COM, gradenergy - sham)| <= 0.5 transitions
Outcomes  labels in the outputs. Directional components (P1-P3): 'SUPPORTED' if all 3 seeds are in the tested
          direction; 'WEAK' if the seed mean and 2 of 3 seeds are; 'NOT SUPPORTED' otherwise; 'NOT EVALUABLE' with
          fewer than 3 seeds or an incomplete window. Closeness components (P4): 'SUPPORTED' if the criterion holds.
          A contrast is labelled 'SUPPORTED' if every component is, 'NOT SUPPORTED' if any component is not, 'WEAK'
          otherwise, 'NOT EVALUABLE' if any component is.
Contrasts gradsc - gradenergy, gradrg - gradenergy, gradsc - sham, gradrg - sham, gradenergy - sham for all measures,
          windows and threshold sets: per-seed differences, mean, t-based 95% CI, seeds per direction, % change.
SAE noise floor: at epoch 0 every arm's network is identical to the sham's, but SAE training on MPS is not bitwise
reproducible over 5,000 steps; each lane's epoch-0 codes are compared with the sham's and the epoch-0 census
difference is reported as the run-to-run noise floor (descriptive).

Outputs in analysis_outputs/paper/RDT_blocking/ (pilot: RDT_blocking/pilot/):
  per_lane/{arm}_seed{S}.json  per-lane count cache (reused while sae_results.json and the thresholds are unchanged)
  rdt_blocking_long.csv        arm, seed, tset, kind (events | census), epoch, transition, layer, measure, value
  rdt_blocking_summary.json    settings, thresholds, validation, per-lane window summaries, contrasts, outcomes
Read-only on all data. CPU only.
Usage: python scripts/paper/blocking_rdt_analysis.py [--workers 3] [--pilot] [--force]
"""
from __future__ import annotations

import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noise_null_lib import PROCS, classify_counts, match  # noqa: E402
from curriculum_recount import load_lane, thr_set  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
E = Path(RDT_DATA + "/extended/experiments")
SHAM = "sham_perepoch/lanes/sham_e00_seed{s}"
LANES = {"sham": SHAM, "gradsc": "blocking_rdt/lanes/gradsc_seed{s}",
         "gradenergy": "blocking_rdt/lanes/gradenergy_seed{s}", "gradrg": "blocking_rdt/lanes/gradrg_seed{s}",
         # stage-matched dose control (descriptive, not part of P1-P4)
         "gradsc_l2": "blocking_rdt/lanes/gradsc_l2_seed{s}",
         "gradenergy_l2": "blocking_rdt/lanes/gradenergy_l2_seed{s}",
         # random directions (descriptive heavy-block control, not part of P1-P4)
         "gradrand": "blocking_rdt/lanes/gradrand_seed{s}"}
PILOT_LANES = {"sham": SHAM, "gradsc": "blocking_rdt_pilot/lanes/gradsc_seed{s}"}
LABEL = {"sham": "sham", "gradsc": "superclass-blocked", "gradenergy": "energy-matched", "gradrg": "random-grouping",
         "gradsc_l2": "superclass-blocked, layer2 to pooled", "gradenergy_l2": "energy-matched, layer2 to pooled",
         "gradrand": "random directions"}
SEEDS = [7, 42, 137]
MAX_EPOCH = 15
WINDOWS = {"block": (1, 10), "release": (11, 15),
           # descriptive: the window in which the energy-matched arm's dose
           # matches the superclass code's at layer2 to the pooled layer (Methods)
           "dose-matched (1-3)": (1, 3)}
CENSUS_LAYERS = ("layer1", "layer2", "layer3", "avgpool")
EV = {"tg_h": "Tg-E", "ab_h": "Ab-E", "di_h": "Di-E", "as_h": "As-E", "de_h": "De-E"}
CEN = ("n_alive", "n_tg", "n_sc", "n_cls", "pct_tg", "pct_sc", "pct_cls")
CONTRASTS = [("gradsc", "gradenergy"), ("gradrg", "gradenergy"), ("gradsc", "sham"), ("gradrg", "sham"),
             ("gradenergy", "sham"), ("gradsc_l2", "gradenergy_l2"), ("gradsc", "gradrg"), ("gradrand", "sham"),
             ("gradsc", "gradrand"), ("gradrg", "gradrand"), ("gradsc_l2", "sham"), ("gradenergy_l2", "sham")]
P4_REL_TOL, P4_COM_TOL = 0.10, 0.5


# counts (per lane, cached)

def census_ex(sel, H, t):
    cnt = (np.abs(H) > 1e-6).sum(axis=0)
    alive = cnt >= max(1, H.shape[0] * 0.01)
    sai = np.asarray(sel["feature_sai"], dtype=np.float64)
    ssi = np.asarray(sel["feature_ssi"], dtype=np.float64)
    csi = np.nan_to_num(np.asarray(sel["feature_csi"], dtype=np.float64), nan=-1.0)
    tg = alive & (sai > t["sai_task_general_thresh"])
    cs = csi > t["csi_differentiation_thresh"]
    sc = alive & (ssi > t["ssi_abstraction_thresh"]) & ~cs & ~tg
    cl = alive & cs & ~tg
    return {"n_alive": int(alive.sum()), "n_tg": int(tg.sum()), "n_sc": int(sc.sum()), "n_cls": int(cl.sum())}


def recount_lane(job):
    arm, s, d, common, max_epoch, out, force = (job[k] for k in ("arm", "seed", "dir", "common", "max_epoch", "out",
                                                                  "force"))
    d, out = Path(d), Path(out)
    t0 = time.time()
    st = (d / "sae_analysis" / "sae_results.json").stat()
    key = {"mtime": st.st_mtime, "size": st.st_size, "common": common, "max_epoch": max_epoch}
    if out.exists() and not force:
        old = json.load(open(out))
        if old.get("key") == key:
            return f"[recount] {arm}_seed{s}: cached"
    lane = load_lane(d)
    labels = [lab for lab in lane.labels if int(lab) <= max_epoch]
    assert labels == [str(e) for e in range(max_epoch + 1)], (arm, s, lane.labels)
    tsets = {"own": thr_set(lane), "common": common}
    sel = lane.results["selectivity"]
    stored = {p["transition"]: p for p in lane.results["process_intensity"]}
    res = {"arm": arm, "seed": s, "lane_dir": str(d), "key": key, "layers": lane.layers, "labels": labels,
           "tsets": tsets, "census": {}, "events": {}, "matching": {}, "validation": {}}
    prev = None
    for lab in labels:
        cur = {L: lane.load_codes(lab, L) for L in lane.layers}
        res["census"][lab] = {ts: {L: census_ex(sel[lab][L], cur[L], t["thresholds"]) for L in lane.layers}
                              for ts, t in tsets.items()}
        if prev is not None:
            plab, pc = prev
            k = f"{plab}->{lab}"
            res["events"][k] = {ts: {} for ts in tsets}
            res["matching"][k] = {}
            for L in lane.layers:
                md = match(pc[L], cur[L])
                res["matching"][k][L] = {x: int(md[x]) for x in ("n_stable", "n_born", "n_died")}
                for ts, t in tsets.items():
                    res["events"][k][ts][L] = classify_counts(pc[L], cur[L], md, sel[plab][L], sel[lab][L], **t)
            rec = {p: sum(res["events"][k]["own"][L][p] for L in lane.layers) for p in PROCS}
            sto = {p: int(stored[k][p]) for p in PROCS} if k in stored else None
            res["validation"][k] = {"stored": sto, "recount_own": rec,
                                    "exact": sto is not None and all(rec[p] == sto[p] for p in PROCS)}
        prev = (lab, cur)
    # epoch-0 SAE codes vs the sham's (identical inputs and settings)
    if arm != "sham":
        sd = E / SHAM.format(s=s) / "sae_analysis" / "activations" / "0"
        res["epoch0_codes_vs_sham"] = {}
        for L in lane.layers:
            a, b = lane.load_codes("0", L), np.load(sd / f"{L}.npy")
            md = match(a, b)
            res["epoch0_codes_vs_sham"][L] = {"identical": bool(np.array_equal(a, b)),
                                              "frac_equal_entries": float((a == b).mean()),
                                              "n_alive": int(np.asarray(md["alive_a"]).sum()),
                                              "n_stable_r05": int(md["n_stable"])}
    res["seconds"] = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"))
    n_ok = sum(v["exact"] for v in res["validation"].values())
    return f"[recount] {arm}_seed{s}: own-threshold recount exact {n_ok}/{len(res['validation'])} ({res['seconds']:.0f}s)"


# tables

def long_rows(r):
    rows = []
    base = {"arm": r["arm"], "arm_label": LABEL[r["arm"]], "seed": r["seed"]}
    for k, by_ts in r["events"].items():
        a, b = (int(x) for x in k.split("->"))
        for ts, by_L in by_ts.items():
            for L in r["layers"] + ["all"]:
                for p in EV:
                    v = sum(by_L[x][p] for x in r["layers"]) if L == "all" else by_L[L][p]
                    rows.append({**base, "tset": ts, "kind": "events", "epoch": b, "transition": k, "layer": L,
                                 "measure": EV[p], "value": v})
        for L, m in r["matching"][k].items():
            for x, v in m.items():
                rows.append({**base, "tset": "n/a", "kind": "matching", "epoch": b, "transition": k, "layer": L,
                             "measure": x, "value": v})
    for lab, by_ts in r["census"].items():
        for ts, by_L in by_ts.items():
            pooled = {c: sum(by_L[L][c] for L in CENSUS_LAYERS) for c in ("n_alive", "n_tg", "n_sc", "n_cls")}
            for L, c in list(by_L.items()) + [("census_pooled", pooled)]:
                na = max(c["n_alive"], 1)
                vals = {**c, "pct_tg": 100 * c["n_tg"] / na, "pct_sc": 100 * c["n_sc"] / na,
                        "pct_cls": 100 * c["n_cls"] / na}
                for m in CEN:
                    rows.append({**base, "tset": ts, "kind": "census", "epoch": int(lab), "transition": "",
                                 "layer": L, "measure": m, "value": vals[m]})
    return rows


def lane_summary(D, arm, s, ts):
    """Window sums / means, COMs and per-transition all-layer counts for one lane under one threshold set."""
    q = D[(D.arm == arm) & (D.seed == s) & (D.tset == ts)]
    ev = q[(q.kind == "events") & (q.layer == "all")].pivot(index="epoch", columns="measure", values="value")
    ce = q[(q.kind == "census") & (q.layer == "census_pooled")].pivot(index="epoch", columns="measure", values="value")
    out = {"windows": {}, "com": {}, "per_epoch": {p: {int(e): int(v) for e, v in ev[p].items()} for p in
                                                   ("Tg-E", "Ab-E", "Di-E")}}
    for w, (lo, hi) in WINDOWS.items():
        need = set(range(lo, hi + 1))
        e_ok, c_ok = need <= set(ev.index), need <= set(ce.index)
        o = {"complete": bool(e_ok and c_ok)}
        e_in, c_in = ev.loc[[e for e in ev.index if lo <= e <= hi]], ce.loc[[e for e in ce.index if lo <= e <= hi]]
        for p in ("Tg-E", "Ab-E", "Di-E"):
            o[f"{p} sum"] = float(e_in[p].sum()) if len(e_in) else float("nan")
        for m in ("pct_sc", "pct_cls", "n_sc", "n_cls", "pct_tg"):
            o[f"{m} mean"] = float(c_in[m].mean()) if len(c_in) else float("nan")
        out["windows"][w] = o
    out["com_complete"] = set(range(1, MAX_EPOCH + 1)) <= set(ev.index)
    for p in ("Tg-E", "Ab-E", "Di-E"):
        t = ev.index.values - 1.0
        n = ev[p].values.astype(float)
        out["com"][p] = float((t * n).sum() / n.sum()) if n.sum() > 0 else float("nan")
    return out


def recovery(S, arm, s, ts):
    """First epoch in 11..15 at which arm's all-layer count >= energy-matched's (same seed); 16 if none."""
    a, b = S.get((arm, s, ts)), S.get(("gradenergy", s, ts))
    if a is None or b is None:
        return None
    out = {}
    for p in ("Ab-E", "Di-E"):
        eps = [e for e in range(11, 16) if e in a["per_epoch"][p] and e in b["per_epoch"][p]]
        if len(eps) < 5:
            return None
        hit = [e for e in eps if a["per_epoch"][p][e] >= b["per_epoch"][p][e]]
        out[p] = hit[0] if hit else 16
    return out


def paired(vals):
    seeds = sorted(k for k, v in vals.items() if v is not None and np.isfinite(v))
    d = np.array([vals[k] for k in seeds], float)
    r = {"n": len(d), "per_seed": {str(k): float(vals[k]) for k in seeds}}
    if not len(d):
        return r
    r.update({"mean": float(d.mean()), "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum())})
    if len(d) >= 2:
        se = d.std(ddof=1) / math.sqrt(len(d))
        h = stats.t.ppf(0.975, len(d) - 1) * se
        r["ci95"] = [float(d.mean() - h), float(d.mean() + h)]
        r["p_two_sided"] = float(2 * stats.t.sf(abs(d.mean() / se), len(d) - 1)) if se > 0 else float("nan")
    return r


def contrasts(S, seeds, ts):
    out = []
    for a, b in CONTRASTS:
        for w in WINDOWS:
            meas = [f"{p} sum" for p in ("Tg-E", "Ab-E", "Di-E")] + [f"{m} mean" for m in
                                                                     ("pct_sc", "pct_cls", "n_sc", "n_cls", "pct_tg")]
            for m in meas:
                diff, rel, comp = {}, {}, True
                for s in seeds:
                    x, y = S.get((a, s, ts)), S.get((b, s, ts))
                    if x is None or y is None:
                        continue
                    xv, yv = x["windows"][w][m], y["windows"][w][m]
                    comp &= x["windows"][w]["complete"] and y["windows"][w]["complete"]
                    diff[s] = xv - yv
                    rel[s] = 100 * (xv / yv - 1) if yv else float("nan")
                r = paired(diff)
                fin = [v for v in rel.values() if np.isfinite(v)]
                out.append({"contrast": f"{a} - {b}", "tset": ts, "window": w, "measure": m, "complete": comp,
                            **r, "pct_change_mean": float(np.mean(fin)) if fin else None})
        for p in ("Tg-E", "Ab-E", "Di-E"):
            diff, comp = {}, True
            for s in seeds:
                x, y = S.get((a, s, ts)), S.get((b, s, ts))
                if x is None or y is None:
                    continue
                comp &= x["com_complete"] and y["com_complete"]
                diff[s] = x["com"][p] - y["com"][p]
            out.append({"contrast": f"{a} - {b}", "tset": ts, "window": "0-15", "measure": f"{p} COM (transition)",
                        "complete": comp, **paired(diff)})
    return out


# outcomes per contrast (P1-P4)

def directional(C, contrast, window, measure, sign):
    r = next((x for x in C if x["contrast"] == contrast and x["window"] == window and x["measure"] == measure), None)
    return _dir(r, sign, f"{contrast} | {window} | {measure} | tested {'>' if sign > 0 else '<'} 0")


def _dir(r, sign, name):
    if r is None or r.get("n", 0) < 3 or not r.get("complete", False):
        return {"component": name, "verdict": "NOT EVALUABLE", "data": r}
    n_dir = r["n_pos"] if sign > 0 else r["n_neg"]
    v = ("SUPPORTED" if n_dir == 3 else "WEAK" if (n_dir >= 2 and r["mean"] * sign > 0) else "NOT SUPPORTED")
    return {"component": name, "verdict": v, "seeds_in_direction": n_dir, "mean": r["mean"],
            "per_seed": r["per_seed"], "ci95": r.get("ci95")}


def combine(comps):
    vs = [c["verdict"] for c in comps]
    if "NOT EVALUABLE" in vs:
        v = "NOT EVALUABLE"
    elif all(x == "SUPPORTED" for x in vs):
        v = "SUPPORTED"
    elif "NOT SUPPORTED" in vs:
        v = "NOT SUPPORTED"
    else:
        v = "WEAK"
    return {"verdict": v, "components": comps}


def verdicts(C, S, seeds):
    ts = "common"
    Cc = [x for x in C if x["tset"] == ts]
    P = {}
    for key, arm in (("P1", "gradsc"), ("P2", "gradrg")):
        P[key] = combine([directional(Cc, f"{arm} - gradenergy", "block", "Ab-E sum", -1),
                          directional(Cc, f"{arm} - gradenergy", "block", "pct_sc mean", -1)])
    comps = [directional(Cc, f"{arm} - gradenergy", "0-15", "Di-E COM (transition)", +1) for arm in ("gradsc", "gradrg")]
    for arm in ("gradsc", "gradrg"):
        rec = {s: recovery(S, arm, s, ts) for s in seeds}
        diff = {s: (r["Di-E"] - r["Ab-E"]) if r else None for s, r in rec.items()}
        r = paired(diff)
        r["complete"] = all(v is not None for v in rec.values()) and len(seeds) == 3
        c = _dir(r, +1, f"{arm}: recovery epoch Di-E - recovery epoch Ab-E (vs gradenergy, epochs 11-15) | tested > 0")
        c["recovery_epochs"] = {str(s): v for s, v in rec.items()}
        comps.append(c)
    P["P3"] = combine(comps)
    comps = []
    for w, m, kind, tol in (("block", "Ab-E sum", "rel", P4_REL_TOL), ("block", "pct_sc mean", "rel", P4_REL_TOL),
                            ("0-15", "Di-E COM (transition)", "abs", P4_COM_TOL)):
        r = next((x for x in Cc if x["contrast"] == "gradenergy - sham" and x["window"] == w and x["measure"] == m),
                 None)
        name = f"gradenergy - sham | {w} | {m} | |mean diff| <= " + (f"{tol:.0%} of sham mean" if kind == "rel"
                                                                       else f"{tol} transitions")
        if r is None or r.get("n", 0) < 3 or not r.get("complete", False):
            comps.append({"component": name, "verdict": "NOT EVALUABLE", "data": r})
            continue
        if kind == "rel":
            sham = [S[("sham", s, ts)]["windows"][w][m] for s in seeds]
            stat = abs(r["mean"]) / abs(np.mean(sham))
        else:
            stat = abs(r["mean"])
        comps.append({"component": name, "verdict": "SUPPORTED" if stat <= tol else "NOT SUPPORTED",
                      "statistic": float(stat), "mean_diff": r["mean"], "per_seed": r["per_seed"]})
    P["P4"] = combine(comps)
    return P


def _c0(D, arm, s, m):
    q = D[(D.arm == arm) & (D.seed == s) & (D.tset == "common") & (D.kind == "census") & (D.epoch == 0)
          & (D.layer == "census_pooled") & (D.measure == m)]
    return float(q.value.iloc[0]) if len(q) == 1 else float("nan")


# main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--pilot", action="store_true", help="sham + pilot superclass-blocked lane, seed 7, epochs 0-1")
    ap.add_argument("--force", action="store_true", help="recount even if the per-lane cache is current")
    a = ap.parse_args()
    out = REPO / "analysis_outputs" / "paper" / "RDT_blocking" / ("pilot" if a.pilot else "")
    out.mkdir(parents=True, exist_ok=True)
    lanes_spec, seeds, max_epoch = (PILOT_LANES, [7], 1) if a.pilot else (LANES, SEEDS, MAX_EPOCH)
    common = {}
    for s in seeds:
        common[s] = thr_set(load_lane(E / SHAM.format(s=s)))
    jobs, missing = [], []
    for arm, pat in lanes_spec.items():
        for s in seeds:
            d = E / pat.format(s=s)
            if not (d / ".sae_done").exists() or not (d / "sae_analysis" / "sae_results.json").exists():
                missing.append(f"{arm}_seed{s}")
                continue
            jobs.append({"arm": arm, "seed": s, "dir": str(d), "common": common[s], "max_epoch": max_epoch,
                         "out": str(out / "per_lane" / f"{arm}_seed{s}.json"), "force": a.force})
    print(f"{len(jobs)} lanes; missing: {missing}; common thresholds: "
          + json.dumps({s: c["thresholds"] for s, c in common.items()}), flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(max(1, min(a.workers, len(jobs)))) as pool:
        for msg in pool.imap_unordered(recount_lane, jobs):
            print(msg, flush=True)
    recount_secs = time.time() - t0
    R = [json.load(open(j["out"])) for j in jobs]
    D = pd.DataFrame([row for r in R for row in long_rows(r)])
    D.to_csv(out / "rdt_blocking_long.csv", index=False)
    S = {(r["arm"], r["seed"], ts): lane_summary(D, r["arm"], r["seed"], ts) for r in R for ts in ("common", "own")}
    C = [c for ts in ("common", "own") for c in contrasts(S, seeds, ts)]
    pd.DataFrame([{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in c.items()} for c in C]).to_csv(
        out / "rdt_blocking_contrasts.csv", index=False)
    val = {f"{r['arm']}_seed{r['seed']}": {"n_transitions": len(r["validation"]),
                                           "n_exact": sum(v["exact"] for v in r["validation"].values()),
                                           "mismatches": {k: v for k, v in r["validation"].items() if not v["exact"]}}
           for r in R}
    summary = {
        "pilot": a.pilot,
        "settings": {"arms": {k: LABEL[k] for k in lanes_spec}, "lanes": {k: str(E / v) for k, v in lanes_spec.items()},
                     "seeds": seeds, "epochs": [0, max_epoch], "windows": WINDOWS, "census_layers": CENSUS_LAYERS,
                     "primary_thresholds": "common = sham_e00_seed{S} stored analysis_thresholds + its stored null "
                                           "mean/s.d. (z-scale), applied to every arm of seed S",
                     "secondary_thresholds": "own = each run's stored set",
                     "event_counts": "raw Phase-6 per-image counts, all five SAE layers summed (and per layer)",
                     "p4_tolerances": {"relative": P4_REL_TOL, "com_transitions": P4_COM_TOL}},
        "common_thresholds": {str(s): c for s, c in common.items()},
        "lanes_missing": missing,
        "validation_own_threshold_recount_vs_stored_process_intensity": val,
        "validation_all_exact": all(v["n_exact"] == v["n_transitions"] for v in val.values()),
        "epoch0_codes_vs_sham": {f"{r['arm']}_seed{r['seed']}": r.get("epoch0_codes_vs_sham") for r in R
                                 if r["arm"] != "sham"},
        "census_noise_floor_epoch0_vs_sham": {
            f"{r['arm']}_seed{r['seed']}": {m: float(_c0(D, r['arm'], r['seed'], m) - _c0(D, 'sham', r['seed'], m))
                                            for m in ("pct_sc", "pct_cls", "pct_tg", "n_sc", "n_cls")}
            for r in R if r["arm"] != "sham" and (D.arm == "sham").any()},
        "recount_seconds": {f"{r['arm']}_seed{r['seed']}": round(r["seconds"], 1) for r in R},
        "recount_wall_seconds": round(recount_secs, 1),
        "lane_summaries": {f"{k[0]}_seed{k[1]}|{k[2]}": v for k, v in S.items()},
        "recovery_epochs_common": {f"{arm}_seed{s}": recovery(S, arm, s, "common") for arm in ("gradsc", "gradrg")
                                   for s in seeds},
        "contrasts": C,
        "predictions": verdicts(C, S, seeds),
    }
    json.dump(summary, open(out / "rdt_blocking_summary.json", "w"), indent=1, default=str)
    print(f"validation (own-threshold recount == stored process_intensity): "
          + ", ".join(f"{k} {v['n_exact']}/{v['n_transitions']}" for k, v in val.items()), flush=True)
    for k, v in summary["predictions"].items():
        print(f"{k}: {v['verdict']}", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
