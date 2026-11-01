#!/usr/bin/env python3
"""Superclass-blind pre-training (blind_pretrain): the two Fig. 6 / ED Fig. 9 measures that
blind_pretrain_analysis.py does not produce (used by scripts/paper/figures/interventions/blind_arm.py).

census   True-superclass-selective SAE features over checkpoints (Fig. 6a), as hierarchy_sae.main for one more lane:
         hierarchy_sae.feature_table / summarise on the stored avgpool SAE codes ('sae') and raw avgpool units ('raw'),
         true superclasses, the seed's imposed random partition for the *_imp columns, 30 null partitions
         (null_partitions(30, true_vec(), 70000)), common SSI threshold of the seed (SAE_outcomes/thresholds_EC.json).
         Lane 'sc_blind_K20', seeds 7, 42, 137 (hierarchy_lib.SEEDS). Writes BLIND_PRETRAIN/census_timecourse.csv
         (schema of DICTFREE/hierarchy/D_sae_census_timecourse.csv).
events   Tg-E / Ab-E / Di-E per real transition at the duplicate sites (layer4 + avgpool): real, same-data null and
         excess, as sae_outcomes events (EC) for arm 'sc_blind_K20', seeds 7, 42, 137, 256, 314. Integrity check
         first (sae_outcomes._integrity_lane), then _events_lane with the seed's common threshold set (scratch_E30
         reference, SAE_outcomes/thresholds_EC.json); only tset 'common' is produced. Writes BLIND_PRETRAIN/events_EC.csv
         (schema of SAE_outcomes/events_EC.csv) and BLIND_PRETRAIN/integrity_EC_blind.csv.
dryrun   Both computations on existing curriculum lanes, compared with the stored rows: census on sc_rand_K20 seed
         42 vs D_sae_census_timecourse.csv, events on sc_true_K20 seed 42 (all three threshold sets, matches
         recomputed) vs SAE_outcomes/events_EC.csv. Writes only to --scratch.

Read-only on all lane data.
Usage: PYTHONHASHSEED=0 python scripts/paper/blind_pretrain_measures.py {census,events,dryrun} [--scratch DIR]
"""
import argparse
import json
import sys
import time
from pathlib import Path
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import sae_outcomes as A4  # noqa: E402
import hierarchy_lib as EL  # noqa: E402
import hierarchy_sae as ES  # noqa: E402
from noise_null_lib import Lane  # noqa: E402

REPO = HERE.parents[1]
RESULTS = REPO / "analysis_outputs" / "paper"
OUT = RESULTS / "BLIND_PRETRAIN"
EXPS = Path(RDT_DATA + "/extended/experiments")
BLIND = EXPS / "blind_pretrain" / "lanes"
ARM = "sc_blind_K20"
SEEDS_ALL = [7, 42, 137, 256, 314]
STORED_CENSUS = RESULTS / "DICTFREE" / "hierarchy" / "D_sae_census_timecourse.csv"
STORED_EVENTS = RESULTS / "SAE_outcomes" / "events_EC.csv"
THRESH_EC = RESULTS / "SAE_outcomes" / "thresholds_EC.json"


def say(*a):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}]", *a, flush=True)


def blind_dir(s):
    return BLIND / f"{ARM}_seed{s}"


def sae_ready(d):
    return A4.is_ready(Path(d))


# census (hierarchy_sae.main, one lane)
def census_lane(d, kind, s, cache=None):
    """Rows of hierarchy_sae.main for lane directory d (kind = lane name written to the 'lane' column)."""
    assert s in EL.SEEDS and s not in EL.EXCLUDED, f"seed {s} is not in the open seed set"
    tv = EL.true_vec()
    nulls = EL.null_partitions(ES.N_NULL, tv, 70000)
    iv = EL.imposed_vec(s)
    thr = ES.THR[str(s)]["thresholds"]["ssi_abstraction_thresh"]
    have_sae = (d / "sae_analysis" / "sae_results.json").exists()
    rows = []
    for sn in EL.snapshots(d):
        X, y = EL.load_acts(d, sn["ms"], ["avgpool"])
        srcs = {"raw": X["avgpool"]}
        ap = d / "sae_analysis" / "activations" / str(sn["ms"]) / "avgpool.npy"
        if have_sae and ap.exists():
            srcs["sae"] = np.load(ap)
        for src, H in srcs.items():
            ft = ES.feature_table(H, y, tv, iv, nulls)
            if cache is not None and sn["epoch"] in (0, 20, 50):
                ft.to_csv(cache / f"sae_feat_{kind}_s{s}_ms{sn['ms']}_{src}.csv.gz", index=False)
            rows.append({"lane": kind, "seed": s, "ms": sn["ms"], "epoch": sn["epoch"], "src": src,
                         "thr": thr, **ES.summarise(ft, thr)})
        say(f"census {kind} s{s} ms{sn['ms']} epoch {sn['epoch']} {list(srcs)}")
    return pd.DataFrame(rows)


def cmd_census():
    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    parts = []
    for s in EL.SEEDS:
        d = blind_dir(s)
        if not sae_ready(d):
            say(f"census: SAE not finished for {d}; seed {s} SKIPPED")
            continue
        parts.append(census_lane(d, ARM, s, cache))
    if not parts:
        raise SystemExit("census: no lane ready")
    C = pd.concat(parts, ignore_index=True)
    cols = pd.read_csv(STORED_CENSUS, nrows=1).columns.tolist()
    C = C[cols]
    C.to_csv(OUT / "census_timecourse.csv", index=False)
    say(f"census_timecourse.csv: {len(C)} rows, seeds {sorted(C.seed.unique().tolist())}")
    print(C[C.src == "sae"].pivot(index="seed", columns="epoch", values="census_genuine").mul(100).round(1))


# events (sae_outcomes events EC, one arm)
def write_null_stub(ln, cdir):
    """The only fields of the threshold-step cache that _events_lane reads: real checkpoints and alive_min,
    computed exactly as sae_outcomes._null_lane does."""
    d = Path(ln["dir"])
    lane = Lane(d)
    ci = A4.real_ckpts(A4.checkpoint_info(d, lane.labels))
    alive_min = max(1, int(np.ceil(lane.n_classes * lane.n_per_class * A4.ALIVE_FRAC)))
    out = {"lane": A4.lane_key(ln), "arm": ln["arm"], "seed": ln["seed"], "lane_dir": str(d),
           "labels_real": [c["label"] for c in ci], "ckpts": ci, "layers": lane.layers, "alive_min": alive_min,
           "note": "stub for _events_lane (ckpts, alive_min only); no permutation null computed"}
    json.dump(out, open(cdir / f"{A4.lane_key(ln)}_null.json", "w"), default=str)
    return out


def events_lane(ln, cdir, T):
    write_null_stub(ln, cdir)
    r = A4._events_lane({"lane": ln, "cache": str(cdir), "T": T, "extra_pairs": []})
    say(r["msg"])
    if "ERROR" in r["msg"]:
        raise RuntimeError(r["msg"])
    return pd.DataFrame(r["rows"])


def integrity_lane(ln):
    r = A4._integrity_lane({"lane": ln})
    say(r["msg"])
    df = pd.DataFrame(r["rows"])
    return df, bool(len(df)) and bool(df["pass"].all())


def cmd_events():
    T = json.load(open(THRESH_EC))
    cdir = OUT / "cache" / "events"
    cdir.mkdir(parents=True, exist_ok=True)
    ints, parts = [], []
    for s in SEEDS_ALL:
        ln = {"arm": ARM, "seed": s, "dir": blind_dir(s)}
        if not sae_ready(ln["dir"]):
            say(f"events: SAE not finished for {ln['dir']}; seed {s} SKIPPED")
            continue
        assert str(s) in T["common"], f"no common threshold set for seed {s}"
        I, ok = integrity_lane(ln)
        ints.append(I)
        pd.concat(ints, ignore_index=True).to_csv(OUT / "integrity_EC_blind.csv", index=False)
        if not ok:
            say(f"events: integrity FAILED for {A4.lane_key(ln)}; lane EXCLUDED (as sae_outcomes passing_lanes)")
            continue
        parts.append(events_lane(ln, cdir, T))
        pd.concat(parts, ignore_index=True).to_csv(OUT / "events_EC.csv", index=False)
    if not parts:
        raise SystemExit("events: no lane ready")
    E = pd.concat(parts, ignore_index=True)
    cols = pd.read_csv(STORED_EVENTS, nrows=1).columns.tolist()
    E = E[cols]
    E.to_csv(OUT / "events_EC.csv", index=False)
    say(f"events_EC.csv: {len(E)} rows, tsets {sorted(E.tset.unique())}, seeds {sorted(E.seed.unique().tolist())}")
    print(E.groupby("seed")[["ab_h_real", "di_h_real"]].sum())


# dry run on existing lanes
def compare(new, old, keys, label):
    num = [c for c in old.columns if c not in keys and pd.api.types.is_numeric_dtype(old[c])]
    m = old.merge(new, on=keys, suffixes=("_old", "_new"), how="outer", indicator=True)
    assert (m["_merge"] == "both").all(), f"{label}: row sets differ\n{m[m._merge != 'both'][keys + ['_merge']]}"
    worst, bad = 0.0, []
    for c in num:
        a, b = m[f"{c}_old"].astype(float).values, m[f"{c}_new"].astype(float).values
        same_nan = np.isnan(a) == np.isnan(b)
        diff = np.where(np.isnan(a) & np.isnan(b), 0.0, np.abs(a - b))
        rel = diff / (1.0 + np.abs(np.nan_to_num(a)))
        worst = max(worst, float(np.nanmax(rel)) if len(rel) else 0.0)
        if not same_nan.all() or (rel > 1e-9).any():
            bad.append((c, float(np.nanmax(rel))))
    print(f"{label}: {len(m)} rows, {len(num)} numeric columns, max rel diff {worst:.3g}; "
          f"{'MATCH' if not bad else 'MISMATCH ' + str(bad)}", flush=True)
    return not bad


def cmd_dryrun(scratch):
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    ok = True
    # census: sc_rand_K20 seed 42
    t0 = time.time()
    s = 42
    new = census_lane(EL.lane_dir("sc_rand_K20", s), "sc_rand_K20", s)
    new.to_csv(scratch / "dry_census_sc_rand_K20_s42.csv", index=False)
    old = pd.read_csv(STORED_CENSUS)
    old = old[(old.lane == "sc_rand_K20") & (old.seed == s)]
    ok &= compare(new, old, ["lane", "seed", "ms", "epoch", "src"], "census sc_rand_K20 s42 vs D_sae_census_timecourse")
    say(f"census dry run {time.time() - t0:.0f}s")
    # events: sc_true_K20 seed 42, matches recomputed (empty cache dir)
    t0 = time.time()
    T = json.load(open(THRESH_EC))
    cdir = scratch / "events_cache"
    cdir.mkdir(exist_ok=True)
    for f in cdir.glob("*"):
        f.unlink()
    ln = {"arm": "sc_true_K20", "seed": s, "dir": EXPS / "curriculum" / "lanes" / f"sc_true_K20_seed{s}"}
    stub = write_null_stub(ln, cdir)
    ref = json.load(open(RESULTS / "SAE_outcomes" / "cache" / "EC" / f"{A4.lane_key(ln)}_null.json"))
    ck_ok = json.loads(json.dumps(stub["ckpts"], default=str)) == ref["ckpts"] and stub["alive_min"] == ref["alive_min"]
    print(f"checkpoint list / alive_min vs SAE_outcomes null cache: {'MATCH' if ck_ok else 'MISMATCH'}", flush=True)
    ok &= ck_ok
    I, iok = integrity_lane(ln)
    I.to_csv(scratch / "dry_integrity_sc_true_K20_s42.csv", index=False)
    print(f"integrity sc_true_K20 s42: {'pass' if iok else 'FAIL'} ({len(I)} ckpt-layers)", flush=True)
    new = events_lane(ln, cdir, T)
    new.to_csv(scratch / "dry_events_sc_true_K20_s42.csv", index=False)
    old = pd.read_csv(STORED_EVENTS)
    old = old[(old.arm == "sc_true_K20") & (old.seed == s)]
    ok &= compare(new, old, ["lane", "arm", "seed", "tset", "transition"], "events sc_true_K20 s42 vs SAE_outcomes/events_EC")
    ok &= compare(new[new.tset == "common"], old[old.tset == "common"], ["lane", "arm", "seed", "tset", "transition"],
                  "  of which tset 'common'")
    say(f"events dry run {time.time() - t0:.0f}s")
    print("DRY RUN", "MATCH" if ok else "MISMATCH", flush=True)
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["census", "events", "dryrun"])
    ap.add_argument("--scratch", default=None, help="dryrun only: output directory")
    a = ap.parse_args()
    if a.cmd == "census":
        cmd_census()
    elif a.cmd == "events":
        cmd_events()
    else:
        assert a.scratch, "--scratch required for dryrun"
        cmd_dryrun(a.scratch)


if __name__ == "__main__":
    main()
