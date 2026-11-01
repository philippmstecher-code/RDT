#!/usr/bin/env python3
"""Event counts for Figs 1b,c / 2a under per-run alive-only thresholds, for the 9 core CIFAR-100 runs and the
Tiny ImageNet run.

Phase-5 matching (noise_null_lib.match) and Phase-6 counting (noise_null_lib.classify_counts) on the stored codes, every layer and
real transition (the duplicate '<k>->terminal' transition is dropped, as in the figures), with
  stored  the run's stored pipeline thresholds and adaptive-threshold z-scales (check: must reproduce
          sae_results.json['process_intensity'] exactly)
  alive   the run's alive-only pooled null thresholds (CENSUS_robust/tables/thresholds_per_run.csv,
          alive_thr_*); z-scales for stable-feature increases kept from the stored null (only the thresholds change).
Writes analysis_outputs/paper/THRESHOLD_impact/core_events.csv.
Usage: core_events_alive.py [--workers 3]
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

os.environ.setdefault("OMP_NUM_THREADS", "2")
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noise_null_lib import REPO, Lane, classify_counts, match  # noqa: E402

EXP = Path(RDT_DATA + "/core/experiments")
CORE_EXP = "abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb"
TIN_EXP = "821eaadb-0160-48f8-9669-ec3d9185c428"
RUNS = [("ResNet-18", 42, CORE_EXP, "7db28a9a", "ResNet18-100class-10ms-seed42"),
        ("ResNet-18", 137, CORE_EXP, "a0bcbcba", "ResNet18-100class-10ms-seed137"),
        ("ResNet-18", 256, CORE_EXP, "935c13ae", "ResNet18-100class-10ms-seed256"),
        ("ViT-Small", 42, CORE_EXP, "10d31151", "VitSmall-100class-10ms-seed42"),
        ("ViT-Small", 137, CORE_EXP, "909d7f68", "VitSmall-100class-10ms-seed137"),
        ("ViT-Small", 256, CORE_EXP, "abad8975", "VitSmall-100class-10ms-seed256"),
        ("CCT-7", 42, CORE_EXP, "4bebc329", "CCT7-100class-10ms-seed42"),
        ("CCT-7", 137, CORE_EXP, "edf8275e", "CCT7-100class-10ms-seed137"),
        ("CCT-7", 256, CORE_EXP, "445dd4d4", "CCT7-100class-10ms-seed256"),
        ("tiny", 42, TIN_EXP, "45476b66", "ResNet18-200class-10ms-seed42")]
OUT = REPO / "analysis_outputs" / "paper" / "THRESHOLD_impact"
P = ("tg_h", "ab_h", "di_h", "as_h", "de_h")


def job(run):
    arch, seed, exp, prefix, name = run
    d = Path(glob.glob(str(EXP / exp / "lanes" / f"{prefix}*"))[0])
    lane = Lane(d, dataset_id="tiny_imagenet" if arch == "tiny" else "cifar100")
    thr = pd.read_csv(REPO / "analysis_outputs" / "paper" / "CENSUS_robust" / "tables" / "thresholds_per_run.csv")
    r = thr[thr.lane == name].iloc[0]
    sets = {"stored": (lane.thresholds, lane.ati),
            "alive": ({"ssi_abstraction_thresh": r.alive_thr_ssi, "csi_differentiation_thresh": r.alive_thr_csi,
                       "sai_task_general_thresh": r.alive_thr_sai}, lane.ati)}
    pi = lane.results["process_intensity"]
    assert pi[-1]["transition"].endswith("->terminal")
    trans = [t["transition"] for t in pi[:-1]]
    sel = lane.results["selectivity"]
    rows = []
    for L in lane.layers:
        codes = {}
        for k, t in enumerate(trans):
            a, b = t.split("->")
            for x in (a, b):
                if x not in codes:
                    codes[x] = lane.load_codes(x, L)
            md = match(codes[a], codes[b])
            for ts, (th, ati) in sets.items():
                c = classify_counts(codes[a], codes[b], md, sel[a][L], sel[b][L], thresholds=th, ati=ati)
                rows.append({"arch": arch, "seed": seed, "run": name, "tset": ts, "layer": L, "transition": k,
                             "label": t, **{p: int(c[p]) for p in P}})
            codes.pop(a, None)
    df = pd.DataFrame(rows)
    st = df[df.tset == "stored"].groupby("transition")[list(P)].sum()
    ok = all(int(st.loc[k, p]) == int(pi[k][p]) for k in range(len(trans)) for p in ("tg_h", "ab_h", "di_h"))
    print(name, "stored reproduced" if ok else "STORED MISMATCH", flush=True)
    df["stored_reproduced"] = ok
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    w = ap.parse_args().workers
    with ProcessPoolExecutor(w) as ex:
        dfs = list(ex.map(job, RUNS))
    pd.concat(dfs).to_csv(OUT / "core_events.csv", index=False)
    print("written")


if __name__ == "__main__":
    main()
