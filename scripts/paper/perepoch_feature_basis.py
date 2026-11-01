#!/usr/bin/env python3
"""Feature-basis process counts for the per-epoch ResNet-18 runs (sham_perepoch, seeds 7, 42, 137, epochs 0-15):
per epoch transition and SAE layer, the number of features carrying a Tg-E, Ab-E or Di-E label (each feature counted
once; matching_lib.Labeler with the run's stored thresholds on the pipeline match), next to the sample-weighted event counts
of Supplementary Fig. 1. Same procedure as the 'steps' of perepoch_cct_analysis.py (CCT-7 feature basis).
Validation: match counts equal the stored feature_matching counts; sample-weighted sums equal process_intensity.
Writes analysis_outputs/paper/SHAM_perepoch/feature_basis_steps.csv and feature_basis_summary.json.
Usage: python scripts/paper/perepoch_feature_basis.py
"""
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matching_lib import EPS_ACT, Labeler, Lane, feature_counts, outcome_from_pipeline, pipeline_match  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis_outputs/paper/SHAM_perepoch"
E = Path(RDT_DATA + "/extended/experiments/sham_perepoch/lanes")
SEEDS = (7, 42, 137)
EPOCHS = range(0, 16)


def com(x):
    x = np.asarray(x, float)
    return float((np.arange(len(x)) * x).sum() / x.sum())


def run_lane(seed):
    lane = Lane(E / f"sham_e00_seed{seed}")
    lab = Labeler(lane)
    labels = set(lane.labels)
    eps = [e for e in EPOCHS if str(e) in labels]
    rows = []
    for L in lane.layers:
        H = {e: np.load(lane.act_path(str(e), L), mmap_mode="r") for e in eps}
        sel = {e: lane.sel(str(e), L) for e in eps}
        for e in eps[:-1]:
            md = pipeline_match(H[e], H[e + 1])
            o = outcome_from_pipeline(md)
            labs = lab.labels(o, sel[e], sel[e + 1])
            w = lab.weighted(o, labs, np.abs(H[e]) > EPS_ACT, np.abs(H[e + 1]) > EPS_ACT)
            st = lane.stored_counts(f"{e}->{e + 1}", L)
            rows.append({"seed": seed, "layer": L, "epoch_a": e, "epoch_b": e + 1,
                         **{k: int(md[k]) for k in ("n_stable", "n_born", "n_died", "n_transformed")},
                         "counts_match_stored": bool(st) and all(int(st[k]) == int(md[k]) for k in
                                                                 ("n_stable", "n_born", "n_died", "n_transformed")),
                         **feature_counts(labs), **{f"w_{k}": v for k, v in w.items()}})
        print(seed, L, flush=True)
    return rows


def main():
    rows = []
    for s in SEEDS:
        rows += run_lane(s)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "feature_basis_steps.csv", index=False)
    res = {"counts_match_stored": f"{int(D.counts_match_stored.sum())}/{len(D)}"}
    for s, g in D.groupby("seed"):
        T = g.groupby("epoch_a")[["f_tg", "f_ab", "f_di", "w_tg_h", "w_ab_h", "w_di_h"]].sum()
        res[str(s)] = {"feature_com_tg_ab_di": [com(T[c]) for c in ("f_tg", "f_ab", "f_di")],
                       "event_com_tg_ab_di": [com(T[c]) for c in ("w_tg_h", "w_ab_h", "w_di_h")],
                       "feature_counts": T[["f_tg", "f_ab", "f_di"]].astype(int).values.tolist(),
                       "event_counts": T[["w_tg_h", "w_ab_h", "w_di_h"]].astype(int).values.tolist()}
        print(s, "feature COM", np.round(res[str(s)]["feature_com_tg_ab_di"], 2),
              "event COM", np.round(res[str(s)]["event_com_tg_ab_di"], 2), flush=True)
    print("counts match stored", res["counts_match_stored"])
    json.dump(res, open(OUT / "feature_basis_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
