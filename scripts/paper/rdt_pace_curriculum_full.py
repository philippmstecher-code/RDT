#!/usr/bin/env python3
"""Pre-building, RDT read-out of class features over the whole run (Fig. 6e).

The census of rdt_pace_curriculum.py (same definitions, same per-seed scratch thresholds, census layers layer1-3 +
pooled; confined class feature = leave-own-class-out sibling enrichment above the 95th percentile under 40 random
groupings) at every stored checkpoint: pre-training epochs 0, 5, 10, 15, the switch and fine-class epochs 1, 5, ..., 30
of the true-superclass, random-grouping and superclass-blind lanes, and epochs 0-30 of scratch. Rows at the checkpoints
that rdt_pace_curriculum.py and blind_pretrain_analysis.py already computed (fine epochs 0, 1, 5, 10, 15) must match
them exactly; the script stops otherwise.
Writes analysis_outputs/paper/RDT_pace/curriculum_full_long.csv (epoch = epochs from the first fine-class label).
Usage: .venv/bin/python scripts/paper/rdt_pace_curriculum_full.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lineage"))
sys.path.insert(0, str(HERE))
import rdt_pace_curriculum as R  # noqa: E402

RESULTS = R.X.REPO / "analysis_outputs/paper"
BLIND = Path(RDT_DATA + "/extended/experiments/blind_pretrain/lanes")
LANES = {"true": (lambda s: R.L / f"sc_true_K20_seed{s}", 20), "rand": (lambda s: R.L / f"sc_rand_K20_seed{s}", 20),
         "blind": (lambda s: BLIND / f"sc_blind_K20_seed{s}", 20), "scratch": (lambda s: R.L / f"scratch_E30_seed{s}", 0)}


def main():
    rows = []
    for s in R.X.SEEDS:
        t = R.thresholds(s)
        for arm, (lane, pre) in LANES.items():
            d = lane(s)
            for m, ep in sorted(R.snaps(d).items()):
                if ep - pre > 30 or not (d / "sae_analysis" / "activations" / str(m)).exists():
                    continue
                n_sc, n_cl, n_conf = R.census(d, m, t)
                rows.append({"arm": arm, "seed": s, "epoch": ep - pre, "milestone": m, "n_sc": n_sc, "n_cl": n_cl,
                             "n_cl_confined": n_conf, "share_confined": n_conf / n_cl if n_cl else np.nan})
            print(arm, s, [(r["epoch"], r["n_cl"], r["n_cl_confined"]) for r in rows if r["arm"] == arm
                           and r["seed"] == s], flush=True)
    D = pd.DataFrame(rows)
    old = pd.read_csv(RESULTS / "RDT_pace" / "curriculum_long.csv")
    b = pd.read_csv(RESULTS / "BLIND_PRETRAIN" / "long.csv", low_memory=False)
    old = pd.concat([old, b[(b.table == "rdt_census") & (b.arm == "blind")][old.columns]], ignore_index=True)
    m = old.merge(D.rename(columns={"epoch": "fine_epoch"}), on=["arm", "seed", "fine_epoch"], suffixes=("_old", ""))
    assert len(m) == len(old), (len(m), len(old))
    for c in ("n_sc", "n_cl", "n_cl_confined"):
        assert (m[c + "_old"].astype(int) == m[c].astype(int)).all(), c
    print(f"matches the {len(m)} published rows")
    D.to_csv(RESULTS / "RDT_pace" / "curriculum_full_long.csv", index=False)
    print(D.groupby(["arm", "epoch"])[["n_cl", "n_cl_confined", "share_confined"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
