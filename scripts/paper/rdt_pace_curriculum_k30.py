#!/usr/bin/env python3
"""Pre-building, RDT nesting after 30 pre-training epochs (Fig. 6 legend e): the census of
rdt_pace_curriculum.py (same definitions, same per-seed scratch thresholds, census layers layer1-3 + pooled; confined
class feature = leave-own-class-out sibling enrichment above the 95th percentile under 40 random groupings) at every
stored checkpoint of the 30-epoch arms (curriculum sc_true_K30, sc_rand_K30; five seeds), so that Fig. 6 can compare
nesting at the switch after 20 and 30 pre-training epochs.
Writes analysis_outputs/paper/RDT_pace/curriculum_k30_long.csv (epoch = epochs from the first fine-class label).
Usage: .venv/bin/python scripts/paper/rdt_pace_curriculum_k30.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lineage"))
sys.path.insert(0, str(HERE))
import rdt_pace_curriculum as R  # noqa: E402

RESULTS = R.X.REPO / "analysis_outputs/paper"
PRE = 30


def main():
    rows = []
    for s in R.X.SEEDS:
        t = R.thresholds(s)
        for arm in ("true", "rand"):
            d = R.L / f"sc_{arm}_K{PRE}_seed{s}"
            for m, ep in sorted(R.snaps(d).items()):
                if ep - PRE > 20 or not (d / "sae_analysis" / "activations" / str(m)).exists():
                    continue
                n_sc, n_cl, n_conf = R.census(d, m, t)
                rows.append({"arm": arm, "pretrain": PRE, "seed": s, "epoch": ep - PRE, "milestone": m, "n_sc": n_sc,
                             "n_cl": n_cl, "n_cl_confined": n_conf, "share_confined": n_conf / n_cl if n_cl else np.nan})
            print(arm, s, [(r["epoch"], r["n_cl"], r["n_cl_confined"]) for r in rows if r["arm"] == arm
                           and r["seed"] == s], flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(RESULTS / "RDT_pace" / "curriculum_k30_long.csv", index=False)
    print(D.groupby(["arm", "epoch"])[["n_cl", "n_cl_confined", "share_confined"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
