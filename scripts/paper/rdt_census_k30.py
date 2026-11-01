#!/usr/bin/env python3
"""RDT census of the 30-epoch pre-training arms (curriculum sc_true_K30, sc_rand_K30; seeds 7, 42, 137):
features selective for the true superclasses at every checkpoint, with the function that produced the 20-epoch census
(blind_pretrain_measures.census_lane = hierarchy_sae.main for one lane). Fig. 6c reads the switch (epoch 30).
Writes SWEEP_RDT/census_K30.csv.
Usage: PYTHONHASHSEED=0 .venv/bin/python scripts/paper/rdt_census_k30.py
"""
import sys
from pathlib import Path

import pandas as pd
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blind_pretrain_measures as BF  # noqa: E402

OUT = HERE.parents[1] / "analysis_outputs" / "paper" / "SWEEP_RDT"
LANES = Path(RDT_DATA + "/extended/experiments/curriculum/lanes")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    C = pd.concat([BF.census_lane(LANES / f"{a}_K30_seed{s}", f"{a}_K30", s)
                   for a in ("sc_true", "sc_rand") for s in BF.EL.SEEDS], ignore_index=True)
    C.to_csv(OUT / "census_K30.csv", index=False)
    print(C[(C.src == "sae") & (C.epoch == 30)].pivot(index="seed", columns="lane", values="census_genuine")
          .mul(100).round(1))


if __name__ == "__main__":
    main()
