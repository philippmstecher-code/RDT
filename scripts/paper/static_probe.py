#!/usr/bin/env python3
"""Superclass (20-way) linear probe on the static-noise runs (clean / between_sc / within_sc, p = 0.3),
so that Extended Data Fig. 8b shows the coarse code and the fine-class cost from the same experiment.

Same read-out as curriculum_probe.py: ridge classifier on standardised avgpool activations of the stored
100 test images per class, five-fold stratified CV. Lanes without a finished final milestone are skipped.
Writes analysis_outputs/paper/STATIC_probe/static_probe.csv.

Usage: python scripts/paper/static_probe.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curriculum_probe import OUT as _OUT, SUPER_OF, load, probe  # noqa: E402

EXP = Path(RDT_DATA + "/extended/experiments")
OUT = _OUT.parent / "STATIC_probe"
ARMS = {"standard_resnet18": "clean", "between_sc_resnet18_throughout": "between_sc",
        "within_sc_resnet18_throughout": "within_sc"}


def main():
    rows = []
    for s in (7, 42, 137, 256, 314):
        for lane_name, arm in ARMS.items():
            lane = EXP / f"static_noise_s{s}" / "lanes" / lane_name / "dev_snapshots"
            f = lane / "metrics" / "devtrain_snapshots.json"
            if not f.exists():
                continue
            snaps = json.load(open(f))
            if not snaps or snaps[-1].get("epoch", 0) < 30:
                print(f"skip s{s} {arm}: unfinished", flush=True)
                continue
            for sn in snaps:
                X, y = load(lane / f"milestone_{sn['milestone_index']}", "avgpool")
                rows.append(dict(seed=s, arm=arm, epoch=sn["epoch"], super20=probe(X, SUPER_OF[y]),
                                 val_acc=sn["val_accuracy"]))
                print(rows[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop_duplicates(["seed", "arm", "epoch"]).to_csv(OUT / "static_probe.csv", index=False)


if __name__ == "__main__":
    main()
