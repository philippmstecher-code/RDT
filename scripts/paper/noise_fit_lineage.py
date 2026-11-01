#!/usr/bin/env python3
"""Lineage of noise-bent class-selective features (pooled layer, fixed-label noise; follows noise_fit_features.py).
For each transition epoch 5(m-1) -> 5m (m = 2..6), RDT feature matching of the stored validation codes (noise_null_lib.match,
stable and transformed matches = continuation). Each class-selective feature at epoch 5m is classified by its
predecessor: born (no predecessor), from a non-class-selective feature, from an unbent class-selective feature, or from
a bent one; 'bent' = noise-fit index below the 5th percentile of the clean null at that epoch (features.csv).
Also: change of the index along stable class-selective chains. Writes analysis_outputs/paper/RDT_noise_fit/
lineage.csv and lineage_summary.csv. Read-only; CPU.
Usage: python scripts/paper/noise_fit_lineage.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import noise_routing_nesting as N  # noqa: E402
from noise_null_lib import match  # noqa: E402
OUT = HERE.parents[1] / "analysis_outputs" / "paper" / "RDT_noise_fit"
F = pd.read_csv(OUT / "features.csv")
lo = F[F.arm == "clean"].groupby("epoch").nfi.quantile(0.05)
F["bent"] = F.nfi < F.epoch.map(lo)
key = {(r.arm, r.seed, r.epoch, r.feature): (r.bent, r.nfi) for r in F.itertuples()}
rows = []
for arm in ("clean", "within", "between"):
    for s in N.SEEDS:
        d = N.lane_dir(arm, s) / "sae_analysis" / "activations"
        for m in range(2, 7):
            md = match(np.load(d / str(m - 1) / "avgpool.npy"), np.load(d / str(m) / "avgpool.npy"))
            pred = {}
            for kname in ("stable", "transformed"):
                for t in md.get(kname, []):
                    pred[int(t[1])] = (int(t[0]), kname)
            for r in F[(F.arm == arm) & (F.seed == s) & (F.epoch == 5 * m)].itertuples():
                p = pred.get(r.feature)
                if p is None:
                    origin, pnfi = "born", np.nan
                else:
                    k = key.get((arm, s, 5 * (m - 1), p[0]))
                    if k is None:
                        origin, pnfi = "from non-class feature", np.nan
                    else:
                        origin, pnfi = ("from bent class feature" if k[0] else "from unbent class feature"), k[1]
                rows.append(dict(arm=arm, seed=s, epoch=5 * m, feature=r.feature, bent=bool(r.bent), nfi=r.nfi,
                                 origin=origin, pred_nfi=pnfi, match=None if p is None else p[1]))
            print(arm, s, m, flush=True)
D = pd.DataFrame(rows)
D.to_csv(OUT / "lineage.csv", index=False)
S = D.groupby(["arm", "seed", "epoch", "bent", "origin"]).size().rename("n").reset_index()
S.to_csv(OUT / "lineage_summary.csv", index=False)
