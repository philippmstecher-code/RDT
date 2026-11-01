#!/usr/bin/env python3
"""Misplacement of class-selective features at equal fine-class accuracy (Supplementary Fig. 6 legend).
For each seed, the off-superclass share of class-selective features of a blocked arm at epochs 4-10 is compared with
the reference arm's share interpolated at the same validation accuracy (reference curve over its stored epochs),
averaged over epochs; mean and t-based 95% CI over five seeds.
Data: analysis_outputs/paper/RDT_completion/feature_routing_long.csv, analysis_outputs/paper/GRADJOINT/long.csv.
Usage: python scripts/paper/matched_acc_misplacement.py"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path(__file__).resolve().parents[2] / "analysis_outputs/paper"
R = pd.read_csv(RESULTS / "RDT_completion/feature_routing_long.csv")
A = pd.read_csv(RESULTS / "GRADJOINT/long.csv")


def matched(arm, ref, eps=range(4, 11)):
    out = []
    for s in sorted(R.seed.unique()):
        ra = R[(R.arm == arm) & (R.seed == s)].set_index("epoch").off_superclass_share
        rr = R[(R.arm == ref) & (R.seed == s)].set_index("epoch").off_superclass_share
        aa = A[(A.arm == arm) & (A.seed == s)].set_index("epoch").val_accuracy
        ar = A[(A.arm == ref) & (A.seed == s)].set_index("epoch").val_accuracy
        common = sorted(set(rr.index) & set(ar.index))
        x = np.array([ar[e] for e in common]); y = np.array([rr[e] for e in common]); o = np.argsort(x)
        d = [ra[e] - np.interp(aa[e], x[o], y[o]) for e in eps if e in ra.index and e in aa.index]
        if d:
            out.append(np.mean(d))
    return np.array(out)


for arm, ref in [("gradsc_l3", "gradjoint_l3"), ("gradsc", "sham"), ("gradsc", "gradjoint"), ("gradjoint", "sham")]:
    v = matched(arm, ref) * 100
    m = v.mean(); h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    print(f"{arm} vs {ref}: {m:+.1f} (95% CI {m - h:+.1f} to {m + h:+.1f}), {int((v > 0).sum())}/{len(v)} seeds positive")
