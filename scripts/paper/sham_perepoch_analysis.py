#!/usr/bin/env python3
"""Process ordering at per-epoch resolution in clean runs (sham_perepoch, ResNet-18, seeds 7/42/137, epochs
0-15, one checkpoint per epoch). Stored per-run pipeline thresholds; counts = sae_results.json['process_intensity'],
all 5 layers (Fig. 1b/c conventions: C/R = (Tg-E + Ab-E) / max(Di-E, 1); shares of Tg+Ab+Di).
Per seed: shares and C/R per epoch transition; centre of mass (transition index, as Fig. 1 / census COM) of Tg-E,
Ab-E, Di-E; Tg-E largest at the first transition; C/R first vs last; transition at which C/R first falls below 1;
epoch of each process's peak share. Also the same quantities on the 5-epoch subsample (0, 5, 10, 15) from the per-epoch
counts summed within blocks (descriptive). Writes analysis_outputs/paper/SHAM_perepoch/results.json.
Usage: python scripts/paper/sham_perepoch_analysis.py
"""
import json
from pathlib import Path

import numpy as np
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

LANES = Path(RDT_DATA + "/extended/experiments/sham_perepoch/lanes")
OUT = Path(__file__).resolve().parents[2] / "analysis_outputs" / "paper" / "SHAM_perepoch"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [7, 42, 137]
P = ("tg_h", "ab_h", "di_h")


def summarise(X):
    """X: (n_transitions, 3) counts Tg, Ab, Di."""
    tot = X.sum(1, keepdims=True)
    sh = 100 * X / np.maximum(tot, 1)
    cr = (X[:, 0] + X[:, 1]) / np.maximum(X[:, 2], 1)
    t = np.arange(len(X), dtype=float)
    com = [float((t * X[:, j]).sum() / X[:, j].sum()) if X[:, j].sum() > 0 else np.nan for j in range(3)]
    below = np.flatnonzero(cr < 1)
    return {"shares": sh.round(1).tolist(), "cr": cr.round(3).tolist(), "com_tg_ab_di": [round(c, 2) for c in com],
            "order_tg_ab_di": bool(com[0] < com[1] < com[2]), "tg_first": bool(com[0] < min(com[1], com[2])),
            "tg_largest_first_transition": bool(X[0, 0] > max(X[0, 1], X[0, 2])),
            "cr_first": float(cr[0]), "cr_last": float(cr[-1]), "cr_declines": bool(cr[-1] < cr[0]),
            "first_transition_cr_below_1": int(below[0]) if len(below) else None,
            "peak_share_transition": [int(np.argmax(sh[:, j])) for j in range(3)]}


def main():
    res = {}
    for s in SEEDS:
        r = json.load(open(LANES / f"sham_e00_seed{s}" / "sae_analysis" / "sae_results.json"))
        pi = r["process_intensity"]
        trans = [t["transition"] for t in pi]
        assert trans == [f"{i}->{i + 1}" for i in range(15)], trans
        X = np.array([[t[p] for p in P] for t in pi], float)
        X5 = np.array([X[i:i + 5].sum(0) for i in (0, 5, 10)])
        res[s] = {"thresholds": {k: r["metadata"]["analysis_thresholds"][k] for k in
                                 ("sai_task_general_thresh", "ssi_abstraction_thresh", "csi_differentiation_thresh")},
                  "per_epoch": summarise(X), "five_epoch_blocks_from_per_epoch_counts": summarise(X5),
                  "counts": X.astype(int).tolist()}
        e = res[s]["per_epoch"]
        print(s, "order", e["order_tg_ab_di"], "Tg first", e["tg_first"], "Tg largest t0", e["tg_largest_first_transition"],
              "C/R", round(e["cr_first"], 2), "->", round(e["cr_last"], 2), "below1 from t", e["first_transition_cr_below_1"],
              "COM", e["com_tg_ab_di"], "peaks", e["peak_share_transition"], flush=True)
    res["summary"] = {k: sum(res[s]["per_epoch"][k] for s in SEEDS) for k in
                      ("order_tg_ab_di", "tg_first", "tg_largest_first_transition", "cr_declines")}
    res["summary"]["n"] = len(SEEDS)
    json.dump(res, open(OUT / "results.json", "w"), indent=1)
    print(json.dumps(res["summary"]))


if __name__ == "__main__":
    main()
