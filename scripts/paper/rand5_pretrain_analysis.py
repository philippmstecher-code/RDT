#!/usr/bin/env python3
"""Five-label controls (Extended Data Fig. 9b; Supplementary Note 4), with the functions that produced the same
measures for the other arms. ARM rand5: five random groups of 20 classes (rand5_pretrain); ARM sup5: five groups of
four whole superclasses (sup5_pretrain). Outputs go to RAND5_PRETRAIN/ (rand5) or SUP5_PRETRAIN/ (sup5).

census   features selective for the true superclasses over checkpoints: blind_pretrain_measures.census_lane
         (hierarchy_sae.main for one lane; seeds 7, 42, 137) -> RAND5_PRETRAIN/census_timecourse.csv
nesting  class-selective features confined to their superclass at every checkpoint: rdt_pace_curriculum.census with
         the seed's scratch thresholds (as rdt_pace_curriculum_full.py) -> RAND5_PRETRAIN/curriculum_full_long.csv
lead     fine-class validation accuracy per fine-class epoch (training record) and the epochs ahead of the same seed's
         curriculum scratch run (fig6_panels.lead_vs_drawn) -> RAND5_PRETRAIN/fine_curves.csv, summary.json
check    the same three computations on an existing arm (sc_rand_K20, seed 42) against the stored rows; writes nothing.
Usage: PYTHONHASHSEED=0 .venv/bin/python scripts/paper/rand5_pretrain_analysis.py {census,nesting,lead,all,check} [rand5|sup5]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lineage"))
import blind_pretrain_measures as BF  # noqa: E402
import rdt_pace_curriculum as R  # noqa: E402

RESULTS = HERE.parents[1] / "analysis_outputs" / "paper"
ARM = sys.argv[2] if len(sys.argv) > 2 else "rand5"
assert ARM in ("rand5", "sup5"), ARM
OUT = RESULTS / f"{ARM.upper()}_PRETRAIN"
EXPS = Path(RDT_DATA + "/extended/experiments")
LANE = lambda s: EXPS / f"{ARM}_pretrain/lanes/sc_{ARM}_K20_seed{s}"
SCRATCH = lambda s: EXPS / f"curriculum/lanes/scratch_E30_seed{s}"
SEEDS, K = [7, 42, 137, 256, 314], 20


def fine_val(d, off):
    rec = {int(e["epoch"]): float(e["val_accuracy"])
           for e in json.load(open(d / "dev_snapshots/metrics/devtrain_metrics.json"))}
    n = json.load(open(d / "_config/lane.json"))["epochs"] - off
    return np.array([rec[off + e] for e in range(1, n + 1)])


def lead_vs_drawn(v, scratch):
    y = np.maximum.accumulate(np.r_[1.0, scratch])
    return float(np.interp(v, y, np.arange(len(y)))) - 1


def nesting(lane, pre):
    rows = []
    for s in SEEDS:
        d, t = lane(s), R.thresholds(s)
        for m, ep in sorted(R.snaps(d).items()):
            if ep - pre > 30 or not (d / "sae_analysis" / "activations" / str(m)).exists():
                continue
            n_sc, n_cl, n_conf = R.census(d, m, t)
            rows.append({"arm": ARM, "seed": s, "epoch": ep - pre, "milestone": m, "n_sc": n_sc, "n_cl": n_cl,
                         "n_cl_confined": n_conf, "share_confined": n_conf / n_cl if n_cl else np.nan})
        BF.say(f"nesting seed {s}: {[(r['epoch'], r['n_cl'], r['n_cl_confined']) for r in rows if r['seed'] == s]}")
    return pd.DataFrame(rows)


def lead(lane):
    rows, ahead = [], []
    for s in SEEDS:
        y, sc = fine_val(lane(s), K), fine_val(SCRATCH(s), 0)
        ahead.append(lead_vs_drawn(y[0], sc))
        rows += [{"arm": ARM, "seed": s, "fine_epoch": e + 1, "val_acc": v} for e, v in enumerate(y)]
    return pd.DataFrame(rows), np.array(ahead)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "check":
        d = EXPS / "curriculum/lanes/sc_rand_K20_seed42"
        old = pd.read_csv(BF.STORED_CENSUS)
        old = old[(old.lane == "sc_rand_K20") & (old.seed == 42)].sort_values(["ms", "src"]).reset_index(drop=True)
        new = BF.census_lane(d, "sc_rand_K20", 42)[old.columns].sort_values(["ms", "src"]).reset_index(drop=True)
        num = old.select_dtypes("number").columns
        print("census sc_rand s42 matches stored:", np.allclose(old[num].values, new[num].values, equal_nan=True))
        full = pd.read_csv(RESULTS / "RDT_pace/curriculum_full_long.csv")
        ref = full[(full.arm == "rand") & (full.seed == 42)].set_index("epoch")
        t = R.thresholds(42)
        for m, ep in sorted(R.snaps(d).items())[:3]:
            got = R.census(d, m, t)
            print("nesting", ep - K, got, "stored", tuple(ref.loc[ep - K, ["n_sc", "n_cl", "n_cl_confined"]].astype(int)))
        y, sc = fine_val(d, K), fine_val(SCRATCH(42), 0)
        print("lead sc_rand s42 after one fine-class epoch:", round(lead_vs_drawn(y[0], sc), 4))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    if cmd in ("census", "all"):
        C = pd.concat([BF.census_lane(LANE(s), f"sc_{ARM}_K20", s) for s in BF.EL.SEEDS], ignore_index=True)
        C[pd.read_csv(BF.STORED_CENSUS, nrows=1).columns.tolist()].to_csv(OUT / "census_timecourse.csv", index=False)
        print(C[C.src == "sae"].pivot(index="seed", columns="epoch", values="census_genuine").mul(100).round(1))
    if cmd in ("nesting", "all"):
        D = nesting(LANE, K)
        D.to_csv(OUT / "curriculum_full_long.csv", index=False)
        print(D.groupby("epoch")[["n_cl", "n_cl_confined", "share_confined"]].mean().round(3))
    if cmd in ("lead", "all"):
        F, ahead = lead(LANE)
        F.to_csv(OUT / "fine_curves.csv", index=False)
        json.dump({"lead_after_one_fine_epoch": dict(zip(map(str, SEEDS), ahead.round(4).tolist())),
                   "mean": float(ahead.mean())}, open(OUT / "summary.json", "w"), indent=1)
        print("lead after one fine-class epoch:", ahead.round(2), "mean", round(float(ahead.mean()), 2))


if __name__ == "__main__":
    main()
