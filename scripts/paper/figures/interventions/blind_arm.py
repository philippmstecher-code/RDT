"""Loaders for the superclass-blind pre-training arm (blind_pretrain) in Fig. 6 and Extended Data Fig. 9.

The arm: 20 epochs on the five-way superclass-blind task (each label pools one class from every superclass), then 30
fine-class epochs, same protocol as sc_true_K20 / sc_rand_K20 (scripts/paper/queue_blind_pretrain.sh). A loader
returns None if its data are missing. Nothing here writes files.

Loader                source                                                         figure panels
  fine_curves()       lane training records (devtrain_metrics.json), only when all     Fig. 6b, d
                      five seeds have finished (.lane_done)
  placement_rows()    stored per-image test predictions (placement_curve.pred_measures, ED 9f
                      the function behind PLACEMENT_curve/long.csv), all five seeds
  confined_rows()     BLIND_PRETRAIN/long.csv, table 'rdt_census' (blind_pretrain_     Fig. 6c
                      analysis.rdt_rows: rdt_pace_curriculum.census, scratch thresholds)
  census_rows()       hierarchy_sae.py census (census_genuine, % of alive pooled SAE  Fig. 6a
                      features) for lane 'sc_blind_K20'; blind_pretrain_analysis does not
                      produce it. Read from BLIND_PRETRAIN/census_timecourse.csv (schema of
                      DICTFREE/hierarchy/D_sae_census_timecourse.csv) or from rows of that file
                      with lane 'sc_blind_K20'.
  events_rows()       common-threshold event counts (sae_outcomes -> SAE_outcomes/   Fig. 6 leads, ED 9g
                      events_EC.csv, tset 'common'); blind_pretrain_analysis does not
                      produce them. Read from BLIND_PRETRAIN/events_EC.csv (same schema) or
                      from rows of SAE_outcomes/events_EC.csv with arm 'sc_blind_K20'.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RESULTS = REPO / "analysis_outputs" / "paper"
OUT = RESULTS / "BLIND_PRETRAIN"
LANES = Path(RDT_DATA + "/extended/experiments/blind_pretrain/lanes")
SEEDS = [7, 42, 137, 256, 314]
K = 20
LANE_NAME = "sc_blind_K20"
LABEL = "pre-built, superclass-blind task"


def lane(s):
    return LANES / f"{LANE_NAME}_seed{s}"


def all_trained():
    return all((lane(s) / ".lane_done").exists() for s in SEEDS)


def _val(s):
    f = lane(s) / "dev_snapshots" / "metrics" / "devtrain_metrics.json"
    return {int(e["epoch"]): float(e["val_accuracy"]) for e in json.loads(f.read_text())}


def fine_curves():
    """{seed: fine-class validation accuracy at fine epochs 1..30}, or None unless all five seeds have finished."""
    if not all_trained():
        return None
    out = {}
    for s in SEEDS:
        c = _val(s)
        out[s] = np.array([c[K + e] for e in range(1, 31)])
    return out


def placement_rows():
    """Superclass accuracy (share of images predicted inside their true superclass) and fine-class accuracy from the
    network's own classifier at fine epochs 1, 5, ..., 30, as PLACEMENT_curve (placement_curve.pred_measures); columns
    family, arm, seed, epoch, placement, fine. None unless all five seeds have finished."""
    if not all_trained():
        return None
    sys.path.insert(0, str(REPO / "scripts" / "paper"))
    from placement_curve import pred_measures   # noqa: E402  (imports torch; only when the arm exists)
    rows = []
    for s in SEEDS:
        snaps = json.load(open(lane(s) / "dev_snapshots" / "metrics" / "devtrain_snapshots.json"))
        snaps = snaps.get("snapshots", snaps) if isinstance(snaps, dict) else snaps
        seen = set()
        for x in snaps:
            fe = int(x["epoch"]) - K
            if fe in (1, 5, 10, 15, 20, 25, 30) and fe not in seen:
                pl, fi = pred_measures(lane(s), int(x["milestone_index"]))
                rows.append(dict(family="curriculum", arm="blind", seed=s, epoch=fe, placement=pl, fine=fi))
                seen.add(fe)
    return pd.DataFrame(rows)


def confined_rows():
    """Class features confined to their superclass (count per checkpoint) at fine epochs 0, 1, 5, 10, 15; columns as
    RDT_pace/curriculum_long.csv with arm 'blind'. None until blind_pretrain_analysis.py has run."""
    f = OUT / "long.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, low_memory=False)
    d = d[(d.table == "rdt_census") & (d.arm == "blind")]
    return d[["arm", "seed", "fine_epoch", "n_cl_confined"]].astype({"seed": int, "fine_epoch": int}) if len(d) else None


def census_rows():
    """Superclass-selective share (census_genuine) over pre-training epochs 0-20 (sources in the module docstring)."""
    for f in (OUT / "census_timecourse.csv", RESULTS / "DICTFREE" / "hierarchy" / "D_sae_census_timecourse.csv"):
        if f.exists():
            d = pd.read_csv(f)
            d = d[(d.lane == LANE_NAME) & (d.src == "sae")]
            if len(d):
                return d
    return None


def events_rows():
    """Common-threshold event counts in the format of curriculum_lib.events_common (sources in the module docstring)."""
    for f in (OUT / "events_EC.csv", RESULTS / "SAE_outcomes" / "events_EC.csv"):
        if f.exists():
            e = pd.read_csv(f)
            e = e[(e.arm == LANE_NAME) & (e.tset == "common")]
            if len(e):
                e = e[e.epoch_b > e.epoch_a].copy()
                e["phase"] = np.where(e.epoch_b <= K, "pre", "fine")
                e = e.rename(columns={"tg_h_real": "tg_h", "ab_h_real": "ab_h", "di_h_real": "di_h"})
                return e[["arm", "seed", "transition", "epoch_a", "epoch_b", "phase", "tg_h", "ab_h", "di_h"]]
    return None


def status():
    n = sum((lane(s) / ".lane_done").exists() for s in SEEDS)
    return (f"blind arm: {n}/5 seeds trained; accuracy {'yes' if fine_curves() is not None else 'no'}; "
            f"confined {'yes' if confined_rows() is not None else 'no'}; census "
            f"{'yes' if census_rows() is not None else 'no'}; events {'yes' if events_rows() is not None else 'no'}")
