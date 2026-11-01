#!/usr/bin/env python3
"""Training cost and overfitting of the curriculum sweep vs the 30-epoch standard runs.

Reads per-epoch metrics (devtrain_metrics.json; switch_e15 also devtrain_batch_log.jsonl because
that run was resumed) and writes
  analysis_outputs/paper/CURRICULUM_sweep/tables/A_training_cost.csv
  analysis_outputs/paper/CURRICULUM_sweep/tables/A_std30_summary.csv
  analysis_outputs/paper/CURRICULUM_sweep/A_training_cost.json
Usage: python scripts/paper/curriculum_training_cost.py
"""
from __future__ import annotations

import csv
import json
import sqlite3
import statistics as st
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

REPO = Path(__file__).resolve().parents[2]
EXP = Path(RDT_DATA + "/core/experiments")
CUR = EXP / "7885dc97-1062-4b2f-9999-61d63b93d097"
STD30 = EXP / "0c28799c-ec9d-4e45-ac30-302dff810a8d"
OUT = REPO / "analysis_outputs" / "paper" / "CURRICULUM_sweep"
THRESH = [50.0, 55.0, 57.0, 58.0]
UPD_PER_EPOCH = 391


def names() -> Dict[str, str]:
    c = sqlite3.connect(f"file:{EXP.parent / 'rcx.db'}?immutable=1", uri=True)
    return {r[0]: r[1] for r in c.execute("select id, name from lanes")}


def condition(name: str) -> str:
    if "noise-standard" in name:
        return "standard30"
    c = name.split("constlr-")[1].rsplit("-seed", 1)[0]
    return "standard50" if c == "standard" else c


def switch_epoch(cond: str) -> int:
    return int(cond.split("_e")[1]) if cond.startswith("switch") else 0


def first_reach(fine: List[Dict[str, Any]], thr: float) -> Optional[int]:
    for i, r in enumerate(fine):
        if r["val_accuracy"] >= thr:
            return i + 1  # number of fine-class epochs completed
    return None


def main() -> None:
    nm = names()
    rows = []
    for exp, ok in ((CUR, lambda n: "constlr" in n), (STD30, lambda n: "noise-standard" in n)):
        for d in sorted((exp / "lanes").iterdir()):
            n = nm.get(d.name, "")
            if not ok(n):
                continue
            cond = condition(n)
            S = switch_epoch(cond)
            mdir = d / "dev_snapshots" / "metrics"
            m = json.load(open(mdir / "devtrain_metrics.json"))
            snaps = json.load(open(mdir / "devtrain_snapshots.json"))
            total_updates = snaps[-1]["weight_updates"]
            note = ""
            if cond == "switch_e15":
                # resumed run: the first attempt trained epochs 1-9, crashed; the resume restored the
                # epoch-5 state (weight_updates 1954) but labelled it epoch 10 and restarted the LR
                # cosine; 6 more superclass epochs (10-15) followed, then 35 fine epochs.
                note = ("resumed run: effective superclass epochs = 5 + 6 = 11 (LR restarted at resume, "
                        "superclass phase never annealed: lr 0.075 at switch); total updates 17985 (=46 epochs)")
            fine = [r for r in m if r["epoch"] > S]
            n_fine = len(fine)
            super_epochs_eff = round(total_updates / UPD_PER_EPOCH) - n_fine
            last = m[-1]
            row: Dict[str, Any] = {
                "condition": cond, "lane": n, "lane_id": d.name, "switch_epoch_nominal": S,
                "superclass_epochs_effective": super_epochs_eff, "fine_epochs": n_fine,
                "total_epochs_effective": super_epochs_eff + n_fine, "total_weight_updates": total_updates,
                "final_val_acc": last["val_accuracy"], "final_train_acc": last["train_accuracy"],
                "final_gap_pp": last["train_accuracy"] - last["val_accuracy"],
                "final_val_loss": last["val_loss"], "final_train_loss": last["train_loss"],
                "best_val_acc": max(r["val_accuracy"] for r in fine),
                "superclass_val_acc_at_switch": (m[[r["epoch"] for r in m].index(S)]["val_accuracy"]
                                                 if S and S in [r["epoch"] for r in m] else None),
                "note": note,
            }
            for t in THRESH:
                k = first_reach(fine, t)
                row[f"fine_epochs_to_{int(t)}"] = k
                row[f"total_epochs_to_{int(t)}"] = None if k is None else super_epochs_eff + k
                row[f"gap_at_first_{int(t)}"] = None if k is None else (
                    fine[k - 1]["train_accuracy"] - fine[k - 1]["val_accuracy"])
            # standard-50 at epoch 30 (mid-cosine; NOT a compute-matched model, reported for completeness)
            if cond == "standard50":
                e30 = [r for r in m if r["epoch"] == 30][0]
                row["note"] = (f"at epoch 30 (mid-schedule, lr {e30['learning_rate']:.3f}): val {e30['val_accuracy']:.2f}, "
                               f"train {e30['train_accuracy']:.2f}")
            rows.append(row)
    order = ["standard50", "switch_e05", "switch_e10", "switch_e15", "switch_e20", "switch_e25", "switch_e30",
             "standard30"]
    rows.sort(key=lambda r: (order.index(r["condition"]), r["lane"]))
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(OUT / "tables" / "A_training_cost.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    s30 = [r for r in rows if r["condition"] == "standard30"]
    summ: Dict[str, Any] = {"n": len(s30)}
    for k in ["final_val_acc", "final_train_acc", "final_gap_pp", "final_val_loss", "best_val_acc"] + \
             [f"{p}_to_{int(t)}" for t in THRESH for p in ("fine_epochs",)] + \
             [f"gap_at_first_{int(t)}" for t in THRESH]:
        v = [r[k] for r in s30 if r[k] is not None]
        summ[k] = {"mean": st.mean(v) if v else None, "sd": st.stdev(v) if len(v) > 1 else None,
                   "min": min(v) if v else None, "max": max(v) if v else None, "n_reached": len(v)}
    with open(OUT / "tables" / "A_std30_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "sd", "min", "max", "n_reached"])
        for k, v in summ.items():
            if isinstance(v, dict):
                w.writerow([k, v["mean"], v["sd"], v["min"], v["max"], v["n_reached"]])

    # matched comparisons (z vs standard-30 seed distribution)
    def z(x, k):
        return (x - summ[k]["mean"]) / summ[k]["sd"]
    matched = []
    for r in rows:
        if r["condition"].startswith("switch") or r["condition"] == "standard50":
            matched.append({"condition": r["condition"], "fine_epochs": r["fine_epochs"],
                            "total_epochs_effective": r["total_epochs_effective"],
                            "final_val_acc": r["final_val_acc"], "z_val_vs_std30": z(r["final_val_acc"], "final_val_acc"),
                            "final_gap_pp": r["final_gap_pp"], "z_gap_vs_std30": z(r["final_gap_pp"], "final_gap_pp"),
                            "fine_epoch_match_to_std30": ("exact (30 fine epochs, identical 30-epoch cosine)"
                                                          if r["fine_epochs"] == 30 else
                                                          f"mismatch ({r['fine_epochs']} vs 30 fine epochs; "
                                                          "no 20/25/35/40/45-epoch standard runs exist)")})
    json.dump({"rows": rows, "std30_summary": summ, "matched_vs_std30": matched},
              open(OUT / "A_training_cost.json", "w"), indent=1)
    for r in rows:
        print(f"{r['condition']:11s} sup={r['superclass_epochs_effective']:2d} fine={r['fine_epochs']:2d} "
              f"val={r['final_val_acc']:.2f} train={r['final_train_acc']:.2f} gap={r['final_gap_pp']:.2f} "
              + " ".join(f"to{int(t)}:{r[f'fine_epochs_to_{int(t)}']}/{r[f'total_epochs_to_{int(t)}']}" for t in THRESH)
              + " " + " ".join(f"g{int(t)}:{r[f'gap_at_first_{int(t)}']:.1f}" if r[f'gap_at_first_{int(t)}'] is not None else f"g{int(t)}:-" for t in THRESH))
    print(json.dumps({k: v for k, v in summ.items()}, indent=0)[:1500])
    for x in matched:
        print(x)


if __name__ == "__main__":
    main()
