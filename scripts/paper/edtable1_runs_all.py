#!/usr/bin/env python3
"""Per-run values of Extended Data Table 1: the per-run table of the run scan (figures/edtable1_scan.py;
figures/final/source_data/edtable1_runs.csv, 174 rows: 164 table runs, the five fixed-label 'No noise' runs listed
again as 'Fine classes only, 30', and the five redrawn random-direction runs) plus the runs not in the scan: the
superclass-blind pre-training arm (blind_pretrain), the placement-given runs (placement_given), the task-general base
transplant runs, the two 30-epoch pre-training arms (Superclasses / Random grouping 30, fine 20) and the two five-label
pre-training arms (five random groups, rand5_pretrain; five groups of four whole superclasses, sup5_pretrain).
Footnote letters are dropped from condition names (the table assigns its own), and 'figure' is set to the table's
'Shown in' cell of each experiment. Final validation accuracy of the added runs from devtrain_metrics.json (last epoch).
Rows are then put in the order of the table, the runs it does not report are dropped, and each row gets the table's
section, purpose and condition label (198 runs, 203 rows). Purpose names the test or alternative of each group of rows.
Writes analysis_outputs/paper/figures/interventions/source_data/edtable1_runs_all.csv.
Usage: python scripts/paper/edtable1_runs_all.py"""
import json
import re
from pathlib import Path

import pandas as pd
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

REPO = Path(__file__).resolve().parents[2]
RUNS_EXTENDED = Path(RDT_DATA + "/extended/experiments")
SCAN_RUNS = REPO / "analysis_outputs/paper/figures/final/source_data/edtable1_runs.csv"
OUT = REPO / "analysis_outputs/paper/figures/interventions/source_data/edtable1_runs_all.csv"
SHOWN = {"Standard replication": "Figs 1–4; ED 1–6; Supp. Figs 4, 5", "Ablations": "ED 1c–f; Supp. Fig. 4",
         "Cross-dataset": "Fig. 2; ED 1, 2", "Per-epoch run": "Supp. Fig. 2", "Self-supervised": "ED 1h; Supp. Note 1",
         "Superclass-blind": "Supp. Note 4", "Suppression, last block": "ED 6c–g, 7g",
         "Label noise, redrawn": "ED 1, 5, 6, 8a,f; Supp. Fig. 3",
         "Label noise, fixed": "ED 1, 5, 6, 7g, 8; Supp. Fig. 3",
         "Sham control": "Fig. 5; ED 6, 7; Supp. Figs 1, 6",
         "Acquisition blocking": "Fig. 5; ED 7; Supp. Fig. 6; Supp. Methods",
         "Placement given": "Supp. Note 3", "Controlled curriculum": "Fig. 6; ED 9; Supp. Note 4",
         "Task-general base": "ED 2e,f",
         "Curriculum sweep": "Fig. 6b; ED 9e", "Five-label pre-training": "ED 9b; Supp. Note 4"}


def final_acc(lane):
    return json.load(open(lane / "dev_snapshots/metrics/devtrain_metrics.json"))[-1]["val_accuracy"]


d = pd.read_csv(SCAN_RUNS)
d["condition"] = d.condition.str.replace(r"[ʰ-˿ᵃ-ᶿⁱⁿ]+$", "", regex=True)   # superscript footnote letters
d.loc[d.condition == "Random, redrawn", "condition"] = "Random directions, redrawn"
d["figure"] = d.experiment.map(SHOWN).fillna(d.figure)
base = d[d.condition == "Superclasses 20, fine 30"].iloc[0]
rows = []
for s in (7, 42, 137, 256, 314):
    r = base.copy(); r["condition"] = "Superclass-blind 20, fine 30"; r["seed"] = s
    r["val_acc_final"] = round(final_acc(RUNS_EXTENDED / f"blind_pretrain/lanes/sc_blind_K20_seed{s}"), 2)
    rows.append(r)
lay = d[d.condition == "Superclass, layer3+"].iloc[0]
for cond, pat in (("Masked loss, coarse block", "lesion_gradsc_e00_seed{s}_*"),
                  ("Masked loss, matched block", "lesion_gradjoint_e00_seed{s}_*"),
                  ("Masked loss, no block", "sham_e00_seed{s}")):
    for s in (7, 42, 137, 256, 314):
        lane = next((RUNS_EXTENDED / "placement_given/lanes").glob(pat.format(s=s)))
        r = lay.copy(); r["experiment"] = "Placement given"; r["figure"] = SHOWN["Placement given"]
        r["condition"] = cond; r["seed"] = s; r["epochs"] = 15; r["branch_epoch"] = 0.0; r["checkpoints"] = 16
        r["checkpoint_epochs_completed"] = " ".join(map(str, range(16))); r["sae_expansion"] = "none"
        r["val_acc_final"] = round(final_acc(lane), 2)
        rows.append(r)
# task-general base: epoch-5 transplant of the frozen first and second stage, later stages trained 15 epochs
for arm, cond in (("B", "Transplant, no removal"), ("C", "Transplant, task-general removed"),
                  ("Crand", "Transplant, random removed")):
    for s in (7, 42, 137):
        lane = RUNS_EXTENDED / f"tgtransplant_e5/lanes/{arm}_seed{s}"
        r = lay.copy(); r["experiment"] = "Task-general base"; r["figure"] = SHOWN["Task-general base"]
        r["condition"] = cond; r["seed"] = s; r["epochs"] = 15; r["branch_epoch"] = 5.0; r["checkpoints"] = 10
        r["checkpoint_epochs_completed"] = ""; r["sae_expansion"] = "none"
        r["val_acc_final"] = round(final_acc(lane), 2)
        rows.append(r)
# five-label pre-training, protocol of the superclass-blind arm
for arm, exp, cond in (("rand5", "rand5_pretrain", "Five random groups 20, fine 30"),
                       ("sup5", "sup5_pretrain", "Five superclass groups 20, fine 30")):
    for s in (7, 42, 137, 256, 314):
        r = base.copy(); r["experiment"] = "Five-label pre-training"; r["figure"] = SHOWN["Five-label pre-training"]
        r["condition"] = cond; r["seed"] = s
        r["val_acc_final"] = round(final_acc(RUNS_EXTENDED / f"{exp}/lanes/sc_{arm}_K20_seed{s}"), 2)
        rows.append(r)
d = pd.concat([d, pd.DataFrame(rows)], ignore_index=True)

# Table order: rows follow the reading flow of each Results section, then the runs shown
# only in Extended Data or the SI. The equal-budget curriculum arms and the redrawn random-direction block are not
# reported. The 'Fine classes only, 30' runs are the fixed-label no-noise runs and keep their run numbers.
P = {"order": "Three-phase ordering (Figs 1–4; ED Figs 1–6)", "tin": "Second dataset (Fig. 2)",
     "sae": "Other SAEs (ED Fig. 1c)", "ssl": "No labels (Supp. Note 1)", "long": "Long training (ED Fig. 1c)",
     "tg": "Reliance on task-general features (ED Fig. 2e,f)", "pe": "Per epoch (Supp. Fig. 2)",
     "sup": "Use of superclass directions (ED Figs 6c–g, 7g)",
     "blk": "Specificity, asymmetry and nesting (Fig. 5; ED Fig. 7)", "l3": "Exactly matched damage (ED Fig. 7c)",
     "rg": "Random grouping (Fig. 5c)", "ctl": "Block control (ED Fig. 7)",
     "fix": "Second route, through the labels, p = 0.3, fixed (ED Fig. 8)",
     "red": "Second route, through the labels, redrawn every epoch (ED Fig. 8a,f; Supp. Fig. 3)",
     "pl": "Beyond assignment to superclasses (Supp. Note 3)",
     "cur": "Head start against equal pre-training, 20 + 30 epochs (Fig. 6; ED Fig. 9; Supp. Note 4)",
     "k30": "More coarse structure, larger head start (Fig. 6c)",
     "swp": "Head start and end point by pre-training length (Fig. 6b; ED Fig. 9e)",
     "lab": "Equal label count (ED Fig. 9b; Supp. Note 4)"}
ORDER = [  # (section, purpose key, experiment, condition, extra filter, table label)
    ("1–4", "order", "Standard replication", "Standard", lambda x: x.architecture == "ResNet-18", "Standard"),
    ("1–4", "order", "Standard replication", "Standard", lambda x: (x.architecture == "ViT-Small") & (x.seed != 256), "Standard"),
    ("1–4", "order", "Standard replication", "Standard", lambda x: (x.architecture == "ViT-Small") & (x.seed == 256), "Standard, 224 px"),
    ("1–4", "order", "Standard replication", "Standard", lambda x: x.architecture == "CCT-7", "Standard"),
    ("1–4", "tin", "Cross-dataset", "Tiny ImageNet, 200 classes", None, "200 classes"),
    ("1–4", "sae", "Ablations", "SAE expansion 8×", None, "SAE expansion 8×"),
    ("1–4", "sae", "Ablations", "Independent SAE seed", None, "Independent SAE seed"),
    ("1–4", "ssl", "Self-supervised", "SimCLR, no labels", None, "SimCLR"),
    ("1–4", "long", "Ablations", "200 epochs", None, "200 epochs"),
    ("1–4", "tg", "Task-general base", "Transplant, no removal", None, "Nothing removed"),
    ("1–4", "tg", "Task-general base", "Transplant, task-general removed", None, "Task-general removed"),
    ("1–4", "tg", "Task-general base", "Transplant, random removed", None, "Random removed"),
    ("1–4", "pe", "Per-epoch run", "Standard, 50 epochs", None, "Per-epoch checkpoints"),
    ("5", "sup", "Suppression, last block", "Superclass directions", None, "Superclass directions"),
    ("5", "sup", "Suppression, last block", "Random directions", None, "Random directions"),
    ("5", "sup", "Suppression, last block", "Random grouping", None, "Random grouping"),
    ("5", "blk", "Sham control", "No projection, epoch 0", None, "No block"),
    ("5", "blk", "Acquisition blocking", "Superclass directions", None, "Coarse block"),
    ("5", "blk", "Acquisition blocking", "Jointly matched", None, "Matched block"),
    ("5", "blk", "Acquisition blocking", "Energy-matched", None, "Energy-matched block"),
    ("5", "l3", "Acquisition blocking", "Superclass, layer3+", None, "Coarse block, layer3 on"),
    ("5", "l3", "Acquisition blocking", "Jointly matched, layer3+", None, "Matched block, layer3 on"),
    ("5", "rg", "Acquisition blocking", "Random grouping", None, "Random-grouping block"),
    ("5", "ctl", "Acquisition blocking", "Random directions", None, "Random-direction block"),
    ("5", "fix", "Label noise, fixed", "No noise", None, "No noise"),
    ("5", "fix", "Label noise, fixed", "Within-superclass, p = 0.3", None, "Within superclasses"),
    ("5", "fix", "Label noise, fixed", "Between-superclass, p = 0.3", None, "Across superclasses"),
    ("5", "fix", "Label noise, fixed", "Similarity-matched, p = 0.3", None, "Matched similarity"),
    ("5", "red", "Label noise, redrawn", "No noise", None, "No noise"),
    ("5", "red", "Label noise, redrawn", "Within-superclass, p = 0.1", None, "Within, p = 0.1"),
    ("5", "red", "Label noise, redrawn", "Within-superclass, p = 0.3", None, "Within, p = 0.3"),
    ("5", "red", "Label noise, redrawn", "Between-superclass, p = 0.1", None, "Across, p = 0.1"),
    ("5", "red", "Label noise, redrawn", "Between-superclass, p = 0.3", None, "Across, p = 0.3"),
    ("5", "red", "Label noise, redrawn", "Random, p = 0.1", None, "Random, p = 0.1"),
    ("5", "red", "Label noise, redrawn", "Random, p = 0.3", None, "Random, p = 0.3"),
    ("5", "pl", "Placement given", "Masked loss, coarse block", None, "Coarse block"),
    ("5", "pl", "Placement given", "Masked loss, matched block", None, "Matched block"),
    ("5", "pl", "Placement given", "Masked loss, no block", None, "No block"),
    ("6", "cur", "Controlled curriculum", "Superclasses 20, fine 30", None, "Superclasses 20 + 30"),
    ("6", "cur", "Controlled curriculum", "Random grouping 20, fine 30", None, "Random grouping 20 + 30"),
    ("6", "cur", "Controlled curriculum", "Superclass-blind 20, fine 30", None, "Blind task 20 + 30"),
    ("6", "cur", "Superclass-blind", "5-way within-superclass", None, "Blind task alone"),
    ("6", "cur", "Controlled curriculum", "Fine classes only, 30", None, "Fine classes only"),
    ("6", "k30", "Controlled curriculum", "Superclasses 30, fine 20", None, "Superclasses 30 + 20"),
    ("6", "k30", "Controlled curriculum", "Random grouping 30, fine 20", None, "Random grouping 30 + 20"),
] + [("6", "swp", "Curriculum sweep", f"{k} superclass epochs", None, f"{k} superclass epochs")
     for k in (0, 5, 10, 11, 20, 25, 30)] + [
    ("6", "lab", "Five-label pre-training", "Five random groups 20, fine 30", None, "Five random groups 20 + 30"),
    ("6", "lab", "Five-label pre-training", "Five superclass groups 20, fine 30", None, "Superclass groups 20 + 30")]
NOT_REPORTED = {"Fine classes only, 20", "Random directions, redrawn"}
assert d.condition.isin(NOT_REPORTED).sum() == 10
parts, n = [], 0
for sec, pk, exp, cond, filt, label in ORDER:
    g = d[(d.experiment == exp) & (d.condition == cond)]
    if filt is not None:
        g = g[filt(g)]
    assert len(g), (exp, cond)
    g = g.copy(); g.insert(0, "section", sec); g.insert(1, "purpose", P[pk]); g["condition"] = label
    if cond == "Fine classes only, 30":   # same networks as the fixed-label no-noise runs
        fix = parts[[i for i, x in enumerate(ORDER) if x[3] == "No noise" and x[2] == "Label noise, fixed"][0]]
        g["run"] = g.seed.map(dict(zip(fix.seed, fix.run))).values
    else:
        g["run"] = range(n + 1, n + len(g) + 1); n += len(g)
    if pk in ("l3", "blk", "rg", "ctl"):  # blocking arms: SAEs at every epoch 0-15 (ED Table 1 footnote d)
        g["sae_expansion"] = "4x"
    parts.append(g)
d = pd.concat(parts, ignore_index=True).drop(columns=["figure"])
d = d[["run"] + [c for c in d.columns if c != "run"]]
assert n == 198 and d.run.max() == 198 and len(d) == 203, (n, len(d))
OUT.parent.mkdir(parents=True, exist_ok=True)
d.to_csv(OUT, index=False)
print(len(d), "rows,", n, "runs;", d.groupby("section").size().to_dict())
