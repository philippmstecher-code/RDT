#!/usr/bin/env python3
"""Is the missing coarse stage in task-irrelevant training (taskirr) explained by slower learning?

Compares task-irrelevant networks with standard training at matched learning progress, per seed (5 seeds):
  * progress = (task accuracy - chance) / (final task accuracy - chance); chance 20% (5-way) / 1% (100-way);
    epoch 0 = progress 0;
  * alternatively matched on the share of class-selective SAE features (avgpool), a task-agnostic progress measure.
Outcomes at matched progress: share of superclass-specific SAE features (exclusive rule, run-specific thresholds,
avgpool; from TASK_irrelevant/taskirr_census.csv) and true-superclass nesting AUC (avgpool; dictionary-free geometry: TASK_irrelevant/geo
for taskirr, DICTFREE/hierarchy and DATA_order/taskirr_share_test caches for the identical scratch_E30 = clean networks).
Standard training is linearly interpolated at the taskirr checkpoints whose progress lies within its trained range.
Writes analysis_outputs/paper/TASK_irrelevant/matched_progress.{json,csv}.

Usage: task_irrelevant_matched.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
A = REPO / "analysis_outputs" / "paper"
OUT = A / "TASK_irrelevant"
SEEDS = [7, 42, 137, 256, 314]
CHANCE = {"taskirr": 20.0, "clean": 1.0}


def geo(arm, s):
    if arm == "taskirr":
        f = OUT / "geo" / f"geo_taskirr_s{s}.csv"
    else:
        f = (A / "DICTFREE" / "hierarchy" / "cache" if s in (7, 42, 137) else A / "DATA_order" / "taskirr_share_test" / "cache") / \
            f"geo_scratch_E30_s{s}.csv"
    g = pd.read_csv(f)
    return g[g.layer == "avgpool"].set_index("epoch").true_auc


def traj(C, arm, s):
    q = C[(C.arm == arm) & (C.seed == s) & (C.part == "true") & (C.layer == "avgpool")].sort_values("epoch")
    q = q.set_index("epoch")
    acc = q.val_accuracy.copy()
    fin = acc.loc[30]
    prog = (acc - CHANCE[arm]) / (fin - CHANCE[arm])
    prog.loc[0] = 0.0
    return pd.DataFrame({"progress": prog, "ab": q.ab, "di": q.di, "auc": geo(arm, s).reindex(q.index)})


def paired(v):
    v = np.asarray(v, float)
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    h = stats.t.ppf(0.975, len(v) - 1) * se
    return {"mean": float(m), "ci": [float(m - h), float(m + h)], "p": float(stats.ttest_1samp(v, 0).pvalue),
            "n_neg": int((v < 0).sum()), "n": len(v), "per_seed": [float(x) for x in v]}


def main():
    C = pd.read_csv(OUT / "taskirr_census.csv")
    rows = []
    for s in SEEDS:
        t, c = traj(C, "taskirr", s), traj(C, "clean", s)
        ct = c[c.index > 0]  # trained checkpoints of standard training
        for ep, r in t[t.index > 0].iterrows():
            row = {"seed": s, "epoch": ep, "progress": r.progress, "ab_taskirr": r.ab, "di_taskirr": r.di,
                   "auc_taskirr": r.auc, "auc_taskirr_init": t.auc.loc[0], "auc_clean_init": c.auc.loc[0]}
            if ct.progress.min() <= r.progress <= ct.progress.max():
                o = ct.sort_values("progress")
                row["ab_clean_at_progress"] = float(np.interp(r.progress, o.progress, o.ab))
                row["auc_clean_at_progress"] = float(np.interp(r.progress, o.progress, o.auc))
            o = ct.sort_values("di")
            if o.di.min() <= r.di <= o.di.max():
                row["ab_clean_at_di"] = float(np.interp(r.di, o.di, o.ab))
            rows.append(row)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "matched_progress.csv", index=False)
    res = {"note": "matched-progress comparison of taskirr with standard training (same seeds)"}
    m = D.dropna(subset=["ab_clean_at_progress"])
    per = m.groupby("seed").apply(lambda g: pd.Series({"ab_t": g.ab_taskirr.mean(), "ab_c": g.ab_clean_at_progress.mean(),
                                                       "auc_t": g.auc_taskirr.mean(), "auc_c": g.auc_clean_at_progress.mean(),
                                                       "n_ck": len(g), "prog_lo": g.progress.min()}))
    res["matched_on_task_progress"] = {
        "checkpoints": "taskirr checkpoints with progress inside the trained range of standard training",
        "per_seed": per.reset_index().to_dict(orient="records"),
        "ab_taskirr_minus_clean": paired(per.ab_t - per.ab_c),
        "ab_ratio_mean": float(per.ab_t.mean() / per.ab_c.mean()),
        "auc_taskirr_minus_clean": paired(per.auc_t - per.auc_c)}
    m2 = D.dropna(subset=["ab_clean_at_di"])
    per2 = m2.groupby("seed").apply(lambda g: pd.Series({"ab_t": g.ab_taskirr.mean(), "ab_c": g.ab_clean_at_di.mean(),
                                                        "n_ck": len(g)}))
    res["matched_on_class_selective_share"] = {"per_seed": per2.reset_index().to_dict(orient="records"),
                                               "ab_taskirr_minus_clean": paired(per2.ab_t - per2.ab_c),
                                               "ab_ratio_mean": float(per2.ab_t.mean() / per2.ab_c.mean())}
    # nesting relative to initialisation over training (no matching needed)
    fin = D[D.epoch == 30].set_index("seed")
    res["auc_change_from_init_epoch30"] = {
        "taskirr": paired(fin.auc_taskirr - fin.auc_taskirr_init),
        "clean": paired([geo("clean", s).loc[30] - geo("clean", s).loc[0] for s in SEEDS])}
    json.dump(res, open(OUT / "matched_progress.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
