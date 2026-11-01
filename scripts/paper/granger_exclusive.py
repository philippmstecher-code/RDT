#!/usr/bin/env python3
"""Granger analysis of Extended Data Fig. 2 (panels a-d) with exclusive labels, per run and autocorrelation-aware.

Input: the raw per-sample transition files of the 9 core CIFAR-100 runs (read only), the same lanes and superclass map as
superclass_series.py / granger_causality_superclass.py. Per transition, layer, sample and event (born or
stable feature entry with its process labels):
  Ab_ex  'ab_h' in the labels and 'di_h' not        Di_ex  'di_h' in the labels and 'ab_h' not
  Ab_all / Di_all  the original (overlapping) counts, for reference; both  = events carrying both labels
Counts are summed per (run, transition, superclass); the duplicated last-checkpoint transition (k_to_terminal) is dropped.
Tests (Ab -> Di, and the reverse Di -> Ab):
  pooled   Di(t+1) ~ Di(t) + Di(t-1) + Ab(t) + run x superclass fixed effects; cluster-robust SE by run (9 clusters)
  per run  the same with superclass fixed effects; sign and p of Ab(t) per run
Writes analysis_outputs/paper/GRANGER_exclusive/{series_exclusive.csv, results.json}.
Usage: python scripts/paper/granger_exclusive.py
"""
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/granger/extractors"))
import granger_causality_superclass as gcs  # noqa: E402

OUT = ROOT / "analysis_outputs" / "paper" / "GRANGER_exclusive"
PROC = re.compile(r"'processes': \[([^\]]*)\]")


def extract(lane_id, exp, name):
    tdir = Path(gcs.EXT_BASE) / "experiments" / exp / "lanes" / lane_id / "sae_analysis" / "transitions"
    rows = []
    for tn in sorted(os.listdir(tdir), key=lambda n: int(n.split("_to_")[0])):
        if tn.endswith("_to_terminal") or not (tdir / tn).is_dir():
            continue
        t = int(tn.split("_to_")[0])
        c = defaultdict(lambda: defaultdict(int))
        for lf in os.listdir(tdir / tn):
            if not lf.endswith(".json"):
                continue
            for smp in json.load(open(tdir / tn / lf)):
                sc = gcs.FINE_TO_COARSE.get(smp["class_idx"])
                if sc is None:
                    continue
                for key in ("born", "stable"):
                    for p in PROC.findall(str(smp.get(key, ""))):
                        ab, di, tg = "'ab_h'" in p, "'di_h'" in p, "'tg_h'" in p
                        c[sc]["tg_all"] += tg
                        c[sc]["tg_ex"] += tg and not ab and not di
                        c[sc]["ab_all"] += ab
                        c[sc]["di_all"] += di
                        c[sc]["ab_ex"] += ab and not di
                        c[sc]["di_ex"] += di and not ab
                        c[sc]["both"] += ab and di
                pc = smp["process_counts"]
                pc = eval(pc) if isinstance(pc, str) else pc  # noqa: S307 (trusted local pipeline output)
                c[sc]["ab_pc"] += pc.get("ab_h", 0)
                c[sc]["di_pc"] += pc.get("di_h", 0)
                c[sc]["tg_pc"] += pc.get("tg_h", 0)
        for sc in range(20):
            rows.append({"lane": name, "transition": t, "sc": sc, **{k: c[sc][k] for k in
                         ("ab_all", "di_all", "ab_ex", "di_ex", "both", "ab_pc", "di_pc", "tg_all", "tg_ex", "tg_pc")}})
    return rows


def panel(df, a, d):
    rows = []
    for (ln, sc), g in df.groupby(["lane", "sc"]):
        g = g.sort_values("transition")
        A, D = g[a].values.astype(float), g[d].values.astype(float)
        for i in range(1, len(g) - 1):
            rows.append({"lane": ln, "sc": sc, "y": D[i + 1], "own1": D[i], "own0": D[i - 1], "x": A[i]})
    return pd.DataFrame(rows)


def fit(p, fe_cols, cluster=None):
    fe = pd.get_dummies(p[fe_cols].astype(str).agg("|".join, axis=1), drop_first=True).astype(float)
    X = sm.add_constant(pd.concat([p[["own1", "own0", "x"]].reset_index(drop=True), fe.reset_index(drop=True)], axis=1))
    y = p["y"].reset_index(drop=True)
    if cluster is not None:
        m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(p[cluster])[0]})
    else:
        m = sm.OLS(y, X).fit()
    return {"beta": float(m.params["x"]), "se": float(m.bse["x"]), "p": float(m.pvalues["x"]), "n": int(len(p))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "series_exclusive.csv"
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        con = sqlite3.connect(Path(gcs.EXT_BASE) / "rcx.db")
        lanes = con.execute("""SELECT l.id, l.name, e.id FROM lanes l JOIN experiments e ON l.experiment_id = e.id
                               WHERE e.name = 'Multi-Arch/-Seed' ORDER BY l.name""").fetchall()
        con.close()
        rows = []
        for lid, name, exp in lanes:
            print("extract", name, flush=True)
            rows += extract(lid, exp, name)
        df = pd.DataFrame(rows)
        df.to_csv(cache, index=False)
    res = {"n_runs": int(df.lane.nunique()),
           "share_events_with_both_labels": {
               "of_ab": float(df.both.sum() / max(df.ab_all.sum(), 1)),
               "of_di": float(df.both.sum() / max(df.di_all.sum(), 1))},
           "check_entry_counts_equal_process_counts": bool(
               (df.ab_all == df.ab_pc).all() and (df.di_all == df.di_pc).all())}
    for lab, (a, d) in {"Ab->Di overlapping": ("ab_all", "di_all"), "Ab->Di exclusive": ("ab_ex", "di_ex"),
                        "Di->Ab exclusive (reverse)": ("di_ex", "ab_ex")}.items():
        p = panel(df, a, d)
        pooled = fit(p, ["lane", "sc"], cluster="lane")
        per = {}
        for ln, g in p.groupby("lane"):
            per[ln] = fit(g, ["sc"])
        betas = np.array([v["beta"] for v in per.values()])
        res[lab] = {"pooled_cluster_by_run": pooled, "per_run": per,
                    "runs_beta_positive": int((betas > 0).sum()),
                    "runs_p_lt_0.05_positive": int(sum(v["beta"] > 0 and v["p"] < 0.05 for v in per.values()))}
        print(lab, json.dumps({k: v for k, v in res[lab].items() if k != "per_run"}), flush=True)
    json.dump(res, open(OUT / "results.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict) or "per_run" not in v}, indent=1))


if __name__ == "__main__":
    main()
