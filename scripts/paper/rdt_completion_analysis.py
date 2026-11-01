#!/usr/bin/env python3
"""RDT completion: events and census of the blocking arms at five seeds, against the jointly matched block, under
one common threshold set per seed (the sham's). Tested here: the superclass block against the jointly matched and the
energy-matched block, and the layer3-to-pooled pair; the other RDT completion analyses are in rdt_completion_*.py.

Reuses blocking_rdt_analysis.py (recount_lane, long_rows, lane_summary, paired); only the lane table, the seeds and
the contrasts differ. Lanes without a finished SAE run are skipped and listed.
"Holds": mean difference with t-based 95% CI excluding 0 and the sign in at least 4 of 5 seeds.

Usage: python scripts/paper/rdt_completion_analysis.py [--workers 3]
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blocking_rdt_analysis as B  # noqa: E402

OUT = B.REPO / "analysis_outputs" / "paper" / "RDT_completion"
SEEDS = [7, 42, 137, 256, 314]
B.LANES.update({"gradjoint": "blocking_rdt/lanes/gradjoint_seed{s}",
                "gradsc_l3": "blocking_rdt/lanes/gradsc_l3_seed{s}",
                "gradjoint_l3": "blocking_rdt/lanes/gradjoint_l3_seed{s}"})
B.LABEL.update({"gradjoint": "jointly matched", "gradsc_l3": "superclass-blocked, layer3 to pooled",
                "gradjoint_l3": "jointly matched, layer3 to pooled"})
ARMS = ["sham", "gradsc", "gradjoint", "gradenergy", "gradrand", "gradrg", "gradsc_l3", "gradjoint_l3"]
CONTRASTS = [("gradsc", "gradjoint"), ("gradsc", "gradenergy"), ("gradrg", "gradjoint"), ("gradsc", "sham"),
             ("gradjoint", "sham"), ("gradsc_l3", "gradjoint_l3")]
MEASURES = ["Ab-E sum", "Di-E sum", "Tg-E sum", "pct_sc mean", "pct_cls mean", "n_sc mean", "n_cls mean"]
WINDOWS = ("block", "release", "dose-matched (1-3)")


def holds(r, sign):
    """CI excludes 0 in the tested direction and >= 4 of 5 seeds share the sign."""
    if r.get("n", 0) < 5 or "ci95" not in r:
        return "NOT EVALUABLE (n < 5)"
    lo, hi = r["ci95"]
    ok_ci = hi < 0 if sign < 0 else lo > 0
    ok_seeds = (r["n_neg"] if sign < 0 else r["n_pos"]) >= 4
    return "HOLDS" if ok_ci and ok_seeds else "DOES NOT HOLD"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    common = {s: B.thr_set(B.load_lane(B.E / B.SHAM.format(s=s))) for s in SEEDS}
    jobs, missing = [], []
    for arm in ARMS:
        for s in SEEDS:
            d = B.E / B.LANES[arm].format(s=s)
            if not (d / ".sae_done").exists() or not (d / "sae_analysis" / "sae_results.json").exists():
                missing.append(f"{arm}_seed{s}")
                continue
            jobs.append({"arm": arm, "seed": s, "dir": str(d), "common": common[s], "max_epoch": B.MAX_EPOCH,
                         "out": str(OUT / "per_lane" / f"{arm}_seed{s}.json"), "force": False})
    print(f"{len(jobs)} lanes; missing: {missing}", flush=True)
    with mp.get_context("spawn").Pool(max(1, min(a.workers, len(jobs)))) as pool:
        for msg in pool.imap_unordered(B.recount_lane, jobs):
            print(msg, flush=True)
    R = [json.load(open(j["out"])) for j in jobs]
    D = pd.DataFrame([row for r in R for row in B.long_rows(r)])
    D.to_csv(OUT / "rdt_completion_long.csv", index=False)
    S = {(r["arm"], r["seed"], "common"): B.lane_summary(D, r["arm"], r["seed"], "common") for r in R}
    rows = []
    for x, y in CONTRASTS:
        for w in WINDOWS:
            for m in MEASURES:
                diff, rel = {}, {}
                for s in SEEDS:
                    p, q = S.get((x, s, "common")), S.get((y, s, "common"))
                    if p is None or q is None:
                        continue
                    diff[s] = p["windows"][w][m] - q["windows"][w][m]
                    rel[s] = 100 * (p["windows"][w][m] / q["windows"][w][m] - 1) if q["windows"][w][m] else float("nan")
                r = B.paired(diff)
                rows.append({"contrast": f"{x} - {y}", "window": w, "measure": m, "n": r.get("n"),
                             "mean": r.get("mean"), "ci_lo": (r.get("ci95") or [None, None])[0],
                             "ci_hi": (r.get("ci95") or [None, None])[1], "n_neg": r.get("n_neg"),
                             "n_pos": r.get("n_pos"),
                             "mean_pct_change": (sum(rel.values()) / len(rel)) if rel else None,
                             "per_seed": json.dumps(r.get("per_seed")), "_r": r})
    C = pd.DataFrame(rows)
    verdicts = {}
    for name, contrast in (("blocking vs jointly matched", "gradsc - gradjoint"), ("blocking vs energy-matched", "gradsc - gradenergy"),
                           ("layer3+ pair", "gradsc_l3 - gradjoint_l3")):
        for m in ("Ab-E sum", "pct_sc mean"):
            r = C[(C.contrast == contrast) & (C.window == "block") & (C.measure == m)].iloc[0]["_r"]
            verdicts[f"{name} | block | {m} < 0"] = holds(r, -1)
    C.drop(columns="_r").to_csv(OUT / "rdt_completion_contrasts.csv", index=False)
    val = {f"{r['arm']}_seed{r['seed']}": f"{sum(v['exact'] for v in r['validation'].values())}/{len(r['validation'])}"
           for r in R}
    json.dump({"lanes_missing": missing, "validation_own_recount_vs_stored": val, "verdicts": verdicts},
              open(OUT / "rdt_completion_summary.json", "w"), indent=1)
    print("validation:", val)
    for k, v in verdicts.items():
        print(f"{k}: {v}")
    show = C[C.window.isin(["block", "release"]) & C.measure.isin(["Ab-E sum", "Di-E sum", "pct_sc mean", "pct_cls mean"])]
    print(show.drop(columns=["_r", "per_seed"]).to_string(index=False))


if __name__ == "__main__":
    main()
