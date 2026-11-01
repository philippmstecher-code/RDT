#!/usr/bin/env python3
"""Fine-grained accuracy against superclass placement.

Tests whether networks with damaged coarse structure stay on the clean relation between placement and fine-class
accuracy (fine learning slowed through placement), whereas damage to fine-grained learning itself puts them below it.

Measures, per network and checkpoint, from the network's own classifier on the 10,000 CIFAR-100 test images:
  placement   share of images whose predicted class lies in their true superclass (%)
  fine        fine-class accuracy (%)
Reference: the clean trajectory of the same seed (placement -> fine, linear interpolation over its checkpoints).
Residual: fine(arm, e) - reference(placement(arm, e)); per seed averaged over a window; paired over seeds 7, 42, 137,
256, 314 (t-based 95% CI; seeds with the sign of the mean). Points outside the reference's placement range are dropped
(counted in the output).
Families and sources:
  block       sham, gradsc, gradrg, gradsc_l3 (coarse blocks), gradjoint, gradjoint_l3, gradenergy, gradrand (controls);
              epochs 0-15, stored per-image predictions (sample_predictions.pt) of the RDT lane copies
              (blocking_rdt, sham_perepoch); reference = sham. Windows 1-3, 4-10 (block), 11-15 (release).
              Validation: gradsc, gradrg, gradrand, gradenergy and sham against SHARED_signal/checkpoints_long.csv.
  noise       clean, within, between, matched; epochs 0, 5, ..., 30 (SHARED_signal/checkpoints_long.csv); reference =
              clean. Windows 5-10, 15-30.
  curriculum  true, rand (fine epochs 1, 5, 10; SHARED_signal); reference = sham of the same seed number (dense epochs
              0-15; scratch has no checkpoint before fine epoch 5), and scratch where its range allows.
Also reported: the fine deficit at matched epoch (arm - reference arm at the same epoch) and the share of it that the
placement deficit accounts for, 1 - residual / deficit.
Writes analysis_outputs/paper/PLACEMENT_curve/{long.csv, residuals.csv, summary.json}.
Usage: python scripts/paper/placement_curve.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis_outputs/paper/PLACEMENT_curve"
E = Path(RDT_DATA + "/extended/experiments")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from curriculum_probe import SUPER_OF  # noqa: E402

SEEDS = [7, 42, 137, 256, 314]
BLOCK_ARMS = ["sham", "gradsc", "gradrg", "gradsc_l3", "gradjoint", "gradjoint_l3", "gradenergy", "gradrand"]
COARSE = {"gradsc", "gradrg", "gradsc_l3", "between", "true", "rand"}


def lane(arm, s):
    return E / (f"sham_perepoch/lanes/sham_e00_seed{s}" if arm == "sham" else f"blocking_rdt/lanes/{arm}_seed{s}")


def pred_measures(d, e):
    p = torch.load(d / "dev_snapshots" / f"milestone_{e}" / "sample_predictions.pt", map_location="cpu",
                   weights_only=False)
    y = np.repeat(np.arange(100), [len(p[c]["predictions"]) for c in range(100)])
    pr = np.concatenate([p[c]["predictions"].numpy() for c in range(100)])
    return 100 * float((SUPER_OF[pr] == SUPER_OF[y]).mean()), 100 * float((pr == y).mean())


def tci(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) < 2:
        return {"mean": float(v.mean()) if len(v) else None, "n": len(v)}
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return {"mean": float(v.mean()), "ci95": [float(v.mean() - h), float(v.mean() + h)], "n": len(v),
            "n_pos": int((v > 0).sum()), "n_neg": int((v < 0).sum())}


def ref_fn(D, fam, ref, s):
    r = D[(D.family == fam) & (D.arm == ref) & (D.seed == s)].sort_values("placement")
    x, y = r.placement.values, r.fine.values
    return lambda q: float(np.interp(q, x, y)) if x.min() <= q <= x.max() else np.nan


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm in BLOCK_ARMS:
        for s in SEEDS:
            for e in range(0, 16):
                pl, fi = pred_measures(lane(arm, s), e)
                rows.append({"family": "block", "arm": arm, "seed": s, "epoch": e, "placement": pl, "fine": fi})
        print("predictions", arm, flush=True)
    M = pd.read_csv(REPO / "analysis_outputs/paper/SHARED_signal/checkpoints_long.csv")
    for _, r in M[M.family.isin(["noise", "curriculum"])].iterrows():
        rows.append({"family": r.family, "arm": r.arm, "seed": int(r.seed), "epoch": int(r.epoch),
                     "placement": 100 * (1 - r.misrouted_rate), "fine": 100 * r.acc})
    D = pd.DataFrame(rows)
    # validation against the mechanism table (same images, the network's own classifier)
    V = M[M.family == "block"].assign(placement=lambda d: 100 * (1 - d.misrouted_rate), fine=lambda d: 100 * d.acc)
    j = D[D.family == "block"].merge(V[["arm", "seed", "epoch", "placement", "fine"]], on=["arm", "seed", "epoch"],
                                     suffixes=("", "_m"))
    val = {"n_compared": len(j), "max_abs_placement": float((j.placement - j.placement_m).abs().max()),
           "max_abs_fine": float((j.fine - j.fine_m).abs().max())}
    print("validation", val, flush=True)
    # curriculum: also on the sham reference (dense early epochs), same seed number
    cur = D[D.family == "curriculum"].copy()
    D = pd.concat([D, cur.assign(family="curriculum_vs_sham")], ignore_index=True)
    sham = D[(D.family == "block") & (D.arm == "sham")].assign(family="curriculum_vs_sham")
    D = pd.concat([D, sham], ignore_index=True)
    D.to_csv(OUT / "long.csv", index=False)

    plan = [("block", "sham", a, w) for a in BLOCK_ARMS[1:] for w in ((1, 3), (4, 10), (11, 15))]
    plan += [("noise", "clean", a, w) for a in ("between", "within", "matched") for w in ((5, 10), (15, 30))]
    plan += [("curriculum", "scratch", a, (5, 10)) for a in ("true", "rand")]
    plan += [("curriculum_vs_sham", "sham", a, w) for a in ("true", "rand") for w in ((1, 1), (5, 10))]
    res, rows_r = {}, []
    for fam, ref, arm, (lo, hi) in plan:
        per_res, per_def, dropped = [], [], 0
        for s in SEEDS:
            f = ref_fn(D, fam, ref, s)
            a = D[(D.family == fam) & (D.arm == arm) & (D.seed == s) & D.epoch.between(lo, hi)]
            rr = [fi - f(pl) for pl, fi in zip(a.placement, a.fine)]
            dropped += int(np.isnan(rr).sum())
            rr = [x for x in rr if np.isfinite(x)]
            per_res.append(np.mean(rr) if rr else np.nan)
            b = D[(D.family == fam) & (D.arm == ref) & (D.seed == s)].set_index("epoch").fine
            dd = [fi - b[e] for e, fi in zip(a.epoch, a.fine) if e in b.index]
            per_def.append(np.mean(dd) if dd and fam != "curriculum_vs_sham" else np.nan)
        k = f"{fam}|{arm} vs {ref}|epochs {lo}-{hi}"
        res[k] = {"residual_at_matched_placement": tci(per_res), "deficit_at_matched_epoch": tci(per_def),
                  "points_outside_reference_range": dropped, "coarse_manipulation": arm in COARSE,
                  "per_seed_residual": [None if not np.isfinite(x) else round(float(x), 3) for x in per_res]}
        dm = res[k]["deficit_at_matched_epoch"].get("mean")
        rm = res[k]["residual_at_matched_placement"].get("mean")
        if dm and rm is not None and abs(dm) > 0.5:
            res[k]["share_of_deficit_from_placement"] = 1 - rm / dm
        rows_r.append({"key": k, "family": fam, "arm": arm, "window": f"{lo}-{hi}", **{f"res_{x}": y for x, y in
                       res[k]["residual_at_matched_placement"].items() if x != "ci95"},
                       "res_ci_lo": (res[k]["residual_at_matched_placement"].get("ci95") or [None, None])[0],
                       "res_ci_hi": (res[k]["residual_at_matched_placement"].get("ci95") or [None, None])[1],
                       "deficit_mean": dm, "share_from_placement": res[k].get("share_of_deficit_from_placement"),
                       "dropped": dropped})
    # paired interaction: coarse block minus its matched control, residuals
    for a, b, w in (("gradsc", "gradjoint", (4, 10)), ("gradsc", "gradjoint", (11, 15)), ("gradsc", "gradjoint", (1, 3)),
                    ("gradsc_l3", "gradjoint_l3", (4, 10)), ("gradsc_l3", "gradjoint_l3", (11, 15)),
                    ("gradrg", "gradjoint", (4, 10))):
        ra = res[f"block|{a} vs sham|epochs {w[0]}-{w[1]}"]["per_seed_residual"]
        rb = res[f"block|{b} vs sham|epochs {w[0]}-{w[1]}"]["per_seed_residual"]
        res[f"interaction|{a} - {b}|epochs {w[0]}-{w[1]}"] = tci(
            [x - y for x, y in zip(ra, rb) if x is not None and y is not None])
    R = pd.DataFrame(rows_r)
    R.to_csv(OUT / "residuals.csv", index=False)
    json.dump({"validation_vs_shared_signal": val, "results": res}, open(OUT / "summary.json", "w"), indent=1)
    pd.set_option("display.width", 220)
    print(R[["key", "res_mean", "res_ci_lo", "res_ci_hi", "res_n_pos", "res_n_neg", "deficit_mean",
             "share_from_placement", "dropped"]].round(2).to_string(index=False))
    for k, v in res.items():
        if k.startswith("interaction"):
            print(k, {x: (np.round(y, 2).tolist() if isinstance(y, (list, float)) else y) for x, y in v.items()})


if __name__ == "__main__":
    main()
