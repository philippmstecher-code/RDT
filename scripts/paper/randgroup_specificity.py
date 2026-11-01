#!/usr/bin/env python3
"""Specificity of the suppression cost: superclass grouping vs the coarse structure that any grouping shares.

Part 1, training runs (last block, code projected out during epochs 1-10, then released; 5 seeds each): superclass
code, random-grouping code (one fixed grouping per seed, RandomState(10000 + seed)), random directions, each minus sham
over the first three epochs, from RANDGROUP_control/routing_decomposition.json
(the network's own classifier with the projection on, as Extended Data Fig. 6e): fine-class accuracy, superclass
assignment and accuracy within the correct superclass, and the share of the superclass code's coarse structure each arm
removes (coarse_overlap.py).
Part 2, dose-response in trained networks (no training): for each sham network (critical_sham sham_e00, 5 seeds) and
epoch (1, 2, 3, 5, 10), the fine-class accuracy lost when the code of each of 20 random groupings is removed from the
last block (PSEUDO_superclass/lastblock_raw.csv, centred removal) against the share of the superclass code's coarse
structure that grouping's code holds (coarse_overlap.csv, subspace removal_<p>, pooled layer); Spearman rho over the 20
groupings per network and epoch.
Writes analysis_outputs/paper/RANDGROUP_control/specificity_arms.csv, dose_response.csv, specificity_summary.json.
Usage: python scripts/paper/randgroup_specificity.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
RG = REPO / "analysis_outputs/paper/RANDGROUP_control"
P11 = REPO / "analysis_outputs/paper/PSEUDO_superclass/lastblock_raw.csv"
SEEDS = [7, 42, 137, 256, 314]


def tci(v):
    v = np.asarray(v, float)
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return [float(v.mean()), float(v.mean() - h), float(v.mean() + h)]


def main():
    R = json.load(open(RG / "routing_decomposition.json"))["per_arm"]
    rows = []
    for arm in ("superclass", "randgroup", "random"):
        for k, s in enumerate(SEEDS):
            rows.append({"arm": arm, "seed": s,
                         "d_fine_pp": R[arm]["d_fine_pp"]["per_seed"][k],
                         "d_assignment_pp": R[arm]["d_superclass_assignment_pp"]["per_seed"][k],
                         "d_within_pp": R[arm]["d_within_superclass_pp"]["per_seed"][k],
                         "coarse_removed_share": R[arm]["coarse_removed_share"]["per_seed"][k]})
    A = pd.DataFrame(rows)
    A.to_csv(RG / "specificity_arms.csv", index=False)
    res = {"arms_vs_sham_epochs_1_3": {}, "contrasts": {}}
    for arm, g in A.groupby("arm"):
        res["arms_vs_sham_epochs_1_3"][arm] = {k: tci(g[k]) for k in ("d_fine_pp", "d_assignment_pp", "d_within_pp",
                                                                      "coarse_removed_share")}
    W = A.pivot(index="seed", columns="arm")
    for k in ("d_fine_pp", "d_assignment_pp", "d_within_pp"):
        d = (W[k]["superclass"] - W[k]["randgroup"]).loc[SEEDS].values
        res["contrasts"][f"superclass - randgroup|{k}"] = {"mean_ci": tci(d), "n_negative": int((d < 0).sum())}
    res["randgroup_share_of_superclass_fine_cost"] = float(W["d_fine_pp"]["randgroup"].mean()
                                                           / W["d_fine_pp"]["superclass"].mean())

    # dose-response over the 20 groupings, in trained sham networks
    r = pd.read_csv(P11)
    r = r[(r.config == "lastblock") & (r.kind == "pseudo") & (r.window == "w01_10")]
    ov = pd.read_csv(RG / "coarse_overlap.csv")
    ov = ov[(ov.layer == "avgpool") & ov.subspace.str.startswith("removal_")].copy()
    ov["partition"] = ov.subspace.str.replace("removal_", "").astype(int)
    D = r.merge(ov[["seed", "epoch", "partition", "captured_coarse"]], on=["seed", "epoch", "partition"])
    D[["seed", "epoch", "partition", "cost_cen", "captured_coarse"]].to_csv(RG / "dose_response.csv", index=False)
    rho = []
    for (s, e), g in D.groupby(["seed", "epoch"]):
        assert len(g) == 20, (s, e, len(g))
        rr = stats.spearmanr(g.captured_coarse, g.cost_cen)
        rho.append({"seed": int(s), "epoch": int(e), "rho": float(rr[0]), "p": float(rr[1])})
    Rh = pd.DataFrame(rho)
    res["dose_response"] = {"n_cases": len(Rh), "n_rho_positive": int((Rh.rho > 0).sum()),
                            "median_rho": float(Rh.rho.median()), "per_case": rho,
                            "captured_share_by_epoch": D.groupby("epoch").captured_coarse.mean().round(3).to_dict()}
    json.dump(res, open(RG / "specificity_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "dose_response"}, indent=1))
    print("dose-response:", res["dose_response"]["n_rho_positive"], "/", res["dose_response"]["n_cases"],
          "median rho", round(res["dose_response"]["median_rho"], 3), res["dose_response"]["captured_share_by_epoch"])


if __name__ == "__main__":
    main()
