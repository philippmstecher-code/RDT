#!/usr/bin/env python3
"""RDT completion, pattern analyses. Three parts, each reusing an existing analysis:

routing   feature-level misrouting of class-selective features (blocking_feature_routing.measure: off-superclass
          share, own-direction ratio, count; pooled-layer SAE), five seeds, against the jointly matched block.
          Tests whether the extra class features under damage are compensating, misrouted features.
dose      per superclass, share of its learning signal blocked (GRADBLOCK/
          superclass_gradblock.csv, mean over epochs 1-10) against the loss of superclass-selective (and class-
          selective) features under the superclass block relative to sham (blocking_superclass_doseresponse.
          per_checkpoint, census layers, epochs 1-10); Spearman over the 20 superclasses per seed, Fisher-z pooled.
turnover  class-feature survival relative to overall turnover: the lineage hazard of class-selective features born
          in epochs 1-10 (lineage_per_lane) over each lane's overall death rate (n_died / (n_stable + n_died) over
          epochs 1-15, all layers; the event recount's matching counts, per_lane). Tests whether longer survival
          under the block is specific to class features or reflects slower turnover of all features.
Thresholds everywhere: one common set per seed, the sham's.
Usage: python scripts/paper/rdt_completion_patterns.py routing|dose|turnover [--arms a,b,c]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blocking_feature_routing as FR  # noqa: E402
import blocking_superclass_doseresponse as DR  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis_outputs/paper/RDT_completion"
SEEDS = [7, 42, 137, 256, 314]
LANES = {"sham": "sham_perepoch/lanes/sham_e00_seed{s}", "gradsc": "blocking_rdt/lanes/gradsc_seed{s}",
         "gradjoint": "blocking_rdt/lanes/gradjoint_seed{s}", "gradenergy": "blocking_rdt/lanes/gradenergy_seed{s}",
         "gradrand": "blocking_rdt/lanes/gradrand_seed{s}", "gradrg": "blocking_rdt/lanes/gradrg_seed{s}",
         "gradsc_l3": "blocking_rdt/lanes/gradsc_l3_seed{s}",
         "gradjoint_l3": "blocking_rdt/lanes/gradjoint_l3_seed{s}"}
CONTRASTS = [("gradsc", "gradjoint"), ("gradsc", "sham"), ("gradjoint", "sham"), ("gradsc", "gradenergy"),
             ("gradrg", "gradjoint"), ("gradsc_l3", "gradjoint_l3")]


def tci(d):
    d = np.asarray(d, float)
    if len(d) < 2:
        return [float(d.mean()) if len(d) else float("nan"), float("nan"), float("nan")]
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return [float(d.mean()), float(d.mean() - h), float(d.mean() + h)]


def paired(T, a, b, k):
    A, B = T.loc[a][k], T.loc[b][k]
    sd = sorted(set(A.index) & set(B.index))
    d = (A.loc[sd] - B.loc[sd]).values
    return {"mean_ci": tci(d), "n": len(sd), "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()),
            "per_seed": dict(zip(map(str, sd), np.round(d, 4).tolist()))}


def ready(arm, s):
    return (FR.E / LANES[arm].format(s=s) / ".sae_done").exists()


def routing(arms):
    rows = []
    for s in SEEDS:
        t = FR.thr_set(FR.load_lane(FR.E / LANES["sham"].format(s=s)))["thresholds"]
        for arm in arms:
            if not ready(arm, s):
                continue
            d = FR.E / LANES[arm].format(s=s)
            for e in FR.EPOCHS:
                rows.append({"arm": arm, "seed": s, "epoch": e, **FR.measure(d, e, t)})
            print("routing", arm, s, rows[-1], flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "feature_routing_long.csv", index=False)
    res = {}
    for w, eps in FR.WINDOWS.items():
        T = D[D.epoch.isin(list(eps))].groupby(["arm", "seed"]).mean(numeric_only=True)
        res[f"levels|{w}"] = T.groupby("arm").mean().round(4).to_dict()
        for a, b in CONTRASTS:
            if a in T.index.get_level_values(0) and b in T.index.get_level_values(0):
                for k in ("off_superclass_share", "own_direction_ratio", "n_class_selective"):
                    res[f"{w}|{a} - {b}|{k}"] = paired(T, a, b, k)
    json.dump(res, open(OUT / "feature_routing_summary.json", "w"), indent=1)
    return res


def dose(arms):
    share = pd.read_csv(REPO / "analysis_outputs/paper/GRADBLOCK/superclass_gradblock.csv")
    share = share[share.epoch.between(1, 10)].groupby(["seed", "superclass"]).blocked_share.mean()
    rows = []
    for s in SEEDS:
        t = DR.thr_set(DR.load_lane(DR.E / LANES["sham"].format(s=s)))["thresholds"]
        for arm in ["sham"] + [a for a in arms if a != "sham"]:
            if not ready(arm, s):
                continue
            d = DR.E / LANES[arm].format(s=s)
            for e in range(1, 11):
                mis, nsc, ncl = DR.per_checkpoint(d, e, t)
                for c in range(20):
                    rows.append({"arm": arm, "seed": s, "epoch": e, "superclass": c, "misassign": 100 * mis[c],
                                 "n_sc": nsc[c], "n_cls": ncl[c]})
            print("dose", arm, s, flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "dose_long.csv", index=False)
    A = D.groupby(["arm", "seed", "superclass"]).mean(numeric_only=True)
    res = {}
    for arm in [a for a in arms if a != "sham"]:
        per = []
        for s in SEEDS:
            if (arm, s) not in A.index.droplevel(2) or ("sham", s) not in A.index.droplevel(2):
                continue
            a, b = A.loc[(arm, s)], A.loc[("sham", s)]
            x = share.loc[s].values
            dsc = np.log((a.n_sc + 1) / (b.n_sc + 1)).values
            dcl = np.log((a.n_cls + 1) / (b.n_cls + 1)).values
            per.append({"seed": s, "rho_share_vs_dsc": float(stats.spearmanr(x, dsc)[0]),
                        "p_share_vs_dsc": float(stats.spearmanr(x, dsc)[1]),
                        "rho_share_vs_dcls": float(stats.spearmanr(x, dcl)[0]),
                        "p_share_vs_dcls": float(stats.spearmanr(x, dcl)[1])})
        z = lambda k: float(np.tanh(np.mean([np.arctanh(np.clip(p[k], -0.999, 0.999)) for p in per]))) if per else None
        res[arm] = {"per_seed": per, "pooled_rho_share_vs_dsc": z("rho_share_vs_dsc"),
                    "pooled_rho_share_vs_dcls": z("rho_share_vs_dcls"),
                    "n_seeds_neg_dsc": int(sum(p["rho_share_vs_dsc"] < 0 for p in per)), "n_seeds": len(per)}
    if "gradsc" in res and res["gradsc"]["n_seeds"] == 5:
        r = res["gradsc"]
        res["pattern verdict"] = "HOLDS" if r["pooled_rho_share_vs_dsc"] < 0 and r["n_seeds_neg_dsc"] >= 4 else "DOES NOT HOLD"
    json.dump(res, open(OUT / "dose_summary.json", "w"), indent=1)
    return res


def turnover(arms):
    rows = []
    for arm in arms:
        for s in SEEDS:
            fl, fr = OUT / "lineage_per_lane" / f"{arm}_seed{s}.json", OUT / "per_lane" / f"{arm}_seed{s}.json"
            if not (fl.exists() and fr.exists()):
                continue
            L, R = json.load(open(fl)), json.load(open(fr))
            st = sum(v["n_stable"] for k, m in R["matching"].items() for v in m.values())
            di = sum(v["n_died"] for k, m in R["matching"].items() for v in m.values())
            death = di / (st + di)
            rows.append({"arm": arm, "seed": s, "class_hazard": L["hazard"], "overall_death_rate": death,
                         "relative_hazard": L["hazard"] / death, "survival_fraction": L["survival_fraction"]})
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "turnover_long.csv", index=False)
    T = T.set_index(["arm", "seed"])
    res = {"levels": T.groupby("arm").mean().round(4).to_dict()}
    for a, b in CONTRASTS:
        if a in T.index.get_level_values(0) and b in T.index.get_level_values(0):
            for k in ("class_hazard", "overall_death_rate", "relative_hazard"):
                res[f"{a} - {b}|{k}"] = paired(T, a, b, k)
    json.dump(res, open(OUT / "turnover_summary.json", "w"), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["routing", "dose", "turnover"])
    ap.add_argument("--arms", default="sham,gradsc,gradjoint")
    a = ap.parse_args()
    arms = a.arms.split(",")
    res = {"routing": routing, "dose": dose, "turnover": turnover}[a.part](arms)
    for k, v in res.items():
        if isinstance(v, dict) and "mean_ci" in v:
            print(f"{k:72s} {v['mean_ci'][0]:+.4f} [{v['mean_ci'][1]:+.4f}, {v['mean_ci'][2]:+.4f}] "
                  f"neg {v['n_neg']} pos {v['n_pos']} / {v['n']}")
        else:
            print(k, json.dumps(v, default=str)[:400])


if __name__ == "__main__":
    main()
