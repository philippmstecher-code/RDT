#!/usr/bin/env python3
"""Placement given: effect of the superclass block on fine learning when every image is placed correctly by construction.

Runs (placement_given; ResNet-18, CIFAR-100, branched from the epoch-0 checkpoints of the blocking runs,
epochs 1-15, block epochs 1-10 at layer1-4 and the pooled layer; training loss = cross-entropy over the five classes of
each image's TRUE superclass only, other logits -inf): no block (sham_e00), superclass block (gradsc), jointly matched
block (gradjoint, rank- and share-matched to the masked learning signal). Classification rules: `verdict` and
`final` below.

Measures per arm, seed, epoch (stored pooled activations of the 10,000 test images):
  oracle_ridge, oracle_logits   5-way accuracy within the true superclass (oracle_within.oracle)
  sc_probe                      fresh 20-way superclass ridge probe (curriculum_probe.probe), superclass decodability
  net_fine, net_assignment      100-way accuracy and superclass assignment of the network's own logits
Dose (lesion_log.jsonl): target (superclass-arm) and achieved learning-signal share, shadow and blocked rank per stage.

  --gate   checks on seed 42 (completion, learning, dose matching and informativeness; not the direction of the effect).
           Prints 'checks: criterion met' or 'checks: criterion not met' with reasons, writes gate.json, exit code
           0 / 1.
  default  final analysis over seeds 7, 42, 137, 256, 314: masked contrasts (superclass - jointly matched, each - no
           block), paired over seeds, mean, t-based 95% CI, seeds with the sign of the mean; interaction with the
           standard-regime contrasts (oracle_summary.json, same seeds); per-cell classification (`verdict`) and aggregation
           over cells; informativeness check over five seeds (pooled-layer share ratio masked / standard >= 0.5).
           Writes long.csv, dose.csv, summary.json, REPORT.md.
Usage: python scripts/paper/placement_given_analysis.py [--gate]
"""
import argparse
import json
import math
import os
import sys
from datetime import date
from pathlib import Path
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coarse_overlap import milestones  # noqa: E402
from oracle_within import oracle, tci  # noqa: E402
from curriculum_probe import SUPER_OF, load, probe  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis_outputs/paper/PLACEMENT_test"
E = Path(RDT_DATA + "/extended/experiments")
LANES = E / "placement_given/lanes"
SEEDS = [7, 42, 137, 256, 314]
PILOT = 42
L5 = "Llayer1-layer2-layer3-layer4-avgpool"
NAME = {"none": "sham_e00_seed{s}", "gradsc": "lesion_gradsc_e00_seed{s}_rp65_" + L5,
        "gradjoint": "lesion_gradjoint_e00_seed{s}_rp65_" + L5}
STD_LANE = {"gradsc_std": E / ("gradblock/lanes/lesion_gradsc_e00_seed{s}_rp65_" + L5),
            "gradjoint_std": E / ("gradjoint/lanes/lesion_gradjoint_e00_seed{s}_rp65_" + L5)}
WINDOWS = {"epochs 1-3": [1, 2, 3], "epochs 8-10": [8, 9, 10], "epochs 11-15": [11, 12, 13, 14, 15]}
EPOCHS = sorted({e for w in WINDOWS.values() for e in w})
LAST_EPOCH = 15
KEYS = ("oracle_ridge", "oracle_logits", "sc_probe", "net_fine", "net_assignment")
PRIMARY = ("oracle_ridge", "oracle_logits")
CONTRASTS = [("gradsc", "gradjoint"), ("gradsc", "none"), ("gradjoint", "none")]
CHANCE_GATE = 40.0           # oracle-within at epoch 10, no-block arm (2 x 20% chance)
SHARE_TOL_EARLY = 0.01       # |target - achieved| at layer3/layer4/avgpool, epochs 1-3 (standard-regime jointly matched runs)
SHORTFALL_TOL_BLOCK = 0.10   # avgpool target - achieved, epochs 1-10 (standard-regime runs 0.081-0.096)
INFORMATIVE_RATIO = 0.5
CLEAN_WINDOW = "epochs 1-3"                      # cleanly matched cells, read as primary
BOUND_WINDOWS = ["epochs 8-10", "epochs 11-15"]  # lighter control: one-sided bounds


def lane(arm, s):
    if arm in STD_LANE:
        return Path(str(STD_LANE[arm]).format(s=s))
    return LANES / NAME[arm].format(s=s)


def measure(ms_dir, cv):
    r = oracle(ms_dir, cv)
    X, y = load(ms_dir, "avgpool")
    r["sc_probe"] = 100 * float(probe(X.astype(np.float64), SUPER_OF[y]))
    return r


def rows(arms, seeds, cache):
    """Measures per (arm, seed, epoch), cached incrementally in `cache`."""
    old = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["arm", "seed", "epoch"])
    have = set(zip(old.arm, old.seed, old.epoch))
    out = old.to_dict("records")
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for arm in arms:
        for s in seeds:
            need = [e for e in EPOCHS if (arm, s, e) not in have]
            if not need:
                continue
            ms = milestones(lane(arm, s))
            for e in need:
                if e not in ms:
                    print("missing", arm, s, e, flush=True)
                    continue
                out.append({"arm": arm, "seed": s, "epoch": e, **measure(ms[e], cv)})
                print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in out[-1].items()}, flush=True)
                pd.DataFrame(out).to_csv(cache, index=False)
    return pd.DataFrame(out)


def dose_rows(ld, regime, s):
    f = ld / "dev_snapshots/metrics/lesion_log.jsonl"
    out = []
    if not f.exists():
        return out
    for line in open(f):
        d = json.loads(line)
        if d.get("event") != "reproject" or not isinstance(d.get("report"), dict):
            continue
        for st, r in d["report"].items():
            out.append({"regime": regime, "arm": d.get("mode"), "seed": s, "epoch": d["epoch"], "wu": d["weight_updates"],
                        "stage": st, "rank": r.get("total_rank"), "shadow_rank": r.get("shadow_rank"),
                        "target_share": r.get("target_grad_share"), "achieved_share": r.get("achieved_grad_share")})
    return out


def dose(seeds):
    R = []
    for s in seeds:
        R += dose_rows(lane("gradjoint", s), "masked", s)
        R += dose_rows(lane("gradsc", s), "masked", s)
        R += dose_rows(lane("gradjoint_std", s), "standard", s)
        R += dose_rows(lane("gradsc_std", s), "standard", s)
    return pd.DataFrame(R)


def wmean(D, regime, arm, stage, eps, col, s=None):
    m = D[(D.regime == regime) & (D.arm == arm) & (D.stage == stage) & D.epoch.isin(eps)]
    if s is not None:
        m = m[m.seed == s]
        return float(m[col].mean()) if len(m) else float("nan")
    return float(m.groupby("seed")[col].mean().mean()) if len(m) else float("nan")


def finite_log(ld):
    """True if every train/val loss and every logged batch loss is finite."""
    m = json.load(open(ld / "dev_snapshots/metrics/devtrain_metrics.json"))
    ok = all(math.isfinite(float(e[k])) for e in m for k in ("train_loss", "val_loss") if k in e)
    bl = ld / "dev_snapshots/metrics/devtrain_batch_log.jsonl"
    if bl.exists():
        for line in open(bl):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                return False  # 'NaN' is valid JSON for Python; malformed lines are suspicious
            v = d.get("batch_loss", d.get("loss"))
            if v is not None and not math.isfinite(float(v)):
                return False
    return ok


def gate():
    OUT.mkdir(parents=True, exist_ok=True)
    reasons, info = [], {}
    s = PILOT
    # 1 technical
    tech = True
    for arm in NAME:
        ld = lane(arm, s)
        mf = ld / "dev_snapshots/metrics/devtrain_metrics.json"
        done = (ld / ".devtrain_done").exists() and mf.exists() and \
            max(int(e["epoch"]) for e in json.load(open(mf))) >= LAST_EPOCH
        fin = done and finite_log(ld)
        info[f"technical|{arm}"] = {"completed": bool(done), "finite": bool(fin)}
        if not (done and fin):
            tech = False
            reasons.append(f"technical: {arm} completed={done} finite={fin}")
    if not tech:
        return finish(False, reasons, info)
    D = rows(list(NAME), [s], OUT / "long.csv")
    D = D[D.seed == s]
    # 2 masked training learns
    r10 = D[(D.arm == "none") & (D.epoch == 10)].iloc[0]
    info["learning|none|epoch10"] = {k: float(r10[k]) for k in KEYS}
    for k in PRIMARY:
        if not r10[k] >= CHANCE_GATE:
            reasons.append(f"learning: no-block {k} at epoch 10 = {r10[k]:.1f} < {CHANCE_GATE}")
    # 3 matching
    Q = dose([s])
    Q.to_csv(OUT / "dose_pilot.csv", index=False)
    J = Q[(Q.regime == "masked") & (Q.arm == "gradjoint") & Q.epoch.isin(range(1, 11))]
    jo = J[J.stage.isin(["layer3", "layer4", "avgpool"])]
    rank_ok = bool(len(jo)) and bool((jo["rank"] == jo["shadow_rank"]).all())
    info["matching|rank_equal_every_reestimation_layer3+"] = rank_ok
    info["matching|rank_equal_every_reestimation_all_stages"] = bool((J["rank"] == J["shadow_rank"]).all())
    if not rank_ok:
        reasons.append("matching: blocked rank != shadow rank at some re-estimation (layer3-pooled)")
    for st in ("layer3", "layer4", "avgpool"):
        t, a = (wmean(Q, "masked", "gradjoint", st, [1, 2, 3], c, s) for c in ("target_share", "achieved_share"))
        info[f"matching|{st}|epochs 1-3"] = {"target": t, "achieved": a}
        if not abs(t - a) <= SHARE_TOL_EARLY:
            reasons.append(f"matching: {st} epochs 1-3 target {t:.3f} vs achieved {a:.3f} (tol {SHARE_TOL_EARLY})")
    t, a = (wmean(Q, "masked", "gradjoint", "avgpool", range(1, 11), c, s) for c in ("target_share", "achieved_share"))
    info["matching|avgpool|epochs 1-10"] = {"target": t, "achieved": a}
    if not (t - a) <= SHORTFALL_TOL_BLOCK:
        reasons.append(f"matching: avgpool epochs 1-10 shortfall {t - a:.3f} > {SHORTFALL_TOL_BLOCK}")
    info["rank|avgpool|epochs 1-10"] = {
        "gradsc_masked": wmean(Q, "masked", "superclass", "avgpool", range(1, 11), "rank", s),
        "gradjoint_masked": wmean(Q, "masked", "gradjoint", "avgpool", range(1, 11), "rank", s),
        "gradsc_standard": wmean(Q, "standard", "superclass", "avgpool", range(1, 11), "rank", s),
        "gradjoint_standard": wmean(Q, "standard", "gradjoint", "avgpool", range(1, 11), "rank", s)}
    # 4 informativeness
    for w, eps in {"epochs 1-10": range(1, 11), "epochs 1-3": [1, 2, 3]}.items():
        for st in ("layer3", "layer4", "avgpool"):
            m = wmean(Q, "masked", "gradjoint", st, eps, "target_share", s)
            sd = wmean(Q, "standard", "gradjoint", st, eps, "target_share", s)
            info[f"informativeness|{st}|{w}"] = {"masked": m, "standard": sd, "ratio": m / sd}
    inf = info["informativeness|avgpool|epochs 1-10"]
    if not inf["ratio"] >= INFORMATIVE_RATIO:
        reasons.append(f"informativeness: pooled share masked {inf['masked']:.3f} / standard {inf['standard']:.3f} = "
                       f"{inf['ratio']:.2f} < {INFORMATIVE_RATIO}: uninformative for routing")
    # effect direction on seed 42: reported only
    eff = {}
    for w, eps in WINDOWS.items():
        M = D[D.epoch.isin(eps)].groupby("arm")[list(KEYS)].mean()
        for k in KEYS:
            eff[f"{w}|{k}"] = {"gradsc-gradjoint": float(M.loc["gradsc", k] - M.loc["gradjoint", k]),
                               "gradsc-none": float(M.loc["gradsc", k] - M.loc["none", k]),
                               "gradjoint-none": float(M.loc["gradjoint", k] - M.loc["none", k])}
    info["pilot_effect_not_gated"] = eff
    return finish(not reasons, reasons, info)


def finish(ok, reasons, info):
    rec = {"date": str(date.today()), "seed": PILOT, "pass": ok, "reasons": reasons, **info}
    json.dump(rec, open(OUT / "gate.json", "w"), indent=1, default=float)
    for k, v in info.items():
        if k != "pilot_effect_not_gated":
            print(f"  {k}: {v}")
    if "pilot_effect_not_gated" in info:
        print("  seed-42 oracle-within contrast (reported only, not part of the checks):")
        for k, v in info["pilot_effect_not_gated"].items():
            print(f"    {k:28s} " + "  ".join(f"{c} {x:+.2f}" for c, x in v.items()))
    print("checks: criterion met" if ok else "checks: criterion not met")
    for r in reasons:
        print("  reason:", r)
    sys.exit(0 if ok else 1)


def verdict(m, lo, hi, S):
    routing = lo <= 0 <= hi and abs(m) < abs(S) / 3
    shared = hi < 0 and m <= S / 2
    return "routing" if routing else "shared modes" if shared else "intermediate"


def final():
    OUT.mkdir(parents=True, exist_ok=True)
    D = rows(list(NAME) + list(STD_LANE), SEEDS, OUT / "long.csv")
    std = json.load(open(REPO / "analysis_outputs/paper/ORACLE_within/oracle_summary.json"))
    res, cells, lines = {}, {}, []
    for w, eps in WINDOWS.items():
        M = D[D.epoch.isin(eps)].groupby(["arm", "seed"])[list(KEYS)].mean()
        for k in KEYS:
            for a, b in CONTRASTS + [("gradsc_std", "gradjoint_std")]:
                d = (M.loc[a][k] - M.loc[b][k]).loc[SEEDS].values
                mci = tci(d)
                res[f"{a} - {b}|{w}|{k}"] = {"mean_ci": mci, "n_sign_of_mean": int((np.sign(d) == np.sign(mci[0])).sum()),
                                             "per_seed": np.round(d, 3).tolist(),
                                             "level_a": float(M.loc[a][k].loc[SEEDS].mean()),
                                             "level_b": float(M.loc[b][k].loc[SEEDS].mean())}
            if k in PRIMARY:
                key = f"jointly matched|{w}|{k}"
                sd = np.array(std[key]["per_seed"])
                S = std[key]["mean_ci"][0]
                dm = np.array(res[f"gradsc - gradjoint|{w}|{k}"]["per_seed"])
                m, lo, hi = res[f"gradsc - gradjoint|{w}|{k}"]["mean_ci"]
                inter = tci(dm - sd)
                v = verdict(m, lo, hi, S)
                cells[f"{w}|{k}"] = {"masked": [m, lo, hi], "standard": std[key]["mean_ci"], "interaction": inter,
                                     "interaction_per_seed": np.round(dm - sd, 3).tolist(),
                                     "n_sign_of_mean": res[f"gradsc - gradjoint|{w}|{k}"]["n_sign_of_mean"],
                                     "verdict": v}
    nr = sum(c["verdict"] == "routing" for c in cells.values())
    ns = sum(c["verdict"] == "shared modes" for c in cells.values())
    overall = "routing" if nr >= 4 and ns == 0 else "shared modes" if ns >= 4 and nr == 0 else "mixed"
    # The epochs 1-3 cells are the cleanly matched cells (primary); the clean-cell classification needs both cells for
    # one account and none for the other. The 6-cell classification above is reported as well.
    clean = {k: c for k, c in cells.items() if k.startswith(CLEAN_WINDOW + "|")}
    nr_c = sum(c["verdict"] == "routing" for c in clean.values())
    ns_c = sum(c["verdict"] == "shared modes" for c in clean.values())
    overall_clean = ("routing" if nr_c == len(clean) and ns_c == 0 else
                     "shared modes" if ns_c == len(clean) and nr_c == 0 else "mixed")
    Q = dose(SEEDS)
    Q.to_csv(OUT / "dose.csv", index=False)
    infm = {}
    for w, eps in {"epochs 1-10": range(1, 11), "epochs 1-3": [1, 2, 3]}.items():
        for st in ("layer3", "layer4", "avgpool"):
            per = {s: (wmean(Q, "masked", "gradjoint", st, eps, "target_share", s),
                       wmean(Q, "standard", "gradjoint", st, eps, "target_share", s)) for s in SEEDS}
            mm, ss = np.nanmean([p[0] for p in per.values()]), np.nanmean([p[1] for p in per.values()])
            infm[f"{st}|{w}"] = {"masked": mm, "standard": ss, "ratio": mm / ss,
                                 "per_seed_ratio": {s: p[0] / p[1] for s, p in per.items()}}
            for c in ("achieved_share", "rank"):
                infm[f"{st}|{w}"][f"masked_{c}"] = wmean(Q, "masked", "gradjoint", st, eps, c)
            infm[f"{st}|{w}"]["masked_gradsc_rank"] = wmean(Q, "masked", "superclass", st, eps, "rank")
    informative = infm["avgpool|epochs 1-10"]["ratio"] >= INFORMATIVE_RATIO
    summary = {"date": str(date.today()), "seeds": SEEDS, "informative": bool(informative),
               "overall": overall if informative else "uninformative for routing",
               "overall_clean_cells": overall_clean if informative else "uninformative for routing",
               "bound_windows": BOUND_WINDOWS, "cells": cells,
               "informativeness": infm, "contrasts": res}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=float)
    lines += [f"# Placement given: final analysis ({date.today()})", "",
              f"Seeds {SEEDS}; paired over seeds, mean, t-based 95% CI, seeds with the sign of the mean.", "",
              f"**Informativeness** (pooled-layer superclass-arm share, epochs 1-10): masked "
              f"{infm['avgpool|epochs 1-10']['masked']:.3f} vs standard {infm['avgpool|epochs 1-10']['standard']:.3f}, "
              f"ratio {infm['avgpool|epochs 1-10']['ratio']:.2f} -> {'informative' if informative else 'UNINFORMATIVE'}",
              "", f"**Overall (six-cell classification):** {summary['overall']} "
              f"({nr} routing cells, {ns} shared-modes cells of 6)", "",
              f"**Clean cells ({CLEAN_WINDOW}, ridge and logits; primary):** "
              f"{summary['overall_clean_cells']} ({nr_c} routing, {ns_c} shared modes of {len(clean)})", "",
              f"Windows {', '.join(BOUND_WINDOWS)}: jointly matched control lighter than its target (structural ceiling); "
              "the masked contrast there is a lower bound on the matched contrast (one-sided).", "",
              "| window | read-out | masked SC - joint | standard | interaction | seeds | outcome |", "|---|---|---|---|---|---|---|"]
    for key, c in cells.items():
        w, k = key.split("|")
        f3 = lambda x: f"{x[0]:+.2f} ({x[1]:+.2f} to {x[2]:+.2f})"
        lines.append(f"| {w} | {k} | {f3(c['masked'])} | {f3(c['standard'])} | {f3(c['interaction'])} | "
                     f"{c['n_sign_of_mean']}/5 | {c['verdict']}{' (bound)' if w in BOUND_WINDOWS else ''} |")
    lines += ["", "| contrast | window | measure | mean (95% CI) | seeds | levels |", "|---|---|---|---|---|---|"]
    for key, v in res.items():
        c, w, k = key.split("|")
        x = v["mean_ci"]
        lines.append(f"| {c} | {w} | {k} | {x[0]:+.2f} ({x[1]:+.2f} to {x[2]:+.2f}) | {v['n_sign_of_mean']}/5 | "
                     f"{v['level_a']:.1f} vs {v['level_b']:.1f} |")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true", help="checks on seed 42")
    a = ap.parse_args()
    gate() if a.gate else final()
