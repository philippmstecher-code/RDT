#!/usr/bin/env python3
"""Critical-period tests of the superclass scaffold and the sham-arm outcomes.

  Critical-period contrast: per seed and window, excess Di-E (real - same-data null, clipped at 0 per transition, 'excess_clip' of
      sae_outcomes) summed over the transitions inside the suppression window, coarse and random; log ratio
      coarse/random. Contrast: log ratio window 1-10 - window 21-30, paired by seed. Criterion (label 'SUPPORTED'):
      mean > 0, the 95% CI (t, n = 5) excludes 0 and >= 4/5 seeds are > 0. Thresholds: sae_outcomes common thresholds with
      each seed's random arm of that window as reference.
  Further measures: the same contrast with window 11-20; behavioural lag (fine val accuracy, coarse - random,
      mean of the first three epochs of each window); SSI census coarse - random at the end of each window and at
      epoch 30 relative to delta; re-encoding (new superclass directions per re-estimation, lesion_log.jsonl);
      final accuracy per window.
  Lasting cost: fine accuracy coarse - sham at epoch 30 < 0, 95% CI excludes 0, >= 4/5 seeds negative (per window).
  Critical period: critical-period contrast criterion met and (coarse - sham @30, window 1-10) - (coarse - sham @30, window 11-20) < 0,
      95% CI excludes 0, >= 4/5 seeds negative.
  Non-neutral control: random - sham at epoch 30, participation ratio <= -4 and superclass variance share >= +1 pp,
      in >= 4/5 seeds.

Inputs: SAE_outcomes/{events,census,thresholds}_SUP (window 1-10), CRITICAL_period/{events,census,thresholds,integrity}_
CP11/CP21 (windows 11-20 and 21-30, same definitions as sae_outcomes), CRITICAL_period/anatomy_epoch30.csv (epoch-30
fine and superclass accuracy, participation ratio and superclass variance share per lane), devtrain_metrics.json and
lesion_log.jsonl of every lane. Writes CRITICAL_period/results.json and per-seed CSVs.

Usage: critical_period.py
"""
from __future__ import annotations

import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis_outputs" / "paper" / "CRITICAL_period"
A4OUT = REPO / "analysis_outputs" / "paper" / "SAE_outcomes"
RUNS_EXTENDED = Path(RDT_DATA + "/extended/experiments")
SEEDS = [7, 42, 137, 256, 314]
WINDOWS = {"1-10": 0, "11-20": 10, "21-30": 20}           # window -> branch (window start) epoch
EXP = {"1-10": ("SUP", A4OUT), "11-20": ("CP11", OUT), "21-30": ("CP21", OUT)}
SHAM = {"1-10": "sham_e00", "11-20": "sham_e10"}
TSETS = ("common", "run", "stored")
SITE = "avgpool"


# statistics (as sae_outcomes.paired / verdict_dir)

def paired(vals, name=""):
    seeds = sorted(s for s, v in vals.items() if v is not None and np.isfinite(v))
    d = np.array([vals[s] for s in seeds], dtype=np.float64)
    n = len(d)
    out = {"name": name, "n": n, "per_seed": {str(s): float(vals[s]) for s in seeds}}
    if n == 0:
        return out
    m = float(d.mean())
    out.update({"mean": m, "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()), "n_zero": int((d == 0).sum())})
    if n >= 2:
        sd = float(d.std(ddof=1))
        se = sd / math.sqrt(n)
        tq = float(stats.t.ppf(0.975, n - 1))
        out.update({"sd": sd, "se": se, "ci95": [m - tq * se, m + tq * se], "df": n - 1})
        if se > 0:
            out.update({"t": m / se, "p_two_sided": float(2 * stats.t.sf(abs(m / se), n - 1))})
    return out


def verdict_dir(r, sign, min_count=4):
    if r.get("n", 0) < 2 or "ci95" not in r:
        return "NOT SUPPORTED (insufficient seeds)"
    lo, hi = r["ci95"]
    ok_mean = r["mean"] * sign > 0
    ok_ci = lo > 0 if sign > 0 else hi < 0
    ok_cnt = (r["n_pos"] if sign > 0 else r["n_neg"]) >= min_count
    r["criteria"] = {"mean_direction": ok_mean, "ci_excludes_0": ok_ci, f"count_ge_{min_count}": ok_cnt}
    return "SUPPORTED" if (ok_mean and ok_ci and ok_cnt) else "NOT SUPPORTED"


def delta_class(r, delta):
    if "ci95" not in r:
        return "n/a"
    lo, hi = r["ci95"]
    if lo >= -delta and hi <= delta:
        return "within +/-delta"
    if hi < -delta:
        return "below -delta"
    if lo > delta:
        return "above +delta"
    return "inconclusive"


def diff(a, b):
    return {s: (None if a.get(s) is None or b.get(s) is None else a[s] - b[s]) for s in SEEDS}


# lanes

def lane_dir(window, arm, seed):
    if arm == "sham":
        return RUNS_EXTENDED / "critical_sham" / "lanes" / f"{SHAM[window]}_seed{seed}"
    c = {"coarse": "scaffold", "random": "random_variance"}[arm]
    e = WINDOWS[window]
    if window == "1-10":
        root = RUNS_EXTENDED / ("prevention_s42_copy" if seed == 42 else "prevention") / "lanes"
    else:
        root = RUNS_EXTENDED / "critical" / "lanes"
    return root / f"lesion_{c}_e{e:02d}_seed{seed}_rp65_Llayer4-avgpool"


def val_curve(d):
    m = json.load(open(d / "dev_snapshots" / "metrics" / "devtrain_metrics.json"))
    return {int(r["epoch"]): float(r["val_accuracy"]) for r in m}


def t0_val(d):
    for s in json.load(open(d / "dev_snapshots" / "metrics" / "devtrain_snapshots.json")):
        if str(s.get("snapshot_type", "")).startswith("t0"):
            return float(s["val_accuracy"])
    return float("nan")


def reencoding(d, w0):
    """reproject events inside the window (epochs w0+1..w0+10): directions added per re-estimation."""
    ev = [json.loads(x) for x in open(d / "dev_snapshots" / "metrics" / "lesion_log.jsonl")]
    rp = [e for e in ev if e.get("event") == "reproject" and w0 + 1 <= int(e["epoch"]) <= w0 + 10]
    out = {"n_reestimations": len(rp), "mode": rp[0].get("mode") if rp else None}
    for L in ("avgpool", "layer4"):
        na = np.array([e["report"][L]["n_added"] for e in rp], float)
        nc = np.array([e["report"][L]["n_candidate_dirs"] for e in rp], float)
        tr = np.array([e["report"][L]["total_rank"] for e in rp], float)
        mr = np.array([e["report"][L]["max_rank"] for e in rp], float)
        sat = np.where(tr >= mr)[0]
        out.update({f"{L}_mean_added": float(na.mean()) if na.size else float("nan"),
                    f"{L}_total_added": float(na.sum()),
                    f"{L}_mean_candidates": float(nc.mean()) if nc.size else float("nan"),
                    f"{L}_final_rank": float(tr[-1]) if tr.size else float("nan"),
                    f"{L}_max_rank": float(mr[-1]) if mr.size else float("nan"),
                    f"{L}_epoch_rank_saturated": float(rp[sat[0]]["epoch"]) if sat.size else float("nan"),
                    f"{L}_mean_added_first_epoch": float(na[[int(e["epoch"]) == w0 + 1 for e in rp]].mean())
                    if na.size else float("nan")})
    # remove event: last epoch with projection
    rem = [e for e in ev if e.get("event") == "remove"]
    out["remove_epoch"] = int(rem[0]["epoch"]) if rem else None
    return out


# SAE outputs

def load_sae(window):
    exp, root = EXP[window]
    E = pd.read_csv(root / f"events_{exp}.csv")
    C = pd.read_csv(root / f"census_{exp}.csv")
    T = json.load(open(root / f"thresholds_{exp}.json"))
    I = json.load(open(root / f"integrity_{exp}_summary.json"))
    b = WINDOWS[window]
    if exp != "SUP":     # critical lanes: epochs are relative to the branch checkpoint
        E["epoch_a"] = E["epoch_a"] + b
        E["epoch_b"] = E["epoch_b"] + b
        C["epoch"] = C["epoch"] + b
    return E, C, T, I


def events_sum(E, arm, seed, tset, w0, col="di_h_excess_clip"):
    q = E[(E.arm == arm) & (E.seed == seed) & (E.tset == tset) & (E.epoch_a >= w0) & (E.epoch_b <= w0 + 10)]
    return (float(q[col].sum()), int(len(q))) if len(q) else (None, 0)


def census_val(C, arm, seed, ep, metric="ssi", tset="common"):
    q = C[(C.arm == arm) & (C.seed == seed) & (C.tset == tset) & (C.layer == SITE) & (np.isclose(C.epoch, ep))]
    return float(q[metric].iloc[0]) if len(q) == 1 else None


# main

def main():
    res = {"seeds": SEEDS, "method_notes": [
               "Seed 42, window 1-10: prevention_s42_copy holds no checkpoint.pt; the fc weights for the epoch-30 "
               "outcomes are read (read-only) from the lane scaffold_lesion/lesion_*_e00_seed42_*, whose "
               "stored activations, predictions and metrics are byte-identical to the copy (SHA-256 checked).",
               "Epoch-30 'fine accuracy' is computed with the fc layer on the stored avgpool activations; it equals the "
               "logged val_accuracy to < 1e-12 pp in every lane.",
               "Behavioural lag and final accuracy use the logged per-epoch val_accuracy (devtrain_metrics.json), as in "
               "the window 1-10 analysis."], "concerns": []}
    sae = {w: load_sae(w) for w in WINDOWS}
    res["integrity"] = {w: {k: v for k, v in sae[w][3].items() if k != "fail_rows"} for w in WINDOWS}
    excluded = {w: sae[w][3].get("excluded_lanes", []) for w in WINDOWS}
    res["thresholds_common"] = {w: {s: v["thresholds"] for s, v in sae[w][2]["common"].items()} for w in WINDOWS}
    res["delta"] = {w: sae[w][2]["delta"] for w in WINDOWS}

    # 1. critical-period contrast: excess Di-E in the window, ratio, log ratio
    ev_rows = []
    logr = {ts: {w: {} for w in WINDOWS} for ts in TSETS}
    for w, w0 in WINDOWS.items():
        E = sae[w][0]
        for s in SEEDS:
            for ts in TSETS:
                row = {"window": w, "seed": s, "tset": ts}
                for arm in ("coarse", "random"):
                    for col, nm in (("di_h_excess_clip", "excess"), ("di_h_real", "real"), ("di_h_null", "null"),
                                    ("di_h_excess", "excess_unclipped")):
                        v, n = events_sum(E, arm, s, ts, w0, col)
                        row[f"{arm}_DiE_{nm}"] = v
                    row[f"{arm}_n_transitions"] = n
                c, r = row["coarse_DiE_excess"], row["random_DiE_excess"]
                ok = c is not None and r is not None and c > 0 and r > 0
                row["ratio"] = c / r if ok else None
                row["log_ratio"] = math.log(c / r) if ok else None
                if not ok and ts == "common":
                    res["method_notes"].append(f"log ratio undefined for window {w} seed {s} (coarse {c}, random {r})")
                logr[ts][w][s] = row["log_ratio"]
                ev_rows.append(row)
    EV = pd.DataFrame(ev_rows)
    EV.to_csv(OUT / "critical_period_excess_DiE_per_seed.csv", index=False)
    hcp = {}
    for ts in TSETS:
        L = logr[ts]
        prim = paired(diff(L["1-10"], L["21-30"]), "log(coarse/random excess Di-E), window 1-10 - window 21-30")
        prim["verdict"] = verdict_dir(prim, +1)
        sec = paired(diff(L["1-10"], L["11-20"]), "log(coarse/random excess Di-E), window 1-10 - window 11-20")
        sec["verdict"] = verdict_dir(sec, +1)
        desc = paired(diff(L["11-20"], L["21-30"]), "descriptive: window 11-20 - window 21-30")
        per_w = {w: paired(L[w], f"log ratio, window {w}") for w in WINDOWS}
        arms = {}
        for w in WINDOWS:
            q = EV[(EV.window == w) & (EV.tset == ts)].set_index("seed")
            for arm in ("coarse", "random"):
                arms[f"{w}_{arm}_excess_DiE"] = paired({s: q.loc[s, f"{arm}_DiE_excess"] for s in SEEDS})
            arms[f"{w}_coarse_minus_random_excess_DiE"] = paired(
                {s: q.loc[s, "coarse_DiE_excess"] - q.loc[s, "random_DiE_excess"] for s in SEEDS})
        hcp[ts] = {"primary_1_10_vs_21_30": prim, "secondary_1_10_vs_11_20": sec,
                   "descriptive_11_20_vs_21_30": desc, "log_ratio_per_window": per_w, "excess_DiE": arms}
    res["critical_period_test"] = {"rule": "log(coarse/random excess Di-E summed over the transitions inside the window); window 1-10 - "
                           "window 21-30, paired by seed; criterion: mean > 0, 95% CI (t, n=5) excludes 0, >= 4/5 seeds > 0",
                   "threshold_sets": "common = primary (each seed's random arm of the same window as reference); run, "
                                     "stored = sensitivity",
                   **hcp}
    # sensitivity analyses (descriptive; not used for the outcome labels)
    sens = {}
    q = EV[EV.tset == "common"]
    lr1 = {}
    for w in WINDOWS:
        g = q[q.window == w].set_index("seed")
        lr1[w] = {s: math.log((g.loc[s, "coarse_DiE_excess"] + 1) / (g.loc[s, "random_DiE_excess"] + 1)) for s in SEEDS}
    sens["pseudocount_plus1"] = {
        "primary_1_10_vs_21_30": paired(diff(lr1["1-10"], lr1["21-30"])),
        "secondary_1_10_vs_11_20": paired(diff(lr1["1-10"], lr1["11-20"])),
        "note": "log((coarse+1)/(random+1)) keeps the two window 11-20 seeds whose random-arm excess is 0"}
    # window 21-30 without the last transition (29->30), where the random arm loses ~4 pp of accuracy
    E21 = sae["21-30"][0]
    lr_no_last = {}
    for s in SEEDS:
        c = E21[(E21.arm == "coarse") & (E21.seed == s) & (E21.tset == "common") & (E21.epoch_a >= 20) & (E21.epoch_b <= 29)]
        r = E21[(E21.arm == "random") & (E21.seed == s) & (E21.tset == "common") & (E21.epoch_a >= 20) & (E21.epoch_b <= 29)]
        cs_, rs_ = c.di_h_excess_clip.sum(), r.di_h_excess_clip.sum()
        lr_no_last[s] = math.log(cs_ / rs_) if cs_ > 0 and rs_ > 0 else None
    sens["window_21_30_without_last_transition"] = {
        "log_ratio_21_30": paired(lr_no_last), "primary_1_10_vs_21_30": paired(diff(logr["common"]["1-10"], lr_no_last))}
    # absolute difference coarse - random of the excess (no ratio)
    absd = {w: {s: float(q[(q.window == w) & (q.seed == s)].coarse_DiE_excess.iloc[0]
                         - q[(q.window == w) & (q.seed == s)].random_DiE_excess.iloc[0]) for s in SEEDS} for w in WINDOWS}
    sens["coarse_minus_random_excess_DiE"] = {
        **{w: paired(absd[w]) for w in WINDOWS},
        "1_10_minus_21_30": paired(diff(absd["1-10"], absd["21-30"])),
        "1_10_minus_11_20": paired(diff(absd["1-10"], absd["11-20"]))}
    # one threshold per seed for all windows (window 1-10 common thresholds applied to the critical lanes)
    lr_sup = {"1-10": logr["common"]["1-10"]}
    rows_sup = []
    for w in ("11-20", "21-30"):
        f = OUT / f"events_{EXP[w][0]}_SUPthr.csv"
        if not f.exists():
            continue
        Eh = pd.read_csv(f)
        Eh["epoch_a"] += WINDOWS[w]
        Eh["epoch_b"] += WINDOWS[w]
        lr_sup[w] = {}
        for s in SEEDS:
            c, _ = events_sum(Eh, "coarse", s, "common", WINDOWS[w])
            r, _ = events_sum(Eh, "random", s, "common", WINDOWS[w])
            lr_sup[w][s] = math.log(c / r) if c and r and c > 0 and r > 0 else None
            rows_sup.append({"window": w, "seed": s, "coarse_DiE_excess_SUPthr": c, "random_DiE_excess_SUPthr": r,
                            "log_ratio_SUPthr": lr_sup[w][s]})
    if len(lr_sup) == 3:
        sens["SUP_thresholds_all_windows"] = {
            **{f"log_ratio_{w}": paired(lr_sup[w]) for w in WINDOWS},
            "primary_1_10_vs_21_30": paired(diff(lr_sup["1-10"], lr_sup["21-30"])),
            "secondary_1_10_vs_11_20": paired(diff(lr_sup["1-10"], lr_sup["11-20"]))}
        pd.DataFrame(rows_sup).to_csv(OUT / "critical_period_excess_DiE_SUPthresholds_per_seed.csv", index=False)
    res["critical_period_sensitivity"] = sens
    res["critical_period_test"]["verdict"] = hcp["common"]["primary_1_10_vs_21_30"]["verdict"]
    res["critical_period_test"]["secondary_11_20_verdict"] = hcp["common"]["secondary_1_10_vs_11_20"]["verdict"]
    critical_period_supported = res["critical_period_test"]["verdict"] == "SUPPORTED"

    # 2. behaviour: val curves, lag over the first three epochs of each window, final accuracy
    curves, beh = [], []
    V = {}
    for w, w0 in WINDOWS.items():
        for arm in ("coarse", "random", "sham"):
            if arm == "sham" and w not in SHAM:
                continue
            for s in SEEDS:
                d = lane_dir(w, arm, s)
                v = val_curve(d)
                v0 = t0_val(d)
                V[(w, arm, s)] = v
                curves.append({"window": w, "arm": arm, "seed": s, "epoch": w0, "val_accuracy": v0, "t0": True})
                curves += [{"window": w, "arm": arm, "seed": s, "epoch": e, "val_accuracy": a, "t0": False}
                           for e, a in sorted(v.items())]
    pd.DataFrame(curves).to_csv(OUT / "val_curves.csv", index=False)

    def lag(w, a1, a2, s):
        w0 = WINDOWS[w]
        x, y = V.get((w, a1, s)), V.get((w, a2, s))
        if x is None or y is None:
            return None
        return float(np.mean([x[e] - y[e] for e in (w0 + 1, w0 + 2, w0 + 3)]))

    def final(w, a1, a2, s):
        x, y = V.get((w, a1, s)), V.get((w, a2, s))
        return None if x is None or y is None else x[30] - y[30]

    behaviour = {}
    for w in WINDOWS:
        b = {"lag_e1_3_coarse_minus_random": paired({s: lag(w, "coarse", "random", s) for s in SEEDS}),
             "lag_e1_10_coarse_minus_random": paired({s: float(np.mean([V[(w, 'coarse', s)][e] - V[(w, 'random', s)][e]
                                                                        for e in range(WINDOWS[w] + 1, WINDOWS[w] + 11)]))
                                                     for s in SEEDS}),
             "final_acc_logged_coarse_minus_random": paired({s: final(w, "coarse", "random", s) for s in SEEDS})}
        if w in SHAM:
            b["lag_e1_3_coarse_minus_sham"] = paired({s: lag(w, "coarse", "sham", s) for s in SEEDS})
            b["lag_e1_3_random_minus_sham"] = paired({s: lag(w, "random", "sham", s) for s in SEEDS})
            b["final_acc_logged_coarse_minus_sham"] = paired({s: final(w, "coarse", "sham", s) for s in SEEDS})
            b["final_acc_logged_random_minus_sham"] = paired({s: final(w, "random", "sham", s) for s in SEEDS})
        # behavioural criterion of the window 1-10 analysis, applied descriptively per window: lag <= -1.0 pp,
        # p < 0.05, >= 4/5 negative
        r = b["lag_e1_3_coarse_minus_random"]
        b["SUP_criterion_descriptive"] = bool(r["mean"] <= -1.0 and r.get("p_two_sided", 1) < 0.05 and r["n_neg"] >= 4)
        behaviour[w] = b
        for s in SEEDS:
            row = {"window": w, "seed": s, "lag_e1_3_vs_random": lag(w, "coarse", "random", s),
                   "lag_e1_3_vs_sham": lag(w, "coarse", "sham", s) if w in SHAM else None,
                   "lag_e1_3_random_vs_sham": lag(w, "random", "sham", s) if w in SHAM else None}
            for arm in ("coarse", "random", "sham"):
                if (w, arm, s) in V:
                    w0 = WINDOWS[w]
                    for e in (w0 + 1, w0 + 2, w0 + 3, 30):
                        row[f"val_{arm}_e{e}"] = V[(w, arm, s)][e]
            beh.append(row)
    behaviour["lag_decline_1_10_minus_21_30"] = paired(
        diff({s: lag("1-10", "coarse", "random", s) for s in SEEDS},
             {s: lag("21-30", "coarse", "random", s) for s in SEEDS}),
        "descriptive: lag(1-10) - lag(21-30), coarse - random")
    behaviour["lag_vs_sham_1_10_minus_11_20"] = paired(
        diff({s: lag("1-10", "coarse", "sham", s) for s in SEEDS}, {s: lag("11-20", "coarse", "sham", s) for s in SEEDS}),
        "descriptive: lag vs sham (1-10) - (11-20)")
    res["behaviour"] = behaviour
    BEH = pd.DataFrame(beh)
    BEH.to_csv(OUT / "behaviour_per_seed.csv", index=False)

    # 3-5. sham-arm outcomes at epoch 30
    A = pd.read_csv(OUT / "anatomy_epoch30.csv")
    res["anatomy_fine_acc_vs_logged_max_abs_pp"] = float(A.fine_acc_minus_logged_pp.abs().max())

    def an(w, arm, s, m):
        q = A[(A.window == w) & (A.arm == arm) & (A.seed == s)]
        return float(q[m].iloc[0]) if len(q) == 1 else None

    scale = {"fine_acc": 100, "sc_acc": 100, "restricted_acc": 100, "sc_var_share": 100, "pr_dim": 1}
    lasting, an_rows = {}, []
    for w in ("1-10", "11-20"):
        blk = {}
        for m, k in scale.items():
            for a1, a2 in (("coarse", "sham"), ("random", "sham"), ("coarse", "random")):
                blk[f"{m}_{a1}_minus_{a2}"] = paired(
                    {s: (None if an(w, a1, s, m) is None or an(w, a2, s, m) is None
                         else k * (an(w, a1, s, m) - an(w, a2, s, m))) for s in SEEDS},
                    f"{m} @30 {a1} - {a2} ({'pp' if k == 100 else 'dims'})")
        lasting[w] = blk
        for s in SEEDS:
            row = {"window": w, "seed": s}
            for m, k in scale.items():
                for arm in ("coarse", "random", "sham"):
                    row[f"{m}_{arm}"] = an(w, arm, s, m)
                row[f"{m}_coarse_minus_sham"] = k * (an(w, "coarse", s, m) - an(w, "sham", s, m))
                row[f"{m}_random_minus_sham"] = k * (an(w, "random", s, m) - an(w, "sham", s, m))
                row[f"{m}_coarse_minus_random"] = k * (an(w, "coarse", s, m) - an(w, "random", s, m))
            an_rows.append(row)
    for s in SEEDS:   # window 21-30: final accuracy only (no lasting-cost outcome)
        row = {"window": "21-30", "seed": s}
        for m, k in scale.items():
            for arm in ("coarse", "random"):
                row[f"{m}_{arm}"] = an("21-30", arm, s, m)
            row[f"{m}_coarse_minus_random"] = k * (an("21-30", "coarse", s, m) - an("21-30", "random", s, m))
        an_rows.append(row)
    lasting["21-30"] = {f"{m}_coarse_minus_random": paired(
        {s: scale[m] * (an("21-30", "coarse", s, m) - an("21-30", "random", s, m)) for s in SEEDS},
        f"{m} @30 coarse - random (descriptive)") for m in scale}
    AN = pd.DataFrame(an_rows)
    AN.to_csv(OUT / "epoch30_per_seed.csv", index=False)

    lasting = {}
    for w in ("1-10", "11-20"):
        r = dict(lasting[w]["fine_acc_coarse_minus_sham"])
        r["verdict"] = verdict_dir(r, -1)
        lasting[w] = r
    res["lasting_cost_epochs1_10"] = {"rule": "fine accuracy coarse - sham at epoch 30 < 0, 95% CI (t, n=5) excludes 0, >= 4/5 "
                                      "seeds negative", "verdict_window_1_10": lasting["1-10"]["verdict"],
                              "verdict_window_11_20": lasting["11-20"]["verdict"], **lasting,
                              "random_minus_sham": {w: lasting[w]["fine_acc_random_minus_sham"] for w in ("1-10", "11-20")},
                              "coarse_minus_random": {w: lasting[w]["fine_acc_coarse_minus_random"] for w in ("1-10", "11-20")}}
    cs = {w: lasting[w]["fine_acc_coarse_minus_sham"]["per_seed"] for w in ("1-10", "11-20")}
    cp = paired({s: cs["1-10"][str(s)] - cs["11-20"][str(s)] for s in SEEDS if str(s) in cs["1-10"] and str(s) in cs["11-20"]},
                "fine acc @30 (coarse - sham), window 1-10 - window 11-20")
    cp_part = verdict_dir(cp, -1)
    cp["behavioural_part"] = cp_part
    res["critical_period_epochs1_10"] = {
        "rule": "critical-period contrast criterion met AND fine accuracy coarse - sham at epoch 30 more negative for window 1-10 than for 11-20 "
                "(paired, 95% CI excludes 0, >= 4/5 seeds)",
        "critical_period_supported": critical_period_supported, "behavioural_contrast": cp,
        "verdict": "SUPPORTED" if (critical_period_supported and cp_part == "SUPPORTED") else "NOT SUPPORTED"}
    nn = {}
    for w in ("1-10", "11-20"):
        per = {}
        for s in SEEDS:
            q = AN[(AN.window == w) & (AN.seed == s)].iloc[0]
            dpr, dsc = float(q["pr_dim_random_minus_sham"]), float(q["sc_var_share_random_minus_sham"])
            per[str(s)] = {"pr_dim_random_minus_sham": dpr, "sc_var_share_random_minus_sham_pp": dsc,
                           "pr_ok": dpr <= -4, "sc_ok": dsc >= 1.0, "both": bool(dpr <= -4 and dsc >= 1.0)}
        k = sum(v["both"] for v in per.values())
        nn[w] = {"per_seed": per, "n_seeds_both": k, "verdict": "NON-NEUTRAL" if k >= 4 else "NOT NON-NEUTRAL",
                 "pr_dim_random_minus_sham": lasting[w]["pr_dim_random_minus_sham"],
                 "sc_var_share_random_minus_sham_pp": lasting[w]["sc_var_share_random_minus_sham"]}
    res["non_neutral_control_epochs1_10"] = {"rule": "random - sham at epoch 30: participation ratio <= -4 AND superclass variance "
                                             "share >= +1 pp, in >= 4/5 seeds", **nn}
    res["epoch30_all_contrasts"] = lasting

    # 6. SSI census relative to delta
    cen, cen_rows = {}, []
    d_sup = sae["1-10"][2]["delta"]["ssi"]
    for w, w0 in WINDOWS.items():
        C = sae[w][1]
        delta = sae[w][2]["delta"]["ssi"]
        blk = {"delta_ssi": delta}
        for lab, ep in (("window_end", w0 + 10), ("epoch30", 30)):
            for ts in TSETS:
                r = paired({s: (None if census_val(C, "coarse", s, ep, "ssi", ts) is None else
                                census_val(C, "coarse", s, ep, "ssi", ts) - census_val(C, "random", s, ep, "ssi", ts))
                            for s in SEEDS}, f"SSI census @epoch {ep}, coarse - random ({ts})")
                r["epoch"] = ep
                r["delta_class"] = delta_class(r, delta)
                r["delta_class_with_SUP_delta"] = delta_class(r, d_sup)
                blk[f"{lab}_{ts}"] = r
        # CSI descriptive
        for lab, ep in (("window_end", w0 + 10), ("epoch30", 30)):
            blk[f"{lab}_csi_common"] = paired({s: (None if census_val(C, "coarse", s, ep, "csi") is None else
                                                   census_val(C, "coarse", s, ep, "csi") - census_val(C, "random", s, ep, "csi"))
                                               for s in SEEDS}, f"CSI census @epoch {ep}, coarse - random")
        cen[w] = blk
        q = C[(C.tset == "common") & (C.layer == SITE)]
        for r_ in q.itertuples():
            cen_rows.append({"window": w, "arm": r_.arm, "seed": r_.seed, "epoch": r_.epoch, "ssi": r_.ssi,
                             "csi": r_.csi, "sai": r_.sai, "n_alive": r_.n_alive})
    res["ssi_census"] = cen
    pd.DataFrame(cen_rows).to_csv(OUT / "census_by_epoch_common_avgpool.csv", index=False)

    # 6b. re-encoding
    re_rows = []
    for w, w0 in WINDOWS.items():
        for arm in ("coarse", "random"):
            for s in SEEDS:
                re_rows.append({"window": w, "arm": arm, "seed": s, **reencoding(lane_dir(w, arm, s), w0)})
    RE = pd.DataFrame(re_rows)
    RE.to_csv(OUT / "reencoding_per_seed.csv", index=False)
    reenc = {}
    for w in WINDOWS:
        q = RE[RE.window == w]
        c = q[q.arm == "coarse"].set_index("seed")
        r = q[q.arm == "random"].set_index("seed")
        reenc[w] = {"coarse_mean_new_superclass_dirs_per_reestimation_avgpool": paired(c["avgpool_mean_added"].to_dict()),
                    "coarse_mean_new_superclass_dirs_per_reestimation_layer4": paired(c["layer4_mean_added"].to_dict()),
                    "random_mean_dirs_per_reestimation_avgpool": paired(r["avgpool_mean_added"].to_dict()),
                    "coarse_minus_random_avgpool": paired((c["avgpool_mean_added"] - r["avgpool_mean_added"]).to_dict()),
                    "coarse_first_epoch_avgpool": paired(c["avgpool_mean_added_first_epoch"].to_dict()),
                    "n_reestimations": {str(k): int(v) for k, v in c["n_reestimations"].items()},
                    "coarse_epoch_rank_saturated_avgpool": {str(k): v for k, v in c["avgpool_epoch_rank_saturated"].items()}}
    reenc["1_10_minus_21_30_coarse_avgpool"] = paired(diff(
        RE[(RE.window == "1-10") & (RE.arm == "coarse")].set_index("seed")["avgpool_mean_added"].to_dict(),
        RE[(RE.window == "21-30") & (RE.arm == "coarse")].set_index("seed")["avgpool_mean_added"].to_dict()))
    res["reencoding"] = reenc

    # compact per-seed table
    comp = []
    for w in WINDOWS:
        for s in SEEDS:
            e = EV[(EV.window == w) & (EV.seed == s) & (EV.tset == "common")].iloc[0]
            b = BEH[(BEH.window == w) & (BEH.seed == s)].iloc[0]
            a = AN[(AN.window == w) & (AN.seed == s)]
            comp.append({"window": w, "seed": s, "lag_e1_3_vs_random_pp": b.lag_e1_3_vs_random,
                         "lag_e1_3_vs_sham_pp": b.lag_e1_3_vs_sham,
                         "fine_acc_e30_coarse_minus_sham_pp": (float(a.fine_acc_coarse_minus_sham.iloc[0])
                                                               if w in SHAM else None),
                         "fine_acc_e30_coarse_minus_random_pp": float(a.fine_acc_coarse_minus_random.iloc[0]),
                         "excess_DiE_coarse": e.coarse_DiE_excess, "excess_DiE_random": e.random_DiE_excess,
                         "log_ratio": e.log_ratio})
    pd.DataFrame(comp).to_csv(OUT / "compact_per_seed.csv", index=False)

    # concerns
    for w in ("11-20", "21-30"):
        if excluded[w]:
            res["concerns"].append(f"integrity failures window {w}: {excluded[w]}")
    for ts in TSETS:
        res.setdefault("sensitivity_summary", {})[ts] = {
            "primary": {k: hcp[ts]["primary_1_10_vs_21_30"].get(k) for k in ("mean", "ci95", "p_two_sided", "n_pos", "verdict")},
            "secondary": {k: hcp[ts]["secondary_1_10_vs_11_20"].get(k) for k in ("mean", "ci95", "p_two_sided", "n_pos", "verdict")}}
    json.dump(res, open(OUT / "results.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    # flat table
    flat = []

    def walk(o, path):
        if isinstance(o, dict) and "per_seed" in o and "n" in o and isinstance(o["per_seed"], dict) \
                and all(isinstance(v, (int, float)) for v in o["per_seed"].values()):
            ci = o.get("ci95", [None, None])
            flat.append({"outcome": "/".join(path), "n": o["n"], "mean": o.get("mean"), "ci_lo": ci[0], "ci_hi": ci[1],
                         "p_two_sided": o.get("p_two_sided"), "n_pos": o.get("n_pos"), "n_neg": o.get("n_neg"),
                         "verdict": o.get("verdict"), **{f"s{k}": v for k, v in o["per_seed"].items()}})
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, path + [str(k)])
    walk(res, [])
    pd.DataFrame(flat).to_csv(OUT / "all_outcomes.csv", index=False)
    pr = hcp["common"]["primary_1_10_vs_21_30"]
    print("critical-period contrast, primary:", {k: pr.get(k) for k in ("mean", "ci95", "p_two_sided", "n_pos", "per_seed", "verdict")})
    print("critical-period contrast, secondary 11-20:", {k: hcp["common"]["secondary_1_10_vs_11_20"].get(k)
                                    for k in ("mean", "ci95", "p_two_sided", "n_pos", "verdict")})
    for w in ("1-10", "11-20"):
        print("lasting", w, {k: lasting[w].get(k) for k in ("mean", "ci95", "p_two_sided", "n_neg", "verdict")})
    print("critical period:", res["critical_period_epochs1_10"]["verdict"], {k: cp.get(k) for k in ("mean", "ci95", "p_two_sided", "n_neg")})
    print("non-neutral:", {w: (nn[w]["verdict"], nn[w]["n_seeds_both"]) for w in nn})


if __name__ == "__main__":
    main()
