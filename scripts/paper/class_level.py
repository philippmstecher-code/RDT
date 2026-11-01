#!/usr/bin/env python3
"""Scaffold-dependency results at the class level (CPU only; stored per-image predictions).

Lanes (per-image test predictions at every milestone; 100 test images per class):
  last block    prevention (+ prevention_s42_copy for seed 42): superclass code removed from layer4+avgpool,
                epochs 1-10 ('coarse'), with its variance-matched random control ('random')
  late windows  critical: the same at epochs 11-20 (e10) and 21-30 (e20)
  cross-layer   xlayer: superclass code removed at all stages, epochs 1-10
  sham          critical_sham sham_e00 / sham_e10 (same branch, no projection)
Per lane, epoch and true class c: fine accuracy, superclass accuracy (prediction in c's superclass), within-superclass
error (wrong class, right superclass) and between-superclass error.

(A) What is lost: fine accuracy factorises exactly as P(fine) = P(superclass right) x P(fine right | superclass right);
    the suppression cost is split into the two factors (log scale), per window, against random and against sham.
(B) Which superclasses carry the weight: per-superclass suppression cost (random - coarse fine accuracy, epochs 1-3)
    vs the data's mode strength of the superclass (DICTFREE/B_timetable predictors_superclass,
    saxe_sc_W64: distance of the superclass mean from the grand mean in whitened pixels, computed before any timing).
    Split-half reliability of the cost profile across seeds; open seeds 7, 42, 137, held-out seeds 256, 314.
Writes analysis_outputs/paper/CLASS_level/.
Usage: class_level.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

REPO = Path(__file__).resolve().parents[2]
EXP = Path(RDT_DATA + "/checkouts/lesion/data/experiments")
OUT = REPO / "analysis_outputs" / "paper" / "CLASS_level"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [7, 42, 137, 256, 314]
OPEN, HELDOUT = [7, 42, 137], [256, 314]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import transfer_lib as T  # noqa: E402

with open(T.CIFAR / "test", "rb") as f:
    _d = pickle.load(f, encoding="latin1")
FINE2SC = np.zeros(100, dtype=int)
for fl, cl in zip(_d["fine_labels"], _d["coarse_labels"]):
    FINE2SC[fl] = cl
SC_NAMES = pickle.load(open(T.CIFAR / "meta", "rb"), encoding="latin1")["coarse_label_names"]


def lane_dir(kind, arm, s):
    if kind == "lastblock":
        base = EXP / ("prevention_s42_copy" if s == 42 else "prevention") / "lanes"
        return base / f"lesion_{'scaffold' if arm == 'coarse' else 'random_variance'}_e00_seed{s}_rp65_Llayer4-avgpool"
    if kind in ("late10", "late20"):
        e = kind[-2:]
        return EXP / "critical" / "lanes" / f"lesion_{'scaffold' if arm == 'coarse' else 'random_variance'}_e{e}_seed{s}_rp65_Llayer4-avgpool"
    if kind == "xlayer":
        return EXP / "xlayer" / "lanes" / f"lesion_{'scaffold' if arm == 'coarse' else 'random_variance'}_e00_seed{s}_rp65_Llayer1-layer2-layer3-layer4-avgpool"
    if kind == "sham":
        return EXP / "critical_sham" / "lanes" / f"sham_{arm}_seed{s}"
    raise ValueError(kind)


def lane_table(d):
    """Rows (epoch, class, n, correct, sc_correct) from every milestone's sample_predictions.pt."""
    snaps = json.load(open(d / "dev_snapshots" / "metrics" / "devtrain_snapshots.json"))
    snaps = snaps.get("snapshots", snaps) if isinstance(snaps, dict) else snaps
    rows, seen = [], set()
    for sn in snaps:
        if sn.get("snapshot_type") == "t0_resume" or sn["epoch"] in seen:
            continue
        seen.add(sn["epoch"])
        p = Path(sn["snapshot_dir"]) / "sample_predictions.pt"
        if not p.exists():
            p = d / "dev_snapshots" / Path(sn["snapshot_dir"]).name / "sample_predictions.pt"
        if not p.exists():
            continue
        sp = torch.load(p, map_location="cpu", weights_only=False)
        for c, v in sp.items():
            pr = v["predictions"].numpy()
            rows.append({"epoch": int(sn["epoch"]), "cls": int(c), "n": len(pr), "correct": int((pr == c).sum()),
                         "sc_correct": int((FINE2SC[pr] == FINE2SC[c]).sum())})
    return pd.DataFrame(rows)


def load_all():
    cache = OUT / "per_class_long.csv"
    if cache.exists():
        return pd.read_csv(cache)
    parts = []
    for s in SEEDS:
        for kind, arms in (("lastblock", ("coarse", "random")), ("late10", ("coarse", "random")),
                           ("late20", ("coarse", "random")), ("xlayer", ("coarse", "random")), ("sham", ("e00", "e10"))):
            for arm in arms:
                d = lane_dir(kind, arm, s)
                if not d.exists():
                    print("missing", d, flush=True)
                    continue
                t = lane_table(d)
                t["seed"], t["kind"], t["arm"] = s, kind, arm
                parts.append(t)
                print(kind, arm, s, len(t), flush=True)
    df = pd.concat(parts, ignore_index=True)
    df["sc"] = FINE2SC[df["cls"].values]
    df.to_csv(cache, index=False)
    return df


def tci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan, np.nan)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return (x.mean(), x.mean() - h, x.mean() + h)


def factorise(sub):
    """Pooled over classes: P(fine), P(sc), P(fine | sc right) and error split."""
    n, c, sc = sub["n"].sum(), sub["correct"].sum(), sub["sc_correct"].sum()
    return {"p_fine": c / n, "p_sc": sc / n, "p_fine_given_sc": c / max(sc, 1),
            "err_within": (sc - c) / n, "err_between": (n - sc) / n}


def part_A(df):
    windows = {"lastblock": {"win 1-3": range(1, 4), "win 1-10": range(1, 11), "post 11-13": range(11, 14)},
               "xlayer": {"win 1-3": range(1, 4), "win 1-10": range(1, 11), "post 11-13": range(11, 14)},
               "late10": {"win 11-13": range(11, 14), "win 11-20": range(11, 21), "post 21-23": range(21, 24)},
               "late20": {"win 21-23": range(21, 24), "win 21-30": range(21, 31)}}
    sham_of = {"lastblock": "e00", "xlayer": "e00", "late10": "e10", "late20": "e10"}
    rows = []
    for kind, wins in windows.items():
        for wname, eps in wins.items():
            for s in SEEDS:
                get = lambda k, a: df[(df.kind == k) & (df.arm == a) & (df.seed == s) & (df.epoch.isin(list(eps)))]
                co, ra, sh = get(kind, "coarse"), get(kind, "random"), get("sham", sham_of[kind])
                if co.empty or ra.empty:
                    continue
                # per-epoch factorisation, then mean over epochs of the window
                def per_epoch(x):
                    return pd.DataFrame([{**factorise(g), "epoch": e} for e, g in x.groupby("epoch")]).set_index("epoch")
                fc, fr = per_epoch(co), per_epoch(ra)
                fs = per_epoch(sh) if not sh.empty else None
                for ref_name, fref in (("random", fr), ("sham", fs)):
                    if fref is None:
                        continue
                    ep = fc.index.intersection(fref.index)
                    a, b = fc.loc[ep], fref.loc[ep]
                    dlog_fine = np.log(a.p_fine / b.p_fine).mean()
                    dlog_sc = np.log(a.p_sc / b.p_sc).mean()
                    dlog_cond = np.log(a.p_fine_given_sc / b.p_fine_given_sc).mean()
                    rows.append({"kind": kind, "window": wname, "ref": ref_name, "seed": s,
                                 "d_fine_pp": 100 * (a.p_fine - b.p_fine).mean(),
                                 "d_sc_pp": 100 * (a.p_sc - b.p_sc).mean(),
                                 "d_err_within_pp": 100 * (a.err_within - b.err_within).mean(),
                                 "d_err_between_pp": 100 * (a.err_between - b.err_between).mean(),
                                 "dlog_fine": dlog_fine, "dlog_sc": dlog_sc, "dlog_cond": dlog_cond,
                                 "share_via_sc": dlog_sc / dlog_fine if dlog_fine != 0 else np.nan})
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "A_factorisation_per_seed.csv", index=False)
    summ = []
    for (k, w, r), g in A.groupby(["kind", "window", "ref"], sort=False):
        row = {"kind": k, "window": w, "ref": r, "n": len(g)}
        for col in ("d_fine_pp", "d_sc_pp", "d_err_within_pp", "d_err_between_pp", "dlog_fine", "dlog_sc", "dlog_cond",
                    "share_via_sc"):
            m, lo, hi = tci(g[col])
            row[col] = f"{m:.3f} [{lo:.3f}, {hi:.3f}]"
        row["seeds_cond_drop"] = f"{int((g.dlog_cond < 0).sum())}/{len(g)}"
        row["seeds_sc_drop"] = f"{int((g.dlog_sc < 0).sum())}/{len(g)}"
        summ.append(row)
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "A_factorisation_summary.csv", index=False)
    print(S[["kind", "window", "ref", "n", "d_fine_pp", "dlog_sc", "dlog_cond", "share_via_sc", "seeds_sc_drop",
             "seeds_cond_drop"]].to_string(index=False))


def sc_cost(df, kind, eps, seeds, ref="random"):
    """Per-superclass cost (ref - coarse fine accuracy, pp), mean over epochs and the given seeds; also per seed."""
    out = {}
    for s in seeds:
        sel = lambda a: df[(df.kind == kind) & (df.arm == a) & (df.seed == s) & (df.epoch.isin(list(eps)))]
        co = sel("coarse").groupby("sc")[["correct", "n"]].sum()
        rf = sel(ref).groupby("sc")[["correct", "n"]].sum() if ref == "random" else \
            df[(df.kind == "sham") & (df.arm == "e00") & (df.seed == s) & (df.epoch.isin(list(eps)))].groupby("sc")[["correct", "n"]].sum()
        if co.empty or rf.empty:
            continue
        out[s] = 100 * (rf.correct / rf.n - co.correct / co.n)
    return pd.DataFrame(out)  # index sc, columns seeds


def part_B(df):
    pred = pd.read_csv(REPO / "analysis_outputs" / "paper" / "DICTFREE" / "B_timetable" / "predictors_superclass.csv").set_index("sc")
    res = {}
    for kind, eps, ref in (("lastblock", range(1, 4), "random"), ("lastblock", range(1, 4), "sham"),
                           ("lastblock", range(1, 11), "random"), ("xlayer", range(1, 4), "random"),
                           ("late10", range(11, 14), "random")):
        C = sc_cost(df, kind, eps, SEEDS, ref)
        if C.empty:
            continue
        key = f"{kind}|{eps.start}-{eps.stop - 1}|vs_{ref}"
        # reliability: mean of open seeds vs mean of held-out seeds
        cd, cc = C[[s for s in OPEN if s in C]].mean(1), C[[s for s in HELDOUT if s in C]].mean(1)
        rel = stats.spearmanr(cd, cc)
        r = {"split_half_rho_disc_vs_conf": round(rel.correlation, 3), "p": round(rel.pvalue, 4),
             "mean_cost_pp": round(C.values.mean(), 2)}
        for pcol in ("saxe_sc_W64", "ratio_sc_over_ws_W64", "lda_pix_sc"):
            x = pred.loc[C.index, pcol]
            rd, rc = stats.spearmanr(x, cd), stats.spearmanr(x, cc)
            r[f"{pcol}: rho_open"] = round(rd.correlation, 3)
            r[f"{pcol}: p_open"] = round(rd.pvalue, 4)
            r[f"{pcol}: rho_heldout"] = round(rc.correlation, 3)
            r[f"{pcol}: p_heldout"] = round(rc.pvalue, 4)
            r[f"{pcol}: per_seed_rho"] = [round(stats.spearmanr(x, C[s]).correlation, 2) for s in C.columns]
        res[key] = r
        tab = pd.DataFrame({"superclass": [SC_NAMES[i] for i in C.index], "cost_mean": C.mean(1).round(2),
                            "cost_open": cd.round(2), "cost_heldout": cc.round(2),
                            "mode_strength": pred.loc[C.index, "saxe_sc_W64"].round(3)})
        tab.sort_values("cost_mean", ascending=False).to_csv(OUT / f"B_sc_cost_{key.replace('|', '_')}.csv", index=False)
        if key == "lastblock|1-3|vs_random":
            print(tab.sort_values("cost_mean", ascending=False).to_string(index=False))
    json.dump(res, open(OUT / "B_superclass_results.json", "w"), indent=1)
    for k, v in res.items():
        print(k, json.dumps(v))


if __name__ == "__main__":
    df = load_all()
    print("rows", len(df), df.groupby(["kind", "arm"]).seed.nunique().to_dict())
    part_A(df)
    part_B(df)
