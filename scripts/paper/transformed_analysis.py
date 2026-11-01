#!/usr/bin/env python3
"""Transformed features (0.2 <= r < 0.5) in the 9 core CIFAR-100 lanes.

Extends scripts/trace_transformed_features.py, whose Phase-6 labelling rules (`Labeller`) are reused, using the
matching cache written by transformed_matching_cache.py.

Conventions:
  * real transitions are consecutive milestones; '<k>->terminal' compares the final model with itself and is
    reported separately as an SAE-refit noise reference;
  * ResNet-18 'avgpool' has the same SAE input as 'layer4'; pooled numbers drop 'avgpool' ("dedup"), while the
    "paper" variants keep all layers to reproduce the stored reference values;
  * a transformed pair (i -> j) is treated as excluded (pipeline: neither died, born nor stable), as turnover
    (i died and j born) or as continuity (a stable link, same z-score rule for labels).
Writes analysis_outputs/paper/TRANSFORMED/transformed_lane_results.pkl, aggregated by transformed_report.py.

Usage: transformed_analysis.py
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "packages" / "ml" / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from trace_transformed_features import Labeller, _at, _sel  # noqa: E402

DATA = Path(RDT_DATA + "/core")
CORE = DATA / "experiments" / "abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb" / "lanes"
OUT = REPO / "analysis_outputs" / "paper" / "TRANSFORMED"
CACHE = OUT / "cache"

CORE_LANES = [
    ("7db28a9a-f436-4d8a-a6b4-37e69fbf54eb", "ResNet-18", 42),
    ("a0bcbcba-94a9-4009-bcb6-d725b47588ee", "ResNet-18", 137),
    ("935c13ae-a026-4eae-a2b2-0306879b2e8c", "ResNet-18", 256),
    ("10d31151-68a2-4491-836d-3ead8e00a7ad", "ViT-Small", 42),
    ("909d7f68-55e1-4cd9-8c7b-68d5793798d3", "ViT-Small", 137),
    ("abad8975-de55-4e7e-b95d-05067f0eed90", "ViT-Small", 256),
    ("4bebc329-c26a-4eb4-b18c-7586e513f49e", "CCT-7", 42),
    ("edf8275e-c99f-4436-a13d-4ed50f072a66", "CCT-7", 137),
    ("445dd4d4-770a-4f33-b82b-c0bc7baed9a7", "CCT-7", 256),
]
ARCHS = ("ResNet-18", "ViT-Small", "CCT-7")
TREATMENTS = ("excluded", "turnover", "continuity")
PROCS = ("tg_h", "ab_h", "di_h", "as_h", "de_h")


def f(x, nd=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), nd)


# per-sample process counts
def count_processes(m, sel_a, sel_b, lab, treatment):
    """Count-exact Phase 6 from cached co-activation counts (== trace script)."""
    alive_a, alive_b, n_a, n_b = m["alive_a"], m["alive_b"], m["n_a"], m["n_b"]
    stable = list(m["stable"]); died = list(m["died"]); born = list(m["born"])
    if treatment == "turnover":
        died += [p[0] for p in m["transformed"]]; born += [p[1] for p in m["transformed"]]
    elif treatment == "continuity":
        stable += list(m["transformed"])
    c = Counter()
    for i, j, _, n in stable:
        if not alive_a[i] or n == 0:
            continue
        c["stable"] += n
        for p in lab.stable(_at(sel_a, i), _at(sel_b, j)):
            c[p] += n
    for i in died:
        if alive_a[i]:
            c["de_h"] += int(n_a[i])
    for j in born:
        if alive_b[j] and n_b[j]:
            for p in lab.born(*_at(sel_b, j)):
                c[p] += int(n_b[j])
    return c


def churn(c):
    ev = sum(c[p] for p in PROCS)
    return ev / (ev + c["stable"]) if ev + c["stable"] else None


def cr(c):
    return (c["tg_h"] + c["ab_h"]) / c["di_h"] if c["di_h"] else float("inf")


def ordering(series):
    """series: list of Counters over real transitions. Peak index of each
    process' share of (tg+ab+di) and event-weighted centre of mass."""
    fr = {k: [] for k in ("tg_h", "ab_h", "di_h")}
    for c in series:
        tot = c["tg_h"] + c["ab_h"] + c["di_h"]
        for k in fr:
            fr[k].append(c[k] / tot if tot else np.nan)
    peak = {k: int(np.nanargmax(v)) for k, v in fr.items()}
    idx = np.arange(len(series))
    com = {}
    for k in fr:
        w = np.array([c[k] for c in series], float)
        com[k] = float((idx * w).sum() / w.sum()) if w.sum() else None
    return {"peak": peak, "peak_order_holds": peak["tg_h"] <= peak["ab_h"] <= peak["di_h"],
            "com": {k: f(v, 3) for k, v in com.items()},
            "com_order_holds": None if None in com.values() else com["tg_h"] < com["ab_h"] < com["di_h"]}


# persistence
def cohort(sel, i, lab):
    ssi, csi, sai = _at(sel, i)
    if sai > lab.sai_t:
        return "SAI"
    if ssi > lab.ssi_t:
        return "SSI"
    if csi > lab.csi_t:
        return "CSI"
    return None


def forward_maps(per_t, trans, treatment):
    """Per transition: dict i -> (j | None, kind) for every alive A feature."""
    maps = []
    for t in trans:
        m = per_t[t]; d = {}
        for i, j, _, _ in m["stable"]:
            d[i] = (j, "stable")
        for i, j, _, _ in m["transformed"]:
            d[i] = (j, "link" if treatment == "continuity" else "transformed")
        for i in m["died"]:
            d[i] = (None, "died")
        maps.append(d)
    return maps


def lifespans_paper_style(per_t, trans, labels_real, selectivity, layer, lab, treatment):
    """Every alive feature at every real checkpoint c < last is a record (the
    convention of scripts/compute_feature_survival*.py). Lifespan = consecutive
    continuing links."""
    maps = forward_maps(per_t, trans, treatment)
    n = len(trans); recs = []
    for c in range(n):
        sel = _sel(selectivity, labels_real[c], layer)
        alive = np.where(per_t[trans[c]]["alive_a"])[0]
        for i in alive:
            cur, life, end = int(i), 0, "censored"
            for k in range(c, n):
                j, kind = maps[k].get(cur, (None, "died"))
                if kind in ("stable", "link"):
                    life += 1; cur = j
                else:
                    end = kind; break
            if treatment == "excluded" and end == "transformed":
                continue  # transformed exits are excluded from the analysis
            ssi, csi, sai = _at(sel, int(i))
            recs.append((cohort(sel, int(i), lab), life, n - c, end, ssi, csi, sai))
    return recs


def backward_persistence(per_t, trans, labels_real, selectivity, layer, lab, treatment):
    """Features alive at the last real checkpoint: how many transitions back do
    they trace? normalised by number of real transitions (Fig. 4c analogue)."""
    n = len(trans)
    back = []
    for t in trans:
        m = per_t[t]; d = {}
        for i, j, _, _ in m["stable"]:
            d[j] = (i, "stable")
        for i, j, _, _ in m["transformed"]:
            d[j] = (i, "link" if treatment == "continuity" else "transformed")
        for j in m["born"]:
            d[j] = (None, "born")
        back.append(d)
    last = trans[-1]
    sel = _sel(selectivity, labels_real[-1], layer)
    out = []
    for j0 in np.where(per_t[last]["alive_b"])[0]:
        cur, age, start = int(j0), 0, "initial"
        for k in range(n - 1, -1, -1):
            i, kind = back[k].get(cur, (None, "born"))
            if kind in ("stable", "link"):
                age += 1; cur = i
            else:
                start = kind; break
        if treatment == "excluded" and start == "transformed":
            continue
        out.append((cohort(sel, int(j0), lab), age / n))
    return out


# lane driver
def analyze_lane(lane_id, arch, seed):
    d = pickle.load(open(CACHE / f"{lane_id}.pkl", "rb"))
    res = json.loads((CORE / lane_id / "sae_analysis" / "sae_results.json").read_text())
    meta = res["metadata"]; lab = Labeller(meta); selectivity = res["selectivity"]
    labels = d["labels"]; assert labels[-1] == "terminal"
    real = labels[:-1]
    trans = [f"{a}->{b}" for a, b in zip(real[:-1], real[1:])]
    tterm = f"{real[-1]}->terminal"
    layers = list(d["layer_data"].keys())
    pooled_layers = [l for l in layers if not (arch == "ResNet-18" and l == "avgpool")]
    out = {"lane_id": lane_id, "arch": arch, "seed": seed, "real_transitions": trans,
           "terminal_transition": tterm, "layers": layers, "pooled_layers": pooled_layers,
           "matching_mismatches": len(d["mismatches"]), "missing_files": len(d["missing"]),
           "thresholds": {"ssi": lab.ssi_t, "csi": lab.csi_t, "sai": lab.sai_t}}

    # 0. process counts per (treatment, transition, layer)
    pc = {tr: {t: {} for t in trans + [tterm]} for tr in TREATMENTS}
    for layer in layers:
        for t in trans + [tterm]:
            m = d["layer_data"][layer][t]
            a, b = t.split("->")
            sa, sb = _sel(selectivity, a, layer), _sel(selectivity, b, layer)
            for tr in TREATMENTS:
                pc[tr][t][layer] = count_processes(m, sa, sb, lab, tr)

    def agg(tr, t, lays):
        c = Counter()
        for l in lays:
            c.update(pc[tr][t][l])
        return c

    # validation vs stored process_intensity (all layers, excluded treatment)
    stored = {r["transition"]: r for r in res["process_intensity"]}
    val = []
    for t in trans + [tterm]:
        c = agg("excluded", t, layers)
        ok = all(int(stored[t][p]) == int(c[p]) for p in PROCS)
        val.append({"transition": t, "match": ok,
                    "stored_churn": f(stored[t]["churn"], 6), "recomputed_churn": f(churn(c), 6)})
    out["process_intensity_validation"] = val

    # per-transition process summaries
    proc = {}
    for variant, lays in (("dedup", pooled_layers), ("paper_all_layers", layers)):
        proc[variant] = {}
        for tr in TREATMENTS:
            series = [agg(tr, t, lays) for t in trans]
            cterm = agg(tr, tterm, lays)
            proc[variant][tr] = {
                "per_transition": [{"transition": t, **{p: int(c[p]) for p in PROCS + ("stable",)},
                                    "churn": f(churn(c)), "cr": f(cr(c))} for t, c in zip(trans, series)],
                "terminal_noise": {**{p: int(cterm[p]) for p in PROCS + ("stable",)},
                                   "churn": f(churn(cterm)), "cr": f(cr(cterm))},
                "cr_first": f(cr(series[0])), "cr_last_real": f(cr(series[-1])),
                "cr_terminal_noise": f(cr(cterm)),
                "di_zero_at_first": series[0]["di_h"] == 0,
                "ordering": ordering(series),
                "first": {p: int(series[0][p]) for p in PROCS + ("stable",)},
                "last_real": {p: int(series[-1][p]) for p in PROCS + ("stable",)},
                "terminal": {p: int(cterm[p]) for p in PROCS + ("stable",)},
            }
    out["processes"] = proc

    # feature-level churn at the first transition under each treatment
    fl = {}
    for variant, lays in (("dedup", pooled_layers), ("paper_all_layers", layers)):
        S = sum(d["layer_data"][l][trans[0]]["stable"].__len__() for l in lays)
        T = sum(len(d["layer_data"][l][trans[0]]["transformed"]) for l in lays)
        B = sum(len(d["layer_data"][l][trans[0]]["born"]) for l in lays)
        D = sum(len(d["layer_data"][l][trans[0]]["died"]) for l in lays)
        fl[variant] = {"n_stable": S, "n_transformed": T, "n_born": B, "n_died": D,
                       "excluded": f((B + D) / (S + B + D)),
                       "turnover": f((B + D + 2 * T) / (S + B + D + 2 * T)),
                       "continuity": f((B + D) / (S + T + B + D))}
    out["feature_level_churn_first_transition"] = fl

    # 1. prevalence + floors
    ctrl = defaultdict(list); ctrl_fd = defaultdict(list); ctrl_st = defaultdict(list)
    for e in res["within_checkpoint_control"]["per_checkpoint"]:
        for x in e["per_layer"]:
            if x["n_alive_primary"]:
                ctrl[x["layer"]].append(x["n_transformed"] / x["n_alive_primary"])
                ctrl_fd[x["layer"]].append(x["false_death_rate"])
                ctrl_st[x["layer"]].append(x["stable_rate"])
    prev = []
    for layer in layers:
        mt = d["layer_data"][layer][tterm]
        aa = int(mt["alive_a"].sum())
        term = {"transformed_rate": len(mt["transformed"]) / aa if aa else None,
                "death_rate": len(mt["died"]) / aa if aa else None,
                "stable_rate": len(mt["stable"]) / aa if aa else None,
                "birth_rate": len(mt["born"]) / int(mt["alive_b"].sum()) if mt["alive_b"].sum() else None}
        for ti, t in enumerate(trans):
            m = d["layer_data"][layer][t]; aa_t = int(m["alive_a"].sum())
            prev.append({"layer": layer, "transition": t, "t_index": ti, "n_alive_a": aa_t,
                         "n_stable": len(m["stable"]), "n_transformed": len(m["transformed"]),
                         "n_died": len(m["died"]), "n_born": len(m["born"]),
                         "transformed_rate": len(m["transformed"]) / aa_t if aa_t else None,
                         "corr_transformed_mean": f(np.mean([p[2] for p in m["transformed"]])) if m["transformed"] else None})
        out.setdefault("floors", {})[layer] = {
            "control_transformed_rate": f(np.mean(ctrl[layer])) if ctrl[layer] else None,
            "control_false_death_rate": f(np.mean(ctrl_fd[layer])) if ctrl_fd[layer] else None,
            "control_stable_rate": f(np.mean(ctrl_st[layer])) if ctrl_st[layer] else None,
            "terminal_same_init": {k: f(v) for k, v in term.items()}}
    out["prevalence"] = prev

    # 2. fates + sequences
    fate = Counter()       # (layer, k, in, out) counts
    triples = Counter()    # (layer, k, in, out1, out2)
    for layer in layers:
        L = d["layer_data"][layer]
        for k in range(len(trans)):
            m1 = L[trans[k]]
            outl = {}
            for i, j, _, _ in m1["stable"]:
                outl[i] = ("stable", j)
            for i, j, _, _ in m1["transformed"]:
                outl[i] = ("transformed", j)
            for i in m1["died"]:
                outl[i] = ("died", None)
            inl = {}
            if k > 0:
                m0 = L[trans[k - 1]]
                for _, j, _, _ in m0["stable"]:
                    inl[j] = "stable"
                for _, j, _, _ in m0["transformed"]:
                    inl[j] = "transformed"
                for j in m0["born"]:
                    inl[j] = "born"
            nxt = None
            if k + 1 < len(trans):
                m2 = L[trans[k + 1]]; nxt = {}
                for i, _, _, _ in m2["stable"]:
                    nxt[i] = "stable"
                for i, _, _, _ in m2["transformed"]:
                    nxt[i] = "transformed"
                for i in m2["died"]:
                    nxt[i] = "died"
            for i, (o1, j) in outl.items():
                o0 = inl.get(i, "initial" if k == 0 else "unmatched")
                fate[(layer, k, o0, o1)] += 1
                if j is not None and nxt is not None:
                    triples[(layer, k, o0, o1, nxt.get(j, "unmatched"))] += 1
    out["fate_counts"] = [list(k) + [v] for k, v in fate.items()]
    out["triple_counts"] = [list(k) + [v] for k, v in triples.items()]

    # 3. selectivity by class
    sel_rows = []
    for layer in layers:
        for t in trans:
            m = d["layer_data"][layer][t]; a, b = t.split("->")
            sa, sb = _sel(selectivity, a, layer), _sel(selectivity, b, layer)
            for cls, idx, s in (("A_stable", [p[0] for p in m["stable"]], sa),
                                ("A_transformed", [p[0] for p in m["transformed"]], sa),
                                ("A_died", m["died"], sa),
                                ("B_stable", [p[1] for p in m["stable"]], sb),
                                ("B_transformed", [p[1] for p in m["transformed"]], sb),
                                ("B_born", m["born"], sb)):
                for i in idx:
                    ssi, csi, sai = _at(s, i)
                    sel_rows.append((layer, t, cls, ssi, csi, sai))
    out["selectivity_rows"] = sel_rows

    # 4. persistence
    pers = {}
    for tr in TREATMENTS:
        recs, back = [], []
        for layer in pooled_layers:
            per_t = d["layer_data"][layer]
            recs += lifespans_paper_style(per_t, trans, real, selectivity, layer, lab, tr)
            back += backward_persistence(per_t, trans, real, selectivity, layer, lab, tr)
        life = {c: [r[1] for r in recs if r[0] == c] for c in ("SAI", "SSI", "CSI")}
        mean = {c: (float(np.mean(v)) if v else None) for c, v in life.items()}
        constr = life["SAI"] + life["SSI"]
        rho_ssi = stats.spearmanr([r[4] for r in recs], [r[1] for r in recs])
        rho_csi = stats.spearmanr([r[5] for r in recs], [r[1] for r in recs])
        bp = {c: [x[1] for x in back if x[0] == c] for c in ("SAI", "SSI", "CSI")}
        bmean = {c: (float(np.mean(v)) if v else None) for c, v in bp.items()}
        pers[tr] = {
            "n_records": len(recs), "n_by_cohort": {c: len(v) for c, v in life.items()},
            "mean_lifespan": {c: f(v) for c, v in mean.items()},
            "construction_over_csi_lifespan": f(np.mean(constr) / mean["CSI"]) if constr and mean["CSI"] else None,
            "sai_over_csi_lifespan": f(mean["SAI"] / mean["CSI"]) if mean["SAI"] and mean["CSI"] else None,
            "ssi_over_csi_lifespan": f(mean["SSI"] / mean["CSI"]) if mean["SSI"] and mean["CSI"] else None,
            "spearman_ssi_lifespan": [f(rho_ssi.correlation), float(rho_ssi.pvalue)],
            "spearman_csi_lifespan": [f(rho_csi.correlation), float(rho_csi.pvalue)],
            "end_reasons": dict(Counter(r[3] for r in recs)),
            "terminal_persistence_mean": {c: f(v) for c, v in bmean.items()},
            "terminal_persistence_n": {c: len(v) for c, v in bp.items()},
            "terminal_persistence_sai_over_csi": f(bmean["SAI"] / bmean["CSI"]) if bmean["SAI"] and bmean["CSI"] else None,
            "terminal_persistence_ssi_over_csi": f(bmean["SSI"] / bmean["CSI"]) if bmean["SSI"] and bmean["CSI"] else None,
        }
    out["persistence"] = pers
    return out


def main():
    results = []
    for lane_id, arch, seed in CORE_LANES:
        if not (CACHE / f"{lane_id}.pkl").exists():
            print(f"missing cache for {arch} {seed}"); continue
        print(f"== {arch} seed {seed}", flush=True)
        results.append(analyze_lane(lane_id, arch, seed))
    with open(OUT / "transformed_lane_results.pkl", "wb") as fh:
        pickle.dump(results, fh)
    print(f"wrote {OUT / 'transformed_lane_results.pkl'}")


if __name__ == "__main__":
    main()
