#!/usr/bin/env python3
"""Aggregate the per-lane matching diagnostics (matching_core.py) and checkpoint-density outputs into CSV tables in
analysis_outputs/paper/MATCHING/.

Real transitions exclude k->terminal (the same model decomposed twice), which serves as a null. Pooled statistics
use ResNet avgpool and drop layer4 (identical data). Percentages pool counts over (layer, transition) within a run;
architecture summaries give the mean and min-max range over the three seeds.

Usage: python scripts/paper/matching_aggregate.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matching_lib import CORE_LANES, OUT  # noqa: E402

RULES = ("hungarian", "mutual_best", "greedy", "confirmed")
ARCHS = ("ResNet-18", "ViT-Small", "CCT-7")


def write_csv(path: Path, rows, cols=None):
    if not rows:
        return
    cols = cols or list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def load():
    lanes = {}
    for lid, arch, seed in CORE_LANES:
        p = OUT / "per_lane" / f"{lid}.json"
        if p.exists():
            lanes[lid] = json.load(open(p))
    return lanes


def keep(r):
    return not (r["arch"] == "ResNet-18" and r["layer"] == "layer4")


def summarize_diag(rows):
    S = lambda k: sum(r[k] for r in rows if r.get(k) is not None)  # noqa: E731
    n_born = S("diag_n_born"); n_died = S("diag_n_died"); n_st = S("diag_n_stable")
    alive_a = S("diag_n_alive_a"); alive_b = S("diag_n_alive_b"); n_tr = S("diag_n_transformed")
    return {
        "n_cells": len(rows), "alive_a": alive_a, "alive_b": alive_b,
        "stable_pct_aliveA": pct(n_st, alive_a), "transformed_pct_aliveA": pct(n_tr, alive_a),
        "died_pct_aliveA": pct(n_died, alive_a), "born_pct_aliveB": pct(n_born, alive_b),
        "n_born": n_born, "contested_birth_r05_pct": pct(S("diag_born_bwd_ge05"), n_born),
        "contested_birth_r02_pct": pct(S("diag_born_bwd_ge02"), n_born),
        "contested_birth_absr05_pct": pct(S("diag_born_bwdabs_ge05"), n_born),
        "swap_scenario_n": S("diag_born_scenario"),
        "swap_scenario_pct_born": pct(S("diag_born_scenario"), n_born),
        "swap_scenario_pct_aliveB": pct(S("diag_born_scenario"), alive_b),
        "scenario_partner_stable": S("diag_born_scenario_partner_stable"),
        "scenario_partner_transformed": S("diag_born_scenario_partner_transformed"),
        "scenario_partner_died": S("diag_born_scenario_partner_died"),
        "n_died": n_died, "contested_death_r05_pct": pct(S("diag_died_fwd_ge05"), n_died),
        "contested_death_r02_pct": pct(S("diag_died_fwd_ge02"), n_died),
        "transformed_with_r05_alt_pct": pct(S("diag_transformed_fwd_ge05"), n_tr),
        "n_stable": n_st, "stable_mutual_best_pct": pct(S("diag_stable_mutual_best"), n_st),
        "stable_rowbest_pct": pct(S("diag_stable_row_best"), n_st),
        "stable_colbest_pct": pct(S("diag_stable_col_best"), n_st),
        "stable_margin_lt0_pct": pct(S("diag_stable_margin_min_lt0"), n_st),
        "stable_margin_lt005_pct": pct(S("diag_stable_margin_min_lt005"), n_st),
        "stable_margin_lt01_pct": pct(S("diag_stable_margin_min_lt01"), n_st),
        "stable_alt_partner_r05_pct": pct(S("diag_stable_alt_ge05"), n_st),
        "splits_pct_aliveA": pct(S("diag_splits_a"), alive_a),
        "merges_pct_aliveB": pct(S("diag_merges_b"), alive_b),
        "merges_on_born_pct_born": pct(S("diag_merges_b_born"), n_born),
        "splits_died_pct_died": pct(S("diag_splits_a_died"), n_died),
    }


def summarize_rules(rows):
    S = lambda k: sum(r[k] for r in rows if r.get(k) is not None)  # noqa: E731
    aa = S("diag_n_alive_a"); ab = S("diag_n_alive_b")
    out = {}
    for ru in RULES:
        out[f"{ru}_stable_pct"] = pct(S(f"{ru}_n_stable"), aa)
        out[f"{ru}_transformed_pct"] = pct(S(f"{ru}_n_transformed"), aa)
        out[f"{ru}_died_pct"] = pct(S(f"{ru}_n_died"), aa)
        out[f"{ru}_born_pct"] = pct(S(f"{ru}_n_born"), ab)
        out[f"{ru}_ambiguousB_pct"] = pct(S(f"{ru}_n_ambiguous_b"), ab)
        for p in ("tg", "ab", "di"):
            out[f"{ru}_f_{p}"] = S(f"{ru}_f_{p}")
        if ru != "hungarian":
            le = S(f"{ru}_lc_n_labelled_either"); lc = S(f"{ru}_lc_n_label_changed")
            out[f"{ru}_label_change_pct_of_labelled"] = pct(lc, le)
            out[f"{ru}_label_change_pct_of_aliveB"] = pct(lc, ab)
            for p in ("tg", "ab", "di"):
                base = S(f"hungarian_f_{p}")
                out[f"{ru}_chg_{p}_pct_of_H{p}"] = pct(S(f"{ru}_lc_chg_{p}"), base)
            if ru in ("mutual_best", "greedy"):
                out[f"{ru}_same_partner_pct"] = pct(S(f"{ru}_stable_same_partner"), S(f"{ru}_stable_overlap"))
    return out


def ordering(series_by_t):
    """series_by_t: list of dicts with tg, ab, di (counts) per real transition in order."""
    T = len(series_by_t)
    t = np.arange(T) / max(T - 1, 1)
    res = {}
    for p in ("tg", "ab", "di"):
        v = np.array([s[p] for s in series_by_t], float)
        res[f"com_{p}"] = float((t * v).sum() / v.sum()) if v.sum() > 0 else float("nan")
    res["order_tg_ab_di"] = bool(res["com_tg"] < res["com_ab"] < res["com_di"])
    res["order_tg_lt_di"] = bool(res["com_tg"] < res["com_di"])
    res["order_ab_lt_di"] = bool(res["com_ab"] < res["com_di"])
    cr = np.array([(s["tg"] + s["ab"]) / s["di"] if s["di"] > 0 else np.inf for s in series_by_t])
    res["cr_first"] = float(cr[0]); res["cr_last"] = float(cr[-1])
    fin = np.isfinite(cr)
    res["cr_spearman_t"] = float(spearmanr(np.arange(T)[fin], cr[fin]).correlation) if fin.sum() > 2 else float("nan")
    res["cr_declines"] = bool(cr[-1] < cr[0])
    res["cr_last_below1"] = bool(cr[-1] < 1)
    res["cr_series"] = ";".join(f"{c:.3g}" for c in cr)
    return res


def arch_summary(rows, keycols, group="arch"):
    out = []
    for a in ARCHS:
        rs = [r for r in rows if r[group] == a]
        if not rs:
            continue
        o = {"arch": a, "n_runs": len(rs)}
        for k in keycols:
            v = np.array([r[k] for r in rs if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)], float)
            if len(v):
                o[f"{k}_mean"] = float(np.nanmean(v)); o[f"{k}_min"] = float(np.nanmin(v)); o[f"{k}_max"] = float(np.nanmax(v))
        out.append(o)
    return out


def main():
    lanes = load()
    print(f"loaded {len(lanes)} lanes")
    # validation
    val, all_tr = [], []
    for lid, d in lanes.items():
        tr = d["transitions"]
        all_tr += tr
        pis = [r for r in d["process"] if "pi_match" in r]
        val.append({"lane": lid, "arch": tr[0]["arch"], "seed": tr[0]["seed"],
                    "n_layer_transitions": len(tr), "counts_exact_match": sum(r["counts_match"] for r in tr),
                    "n_transitions_process_intensity": len(pis),
                    "process_intensity_exact_match": sum(r["pi_match"] for r in pis)})
    write_csv(OUT / "validation.csv", val)
    write_csv(OUT / "transitions_all.csv", all_tr)

    # diagnostics + rules per lane (real vs terminal null)
    diag_rows, rule_rows, layer_rows = [], [], []
    for lid, d in lanes.items():
        tr = [r for r in d["transitions"] if keep(r)]
        arch, seed = tr[0]["arch"], tr[0]["seed"]
        for kind, sub in (("real", [r for r in tr if not r["is_terminal"]]), ("terminal_null", [r for r in tr if r["is_terminal"]])):
            diag_rows.append({"lane": lid, "arch": arch, "seed": seed, "transitions": kind, **summarize_diag(sub)})
            rule_rows.append({"lane": lid, "arch": arch, "seed": seed, "transitions": kind, **summarize_rules(sub)})
        for layer in dict.fromkeys(r["layer"] for r in d["transitions"]):
            sub = [r for r in d["transitions"] if r["layer"] == layer and not r["is_terminal"]]
            layer_rows.append({"lane": lid, "arch": arch, "seed": seed, "layer": layer, **summarize_diag(sub),
                               **{k: v for k, v in summarize_rules(sub).items() if "label_change" in k}})
    write_csv(OUT / "diag_by_run.csv", diag_rows)
    write_csv(OUT / "diag_by_run_layer.csv", layer_rows)
    write_csv(OUT / "alt_rules_by_run.csv", rule_rows)
    dkeys = [k for k in diag_rows[0] if k not in ("lane", "arch", "seed", "transitions")]
    rkeys = [k for k in rule_rows[0] if k not in ("lane", "arch", "seed", "transitions")]
    diag_arch = []
    for kind in ("real", "terminal_null"):
        for o in arch_summary([r for r in diag_rows if r["transitions"] == kind], dkeys):
            diag_arch.append({"transitions": kind, **o})
    write_csv(OUT / "diag_by_arch.csv", diag_arch)
    rule_arch = []
    for kind in ("real", "terminal_null"):
        for o in arch_summary([r for r in rule_rows if r["transitions"] == kind], rkeys):
            rule_arch.append({"transitions": kind, **o})
    write_csv(OUT / "alt_rules_by_arch.csv", rule_arch)

    # process ordering under each rule (sample-weighted and feature-level)
    ord_rows = []
    for lid, d in lanes.items():
        labels = d["labels"]; real_pairs = [f"{a}->{b}" for a, b in zip(labels[:-1], labels[1:]) if b != "terminal"]
        arch, seed = d["transitions"][0]["arch"], d["transitions"][0]["seed"]
        ref = {}
        for ru in RULES:
            # sample-weighted, dedup layer4
            pr = {r["pair"]: r for r in d["process"] if r["rule"] == ru and r["dedup_layer4"]}
            ser_w = [{"tg": pr[p]["tg_h"], "ab": pr[p]["ab_h"], "di": pr[p]["di_h"]} for p in real_pairs]
            # feature-level
            ser_f = []
            for p in real_pairs:
                rs = [r for r in d["transitions"] if r["pair"] == p and keep(r)]
                ser_f.append({q: sum(r[f"{ru}_f_{q}"] for r in rs) for q in ("tg", "ab", "di")})
            for wt, ser in (("sample_weighted", ser_w), ("feature_level", ser_f)):
                row = {"lane": lid, "arch": arch, "seed": seed, "rule": ru, "weighting": wt,
                       "n_real_transitions": len(real_pairs), **ordering(ser)}
                if ru == "hungarian":
                    ref[wt] = ser
                for q in ("tg", "ab", "di"):
                    a = [x[q] for x in ref[wt]]; b = [x[q] for x in ser]
                    row[f"rho_vs_hungarian_{q}"] = float(spearmanr(a, b).correlation) if np.std(a) > 0 and np.std(b) > 0 else float("nan")
                ord_rows.append(row)
        # pipeline-style (all layers incl. layer4), Hungarian, as the stored process_intensity counts them
        pr = {r["pair"]: r for r in d["process"] if r["rule"] == "hungarian" and not r["dedup_layer4"]}
        ser = [{"tg": pr[p]["tg_h"], "ab": pr[p]["ab_h"], "di": pr[p]["di_h"]} for p in real_pairs]
        ord_rows.append({"lane": lid, "arch": arch, "seed": seed, "rule": "hungarian_pipeline_all_layers",
                         "weighting": "sample_weighted", "n_real_transitions": len(real_pairs), **ordering(ser)})
    write_csv(OUT / "process_ordering_by_rule.csv", ord_rows)

    # skip-one (core)
    sk_rows = []
    for lid, d in lanes.items():
        sk = [r for r in d["skip_one"] if not (r["arch"] == "ResNet-18" and r["layer"] == "layer4")]
        arch, seed = sk[0]["arch"], sk[0]["seed"]
        S = lambda k: sum(r.get(k, 0) for r in sk)  # noqa: E731
        sk_rows.append({"lane": lid, "arch": arch, "seed": seed, **skip_summary(sk)})
    write_csv(OUT / "skip_one_by_run.csv", sk_rows)
    write_csv(OUT / "skip_one_by_arch.csv", arch_summary(sk_rows, [k for k in sk_rows[0] if k not in ("lane", "arch", "seed")]))

    # lesion density
    lp = OUT / "per_lane" / "lesion_density.json"
    if lp.exists():
        L = json.load(open(lp))
        les_win, les_skip, les_step = [], [], []
        for ln, d in L.items():
            for r in d["windows"]:
                les_win.append(window_summary(r))
            for grp, layers in (("layers1-3", ("layer1", "layer2", "layer3")), ("avgpool", ("avgpool",))):  # skip-one by group
                sk = [r for r in d["skip_one"] if r["layer"] in layers]
                les_skip.append({"lane": ln, "layers": grp, **skip_summary(sk)})
                for wnd in ("0->5", "5->10", "10->15"):
                    if grp == "avgpool":
                        break  # per-layer avgpool rows already written above
                    ws = [r for r in d["windows"] if r["layer"] in layers and r["window"] == wnd]
                    agg = {k: sum(r[k] for r in ws) for k in ws[0] if isinstance(ws[0][k], (int, float)) and not isinstance(ws[0][k], bool)}
                    agg.update({"lane": ln, "layer": grp, "window": wnd})
                    les_win.append(window_summary(agg))
            for r in d["steps"]:
                les_step.append(r)
        write_csv(OUT / "lesion_windows.csv", les_win)
        write_csv(OUT / "lesion_skip_one.csv", les_skip)
        write_csv(OUT / "lesion_steps.csv", les_step)
    print("done")


def skip_summary(sk):
    S = lambda k: sum(r.get(k, 0) for r in sk)  # noqa: E731
    n = S("n_alive"); c = S("n_chain_stable"); d = S("n_direct_stable"); b = S("n_both")
    out = {"n_cells": len(sk), "n_alive": n,
           "chain_stable_pct": pct(c, n), "direct_stable_pct": pct(d, n),
           "agreement_pct": pct(b + S("n_neither"), n),
           "P_direct_given_chain_pct": pct(b, c), "P_chain_given_direct_pct": pct(b, d),
           "same_endpoint_pct_of_both": pct(S("n_both_same_partner"), b),
           "chain_only_pct_alive": pct(S("n_chain_only"), n), "direct_only_pct_alive": pct(S("n_direct_only"), n),
           "direct_only_via_death_pct_of_direct_only": pct(S("direct_only_break_died"), S("n_direct_only")),
           "direct_only_via_transformed_pct_of_direct_only": pct(S("direct_only_break_transformed"), S("n_direct_only")),
           "n_died_step1": S("n_died_step1"), "n_flicker": S("n_flicker_step1"),
           "flicker_pct_of_step1_deaths": pct(S("n_flicker_step1"), S("n_died_step1")),
           "flicker_pct_alive": pct(S("n_flicker_step1"), n),
           "flicker_consensus_pct_of_consensus_deaths": pct(S("n_flicker_step1_consensus"), S("n_died_step1_consensus")),
           "blip_births_pct_of_births": pct(S("n_blip_births"), S("n_born_step1"))}
    # pooled kappa
    p1 = c / n if n else np.nan; p2 = d / n if n else np.nan
    po = (b + S("n_neither")) / n if n else np.nan
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    out["kappa"] = (po - pe) / (1 - pe) if n else np.nan
    return out


def window_summary(r):
    n = r["n_alive"]
    o = {"lane": r["lane"], "layer": r["layer"], "window": r["window"], "n_alive_start": n,
         "chain_stable_pct": pct(r["n_chain_stable"], n), "direct_stable_pct": pct(r["n_direct_stable"], n),
         "P_direct_given_chain_pct": pct(r["n_both"], r["n_chain_stable"]),
         "P_chain_given_direct_pct": pct(r["n_both"], r["n_direct_stable"]),
         "gap_any_death_pct_of_direct": pct(r["n_gap_any_death"], r["n_direct_stable"]),
         "flicker_pct_of_step1_deaths": pct(r["n_flicker_step1"], r["n_died_step1"]),
         "coarse_born": r["coarse_n_born"], "dense_sum_born": r["dense_sum_n_born"],
         "dense_births_surviving": r["dense_births_surviving_to_end"],
         "coarse_died": r["coarse_n_died"], "dense_sum_died": r["dense_sum_n_died"],
         "end_new_dense": r["end_new"], "end_new_coarse_born": r["end_new_coarse_born"],
         "end_persistent_coarse_born": r["end_persistent_coarse_born"]}
    for p in ("tg", "ab", "di"):
        o[f"coarse_f_{p}"] = r[f"coarse_f_{p}"]; o[f"dense_sum_f_{p}"] = r[f"dense_sum_f_{p}"]
        o[f"coarse_w_{p}"] = r[f"coarse_w_{p}_h"]; o[f"dense_sum_w_{p}"] = r[f"dense_sum_w_{p}_h"]
    o["coarse_CR_w"] = (o["coarse_w_tg"] + o["coarse_w_ab"]) / o["coarse_w_di"] if o["coarse_w_di"] else float("inf")
    o["dense_CR_w"] = (o["dense_sum_w_tg"] + o["dense_sum_w_ab"]) / o["dense_sum_w_di"] if o["dense_sum_w_di"] else float("inf")
    return o


if __name__ == "__main__":
    main()
