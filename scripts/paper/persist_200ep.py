#!/usr/bin/env python3
"""200-epoch ablation of the feature-lifespan profile: overall survival, SSI/CSI lifespan ratio and SAI survival
ratio in the 50-epoch and 200-epoch runs.

Compares ResNet-18 CIFAR-100 seed 42, 50 epochs (abf16d20/7db28a9a, core run) with the
200-epoch run of the same model (ce93d3f6/c645c39e; label 'ResNet18-CIFAR100-200ep-seed42'
in scripts/compute_feature_survival*.py).

Steps
 1. Build the matching cache for the 200-epoch lane (pipeline's own Phase-5 matching, via
    scripts/paper/transformed_matching_cache.process_lane) if missing.
 2. Reproduce the stored reference values (REFERENCE) with the reference construction
    (persist_lib.records_reference: duplicate '<k>->terminal' transition included,
    non-alive features included, all layers) under several candidate definitions.
 3. Recompute under the real-transition construction 'stored|paper'
    (persist_lib.records_real(rule='paper'): duplicate transition removed, pipeline alive
    mask, ResNet avgpool dropped; stored within-run thresholds).

Writes analysis_outputs/paper/PERSIST_robust/ep200.json. Read-only on the data directory.
Usage: python scripts/paper/persist_200ep.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from persist_lib import (CACHE, DATA, OUT, records_real, records_reference,  # noqa: E402
                         terminal_persistence, terminal_persistence_reference)

RUNS = {
    "50ep": ("abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb", "7db28a9a-f436-4d8a-a6b4-37e69fbf54eb"),
    "200ep": ("ce93d3f6-282b-4811-84f8-be6c589c0500", "c645c39e-e21f-4cff-853f-967bf69ee857"),
}
STORED_LABEL = {"50ep": "ResNet18-CIFAR100-seed42", "200ep": "ResNet18-CIFAR100-200ep-seed42"}
REFERENCE = {"survival_pct": (22.0, 22.1), "ssi_over_csi": (1.49, 1.45), "sai_over_csi": (1.35, 2.17)}
C = {k: i for i, k in enumerate(("layer_i", "birth", "fid", "life", "maxp", "cens", "ssi", "csi", "sai",
                                  "chain_start"))}


def ensure_cache(exp: str, lane: str):
    dst = CACHE / f"{lane}.pkl"
    if dst.exists():
        return
    import transformed_matching_cache as a5
    a5.CORE_EXP = DATA / "experiments" / exp  # process_lane reads this module global
    print(f"building matching cache for {lane} ...", flush=True)
    d = a5.process_lane(lane, "ResNet-18", 42)
    print(f"   mismatches vs sae_results.json: {len(d['mismatches'])}", flush=True)
    with open(dst, "wb") as f:
        pickle.dump(d, f)


def load(exp: str, lane: str):
    ensure_cache(exp, lane)
    d = pickle.load(open(CACHE / f"{lane}.pkl", "rb"))
    res = json.loads((DATA / "experiments" / exp / "lanes" / lane / "sae_analysis" / "sae_results.json").read_text())
    return d, res


def stored_thr(res):
    at = res["metadata"]["analysis_thresholds"]
    return {"sai": at["sai_task_general_thresh"], "ssi": at["ssi_abstraction_thresh"],
            "csi": at["csi_differentiation_thresh"]}


def masks(R, thr, exclusive):
    ssi, csi, sai = R[:, C["ssi"]], R[:, C["csi"]], R[:, C["sai"]]
    A, B, Cc = sai > thr["sai"], ssi > thr["ssi"], csi > thr["csi"]
    if exclusive:
        return A, B & ~A, Cc & ~A & ~B
    return A, B, Cc


def mean_ratio(x, a, b):
    return float(x[a].mean() / x[b].mean()) if a.any() and b.any() and x[b].mean() > 0 else float("nan")


def candidates(R, T, thr, n_t_total):
    """All candidate statistics for one record set R and terminal-persistence set T."""
    life, cens = R[:, C["life"]], R[:, C["cens"]]
    out = {"n_records": int(len(R)), "survival_pct_records": 100 * float(cens.mean()),
           "n_features_unique": int(len({(int(l), int(f)) for l, f in zip(R[:, C["layer_i"]], R[:, C["fid"]])}))}
    b0 = R[:, C["birth"]] == 0
    out["survival_pct_born_at_0"] = 100 * float(cens[b0].mean())
    # Task-general (SAI > 0.9) vs rest, as the 'tg' vs 'non_tg' cohorts of feature_survival_tg_expanded.json
    tg = R[:, C["sai"]] > 0.9
    out["tg_vs_nontg|surv|SAI>0.9/SAI<=0.9"] = mean_ratio(cens, tg, ~tg)
    out["tg_vs_nontg|life|SAI>0.9/SAI<=0.9"] = mean_ratio(life, tg, ~tg)
    out["tg_vs_nontg|surv_pct|tg,nontg"] = [100 * float(cens[tg].mean()), 100 * float(cens[~tg].mean())]
    out["tg_vs_nontg|n|tg,nontg"] = [int(tg.sum()), int((~tg).sum())]
    for thr_name, th in (("stored", thr), ("stored_sai0.9", {**thr, "sai": 0.9})):
        for excl in (False, True):
            A, B, Cc = masks(R, th, excl)
            tag = f"{thr_name}|{'excl' if excl else 'overlap'}"
            # mean lifespan ratios over all records
            out[f"{tag}|life|SSI/CSI"] = mean_ratio(life, B, Cc)
            out[f"{tag}|life|SAI/CSI"] = mean_ratio(life, A, Cc)
            # survival-rate ratios
            out[f"{tag}|surv|SSI/CSI"] = mean_ratio(cens, B, Cc)
            out[f"{tag}|surv|SAI/CSI"] = mean_ratio(cens, A, Cc)
            # records reaching terminal, lifespan/n_t
            tm = cens == 1
            out[f"{tag}|fig4c|SSI/CSI"] = mean_ratio(life, B & tm, Cc & tm)
            out[f"{tag}|fig4c|SAI/CSI"] = mean_ratio(life, A & tm, Cc & tm)
            # backward-traced developmental persistence of terminal features
            ssi, csi, sai, p = T[:, 2], T[:, 3], T[:, 4], T[:, 1]
            tA, tB, tC = sai > th["sai"], ssi > th["ssi"], csi > th["csi"]
            if excl:
                tB, tC = tB & ~tA, tC & ~tA & ~tB
            out[f"{tag}|termpers|SSI/CSI"] = mean_ratio(p, tB, tC)
            out[f"{tag}|termpers|SAI/CSI"] = mean_ratio(p, tA, tC)
            out[f"{tag}|n|SAI,SSI,CSI"] = [int(A.sum()), int(B.sum()), int(Cc.sum())]
    return out


def headline_stats(R, thr):
    life, cens = R[:, C["life"]], R[:, C["cens"]]
    A, B, Cc = masks(R, {**thr, "sai": 0.9}, exclusive=False)
    return np.array([100 * cens.mean(), mean_ratio(life, B, Cc), mean_ratio(cens, A, ~A)])


def bootstrap(R, thr, n_boot=2000, seed=0):
    """Record-level bootstrap (ignores dependence between records of the same chain, so CIs are
    optimistic). Returns (n_boot, 3) array: survival %, SSI/CSI life ratio, SAI>0.9 survival ratio."""
    rng = np.random.default_rng(seed)
    return np.array([headline_stats(R[rng.integers(0, len(R), len(R))], thr) for _ in range(n_boot)])


def main():
    stored_json = json.loads((DATA / "feature_survival_all_lanes.json").read_text())
    stored_tg = json.loads((DATA / "feature_survival_tg_expanded.json").read_text())
    result = {"lanes": {k: {"experiment": e, "lane": l} for k, (e, l) in RUNS.items()},
              "overlapping": REFERENCE, "stored_original_outputs": {}, "original": {}, "corrected": {}}
    boots = {}
    for run, (exp, lane) in RUNS.items():
        d, res = load(exp, lane)
        thr = stored_thr(res)
        n_t = len(d["labels"]) - 1
        s, t = stored_json[STORED_LABEL[run]], stored_tg[STORED_LABEL[run]]["cohorts"]
        result["stored_original_outputs"][run] = {
            "source": "feature_survival_all_lanes.json / feature_survival_tg_expanded.json",
            "overall_survival_rate": s["overall_survival_rate"], "n_total_tracked": s["n_total_tracked"],
            "mean_life_high_ssi": t["high_ssi"]["mean_lifespan"], "mean_life_high_csi": t["high_csi"]["mean_lifespan"],
            "mean_life_tg": t["tg"]["mean_lifespan"],
            "high_ssi_label": t["high_ssi"]["name"], "high_csi_label": t["high_csi"]["name"]}
        Ro = records_reference(d, res)
        To = terminal_persistence_reference(d, res)
        result["original"][run] = {"thresholds": thr, **candidates(Ro, To, thr, n_t)}
        Rc = records_real(d, res, "paper")
        Tc = terminal_persistence(d, res, "paper")
        result["corrected"][run] = {"thresholds": thr, **candidates(Rc, Tc, thr, n_t - 1)}
        boots[("original", run)] = bootstrap(Ro, thr)
        boots[("corrected", run)] = bootstrap(Rc, thr)
        print(run, "reference n", len(Ro), "stored n", s["n_total_tracked"], "| real-transition n", len(Rc), flush=True)

    keys = [k for k in result["original"]["50ep"] if k != "thresholds"]
    print(f"\n{'statistic':<40}{'ref 50':>10}{'ref 200':>10}{'real 50':>10}{'real 200':>10}")
    for k in keys:
        v = [result[s][r][k] for s in ("original", "corrected") for r in ("50ep", "200ep")]
        if isinstance(v[0], list):
            print(f"{k:<40}" + "".join(f"{str(x):>18}" for x in v))
        else:
            print(f"{k:<40}" + "".join(f"{x:>10.3f}" for x in v))
    result["_keys"] = keys

    # headline: definitions that reproduce the stored reference values, recomputed under the real-transition construction
    defs = {
        "survival_pct": ("survival_pct_records",
                         "% of feature records (every feature x checkpoint it is tracked at) whose forward "
                         "chain of stable matches reaches the final checkpoint"),
        "ssi_over_csi": ("stored|overlap|life|SSI/CSI",
                         "mean lifespan (n transitions survived) of records with SSI > stored t_ssi divided by "
                         "that of records with CSI > stored t_csi (overlapping cohorts)"),
        "sai_over_nontg": ("tg_vs_nontg|surv|SAI>0.9/SAI<=0.9",
                           "survival-to-final-checkpoint rate of records with SAI > 0.9 divided by that of "
                           "records with SAI <= 0.9 (NOT relative to CSI; 'tg' vs 'non_tg' cohorts of "
                           "feature_survival_tg_expanded.json)"),
        "sai_over_csi_fig4a_style": ("stored_sai0.9|overlap|life|SAI/CSI",
                                     "mean lifespan SAI > 0.9 / CSI > stored t_csi (overlapping; does NOT "
                                     "reproduce the reference values 1.35/2.17, given for comparison)"),
    }
    head = {}
    for name, (key, desc) in defs.items():
        head[name] = {"definition": desc, "key": key,
                      "reference_50ep_200ep": REFERENCE.get(name if name != "sai_over_nontg" else "sai_over_csi"),
                      "original_50ep_200ep": [round(result["original"][r][key], 3) for r in ("50ep", "200ep")],
                      "corrected_50ep_200ep": [round(result["corrected"][r][key], 3) for r in ("50ep", "200ep")]}
    head["sai_over_csi_fig4a_style"]["reference_50ep_200ep"] = None
    # record-level bootstrap 95% CIs and 200ep-minus-50ep differences (optimistic: records not independent)
    for i, name in enumerate(("survival_pct", "ssi_over_csi", "sai_over_nontg")):
        for s in ("original", "corrected"):
            b50, b200 = boots[(s, "50ep")][:, i], boots[(s, "200ep")][:, i]
            diff = b200 - b50
            head[name][f"{s}_bootstrap"] = {
                "ci95_50ep": [round(float(x), 3) for x in np.nanpercentile(b50, [2.5, 97.5])],
                "ci95_200ep": [round(float(x), 3) for x in np.nanpercentile(b200, [2.5, 97.5])],
                "ci95_diff_200_minus_50": [round(float(x), 3) for x in np.nanpercentile(diff, [2.5, 97.5])],
                "p_two_sided_diff": round(float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))), 4)}
    result["headline"] = head
    result["notes"] = [
        "Reference values '22.1% vs 22.0%' are 200ep, 50ep (reference construction: 50ep 21.98%, 200ep 22.06%; stored JSON 0.2198 / 0.2208).",
        "original = reference construction, persist_lib.records_reference (duplicate '<k>->terminal' transition, non-alive features, "
        "all layers incl. ResNet avgpool); n records match feature_survival_all_lanes.json (10096 / 9923).",
        "corrected = real-transition construction, persist_lib.records_real(rule='paper') ('stored|paper'): last real checkpoint is the "
        "censoring point, alive features only, avgpool dropped; stored within-run thresholds, SAI cut 0.9.",
        "200ep matching cache built with the pipeline's Phase-5 matching: 1 pair differs from sae_results.json "
        "(2->3 layer4: n_stable 135 vs 134, n_transformed 84 vs 85).",
    ]
    print(json.dumps(head, indent=1))
    (OUT / "ep200.json").write_text(json.dumps(result, indent=1, default=float))
    return result


if __name__ == "__main__":
    main()
