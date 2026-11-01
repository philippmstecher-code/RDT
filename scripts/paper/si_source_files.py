#!/usr/bin/env python3
"""Producer of the Supplementary Information source files that are assembled from other analysis outputs.

  SI/stored_thresholds_per_run.csv            each run's stored pipeline thresholds (SSI, CSI, SAI) by experiment
  SI/transformed_treatments.csv   ordering, C/R and persistence per run under three treatments of
                                              transformed matches (from TRANSFORMED/transformed_results.json)
  SI/same_checkpoint_lifespan.csv coarse vs fine lifespan per run (all) and birth-stratified hazard
                                              ratio (THRESHOLD_impact/fig4_stored_sai.json)
  SI/superclass_process_fractions_per_run.csv, ..._arch_means.csv
                                              process fractions per superclass, 9 core runs, final transition excluded
  SHAM_perepoch/flicker_density.json          flicker and dense/coarse birth ratios per lane
                                              (MATCHING/per_lane/clean_density.json, from checkpoint_density.py)
  THRESHOLD_impact/thresholds_stored_events.json           total differentiation relative to standard and superclass-phase share,
                                              curriculum sweep (CURRICULUM_threshold_free/tables/B_per_lane.csv)

Usage: python scripts/paper/si_source_files.py [--out DIR]   (DIR replaces analysis_outputs/paper)
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "analysis_outputs/paper"
RUNS_EXTENDED = Path(RDT_DATA + "/extended/experiments")
sys.path.insert(0, str(REPO / "scripts/paper"))
sys.path.insert(0, str(REPO / "scripts/paper/figures"))


def stored_thresholds(out):
    rows = []
    names = {"EN": "fixed label noise", "EC": "curriculum",
             "SUP": "suppression from birth, epochs 1-10", "RN": "label noise redrawn every epoch",
             "SW": "curriculum sweep", "PT": "ImageNet pre-trained fine-tuning"}
    for exp, desc in names.items():
        T = json.load(open(RESULTS / "SAE_outcomes" / f"thresholds_{exp}.json"))["stored"]
        for k, v in T.items():
            arm, s = k.rsplit("__s", 1)
            t = v["thresholds"]
            rows.append({"experiment": desc, "condition": arm, "seed": int(s), "SSI": t["ssi_abstraction_thresh"],
                         "CSI": t["csi_differentiation_thresh"], "SAI": t["sai_task_general_thresh"]})
    grp = {"core": "core runs (Figs 1-4)", "tinyimagenet": "Tiny ImageNet",
           "indep_init": "independent SAE initialisation"}
    for r in pd.read_csv(RESULTS / "CENSUS_robust/tables/thresholds_per_run.csv").itertuples():
        rows.append({"experiment": grp[r.group], "condition": r.arch, "seed": int(r.lane.split("seed")[1].split("-")[0]),
                     "SSI": r.stored_thr_ssi, "CSI": r.stored_thr_csi, "SAI": r.stored_thr_sai})
    for pat, exp, cond in (("taskirr/lanes/taskirr_within_index_seed*", "superclass-blind task",
                            "task-cutting labels"),
                           ("sham_perepoch/lanes/sham_e00_seed*", "clean per-epoch runs", "standard")):
        for d in sorted(glob.glob(str(RUNS_EXTENDED / pat))):
            t = json.load(open(d + "/sae_analysis/sae_results.json"))["metadata"]["analysis_thresholds"]
            rows.append({"experiment": exp, "condition": cond, "seed": int(d.split("seed")[-1]),
                         "SSI": t["ssi_abstraction_thresh"], "CSI": t["csi_differentiation_thresh"],
                         "SAI": t["sai_task_general_thresh"]})
    for x in json.load(open(RESULTS / "THRESHOLD_impact/dense_thresholds.json")):
        t = x["stored"]
        rows.append({"experiment": "dense first-epoch runs", "condition": "standard", "seed": x["seed"],
                     "SSI": t["ssi_abstraction_thresh"], "CSI": t["csi_differentiation_thresh"],
                     "SAI": t["sai_task_general_thresh"]})
    df = pd.DataFrame(rows).round(4).sort_values(["experiment", "condition", "seed"])
    df.to_csv(out / "SI/stored_thresholds_per_run.csv", index=False)


def supp_table1(out):
    r = json.load(open(RESULTS / "TRANSFORMED/transformed_results.json"))
    P, Q = r["processes"]["paper_all_layers"], r["persistence"]
    rows = []
    for tr in ("excluded", "turnover", "continuity"):
        for a, b in zip(P[tr]["per_lane"], Q[tr]["per_lane"]):
            assert a["arch"] == b["arch"] and a["seed"] == b["seed"]
            crf = a["cr_first"] if a["cr_first"] is not None else np.inf
            rows.append({"treatment": tr, "arch": a["arch"], "seed": a["seed"], "com_order_Tg<Ab<Di": a["com_order_holds"],
                         "peak_order": a["peak_order_holds"], "cr_first": a["cr_first"], "cr_last_real": a["cr_last_real"],
                         "cr_declines": bool(crf > a["cr_last_real"]), "coarse_over_fine_lifespan": b["constr_over_csi"],
                         "sai_over_csi": b["sai_over_csi"], "ssi_over_csi": b["ssi_over_csi"]})
    pd.DataFrame(rows).to_csv(out / "SI/transformed_treatments.csv", index=False)


def supp_table2(out):
    r = json.load(open(RESULTS / "THRESHOLD_impact/fig4_stored_sai.json"))["fig4a"]
    df = pd.DataFrame([{"arch": x["arch"], "seed": x["seed"], "ratio_all": round(x["sai_stored"]["ratio"], 2),
                        "HR_birth_stratified": round(x["sai_stored"]["hr_birth"], 2),
                        "P": float("%.2g" % x["sai_stored"]["hr_birth_p"]),
                        "n_coarse": x["sai_stored"]["n_coarse"], "n_fine": x["sai_stored"]["n_fine"]} for x in r])
    df.to_csv(out / "SI/same_checkpoint_lifespan.csv", index=False)


def supp_fig3(out):
    import core_runs_lib as L
    from process_composition import CORE, CORE_EXP
    rows = []
    for arch, seed, pre in CORE:
        for sc, v in L.superclass_fractions(CORE_EXP, pre, drop=True).items():
            rows.append({"arch": arch, "seed": seed, "superclass": sc, **{p: float(x) for p, x in zip(L.PROCS5, v)}})
    df = pd.DataFrame(rows)
    df.to_csv(out / "SI/superclass_process_fractions_per_run.csv", index=False)
    df.groupby(["superclass", "arch"])[list(L.PROCS5)].mean().reset_index().to_csv(
        out / "SI/superclass_process_fractions_arch_means.csv", index=False)


def flicker_density(out):
    from matching_aggregate import skip_summary
    L = json.load(open(RESULTS / "MATCHING/per_lane/clean_density.json"))
    res = {}
    for ln, d in L.items():
        fl = {}
        for grp, layers in (("layers1-3", ("layer1", "layer2", "layer3")), ("avgpool", ("avgpool",))):
            fl[grp] = round(skip_summary([r for r in d["skip_one"] if r["layer"] in layers])["flicker_pct_alive"], 2)
        ratios, surv = [], []
        for w in ("0->5", "5->10", "10->15"):
            ws = [r for r in d["windows"] if r["layer"] in ("layer1", "layer2", "layer3") and r["window"] == w]
            cb = sum(r["coarse_n_born"] for r in ws)
            db = sum(r["dense_sum_n_born"] for r in ws)
            sv = sum(r["dense_births_surviving_to_end"] for r in ws)
            ratios.append(round(db / cb, 2) if cb else None)
            surv.append(round(100 * sv / db, 1) if db else None)
        res[ln] = {"flicker_pct_alive_all": round(skip_summary(d["skip_one"])["flicker_pct_alive"], 2),
                   "flicker_by_group": fl, "dense_over_coarse_births_l13": ratios, "dense_births_surviving_pct_l13": surv}
    json.dump(res, open(out / "SHAM_perepoch/flicker_density.json", "w"), indent=1)


def thresholds_stored_events(out):
    b = pd.read_csv(RESULTS / "CURRICULUM_threshold_free/tables/B_per_lane.csv")
    K = ("switch_e05", "switch_e10", "switch_e15", "switch_e20", "switch_e25", "switch_e30")
    res = {}
    for m in ("own_runspecific", "B2b"):
        x = b[b.measure == m].set_index("cond")
        rel = x.real_all_total / x.loc["standard50", "real_all_total"]
        sh_excl = x.real_all_superclass / x.real_all_total * 100
        sh_incl = (x.real_all_superclass + x.real_all_switch) / x.real_all_total * 100
        res[m] = {c: {"rel_total": round(rel[c], 3), "share_sc_excl_switch": round(sh_excl[c], 1),
                      "share_sc_incl_switch": round(sh_incl[c], 1)} for c in K}
    json.dump(res, open(out / "THRESHOLD_impact/thresholds_stored_events.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS))
    out = Path(ap.parse_args().out)
    for sub in ("SI", "SHAM_perepoch", "THRESHOLD_impact"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for fn in (stored_thresholds, supp_table1, supp_table2, supp_fig3, flicker_density, thresholds_stored_events):
        fn(out)
        print("written:", fn.__name__, flush=True)


if __name__ == "__main__":
    main()
