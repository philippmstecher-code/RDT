#!/usr/bin/env python3
"""SAE totals and reconstruction quality over all SAE-tracked runs of Extended Data Table 1 (Methods).

Runs: analysis_outputs/run_inventory/edtable1_scan.json (155 runs; the 10 sham runs have no SAEs). Per run, from
sae_analysis/sae_results.json: reconstruction_quality (cosine similarity of every SAE, one per checkpoint and layer)
and feature_landscape (n_alive per SAE).
Writes analysis_outputs/paper/SI/sae_totals.json: number of runs and SAEs, alive features summed over SAEs,
reconstruction cosine (mean, s.d. with ddof 1, minimum, number <= 0.93) overall and per experiment block, and the same
totals for the analyses as the Methods count them ("analyses"): the five 'Fine classes only, 30' runs, which are the
same networks as the fixed-label 'No noise' runs (identical training records), are counted once, and the per-epoch SAE
analyses are added: five sham runs (sham_perepoch, Supplementary Fig. 1; Fig. 5) and, of the acquisition-blocking
runs, the superclass, random-grouping, random-direction, energy-matched and jointly matched arms of all five seeds and
the pair confined to layer3 onwards (blocking_rdt, epochs 0-15; Fig. 5; ED Fig. 7c; ED Table 1). The layer2 pair
(*_l2_*) is not reported and not counted. Runs left out of ED Table 1 (scratch E20 and the redrawn
random-direction block) are not counted. The 30-epoch pre-training arms (K30) and the two five-label
pre-training arms (rand5_pretrain, sup5_pretrain) are in ED Table 1 and counted.
Usage: python scripts/paper/sae_totals.py
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

RUNS_EXTENDED = Path(RDT_DATA + "/extended/experiments")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis_outputs/paper/SI/sae_totals.json"


def read(lane, key):
    j = json.load(open(os.path.join(lane, "sae_analysis", "sae_results.json")))
    cos = [v["cosine_sim"] for ck in j.get("reconstruction_quality", {}).values() for v in ck.values()
           if isinstance(v, dict) and "cosine_sim" in v]
    alive = [v["n_alive"] for ck in j.get("feature_landscape", {}).values() for v in ck.values()
             if isinstance(v, dict) and "n_alive" in v]
    return {"key": key, "lane": lane, "cos": cos, "alive": sum(alive)}


def training_record(lane):
    return [(e["epoch"], e["val_accuracy"], e.get("train_loss"))
            for e in json.load(open(Path(lane) / "dev_snapshots/metrics/devtrain_metrics.json"))]


def is_duplicate(entry):
    """'Fine classes only, 30' run with the same training record as the fixed-label no-noise run of its seed."""
    if entry["key"] != "ctrl|scratch_E30":
        return False
    seed = int(entry["lane"].rsplit("seed", 1)[1])
    return training_record(entry["lane"]) == training_record(RUNS_EXTENDED / f"static_noise_s{seed}/lanes/standard_resnet18")


BLOCK_RDT_ARMS = ("gradsc", "gradrg", "gradrand", "gradenergy", "gradjoint", "gradsc_l3", "gradjoint_l3")
NOT_REPORTED = {"ctrl|scratch_E20", "gradblock|randfresh"}


def perepoch_analyses():
    """Per-epoch SAE analyses outside ED Table 1's SAE column: sham runs and the reported blocking arms."""
    out = [read(str(d), "perepoch|sham") for d in sorted((RUNS_EXTENDED / "sham_perepoch/lanes").glob("sham_e00_seed*"))]
    for arm in BLOCK_RDT_ARMS:
        for d in sorted((RUNS_EXTENDED / "blocking_rdt/lanes").glob(f"{arm}_seed*")):
            out.append(read(str(d), f"perepoch|{arm}"))
    assert len(out) == 5 + 5 * len(BLOCK_RDT_ARMS), len(out)
    return out


def added_runs():
    """Runs outside the run scan that carry SAEs: the superclass-blind pre-training arm of the
    controlled curriculum (blind_pretrain, five seeds). The placement-given runs and the redrawn random-direction
    blocking runs are not traced (RDT '-')."""
    out = [read(str(d), "ctrl|sc_blind_K20") for d in sorted((RUNS_EXTENDED / "blind_pretrain/lanes").glob("sc_blind_K20_seed*"))]
    for arm in ("rand5", "sup5"):     # five-label pre-training
        out += [read(str(d), f"ctrl|sc_{arm}_K20")
                for d in sorted((RUNS_EXTENDED / f"{arm}_pretrain/lanes").glob(f"sc_{arm}_K20_seed*"))]
    assert len(out) == 15, len(out)
    return out


def summary(rows):
    c = np.concatenate([x["cos"] for x in rows])
    return {"n_analyses": len(rows), "n_saes": int(len(c)), "alive_sum": int(sum(x["alive"] for x in rows)),
            "cos_mean": round(float(c.mean()), 4), "cos_sd": round(float(c.std(ddof=1)), 4),
            "cos_min": round(float(c.min()), 4), "frac_gt_0.93": round(float((c > 0.93).mean()), 4)}


def main():
    scan = json.load(open(REPO / "analysis_outputs/run_inventory/edtable1_scan.json"))
    rows = [read(r["lane"], r["key"]) for r in scan if r["key"] not in NOT_REPORTED
            and os.path.exists(os.path.join(r["lane"], "sae_analysis", "sae_results.json"))] + added_runs()
    allc = np.concatenate([x["cos"] for x in rows])
    out = {"source": "sae_results.json of the SAE-tracked runs of ED Table 1 (edtable1_scan.json, plus the superclass-blind and the five-label pre-training runs); sham runs have no SAEs",
           "n_runs": len(rows), "n_saes": int(len(allc)),
           "alive_feature_observations_sum_over_saes": int(sum(x["alive"] for x in rows)),
           "recon_cosine": {"all": {"mean": round(float(allc.mean()), 4), "sd": round(float(allc.std(ddof=1)), 4),
                                    "min": round(float(allc.min()), 4), "n_le_0.93": int((allc <= 0.93).sum()),
                                    "frac_gt_0.93": round(float((allc > 0.93).mean()), 4)}}}
    by = defaultdict(list)
    for x in rows:
        by[x["key"].split("|")[0]] += x["cos"]
    for g, c in sorted(by.items()):
        c = np.array(c)
        out["recon_cosine"][g] = {"n": len(c), "mean": round(float(c.mean()), 4), "sd": round(float(c.std(ddof=1)), 4),
                                  "min": round(float(c.min()), 4)}
    perepoch = perepoch_analyses()
    kept = [x for x in rows if not is_duplicate(x)]
    out["analyses"] = {**summary(kept + perepoch), "n_duplicates_excluded": len(rows) - len(kept),
                       "n_perepoch_added": len(perepoch)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "recon_cosine"}), out["recon_cosine"]["all"])


if __name__ == "__main__":
    main()
