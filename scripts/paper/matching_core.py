#!/usr/bin/env python3
"""Matching diagnostics and skip-one consistency for the 9 core CIFAR-100 runs.

Per lane, layer and consecutive transition (including the k->terminal null):
  * stored n_stable / n_born / n_died / n_transformed reproduced with the pipeline's _phase5_feature_matching,
  * Hungarian diagnostics (contested births and deaths, mutual best, margins, splits and merges),
  * alternative matching rules (mutual best, greedy, confirmed): counts, Tg/Ab/Di labels, label changes relative
    to Hungarian, sample-weighted process counts (Hungarian counts checked against the stored process_intensity),
  * skip-one consistency, k->k+2 direct vs k->k+1->k+2 chained (real checkpoints only).
Writes analysis_outputs/paper/MATCHING/per_lane/<lane_id>.json and <lane_id>_margins.npz.

Usage: python scripts/paper/matching_core.py --lane <lane_id> [--lane ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matching_lib import (CORE_EXP, CORE_LANES, EPS_ACT, OUT, Labeler, Lane, chain_compare,  # noqa: E402
                    confirmed_outcome, counts, diagnostics, feature_counts, greedy_outcome,
                    label_change, mutual_best_outcome, outcome_from_pipeline, pipeline_match)

RULES = ("hungarian", "mutual_best", "greedy", "confirmed")


def run_lane(lane_id: str) -> None:
    arch_seed = {l: (a, s) for l, a, s in CORE_LANES}
    lane = Lane(CORE_EXP / "lanes" / lane_id)
    arch, seed = arch_seed[lane_id]
    assert lane.arch == arch, (lane.arch, arch)
    labels = lane.labels
    real = [l for l in labels if l != "terminal"]
    assert labels[-1] == "terminal" and len(real) == len(labels) - 1
    lab = Labeler(lane)
    rows, skip_rows = [], []
    margins = {}
    weighted_by_pair = {}  # pair -> rule -> {layer: counts}
    t0 = time.time()
    for layer in lane.layers:
        H = {l: lane.load(l, layer) for l in labels}
        act = {l: np.abs(H[l]) > EPS_ACT for l in labels}
        sel = {l: lane.sel(l, layer) for l in labels}
        hung_out = {}
        for k, (la, lb) in enumerate(zip(labels[:-1], labels[1:])):
            pair = f"{la}->{lb}"
            md = pipeline_match(H[la], H[lb])
            stored = lane.stored_counts(pair, layer)
            ho = outcome_from_pipeline(md)
            hung_out[pair] = ho
            ia = np.where(md["alive_a"])[0]; ib = np.where(md["alive_b"])[0]
            Cs = md["correlation_matrix"][np.ix_(ia, ib)]
            outs = {"hungarian": ho,
                    "mutual_best": mutual_best_outcome(Cs, ia, ib),
                    "greedy": greedy_outcome(Cs, ia, ib),
                    "confirmed": confirmed_outcome(ho, Cs, ia, ib)}
            diag, arrs = diagnostics(md, ho)
            row = {"lane": lane_id, "arch": arch, "seed": seed, "layer": layer, "pair": pair, "k": k,
                   "is_terminal": lb == "terminal"}
            rep = {key: md[key] for key in ("n_stable", "n_born", "n_died", "n_transformed")}
            for key, v in rep.items():
                row[f"rep_{key}"] = int(v)
                row[f"stored_{key}"] = int(stored[key]) if stored else None
            row["counts_match"] = bool(stored) and all(int(stored[k2]) == int(v) for k2, v in rep.items())
            row.update({f"diag_{k2}": v for k2, v in diag.items()})
            labs = {}
            for rname, o in outs.items():
                c = counts(o)
                row.update({f"{rname}_{k2}": v for k2, v in c.items()})
                labs[rname] = lab.labels(o, sel[la], sel[lb])
                row.update({f"{rname}_{k2}": v for k2, v in feature_counts(labs[rname]).items()})
                w = lab.weighted(o, labs[rname], act[la], act[lb])
                row.update({f"{rname}_w_{k2}": v for k2, v in w.items()})
                weighted_by_pair.setdefault(pair, {}).setdefault(rname, {})[layer] = w
                if rname != "hungarian":
                    chg = label_change(labs["hungarian"], labs[rname], ib)
                    row.update({f"{rname}_lc_{k2}": v for k2, v in chg.items()})
                    # partner changes among pairs stable under both
                    if rname in ("mutual_best", "greedy"):
                        both = set(ho["stable"]) & set(o["stable"])
                        row[f"{rname}_stable_same_partner"] = sum(ho["stable"][i][0] == o["stable"][i][0] for i in both)
                        row[f"{rname}_stable_overlap"] = len(both)
            rows.append(row)
            key = f"{layer}|{pair}"
            margins[key] = arrs
        # skip-one (real checkpoints only)
        for k in range(len(real) - 2):
            la, lm, lb = real[k], real[k + 1], real[k + 2]
            md2 = pipeline_match(H[la], H[lb])
            direct = outcome_from_pipeline(md2)
            ia = np.where(md2["alive_a"])[0]
            prev = hung_out[f"{real[k-1]}->{la}"] if k > 0 else None
            res = chain_compare([hung_out[f"{la}->{lm}"], hung_out[f"{lm}->{lb}"]], direct, ia, prev=prev)
            # blip births: born at la->lm, died at lm->lb
            born1 = hung_out[f"{la}->{lm}"]["born"]
            died2 = hung_out[f"{lm}->{lb}"]["died"]
            res.update({"lane": lane_id, "arch": arch, "seed": seed, "layer": layer,
                        "window": f"{la}->{lb}", "k": k,
                        "direct_n_stable": md2["n_stable"], "direct_n_born": md2["n_born"],
                        "direct_n_died": md2["n_died"], "direct_n_transformed": md2["n_transformed"],
                        "n_born_step1": len(born1), "n_blip_births": len(born1 & died2)})
            skip_rows.append(res)
        del H, act
        print(f"  [{lane_id[:8]}] {layer} done ({time.time()-t0:.0f}s)", flush=True)

    # process_intensity replication (sum over all layers, as the pipeline does)
    pi_stored = {e["transition"]: e for e in lane.results.get("process_intensity", [])}
    pi_rows = []
    for pair, per_rule in weighted_by_pair.items():
        for rname, per_layer in per_rule.items():
            for dedup in (False, True):
                tot = {}
                for layer, w in per_layer.items():
                    if dedup and arch == "ResNet-18" and layer == "layer4":
                        continue
                    for k2, v in w.items():
                        tot[k2] = tot.get(k2, 0) + v
                r = {"lane": lane_id, "arch": arch, "seed": seed, "pair": pair, "rule": rname,
                     "dedup_layer4": dedup, "is_terminal": pair.endswith("terminal"), **tot}
                if rname == "hungarian" and not dedup and pair in pi_stored:
                    s = pi_stored[pair]
                    r["pi_match"] = all(int(s[p]) == int(tot[p]) for p in ("tg_h", "ab_h", "di_h", "as_h", "de_h"))
                    r.update({f"stored_{p}": s[p] for p in ("tg_h", "ab_h", "di_h", "as_h", "de_h")})
                pi_rows.append(r)
    outdir = OUT / "per_lane"
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / f"{lane_id}.json", "w") as f:
        json.dump({"transitions": rows, "skip_one": skip_rows, "process": pi_rows,
                   "ati": lane.ati, "labels": labels, "layers": lane.layers}, f, default=float)
    flat = {}
    for key, arrs in margins.items():
        for nm, a in arrs.items():
            flat[f"{key}|{nm}"] = np.asarray(a, dtype=np.float32)
    np.savez_compressed(outdir / f"{lane_id}_margins.npz", **flat)
    n_ok = sum(r["counts_match"] for r in rows)
    n_pi = [r["pi_match"] for r in pi_rows if "pi_match" in r]
    print(f"[{lane_id[:8]}] {arch} s{seed}: counts match {n_ok}/{len(rows)}; process_intensity match "
          f"{sum(n_pi)}/{len(n_pi)}; {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", action="append", required=True)
    a = ap.parse_args()
    for l in a.lane:
        run_lane(l)


if __name__ == "__main__":
    main()
