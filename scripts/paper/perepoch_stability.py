#!/usr/bin/env python3
"""Stability of features matched as stable over a 5-epoch interval, followed at every epoch in between.

Inputs
  Per-epoch clean runs (ResNet-18, CIFAR-100; seeds 7, 42, 137; one checkpoint per epoch, epochs 0-15):
    /data/rdt/extended/experiments/sham_perepoch/lanes/sham_e00_seed{s}/sae_analysis/
      activations/<epoch>/<layer>.npy   per-checkpoint SAE codes H (10,000 test images x d_hidden, float32)
      sae_results.json                  stored pipeline output (feature_matching counts; metadata thresholds)
      transitions/<e>_to_<e+1>/<layer>.json   stored per-sample event lists (stable fid_a -> fid_b pairs)
  analysis_outputs/paper/figures/final/source_data/supp1_b.csv, supp1_c.csv (Source Data of Supplementary Fig. 1)
  analysis_outputs/paper/SHAM_perepoch/results.json (counts behind Supplementary Fig. 1)

Rules (as packages/ml/src/saeanalysis.py::_phase5_feature_matching)
  alive: |h| > 1e-6 on >= max(1, 1% of samples); Pearson r of activation columns (float32, centred, normalised with
  eps 1e-10); Hungarian (scipy linear_sum_assignment on -r) over alive x alive; assigned pair r >= 0.5 stable,
  0.2 <= r < 0.5 transformed, r < 0.2 died; alive A features left unassigned died; alive B features not claimed by
  a stable or transformed pair born. Thresholds read from each run's metadata (0.5 / 0.2 in all three runs).
  All five SAE layers the pipeline analysed (avgpool, layer1-4).

(a) For intervals 0->5, 5->10, 10->15: direct Hungarian match epoch e vs e+5. Every feature that the direct match
    classes stable (partner j) is followed along the consecutive Hungarian matches e->e+1->...->e+5:
      intact            stable at all five steps and the composed chain ends at j
      different_partner stable at all five steps but the chain ends at another epoch-(e+5) feature
      transformed       first non-stable step is a transformed match
      death             first non-stable step is a death (r < 0.2 or unassigned)
    Extra columns: the epoch of the first break; the "any-link" chain (follow stable or transformed links, stop at a
    death) and whether it ends at j; for broken chains, the backward lineage of j (follow consecutive stable or
    transformed links backwards from e+5) and the epoch at which it starts (birth), if inside the interval.
    Secondary: of all features with an all-stable consecutive chain e->e+5, how the direct match classes them.
    Validation: consecutive matches recomputed for all 15 transitions x 5 layers per run and compared with
    sae_results.json['feature_matching'] (n_stable, n_born, n_died, n_transformed); stable pairs additionally
    checked against the stored per-sample transition files for transitions 4->5 and 9->10.
(b) Centre of mass (transition index t = epoch t -> t+1, weighted by event count, as supp1.py) of Tg-E, Ab-E, Di-E
    per run from supp1_c.csv, recomputed from SHAM_perepoch/results.json counts; ordered = Tg < Ab < Di with both
    gaps >= 0.05 transitions (smaller gaps count as ties, i.e. NOT ordered).
(c) From supp1_b.csv ('Ab-E events', 'Di-E events' by seed, transition_index): per run and for the mean of the three
    runs' counts, the first transition with Di-E > Ab-E, and the first transition from which Di-E > Ab-E at every
    later transition.

Outputs (analysis_outputs/paper/PEREPOCH_stability/)
  chains_long.csv       one row per direct-stable feature (seed, interval, layer, ...)
  validation_long.csv   recomputed vs stored consecutive counts, per seed x transition x layer
  summary.json          (a) per seed x interval x layer, per seed x interval, pooled over seeds; (b); (c);
                        validation result; runtime

Usage: OMP_NUM_THREADS=2 python scripts/paper/perepoch_stability.py
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

REPO = Path(__file__).resolve().parents[2]
LANES = Path(RDT_DATA + "/extended/experiments/sham_perepoch/lanes")
OUT = REPO / "analysis_outputs" / "paper" / "PEREPOCH_stability"
SD = REPO / "analysis_outputs" / "paper" / "figures" / "final" / "source_data"
SEEDS = (7, 42, 137)
LAYERS = ("avgpool", "layer1", "layer2", "layer3", "layer4")
EPOCHS = range(16)
INTERVALS = ((0, 5), (5, 10), (10, 15))
STEP = 5
VALIDATE_JSON = ((4, 5), (9, 10))
OUTCOMES = ("intact", "death", "transformed", "different_partner")
TIE = 0.05


def prep(H):
    """Pipeline arithmetic: alive mask and centred, normalised float32 columns."""
    n = H.shape[0]
    alive = (np.abs(H) > 1e-6).sum(axis=0) >= max(1, n * 0.01)
    A = H.copy()
    A -= A.mean(axis=0, keepdims=True)
    A /= np.sqrt((A ** 2).sum(axis=0, keepdims=True) + 1e-10)
    return A, alive


def match(pa, pb, st, dt):
    """_phase5_feature_matching for one layer. Returns per-A status (0 not alive, 1 stable, 2 transformed, 3 died),
    per-A partner (-1 none), per-A r, counts, and per-B claim map."""
    (A_a, alive_a), (A_b, alive_b) = pa, pb
    C = A_a.T @ A_b
    na, nb = C.shape
    status = np.zeros(na, int)
    partner = np.full(na, -1)
    corr = np.full(na, np.nan)
    ia, ib = np.where(alive_a)[0], np.where(alive_b)[0]
    status[ia] = 3
    claimed = np.zeros(nb, bool)
    if len(ia) and len(ib):
        Cs = C[np.ix_(ia, ib)]
        r_, c_ = linear_sum_assignment(-Cs)
        for r, c in zip(r_, c_):
            i, j, v = int(ia[r]), int(ib[c]), float(Cs[r, c])
            corr[i] = v
            if v >= st:
                status[i], partner[i] = 1, j
                claimed[j] = True
            elif v >= dt:
                status[i], partner[i] = 2, j
                claimed[j] = True
            else:
                status[i] = 3
    n = {"n_stable": int((status == 1).sum()), "n_transformed": int((status == 2).sum()),
         "n_died": int((status == 3).sum()), "n_born": int((alive_b & ~claimed).sum())}
    return {"status": status, "partner": partner, "corr": corr, "counts": n}


def back_lineage(cons, e, j):
    """Follow epoch-(e+5) feature j backwards over consecutive stable/transformed links. Returns (birth epoch or None
    if traced back to epoch e, origin feature at epoch e or -1)."""
    k = j
    for s in range(STEP - 1, -1, -1):
        m = cons[e + s]
        src = np.flatnonzero((m["partner"] == k) & ((m["status"] == 1) | (m["status"] == 2)))
        if len(src) == 0:
            return e + s + 1, -1
        k = int(src[0])
    return None, k


def stored_pairs(sd, a, b, layer):
    pairs = set()
    for rec in json.load(open(sd / "transitions" / f"{a}_to_{b}" / f"{layer}.json")):
        for x in rec.get("stable", []):
            pairs.add((x["fid_a"], x["fid_b"]))
    return pairs


def run_seed(seed):
    t0 = time.time()
    sd = LANES / f"sham_e00_seed{seed}" / "sae_analysis"
    res = json.load(open(sd / "sae_results.json"))
    th = res["metadata"]["analysis_thresholds"]
    st, dt = th["match_stable_thresh"], th["match_death_thresh"]
    assert res["metadata"]["layers"] == list(LAYERS)
    stored = res["feature_matching"]
    chains, val, sec, pair_checks = [], [], [], []
    for layer in LAYERS:
        P = {e: prep(np.load(sd / "activations" / str(e) / f"{layer}.npy")) for e in EPOCHS}
        cons = {e: match(P[e], P[e + 1], st, dt) for e in range(15)}
        for e in range(15):
            ref = stored[f"{e}->{e + 1}"][layer]
            mine = cons[e]["counts"]
            val.append({"seed": seed, "transition": f"{e}->{e + 1}", "layer": layer,
                        **{f"{k}_stored": ref[k] for k in mine}, **{f"{k}_recomputed": mine[k] for k in mine},
                        "exact": all(ref[k] == mine[k] for k in mine)})
        for a, b in VALIDATE_JSON:
            sp = stored_pairs(sd, a, b, layer)
            m = cons[a]
            mine_pairs = {(int(i), int(m["partner"][i])) for i in np.flatnonzero(m["status"] == 1)}
            pair_checks.append({"seed": seed, "transition": f"{a}->{b}", "layer": layer,
                                "stored_pairs_seen": len(sp), "in_recomputed": len(sp & mine_pairs),
                                "recomputed_stable": len(mine_pairs)})
        for e, f in INTERVALS:
            d = match(P[e], P[f], st, dt)
            for i in np.flatnonzero(d["status"] == 1):
                j = int(d["partner"][i])
                cur, outcome, brk, rs = int(i), None, None, []
                for s in range(STEP):
                    m = cons[e + s]
                    sti = m["status"][cur]
                    rs.append(float(m["corr"][cur]))
                    if sti == 1:
                        cur = int(m["partner"][cur])
                    else:
                        outcome, brk = ("transformed" if sti == 2 else "death"), e + s
                        break
                end_stable = cur if outcome is None else -1
                if outcome is None:
                    outcome = "intact" if cur == j else "different_partner"
                # any-link chain (stable or transformed), stops at death
                cur2, any_out = int(i), None
                for s in range(STEP):
                    m = cons[e + s]
                    if m["status"][cur2] in (1, 2):
                        cur2 = int(m["partner"][cur2])
                    else:
                        any_out = f"death_{e + s}->{e + s + 1}"
                        break
                if any_out is None:
                    any_out = "ends_at_partner" if cur2 == j else "ends_elsewhere"
                birth, origin = back_lineage(cons, e, j) if outcome != "intact" else (None, int(i))
                chains.append({"seed": seed, "interval": f"{e}->{f}", "layer": layer, "fid_e": int(i), "fid_e5": j,
                               "r_direct": round(float(d["corr"][i]), 4), "outcome": outcome,
                               "break_transition": f"{brk}->{brk + 1}" if brk is not None else "",
                               "r_steps_until_break": ";".join("nan" if np.isnan(x) else f"{x:.4f}" for x in rs),
                               "stable_chain_end": end_stable, "anylink_chain": any_out,
                               "partner_backward_birth_epoch": birth if birth is not None else "",
                               "partner_backward_origin_at_e": origin})
            # secondary: all-stable consecutive chains from epoch e and their direct-match class
            alive_e = np.flatnonzero(P[e][1])
            cls = {1: "stable", 2: "transformed", 3: "died"}
            tally = {"n_all_stable_chains": 0, "direct_stable_same_partner": 0, "direct_stable_other_partner": 0,
                     "direct_transformed": 0, "direct_died": 0}
            for i in alive_e:
                cur, ok = int(i), True
                for s in range(STEP):
                    m = cons[e + s]
                    if m["status"][cur] != 1:
                        ok = False
                        break
                    cur = int(m["partner"][cur])
                if not ok:
                    continue
                tally["n_all_stable_chains"] += 1
                c = cls[int(d["status"][i])]
                if c == "stable":
                    tally["direct_stable_same_partner" if d["partner"][i] == cur else "direct_stable_other_partner"] += 1
                else:
                    tally[f"direct_{c}"] += 1
            sec.append({"seed": seed, "interval": f"{e}->{f}", "layer": layer, "n_alive_e": int(len(alive_e)),
                        "n_direct_stable": d["counts"]["n_stable"], **tally})
        del P
        print(f"seed {seed} {layer} done {time.time() - t0:.0f}s", flush=True)
    ctx = []
    for c in res["within_checkpoint_control"]["per_checkpoint"]:
        for x in c["per_layer"]:
            ctx.append({"seed": seed, "checkpoint": int(c["checkpoint"]), "layer": x["layer"],
                        "control_stable_rate": x["stable_rate"]})
    for k, v in stored.items():
        a = int(k.split("->")[0])
        for l, x in v.items():
            na = x["n_stable"] + x["n_transformed"] + x["n_died"]
            ctx.append({"seed": seed, "transition_from": a, "layer": l,
                        "consecutive_stable_rate": x["n_stable"] / na if na else np.nan})
    return {"chains": chains, "val": val, "sec": sec, "pairs": pair_checks, "ctx": ctx,
            "sec_runtime": time.time() - t0}


def tally(df):
    n = len(df)
    c = {k: int((df["outcome"] == k).sum()) for k in OUTCOMES}
    out = {"n_direct_stable": n, **{f"n_{k}": c[k] for k in OUTCOMES},
           **{f"frac_{k}": (c[k] / n if n else None) for k in OUTCOMES}}
    out["n_anylink_ends_at_partner"] = int((df["anylink_chain"] == "ends_at_partner").sum())
    br = df[df["outcome"] != "intact"]
    out["broken_anylink_ends_at_partner"] = int((br["anylink_chain"] == "ends_at_partner").sum())
    out["broken_partner_born_inside_interval"] = int((br["partner_backward_birth_epoch"] != "").sum())
    return out


def part_a(chains, sec):
    ch = chains.copy()
    ch["partner_backward_birth_epoch"] = ch["partner_backward_birth_epoch"].astype(str).replace("nan", "")
    A = {"per_seed_interval_layer": [], "per_seed_interval": [], "pooled_over_seeds_per_interval": [],
         "pooled_over_seeds_per_interval_layer": [], "pooled_all": None, "secondary_all_stable_chains": None}
    for (s, iv, l), g in ch.groupby(["seed", "interval", "layer"], sort=False):
        A["per_seed_interval_layer"].append({"seed": int(s), "interval": iv, "layer": l, **tally(g)})
    for (s, iv), g in ch.groupby(["seed", "interval"], sort=False):
        A["per_seed_interval"].append({"seed": int(s), "interval": iv, **tally(g)})
    psi = pd.DataFrame(A["per_seed_interval"])
    for iv, g in ch.groupby("interval", sort=False):
        r = psi[psi["interval"] == iv]
        A["pooled_over_seeds_per_interval"].append({"interval": iv, **tally(g), "frac_intact_range_over_seeds":
                                                    [round(r["frac_intact"].min(), 4), round(r["frac_intact"].max(), 4)]})
    for (iv, l), g in ch.groupby(["interval", "layer"], sort=False):
        r = pd.DataFrame([x for x in A["per_seed_interval_layer"] if x["interval"] == iv and x["layer"] == l])
        A["pooled_over_seeds_per_interval_layer"].append({"interval": iv, "layer": l, **tally(g),
                                                          "frac_intact_range_over_seeds":
                                                          [round(r["frac_intact"].min(), 4), round(r["frac_intact"].max(), 4)]})
    A["pooled_all"] = {**tally(ch), "frac_intact_range_over_seed_x_interval":
                       [round(psi["frac_intact"].min(), 4), round(psi["frac_intact"].max(), 4)]}
    A["pooled_per_layer"] = [{"layer": l, **tally(g)} for l, g in ch.groupby("layer", sort=False)]
    A["break_transition_counts"] = {k: int(v) for k, v in ch.loc[ch["outcome"] != "intact", "break_transition"]
                                    .value_counts().sort_index().items()}
    # survival of the all-stable chain over steps 1..5 (direct-stable features) and r at the transformed break
    nstep = ch["r_steps_until_break"].str.count(";") + 1
    A["chain_survival_after_k_steps"] = {
        iv: [round(float(((g["outcome"].isin(["intact", "different_partner"])) |
                          (nstep[g.index] > k)).mean()), 4) for k in range(1, STEP + 1)]
        for iv, g in ch.groupby("interval", sort=False)}
    A["chain_survival_after_k_steps"]["all"] = [round(float(((ch["outcome"].isin(["intact", "different_partner"])) |
                                                             (nstep > k)).mean()), 4) for k in range(1, STEP + 1)]
    rb = ch.loc[ch["outcome"] == "transformed", "r_steps_until_break"].str.split(";").str[-1].astype(float)
    A["r_at_transformed_break"] = {"median": round(float(rb.median()), 4), "q25": round(float(rb.quantile(.25)), 4),
                                   "q75": round(float(rb.quantile(.75)), 4), "frac_ge_0.4": round(float((rb >= 0.4).mean()), 4)}
    S = pd.DataFrame(sec)
    cols = ["n_alive_e", "n_direct_stable", "n_all_stable_chains", "direct_stable_same_partner",
            "direct_stable_other_partner", "direct_transformed", "direct_died"]
    A["secondary_all_stable_chains"] = {"definition": "alive epoch-e features stable at all five consecutive steps; "
                                                      "class given by the direct e->e+5 match",
                                        "pooled": {c: int(S[c].sum()) for c in cols},
                                        "per_seed_interval": S.groupby(["seed", "interval"], sort=False)[cols].sum()
                                        .reset_index().to_dict("records")}
    return A


def context(rows):
    """Stored per-step stable rate (n_stable / alive at e) of consecutive matches vs the stored within-checkpoint
    control stable rate (two SAEs trained on the same checkpoint), per layer, epochs >= 5 and all."""
    d = pd.DataFrame(rows)
    out = {}
    for l, g in d.groupby("layer"):
        cs, cc = g.dropna(subset=["consecutive_stable_rate"]), g.dropna(subset=["control_stable_rate"])
        out[l] = {"consecutive_stable_rate_mean": round(float(cs["consecutive_stable_rate"].mean()), 4),
                  "consecutive_stable_rate_mean_from_epoch5": round(float(cs.loc[cs["transition_from"] >= 5,
                                                                              "consecutive_stable_rate"].mean()), 4),
                  "within_checkpoint_control_stable_rate_mean": round(float(cc["control_stable_rate"].mean()), 4),
                  "within_checkpoint_control_stable_rate_mean_from_epoch5":
                      round(float(cc.loc[cc["checkpoint"] >= 5, "control_stable_rate"].mean()), 4)}
    return out


def part_b():
    c = pd.read_csv(SD / "supp1_c.csv")
    r = json.loads((REPO / "analysis_outputs" / "paper" / "SHAM_perepoch" / "results.json").read_text())
    rows = []
    for _, x in c.iterrows():
        s = int(x["seed"])
        X = np.array(r[str(s)]["counts"], float)
        t = np.arange(len(X))
        com = (t[:, None] * X).sum(0) / X.sum(0)
        stored = np.array([x["Tg-E centre of mass (transition)"], x["Ab-E centre of mass (transition)"],
                           x["Di-E centre of mass (transition)"]])
        assert np.abs(com.round(4) - stored).max() < 1e-9, (s, com, stored)
        tg, ab, di = com
        rows.append({"seed": s, "com_tg": round(tg, 2), "com_ab": round(ab, 2), "com_di": round(di, 2),
                     "gap_ab_minus_tg": round(ab - tg, 4), "gap_di_minus_ab": round(di - ab, 4),
                     "strictly_ordered": bool(ab - tg >= TIE and di - ab >= TIE),
                     "ordered_ignoring_ties": bool(tg < ab < di)})
    return {"source": "supp1_c.csv (centre of mass in transition index t = epoch t -> t+1), recomputed from "
                      "SHAM_perepoch/results.json counts as in supp1.py (match to 4 dp)",
            "tie_rule": f"gap < {TIE} transitions = not ordered", "per_run": rows,
            "n_strictly_ordered": sum(x["strictly_ordered"] for x in rows), "n_runs": len(rows)}


def onsets(ab, di):
    gt = di > ab
    first_any = int(np.flatnonzero(gt)[0]) if gt.any() else None
    persistent = None
    for t in range(len(gt)):
        if gt[t:].all():
            persistent = t
            break
    f = lambda t: None if t is None else {"transition_index": t, "transition": f"{t}->{t + 1}", "epoch_from": t}
    return {"first_di_gt_ab": f(first_any), "di_gt_ab_at_every_later_transition_from": f(persistent),
            "transitions_di_le_ab": np.flatnonzero(~gt).tolist()}


def part_c():
    b = pd.read_csv(SD / "supp1_b.csv")
    b = b[b["seed"].isin([str(s) for s in SEEDS])].copy()
    b["seed"] = b["seed"].astype(int)
    out = {"source": "supp1_b.csv columns 'Ab-E events', 'Di-E events' (by 'seed', 'transition_index'; transition t = "
                     "epoch t -> t+1; counts summed over the five SAE layers)", "per_run": {}}
    M = []
    for s in SEEDS:
        g = b[b["seed"] == s].sort_values("transition_index")
        ab, di = g["Ab-E events"].to_numpy(float), g["Di-E events"].to_numpy(float)
        M.append((ab, di))
        out["per_run"][str(s)] = onsets(ab, di)
    ab_m, di_m = np.mean([m[0] for m in M], 0), np.mean([m[1] for m in M], 0)
    out["mean_of_runs_counts"] = onsets(ab_m, di_m)
    out["mean_of_runs_counts"]["t0_ab_di_mean"] = [round(ab_m[0], 1), round(di_m[0], 1)]
    return out


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=3) as ex:
        R = dict(zip(SEEDS, ex.map(run_seed, SEEDS)))
    chains = pd.DataFrame([x for s in SEEDS for x in R[s]["chains"]])
    val = pd.DataFrame([x for s in SEEDS for x in R[s]["val"]])
    pairs = pd.DataFrame([x for s in SEEDS for x in R[s]["pairs"]])
    sec = [x for s in SEEDS for x in R[s]["sec"]]
    chains.to_csv(OUT / "chains_long.csv", index=False)
    val.to_csv(OUT / "validation_long.csv", index=False)
    summary = {
        "validation": {"n_seed_transition_layer": int(len(val)), "n_exact": int(val["exact"].sum()),
                       "non_exact": val[~val["exact"]].to_dict("records"),
                       "stable_pairs_vs_stored_transition_files": pairs.to_dict("records"),
                       "stable_pairs_all_contained": bool((pairs["stored_pairs_seen"] == pairs["in_recomputed"]).all())},
        "a": part_a(chains, sec), "b": part_b(), "c": part_c(),
        "context_matching_noise": context([x for s in SEEDS for x in R[s]["ctx"]]),
        "runtime_s": round(time.time() - t0, 1),
    }
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({"validation_exact": f"{summary['validation']['n_exact']}/{summary['validation']['n_seed_transition_layer']}",
                      "pairs_ok": summary["validation"]["stable_pairs_all_contained"],
                      "pooled": summary["a"]["pooled_all"], "runtime_s": summary["runtime_s"]}, indent=1))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    main()
