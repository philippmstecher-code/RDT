#!/usr/bin/env python3
"""Aggregate transformed_lane_results.pkl into analysis_outputs/paper/TRANSFORMED/transformed_results.json.

See transformed_analysis.py for the conventions (real vs terminal-noise transitions, ResNet avgpool dedup,
three treatments of transformed pairs).

Usage: transformed_report.py
"""
from __future__ import annotations

import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis_outputs" / "paper" / "TRANSFORMED"
ARCHS = ("ResNet-18", "ViT-Small", "CCT-7")
TREATMENTS = ("excluded", "turnover", "continuity")


def r4(x, nd=4):
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, nd)


def pct(x, nd=1):
    return "–" if x is None else f"{100 * x:.{nd}f}%"


def fx(x, nd=2):
    if x is None:
        return "–"
    return "∞" if not np.isfinite(x) else f"{x:.{nd}f}"


def cmh(tables):
    """Mantel–Haenszel pooled OR and CMH chi2 for a list of 2x2 [[a,b],[c,d]]."""
    num = den = 0.0; s_a = s_e = s_v = 0.0
    for (a, b), (c, d) in tables:
        n = a + b + c + d
        if n < 2:
            continue
        num += a * d / n; den += b * c / n
        r1, c1 = a + b, a + c
        s_a += a; s_e += r1 * c1 / n
        s_v += r1 * (n - r1) * c1 * (n - c1) / (n * n * (n - 1))
    orr = num / den if den else float("inf")
    chi2 = (abs(s_a - s_e) - 0.5) ** 2 / s_v if s_v else 0.0
    return orr, chi2, float(stats.chi2.sf(chi2, 1))


def main():
    R = pickle.load(open(OUT / "transformed_lane_results.pkl", "rb"))
    ARCHS = tuple(a for a in ("ResNet-18", "ViT-Small", "CCT-7") if any(r["arch"] == a for r in R))
    res = {"n_lanes": len(R), "lanes": [(r["arch"], r["seed"], r["lane_id"]) for r in R]}
    L = []

    # validation
    val = {f"{r['arch']} s{r['seed']}": {"matching_mismatches": r["matching_mismatches"],
                                          "process_intensity_all_match": all(v["match"] for v in r["process_intensity_validation"])}
           for r in R}
    res["validation"] = val

    # 1. prevalence
    prev = {}
    for arch in ARCHS:
        rs = [r for r in R if r["arch"] == arch]
        S = T = D = B = A = 0
        per_t = defaultdict(list); lane_layer = []
        for r in rs:
            byt = defaultdict(lambda: [0, 0])
            for p in r["prevalence"]:
                if p["layer"] not in r["pooled_layers"]:
                    continue
                S += p["n_stable"]; T += p["n_transformed"]; D += p["n_died"]; B += p["n_born"]; A += p["n_alive_a"]
                byt[p["t_index"]][0] += p["n_transformed"]; byt[p["t_index"]][1] += p["n_alive_a"]
            for ti, (t, a) in byt.items():
                per_t[ti].append(t / a if a else np.nan)
            for layer in r["pooled_layers"]:
                ps = [p for p in r["prevalence"] if p["layer"] == layer]
                cross = sum(p["n_transformed"] for p in ps) / max(1, sum(p["n_alive_a"] for p in ps))
                fl = r["floors"][layer]
                lane_layer.append({"seed": r["seed"], "layer": layer, "cross": cross,
                                   "control": fl["control_transformed_rate"],
                                   "terminal_same_init": fl["terminal_same_init"]["transformed_rate"],
                                   "control_false_death": fl["control_false_death_rate"],
                                   "terminal_death": fl["terminal_same_init"]["death_rate"]})
        ctrl = [x["control"] for x in lane_layer if x["control"] is not None]
        term = [x["terminal_same_init"] for x in lane_layer if x["terminal_same_init"] is not None]
        cross = [x["cross"] for x in lane_layer]
        prev[arch] = {
            "pooled_real": {"stable": S / A, "transformed": T / A, "died": D / A, "n_alive_a": A,
                            "transformed_to_died": T / D if D else None, "transformed_to_born": T / B if B else None},
            "per_transition_index_mean": {int(k): r4(np.nanmean(v)) for k, v in sorted(per_t.items())},
            "lane_layer_mean_cross": r4(np.mean(cross)),
            "lane_layer_mean_control_floor": r4(np.mean(ctrl)),
            "lane_layer_mean_terminal_floor": r4(np.mean(term)),
            "n_lane_layers": len(lane_layer),
            "n_cross_above_control": int(sum(1 for x in lane_layer if x["control"] is not None and x["cross"] > x["control"])),
            "n_cross_above_terminal": int(sum(1 for x in lane_layer if x["terminal_same_init"] is not None and x["cross"] > x["terminal_same_init"])),
            "lane_layer": lane_layer,
        }
    res["prevalence"] = prev

    # per layer (arch, layer) mean over seeds
    per_layer = defaultdict(list)
    for arch in ARCHS:
        for x in prev[arch]["lane_layer"]:
            per_layer[(arch, x["layer"])].append(x)
    res["prevalence_per_layer"] = [{"arch": a, "layer": l,
                                    "cross": r4(np.mean([x["cross"] for x in v])),
                                    "control": r4(np.mean([x["control"] for x in v if x["control"] is not None])),
                                    "terminal_same_init": r4(np.mean([x["terminal_same_init"] for x in v if x["terminal_same_init"] is not None]))}
                                   for (a, l), v in per_layer.items()]

    # 2. fates
    fates = {}
    for arch in ARCHS + ("all",):
        rs = [r for r in R if arch == "all" or r["arch"] == arch]
        tot = defaultdict(Counter); strata = defaultdict(lambda: defaultdict(Counter))
        for r in rs:
            for layer, k, o0, o1, n in r["fate_counts"]:
                if layer not in r["pooled_layers"]:
                    continue
                tot[o0][o1] += n
                strata[(r["lane_id"], layer, k)][o0][o1] += n
        pdied = {o: (tot[o]["died"] / sum(tot[o].values()) if sum(tot[o].values()) else None) for o in ("stable", "transformed", "born", "initial")}
        ptr = {o: (tot[o]["transformed"] / sum(tot[o].values()) if sum(tot[o].values()) else None) for o in ("stable", "transformed", "born", "initial")}

        def tables(x, y):
            out = []
            for s in strata.values():
                a = s[x]["died"]; b = sum(s[x].values()) - a
                c = s[y]["died"]; d = sum(s[y].values()) - c
                out.append(((a, b), (c, d)))
            return out
        or_ts = cmh(tables("transformed", "stable")); or_tb = cmh(tables("transformed", "born"))
        # triples
        trip = Counter(); strata_k1 = defaultdict(Counter); exp_std = 0.0; obs_std = 0
        death_rate_k = {}
        for r in rs:
            for layer, k, o0, o1, n in r["fate_counts"]:
                if layer in r["pooled_layers"]:
                    death_rate_k.setdefault((r["lane_id"], layer, k), Counter())
                    death_rate_k[(r["lane_id"], layer, k)]["n"] += n
                    if o1 == "died":
                        death_rate_k[(r["lane_id"], layer, k)]["d"] += n
        for r in rs:
            for layer, k, o0, o1, o2, n in r["triple_counts"]:
                if layer not in r["pooled_layers"]:
                    continue
                trip[(o0, o1, o2)] += n
                if o0 == "stable" and o1 == "transformed":
                    dk = death_rate_k.get((r["lane_id"], layer, k + 1))
                    if dk and dk["n"]:
                        exp_std += n * dk["d"] / dk["n"]
                    if o2 == "died":
                        obs_std += n

        def pd(o0, o1):
            n = sum(v for (a, b, c), v in trip.items() if a == o0 and b == o1)
            d = trip[(o0, o1, "died")]
            return (d / n if n else None), n
        fates[arch] = {
            "counts_in_by_out": {o: dict(tot[o]) for o in tot},
            "p_died_given_in": {k: r4(v) for k, v in pdied.items()},
            "p_transformed_given_in": {k: r4(v) for k, v in ptr.items()},
            "cmh_transformed_vs_stable_died": {"OR": r4(or_ts[0]), "chi2": r4(or_ts[1], 2), "p": or_ts[2]},
            "cmh_transformed_vs_born_died": {"OR": r4(or_tb[0]), "chi2": r4(or_tb[1], 2), "p": or_tb[2]},
            "seq_S_T_D": {"observed": obs_std, "expected_stratified_marginal": r4(exp_std, 1),
                          "obs_over_exp": r4(obs_std / exp_std) if exp_std else None,
                          "poisson_p_upper": float(stats.poisson.sf(obs_std - 1, exp_std)) if exp_std else None},
            "p_died_next_given_two_links": {f"{a}->{b}": {"p": r4(pd(a, b)[0]), "n": pd(a, b)[1]}
                                            for a in ("stable", "transformed", "born", "initial") for b in ("stable", "transformed")},
        }
    # false-death references
    for arch in ARCHS:
        ll = prev[arch]["lane_layer"]
        fates[arch]["false_death_control_mean"] = r4(np.mean([x["control_false_death"] for x in ll if x["control_false_death"] is not None]))
        fates[arch]["death_rate_terminal_same_init_mean"] = r4(np.mean([x["terminal_death"] for x in ll if x["terminal_death"] is not None]))
    res["fates"] = fates

    # 3. selectivity
    sel = {}
    for arch in ARCHS + ("all",):
        rs = [r for r in R if arch == "all" or r["arch"] == arch]
        by = defaultdict(lambda: {"ssi": [], "csi": [], "sai": [], "above": Counter(), "n": 0})
        for r in rs:
            th = r["thresholds"]
            for layer, t, cls, ssi, csi, sai in r["selectivity_rows"]:
                if layer not in r["pooled_layers"]:
                    continue
                g = by[cls]; g["ssi"].append(ssi); g["csi"].append(csi); g["sai"].append(sai); g["n"] += 1
                a_sai, a_ssi, a_csi = sai > th["sai"], ssi > th["ssi"], csi > th["csi"]
                g["above"]["sai"] += a_sai; g["above"]["ssi"] += a_ssi; g["above"]["csi"] += a_csi
                g["above"]["none"] += not (a_sai or a_ssi or a_csi)
        s = {}
        for cls, g in by.items():
            s[cls] = {"n": g["n"], **{f"{k}_mean": r4(np.mean(g[k])) for k in ("ssi", "csi", "sai")},
                      **{f"{k}_median": r4(np.median(g[k])) for k in ("ssi", "csi", "sai")},
                      **{f"frac_above_{k}": r4(g["above"][k] / g["n"]) for k in ("sai", "ssi", "csi", "none")}}
        tests = {}
        for a_, b_ in (("A_transformed", "A_stable"), ("A_transformed", "A_died"), ("B_transformed", "B_born"), ("B_transformed", "B_stable")):
            if a_ in by and b_ in by:
                tests[f"{a_}_vs_{b_}"] = {}
                for k in ("ssi", "csi", "sai"):
                    u = stats.mannwhitneyu(by[a_][k], by[b_][k], alternative="two-sided")
                    rb = 1 - 2 * u.statistic / (len(by[a_][k]) * len(by[b_][k]))
                    tests[f"{a_}_vs_{b_}"][k] = {"rank_biserial": r4(-rb), "p": float(u.pvalue)}
        sel[arch] = {"by_class": s, "mannwhitney": tests}
    res["selectivity"] = sel

    # 4. processes under treatments
    procs = {}
    for variant in ("paper_all_layers", "dedup"):
        procs[variant] = {}
        for tr in TREATMENTS:
            rows = []
            for r in R:
                p = r["processes"][variant][tr]
                rows.append({"arch": r["arch"], "seed": r["seed"], "cr_first": p["cr_first"],
                             "cr_last_real": p["cr_last_real"], "cr_terminal_noise": p["cr_terminal_noise"],
                             "di_zero_at_first": p["di_zero_at_first"],
                             "peak": p["ordering"]["peak"], "peak_order_holds": p["ordering"]["peak_order_holds"],
                             "com": p["ordering"]["com"], "com_order_holds": p["ordering"]["com_order_holds"],
                             "churn_first": p["per_transition"][0]["churn"],
                             "churn_last_real": p["per_transition"][-1]["churn"],
                             "churn_terminal_noise": p["terminal_noise"]["churn"]})

            def safe(v):
                return np.inf if v is None else v
            dec_real = sum(safe(x["cr_last_real"]) < safe(x["cr_first"]) for x in rows)
            dec_term = sum(safe(x["cr_terminal_noise"]) < safe(x["cr_first"]) for x in rows)
            cross_real = sum(safe(x["cr_last_real"]) < 1 for x in rows)
            cross_term = sum(safe(x["cr_terminal_noise"]) < 1 for x in rows)
            pool = lambda key: sum(sum(r["processes"][variant][tr][key][p] for p in ("tg_h", "ab_h")) for r in R) / \
                max(1, sum(r["processes"][variant][tr][key]["di_h"] for r in R))
            n = len(rows)
            procs[variant][tr] = {
                "per_lane": rows,
                "n_decline_first_to_last_real": int(dec_real),
                "sign_test_p_decline_real": float(stats.binomtest(int(dec_real), n, 0.5, alternative="greater").pvalue),
                "n_decline_first_to_terminal": int(dec_term),
                "n_crossover_last_real_below_1": int(cross_real),
                "n_crossover_terminal_below_1": int(cross_term),
                "n_di_zero_first": int(sum(x["di_zero_at_first"] for x in rows)),
                "n_peak_order_holds": int(sum(x["peak_order_holds"] for x in rows)),
                "n_com_order_holds": int(sum(bool(x["com_order_holds"]) for x in rows)),
                "pooled_cr_first": r4(pool("first")), "pooled_cr_last_real": r4(pool("last_real")),
                "pooled_cr_terminal_noise": r4(pool("terminal")),
                "mean_cr_terminal_noise": r4(np.mean([x["cr_terminal_noise"] for x in rows if x["cr_terminal_noise"] is not None])),
            }
    res["processes"] = procs

    # 5. churn at the first transition (ResNet-18)
    rn = [r for r in R if r["arch"] == "ResNet-18"]
    churn815 = {}
    for variant in ("paper_all_layers", "dedup"):
        churn815[variant] = {tr: {"seed42": next(r["processes"][variant][tr]["per_transition"][0]["churn"] for r in rn if r["seed"] == 42),
                                  "resnet_mean": r4(np.mean([r["processes"][variant][tr]["per_transition"][0]["churn"] for r in rn])),
                                  "all9_mean": r4(np.mean([r["processes"][variant][tr]["per_transition"][0]["churn"] for r in R]))}
                             for tr in TREATMENTS}
        churn815[variant]["feature_level_seed42"] = next(r["feature_level_churn_first_transition"][variant] for r in rn if r["seed"] == 42)
        churn815[variant]["feature_level_resnet_mean"] = {tr: r4(np.mean([r["feature_level_churn_first_transition"][variant][tr] for r in rn])) for tr in TREATMENTS}
    res["churn_first_transition_resnet"] = churn815

    # 6. persistence
    pers = {}
    for tr in TREATMENTS:
        rows = []
        for r in R:
            p = r["persistence"][tr]
            rows.append({"arch": r["arch"], "seed": r["seed"], "constr_over_csi": p["construction_over_csi_lifespan"],
                         "sai_over_csi": p["sai_over_csi_lifespan"], "ssi_over_csi": p["ssi_over_csi_lifespan"],
                         "rho_ssi": p["spearman_ssi_lifespan"], "rho_csi": p["spearman_csi_lifespan"],
                         "mean_lifespan": p["mean_lifespan"], "n_by_cohort": p["n_by_cohort"],
                         "term_pers_sai_over_csi": p["terminal_persistence_sai_over_csi"],
                         "term_pers_ssi_over_csi": p["terminal_persistence_ssi_over_csi"],
                         "term_pers_n": p["terminal_persistence_n"]})
        v = [x["constr_over_csi"] for x in rows if x["constr_over_csi"] is not None]
        tp = [x["term_pers_sai_over_csi"] for x in rows if x["term_pers_sai_over_csi"] is not None]
        pers[tr] = {"per_lane": rows,
                    "n_constr_over_csi_gt1": int(sum(x > 1 for x in v)), "constr_over_csi_range": [r4(min(v)), r4(max(v))] if v else None,
                    "n_rho_ssi_pos": int(sum(x["rho_ssi"][0] is not None and x["rho_ssi"][0] > 0 for x in rows)),
                    "n_rho_ssi_pos_sig": int(sum(x["rho_ssi"][0] is not None and x["rho_ssi"][0] > 0 and x["rho_ssi"][1] < 1e-6 for x in rows)),
                    "n_rho_csi_neg": int(sum(x["rho_csi"][0] is not None and x["rho_csi"][0] < 0 for x in rows)),
                    "n_rho_csi_neg_sig": int(sum(x["rho_csi"][0] is not None and x["rho_csi"][0] < 0 and x["rho_csi"][1] < 1e-6 for x in rows)),
                    "n_term_pers_sai_over_csi_gt1": int(sum(x > 1 for x in tp)), "term_pers_sai_over_csi_range": [r4(min(tp)), r4(max(tp))] if tp else None}
    res["persistence"] = pers

    # 7. terminal noise reference vs last real transition
    noise = {}
    for arch in ARCHS:
        rs = [r for r in R if r["arch"] == arch]
        last_T = last_D = last_A = 0
        for r in rs:
            lt = r["real_transitions"][-1]
            for p in r["prevalence"]:
                if p["transition"] == lt and p["layer"] in r["pooled_layers"]:
                    last_T += p["n_transformed"]; last_D += p["n_died"]; last_A += p["n_alive_a"]
        ti = [r["floors"][l]["terminal_same_init"] for r in rs for l in r["pooled_layers"]]
        noise[arch] = {"last_real_transformed_rate": r4(last_T / last_A), "last_real_death_rate": r4(last_D / last_A),
                       "terminal_transformed_rate_mean": r4(np.mean([x["transformed_rate"] for x in ti])),
                       "terminal_death_rate_mean": r4(np.mean([x["death_rate"] for x in ti])),
                       "terminal_stable_rate_mean": r4(np.mean([x["stable_rate"] for x in ti])),
                       "churn_last_real_mean": r4(np.mean([r["processes"]["dedup"]["excluded"]["per_transition"][-1]["churn"] for r in rs])),
                       "churn_terminal_mean": r4(np.mean([r["processes"]["dedup"]["excluded"]["terminal_noise"]["churn"] for r in rs]))}
    res["terminal_noise_reference"] = noise

    (OUT / "transformed_results.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"wrote {OUT / 'transformed_results.json'}")


if __name__ == "__main__":
    main()
