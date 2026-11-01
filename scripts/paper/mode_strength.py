"""mode_strength: input-output mode strength predicts when each superclass is learned, tested on held-out networks.

timetable_saxe.main (mode strength s from training pixels whitened with 64 PCs; superclass-probability timing lead with
chance baseline; t50 at 0.5 interpolated in log(1+epoch); partial Spearman given the final level) is called unchanged
with only `lanes`, `CACHE`, `OUT` re-pointed to the 7 held-out networks (logits from information_share.py, argmax agreement
>= 0.999).

Criteria:
  (1) Spearman rho(s, lead) >= 0.6 in every one of the 7 networks;
  (2) pooled partial rho (given the final level) > 0: Fisher-z DerSimonian-Laird random-effects meta-analysis over the
      7 networks, per-network SE = 1/sqrt(n-4) with n = 20 superclasses, one-sided p < 0.05.
Further measures:
  slope of log t50 on log s within [-2, -0.7] in the three core networks (timetable_saxe slope);
  within-superclass timing (pws lead; timetable_curves.behaviour + timetable_timing.unit_curves/measures) is predicted better
  by the class-within-superclass contrast strength (saxe_ws_W64) than by the class-versus-rest strength (saxe_cls_W64),
  partial Spearman given the class final level (pt final), as in timetable_predict.main.
Usage: python scripts/paper/mode_strength.py [saxe|decide|all]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from information_share import DATA_ORDER, CACHE_ORDER, CORE_KEYS, TIMETABLE, seven, verify_logits  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

OUT_MS = DATA_ORDER / "mode_strength"
N_UNITS = 20
METHOD_NOTES = [
    "Mode strengths recomputed by timetable_saxe.main (verbatim); identical to the stored DICTFREE values (max |diff| 0).",
    "Within-superclass secondary evaluated per network (DICTFREE: per architecture-set consensus); predictors and the "
    "control (class final level, pt final) as in timetable_predict.main.",
    "timetable_saxe.main also runs its 30 random 20x5 partitions (rng seed 1); their comparison is reported, not used in the criteria.",
]


def step_saxe():
    import timetable_saxe
    OUT_MS.mkdir(parents=True, exist_ok=True)
    timetable_saxe.CACHE = CACHE_ORDER
    timetable_saxe.OUT = OUT_MS
    timetable_saxe.lanes = seven
    timetable_saxe.main()


def dersimonian_laird(r, n):
    z = np.arctanh(np.asarray(r, float))
    v = np.full(len(z), 1.0 / (n - 4))
    w = 1 / v
    z_fe = np.sum(w * z) / np.sum(w)
    Q = float(np.sum(w * (z - z_fe) ** 2))
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - (len(z) - 1)) / C)
    ws = 1 / (v + tau2)
    z_re = float(np.sum(ws * z) / np.sum(ws))
    se = float(np.sqrt(1 / np.sum(ws)))
    Z = z_re / se
    return dict(k=len(z), z_re=z_re, se=se, Z=Z, p_one_sided=float(norm.sf(Z)), pooled_rho=float(np.tanh(z_re)),
                ci95_rho=[float(np.tanh(z_re - 1.96 * se)), float(np.tanh(z_re + 1.96 * se))], tau2=tau2, Q=Q)


def within_superclass_secondary():
    from timetable_curves import behaviour
    from timetable_predict import partial_spearman
    from timetable_timing import CHANCE, measures, unit_curves
    pc = pd.read_csv(TIMETABLE / "predictors_class.csv").set_index("cls")  # training-pixel statistics only
    rows = []
    for key, lane in seven().items():
        img = {"pt": [], "pws": []}
        for lab in lane["labels"]:
            b = behaviour(np.load(CACHE_ORDER / f"{key}_m{lab}_logits.npy"))
            for q in img:
                img[q].append(b[q].astype(np.float32))
        img = {q: np.stack(v) for q, v in img.items()}
        lead = measures(unit_curves(img, "pws"), CHANCE["pws"])["lead"]
        fin = measures(unit_curves(img, "pt"), CHANCE["pt"])["final"]
        r_ws = partial_spearman(pc["saxe_ws_W64"].values, lead, fin)
        r_cls = partial_spearman(pc["saxe_cls_W64"].values, lead, fin)
        rows.append(dict(lane=key, partial_ws_contrast=r_ws, partial_cls_vs_rest=r_cls, ws_better=bool(r_ws > r_cls)))
    d = pd.DataFrame(rows)
    d.to_csv(OUT_MS / "mode_strength_within_superclass.csv", index=False)
    return d


def decide():
    P = pd.read_csv(OUT_MS / "saxe_partition_test.csv")
    U = pd.read_csv(OUT_MS / "saxe_superclass_units.csv")
    # mode strengths depend only on training pixels and must equal the stored values in TIMETABLE
    s_new = U[U.lane == U.lane.iloc[0]].sort_values("sc").s.values
    Ue = pd.read_csv(TIMETABLE / "saxe_superclass_units.csv")
    s_timetable = Ue[Ue.lane == Ue.lane.iloc[0]].sort_values("sc").s.values
    s_pred = pd.read_csv(TIMETABLE / "predictors_superclass.csv").sort_values("sc").saxe_sc_W64.values
    s_check = dict(max_abs_diff_vs_timetable_saxe=float(np.max(np.abs(s_new - s_timetable))),
                   max_abs_diff_vs_timetable_predictors=float(np.max(np.abs(s_new - s_pred))))
    T = P[P.partition == "true"].set_index("lane").loc[list(seven())]
    R = P[P.partition != "true"]
    T["frac_rand_rho_lead_ge_true"] = [float((R[R.lane == k].rho_lead >= T.loc[k, "rho_lead"]).mean()) for k in T.index]
    T["frac_rand_partial_ge_true"] = [float((R[R.lane == k].partial_lead >= T.loc[k, "partial_lead"]).mean()) for k in T.index]
    T["crit1_rho_ge_0.6"] = T.rho_lead >= 0.6
    T.to_csv(OUT_MS / "mode_strength_per_network.csv")
    meta = dersimonian_laird(T.partial_lead.values, N_UNITS)
    c1 = bool(T["crit1_rho_ge_0.6"].all())
    c2 = bool(meta["z_re"] > 0 and meta["p_one_sided"] < 0.05)
    verdict = "criteria met" if (c1 and c2) else "criteria not met"
    core = T.loc[CORE_KEYS]
    slope_ok = {k: bool(-2 <= v <= -0.7) for k, v in core.slope_logt50_logs.items()}
    W = within_superclass_secondary()
    res = dict(
        test="mode_strength", verdict=verdict,
        rule="(1) Spearman rho(s, lead) >= 0.6 in every one of the 7 networks AND (2) pooled partial rho given the final "
             "level > 0 (Fisher-z DerSimonian-Laird random effects, SE 1/sqrt(20-4), one-sided p < 0.05)",
        open_estimate="rho(s, lead) 0.86/0.86/0.88 (core ResNet/ViT/CCT s42), 0.68-0.85 (six 30-epoch ResNets); "
                             "partial 0.58/0.63/0.70 and 0.07-0.61",
        criterion1=dict(holds=c1, n_networks=int(T["crit1_rho_ge_0.6"].sum()), of=len(T)),
        criterion2=dict(holds=c2, **meta),
        per_network=T[["arch", "rho_lead", "partial_lead", "rho_final", "rho_t50", "slope_logt50_logs", "n_t50",
                       "frac_rand_rho_lead_ge_true", "frac_rand_partial_ge_true"]].round(4).reset_index().to_dict(orient="records"),
        mode_strength_check=s_check,
        secondary=dict(
            slope_core=dict(rule="slope of log t50 on log s within [-2, -0.7] in the three core networks",
                            slopes=core.slope_logt50_logs.round(3).to_dict(), within=slope_ok, holds=all(slope_ok.values())),
            within_superclass=dict(rule="partial rho(pws lead | pt final): class-within-superclass contrast strength > "
                                        "class-versus-rest strength", per_network=W.round(4).to_dict(orient="records"),
                                   holds_in=int(W.ws_better.sum()), of=len(W)),
        ),
        multiplicity="no correction (a single test with its own criteria)",
    )
    json.dump(res, open(OUT_MS / "mode_strength_results.json", "w"), indent=2, default=float)
    with pd.option_context("display.width", 250):
        print(T.round(3).to_string())
        print(W.round(3).to_string())
    print(json.dumps(dict(meta=meta, s_check=s_check, slope_ok=slope_ok), indent=1))
    print("mode_strength:", verdict)
    return res


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    verify_logits()
    if step in ("saxe", "all"):
        step_saxe()
    if step in ("decide", "all"):
        decide()
