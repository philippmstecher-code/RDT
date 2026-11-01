#!/usr/bin/env python3
"""Fig. 4a and the 50 vs 200 epoch comparison (ResNet-18 seed 42) with the stored task-general threshold (stored t_sai, as Fig. 4c and the pipeline)
instead of the fixed SAI > 0.9; everything else exactly as scripts/paper/figures/fig4_lifespan.py (same records,
cohort_stats 'methods' rule, birth x layer matched ratio, birth-stratified log-rank; ep200 record bootstrap).
Read-only; writes analysis_outputs/paper/THRESHOLD_impact/fig4_stored_sai.json (fig4_lifespan outputs untouched).
Usage: fig4_stored_sai.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "figures"))
sys.path.insert(0, str(HERE))
import fig4_lifespan as F  # noqa: E402
from persist_analyze import C, cohort_masks, cohort_stats, strat_logrank  # noqa: E402
from persist_lib import CORE_LANES, load_lane, records_real  # noqa: E402

OUT = HERE.parents[1] / "analysis_outputs" / "paper" / "THRESHOLD_impact" / "fig4_stored_sai.json"


def ep200_vec(R, thr):
    life, cens = R[:, C["life"]], R[:, C["cens"]]
    ssi, csi, sai = R[:, C["ssi"]], R[:, C["csi"]], R[:, C["sai"]]
    A, S, Cs = sai > thr["sai"], ssi > thr["ssi"], csi > thr["csi"]
    ratio = lambda x, a, b: x[a].mean() / x[b].mean() if a.any() and b.any() and x[b].mean() > 0 else np.nan  # noqa: E731
    return np.array([100 * cens.mean(), ratio(cens, A, ~A), ratio(life, S & ~Cs & ~A, Cs & ~A)])


def main():
    res = {"fig4a": [], "fig4d": {}}
    for lane, arch, seed in CORE_LANES:
        d, rr = load_lane(lane)
        at = rr["metadata"]["analysis_thresholds"]
        thr = {"sai": at["sai_task_general_thresh"], "ssi": at["ssi_abstraction_thresh"], "csi": at["csi_differentiation_thresh"]}
        R = records_real(d, rr, "paper", dedup=True)
        out = {"arch": arch, "seed": seed, "thr_sai": thr["sai"]}
        for nm, t in (("sai_0.9", {**thr, "sai": 0.9}), ("sai_stored", thr)):
            st = cohort_stats(R, t, "methods")
            mm = cohort_masks(R, t, "methods")
            _, p, hr = strat_logrank(R[:, C["life"]], R[:, C["cens"]], mm["coarse"], mm["fine"], R[:, C["birth"]])
            out[nm] = {"ratio": float(st["ratio"]), "n_coarse": int(st["n_coarse"]), "n_fine": int(st["n_fine"]),
                       "ratio_birth_layer_matched": float(st["ratio_birth_layer_matched"]),
                       "hr_birth": float(hr), "hr_birth_p": float(p)}
        res["fig4a"].append(out)
        print(arch, seed, {k: (round(v["ratio"], 3), round(v["ratio_birth_layer_matched"], 3), v["n_coarse"]) for k, v in out.items()
                           if isinstance(v, dict)}, flush=True)
    from persist_200ep import RUNS, load, stored_thr
    boots = {}
    for run, (exp, lane) in RUNS.items():
        d, rr = load(exp, lane)
        thr = stored_thr(rr)
        at = rr["metadata"]["analysis_thresholds"]
        thr = {**thr, "sai": at["sai_task_general_thresh"]}
        R = records_real(d, rr, "paper")
        est = ep200_vec(R, thr)
        rng = np.random.default_rng(0)
        B = np.array([ep200_vec(R[rng.integers(0, len(R), len(R))], thr) for _ in range(F.EP_BOOT)])
        boots[run] = B
        res["fig4d"][run] = {"thr": thr, "n_tg": int((R[:, C["sai"]] > thr["sai"]).sum()),
                             **{nm: float(est[i]) for i, nm in enumerate(("survival_pct", "tg_surv_adv", "ssi_csi_life"))},
                             **{nm + "_ci95": [float(x) for x in np.nanpercentile(B[:, i], [2.5, 97.5])]
                                for i, nm in enumerate(("survival_pct", "tg_surv_adv", "ssi_csi_life"))}}
    for i, nm in enumerate(("survival_pct", "tg_surv_adv", "ssi_csi_life")):
        diff = boots["200ep"][:, i] - boots["50ep"][:, i]
        res["fig4d"][f"diff_{nm}"] = {"ci95": [float(x) for x in np.nanpercentile(diff, [2.5, 97.5])],
                                      "p_two_sided": float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean())))}
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    print(json.dumps(res["fig4d"], indent=1, default=float))


if __name__ == "__main__":
    main()
