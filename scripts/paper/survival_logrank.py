#!/usr/bin/env python3
"""Methods 'Feature survival analysis': log-rank comparison of high- vs low-SSI features (median split of SSI) on the
Fig. 4 lifespan records (persist_lib.records_real 'paper', alive features, duplicate final transition excluded,
ResNet avgpool deduplicated), unstratified and stratified by birth checkpoint (persist_analyze.strat_logrank;
hazard ratio < 1: high-SSI features live longer). Primary run = ResNet-18 seed 42; all 9 core runs reported.
Writes analysis_outputs/paper/SI/survival_logrank.json.
Usage: python scripts/paper/survival_logrank.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from persist_analyze import C, strat_logrank  # noqa: E402
from persist_lib import CORE_LANES, load_lane, records_real  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "analysis_outputs" / "paper" / "SI" / "survival_logrank.json"


def main():
    res = {}
    for lane, arch, seed in CORE_LANES:
        d, rr = load_lane(lane)
        R = records_real(d, rr, "paper", dedup=True)
        life, cens, ssi, birth = R[:, C["life"]], R[:, C["cens"]], R[:, C["ssi"]], R[:, C["birth"]]
        med = float(np.median(ssi))
        hi = ssi > med
        out = {"n_records": int(len(R)), "median_ssi": med}
        for nm, strata in (("unstratified", np.zeros(len(R))), ("birth_stratified", birth)):
            chi2, p, hr = strat_logrank(life, cens, hi, ~hi, strata)
            out[nm] = {"chi2": float(chi2), "p": float(p), "hazard_ratio": float(hr)}
        res[f"{arch} s{seed}"] = out
        print(arch, seed, json.dumps(out), flush=True)
    res["primary_run"] = "ResNet-18 s42"
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
