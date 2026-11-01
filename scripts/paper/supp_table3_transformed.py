#!/usr/bin/env python3
"""Three treatments of transformed matches on the Fig. 4a feature records (persist_lib.records_real, alive features, duplicate
final transition excluded, ResNet avgpool deduplicated) with stored per-run thresholds incl. SAI (Methods rule):
  turnover   = Fig. 4a rule 'paper' (a transformed match ends the chain)
  continuity = rule 'continuity' (a transformed match continues the chain)
  excluded   = rule 'paper', but chains that end at a transformed match are dropped (as in TRANSFORMED)
Per run: coarse/fine mean-lifespan ratio (persist_analyze.cohort_stats 'methods') and Spearman rho of SSI with lifespan,
and of CSI with lifespan (columns csi_lifespan_rho, csi_lifespan_p; the 'turnover' rows equal Fig. 4b, fig4_b.csv).
Writes analysis_outputs/paper/SI/supp_table3_transformed.csv.
Usage: python scripts/paper/supp_table3_transformed.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from persist_analyze import C, cohort_stats  # noqa: E402
from persist_lib import CORE_LANES, load_lane, records_real  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "analysis_outputs" / "paper" / "SI" / "supp_table3_transformed.csv"


def ends_transformed(d, dedup=True):
    """Boolean per record of records_real(d, res, 'paper'): chain ended at a transformed match."""
    real = d["labels"][:-1]
    trans = [f"{a}->{b}" for a, b in zip(real[:-1], real[1:])]
    n_t = len(trans)
    layers = [l for l in d["layers"] if not (dedup and d["arch"] == "ResNet-18" and l == "avgpool")]
    flags = []
    for layer in layers:
        per_t = d["layer_data"][layer]
        st = [{i: j for i, j, _, _ in per_t[t]["stable"]} for t in trans]
        tr = [{i for i, _, _, _ in per_t[t]["transformed"]} for t in trans]
        for b in range(n_t):
            for fid in np.where(per_t[trans[b]]["alive_a"])[0]:
                cur, k, end_tr = int(fid), b, False
                while k < n_t:
                    if cur in st[k]:
                        cur = st[k][cur]; k += 1
                    else:
                        end_tr = cur in tr[k]
                        break
                flags.append(end_tr)
    return np.array(flags, bool)


def main():
    rows = []
    for lane, arch, seed in CORE_LANES:
        d, res = load_lane(lane)
        at = res["metadata"]["analysis_thresholds"]
        thr = {"sai": at["sai_task_general_thresh"], "ssi": at["ssi_abstraction_thresh"], "csi": at["csi_differentiation_thresh"]}
        Rp = records_real(d, res, "paper", dedup=True)
        Rc = records_real(d, res, "continuity", dedup=True)
        et = ends_transformed(d)
        assert len(et) == len(Rp), (len(et), len(Rp))
        for name, R in (("turnover (Fig. 4a rule)", Rp), ("continuity", Rc), ("excluded", Rp[~et])):
            st = cohort_stats(R, thr, "methods")
            rho, p = stats.spearmanr(R[:, C["ssi"]], R[:, C["life"]])
            rc, pc = stats.spearmanr(R[:, C["csi"]], R[:, C["life"]])
            rows.append({"treatment": name, "arch": arch, "seed": seed, "n_records": len(R),
                         "coarse_over_fine": round(float(st["ratio"]), 3), "ssi_lifespan_rho": round(float(rho), 3),
                         "ssi_lifespan_p": float("%.2g" % p), "csi_lifespan_rho": round(float(rc), 3),
                         "csi_lifespan_p": float("%.2g" % pc)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    for t, g in df.groupby("treatment", sort=False):
        print(f"{t}: ratio {g.coarse_over_fine.min():.2f}-{g.coarse_over_fine.max():.2f} ({(g.coarse_over_fine > 1).sum()}/9 > 1); "
              f"SSI rho {g.ssi_lifespan_rho.min():+.3f}..{g.ssi_lifespan_rho.max():+.3f}, positive {(g.ssi_lifespan_rho > 0).sum()}/9, "
              f"positive & P<0.05 {((g.ssi_lifespan_rho > 0) & (g.ssi_lifespan_p < 0.05)).sum()}/9; "
              f"CSI rho {g.csi_lifespan_rho.min():+.3f}..{g.csi_lifespan_rho.max():+.3f}, negative {(g.csi_lifespan_rho < 0).sum()}/9, "
              f"negative & P<0.05 {((g.csi_lifespan_rho < 0) & (g.csi_lifespan_p < 0.05)).sum()}/9")


if __name__ == "__main__":
    main()
