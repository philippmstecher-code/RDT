"""Developmental timetable, step 5: do data statistics predict unit timing? Spearman and partial Spearman (controlling the unit's final level).

Targets (consensus = mean of within-lane z-scores):
  superclass level (n=20): psc lead / early, sc lead / early, nc_sc lead / early, final psc
  class level (n=100):     pt lead / early, fine lead / early, ws lead / early, pws lead / early, final pt
Network sets: ResNet (7 open ResNet lanes), ViT s42, CCT s42.
Predictors: predictors_class.csv / predictors_superclass.csv + init-network nearest-centroid recall (milestone 0).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timetable_lib import OUT, lanes  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.c_[np.ones_like(rz), rz]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return np.corrcoef(ex, ey)[0, 1]


def main():
    L = lanes()
    T = pd.read_csv(OUT / "unit_timing_long.csv")
    sets = {"ResNet": [k for k in L if L[k]["arch"] == "ResNet-18"], "ViT": ["core_ViT-Small_s42"], "CCT": ["core_CCT-7_s42"]}

    def consensus(score, measure, keys):
        d = T[(T.score == score) & (T.measure == measure) & (T.lane.isin(keys))]
        p = d.pivot(index="unit", columns="lane", values="value")
        return ((p - p.mean()) / p.std()).mean(1).values

    pc = pd.read_csv(OUT / "predictors_class.csv").set_index("cls")
    ps = pd.read_csv(OUT / "predictors_superclass.csv").set_index("sc")
    # init-network separability (milestone 0), averaged within set
    for nm, keys in sets.items():
        pc[f"init_nc_ws_{nm}"] = consensus("nc_ws", "init", keys)
        ps[f"init_nc_sc_{nm}"] = consensus("nc_sc", "init", keys)
    rows = []
    targets = {"sc": [("psc", "lead"), ("psc", "early"), ("sc", "lead"), ("sc", "early"), ("nc_sc", "lead"),
                      ("nc_sc", "early"), ("psc", "final")],
               "cls": [("pt", "lead"), ("pt", "early"), ("fine", "lead"), ("ws", "lead"), ("ws", "early"),
                       ("pws", "lead"), ("nc_ws", "lead"), ("pt", "final"), ("pws", "final")]}
    for level, P, finals in (("sc", ps, ("psc", "final")), ("cls", pc, ("pt", "final"))):
        preds = [c for c in P.columns if c not in ("superclass",)]
        for nm, keys in sets.items():
            fin = consensus(*finals, keys)
            for sc_, ms in targets[level]:
                y = consensus(sc_, ms, keys)
                for p in preds:
                    if p.startswith("init_nc") and not p.endswith(nm):
                        continue
                    x = P[p].values
                    rows.append(dict(level=level, set=nm, target=f"{sc_}_{ms}", predictor=p.replace(f"_{nm}", ""),
                                     rho=spearmanr(x, y)[0],
                                     partial_rho_given_final=partial_spearman(x, y, fin) if ms != "final" else np.nan))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "predictor_correlations.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 1000):
        for level in ("sc", "cls"):
            r = R[R.level == level].pivot_table(index=["target", "predictor"], columns="set",
                                                values=["rho", "partial_rho_given_final"]).round(2)
            print(r.to_string())


if __name__ == "__main__":
    main()
