"""Developmental timetable, step 3: unit-level (class / superclass) timing measures and their reliability.

Measures per unit u and score q (curves a_u(t) over checkpoints t=0..T, network curve A(t)):
  lead   : mean over intermediate t of [ (a_u(t)-b)/(a_u(T)-b) - (A(t)-b)/(A(T)-b) ], b = chance  (final-level normalised)
  early  : a_u(t1) at the first checkpoint where A gained >= 40% of its total above chance (raw level, not final-normalised)
  final  : a_u(T)
Reliability: (i) split-half within network (even vs odd images; Spearman-Brown), (ii) between-network Spearman by pair type.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timetable_lib import CACHE, CLS_OF_IMG, OUT, SC_OF_IMG, lanes  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

UNIT = {"fine": "cls", "ws": "cls", "pt": "cls", "pws": "cls", "nc_ws": "cls",
        "sc": "sc", "psc": "sc", "nc_sc": "sc", "fine_sc": "sc", "ws_sc": "sc", "sc_cls": "cls"}


def load(key):
    d = np.load(CACHE / f"{key}_img.npz")
    return {k: d[k] for k in d.files}


def unit_curves(img, q, mask=None):
    """(T+1, n_units) mean of score per unit; mask selects images."""
    base = {"fine_sc": "fine", "ws_sc": "ws", "sc_cls": "sc"}.get(q, q)
    M = img[base]
    lab = SC_OF_IMG if UNIT[q] == "sc" else CLS_OF_IMG
    n = lab.max() + 1
    m = np.ones(M.shape[1], bool) if mask is None else mask
    return np.stack([np.bincount(lab[m], M[t, m], n) / np.bincount(lab[m], None, n) for t in range(M.shape[0])])


CHANCE = {"fine": .01, "pt": .01, "fine_sc": .01, "sc": .05, "psc": .05, "sc_cls": .05, "nc_sc": .05,
          "ws": .2, "pws": .2, "ws_sc": .2, "nc_ws": .2}


def measures(C, b):
    """b = chance baseline (not the init value: init-specific biases would otherwise be shared by same-init pairs)."""
    A = C.mean(1)
    gain = C[-1] - b
    rel = (C[1:-1] - b) / np.where(np.abs(gain) < 1e-3, np.nan, gain)
    Ag = (A[1:-1] - b) / (A[-1] - b)
    lead = np.nanmean(rel - Ag[:, None], 0)
    t1 = int(np.argmax((A - b) / (A[-1] - b) >= 0.4))
    early = C[t1]
    return dict(lead=lead, early=early, final=C[-1], init=C[0])


def sb(r):
    return 2 * r / (1 + r)


def pair_type(a, b, L):
    A, B = L[a], L[b]
    if A["arch"] != B["arch"]:
        return "cross-arch (core s42)"
    if A["group"] == "core" or B["group"] == "core":
        return "ResNet core-vs-30ep"
    if A["seed"] == B["seed"]:
        return "ResNet same-init (diff SGD run)"
    return "ResNet diff-seed"


def main():
    L = lanes()
    keys = [k for k in L if (CACHE / f"{k}_img.npz").exists()]
    j = np.tile(np.arange(100), 100)
    rows, rel_rows, tabs = [], [], {}
    for k in keys:
        img = load(k)
        for q in UNIT:
            C = unit_curves(img, q)
            m = measures(C, CHANCE[q])
            for meas, v in m.items():
                tabs[(k, q, meas)] = v
            Ce, Co = unit_curves(img, q, j % 2 == 0), unit_curves(img, q, j % 2 == 1)
            me, mo = measures(Ce, CHANCE[q]), measures(Co, CHANCE[q])
            for meas in ["lead", "early", "final"]:
                ok = np.isfinite(me[meas]) & np.isfinite(mo[meas])
                r = spearmanr(me[meas][ok], mo[meas][ok])[0]
                rel_rows.append(dict(lane=k, score=q, measure=meas, split_half_r=r, spearman_brown=sb(r)))
    rel = pd.DataFrame(rel_rows)
    rel.to_csv(OUT / "reliability_split_half.csv", index=False)
    for a, b in itertools.combinations(keys, 2):
        for q in UNIT:
            for meas in ["lead", "early", "final"]:
                x, y = tabs[(a, q, meas)], tabs[(b, q, meas)]
                ok = np.isfinite(x) & np.isfinite(y)
                r = spearmanr(x[ok], y[ok])[0]
                ra = rel[(rel.lane == a) & (rel.score == q) & (rel.measure == meas)].spearman_brown.iloc[0]
                rb = rel[(rel.lane == b) & (rel.score == q) & (rel.measure == meas)].spearman_brown.iloc[0]
                rows.append(dict(a=a, b=b, type=pair_type(a, b, L), score=q, measure=meas, r=r,
                                 r_ceiling_corrected=r / np.sqrt(max(ra, 1e-3) * max(rb, 1e-3))))
    pr = pd.DataFrame(rows)
    pr.to_csv(OUT / "reliability_between_networks.csv", index=False)
    s = pr.groupby(["score", "measure", "type"]).r.mean().unstack("type").round(2)
    s = s.join(rel.groupby(["score", "measure"]).spearman_brown.mean().round(2).rename("split-half(SB)"))
    s.to_csv(OUT / "reliability_summary.csv")
    sh = ""
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(sh)
        print(s)
    # unit tables
    out = []
    for (k, q, meas), v in tabs.items():
        for u, val in enumerate(v):
            out.append(dict(lane=k, score=q, measure=meas, unit=u, value=val))
    pd.DataFrame(out).to_csv(OUT / "unit_timing_long.csv", index=False)


if __name__ == "__main__":
    main()
