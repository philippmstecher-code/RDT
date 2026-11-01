#!/usr/bin/env python3
"""Per-superclass, per-transition process counts of the core CIFAR-100 runs and the Tiny ImageNet run.

Extracted with the extractors in scripts/granger/extractors from the raw transition files on the
data drive (read only); the transition label is kept so that the duplicated ``k_to_terminal`` transition can be
dropped downstream (granger_stats.py).

Output: analysis_outputs/paper/SUPERCLASS_series/superclass_transition_series_{cifar100,tiny}.csv
Usage: python scripts/paper/figures/superclass_series.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/granger/extractors"))
import granger_causality_onset as gco  # noqa: E402  (extractor + Tiny ImageNet map)
import granger_causality_superclass as gcs  # noqa: E402  (CIFAR-100 extractor)

OUT = ROOT / "analysis_outputs/paper/SUPERCLASS_series"


def add_labels(df: pd.DataFrame, exp: str, lane_id: str) -> pd.DataFrame:
    tdir = Path(gco.EXT_BASE) / "experiments" / exp / "lanes" / lane_id / "sae_analysis" / "transitions"
    lab = {int(n.split("_to_")[0]): n for n in (p.name for p in tdir.iterdir() if p.is_dir())}
    df["transition_label"] = df["transition"].map(lab)
    df["is_terminal"] = df["transition_label"].str.endswith("_to_terminal")
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(Path(gco.EXT_BASE) / "rcx.db")
    lanes = con.execute("""SELECT l.id, l.name, e.id FROM lanes l JOIN experiments e ON l.experiment_id = e.id
                           WHERE e.name = 'Multi-Arch/-Seed' ORDER BY l.name""").fetchall()
    con.close()
    parts = []
    for lane_id, name, exp in lanes:
        print("extract", name, flush=True)
        d = pd.DataFrame(gcs.extract_superclass_transition_data(lane_id, exp, name))
        parts.append(add_labels(d, exp, lane_id))
    cif = pd.concat(parts, ignore_index=True)
    cif.to_csv(OUT / "superclass_transition_series_cifar100.csv", index=False)

    def tiny_map(i):
        sc = gco.TINY_IDX_TO_SUPER.get(i)
        return sc if sc and sc != "unknown" else None
    print("extract", gco.TINY_LANE_NAME, flush=True)
    t = pd.DataFrame(gco.extract_transition_data(gco.TINY_LANE_ID, gco.TINY_EXP_ID, gco.TINY_LANE_NAME,
                                                 tiny_map, gco.TINY_SC_NAMES))
    t = add_labels(t, gco.TINY_EXP_ID, gco.TINY_LANE_ID)
    t.to_csv(OUT / "superclass_transition_series_tiny.csv", index=False)
    print("done", len(cif), len(t))


if __name__ == "__main__":
    main()
