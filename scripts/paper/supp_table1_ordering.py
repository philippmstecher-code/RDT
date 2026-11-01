#!/usr/bin/env python3
"""Supplementary Table 1: the developmental ordering counted per event and per feature in the 9 standard runs.

Source: analysis_outputs/paper/MATCHING/process_ordering_by_rule.csv (scripts/paper/matching_core.py ->
matching_aggregate.py), real transitions only (the duplicate final transition excluded). Centres of mass on normalised
transition position (0 = first, 1 = last real transition).
  event basis    sample-weighted Tg-E / Ab-E / Di-E events, pipeline (Hungarian) matching, all SAE layers
                 (rule 'hungarian_pipeline_all_layers', as Figs 1-2)
  feature basis  features carrying a Tg-E / Ab-E / Di-E label at a transition, each counted once, pipeline matching;
                 ResNet-18 layer4 left out because it decomposes the same activations as the pooled layer
                 (rule 'hungarian', weighting 'feature_level')
  matching rules feature-basis strict order under mutual-best, greedy and confirmed matching (same file)
Writes analysis_outputs/paper/SI/supp_table1_ordering.csv.
Usage: python scripts/paper/supp_table1_ordering.py
"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "analysis_outputs/paper/MATCHING/process_ordering_by_rule.csv"
OUT = REPO / "analysis_outputs/paper/SI/supp_table1_ordering.csv"
ARCH_ORDER = {"ResNet-18": 0, "ViT-Small": 1, "CCT-7": 2}


def main():
    d = pd.read_csv(SRC)
    ev = d[(d.rule == "hungarian_pipeline_all_layers") & (d.weighting == "sample_weighted")].set_index("lane")
    fe = d[(d.rule == "hungarian") & (d.weighting == "feature_level")].set_index("lane")
    alt = {r: d[(d.rule == r) & (d.weighting == "feature_level")].set_index("lane").order_tg_ab_di
           for r in ("mutual_best", "greedy", "confirmed")}
    assert len(ev) == len(fe) == 9
    rows = []
    for lane, e in ev.iterrows():
        f = fe.loc[lane]
        rows.append({"arch": e.arch, "seed": int(e.seed), "n_transitions": int(e.n_real_transitions),
                     **{f"event_com_{p}": e[f"com_{p}"] for p in ("tg", "ab", "di")},
                     "event_order": bool(e.order_tg_ab_di), "event_cr_first": e.cr_first, "event_cr_last": e.cr_last,
                     "event_cr_declines": bool(e.cr_declines), "event_cr_last_below1": bool(e.cr_last_below1),
                     **{f"feature_com_{p}": f[f"com_{p}"] for p in ("tg", "ab", "di")},
                     "feature_order": bool(f.order_tg_ab_di), "feature_cr_first": f.cr_first,
                     "feature_cr_last": f.cr_last, "feature_cr_declines": bool(f.cr_declines),
                     "feature_cr_last_below1": bool(f.cr_last_below1),
                     **{f"feature_order_{r}": bool(alt[r].loc[lane]) for r in alt}})
    T = pd.DataFrame(rows).sort_values(["arch", "seed"], key=lambda s: s.map(ARCH_ORDER) if s.name == "arch" else s)
    T.to_csv(OUT, index=False)
    print(T.round(3).to_string(index=False))
    print({c: int(T[c].sum()) for c in T.columns if T[c].dtype == bool})


if __name__ == "__main__":
    main()
