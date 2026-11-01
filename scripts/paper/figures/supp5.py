#!/usr/bin/env python3
"""Supplementary Fig. 5: noise floor of feature events and robustness of the developmental ordering.

a  same-SAE noise floor, ResNet-18 (NOISE_null/tables/resnet_dup_null_per_transition.csv).
b  robustness ledger of the 9 standard runs (MATCHING/process_ordering_by_rule.csv; TRANSFORMED/transformed_results.json;
   no-SAE rise and fall, validation_panels.load_nosae).
Drawing and data code are shared with ed1.py.

Usage: python scripts/paper/figures/supp5.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nmi_style as ns  # noqa: E402
import validation_panels  # noqa: E402
from checks_lib import MMCanvas  # noqa: E402

FIG = "supp5"
W_MM, H_MM = 180, 66


def main():
    ns.apply()
    fig = plt.figure(figsize=(W_MM * ns.MM, H_MM * ns.MM))
    C = MMCanvas(fig, W_MM, H_MM)
    ax_a = C.ax(16, 9, 56, 44)
    ax_b = C.ax(134, 14, 40, 47)
    kdf = validation_panels.panel_noise_floor(ax_a)
    validation_panels.ax_style(ax_a)
    peaks = validation_panels.interior_peaks(validation_panels.load_nosae())
    runs, rows = validation_panels.ledger_data(peaks)
    validation_panels.panel_l(ax_b, runs, rows)
    ns.panel(ax_a, "a", x=-15 / 56, y=1 + 7.5 / 44)
    fig.text(88 / W_MM, 1 - 6.5 / H_MM, "b", fontsize=8, fontweight="bold", va="top", ha="left")
    fig.text(132 / W_MM, 1 - 7.5 / H_MM, "Robustness across runs", fontsize=6.8, va="top", ha="center")

    ns.source_data(kdf, FIG, "a")
    led = pd.DataFrame([{"section": r["section"], "row": r["label"], "count": r["count"],
                         **{f"{a} s{s}": ("not tested" if v is None else ("holds" if v else "fails"))
                            for (a, s), v in zip(runs, r["vals"])}} for r in rows])
    ns.source_data(led, FIG, "b")
    out = ns.save_final(fig, FIG, W_MM, ed=False)
    keep = {k: v for k, v in validation_panels.STORE.items() if k.startswith(("k_null", "l_"))}
    ns.source_data(pd.DataFrame([dict(key=k, value=json.dumps(v)) for k, v in keep.items()]), FIG, "stored_values")
    print(out)
    for k, v in keep.items():
        print(k, v)


if __name__ == "__main__":
    main()
