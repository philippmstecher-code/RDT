#!/usr/bin/env python3
"""Per-seed process events per transition, without the duplicated final transition: a per-seed view of the runs
shown in Extended Data Fig. 3 (extended/ed3.py). It saves under the name ed3, so ES_FIG_OUT must point elsewhere.

3 x 3 grid, rows ResNet-18 / ViT-Small / CCT-7, columns Ab-E / Di-E / Tg-E events per 1,000 weight updates. Seeds in
neutral greys with distinct line styles and markers (42 dark grey solid circles, 137 mid grey dashed squares, 256
light grey dotted diamonds), so that no seed colour collides with the process hues of the columns. The stored
``k->terminal`` transition (last checkpoint vs itself) is removed.
Value = process_intensity[t][ab_h|di_h|tg_h] / delta weight updates x 1,000.
Usage: ES_FIG_OUT=<dir> python scripts/paper/figures/seed_events.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nmi_style as ns  # noqa: E402
from turnover_lib import ARCHS, SEEDS, load_core, per1k  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

NAME = "ed3"
SEED_STYLE = {42: dict(color=ns.REF_DARK, ls="-", marker="o"),
              137: dict(color=ns.NEUTRAL, ls="--", marker="s"),
              256: dict(color="#9CA3AF", ls=":", marker="D")}
SEED_COL = {s: st["color"] for s, st in SEED_STYLE.items()}
SEED_MS = {"o": 3.0, "s": 2.7, "D": 2.5}   # visually matched marker sizes
PROCS = [("ab_h", "Ab-E"), ("di_h", "Di-E"), ("tg_h", "Tg-E")]
LETTERS = "abcdefghi"
LEG_LOC = {("ViT-Small", "Di-E"): "upper left"}
HEADROOM = {("ViT-Small", "Ab-E"): 1.38}


def kfmt(v, _):
    if v == 0:
        return "0"
    return f"{v / 1000:.0f}k"


def main():
    ns.apply()
    runs = load_core()
    rows = []
    for r in runs:
        for i, t in enumerate(r["tr"]):
            rows.append({"arch": r["arch"], "seed": r["seed"], "transition_index": i, "transition": t["transition"],
                         "delta_weight_updates": t["dt_updates"],
                         **{f"{lab}_events": t[k] for k, lab in PROCS},
                         **{f"{lab}_per_1k_updates": per1k(t[k], t["dt_updates"]) for k, lab in PROCS}})
    df = pd.DataFrame(rows)

    def val(a, s, i, lab):
        return df[(df.arch == a) & (df.seed == s) & (df.transition_index == i)][f"{lab}_per_1k_updates"].item()
    # spot checks against reference values
    assert abs(val("ResNet-18", 42, 1, "Ab-E") - 20.6e3) < 300
    assert abs(val("ViT-Small", 256, 1, "Tg-E") - 459e3) < 3e3
    assert abs(val("CCT-7", 137, 1, "Ab-E") - 49.5e3) < 700

    fig = plt.figure(figsize=(180 * ns.MM, 158 * ns.MM))
    left, right, bottom, top = 0.105, 0.985, 0.075, 0.955
    wspace, hspace = 0.075, 0.075
    w = (right - left - 2 * wspace) / 3
    h = (top - bottom - 2 * hspace) / 3
    for ri, arch in enumerate(ARCHS):
        for ci, (k, lab) in enumerate(PROCS):
            ax = fig.add_axes([left + ci * (w + wspace), top - (ri + 1) * h - ri * hspace, w, h])
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(True)
            n_tr = 0
            for s in SEEDS:
                d = df[(df.arch == arch) & (df.seed == s)].sort_values("transition_index")
                n_tr = max(n_tr, len(d))
                st = SEED_STYLE[s]
                ax.plot(d.transition_index, d[f"{lab}_per_1k_updates"], color=st["color"], ls=st["ls"],
                        marker=st["marker"], alpha=0.9, lw=1.0, ms=SEED_MS[st["marker"]], label=f"seed {s}")
            ax.set_xlim(-0.5, n_tr - 0.5)
            ax.set_xticks(range(0, n_tr, 2))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            ax.yaxis.set_major_formatter(FuncFormatter(kfmt))
            ymax = ax.get_ylim()[1]
            head = HEADROOM.get((arch, lab), 1.04)   # room for the legend where seeds peak late
            ax.set_ylim(-0.04 * ymax, ymax * head)
            if ri == 0:
                ax.set_title(f"{lab} events (n = 3 seeds)", fontsize=7)
            if ci == 0:
                ax.set_ylabel(f"{arch}\nEvents / 1k updates")
            if ri == 2:
                ax.set_xlabel("Training transition")
            ax.legend(loc=LEG_LOC.get((arch, lab), "upper right"), fontsize=6, handlelength=2.4, labelspacing=0.3, borderaxespad=0.4)
            ns.panel(ax, LETTERS[ri * 3 + ci], x=-0.2, y=1.1)

    ns.source_data(df, NAME, "a-i")
    out = ns.save_final(fig, NAME, 180, ed=True)
    print("saved", out)
    print("transitions drawn per arch:", {a: int(df[df.arch == a].transition_index.max()) + 1 for a in ARCHS})


if __name__ == "__main__":
    import os as _os
    if not _os.environ.get("ES_FIG_OUT"):
        raise SystemExit("seed_events.py saves under the name ed3 (Extended Data Fig. 3 is drawn by extended/ed3.py); "
                         "set ES_FIG_OUT to a separate output folder to draw this per-seed view.")
    main()
