#!/usr/bin/env python3
"""Supplementary Fig. 7: class-selective features grow from features that already lean towards their superclass,
and the coarse block removes that lean (ancestry analysis on the sham runs and blocking arms of the blocking
experiment, five seeds, epochs 0-15, census layers pooled, one common threshold set per seed).

  a  predecessor sibling preference (log2 of the leave-own-class-out preference for the other classes of the new
     feature's superclass), no block, against 20 random partitions of the classes into groups of five; seeds joined
  b  origins of newly class-selective features without block: from an existing feature; with a superclass-selective
     predecessor of their own superclass (against 100 random partitions); with a task-general predecessor
  c  predecessor sibling preference during the block (epochs 2-10) and after release (11-15) under no block, the
     matched block, the coarse block and the random-grouping block
Data: analysis_outputs/paper/RDT_lineage/analyses/ancestry_summary.json and ancestry_long.csv.

Usage: [ES_FIG_OUT=<dir>] python scripts/paper/figures/interventions/supp7.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import nmi_style as ns  # noqa: E402

A1 = HERE.parents[3] / "analysis_outputs/paper/RDT_lineage/analyses"
NAME, W, H = "supp7", 180, 62
SEEDS = [7, 42, 137, 256, 314]
BLUE, GREYS = ns.BLUE, ns.GREYS
DARK = GREYS[0]


def ci(v):
    v = np.asarray(v, float)
    m = v.mean(); h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return m, m - h, m + h


def bar(ax, x, v, color, w=0.6, hatch=None, fc=None):
    m, lo, hi = ci(v)
    ax.bar(x, m, w, color=fc or color, edgecolor=color, hatch=hatch, lw=0.8, zorder=2)
    ax.plot([x, x], [lo, hi], color=DARK, lw=0.8, zorder=4)
    rng = np.random.default_rng(int(x * 10) + 3)
    ax.scatter(x + rng.uniform(-0.13, 0.13, len(v)), v, s=6, facecolor="white", edgecolor=DARK, lw=0.5, zorder=5)
    return m, lo, hi


def main():
    ns.apply()
    S = json.load(open(A1 / "ancestry_summary.json"))
    L = pd.read_csv(A1 / "ancestry_long.csv")
    ev = L[(L.kind == "event")]
    null = ev[ev.arm == "sham"].groupby("seed").pred_sib_null.mean().reindex(SEEDS)
    tg = ev[ev.arm == "sham"].groupby("seed").pred_tg.mean().reindex(SEEDS)
    fig = plt.figure(figsize=(W * ns.MM, H * ns.MM))
    mm = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
    a, b, c = mm(14, 12, 30, 36), mm(70, 12, 38, 36), mm(124, 12, 54, 36)
    rows = []

    # a
    obs = np.array(S["all|pooled|sham"]["log2_pred_sib"]["per_seed"]); nul = np.log2(null.values)
    for i in range(5):
        a.plot([0, 1], [obs[i], nul[i]], color=GREYS[2], lw=0.5, zorder=1)
    m1 = bar(a, 0, obs, BLUE[0]); m0 = bar(a, 1, nul, GREYS[2], fc="white")
    a.axhline(0, color=GREYS[2], lw=0.6, ls="--", zorder=0)
    a.text(0, m1[2] + 0.12, f"{2 ** m1[0]:.1f}×", ha="center", fontsize=7, color=ns.BLUE_TXT[0])
    a.set_xticks([0, 1]); a.set_xticklabels(["class-feature\npredecessors", "random\npartitions"], fontsize=6.5)
    a.set_ylabel("Preference for the other classes\nof the superclass (log2)")
    a.set_ylim(-0.4, 2.3); a.set_xlim(-0.6, 1.6)
    rows += [dict(panel="a", group=g, seed=s, value=v) for g, vv in (("predecessors, no block", obs),
             ("20 random partitions", nul)) for s, v in zip(SEEDS, vv)]

    # b
    ex = np.array(S["all|pooled|sham"]["share_from_existing"]["per_seed"]) * 100
    sc = np.array(S["all|pooled|sham"]["direct_obs"]["per_seed"]) * 100
    scn = np.array(S["all|pooled|sham"]["direct_null"]["per_seed"]) * 100
    tgv = tg.values * 100
    items = [("existing\nfeature", ex, BLUE[2], None), ("superclass-\nselective", sc, BLUE[0], None),
             ("same,\nrandom", scn, GREYS[2], "white"), ("task-\ngeneral", tgv, GREYS[1], None)]
    for k, (lab, v, col, fc) in enumerate(items):
        m = bar(b, k, v, col, fc=fc)
        b.text(k, max(m[2], v.max()) + 3, f"{m[0]:.1f}%" if m[0] < 10 else f"{m[0]:.0f}%", ha="center", fontsize=6.5,
               color=DARK)
        rows += [dict(panel="b", group=lab.replace("\n", " "), seed=s, value=x) for s, x in zip(SEEDS, v)]
    b.set_xticks(range(4)); b.set_xticklabels([i[0] for i in items], fontsize=6)
    b.set_ylabel("Newly class-selective\nfeatures (%)"); b.set_ylim(0, 92)

    # c
    arms = [("sham", "no block", GREYS[2], None), ("gradjoint", "matched\nblock", BLUE[2], None),
            ("gradsc", "coarse\nblock", BLUE[0], None), ("gradrg", "random\ngrouping", BLUE[1], "///")]
    for wi, (win, wlab) in enumerate((("block", "during the block"), ("release", "after release"))):
        for k, (arm, lab, col, hatch) in enumerate(arms):
            v = np.array(S[f"{win}|pooled|{arm}"]["log2_pred_sib"]["per_seed"])
            x = wi * 5 + k
            bar(c, x, v, col, w=0.75, hatch=hatch, fc="white" if hatch else None)
            rows += [dict(panel="c", group=f"{wlab}, {lab.replace(chr(10), ' ')}", seed=s, value=y)
                     for s, y in zip(SEEDS, v)]
    c.set_xticks([1.5, 6.5]); c.set_xticklabels(["during the block", "after release"], fontsize=6.5)
    c.tick_params(axis="x", length=0)
    from matplotlib.patches import Patch
    c.legend(handles=[Patch(facecolor="white" if h else col, edgecolor=col, hatch=h, label=lab.replace("\n", " "))
                      for _, lab, col, h in arms], loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2,
             fontsize=6, frameon=False, handlelength=1.0, columnspacing=0.8, handletextpad=0.4)
    c.axhline(0, color=GREYS[2], lw=0.6, ls="--", zorder=0)
    c.set_ylabel("Predecessor preference\n(log2)"); c.set_ylim(-0.2, 3.0)
    for ax in (a, b, c):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    T = H - 2
    for ax, letter, x, title in ((a, "a", 1, "Class features grow from features\nthat already favour their superclass"),
                                 (b, "b", 56, "Where new class features\ncome from (no block)"),
                                 (c, "c", 112, "The coarse block weakens the lean,\nalso after release")):
        fig.text(x / W, T / H, letter, fontsize=8, fontweight="bold", va="top")
        fig.text((x + 3.8) / W, (T - 0.2) / H, title, fontsize=7, va="top", linespacing=1.1)
    ns.source_data(pd.DataFrame(rows), NAME, "a_c")
    print("saved", ns.save_final(fig, NAME, 180, strict=False))


if __name__ == "__main__":
    main()
