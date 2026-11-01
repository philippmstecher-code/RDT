#!/usr/bin/env python3
"""Extended Data Fig. 2: panels a-d of extended/ed2_panels.py, plus a row e, f on the task-general base.

extended/ed2_panels.main() draws a-d on its 180 x 118 mm canvas; its save is intercepted, the canvas grows by ROW_MM at the
bottom, every artist of a-d keeps its absolute position, and e, f are drawn in the new row.
  e  Fine-class accuracy lost when a subspace is projected out of the frozen first and second stage of the clean
     networks (epochs 0-15), read out by a fresh linear classifier on the second stage: task-general features, a
     random subspace of equal rank and variance, the leading principal directions of equal variance. Mean and t-based
     95% CI over five seeds (TG_dependency/proxy_long.csv; scripts/paper/tg_dependency/substrate_proxy.py).
  f  Extra fine-class accuracy lost by removing the task-general features against the matched random subspace at
     epoch 15: removal in the first stage only, in the second stage only (read-out proxy, five seeds), and with the
     later stages retrained on frozen first and second stages taken at epoch 5 (transplant, three seeds,
     TG_dependency/transplant_long.csv). Bars, means; dots, seeds; error bars, 95% CI.
Level colour rule: fine-class accuracy in ambers; task-general removal darkest solid, random mid dashed, principal
directions light dotted.
Run: ES_FIG_OUT=<repo>/analysis_outputs/paper/figures/interventions python scripts/paper/figures/interventions/ed2.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "extended"))
sys.path.insert(0, str(HERE.parent))
import importlib.util  # noqa: E402

import nmi_style as ns  # noqa: E402

# extended/ed2_panels.py (Extended Data Fig. 2 a-d) by path: figures/validation_panels.py (Extended Data Fig. 1) shadows the name on sys.path
_spec = importlib.util.spec_from_file_location("ed2_panels", HERE.parent / "extended" / "ed2_panels.py")
ED = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ED)

TGD = ns.REPO / "analysis_outputs/paper/TG_dependency" if hasattr(ns, "REPO") else \
    Path(__file__).resolve().parents[4] / "analysis_outputs/paper/TG_dependency"
NAME = "ed2"
H0_MM, ROW_MM = 118, 52
COND = [("tg", "task-general features", ns.AMBER[0], "-", "o"),
        ("rand", "random, equal rank and variance", ns.AMBER[1], "--", "s"),
        ("pcv", "leading directions, equal variance", ns.AMBER[2], ":", "^")]


def ci(v):
    v = np.asarray(v, float)
    h = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), h


def data_e():
    p = pd.read_csv(TGD / "proxy_long.csv")
    base = p[p.cond == "none"].set_index(["seed", "epoch"]).fine
    rows = []
    for c, *_ in COND:
        x = p[p.cond == c].set_index(["seed", "epoch"]).fine
        lost = (base - x).dropna().rename("lost").reset_index()
        lost["cond"] = c
        rows.append(lost)
    return pd.concat(rows, ignore_index=True)


def data_f():
    p = pd.read_csv(TGD / "proxy_long.csv")
    p = p[p.epoch == 15].set_index(["seed", "cond"]).fine
    out = []
    for lab, a, b in (("first stage", "tg@layer1", "rand@layer1"), ("second stage", "tg@layer2", "rand@layer2")):
        for s in sorted(p.index.get_level_values(0).unique()):
            out.append(dict(bar=lab, seed=s, extra=p[(s, b)] - p[(s, a)], source="read-out proxy, epoch 15"))
    t = pd.read_csv(TGD / "transplant_long.csv")
    t = t[(t.run == "tgtransplant_e5") & (t.epoch == 15)].set_index(["seed", "arm"]).val_acc
    for s in sorted(t.index.get_level_values(0).unique()):
        out.append(dict(bar="later stages\nretrained", seed=s, extra=t[(s, "Crand")] - t[(s, "C")],
                        source="transplant from epoch 5, 15 epochs of training"))
    return pd.DataFrame(out)


def main():
    import matplotlib.pyplot as plt
    captured = {}
    save = ns.save_final
    ED.ns.save_final = lambda fig, name, w, ed=False, strict=True: captured.setdefault("fig", fig)
    ED.NAME = NAME
    ED.main()
    ns.save_final = save
    fig = captured["fig"]
    H1 = H0_MM + ROW_MM
    k = H0_MM / H1
    off = ROW_MM / H1
    fig.set_size_inches(180 * ns.MM, H1 * ns.MM)
    for ax in fig.axes:
        b = ax.get_position()
        ax.set_position([b.x0, b.y0 * k + off, b.width, b.height * k])
    for t in fig.texts:
        x, y = t.get_position()
        t.set_position((x, y * k + off))
    for lg in fig.legends:
        lg.set_bbox_to_anchor((0.5, 0.995 * k + off), transform=fig.transFigure)

    mm_y = lambda mm: 1 - mm / H1                       # top-down mm -> figure fraction
    mm_x = lambda mm: mm / 180
    top, hgt = H0_MM + 12, 30                            # row e, f: axes top at 130 mm, 30 mm tall

    # e: accuracy lost over epochs
    E = data_e()
    ax = fig.add_axes([mm_x(24), mm_y(top + hgt), mm_x(80), hgt / H1])
    for c, lab, col, ls, mk in COND:
        g = E[E.cond == c].groupby("epoch").lost
        m = g.mean()
        h = g.apply(lambda v: ci(v)[1])
        ax.fill_between(m.index, m - h, m + h, color=col, alpha=0.18, lw=0)
        ax.plot(m.index, m.values, ls=ls, color=col, marker=mk, ms=2.6, lw=1.0, label=lab)
    ax.axhline(0, color=ns.GREYS[2], lw=0.5, zorder=0)
    ax.set_xlim(-0.5, 15.5)
    ax.set_xticks([0, 1, 2, 3, 5, 10, 15])
    ax.set_xlabel("Epoch of the frozen network")
    ax.set_ylabel("Fine-class accuracy lost (pp)")
    ax.legend(loc="upper left", fontsize=6, frameon=False, handlelength=2.2, borderaxespad=0.2)
    ax.set_title("From epoch 1, the task-general features cost more\nthan matched removals",
                 loc="left", fontsize=7, pad=4)
    fig.text(0.008, ax.get_position().y1 + 0.03, "e", fontsize=8, fontweight="bold", va="bottom")

    # f: extra cost at epoch 15
    F = data_f()
    ax = fig.add_axes([mm_x(128), mm_y(top + hgt), mm_x(48), hgt / H1])
    bars = list(dict.fromkeys(F.bar))
    for i, b in enumerate(bars):
        v = F[F.bar == b].extra.values
        m, h = ci(v)
        ax.bar(i, m, width=0.62, color=ns.AMBER[0] if i != 1 else ns.AMBER[3], edgecolor=ns.AMBER[0], lw=0.6)
        ax.errorbar(i, m, yerr=h, color=ns.GREYS[0], lw=0.8, capsize=2)
        ax.scatter(i + np.linspace(-0.12, 0.12, len(v)), v, s=5, color=ns.GREYS[1], zorder=3, lw=0)
        ax.text(i, max(m + h, 0) + 0.35, f"{m:.1f}", ha="center", va="bottom", fontsize=6.2)
    ax.axhline(0, color=ns.GREYS[2], lw=0.5)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels(bars, fontsize=6.2, linespacing=1.0)
    ax.set_ylabel("Extra accuracy lost vs\nrandom subspace (pp)", linespacing=1.0)
    ax.set_ylim(min(-1, F.extra.min() - 0.5), F.extra.max() + 1.6)
    ax.set_title("The reliance sits in the first stage and\nholds with later stages retrained",
                 loc="left", fontsize=7, pad=4)
    fig.text(mm_x(108), ax.get_position().y1 + 0.03, "f", fontsize=8, fontweight="bold", va="bottom")

    ns.source_data(E, NAME, "e")
    ns.source_data(F, NAME, "f")
    for c, *_ in COND:
        g = E[(E.cond == c)].groupby("epoch").lost
        print(c, g.mean().round(2).to_dict())
    print(F.groupby("bar").extra.agg(["mean", "count"]).round(2))
    out = ns.save_final(fig, NAME, 180, ed=True)
    print("saved", out)


if __name__ == "__main__":
    main()
