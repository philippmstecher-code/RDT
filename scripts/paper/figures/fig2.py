#!/usr/bin/env python3
"""Fig. 2: replication across architectures, seeds and datasets.

a  Construction/refinement ratio (Tg-E + Ab-E)/Di-E at the first (0->1) and the last real transition, mean over
   3 seeds per CIFAR-100 architecture, and the Tiny ImageNet run (#F472B6). Initial bars light (alpha 0.35, edge),
   final bars dark (alpha 0.85). Di-E floored at 1 event (ResNet-18 seeds 137/256 have Di-E = 0 at 0->1).
   Duplicate final transition removed (as in process_composition.py).
b  Process fractions per superclass (share of As-E, De-E, Ab-E, Di-E, Tg-E among all events of the classes of a
   superclass, summed over transitions): CIFAR-100, 20 superclasses, mean over the 9 runs; Tiny ImageNet,
   27 superclasses, one run. Horizontal bars: means over superclasses. Source:
   sae_results.json['class_process_summary'][*]['per_transition'], which sums exactly to the stored
   superclass_summary.process_breakdown; the duplicate final transition is removed (DROP_TERMINAL).
c, d  Mean SSI (c) and mean CSI (d) per checkpoint (sae_results.json['selectivity_evolution']) for the three
   ResNet-18 CIFAR-100 seeds (ResNet-18 colour; solid / dashed / dotted,
   band = range of the three seeds) and Tiny ImageNet
   (#F472B6, markers). Real checkpoints only (CIFAR-100 0-10, Tiny ImageNet 0-8); the duplicate 'terminal'
   checkpoint is removed.
Usage: python scripts/paper/figures/fig2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nmi_style as ns  # noqa: E402
import core_runs_lib as L  # noqa: E402
from core_runs_lib import MM, add_axes_mm, apply_style, f12, plain_log_ticks  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

NAME = "fig2"
W, H = 180.0, 132.0
ARCHS = ("ResNet-18", "ViT-Small", "CCT-7")
GREY_LEG = "#9CA3AF"
CIFAR_DOT = ns.REF_DARK
SEED_LS = {42: "-", 137: "--", 256: ":"}
ORDER = ("as_h", "de_h", "ab_h", "di_h", "tg_h")  # panel b order


def draw_a(ax, runs, tiny):
    cr = f12.cr
    groups = [(a, a) for a in ARCHS] + [("ResNet-18\n(Tiny ImageNet)", "tiny")]
    vals = []
    for lab, key in groups:
        if key == "tiny":
            ini, fin, col = cr(tiny[0]), cr(tiny[-1]), ns.TINY
        else:
            rr = [r for r in runs if r["arch"] == key]
            ini = float(np.mean([cr(r["pi"][0]) for r in rr]))
            fin = float(np.mean([cr(r["pi"][-1]) for r in rr]))
            col = ns.ARCH[key]
        vals.append((lab, ini, fin, col))
    w = 0.3
    for i, (lab, ini, fin, col) in enumerate(vals):
        ax.bar(i - w / 2, ini, w, color=col, alpha=0.35, edgecolor=col, lw=0.8, zorder=2)
        ax.bar(i + w / 2, fin, w, color=col, alpha=0.85, lw=0, zorder=2)
    ax.axhline(1, color="#555555", ls="--", lw=0.5, alpha=0.8, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.38, 1e5)
    plain_log_ticks(ax, [1, 10, 100, 1000, 10000, 100000])
    ax.set_xlim(-0.5, len(vals) - 0.5)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([v[0] for v in vals])
    ax.set_ylabel("(Tg-E + Ab-E) / Di-E ratio")
    ax.set_title("Construction / refinement ratio")
    # direct labels: initial / final on the first group, final values in large type
    ax.text(-w / 2, vals[0][1] * 1.35, "initial", ha="center", va="bottom", fontsize=6, color="#4B5563")
    ax.text(w / 2, vals[0][2] * 5.5, "final", ha="center", va="bottom", fontsize=6, color="#4B5563")
    for i, (lab, ini, fin, col) in enumerate(vals):
        ax.text(i + w / 2 + 0.02, fin * 1.3, f"{fin:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold",
                color=col)
    return vals


def draw_b(ax, cifar, tin):
    rng = np.random.default_rng(0)
    off, w = 0.18, 0.13
    for j, p in enumerate(ORDER):
        for vals, dx, col in ((cifar[:, j], -off, CIFAR_DOT), (tin[:, j], off, ns.TINY)):
            jit = rng.uniform(-w, w, len(vals))
            ax.scatter(j + dx + jit, vals, s=6, color=col, alpha=0.6, lw=0, zorder=3)
            m = vals.mean()
            ax.plot([j + dx - 0.17, j + dx + 0.17], [m, m], color=col, lw=1.1, zorder=4)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([L.PROC5_LAB[p] for p in ORDER])
    ax.set_xlim(-0.6, len(ORDER) - 0.4)
    ax.set_ylim(-2, 78)
    ax.set_yticks(range(0, 61, 10))
    ax.set_ylabel("Process fraction (%)")
    ax.set_title("Process fractions per superclass")
    ax.text(-0.52, 68.5, "CIFAR-100\n(20 superclasses)", ha="left", va="bottom", fontsize=6.5, color=CIFAR_DOT,
            linespacing=1.05)
    ax.text(1.02, 68.5, "Tiny ImageNet\n(27 superclasses)", ha="left", va="bottom", fontsize=6.5, color=ns.TINY,
            linespacing=1.05)

def draw_traj(ax, key, cif, tin, ep42, ylab, title, legend, tiny_lab=None, col=None):
    n = len(next(iter(cif.values())))
    x = np.arange(n)
    Y = np.array([cif[s] for s in (42, 137, 256)])
    ax.fill_between(x, Y.min(0), Y.max(0), color=col, alpha=0.15, lw=0, zorder=1)
    for s in (42, 137, 256):
        ax.plot(x, cif[s], color=col, ls=SEED_LS[s], lw=1.0, zorder=3,
                label=f"CIFAR-100 seed {s}")
    ax.plot(np.arange(len(tin)), tin, color=ns.TINY, lw=1.0, marker="o", ms=2.6, zorder=4, label="Tiny ImageNet")
    ticks = list(range(0, n, 2))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}\n({ep42[t]})" for t in ticks])
    ax.set_xlim(-0.4, n + 1.3)
    ax.set_xlabel("Checkpoint (epoch, CIFAR-100 seed 42)")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.text(n - 1 + 0.15, Y[:, -1].mean(), "CIFAR-100", color=col, fontsize=6.5, ha="left",
            va="center")
    tx, ty, tha = tiny_lab if tiny_lab else (len(tin) - 1 + 0.2, tin[-1], "left")
    ax.text(tx, ty, "Tiny\nImageNet", color=ns.TINY, fontsize=6.5, ha=tha, va="center", linespacing=1.0)
    if legend:
        h = [Line2D([], [], color=col, ls=SEED_LS[s_], lw=1.0, label=f"seed {s_}") for s_ in (42, 137, 256)]
        ax.legend(handles=h, loc="lower right", handlelength=2.4, fontsize=6.5, labelspacing=0.25, title="CIFAR-100",
                  title_fontsize=6.5)
    return Y


def main():
    apply_style()
    runs, tiny = f12.load_all()

    # b data
    per_run = [L.superclass_fractions(L.CORE_EXP, pre) for _, _, pre in L.CORE]
    scs = list(per_run[0])
    assert all(list(r) == scs for r in per_run)
    cifar = np.array([np.mean([r[s] for r in per_run], 0) for s in scs])  # (20, 5) in PROCS5 order
    tin_d = L.superclass_fractions(*L.TINY_LANE)
    tin = np.array(list(tin_d.values()))
    idx = [L.PROCS5.index(p) for p in ORDER]
    cifar_o, tin_o = cifar[:, idx], tin[:, idx]

    # c, d data
    r18 = {seed: pre for arch, seed, pre in L.CORE if arch == "ResNet-18"}
    se = {s: L.selectivity_evolution(L.CORE_EXP, pre) for s, pre in r18.items()}
    se_t = L.selectivity_evolution(*L.TINY_LANE)
    ep42 = L.checkpoint_epochs(L.CORE_EXP, r18[42])
    assert len(ep42) == len(se[42])

    fig = plt.figure(figsize=(W * MM, H * MM))
    ax_a = add_axes_mm(fig, 18, 79, 64, 44)
    ax_b = add_axes_mm(fig, 111, 79, 66, 44)
    ax_c = add_axes_mm(fig, 18, 15, 64, 44)
    ax_d = add_axes_mm(fig, 111, 15, 66, 44)
    vals = draw_a(ax_a, runs, tiny)
    draw_b(ax_b, cifar_o, tin_o)
    Yc = draw_traj(ax_c, "mean_ssi", {s: [e["mean_ssi"] for e in v] for s, v in se.items()},
                   [e["mean_ssi"] for e in se_t], ep42, "Mean SSI", "Cross-dataset SSI trajectories", True,
                   tiny_lab=(4.3, 0.258, "left"), col=ns.ARCH["ResNet-18"])
    Yd = draw_traj(ax_d, "mean_csi", {s: [e["mean_csi"] for e in v] for s, v in se.items()},
                   [e["mean_csi"] for e in se_t], ep42, "Mean CSI", "Cross-dataset CSI trajectories", True,
                   col=ns.ARCH["ResNet-18"])
    for ax, k in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d")):
        ns.panel(ax, k, x=-0.25, y=1.12)
        ns.rdt_tag(ax, x=0.02, ha="left") if k == "d" else ns.rdt_tag(ax)

    # Source data
    rows = [{"bar": lab.replace("\n", " "), "initial_transition": "0->1", "initial_ratio": ini,
             "final_transition": "last real", "final_ratio": fin} for lab, ini, fin, _ in vals]
    rows += [{"bar": f"{r['arch']} seed {r['seed']}", "initial_transition": r["pi"][0]["transition"],
              "initial_ratio": f12.cr(r["pi"][0]), "final_transition": r["pi"][-1]["transition"],
              "final_ratio": f12.cr(r["pi"][-1])} for r in runs]
    rows += [{"bar": "Tiny ImageNet run", "initial_transition": tiny[0]["transition"], "initial_ratio": f12.cr(tiny[0]),
              "final_transition": tiny[-1]["transition"], "final_ratio": f12.cr(tiny[-1])}]
    ns.source_data(pd.DataFrame(rows), NAME, "a")
    db = pd.concat([pd.DataFrame(cifar_o, columns=[L.PROC5_LAB[p] + "_pct" for p in ORDER]).assign(
                        dataset="CIFAR-100 (mean of 9 runs)", superclass=scs),
                    pd.DataFrame(tin_o, columns=[L.PROC5_LAB[p] + "_pct" for p in ORDER]).assign(
                        dataset="Tiny ImageNet", superclass=list(tin_d))], ignore_index=True)
    ns.source_data(db[["dataset", "superclass"] + [L.PROC5_LAB[p] + "_pct" for p in ORDER]], NAME, "b")
    for key, panel in (("mean_ssi", "c"), ("mean_csi", "d")):
        rr = []
        for s, v in se.items():
            rr += [{"run": f"ResNet-18 CIFAR-100 seed {s}", "checkpoint": e["checkpoint"],
                    "epoch": ep42[i] if s == 42 else L.checkpoint_epochs(L.CORE_EXP, r18[s])[i], key: e[key]}
                   for i, e in enumerate(v)]
        rr += [{"run": "ResNet-18 Tiny ImageNet seed 42", "checkpoint": e["checkpoint"], "epoch": "", key: e[key]}
               for e in se_t]
        ns.source_data(pd.DataFrame(rr), NAME, panel)

    print("2a bars (initial, final):", [(v[0].replace("\n", " "), round(v[1], 4), round(v[2], 4)) for v in vals])
    print("2b CIFAR-100 means", dict(zip([L.PROC5_LAB[p] for p in ORDER], cifar_o.mean(0).round(2))),
          "Tiny", dict(zip([L.PROC5_LAB[p] for p in ORDER], tin_o.mean(0).round(2))))
    out = ns.save_final(fig, NAME, 180)
    print("saved", out)


if __name__ == "__main__":
    main()
