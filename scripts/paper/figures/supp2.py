#!/usr/bin/env python3
"""Supplementary Fig. 2: CCT-7 at one checkpoint per epoch.

One CCT-7 run on CIFAR-100 (seed 42), one checkpoint per epoch, epochs 0-50 (50 transitions; transition e = epoch
e -> e + 1), stored per-run thresholds; analysis_outputs/paper/PEREPOCH_cct (scripts/paper/perepoch_cct_analysis.py).
a  process composition (share of Tg-E, Ab-E, Di-E among Tg + Ab + Di events, six SAE layers) per transition;
   triangles, count-weighted centre of mass of each process (results.json process.all.com_tg_ab_di).
b  as a, counting each feature once per transition (steps_per_layer.csv).
c  SAE-free superclass share of between-class variance at the pooled layer per epoch (no-SAE measure of ed1.py;
   band, image-bootstrap SD); grey, mean and 95th percentile of 100 random 20 x 5 groupings.
d  events counted by matching directly across an interval of the standard CCT-7 checkpoint grids (seeds 42, 137, 256;
   checkpoint epochs) as % of the events summed over the one-epoch steps inside it, by interval length; identical
   intervals of different grids drawn once; one-epoch intervals equal 100% by construction (not drawn).
e  flicker: features alive at epoch k that are counted dead at k -> k+1 and matched stably again at k+2, % of alive
   features per k; dashed line, pooled rate.

Usage: python scripts/paper/figures/supp2.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nmi_style as ns  # noqa: E402
from core_runs_lib import apply_style, plain_log_ticks  # noqa: E402
from checks_lib import RESULTS, MMCanvas  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

FIG = "supp2"
W_MM, H_MM = 180, 122
D = RESULTS / "PEREPOCH_cct"
PROC = ("Tg-E", "Ab-E", "Di-E")
COLS = {"Tg-E": "tg", "Ab-E": "ab", "Di-E": "di"}
MK = {"Tg-E": "o", "Ab-E": "s", "Di-E": "D"}
GREY = "#6B7280"
XLAB = "Transition (epoch e → e + 1)"
ROW1, ROW2, AH = 10.0, 72.0, 38.0
AX = {"a": (16.0, ROW1, 70.0), "b": (106.0, ROW1, 70.0), "c": (16.0, ROW2, 40.0), "d": (76.0, ROW2, 44.0),
      "e": (138.0, ROW2, 40.0)}


def load():
    R = json.loads((D / "results.json").read_text())
    assert R["validation"] == {"feature_matching_counts_match": "300/300", "process_intensity_match": "50/50"}
    P = pd.read_csv(D / "process_per_transition.csv")
    assert len(P) == 50
    S = pd.read_csv(D / "sae_free_share_per_epoch.csv").sort_values("epoch")
    V = pd.read_csv(D / "sparse_vs_dense.csv")
    F = pd.read_csv(D / "flicker_per_epoch.csv").sort_values("k")
    St = pd.read_csv(D / "steps_per_layer.csv")
    assert St.counts_match_stored.all() and St.layer.nunique() == 6
    G = St.groupby("epoch_a")[["f_tg", "f_ab", "f_di"]].sum().reset_index()
    assert len(G) == 50
    return R, P, S, V, F, G


def com(x):
    x = np.asarray(x, float)
    return float((np.arange(len(x)) * x).sum() / x.sum())


def draw_a(ax, m, com, pos, title, ylabel=True):
    n = len(m)
    x = np.arange(n)
    c1, c2 = m[:, 0], m[:, 0] + m[:, 1]
    for lo, hi, k in ((np.zeros(n), c1, "Tg-E"), (c1, c2, "Ab-E"), (c2, np.full(n, 100.0), "Di-E")):
        ax.fill_between(x, lo, hi, facecolor=ns.PROC[k], edgecolor=ns.PROC[k], alpha=ns.FILL_ALPHA, lw=0.8, zorder=2)
    ax.plot(x, c1, color="white", lw=0.7, zorder=3)
    ax.plot(x, c2, color="white", lw=0.7, zorder=3)
    lo_hi = {"Tg-E": (np.zeros(n), c1), "Ab-E": (c1, c2), "Di-E": (c2, np.full(n, 100.0))}
    for k, xi in zip(PROC, pos):
        lo, hi = lo_hi[k]
        ax.text(xi, float(np.interp(xi, x, (lo + hi) / 2)), k, ha="center", va="center", fontsize=7, color="white",
                zorder=5)
    for j, k in enumerate(PROC):
        ax.plot(com[j], 103.0, marker="v", ms=3.6, color=ns.PROC[k], mec="white", mew=0.3, ls="none", clip_on=False,
                zorder=6)
        ax.text(com[j], 107.5, f"{com[j]:.1f}", fontsize=6, ha="center", va="bottom", color=ns.PROC[k], clip_on=False)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 106)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xticks(np.arange(0, n + 1, 5))
    ax.set_xlabel(XLAB)
    ax.set_ylabel("Process composition (%)")
    ax.set_title(title, fontsize=7, pad=13)
    return m


def draw_b(ax, S):
    e = S.epoch.to_numpy(int)
    y, sd = S.raw_share_sc.to_numpy(float), S.raw_share_boot_sd.to_numpy(float)
    ax.fill_between(e, y - sd, y + sd, color=ns.PROC["Ab-E"], alpha=0.25, lw=0, zorder=2)
    ax.plot(e, y, color=ns.PROC["Ab-E"], lw=1.1, zorder=3)
    k = int(np.argmax(y))
    ax.plot(e[k], y[k] + 0.035, marker="v", ms=3.6, color=ns.PROC["Ab-E"], mec="white", mew=0.3, ls="none", zorder=5)
    ax.plot(e, S.raw_share_sc_null_mean, color=GREY, lw=0.8, zorder=2)
    ax.plot(e, S.raw_share_sc_null_p95, color=GREY, lw=0.6, ls="--", zorder=2)
    ax.text(50, 0.28, "random groupings\n(mean, 95th percentile)", fontsize=6, color=GREY, ha="right", va="bottom",
            linespacing=1.05)
    ax.set_xlim(-1, 51)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_ylim(0, 0.8)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Superclass share of\nbetween-class variance")
    return k


def draw_c(ax, V):
    U = V[V.length_epochs > 1].drop_duplicates(["epoch_a", "epoch_b"]).sort_values(["length_epochs", "epoch_a"])
    rows = []
    for k in PROC:
        cap = 100 * (1 - U[f"missed_frac_{COLS[k]}_events"].to_numpy(float))
        ax.plot(U.length_epochs, cap, marker=MK[k], ms=3.4, ls="none", color=ns.PROC[k], mec="white", mew=0.3,
                alpha=0.9, zorder=3)
        for (_, r), c in zip(U.iterrows(), cap):
            rows.append(dict(process=k, epoch_a=int(r.epoch_a), epoch_b=int(r.epoch_b),
                             length_epochs=int(r.length_epochs), grids=", ".join(
                                 str(g) for g in sorted(V[(V.epoch_a == r.epoch_a) & (V.epoch_b == r.epoch_b)].grid_seed)),
                             captured_pct_of_dense=c))
    ax.axhline(100, color=GREY, lw=0.5, ls="--", zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(1.6, 45)
    plain_log_ticks(ax, [2, 4, 8, 16, 32], axis="x")
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Checkpoint interval (epochs)")
    ax.set_ylabel("Events counted across the interval\n(% of one-epoch count)")
    h = [Line2D([], [], marker=MK[k], ms=3.4, ls="none", color=ns.PROC[k], mec="white", mew=0.3, label=k) for k in PROC]
    ax.legend(handles=h, loc="center right", fontsize=6.5, frameon=False, handletextpad=0.2, borderaxespad=0.2)
    return pd.DataFrame(rows)


def draw_d(ax, F, pooled):
    ax.plot(F.k, F.flicker_pct_alive, "o-", color=ns.REF_DARK, lw=0.8, ms=2.2, zorder=3)
    ax.axhline(pooled, color=GREY, lw=0.7, ls="--", zorder=2)
    ax.text(48, 1.9, f"dashed line, pooled rate ({pooled:.1f}%)", fontsize=6, color=GREY, ha="right", va="top")
    ax.set_xlim(-1, 49)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_ylim(0, 2.0)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xlabel("Epoch k")
    ax.set_ylabel("Features counted dead and\nback one epoch later (% of alive)")


def main():
    apply_style()
    R, P, S, V, F, G = load()
    com_e = R["process"]["all"]["com_tg_ab_di"]
    fv = G[["f_tg", "f_ab", "f_di"]].to_numpy(float)
    com_f = [com(fv[:, j]) for j in range(3)]
    pooled = R["flicker"]["all"]["flicker_pct_alive_pooled"]
    fig = plt.figure(figsize=(W_MM * ns.MM, H_MM * ns.MM))
    C = MMCanvas(fig, W_MM, H_MM)
    axs = {k: C.ax(x, y, w, AH) for k, (x, y, w) in AX.items()}
    draw_a(axs["a"], P[["share_tg", "share_ab", "share_di"]].to_numpy(float), com_e, (4.0, 46.0, 38.0),
           "Events (weighted by images)")
    fsh = 100 * fv / fv.sum(1, keepdims=True)
    draw_a(axs["b"], fsh, com_f, (3.0, 30.0, 40.0), "Features (each counted once)")
    kpk = draw_b(axs["c"], S)
    c_df = draw_c(axs["d"], V)
    draw_d(axs["e"], F, pooled)
    for k, (x, y, w) in AX.items():
        fig.text((x - 14.0) / W_MM, 1 - (y - 7.5) / H_MM, k, fontsize=8, fontweight="bold", va="top", ha="left")

    a_df = P[["transition", "epoch_a", "epoch_b", "tg_h", "ab_h", "di_h", "share_tg", "share_ab", "share_di", "cr"]]
    a_df = a_df.rename(columns={"tg_h": "Tg-E events", "ab_h": "Ab-E events", "di_h": "Di-E events",
                                "share_tg": "Tg-E share (%)", "share_ab": "Ab-E share (%)", "share_di": "Di-E share (%)",
                                "cr": "construction/refinement ratio"})
    ns.source_data(a_df.round(4), FIG, "a")
    b_df = pd.DataFrame({"transition": a_df.transition, "Tg-E features": fv[:, 0].astype(int),
                         "Ab-E features": fv[:, 1].astype(int), "Di-E features": fv[:, 2].astype(int),
                         "Tg-E share (%)": fsh[:, 0], "Ab-E share (%)": fsh[:, 1], "Di-E share (%)": fsh[:, 2],
                         "construction/refinement ratio": (fv[:, 0] + fv[:, 1]) / np.maximum(fv[:, 2], 1)})
    ns.source_data(b_df.round(4), FIG, "b")
    ns.source_data(pd.DataFrame([{"basis": b, "process": p, "centre of mass (transition)": round(c, 4)}
                                 for b, cc in (("events", com_e), ("features", com_f)) for p, c in zip(PROC, cc)]),
                   FIG, "a_b_centre_of_mass")
    ns.source_data(S[["epoch", "val_acc", "raw_share_sc", "raw_share_boot_sd", "raw_share_sc_null_mean",
                      "raw_share_sc_null_p95"]].round(5), FIG, "c")
    ns.source_data(c_df.round(3), FIG, "d")
    ns.source_data(F[["k", "n_alive", "n_died_step1", "n_flicker_step1", "flicker_pct_alive"]].round(4), FIG, "e")
    out = ns.save_final(fig, FIG, W_MM, ed=False)
    print(out, f"{W_MM} x {H_MM} mm")
    print("COM events", com_e, "features", np.round(com_f, 3).tolist(), "| peak share epoch", int(S.epoch.iloc[kpk]))


if __name__ == "__main__":
    main()
