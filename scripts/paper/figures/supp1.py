#!/usr/bin/env python3
"""Supplementary Fig. 1: process composition at one checkpoint per epoch.

Three clean ResNet-18 runs on CIFAR-100 (seeds 7, 42, 137), one checkpoint per epoch, epochs 0-15, i.e. 15 epoch
transitions (transition e = epoch e -> e + 1). Counts are the stored per-transition Tg-E / Ab-E / Di-E event counts
summed over the five SAE layers (analysis_outputs/paper/SHAM_perepoch/results.json, written by
scripts/paper/sham_perepoch_analysis.py). Conventions of Fig. 1b,c.
a  process composition (share of Tg-E, Ab-E, Di-E among Tg + Ab + Di events), mean of the three seeds; white
   boundaries and in-band labels as Fig. 1b; triangles, transition of each process's largest mean share (Fig. 1b).
b  construction/refinement ratio (Tg-E + Ab-E) / Di-E (Di-E counted as at least one event), each seed and the
   median, log axis, dashed line at 1 (Fig. 1c).
c  centre of mass (transition index, weighted by event count) of Tg-E, Ab-E and Di-E per seed.
d  Ab-E share of Tg + Ab + Di events per transition, each seed; triangles, each seed's peak; dark segments, mean
   over the three seeds and transitions 0-2, 6-10 and 12-14.
e, f  as a and c, counting each feature once per transition (feature basis,
   analysis_outputs/paper/SHAM_perepoch/feature_basis_summary.json).
Seed styles from seed_events.py (neutral greys with distinct line styles and markers; seed 7 takes the seed-256 style: light
grey, dotted, diamonds); in d the same styles in the blue ramp (Ab-E share = superclass level). Overlapping peak
triangles in a and e are offset horizontally so that every process's triangle stays visible.

Usage: python scripts/paper/figures/supp1.py
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
from seed_events import SEED_MS, SEED_STYLE as ED3_SEED_STYLE  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

FIG = "supp1"
W_MM, H_MM = 180, 190
SRC = RESULTS / "SHAM_perepoch" / "results.json"
SEEDS = (7, 42, 137)
SEED_STYLE = {7: ED3_SEED_STYLE[256], 42: ED3_SEED_STYLE[42], 137: ED3_SEED_STYLE[137]}
SEED_COL = {s: st["color"] for s, st in SEED_STYLE.items()}
PROC = ("Tg-E", "Ab-E", "Di-E")
WINDOWS = ((0, 2), (6, 10), (12, 14))   # inclusive transition ranges for the window means in d
SEED_KW = dict(lw=1.0, alpha=0.9)
SUMMARY_KW = dict(color=ns.REF_DARK, lw=2.6, alpha=0.4)   # median / window means: broad translucent line, so the
                                                          # dark-grey seed-42 line stays distinct


# d measures the superclass level (Ab-E share), so its seeds take the blue ramp, darkest for seed 42 (solid, circles), then 137 (dashed, squares) and 7 (dotted, diamonds)
SEED_BLUE = {42: ns.BLUE[0], 137: ns.BLUE[1], 7: ns.BLUE[2]}


def seed_kw(s, color=None):
    st = SEED_STYLE[s]
    return dict(color=color or st["color"], ls=st["ls"], marker=st["marker"], ms=SEED_MS[st["marker"]] * 2.6 / 3.0,
                **SEED_KW)
XLAB = "Transition (epoch e → e + 1)"

# layout (mm from the top-left corner)
ROW1, ROW2, ROW3, AH = 6.0, 70.0, 136.0, 42.0
AX = {"a": (14.0, ROW1, 70.0), "b": (108.0, ROW1, 70.0), "c": (24.0, ROW2, 46.0), "d": (90.0, ROW2, 88.0),
      "e": (14.0, ROW3, 70.0), "f": (132.0, ROW3, 46.0)}
FB = RESULTS / "SHAM_perepoch" / "feature_basis_summary.json"


def load():
    r = json.loads(SRC.read_text())
    X = np.array([r[str(s)]["counts"] for s in SEEDS], float)                     # seeds x 15 x (Tg, Ab, Di)
    assert X.shape == (3, 15, 3)
    S = 100 * X / X.sum(2, keepdims=True)
    CR = (X[:, :, 0] + X[:, :, 1]) / np.maximum(X[:, :, 2], 1)
    t = np.arange(X.shape[1])
    COM = (t[None, :, None] * X).sum(1) / X.sum(1)                                  # seeds x 3
    # recomputed values must equal the stored (rounded) ones
    for i, s in enumerate(SEEDS):
        e = r[str(s)]["per_epoch"]
        assert np.abs(S[i].round(1) - np.array(e["shares"])).max() < 1e-9
        assert np.abs(CR[i].round(3) - np.array(e["cr"])).max() < 1e-9
        assert np.abs(COM[i].round(2) - np.array(e["com_tg_ab_di"])).max() < 1e-9
        assert int(np.argmax(S[i, :, 1])) == e["peak_share_transition"][1]
    return r, X, S, CR, COM


def load_features():
    """Feature basis (scripts/paper/perepoch_feature_basis.py): features carrying a Tg-E / Ab-E / Di-E label per
    transition, each counted once, summed over the five SAE layers; event counts there equal SRC (asserted)."""
    r = json.loads(FB.read_text())
    assert r["counts_match_stored"].split("/")[0] == r["counts_match_stored"].split("/")[1]
    F = np.array([r[str(s)]["feature_counts"] for s in SEEDS], float)
    assert F.shape == (3, 15, 3)
    SF = 100 * F / F.sum(2, keepdims=True)
    t = np.arange(F.shape[1])
    COMF = (t[None, :, None] * F).sum(1) / F.sum(1)
    return F, SF, COMF


def draw_a(ax, S):
    m = S.mean(0)
    n = m.shape[0]
    x = np.arange(n)
    c1, c2 = m[:, 0], m[:, 0] + m[:, 1]
    for lo, hi, k in ((np.zeros(n), c1, "Tg-E"), (c1, c2, "Ab-E"), (c2, np.full(n, 100.0), "Di-E")):
        ax.fill_between(x, lo, hi, facecolor=ns.PROC[k], edgecolor=ns.PROC[k], alpha=ns.FILL_ALPHA, lw=0.8, zorder=2)
    ax.plot(x, c1, color="white", lw=0.7, zorder=3)
    ax.plot(x, c2, color="white", lw=0.7, zorder=3)
    lo_hi = {"Tg-E": (np.zeros(n), c1), "Ab-E": (c1, c2), "Di-E": (c2, np.full(n, 100.0))}
    for k, xi in (("Tg-E", 0.9), ("Ab-E", 9.0), ("Di-E", 7.0)):
        lo, hi = lo_hi[k]
        yy = float(np.interp(xi, x, (lo + hi) / 2))
        ax.text(xi, yy, k, ha="center", va="center", fontsize=7, color="white", zorder=5)
    peaks = [int(np.argmax(m[:, j])) for j in range(3)]
    for j, k in enumerate(PROC):
        same = [jj for jj in range(3) if peaks[jj] == peaks[j]]      # offset triangles that share a transition
        dx = 0.32 * (same.index(j) - (len(same) - 1) / 2)
        ax.plot(peaks[j] + dx, 103.0, marker="v", ms=3.6, color=ns.PROC[k], mec="white", mew=0.3, ls="none",
                clip_on=False, zorder=6)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 106)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xticks(np.arange(0, n, 2))
    ax.set_xlabel(XLAB)
    ax.set_ylabel("Process composition (%)")
    return m, peaks


def draw_b(ax, CR):
    n = CR.shape[1]
    x = np.arange(n)
    for i, s in enumerate(SEEDS):
        ax.plot(x, CR[i], zorder=3, **seed_kw(s))
    med = np.median(CR, 0)
    ax.plot(x, med, zorder=4, **SUMMARY_KW)
    ax.axhline(1, color="#999999", ls="--", lw=0.5, zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(0.1, 60)
    plain_log_ticks(ax, [0.1, 1, 10])
    ax.text(6.6, 1.12, "ratio = 1", color="#6B7280", fontsize=6, ha="left", va="bottom")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xticks(np.arange(0, n, 2))
    ax.set_xlabel(XLAB)
    ax.set_ylabel("Construction/refinement ratio")
    h = [Line2D([], [], label="Median (n = 3)", **SUMMARY_KW),
         *[Line2D([], [], label=f"seed {s}", **seed_kw(s)) for s in SEEDS]]
    ax.legend(handles=h, loc="upper right", bbox_to_anchor=(1.0, 1.02), handlelength=2.0, labelspacing=0.3,
              fontsize=6.5, borderaxespad=0.2)
    return med


def draw_c(ax, COM):
    off = {"Tg-E": -0.24, "Ab-E": 0.0, "Di-E": 0.24}
    for i, s in enumerate(SEEDS):
        for j, p in enumerate(PROC):
            y = i + off[p]
            ax.plot(COM[i, j], y, marker="o", ms=4.0, color=ns.PROC[p], mec="white", mew=0.4, ls="none", zorder=3)
            ax.text(COM[i, j] + 0.35, y, f"{np.floor(COM[i, j] * 10 + 0.5) / 10:.1f}", fontsize=6, va="center", ha="left", color=ns.PROC[p])
    ax.set_yticks(range(len(SEEDS)))
    ax.set_yticklabels([f"seed {s}" for s in SEEDS])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(len(SEEDS) - 0.5, -0.6)
    ax.set_xlim(0, 14)
    ax.set_xticks(np.arange(0, 15, 2))
    ax.set_xlabel("Centre of mass (transition)")
    h = [Line2D([], [], marker="o", ms=4.0, color=ns.PROC[p], mec="white", mew=0.4, ls="none", label=p) for p in PROC]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3, fontsize=6.5, handletextpad=0.1,
              columnspacing=0.9, borderaxespad=0.0)


def draw_d(ax, S):
    ab = S[:, :, 1]
    n = ab.shape[1]
    x = np.arange(n)
    for a, b in WINDOWS:
        ax.axvspan(a - 0.3, b + 0.3, color=ns.NEUTRAL, alpha=0.08, lw=0, zorder=0)
    peaks = []
    for i, s in enumerate(SEEDS):
        ax.plot(x, ab[i], zorder=3, **seed_kw(s, SEED_BLUE[s]))
        k = int(np.argmax(ab[i]))
        peaks.append((s, k, ab[i, k]))
        ax.plot(k, ab[i, k] + 2.6, marker="v", ms=3.6, color=SEED_BLUE[s], mec="white", mew=0.3, ls="none", zorder=5)
    means = []
    for a, b in WINDOWS:
        mv = ab[:, a:b + 1].mean()
        means.append((a, b, mv))
        ax.plot([a - 0.3, b + 0.3], [mv, mv], solid_capstyle="butt", zorder=4, **SUMMARY_KW)
        ax.text((a + b) / 2, 55.0, f"{mv:.0f}%", ha="center", va="top", fontsize=6.5, color=ns.REF_DARK, zorder=6)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 56)
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_xticks(np.arange(0, n, 1))
    ax.set_xlabel(XLAB)
    ax.set_ylabel("Ab-E share of events (%)")
    h = [*[Line2D([], [], label=f"seed {s}", **seed_kw(s, SEED_BLUE[s])) for s in SEEDS],
         Line2D([], [], color=ns.BLUE[1], marker="v", ms=3.6, ls="none", label="Peak"),
         Line2D([], [], label="Window mean", **SUMMARY_KW)]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, handlelength=1.8, fontsize=6.5,
              borderaxespad=0.3, columnspacing=1.0, handletextpad=0.5)
    return peaks, means


def main():
    apply_style()
    r, X, S, CR, COM = load()
    fig = plt.figure(figsize=(W_MM * ns.MM, H_MM * ns.MM))
    C = MMCanvas(fig, W_MM, H_MM)
    axs = {k: C.ax(x, y, w, AH) for k, (x, y, w) in AX.items()}
    m, peaks_a = draw_a(axs["a"], S)
    med = draw_b(axs["b"], CR)
    draw_c(axs["c"], COM)
    peaks_d, means_d = draw_d(axs["d"], S)
    F, SF, COMF = load_features()
    mf, peaks_e = draw_a(axs["e"], SF)
    axs["e"].set_title("Counted per feature", fontsize=7, pad=9)
    for t in axs["e"].texts:
        if t.get_text() == "Tg-E":                      # thin band: label above it in the process colour
            t.set_position((1.6, 11.0)); t.set_color(ns.PURPLE[0]); t.set_ha("left")   # dark purple: legible on blue
    draw_c(axs["f"], COMF)
    axs["f"].set_title("Centre of mass, counted per feature", fontsize=7, pad=14)
    for k, (x, y, w) in AX.items():
        lx = x - (21.5 if k in ("c", "f") else 11.5)
        fig.text(lx / W_MM, 1 - (y - 3.5) / H_MM, k, fontsize=8, fontweight="bold", va="top", ha="left")

    # Source Data
    tr = [f"{i}->{i + 1}" for i in range(S.shape[1])]
    a_df = pd.DataFrame({"transition_index": range(len(tr)), "transition": tr,
                         **{f"{p} share mean of seeds (%)": m[:, j] for j, p in enumerate(PROC)}})
    for i, s in enumerate(SEEDS):
        for j, p in enumerate(PROC):
            a_df[f"{p} share seed {s} (%)"] = S[i, :, j]
    ns.source_data(a_df.round(4), FIG, "a")
    rows = []
    for i, s in enumerate(SEEDS):
        for t in range(len(tr)):
            rows.append({"seed": s, "transition_index": t, "transition": tr[t], "Tg-E events": int(X[i, t, 0]),
                         "Ab-E events": int(X[i, t, 1]), "Di-E events": int(X[i, t, 2]),
                         "construction/refinement ratio": round(CR[i, t], 6)})
    for t in range(len(tr)):
        rows.append({"seed": "median of 3", "transition_index": t, "transition": tr[t],
                     "construction/refinement ratio": round(med[t], 6)})
    ns.source_data(pd.DataFrame(rows), FIG, "b")
    ns.source_data(pd.DataFrame([{"seed": s, **{f"{p} centre of mass (transition)": round(COM[i, j], 4)
                                                for j, p in enumerate(PROC)}} for i, s in enumerate(SEEDS)]), FIG, "c")
    d_df = pd.DataFrame({"transition_index": range(len(tr)), "transition": tr,
                         **{f"Ab-E share seed {s} (%)": S[i, :, 1].round(4) for i, s in enumerate(SEEDS)}})
    d_df["peak of seeds"] = [", ".join(str(s) for s, k, _ in peaks_d if k == t) for t in range(len(tr))]
    d_df["window mean of seeds (%)"] = [next((round(mv, 4) for a, b, mv in means_d if a <= t <= b), np.nan)
                                        for t in range(len(tr))]
    ns.source_data(d_df, FIG, "d")
    e_df = pd.DataFrame({"transition_index": range(len(tr)), "transition": tr,
                         **{f"{p} feature share mean of seeds (%)": mf[:, j] for j, p in enumerate(PROC)}})
    for i, s in enumerate(SEEDS):
        for j, p in enumerate(PROC):
            e_df[f"{p} features seed {s}"] = F[i, :, j].astype(int)
    ns.source_data(e_df.round(4), FIG, "e")
    ns.source_data(pd.DataFrame([{"seed": s, **{f"{p} centre of mass, features (transition)": round(COMF[i, j], 4)
                                                for j, p in enumerate(PROC)}} for i, s in enumerate(SEEDS)]), FIG, "f")

    out = ns.save_final(fig, FIG, W_MM, ed=False)
    print(out, f"{W_MM} x {H_MM} mm")
    print("a  mean shares (Tg, Ab, Di) per transition:")
    for t in range(len(tr)):
        print(f"   {t:2d}  {m[t, 0]:5.1f} {m[t, 1]:5.1f} {m[t, 2]:5.1f}")
    print("   peak transition of the mean share (Tg, Ab, Di):", peaks_a)
    print("b  C/R first -> last per seed:", {s: (round(CR[i, 0], 2), round(CR[i, -1], 2)) for i, s in enumerate(SEEDS)})
    print("   median per transition:", med.round(2).tolist())
    print("   range all seeds:", round(CR.min(), 3), round(CR.max(), 3))
    print("   first transition below 1 per seed:", {s: int(np.flatnonzero(CR[i] < 1)[0]) for i, s in enumerate(SEEDS)})
    print("   transitions with median < 1:", np.flatnonzero(med < 1).tolist())
    print("c  COM (Tg, Ab, Di):", {s: COM[i].round(2).tolist() for i, s in enumerate(SEEDS)})
    print("d  peaks (seed, transition, share):", [(s, k, round(v, 1)) for s, k, v in peaks_d])
    print("   window means:", [(a, b, round(mv, 2)) for a, b, mv in means_d])
    print("   per-seed window means:", {f"{a}-{b}": S[:, a:b + 1, 1].mean(1).round(1).tolist() for a, b in WINDOWS})


if __name__ == "__main__":
    main()
