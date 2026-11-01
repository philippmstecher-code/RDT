"""Pre-building the coarse structure: loaders, helpers (band, end_labels, cumulative, lead_vs_drawn) and panels used by
Fig. 6 (interventions/fig6.py) and Extended Data Fig. 9 (interventions/ed9.py). Run directly,
the script draws the 2 x 2 layout below, 180 mm; top row coarse (blue), bottom row fine (amber).

Colour = level measured (nmi_style.BLUE / AMBER / GREYS); condition = shade + line style (true superclasses darkest
solid, random grouping mid dashed, superclass-blind task light dotted, scratch thin grey solid); direct labels.

a  share of alive pooled-layer SAE features selective for the true superclasses (census_genuine: SSI above the common
   threshold and above the feature's breadth-matched null, hierarchy_sae.py) over pre-training epochs 0-20, superclass and
   random-grouping pre-training; reference, fine-class training from scratch over the same epochs; light dotted, the
   random-grouping arm's selectivity for its own groups (census_imp_genuine).
   DICTFREE/hierarchy/D_sae_census_timecourse.csv, src 'sae'; seeds 7, 42, 137.
b  cumulative superclass events (Ab-E; one common threshold set per seed, last block, raw counts;
   curriculum_lib.events_common), aligned at the first fine-class epoch; dots, share recorded before the first
   fine-class label (range over seeds and both groupings). n = 5.
c  class features confined to their superclass (count per checkpoint; RDT_pace/curriculum_long.csv n_cl_confined)
   at fine-class epochs, pre-built against scratch. n = 5.
d  fine-class validation accuracy aligned at the first fine-class epoch (L.fine_val); dotted grey, the scratch mean
   shifted four epochs earlier; lead after one fine-class epoch per seed: epoch at which the same seed's scratch run
   (running maximum from 1% at epoch 0) reaches the pre-built accuracy, minus 1. n = 5.
The superclass-blind arm (blind_pretrain) is added to every panel whose data exist (interventions/blind_arm.py). Curves are
means with t-based 95% CI over seeds.

Usage: [ES_FIG_OUT=<dir>] python scripts/paper/figures/interventions/fig6_panels.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # scripts/paper/figures
sys.path.insert(0, str(HERE))
import nmi_style as ns  # noqa: E402
import curriculum_lib as L  # noqa: E402
import blind_arm as B  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from scipy import stats  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

RESULTS = ns.OUT.parent
NAME, W, H = "fig6", 180, 124
SEEDS = [7, 42, 137, 256, 314]
CENSUS_SEEDS = [7, 42, 137]
SHADE = ns.COND_SHADE           # arm -> (ramp index, line style)
SCR = ns.SCRATCH_STYLE
GT = ns.GREY_TXT
WINDOW = "#F3F4F6"
LW = 1.5
SRC = {}
LABEL = {"sc_true": "superclasses", "sc_rand": "random grouping", "sc_blind": "superclass-blind task"}


def band(ax, x, Y, color, ls, lw=LW, marker=None, ms=2.6, alpha=0.2, z=3):
    Y = np.asarray(Y, float)
    m = np.nanmean(Y, 0)
    if Y.shape[0] > 1:                    # t-based 95% CI over seeds (Methods), per point with its own n
        n = np.sum(~np.isnan(Y), 0)
        se = np.nanstd(Y, 0, ddof=1) / np.sqrt(n)
        h = np.where(n > 1, stats.t.ppf(0.975, np.maximum(n - 1, 1)), np.nan) * se
        ax.fill_between(x, m - h, m + h, color=color, alpha=alpha, lw=0, zorder=z - 1)
    ax.plot(x, m, color=color, ls=ls, lw=lw, marker=marker, ms=ms, zorder=z)
    return m


def end_labels(ax, items, x, min_gap, **kw):
    """Direct labels at the right end of curves: items = [(y, text, colour)], pushed apart by min_gap (data units)."""
    items = sorted(items, key=lambda t: -t[0])
    placed = []
    for y, txt, c in items:
        if placed and placed[-1] - y < min_gap:
            y = placed[-1] - min_gap
        ax.text(x, y, txt, color=c, fontsize=6, ha="left", va="center", linespacing=0.95, clip_on=False, **kw)
        placed.append(y)


# a: pre-training builds the coarse structure
def panel_a(ax):
    d = pd.read_csv(RESULTS / "DICTFREE" / "hierarchy" / "D_sae_census_timecourse.csv")
    d = d[(d.src == "sae") & (d.epoch <= 20)]
    blind = B.census_rows()
    if blind is not None:
        d = pd.concat([d, blind[blind.epoch <= 20]], ignore_index=True)
    series = [("sc_true_K20", "census_genuine", ns.BLUE[0], "-", LW, "o", "superclasses"),
              ("sc_rand_K20", "census_genuine", ns.BLUE[1], "--", LW, "o", "random grouping"),
              ("sc_blind_K20", "census_genuine", ns.BLUE[2], ":", LW, "o", "superclass-blind task"),
              ("scratch_E30", "census_genuine", SCR["color"], "-", SCR["lw"], "o", "scratch\n(fine classes)"),
              ("sc_rand_K20", "census_imp_genuine", ns.BLUE[3], (0, (1, 1.2)), 1.0, None,
               "random grouping,\nits own groups")]
    rows, ends = [], []
    for lane, col, c, ls, lw, mk, lab in series:
        x = d[d.lane == lane]
        if not len(x):
            continue
        P = x.pivot(index="seed", columns="epoch", values=col) * 100
        # three seeds only: thin per-seed lines instead of a 95% band (t with 2 df makes the band uninformative)
        if col == "census_genuine":
            for _, r in P.iterrows():
                ax.plot(P.columns.values, r.values, color=c, ls="-", lw=0.5, alpha=0.45, zorder=2)
        m = band(ax, P.columns.values, P.values, c, ls, lw=lw, marker=mk, ms=2.4, alpha=0.0)
        tc = {ns.BLUE[0]: ns.BLUE_TXT[0], ns.BLUE[1]: ns.BLUE_TXT[1], ns.BLUE[2]: ns.BLUE_TXT[1],
              ns.BLUE[3]: ns.BLUE_TXT[1]}.get(c, GT)
        ends.append((m[-1], lab, tc))
        rows += [dict(lane=lane, measure=("true superclasses" if col == "census_genuine" else "own random groups"),
                      seed=int(sd), epoch=int(e), pct_of_alive=v) for sd, r in P.iterrows() for e, v in r.items()]
        print(f"[a] {lane} {col} epoch 20: mean {P[20].mean():.1f}, range {P[20].min():.1f}-{P[20].max():.1f} "
              f"(seeds {sorted(P.index.tolist())}); epoch 5 mean {P[5].mean():.1f}")
    end_labels(ax, ends, 20.8, 3.4)
    ax.set_xlim(-0.8, 20.5)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_ylim(-1, 36)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_xlabel("Pre-training epoch\n(scratch: fine-class epoch)", linespacing=1.0)
    ax.set_ylabel("Features selective for the\ntrue superclasses (% of alive)")
    ax.set_title("Pre-training builds the coarse structure,\neven from labels that do not describe it",
                 fontsize=7, pad=3)
    SRC["a"] = pd.DataFrame(rows)


# b: superclass events before the first fine-class label
def cumulative(ev, lane, col, off):
    x = ev[ev.arm == lane]
    P = x.pivot(index="seed", columns="epoch_b", values=col).sort_index().cumsum(axis=1) / 1e3
    ep = np.r_[-off, P.columns.values - off]
    Y = np.c_[np.zeros(len(P)), P.values]
    return P.index.tolist(), ep, Y


def events_panel(ax, proc, palette, txt):
    """Cumulative superclass (ab) or differentiation (di) events aligned at the first fine-class epoch; dots and label,
    share of the run's events recorded before the first fine-class label (range over seeds and pre-built arms).
    Returns (rows, share rows)."""
    col = f"{proc}_h"
    ev = L.events_common()
    blind = B.events_rows()
    arms = [("scratch", "scratch_E30"), ("sc_true", "sc_true_K20"), ("sc_rand", "sc_rand_K20")]
    if blind is not None:
        ev = pd.concat([ev, blind], ignore_index=True)
        arms.append(("sc_blind", B.LANE_NAME))
    ax.axvspan(-20.5, 0, color=WINDOW, lw=0, zorder=0)
    rows, share, ends = [], [], []
    for a, lane in arms:
        off = 0 if a == "scratch" else 20
        seeds, ep, Y = cumulative(ev, lane, col, off)
        if a == "scratch":
            c, ls, lw, tc = SCR["color"], "-", SCR["lw"], GT
        else:
            i, ls = SHADE[a]
            c, lw, tc = palette[i], LW, txt[0] if i == 0 else txt[1]
        m = band(ax, ep, Y, c, ls, lw=lw)
        ends.append((m[-1], "scratch" if a == "scratch" else LABEL[a], tc))
        rows += [dict(process={"ab": "Ab-E", "di": "Di-E"}[proc], arm=lane, seed=s, epoch_rel=int(e), cumulative_k=v)
                 for s, y in zip(seeds, Y) for e, v in zip(ep, y)]
        if a != "scratch":
            i0 = list(ep).index(0)
            sh = Y[:, i0] / Y[:, -1] * 100
            share += [dict(arm=lane, seed=s, pct_before_first_fine_label=v) for s, v in zip(seeds, sh)]
            ax.plot([0], [Y[:, i0].mean()], "o", ms=3.0, color=c, mec="white", mew=0.4, zorder=6)
    S = pd.DataFrame(share)
    pre = S[S.arm.isin(["sc_true_K20", "sc_rand_K20"])]
    lo, hi = pre.pct_before_first_fine_label.min(), pre.pct_before_first_fine_label.max()
    ax.text(-19.6, 0.95, f"{lo:.0f}–{hi:.0f}% before the first\nfine-class label", transform=ax.get_xaxis_transform(),
            fontsize=6, color=txt[0], ha="left", va="top", linespacing=1.0)
    end_labels(ax, ends, 31.6, 0.09 * ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 20)
    ax.set_xlim(-20.5, 31)
    ax.set_xticks([-20, -10, 0, 10, 20, 30])
    ax.set_xlabel("Epoch relative to the first fine-class epoch")
    for a in S.arm.unique():
        v = S[S.arm == a].pct_before_first_fine_label
        print(f"[{proc}] share before first fine label {a}: mean {v.mean():.1f}, range {v.min():.1f}-{v.max():.1f} "
              f"(per seed {np.round(v.values, 1).tolist()})")
    print(f"[{proc}] range over seeds and both groupings: {lo:.1f}-{hi:.1f}")
    return rows, S


def panel_b(ax):
    ax.set_ylim(0, 300)
    rows, S = events_panel(ax, "ab", ns.BLUE, ns.BLUE_TXT)
    ax.set_ylabel("Superclass events\n(×10³, cumulative)")
    ax.set_title("Much of the coarse structure forms\nbefore the first fine-class label", fontsize=7, pad=3)
    SRC["b"] = pd.DataFrame(rows)
    SRC["b_share_before_first_fine_label"] = S


# c: class features confined to their superclass
def panel_c(ax):
    D = pd.read_csv(RESULTS / "RDT_pace" / "curriculum_long.csv")
    blind = B.confined_rows()
    arms = [("scratch", "scratch", [0, 5, 10, 15]), ("sc_true", "true", [1, 5, 10, 15]),
            ("sc_rand", "rand", [1, 5, 10, 15])]
    if blind is not None:
        D = pd.concat([D, blind], ignore_index=True)
        arms.append(("sc_blind", "blind", [1, 5, 10, 15]))
    rows, ends, M = [], [], {}
    for a, key, eps in arms:
        x = D[D.arm == key]
        P = x.pivot(index="seed", columns="fine_epoch", values="n_cl_confined")[eps]
        if a == "scratch":
            c, ls, lw, tc = SCR["color"], "-", SCR["lw"], GT
        else:
            i, ls = SHADE[a]
            c, lw, tc = ns.AMBER[i], LW, ns.AMBER_TXT[0] if i == 0 else ns.AMBER_TXT[1]
        m = band(ax, eps, P.values, c, ls, lw=lw, marker="o", ms=2.6)
        M[a] = P
        ends.append((m[-1], "scratch" if a == "scratch" else LABEL[a], tc))
        rows += [dict(arm=key, seed=int(s), fine_epoch=int(e), confined_class_features=v)
                 for s, r in P.iterrows() for e, v in r.items()]
    end_labels(ax, ends, 15.6, 3.6)
    ax.set_xlim(-0.4, 15.5)
    ax.set_xticks([0, 1, 5, 10, 15])
    ax.set_ylim(0, 55)
    ax.set_xlabel("Fine-class epoch")
    ax.set_ylabel("Class features confined to\ntheir superclass (per checkpoint)")
    ax.set_title("Class features then form inside\nthe coarse structure, faster", fontsize=7, pad=3)
    SRC["c"] = pd.DataFrame(rows)
    for e in (5, 10, 15):
        d_ = {a: (M[a][e] - M["scratch"][e]) for a in M if a != "scratch"}
        print(f"[c] fine epoch {e}: " + ", ".join(f"{a} {M[a][e].mean():.1f} (minus scratch {v.mean():+.1f}, "
                                                   f"{(v > 0).sum()}/{len(v)} above)" for a, v in d_.items())
              + f"; scratch {M['scratch'][e].mean():.1f}")
    print(f"[c] fine epoch 1: " + ", ".join(f"{a} {M[a][1].mean():.1f} ({M[a][1].min():.0f}-{M[a][1].max():.0f})"
                                            for a in M if a != "scratch"))


# d: fine-class accuracy
def lead_vs_drawn(v1, scratch_curve):
    """Epoch at which the scratch run (running maximum, 1% at epoch 0) reaches v1, minus 1."""
    y = np.maximum.accumulate(np.r_[1.0, scratch_curve])
    return float(np.interp(v1, y, np.arange(len(y)))) - 1


def panel_d(ax):
    x = np.arange(1, 31)
    curves = {a: {s: L.fine_val(a, s) for s in SEEDS} for a in ("scratch", "sc_true", "sc_rand")}
    blind = B.fine_curves()
    if blind is not None:
        curves["sc_blind"] = blind
    rows, lead, handles = [], [], []
    for a, C in curves.items():
        Y = np.array([C[s] for s in SEEDS])
        if a == "scratch":
            c, ls, lw = SCR["color"], "-", SCR["lw"]
        else:
            i, ls = SHADE[a]
            c, lw = ns.AMBER[i], LW
        band(ax, x, Y, c, ls, lw=lw, z=4 if a == "sc_true" else 3)
        handles.append((Line2D([], [], color=c, ls=ls, lw=lw), "scratch" if a == "scratch" else LABEL[a]))
        rows += [dict(arm=a, seed=s, fine_epoch=int(e), val_acc=v) for s, y in zip(SEEDS, Y) for e, v in zip(x, y)]
        if a != "scratch":
            lead += [dict(arm=a, seed=s, acc_fine_epoch1=C[s][0],
                          lead_epochs=lead_vs_drawn(C[s][0], curves["scratch"][s])) for s in SEEDS]
    Ysc = np.array([curves["scratch"][s] for s in SEEDS]).mean(0)
    ax.plot(x[:12], Ysc[4:16], color=ns.GREYS[0], lw=0.9, ls=(0, (1, 1.5)), zorder=8)
    T = pd.DataFrame(lead)
    lt = {a: T[T.arm == a].lead_epochs for a in T.arm.unique()}
    for a, v in lt.items():
        print(f"[d] lead after one fine-class epoch, {a}: {v.mean():.2f} (range {v.min():.2f}-{v.max():.2f}, "
              f"per seed {np.round(v.values, 2).tolist()}); accuracy at fine epoch 1 "
              f"{T[T.arm == a].acc_fine_epoch1.mean():.1f}")
    fin = {a: np.array([curves[a][s][-1] for s in SEEDS]) for a in curves}
    print("[d] final (fine epoch 30): " + ", ".join(f"{a} {v.mean():.2f} ({v.min():.2f}-{v.max():.2f})"
                                                     for a, v in fin.items()))
    print(f"[d] fine epoch 1: scratch {np.mean([curves['scratch'][s][0] for s in SEEDS]):.1f}")
    # lead bracket at 30% accuracy: scratch crossing <-> pre-built (superclasses) at fine epoch 1
    y0 = 30.0
    xs = float(np.interp(y0, np.maximum.accumulate(Ysc), x))
    ax.annotate("", xy=(1.0, y0), xytext=(xs, y0),
                arrowprops=dict(arrowstyle="|-|", color=ns.AMBER_TXT[0], lw=0.9, mutation_scale=2.5, shrinkA=0,
                                shrinkB=0), zorder=9)
    v = [f"{lt[a].mean():.1f}" for a in lt]
    txt = "ahead of scratch\nafter one epoch:\n" + (", ".join(v) if len(v) > 2 else " and ".join(v)) + " epochs"
    ax.text(xs + 0.8, y0 - 0.8, txt, fontsize=6, color=ns.AMBER_TXT[0], ha="left", va="top", linespacing=1.0)
    ax.text(1.0, 61.5, "grey dotted: scratch,\n4 epochs earlier", fontsize=6, color=ns.GREYS[0], ha="left", va="top",
            linespacing=0.95)
    # the superclass-blind arm ends lower (Supplementary Note 4), so the range covers the coarse arms and scratch
    allfin = np.concatenate([v for a, v in fin.items() if a != "sc_blind"])
    ax.text(30, 40, f"coarse pre-training and\nscratch end at {allfin.min():.1f}–{allfin.max():.1f}%", fontsize=6, color=ns.AMBER_TXT[0], linespacing=0.95,
            ha="right", va="center")
    leg = ax.legend([h for h, _ in handles[1:]] + [handles[0][0]], [l for _, l in handles[1:]] + ["scratch"],
                    loc="lower right", fontsize=6, frameon=False, handlelength=2.2, labelspacing=0.25,
                    borderaxespad=0.2, title="pre-built on", title_fontsize=6, alignment="left")
    for t, (h, _) in zip(leg.get_texts(), handles[1:] + [handles[0]]):
        t.set_color(ns.AMBER_TXT[0] if h.get_color() != SCR["color"] else GT)
    leg.get_title().set_color(ns.AMBER_TXT[0])
    ax.set_xlim(0, 31)
    ax.set_xticks([1, 10, 20, 30])
    ax.set_ylim(5, 64)
    ax.set_xlabel("Fine-class epoch")
    ax.set_ylabel("Fine-class accuracy (%)")
    ax.set_title("Fine-class accuracy starts about four\nepochs ahead and ends at the same level", fontsize=7,
                 pad=3)
    SRC["d"] = pd.DataFrame(rows)
    SRC["d_lead_after_one_epoch"] = T


def main():
    ns.apply()
    print(B.status())
    for f in (ns.FINAL_OUT / "source_data").glob(f"{NAME}_*.csv"):
        f.unlink()
    fig = plt.figure(figsize=(W * ns.MM, H * ns.MM))
    mm = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
    top, bot, h = 74, 12, 35
    a = mm(16, top, 52, h)
    b = mm(110, top, 46, h)
    c = mm(16, bot, 52, h)
    d = mm(110, bot, 66, h)
    panel_a(a)
    panel_b(b)
    panel_c(c)
    panel_d(d)
    for ax, letter, x, y in ((a, "a", 0.5, H - 1), (b, "b", 92, H - 1), (c, "c", 0.5, bot + h + 14),
                             (d, "d", 92, bot + h + 14)):
        bb = ax.get_position()
        ns.panel(ax, letter, x=(x / W - bb.x0) / bb.width, y=(y / H - bb.y0) / bb.height)
    for k, df in SRC.items():
        ns.source_data(df, NAME, k)
    print("saved", ns.save_final(fig, NAME, 180, strict=True))


if __name__ == "__main__":
    main()
