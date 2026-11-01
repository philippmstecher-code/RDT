#!/usr/bin/env python3
"""Fig. 1: RDT pipeline (a), process composition (b) and construction/refinement ratio (c).

a  six-step pipeline schematic, one tile per step with a one-sentence description column (DESC). Step 1 spaces
   checkpoints by training accuracy; step 3 keys the match states by r band (stable r >= 0.5, transformed
   0.2 <= r < 0.5, born / died r < 0.2); step 5 takes the SSI and CSI thresholds from permutation nulls and the SAI
   threshold from the observed SAI values.
b  share of Tg-E / Ab-E / Di-E events per transition, mean over the 9 CIFAR-100 runs, on the transitions common to
   all runs (0->1 ... 7->8); duplicate final transition removed.
c  construction/refinement ratio (Tg-E + Ab-E)/Di-E on the same transitions: median and IQR over the 9 runs,
   architecture means (3 seeds), open circles = each run at transition 7. Di-E is floored at 1 event; ratios above
   1,000 are drawn at 1,000.
Data: <lane>/sae_analysis/sae_results.json['process_intensity'] via process_composition.load_all.
Usage: python scripts/paper/figures/fig1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nmi_style as ns  # noqa: E402
from core_runs_lib import MM, add_axes_mm, apply_style, f12, plain_log_ticks  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Patch, Polygon  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402

NAME = "fig1"
W, H = 180.0, 170.0
ARCHS = ("ResNet-18", "ViT-Small", "CCT-7")
CR_CEIL = 1e3

# panel a palette
TILE = "#F7F7FA"
LAV = "#EAE5F1"
PURPLE = "#7E57C2"
TITLE = "#2D2D2D"
CURVE = "#172C51"
NET = "#90A4AE"
SAE_GRAD = ["#D1C4E9", "#B39DDB", "#9575CD", "#7E57C2", "#673AB7"]  # top (pooled) -> bottom (block 1)
INACTIVE = "#D8D8DE"
ARROW = "#C8C8D0"
STABLE, BORN, DIED_CELL, DIED = "#5B9BD5", "#70AD47", "#ED7D31", "#D55E00"
TRANSF = "#9DC3E6"  # transformed (0.2 <= r < 0.5): light tint of the stable blue (partial continuity)
TRANSF_TXT = "#1F3F66"  # dark label on the light chip (as the light SAE boxes of step 2 carry dark text)
CARD_ON, CARD_OFF, CARD_BOX = "#009E73", "#A5A5A5", "#BFBFBF"
NULL_FILL, NULL_LINE, OBS = "#D4D4D6", "#565657", "#0072B2"
CONNECT = "#A5A5A5"

DESC = {   # one short sentence per step; details in the legend and Methods
    1: "Checkpoints are spaced by training accuracy: dense early, sparse late.",
    2: "One sparse autoencoder per layer and checkpoint decomposes the activations into features.",
    3: "Features are matched one-to-one between checkpoints by their activation profiles.",
    4: "Three indices measure how broadly each feature responds.",
    5: "Each index is tested against a permutation null.",
    6: "Lifecycle state and selectivity label each feature event.",
}
TC = 25.0     # width of the lavender description column (mm)
FS_D = 6.0    # description text

FS = 6.0      # diagram labels
FS_T = 7.0    # step titles


# panel a primitives (mm coordinates)
def rbox(ax, x, y, w, h, fc, ec="none", lw=0.5, ls="-", r=0.6, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                ls=ls, zorder=z))


def txt(ax, x, y, s, size=FS, color=TITLE, ha="center", va="center", weight="normal", style="normal", z=5, **kw):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, fontstyle=style, zorder=z, **kw)


def arrow(ax, x0, y0, x1, y1, color=ARROW, lw=0.8, ms=5, z=3):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=z,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=ms, shrinkA=0, shrinkB=0))


def wrap_mm(s, width_mm, size):
    """Greedy word wrap to a width in mm, measured with the Arial glyph outlines."""
    fp = FontProperties(family="Arial", size=size)
    wmm = lambda t: TextPath((0, 0), t, prop=fp).get_extents().width * 25.4 / 72  # noqa: E731
    lines, cur = [], ""
    for w in s.split():
        cand = (cur + " " + w).strip()
        if cur and wmm(cand) > width_mm:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    lines.append(cur)
    return lines


def tile(ax, x0, y0, tw, th, num, title, desc):
    rbox(ax, x0, y0, tw, th, TILE, r=0.0, z=0)
    cx0 = x0 + tw - TC
    rbox(ax, cx0, y0, TC, th, LAV, r=0.0, z=1)  # lavender description column
    ax.add_patch(Circle((x0 + 3.6, y0 + th - 4.0), 2.4, fc=PURPLE, ec="none", zorder=3))
    txt(ax, x0 + 3.6, y0 + th - 4.0, str(num), size=7, color="white", weight="bold")
    txt(ax, x0 + 7.2, y0 + th - 4.0, title, size=FS_T, ha="left", weight="bold")
    lines = wrap_mm(desc, TC - 3.6, FS_D)
    txt(ax, cx0 + 1.8, y0 + th / 2, "\n".join(lines), size=FS_D, ha="left", va="center", color=TITLE,
        linespacing=1.2)
    return x0 + 2.5, y0 + 1.8, tw - TC - 5.0, th - 8.6  # diagram box (x, y, w, h)


def step1(ax, bx, by, bw, bh):
    """Accuracy curve with checkpoints at equal steps of training accuracy (Methods)."""
    x0, y0, x1, y1 = bx + 4.5, by + 3.4, bx + bw - 16, by + bh - 0.5
    arrow(ax, x0, y0, x1 + 2.5, y0, color="black", lw=0.6, ms=4)
    arrow(ax, x0, y0, x0, y1, color="black", lw=0.6, ms=4)
    txt(ax, (x0 + x1) / 2, by + 1.2, "Training epochs", size=FS, color="black")
    txt(ax, x0 - 2.2, (y0 + y1) / 2, "Training accuracy", size=FS, color="black", rotation=90)
    tau, amax = 0.2, 0.92
    e = np.linspace(0, 1, 200)
    acc = amax * (1 - np.exp(-e / tau))
    X = lambda u: x0 + 1.0 + u * (x1 - x0 - 1.5)  # noqa: E731
    Y = lambda a: y0 + 1.0 + a * (y1 - y0 - 2.5)  # noqa: E731
    ax.plot(X(e), Y(acc), color=CURVE, lw=0.9, zorder=3)
    levels = np.linspace(0, 0.89, 10)  # equal steps in accuracy -> dense early, sparse late
    ec = -tau * np.log(1 - levels / amax)
    ax.scatter(X(ec), Y(levels), s=7, color="black", zorder=4, lw=0)
    # direct labels
    txt(ax, X(ec[3]) + 1.6, Y(levels[3]) - 0.6, "dense early", size=FS, color="black", ha="left", va="top")
    txt(ax, X(ec[-1]), Y(levels[-1]) - 1.6, "sparse late", size=FS, color="black", ha="center", va="top")
    # key (right)
    kx = x1 + 3.0
    ax.scatter([kx], [y1 - 3.2], s=7, color="black", lw=0, zorder=4)
    txt(ax, kx + 1.6, y1 - 3.2, "checkpoint", size=FS, color="black", ha="left")


def step2(ax, bx, by, bw, bh):
    """ResNet-18 blocks -> one SAE per layer -> sparse feature dictionary."""
    names = ["pooled", "block 4", "block 3", "block 2", "block 1"]
    top = by + bh - 3.4
    bhgt, gap = 2.85, 0.55
    nx, nw = bx, 11.0
    sx, sw = bx + 15.0, 11.0
    txt(ax, nx + nw / 2, by + bh - 1.0, "ResNet-18", size=FS, weight="bold")
    txt(ax, sx + sw / 2, by + bh - 1.0, "SAE per layer", size=FS, weight="bold")
    rbox(ax, nx - 0.6, top - 5 * bhgt - 4 * gap - 0.6, nw + 1.2, 5 * bhgt + 4 * gap + 1.2, "white", ec="#D9D9DF",
         lw=0.5, r=0.5, z=1)
    rbox(ax, sx - 0.6, top - 5 * bhgt - 4 * gap - 0.6, sw + 1.2, 5 * bhgt + 4 * gap + 1.2, "white", ec="#D1C4E9",
         lw=0.5, r=0.5, z=1)
    for i, nm in enumerate(names):
        y = top - (i + 1) * bhgt - i * gap
        rbox(ax, nx, y, nw, bhgt, NET, r=0.5)
        txt(ax, nx + nw / 2, y + bhgt / 2, nm, color="white")
        rbox(ax, sx, y, sw, bhgt, SAE_GRAD[i], r=0.5)
        txt(ax, sx + sw / 2, y + bhgt / 2, "SAE", color="white" if i >= 2 else "#4C3A78")
        arrow(ax, nx + nw + 0.7, y + bhgt / 2, sx - 0.9, y + bhgt / 2, color=ARROW, lw=0.7, ms=4)
    # dictionary of the block-2 SAE
    dx = sx + sw + 5.5
    yb2 = top - 4 * bhgt - 3 * gap + bhgt / 2
    arrow(ax, sx + sw + 0.9, yb2, dx - 1.0, yb2, color=PURPLE, lw=1.0, ms=5)
    cw, ch, cg = 3.4, 3.4, 0.7
    txt(ax, dx, by + bh - 1.0, "Dictionary (block 2)", size=FS, weight="bold", ha="left")
    txt(ax, dx, by + bh - 4.2, "X \u2248 D \u00b7 Z  (Z sparse)", size=FS, ha="left")
    pat = [[1, 0, 1, 0, 0], [0, 1, 0, 0, 1]]
    gy = yb2 + 0.2
    for r, row in enumerate(pat):
        for c, on in enumerate(row):
            rbox(ax, dx + c * (cw + cg), gy - r * (ch + cg), cw, ch, PURPLE if on else INACTIVE, r=0.4)
    ky = gy - (ch + cg) - 2.4
    rbox(ax, dx, ky - 0.9, 1.8, 1.8, PURPLE, r=0.2)
    txt(ax, dx + 2.4, ky, "active", ha="left")
    rbox(ax, dx + 9.4, ky - 0.9, 1.8, 1.8, INACTIVE, r=0.2)
    txt(ax, dx + 11.8, ky, "inactive", ha="left")


def step3(ax, bx, by, bw, bh):
    """Matching between consecutive checkpoints; lifecycle state of each feature."""
    cw, ch, cg = 3.2, 3.4, 0.7
    gw = 5 * cw + 4 * cg
    lx, rx = bx + 0.5, bx + bw - gw - 0.5
    gy = by + bh - 8.4
    txt(ax, lx + gw / 2, by + bh - 1.0, "Checkpoint t", weight="bold")
    txt(ax, rx + gw / 2, by + bh - 1.0, "Checkpoint t + 1", weight="bold")
    left = [[1, 0, 1, 0, 0], [0, 1, 0, 0, 1]]
    right = [[STABLE, INACTIVE, DIED_CELL, BORN, INACTIVE], [INACTIVE, STABLE, INACTIVE, INACTIVE, TRANSF]]
    for r in range(2):
        for c in range(5):
            y = gy - r * (ch + cg)
            rbox(ax, lx + c * (cw + cg), y, cw, ch, PURPLE if left[r][c] else INACTIVE, r=0.4)
            rbox(ax, rx + c * (cw + cg), y, cw, ch, right[r][c], r=0.4)
    mx0, mx1 = lx + gw + 2.0, rx - 2.0
    my = gy - cg / 2 + 0.6
    arrow(ax, mx0, my, mx1, my, color="black", lw=0.9, ms=5)
    txt(ax, (mx0 + mx1) / 2, my + 2.3, "Pearson r", weight="bold")
    txt(ax, (mx0 + mx1) / 2, my - 2.4, "Hungarian\nmatching", weight="bold", va="top", linespacing=1.1)
    # lifecycle key with the Methods' r bands underneath
    ky = by + 4.2
    chips = (("Stable", STABLE, 8.4), ("Transformed", TRANSF, 14.6), ("Born", BORN, 7.0), ("Died", DIED, 7.0))
    gap = 1.0
    x = bx + bw - sum(c[2] for c in chips) - gap * (len(chips) - 1)
    xc = {}
    for lab, col, w in chips:
        rbox(ax, x, ky - 1.3, w, 2.6, col, r=0.4)
        txt(ax, x + w / 2, ky, lab, color=TRANSF_TXT if col == TRANSF else "white", weight="bold", style="italic")
        xc[lab] = (x, x + w)
        x += w + gap
    ry = by + 1.4
    txt(ax, sum(xc["Stable"]) / 2, ry, "r \u2265 0.5", color="black")
    txt(ax, sum(xc["Transformed"]) / 2, ry, "0.2 \u2264 r < 0.5", color="black")
    txt(ax, (xc["Born"][0] + xc["Died"][1]) / 2, ry, "r < 0.2", color="black")


def step4(ax, bx, by, bw, bh):
    """Three selectivity indices: which superclass / class cards a feature responds to."""
    rows = (("SAI", "task-general", [1, 1, 1], [1, 1, 1]),
            ("SSI", "superclass", [1, 0, 0], [1, 1, 1]),
            ("CSI", "class", [0, 0, 0], [1, 0, 0]))
    lw_ = 13.0
    cw, ch, cg = 8.3, 3.6, 0.5
    sx = bx + lw_
    cx = sx + 3 * cw + 2 * cg + 1.8
    top = by + bh - 4.6
    txt(ax, sx + (3 * cw + 2 * cg) / 2, by + bh - 2.0, "Superclasses", weight="bold")
    txt(ax, cx + (3 * cw + 2 * cg) / 2, by + bh - 2.0, "Classes of ‘flowers’", weight="bold")
    sups, clss = ("flowers", "fish", "vehicles"), ("rose", "orchid", "tulip")
    for i, (ab, nm, s_on, c_on) in enumerate(rows):
        y = top - (i + 1) * ch - i * 1.6
        txt(ax, bx, y + ch / 2 + 0.85, ab, ha="left", weight="bold", style="italic")
        txt(ax, bx, y + ch / 2 - 1.15, nm, ha="left", color="#4B5563")
        for j in range(3):
            rbox(ax, sx + j * (cw + cg), y, cw, ch, CARD_ON if s_on[j] else CARD_OFF, r=0.5)
            txt(ax, sx + j * (cw + cg) + cw / 2, y + ch / 2, sups[j], color="white")
            rbox(ax, cx + j * (cw + cg), y, cw, ch, CARD_ON if c_on[j] else CARD_OFF, r=0.5)
            txt(ax, cx + j * (cw + cg) + cw / 2, y + ch / 2, clss[j], color="white")
        rbox(ax, cx - 0.7, y - 0.45, 3 * cw + 2 * cg + 1.4, ch + 0.9, "none", ec=CARD_BOX, lw=0.5, ls=(0, (2, 1.5)),
             r=0.3, z=3)


def step5(ax, bx, by, bw, bh):
    """Reference distribution (permutation null for SSI and CSI; observed values for SAI), 95th-percentile
    threshold, observed index (Methods, 'Null baseline and adaptive thresholds')."""
    x0, x1, y0 = bx + 3, bx + bw - 14, by + 4.2
    u = np.linspace(-3.2, 3.2, 300)
    g = np.exp(-u ** 2 / 2)
    X = x0 + (u + 3.2) / 6.4 * (x1 - x0)
    Y = y0 + g * (bh - 9.0)
    ax.add_patch(Polygon(np.c_[np.r_[X, X[::-1]], np.r_[Y, np.full_like(Y, y0)]], closed=True, fc=NULL_FILL,
                         ec="none", zorder=2))
    ax.plot(X, Y, color=NULL_LINE, lw=1.0, zorder=3)
    txt(ax, x0 + (x1 - x0) / 2, y0 + 3.4, "null (SSI, CSI) or\nall features (SAI)", color="#4B5563",
        linespacing=1.1)
    xt = x0 + (1.645 + 3.2) / 6.4 * (x1 - x0)
    xo = x0 + (2.45 + 3.2) / 6.4 * (x1 - x0)
    ax.plot([xt, xt], [y0 - 0.6, y0 + bh - 7.4], color=DIED, lw=0.9, ls=(0, (2.5, 1.6)), zorder=4)
    txt(ax, xt, y0 + bh - 6.4, "95th percentile = ATS", color=DIED, ha="right", va="bottom")
    ax.plot([xo, xo], [y0, y0 + bh - 9.8], color=OBS, lw=1.1, zorder=4)
    txt(ax, xo + 1.0, y0 + bh - 10.8, "observed", color=OBS, ha="left", va="center")
    txt(ax, x0 + (x1 - x0) / 2, by + 1.3, "Selectivity index (pooled over checkpoints and layers)", color="black")


def step6(ax, bx, by, bw, bh):
    """Lifecycle state x selectivity -> five feature events."""
    xs = np.linspace(bx + 5.5, bx + bw - 5.5, 5)
    cols = dict(zip(("As-E", "Tg-E", "Ab-E", "Di-E", "De-E"), xs))
    ct = by + bh - 1.8
    chips = (("Born", BORN, cols["As-E"], 12.0), ("Stable", STABLE, cols["Ab-E"], 12.0),
             ("Died", DIED, cols["De-E"], 11.0))
    for lab, col, x, w in chips:
        rbox(ax, x - w / 2, ct - 2.2, w, 3.4, col, r=0.5)
        txt(ax, x, ct - 0.5, lab, color="white", weight="bold", style="italic", size=6.5)
    yb = ct - 5.4
    lwc = 0.7
    ya = yb - 2.6
    # Born feeds two routes: every born feature is an assembly event (own arrow to As-E), and
    # born and stable features enter the three selectivity tests (a second Born stem joins the bus to Tg-E, Ab-E, Di-E)
    xb2 = cols["As-E"] + 4.2
    arrow(ax, cols["As-E"], ct - 2.2, cols["As-E"], ya, color=CONNECT, lw=lwc, ms=4)
    ax.plot([xb2, xb2], [ct - 2.2, yb], color=CONNECT, lw=lwc, zorder=2)
    ax.plot([cols["Ab-E"], cols["Ab-E"]], [ct - 2.2, yb], color=CONNECT, lw=lwc, zorder=2)
    ax.plot([xb2, cols["Di-E"]], [yb, yb], color=CONNECT, lw=lwc, zorder=2)
    for k in ("Tg-E", "Ab-E", "Di-E"):
        arrow(ax, cols[k], yb, cols[k], ya, color=CONNECT, lw=lwc, ms=4)
    arrow(ax, cols["De-E"], ct - 2.2, cols["De-E"], ya, color=CONNECT, lw=lwc, ms=4)
    for k, cond in (("Tg-E", "SAI > ATS"), ("Ab-E", "SSI > ATS"), ("Di-E", "CSI > ATS")):
        txt(ax, cols[k], ya - 2.4, cond, style="italic")
    for k in cols:
        txt(ax, cols[k], by + 1.4, k, size=7, weight="bold", color="black")
    for xd in (xs[:-1] + xs[1:]) / 2:
        ax.plot([xd, xd], [by + 0.2, ya - 0.6], color=CARD_BOX, lw=0.5, zorder=1)


def draw_a(fig, top):
    twl, twr, th, gx, gy = 82.0, 95.5, 32.5, 2.5, 2.0   # left steps compress more than right ones
    ha = 3 * th + 2 * gy
    ax = add_axes_mm(fig, 0, top - ha, W, ha)
    ax.set_xlim(0, W)
    ax.set_ylim(0, ha)
    ax.set_axis_off()
    steps = ((1, "Training and checkpoint capture", step1), (2, "SAE decomposition into sparse features", step2),
             (3, "Feature matching and lifecycle", step3), (4, "Selectivity measurement", step4),
             (5, "Adaptive threshold (ATS) calibration", step5), (6, "Process classification", step6))
    for k, (num, title, fn) in enumerate(steps):
        col, row = k // 3, k % 3
        x0, tw = (0.0, twl) if col == 0 else (twl + gx, twr)
        y0 = ha - (row + 1) * th - row * gy
        fn(ax, *tile(ax, x0, y0, tw, th, num, title, DESC[num]))
    return ax


# panels b, c
def draw_b(ax, runs, n_tr):
    x = np.arange(n_tr)
    shares = []
    for r in runs:
        s = np.array([[t["tg_h"], t["ab_h"], t["di_h"]] for t in r["pi"][:n_tr]], float)
        shares.append(100 * s / s.sum(1, keepdims=True))
    m = np.mean(shares, 0)
    xmax = n_tr - 0.5
    for x0, x1, c, lab in ((-0.5, 2, ns.PROC["Tg-E"], "Phase I\nTask-general"),
                           (2, 5, ns.PROC["Ab-E"], "Phase II\nAbstraction"),
                           (5, xmax, ns.PROC["Di-E"], "Phase III\nDifferentiation")):
        ax.axvspan(x0, x1, color=c, alpha=0.06, lw=0, zorder=0)
        ax.text((x0 + x1) / 2, 110, lab, ha="center", va="center", fontsize=6.5, color="#111111", linespacing=1.05)
    for xb, c in ((2, ns.PROC["Tg-E"]), (5, ns.PROC["Ab-E"])):
        ax.axvline(xb, color=c, alpha=0.12, lw=0.8, zorder=0)
    c1, c2 = m[:, 0], m[:, 0] + m[:, 1]
    for lo, hi, k in ((np.zeros(n_tr), c1, "Tg-E"), (c1, c2, "Ab-E"), (c2, np.full(n_tr, 100.0), "Di-E")):
        ax.fill_between(x, lo, hi, facecolor=ns.PROC[k], edgecolor=ns.PROC[k], alpha=ns.FILL_ALPHA, lw=0.8, zorder=2)
    ax.plot(x, c1, color="white", lw=0.7, zorder=3)
    ax.plot(x, c2, color="white", lw=0.7, zorder=3)
    ax.set_xlim(-0.5, xmax)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xticks(np.arange(0, n_tr, 1))
    ax.set_xlabel("Transition")
    ax.set_ylabel("Process composition (%)")
    # direct in-band labels instead of a legend box
    for k, xi, y in (("Tg-E", 1.5, None), ("Ab-E", 3.5, None), ("Di-E", 6.4, None)):
        i0, i1 = int(np.floor(xi)), int(np.ceil(xi))
        f = xi - i0
        lo = {"Tg-E": 0 * c1, "Ab-E": c1, "Di-E": c2}[k]
        hi = {"Tg-E": c1, "Ab-E": c2, "Di-E": np.full(n_tr, 100.0)}[k]
        yy = ((1 - f) * (lo[i0] + hi[i0]) + f * (lo[i1] + hi[i1])) / 2
        ax.text(xi, yy, k, ha="center", va="center", fontsize=7, fontweight="bold", color="white", zorder=5)
    # peak markers: small downward triangle in the process colour on the top edge, at the transition where the
    # mean share of that process is largest (from m, the data in Source Data fig1_b)
    for j, k in enumerate(("Tg-E", "Ab-E", "Di-E")):
        ax.plot(int(np.argmax(m[:, j])), 102.2, marker="v", ms=3.6, color=ns.PROC[k], mec="white", mew=0.3,
                ls="none", clip_on=False, zorder=6)
    return m


def draw_c(ax, runs, n_common):
    cr = f12.cr
    C = np.array([[min(cr(t), CR_CEIL) for t in r["pi"][:n_common]] for r in runs])
    xc = np.arange(n_common)
    med = np.median(C, 0)
    q1, q3 = np.percentile(C, 25, 0), np.percentile(C, 75, 0)
    ax.fill_between(xc, q1, q3, facecolor=ns.NEUTRAL, alpha=0.12, edgecolor=ns.NEUTRAL, lw=1.0, zorder=1)
    ax.plot(xc, med, color=ns.REF_DARK, lw=1.2, zorder=4)
    am = {}
    for a in ARCHS:
        rr = [r for r in runs if r["arch"] == a]
        am[a] = np.array([[min(cr(t), CR_CEIL) for t in r["pi"][:n_common]] for r in rr]).mean(0)
        ax.plot(xc, am[a], color=ns.ARCH[a], ls=ns.ARCH_LS[a], lw=1.0, alpha=0.85, zorder=3)
    last = []
    off = {"ResNet-18": -0.12, "ViT-Small": 0.0, "CCT-7": 0.12}
    k = n_common - 1  # transition 7 (7->8), the last transition all runs share
    for i, r in enumerate(runs):
        v = cr(r["pi"][k])
        seed_off = ((r["seed"] == 137) * 0.06 - (r["seed"] == 256) * 0.06)
        ax.scatter(k + off[r["arch"]] + seed_off, v, s=11, facecolor="white", edgecolor=ns.ARCH[r["arch"]], lw=0.8,
                   zorder=6)
        last.append((r["arch"], r["seed"], r["pi"][k]["transition"], k, v))
    ax.axhline(1, color="#999999", ls="--", lw=0.5, zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(0.03, 1e4)
    plain_log_ticks(ax, [0.1, 1, 10, 100, 1000])
    ax.set_xlim(-0.5, n_common - 0.5)
    ax.set_xticks(np.arange(0, n_common, 1))
    ax.set_xlabel("Transition")
    ax.set_ylabel("Construction/refinement ratio")
    n = len(runs)
    # direct labels (architecture means), ratio = 1 reference, and the key effect in large type
    for a, (xt, yt, ha, va) in {"ResNet-18": (2.9, 1.9, "center", "top"), "ViT-Small": (1.3, 260, "left", "bottom"),
                                "CCT-7": (2.2, 80, "left", "bottom")}.items():
        ax.text(xt, yt, a, color=ns.ARCH[a], fontsize=6.5, ha=ha, va=va, zorder=7)
    ax.text(-0.4, 1.12, "ratio = 1", color="#6B7280", fontsize=6, ha="left", va="bottom")
    n_below = sum(v < 1 for *_, v in last)
    ax.text(4.6, 0.115, f"{n_below} of {n} runs below 1 at transition {k}", fontsize=7, fontweight="bold", color=ns.REF_DARK,
            ha="center", va="center")
    h = [Line2D([], [], color=ns.REF_DARK, lw=1.2, label=f"Median (n = {n})"),
         Patch(facecolor=ns.NEUTRAL, alpha=0.2, edgecolor=ns.NEUTRAL, lw=0.8, label="IQR"),
         Line2D([], [], ls="none", marker="o", ms=3.4, mfc="white", mec="#6B7280", mew=0.8,
                label=f"Each run, transition {k}")]
    ax.legend(handles=h, loc="upper right", bbox_to_anchor=(1.0, 1.02), handlelength=2.0, labelspacing=0.3,
              fontsize=6.5, borderaxespad=0.2)
    return C, med, q1, q3, am, last


def main():
    apply_style()
    runs, _ = f12.load_all()
    n_common = min(len(r["pi"]) for r in runs)
    assert n_common == 8
    fig = plt.figure(figsize=(W * MM, H * MM))

    top_a = H - 4.5
    ax_a = draw_a(fig, top_a)
    ha = ax_a.get_ylim()[1]
    ns.panel(ax_a, "a", x=0.0, y=1 + 4.0 / ha)

    yb, hb = 11.0, 46.0
    ax_b = add_axes_mm(fig, 14.0, yb, 70.0, hb)
    ax_c = add_axes_mm(fig, 108.0, yb, 70.0, hb)
    m = draw_b(ax_b, runs, n_common)
    C, med, q1, q3, am, last = draw_c(ax_c, runs, n_common)
    ns.panel(ax_b, "b", x=-0.2, y=1.08)
    ns.panel(ax_c, "c", x=-0.2, y=1.08)
    ns.rdt_tag(ax_b, y=0.03, va="bottom")
    ns.rdt_tag(ax_c, x=0.03, y=0.03, ha="left", va="bottom")

    # Source data
    trs = [t["transition"] for t in runs[0]["pi"][:n_common]]
    ns.source_data(pd.DataFrame({"transition_index": range(n_common), "transition": trs,
                                 "Tg-E_share_pct": m[:, 0], "Ab-E_share_pct": m[:, 1], "Di-E_share_pct": m[:, 2]}),
                   NAME, "b")
    rows = []
    for r in runs:
        for i, t in enumerate(r["pi"][:n_common]):
            rows.append({"arch": r["arch"], "seed": r["seed"], "transition_index": i, "transition": t["transition"],
                         "tg_h": t["tg_h"], "ab_h": t["ab_h"], "di_h": t["di_h"], "cr_ratio": f12.cr(t),
                         "cr_ratio_drawn_cap1000": min(f12.cr(t), CR_CEIL)})
    df = pd.DataFrame(rows)
    summ = []
    for i in range(n_common):
        summ.append({"arch": "all (n = 9)", "seed": "", "transition_index": i, "median": med[i], "iqr_low": q1[i],
                     "iqr_high": q3[i], "n_runs_below_1": int((C[:, i] < 1).sum())})
    for a in ARCHS:
        for i, v in enumerate(am[a]):
            summ.append({"arch": f"{a} mean (3 seeds)", "seed": "", "transition_index": i, "mean_drawn": v})
    df_c = pd.concat([df, pd.DataFrame(summ)], ignore_index=True)
    ns.source_data(df_c, NAME, "c")
    n_below = sum(v < 1 for *_, v in last)
    print(f"Fig. 1c: runs below 1 at transition 7: {n_below}/9")
    for a in ARCHS:
        print(f"  {a} mean at transition 7: {am[a][-1]:.3f}")
    for row in last:
        print("   ", row)
    out = ns.save_final(fig, NAME, 180)
    print("saved", out)


if __name__ == "__main__":
    main()
