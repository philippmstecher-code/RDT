#!/usr/bin/env python3
"""Supplementary Fig. 3: label-noise analyses with labels redrawn every epoch, and the fixed-label experiment.

a  superclass probe over epochs: redrawn labels at p = 0.1 and p = 0.3, fixed labels at p = 0.3
b  fine-class accuracy, same layout
c  share of Ab-E and Di-E events per condition (noise_lib.process_fractions)
d  alive SAE features per layer at epoch 30 (final/source_data_ed/ed6_e_per_lane_layer.csv)
e  fixed labels: fine-class accuracy against superclass probe (STATIC_probe/static_probe.csv), and the accuracy offset
   of noise-across networks at equal probe
Colour = level measured (blue, superclass probe and Ab-E; amber, fine-class accuracy and Di-E; e in ambers as both
levels are on its axes); condition by shade and line style (darkest solid, noise across superclasses; mid dashed,
noise within superclasses; light dotted, random noise; thin grey, clean/standard training); open markers (a, b) and
open bars (c, d), p = 0.1.
Usage: python scripts/paper/figures/extended/supp3.py  (ES_FIG_OUT=<dir> redirects the output)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nmi_style as ns  # noqa: E402
import noise_lib as L  # noqa: E402
from fig_common import RESULTS  # noqa: E402

NAME = "supp3"
W, H = 180, 168
SC, CL = ns.INDEX["SSI"], ns.INDEX["CSI"]
CLEAN_C = ns.GREYS[2]       # reference: thin grey solid
NEUTRAL = "#4B5563"         # d: alive SAE features (no level)
LS = {"standard": "-", "between": "-", "within": "--", "random": ":"}
SHADE = {"between": 0, "within": 1, "random": 2}       # ramp index per noise kind
LW = {"standard": 0.8, "between": 1.3, "within": 1.1, "random": 1.2}
ALPHA = ns.BAR_ALPHA_CORE
ZERO = L.ZERO
AS = 6.5
FINAL_SD = RESULTS / "figures" / "final" / "source_data_ed"


def tint(c, f=0.5):
    r = np.array(to_rgb(c))
    return tuple(r + (1 - r) * f)


def mm_axes(fig, x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def title(ax, txt, x=0.0, pad_mm=2.0):
    """Claim title, left-aligned with the axes, pad_mm above its top."""
    y = 1.0 + pad_mm / (ax.get_position().height * H)
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=7, ha="left", va="bottom", linespacing=1.05)


# a, b: probe and accuracy by dose
def dose_panels(axes, measure, ramp, ylab, ylim, yt, sd_panel, xlabels=True, headers=False):
    R = L.redrawn_metric(measure)
    if measure == "probe_sc_avgpool":
        fixed = L.static_probe()
    else:
        A = L.static_accuracy()
        fixed = {a: pd.DataFrame(A[a], index=L.SEEDS, columns=range(31)) for a in A}
    cols = (("labels redrawn every epoch", "p = 0.1"), ("labels redrawn every epoch", "p = 0.3"),
            ("labels kept fixed", "p = 0.3"))
    rows = []
    for k, ax in enumerate(axes):
        if k < 2:
            suf = "p01" if k == 0 else "p03"
            series = [("standard", R["standard"])] + [(n, R[f"{n}_sc_{suf}" if n != "random" else f"random_{suf}"])
                                                       for n in ("within", "between", "random")]
        else:
            series = [("standard", fixed["clean"]), ("within", fixed["within_sc"]), ("between", fixed["between_sc"])]
        open_ = k == 0                                  # p = 0.1: open markers
        for n, df in series:
            x = df.columns.values.astype(float)
            c = CLEAN_C if n == "standard" else ramp[SHADE[n]]
            m, h = L.traj(ax, x, df.values, c, LS[n], band="ci", lw=LW[n], ms=2.2 if open_ else 1.8,
                          markevery=5 if len(x) > 10 else None, zorder={"random": 5, "between": 4}.get(n, 3),
                          band_alpha=0.12, mfc="white" if (open_ and n != "standard") else None)
            for e, mm_, hh in zip(x, m, h):
                rows.append(dict(design=cols[k][0], noise_rate=cols[k][1],
                                 series="clean" if (k == 2 and n == "standard") else n,
                                 epoch=int(e), mean=mm_, ci_lo=mm_ - hh, ci_hi=mm_ + hh, n=df.shape[0]))
            if x[-1] == 30:
                print(f"[{sd_panel}] {cols[k][0]}, {cols[k][1]}, {n}: epoch 30 {m[-1]:.2f}")
        if headers:
            ax.text(0.5, 1.03, f"{cols[k][0].capitalize()}, {cols[k][1]}", transform=ax.transAxes, fontsize=6.5,
                    ha="center", va="bottom", color=L.ERR)
        ax.set_xlim(-1, 31)
        ax.set_xticks([0, 10, 20, 30])
        ax.set_ylim(*ylim)
        ax.set_yticks(yt)
        if xlabels:
            ax.set_xlabel("Epoch", labelpad=1.5)
        else:
            ax.set_xticklabels([])
        if k == 0:
            ax.set_ylabel(ylab)
        else:
            ax.set_yticklabels([])
    ns.source_data(pd.DataFrame(rows), NAME, sd_panel)


# c, d: condition blocks
POS = {"standard": 0.0, "within_sc_p01": 1.25, "within_sc_p03": 2.05, "between_sc_p01": 3.3, "between_sc_p03": 4.1,
       "random_p01": 5.35, "random_p03": 6.15}
BLOCKS = [("Within", 0.75, 2.55), ("Across", 2.8, 4.6), ("Random", 4.85, 6.65)]


def cond_key(c):
    if c == "standard":
        return "standard", None
    return c.split("_")[0], (0.1 if c.endswith("p01") else 0.3)


def block_axis(ax, ytop):
    for name, a, b in BLOCKS:
        ax.text((a + b) / 2, ytop, name, ha="center", va="top", fontsize=6, color=L.ERR)
    for b in (0.65, 2.675, 4.725):
        ax.axvline(b, color="#E5E7EB", lw=0.5, zorder=0)
    ax.set_xticks(list(POS.values()))
    ax.set_xticklabels(["Standard"] + ["0.1", "0.3"] * 3, fontsize=6.5)
    ax.set_xlabel("Noise rate p", labelpad=1.5)
    ax.set_xlim(-0.6, 6.7)


def panel_c(ax):
    F = L.process_fractions()
    Fk = F[~F.excluded]
    w = 0.34
    rows = []
    for c, x in POS.items():
        kind, p = cond_key(c)
        for j, (col, ramp) in enumerate((("ab_frac", ns.BLUE), ("di_frac", ns.AMBER))):
            v = Fk[Fk.condition == c][col].values
            m, lo, hi = L.t_ci(v)
            xb = x + (j - 0.5) * (w + 0.02)
            hue = CLEAN_C if kind == "standard" else ramp[SHADE[kind]]
            if p == 0.1:                                 # p = 0.1: open bars
                ax.bar(xb, m, width=w - 0.03, facecolor="white", edgecolor=hue, lw=0.8, zorder=2)
            else:
                ax.bar(xb, m, width=w, color=hue, alpha=ALPHA, lw=0, zorder=2)
            ax.errorbar(xb, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=L.ERR, elinewidth=0.7, capsize=1.6,
                        capthick=0.7, zorder=3)
            rows.append(dict(condition=c, process="Di-E" if col == "di_frac" else "Ab-E", mean=m, ci_lo=lo, ci_hi=hi,
                             n=len(v), seeds=";".join(str(s) for s in Fk[Fk.condition == c].seed)))
    st = Fk[Fk.condition == "standard"]
    for col in ("di_frac", "ab_frac"):                  # standard-training levels
        ax.axhline(st[col].mean(), color=CLEAN_C, ls=":", lw=0.7, zorder=1)
    block_axis(ax, 80)
    ax.set_ylim(0, 80)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_ylabel("Share of events (%)")
    title(ax, "Redrawn labels: random noise lowers abstraction (Ab-E) as noise across superclasses does")
    print("[c] process fractions:", ", ".join(f"{r['condition']} {r['process']} {r['mean']:.1f}" for r in rows))
    sd = F.rename(columns={"di_frac": "di_frac_real_transitions", "ab_frac": "ab_frac_real_transitions"})
    ns.source_data(pd.concat([sd.assign(table="per lane"), pd.DataFrame(rows).assign(table="condition mean (95% CI)")],
                             ignore_index=True), NAME, "c")


def panel_d(ax):
    A = L.alive_features(FINAL_SD / "ed6_e_per_lane_layer.csv")
    ns.source_data(A, NAME, "d_per_lane_layer")
    A6 = A[A.checkpoint == 6].groupby(["condition", "seed"]).n_alive.mean().reset_index()
    A6 = A6[[(c, s) != L.EXCLUDED_LANE for c, s in zip(A6.condition, A6.seed)]]
    std = A6[A6.condition == "standard"].set_index("seed").n_alive
    rows = []
    for c, x in POS.items():
        kind, p = cond_key(c)
        d = A6[A6.condition == c].set_index("seed").n_alive
        m, lo, hi = L.t_ci(d.values)
        col = CLEAN_C if kind == "standard" else NEUTRAL
        if p == 0.1:                                     # p = 0.1: open bars
            ax.bar(x, m, width=0.59, facecolor="white", edgecolor=col, lw=0.8, zorder=2)
        else:
            ax.bar(x, m, width=0.62, color=col, alpha=ALPHA, lw=0, zorder=2)
        ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=L.ERR, elinewidth=0.7, capsize=1.6,
                    capthick=0.7, zorder=3)
        rel = 100 * (d / std.loc[d.index] - 1)
        rm, rlo, rhi = L.t_ci(rel.values) if c != "standard" else (0.0, 0.0, 0.0)
        rows.append(dict(condition=c, n=len(d), alive_per_layer_mean=m, ci_lo=lo, ci_hi=hi,
                         alive_total_5_layers_mean=5 * m, change_vs_standard_pct=rm, change_ci_lo=rlo,
                         change_ci_hi=rhi, seeds=";".join(str(s) for s in d.index)))
        print(f"[d] {c}: {m:.1f} per layer, {rm:+.1f}% [{rlo:+.1f}, {rhi:+.1f}] (n = {len(d)})")
    ax.axhline(std.mean(), color=ZERO, ls="--", lw=0.6, zorder=1)
    r = {x["condition"]: x["change_vs_standard_pct"] for x in rows}
    for c in ("between_sc_p03", "random_p03"):
        hi = [x for x in rows if x["condition"] == c][0]["ci_hi"]
        ax.text(POS[c], hi + 8, f"+{r[c]:.0f}%", fontsize=AS, ha="center", va="bottom")
    block_axis(ax, 425)
    ax.set_ylim(0, 430)
    ax.set_yticks([0, 100, 200, 300])
    ax.set_ylabel("Alive SAE features\nper layer (epoch 30)")
    title(ax, "Noise across superclasses and random\nnoise expand the alive SAE features")
    ns.source_data(pd.DataFrame(rows), NAME, "d")


# e: accuracy against probe, fixed labels
def panel_e(ax, ax_o):
    D = pd.read_csv(RESULTS / "STATIC_probe/static_probe.csv")
    D["probe"] = 100 * D.super20
    # both levels on the axes: conditions in amber shades (across darkest solid, within mid dashed), clean thin grey
    sty = (("clean", "standard", CLEAN_C), ("within_sc", "within", ns.AMBER[1]), ("between_sc", "between", ns.AMBER[0]))
    for arm, k, col in sty:
        g = D[D.arm == arm]
        for _, gs in g.groupby("seed"):
            gs = gs.sort_values("epoch")
            ax.plot(gs.probe, gs.val_acc, color=col, ls=LS[k], lw=0.5, alpha=0.4)
        mm = g.groupby("epoch")[["probe", "val_acc"]].mean()
        ax.plot(mm.probe, mm.val_acc, color=col, ls=LS[k], lw={"standard": 0.9}.get(k, 1.2), marker="o",
                ms=2.2, zorder={"between": 5, "within": 4}.get(k, 3))
    ax.text(66.5, 46, "noise within", fontsize=6, color=ns.AMBER_TXT[1], ha="left", va="center")
    ax.text(57.5, 51, "noise across", fontsize=6, color=ns.AMBER_TXT[0], ha="right", va="center")
    ax.text(64.0, 61, "clean", fontsize=6, color=ns.GREY_TXT, ha="right", va="center")
    ax.set_xlabel("Superclass probe (%)", labelpad=1.5)
    ax.set_ylabel("Fine-class accuracy (%)")
    ax.set_xlim(20, 70)
    ax.set_ylim(0, 64)
    ax.set_yticks([0, 20, 40, 60])
    rows = []
    for s, g in D.groupby("seed"):
        for ref in ("within_sc", "clean"):
            w = g[g.arm == ref].sort_values("probe")
            for _, r in g[(g.arm == "between_sc") & (g.epoch > 0)].iterrows():
                rows.append(dict(reference=ref, seed=int(s), epoch=int(r.epoch), probe=r.probe, val_acc=r.val_acc,
                                 offset_pp=r.val_acc - np.interp(r.probe, w.probe, w.val_acc)))
    R = pd.DataFrame(rows)
    # noise-across networks against each reference: darkest against noise within, light against clean training
    for ref, col, tcol, lab in (("within_sc", ns.AMBER[0], ns.AMBER_TXT[0], "vs within"),
                                ("clean", ns.AMBER[2], ns.AMBER_TXT[1], "vs clean")):
        P = R[R.reference == ref].pivot(index="seed", columns="epoch", values="offset_pp")
        m, lo, hi = L.t_ci(P.values)
        x = P.columns.values
        ax_o.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
        ax_o.plot(x, m, color=col, ls=LS["between"], lw=1.2, marker="o", ms=2.2)
        ax_o.text(31.2, m[-1], lab, fontsize=6, color=tcol, va="center", ha="left", clip_on=False)
        print(f"[e] between offset {ref}: " + ", ".join(f"{e}: {v:+.2f} [{a:+.2f}, {b:+.2f}]"
                                                       for e, v, a, b in zip(x, m, lo, hi)))
    ax_o.axhline(0, color=ZERO, ls="--", lw=0.6)
    ax_o.set_xticks([5, 10, 20, 30])
    ax_o.set_xlim(3, 31)
    ax_o.set_ylim(-3, 12)
    ax_o.set_yticks([0, 5, 10])
    ax_o.set_xlabel("Epoch", labelpad=1.5)
    ax_o.set_ylabel("Accuracy offset at\nequal probe (pp)")
    title(ax, "Fixed labels: at equal probe, noise across superclasses stays within 3 percentage points of clean training")
    ns.source_data(R, NAME, "e")


def legend_strip(fig):
    dk = L.ERR
    h = [Line2D([], [], color=dk, ls=LS["between"], lw=LW["between"], label="noise across superclasses"),
         Line2D([], [], color=dk, ls=LS["within"], lw=LW["within"], label="noise within superclasses"),
         Line2D([], [], color=dk, ls=LS["random"], lw=LW["random"], label="random noise"),
         Line2D([], [], color=CLEAN_C, ls="-", lw=LW["standard"], label="clean / standard training"),
         Patch(color=ns.BLUE[1], alpha=ALPHA, lw=0, label="superclass (probe, Ab-E)"),
         Patch(color=ns.AMBER[1], alpha=ALPHA, lw=0, label="class (accuracy, Di-E)"),
         Line2D([], [], color=dk, ls="none", marker="o", ms=3.2, mfc="white", mew=0.8, label="open, p = 0.1")]
    fig.legend(handles=h[:4], loc="upper center", bbox_to_anchor=(0.5, 1 - 0.3 / H), ncol=4, fontsize=6.5,
               handlelength=2.6, columnspacing=1.8, frameon=False)
    fig.legend(handles=h[4:], loc="upper center", bbox_to_anchor=(0.5, 1 - 4.0 / H), ncol=3, fontsize=6.5,
               handlelength=2.6, columnspacing=1.8, frameon=False)


def main():
    ns.apply()
    fig = plt.figure(figsize=(W * ns.MM, H * ns.MM))
    x0, wa, ga = 18, 50, 5
    ya, yb, hab = 125, 90, 21
    ax_a = [mm_axes(fig, x0 + k * (wa + ga), ya, wa, hab) for k in range(3)]
    ax_b = [mm_axes(fig, x0 + k * (wa + ga), yb, wa, hab) for k in range(3)]
    y2, h2 = 49, 22
    ax_c = mm_axes(fig, 18, y2, 92, h2)
    ax_d = mm_axes(fig, 128, y2, 50, h2)
    y3, h3 = 9, 22
    ax_e = mm_axes(fig, 18, y3, 70, h3)
    ax_eo = mm_axes(fig, 116, y3, 46, h3)
    dose_panels(ax_a, "probe_sc_avgpool", ns.BLUE, "Superclass\nprobe (%)", (15, 72), [20, 40, 60], "a", xlabels=False,
                headers=True)
    dose_panels(ax_b, "fine_acc", ns.AMBER, "Fine-class\naccuracy (%)", (0, 62), [0, 20, 40, 60], "b")
    title(ax_a[0], "Redrawn labels: random noise lowers the probe as noise across superclasses does, and the gap "
                   "closes by epoch 30", x=0.0)
    title(ax_b[0], "All noise slows fine-class accuracy; with fixed labels, crossing superclasses slows it most")
    panel_c(ax_c)
    panel_d(ax_d)
    panel_e(ax_e, ax_eo)
    legend_strip(fig)
    # a's claim title sits above the column headers
    ax_a[0].texts[-1].set_position((0.0, 1.0 + 6.0 / hab))
    for ax, letter, xmm, ymm in ((ax_a[0], "a", 1, ya + hab + 10.5), (ax_b[0], "b", 1, yb + hab + 6),
                                 (ax_c, "c", 1, y2 + h2 + 6), (ax_d, "d", 113, y2 + h2 + 9),
                                 (ax_e, "e", 1, y3 + h3 + 6)):
        b = ax.get_position()
        ns.panel(ax, letter, x=(xmm / W - b.x0) / b.width, y=(ymm / H - b.y0) / b.height)
    print("saved", ns.save_final(fig, NAME, 180))


if __name__ == "__main__":
    main()
