#!/usr/bin/env python3
"""Fig. 6: building the scaffold in advance gives fine-grained learning a head start, one panel per step
of the argument. Loaders from fig6_panels.py, curriculum_lib.py, blind_arm.py and ed9.py.

  a  the same CIFAR-100 test images (three of the 20 superclasses, five classes each;
     figures/inputs/fig6_cifar_examples.npz from scripts/paper/fig6_cifar_examples.py) and how each pre-training task
     labels them, cell for cell (true superclasses, one colour per row; random groups, mixed; superclass-blind task,
     one colour per column), then the fine classes
  b  the 50-epoch sweep (seed 42): fine-class accuracy against epochs from the first fine-class label, one curve per
     pre-training length, the first 10 fine-class epochs stretched fourfold; arrow, the head start of the 30-epoch run
     (fig6_panels.lead_vs_drawn)
  c  dose-response, the five-seed arms at 0, 20 and 30 pre-training epochs: head start (mean, 95% CI, seeds) and RDT
     features selective for the true superclasses at the switch (seeds 7, 42, 137; 0 = initialisation;
     DICTFREE/hierarchy/D_sae_census_timecourse.csv, SWEEP_RDT/census_K30.csv)
  d  specificity: 20 epochs on the superclass-blind task, random groups or true superclasses; RDT features selective
     for the true superclasses at the switch (open, the random groups' own groups) and head start (dots, seeds)
  e  RDT share of class-selective features confined to their superclass during pre-training (RDT_pace/
     curriculum_full_long.csv, rdt_pace_curriculum_full.py; means; dashed, chance)
Run: ES_FIG_OUT=<repo>/analysis_outputs/paper/figures/fig6 .venv/bin/python scripts/paper/figures/interventions/fig6.py
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("ES_FIG_OUT", str(HERE.parents[3] / "analysis_outputs" / "paper" / "figures" / "fig6"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import nmi_style as ns  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import fig6_panels as G  # noqa: E402
import ed9 as E  # noqa: E402

B, L = G.B, G.L
NAME, W, H = "fig6", 180, 150
SEEDS = G.SEEDS
BLUE, AMBER, GREYS = ns.BLUE, ns.AMBER, ns.GREYS
BT, AT, GT, DARK = ns.BLUE_TXT, ns.AMBER_TXT, ns.GREY_TXT, ns.GREYS[0]
EXAMPLES = G.RESULTS / "figures" / "inputs" / "fig6_cifar_examples.npz"
CENSUS = G.RESULTS / "DICTFREE" / "hierarchy" / "D_sae_census_timecourse.csv"
CENSUS30 = G.RESULTS / "SWEEP_RDT" / "census_K30.csv"
ROWNAME = {"flowers": "flowers", "vehicles_1": "vehicles", "large_carnivores": "large carnivores"}
SRC = {}


def switch_census(path, lane, k, col):
    d = pd.read_csv(path)
    d = d[(d.src == "sae") & (d.lane == lane) & (d.epoch == k)].set_index("seed")[col] * 100
    return d.loc[E.SEEDS_RDT].values


def panel_design(ax):
    """The same 15 images, how each pre-training task labels them (cell for cell), then the fine classes."""
    ex = np.load(EXAMPLES)
    s, gap = 1.0, 0.12
    for r, sname in enumerate(ex["superclasses"]):
        y = (2 - r) * (s + gap)
        for c in range(5):
            ax.imshow(ex["images"][r, c], extent=(c * (s + gap), c * (s + gap) + s, y, y + s), interpolation="lanczos",
                      zorder=2)
        ax.text(-0.2, y + s / 2, ROWNAME[str(sname)], ha="right", va="center", fontsize=6.5, color=DARK)
    top = 3 * (s + gap)
    ax.text(2.8, top + 0.15, "CIFAR-100 images\n(3 of 20 superclasses, 5 classes each)", ha="center", va="bottom",
            fontsize=6.5, color=DARK, linespacing=1.0)
    rng = np.random.default_rng(3)
    while True:                                   # random groups of five that mix superclasses
        perm = rng.permutation(15).reshape(3, 5)
        if all(len({int(i) // 5 for i in g}) >= 2 for g in perm):
            break
    rand = np.empty(15, int)
    for g, cells in enumerate(perm):
        rand[cells] = g
    fine_cols = plt.get_cmap("YlOrBr")(np.linspace(0.25, 0.85, 15))
    tasks = [("true superclasses", "20 labels", lambda r, c: BLUE[[0, 2, 1][r]], BT[0],
              "builds the coarse\nstructure"),
             ("random groups", "20 labels", lambda r, c: BLUE[[0, 2, 1][rand[r * 5 + c]]], BT[1],
              "tests whether the\nimages carry it"),
             ("superclass-blind", "5 labels", lambda r, c: GREYS[[0, 2, 1, 3, 1][c]], GT,
              "equal training,\nlittle coarse structure"),
             ("fine classes", "100 labels", lambda r, c: fine_cols[(r * 5 + c * 7) % 15], AT[0],
              "every image its class,\nfor the rest of training")]
    cs = 0.62
    for t, (name, nlab, col, tc, why) in enumerate(tasks):
        x0 = 7.6 + t * 4.4 + (1.6 if t == 3 else 0)
        for r in range(3):
            for c in range(5):
                ax.add_patch(Rectangle((x0 + c * cs, 2.2 - r * cs * 1.15 + 0.3), cs * 0.9, cs * 0.9 * 1.15,
                                       color=col(r, c), lw=0, zorder=2))
        ax.text(x0 + 2.5 * cs, top + 0.15, f"{name}\n{nlab}", ha="center", va="bottom", fontsize=6.5, color=tc,
                linespacing=1.0)
        ax.text(x0 + 2.5 * cs, 0.95, why, ha="center", va="top", fontsize=6, color=GREYS[1], linespacing=1.0)
    ax.annotate("", xy=(7.3, 1.75), xytext=(5.9, 1.75),
                arrowprops=dict(arrowstyle="-|>", color=GREYS[1], lw=1.0, mutation_scale=8))
    ax.plot([7.5, 7.5 + 3 * 4.4 - 1.2], [-0.25] * 2, color=GREYS[2], lw=0.6)
    ax.text(7.5 + (3 * 4.4 - 1.3) / 2, -0.4, "pre-training labels for the same 15 images (one colour, one label)",
            ha="center", va="top", fontsize=6.5, color=DARK)
    xf = 7.6 + 3 * 4.4 + 1.6                      # left edge of the fine-class grid
    ax.plot([xf - 0.1, xf + 5 * cs], [-0.25] * 2, color=GREYS[2], lw=0.6)
    ax.text(xf + 2.5 * cs, -0.4, "fine-class training", ha="center", va="top", fontsize=6.5, color=DARK)
    ax.annotate("", xy=(xf - 0.2, 1.75), xytext=(xf - 2.6, 1.75),
                arrowprops=dict(arrowstyle="-|>", color=GREYS[1], lw=1.0, mutation_scale=8))
    ax.text(xf - 1.4, 1.95, "new\nclassifier", ha="center", va="bottom", fontsize=6, color=GREYS[1], linespacing=1.0)
    ax.text((-3.0 + xf + 5 * cs) / 2, -1.15, "Each network is pre-trained on one of the three tasks (20 epochs; the true superclasses "
            "also 5–30), then its classifier is replaced and it learns the 100 fine classes for the rest of training.",
            ha="center", va="top", fontsize=6, color=GREYS[1])
    ax.set_xlim(-3.0, xf + 5 * cs + 0.3); ax.set_ylim(-1.75, top + 1.3)
    ax.set_aspect("equal"); ax.set_anchor("C"); ax.axis("off")


def band(ax, x, Y, c, lw=1.3):
    m = Y.mean(0)
    lo, hi = zip(*[L.tci(Y[:, i])[1:] for i in range(Y.shape[1])])
    ax.fill_between(x, lo, hi, color=c, alpha=0.2, lw=0)
    ax.plot(x, m, color=c, lw=lw)
    return m


STRETCH, GAIN = 10, 4.0       # panel b: epochs 0-10 drawn GAIN times wider than the rest, so the head start is legible


def stretch(x):
    x = np.asarray(x, float)
    return np.where(x <= STRETCH, x * GAIN, STRETCH * GAIN + (x - STRETCH))


def unstretch(y):
    y = np.asarray(y, float)
    return np.where(y <= STRETCH * GAIN, y / GAIN, STRETCH + (y - STRETCH * GAIN))


def panel_sweep(ax):
    """The 50-epoch sweep (seed 42) on the fine-class time axis: fine-class accuracy against epochs from the first
    fine-class label, one curve per superclass pre-training length (amber, light to dark; 0 = scratch, grey); the first
    10 epochs stretched; arrow, the head start of the 30-epoch run, the epochs scratch needs to reach its accuracy after
    one fine-class epoch."""
    ref = L.sweep_curve("standard50")
    rows = []
    for c in L.COND:
        v = L.sweep_curve(c)
        k = L.KNOM[c]
        tx = (lambda e: e - E.RESUME_OFFSET if e >= 10 else e) if c == E.RESUMED else (lambda e: e)
        kt = tx(k) if k else 0
        col = E.AMB_RAMP(E.ramp_t(kt)) if k else ns.SCRATCH_STYLE["color"]
        es = [e for e in sorted(v) if tx(e) > kt]
        x = np.array([tx(e) - kt for e in es]); y = np.array([v[e] for e in es])
        ax.plot(x, y, color=col, lw=1.0 if k else 0.9, ls="--" if c == E.RESUMED else "-", zorder=3 if k else 2)
        ax.plot(x[-1], y[-1], "o", ms=2.4, color=col, zorder=4)
        ax.text(x[-1], y[-1] + 1.6, f"{kt}", fontsize=6, color=GT if not k else AT[0], ha="center", va="bottom")
        rows += [dict(condition=c, pretrain_epochs=kt, epoch_from_first_fine_label=int(xx), val_acc=yy)
                 for xx, yy in zip(x, y)]
    v1 = L.sweep_curve("switch_e30")[31]
    lead = G.lead_vs_drawn(v1, np.array([ref[e] for e in range(1, 51)]))
    ax.annotate("", xy=(1 + lead, v1), xytext=(1, v1), zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=0.9, mutation_scale=7, shrinkA=0, shrinkB=0))
    ax.plot([1], [v1], "o", ms=3.4, color=E.AMB_RAMP(E.ramp_t(30)), zorder=7)
    ax.annotate(f"head start of the\n30-epoch run, {lead:.1f} epochs", xy=(1 + lead / 2, v1 + 0.4), xytext=(2.2, 50),
                fontsize=6, color=DARK, ha="left", va="bottom", linespacing=0.95,
                arrowprops=dict(arrowstyle="-", color=GREYS[1], lw=0.5, shrinkA=1, shrinkB=1))
    ax.axvline(STRETCH, color=GREYS[3], lw=0.6, zorder=0)
    ax.text(STRETCH / 2 + 0.5, 11.5, "first 10 epochs stretched", fontsize=6, color=GT, ha="center", va="bottom")
    ax.axhspan(57, 59, color=L.WINDOW, ec="#D1D5DB", lw=0.5, zorder=0)
    ax.text(1.0, 62.6, "57–59%", fontsize=6, ha="left", va="center", color=GT)
    ax.set_xscale("function", functions=(stretch, unstretch))
    ax.set_xlim(0, 52); ax.set_xticks([1, 5, 10, 20, 30, 40, 50])
    ax.set_ylim(10, 66); ax.set_yticks(range(10, 61, 10))
    ax.set_xlabel("Epoch from the first fine-class label")
    ax.set_ylabel("Fine-class accuracy (%)")
    SRC["b"] = pd.DataFrame(rows)
    print(f"[b] head start of the 30-epoch run {lead:.2f} epochs at {v1:.2f}%")


def panel_dose(axh, axc):
    """The five-seed arms at 0, 20 and 30 pre-training epochs: head start (top) and RDT census at the switch (bottom),
    on one x-axis (the one-seed sweep is panel b)."""
    rows, means = [], [(0, 0.0)]
    for k in (20, 30):
        v = np.array([E.lead1("sc_true", s, k) for s in SEEDS])
        m, lo, hi = L.tci(v)
        means.append((k, m))
        axh.errorbar(k, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=4.5, color=AMBER[0], elinewidth=0.9, zorder=4)
        axh.scatter([k + 0.9] * len(v), v, s=4, color=DARK, zorder=5)
        axh.text(k - 1.0, m + 0.9, f"{m:.1f}", fontsize=6, color=AT[0], ha="right", va="bottom")
        rows += [dict(measure="head start (epochs)", source="controlled", pretrain_epochs=k, seed=s, value=x)
                 for s, x in zip(SEEDS, v)]
    axh.plot(*zip(*means), color=AMBER[0], lw=1.0, zorder=3)
    axh.plot([0], [0], "o", ms=4, color=ns.SCRATCH_STYLE["color"], zorder=4)
    axh.set_ylim(-0.5, 7.5); axh.set_yticks([0, 3, 6]); axh.set_ylabel("Head start\n(epochs)")
    axh.set_xticklabels([])
    pts = [(0, switch_census(CENSUS, "sc_true_K20", 0, "census_genuine")),
           (20, switch_census(CENSUS, "sc_true_K20", 20, "census_genuine")),
           (30, switch_census(CENSUS30, "sc_true_K30", 30, "census_genuine"))]
    axc.plot([p[0] for p in pts], [p[1].mean() for p in pts], color=BLUE[0], lw=1.0, zorder=3)
    for k, v in pts:                              # the top row's markers: mean dot, seed dots offset right
        if k:
            axc.plot([k], [v.mean()], "o", ms=4.5, color=BLUE[0], zorder=4)
            axc.scatter([k + 0.9] * len(v), v, s=4, color=DARK, zorder=5)
            axc.text(k - 1.0, v.mean() + 2.5, f"{v.mean():.0f}%", fontsize=6, color=BT[0], ha="right", va="bottom")
        else:
            axc.plot([0], [v.mean()], "o", ms=4, color=ns.SCRATCH_STYLE["color"], zorder=4)
        rows += [dict(measure="RDT features selective for the true superclasses at the switch (% of alive)",
                      source="controlled", pretrain_epochs=k, seed=s, value=x) for s, x in zip(E.SEEDS_RDT, v)]
    axc.set_ylim(-2, 36); axc.set_yticks([0, 15, 30]); axc.set_ylabel("Coarse features\n(RDT, %)")
    for a in (axh, axc):
        a.set_xlim(-2, 35); a.set_xticks([0, 10, 20, 30])
    axc.set_xlabel("Superclass pre-training (epochs)")
    SRC["c"] = pd.DataFrame(rows)


def panel_tasks(axc, axh):
    b = B.census_rows(); b = b[b.epoch == 20].set_index("seed").census_genuine * 100
    census = [b.loc[E.SEEDS_RDT].values, switch_census(CENSUS, "sc_rand_K20", 20, "census_genuine"),
              switch_census(CENSUS, "sc_true_K20", 20, "census_genuine")]
    own = switch_census(CENSUS, "sc_rand_K20", 20, "census_imp_genuine")
    lead = [np.array([E.lead1(a, s) for s in SEEDS]) for a in ("sc_blind", "sc_rand", "sc_true")]
    tasks = ("superclass-blind task", "random groups", "true superclasses")
    rows = []
    for ax, vals, ramp, txt, fmt, seeds, meas in (
            (axc, census, BLUE, BT, "{:.0f}%", E.SEEDS_RDT,
             "RDT features selective for the true superclasses at the switch (% of alive)"),
            (axh, lead, AMBER, AT, "{:.1f}", SEEDS, "head start (epochs)")):
        for i, (v, sty) in enumerate(zip(vals, ("open", "hatch", "fill"))):
            c = ramp[[2, 1, 0][i]]
            if sty == "open":
                ax.bar(i, v.mean(), width=0.7, facecolor="white", edgecolor=c, lw=0.9, ls=":")
            elif sty == "hatch":
                ax.bar(i, v.mean(), width=0.7, facecolor=c, edgecolor=c, alpha=0.45, hatch=L.HATCH, lw=0.9)
            else:
                ax.bar(i, v.mean(), width=0.7, facecolor=c, edgecolor=c, alpha=0.85, lw=0.9)
            ax.scatter(i + np.linspace(-0.15, 0.15, len(v)), v, s=5, color=DARK, zorder=3)
            ax.text(i, max(v) + (1.2 if ax is axc else 0.3), fmt.format(v.mean()), fontsize=6, color=txt[0],
                    ha="center", va="bottom")
            rows += [dict(task=tasks[i], measure=meas, seed=s, value=x) for s, x in zip(seeds, v)]
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["blind", "random", "true"], fontsize=6.5)
        ax.set_xlim(-0.6, 2.6)
    axc.scatter(1 + np.linspace(-0.15, 0.15, 3), own, s=7, facecolor="white", edgecolor=BLUE[0], lw=0.7, zorder=4)
    axc.text(1, own.max() + 1.5, "own\ngroups", fontsize=6, color=BT[0], ha="center", va="bottom", linespacing=0.9,
             bbox=dict(facecolor="white", edgecolor="none", pad=0.6), zorder=5)
    rows += [dict(task="random groups", measure="RDT features selective for its own groups at the switch (% of alive)",
                  seed=s, value=x) for s, x in zip(E.SEEDS_RDT, own)]
    axc.set_ylim(0, 36); axc.set_yticks([0, 15, 30]); axc.set_ylabel("Features for the true\nsuperclasses (RDT, %)")
    axh.set_ylim(0, 6); axh.set_yticks([0, 2, 4, 6]); axh.set_ylabel("Head start (epochs)")
    for ax in (axc, axh):
        ax.text(1, -0.2, "20 pre-training epochs each", transform=ax.get_xaxis_transform(), fontsize=6, color=GT,
                ha="center", va="top")
    SRC["d"] = pd.DataFrame(rows)


def panel_nesting(ax):
    """Share of class features confined to their superclass during pre-training (stored epochs 5-20)."""
    D = pd.read_csv(G.RESULTS / "RDT_pace" / "curriculum_full_long.csv")
    rows = []
    for key, lab, c, ls in (("blind", "superclass-blind", AMBER[2], ":"), ("rand", "random groups", AMBER[1], "--"),
                            ("true", "true superclasses", AMBER[0], "-")):
        x = D[(D.arm == key) & (D.epoch >= -15) & (D.epoch <= 0)]
        P = x.pivot(index="seed", columns="epoch", values="share_confined") * 100
        ep = P.columns.values + 20
        m = P.values.mean(0)
        ax.plot(ep, m, color=c, ls=ls, lw=1.3, marker="o", ms=2.4)
        ax.text(20.6, m[-1] + {"true": 2.5, "rand": -2.5, "blind": 0}[key], f"{lab}, {m[-1]:.0f}%", fontsize=6,
                color=AT[0 if key == "true" else 1], ha="left", va="center")
        rows += [dict(task=lab, seed=int(sd), pretrain_epoch=int(e) + 20, share_confined_pct=v)
                 for sd in P.index for e, v in zip(P.columns, P.loc[sd])]
    ax.axhline(5, color=DARK, lw=0.7, ls=(0, (3, 2)))
    ax.text(20.6, 6.5, "chance", fontsize=6, color=DARK, ha="left", va="bottom")
    ax.set_xlim(4, 34); ax.set_xticks([5, 10, 15, 20]); ax.set_ylim(0, 72); ax.set_yticks([0, 20, 40, 60])
    ax.set_xlabel("Pre-training epoch (before any fine-class label)")
    ax.set_ylabel("Class features within\ntheir superclass (%)")
    SRC["e"] = pd.DataFrame(rows)


def main():
    ns.apply()
    for f in (ns.FINAL_OUT / "source_data").glob(f"{NAME}_*.csv"):
        f.unlink()
    fig = plt.figure(figsize=(W * ns.MM, H * ns.MM))

    def box(x0, x1, top, h):
        return fig.add_axes([x0 / W, 1 - (top + h) / H, (x1 - x0) / W, h / H])

    panel_design(box(2, 178, 6, 40))
    panel_sweep(box(14, 84, 58, 32))
    panel_dose(box(118, 172, 58, 14), box(118, 172, 76, 14))
    panel_tasks(box(14, 44, 112, 28), box(60, 90, 112, 28))
    panel_nesting(box(118, 176, 112, 28))
    titles = {"a": (1, 2, "Same images, different pre-training labels"),
              "b": (1, 51, "Longer superclass pre-training, faster fine-grained learning"),
              "c": (100, 51, "Dose-response: more coarse structure, larger head start"),
              "d": (1, 105, "Specificity: equal training without coarse structure gives little head start"),
              "e": (100, 105, "Class features form inside the coarse structure before any fine label")}
    for k, (x, y, t) in titles.items():
        fig.text(x / W, 1 - y / H, k, fontsize=8, fontweight="bold", va="top", ha="left")
        fig.text((x + 3.8) / W, 1 - (y + 0.25) / H, t, fontsize=7, va="top", ha="left")
    for k, df in SRC.items():
        ns.source_data(df, NAME, k)
    print("saved", ns.save_final(fig, NAME, 180, strict=True))


if __name__ == "__main__":
    main()
