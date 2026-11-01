#!/usr/bin/env python3
"""Suppression, blocking and removal panels: the network uses the superclass directions while it learns. Loaders and
panel functions used by routing/routing_panels.py (Extended Data Fig. 6), intervention_panels.py and fig4.py.

Row 1, suppression in the last block, epochs 1-10: a cost vs sham and random directions (intervention_panels.panel_supp_cost);
  b fine-class accuracy minus sham, epochs 1-30 (CRITICAL_period/val_curves.csv); c cost split into superclass assignment
  and class given the true superclass (intervention_panels.panel_supp_where).
Row 2: d fresh read-out of the unprojected activations (RANDGROUP_control/fresh_readout.csv); e superclass share of
  the classifier input relative to sham (REGENERATION).
Row 3, acquisition blocking: f assignment and accuracy per arm (GRADJOINT/long.csv); g random-grouping block
  (GRADBLOCK/final_long.csv); h stage-matched block, layer3 to pooled (effect_units); i misrouted images (intervention_panels.panel_sr_i).
Row 4, removal from trained networks: j superclass directions vs 20 random groupings (PSEUDO_superclass);
  k specificity and l dose-response over groupings (RANDGROUP_control).
Row 5: m removing one superclass's features (intervention_panels.panel_removal); n shared learning signal (intervention_panels.panel_f_sharing).
Row 6: o per-superclass dose-response of blocking (GRADBLOCK/superclass_gradblock.csv).
suppression_first3() and factor_rows() are used by intervention_panels.py; main() draws all panels as one stand-alone
overview figure and runs only when ES_FIG_OUT is set.

Usage: ES_FIG_OUT=<dir> python scripts/paper/figures/suppression_panels.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Patch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nmi_style as ns  # noqa: E402
from fig_common import DOT, ERR, REF, RESULTS, SEEDS, RUNS_EXTENDED, WINDOW, val_curve  # noqa: E402

# hue of the fine-class accuracy quantities
ACC_HUE = ns.INDEX["CSI"]          # amber #F59E0B
SSI, CSI = ns.INDEX["SSI"], ns.INDEX["CSI"]
SLATE = ns.SLATE                   # null / random-group reference
# intervention arms coloured by ns.ARM; panels coded by outcome use blue/amber
ARM_C = {"superclass": ns.ARM["superclass"], "random": ns.ARM["random_directions"],
         "grouping": ns.ARM["random_grouping"], "sham": ns.ARM["sham"]}
RED = ns.RED_600                   # effect arrows
BAR_A = ns.BAR_ALPHA_CORE
BAND_A = ns.BAND_ALPHA_CORE
FIG = "ed8"
W_MM, H_MM = 180, 274   # six rows (a-c, d-e, f-i, j-l, m-n, o)
LS = ns.COND_LS                    # reference solid, superclass-level dashed, random dotted
BIG = 7.0                          # effect numbers (pt)

V = pd.read_csv(RESULTS / "CRITICAL_period" / "val_curves.csv")
X = pd.read_csv(RESULTS / "XLAYER" / "val_acc_curves.csv")
P_RAW = pd.read_csv(RESULTS / "PSEUDO_superclass" / "lastblock_raw.csv")
P_NET = pd.read_csv(RESULTS / "PSEUDO_superclass" / "lastblock_per_network.csv")


def tci(v):
    v = np.asarray(v, float)
    m = v.mean()
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return m, m - h, m + h


def mm_axes(fig, x, y_top, w, h):
    """Axes placed in mm from the top-left corner of the canvas."""
    return fig.add_axes([x / W_MM, 1 - (y_top + h) / H_MM, w / W_MM, h / H_MM])


def minus(s):
    return s.replace("-", "−")


def bars(ax, xs, vals, color, light, rng, width=0.62, dot_s=7):
    """Bar = mean, grey seed dots, dark 95% CI; light = hatched variant."""
    out = []
    for x, v in zip(xs, vals):
        m, lo, hi = tci(v)
        if light:
            ax.bar(x, m, width=width, color=color, alpha=0.25, lw=0, zorder=1)
            ax.bar(x, m, width=width, facecolor="none", edgecolor=color, lw=0.8, hatch="//////", zorder=1)
        else:
            ax.bar(x, m, width=width, color=color, alpha=BAR_A, lw=0, zorder=1)
        ax.scatter(x + rng.uniform(-0.2, 0.2, len(v)) * width, v, s=dot_s, color=DOT, lw=0, zorder=3)
        ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=ERR, elinewidth=0.9, capsize=2.5,
                    capthick=0.9, zorder=4)
        out.append((m, lo, hi))
    return out


# Removal from trained networks
SEGS = [[1, 2, 3, 5, 10, 11, 12, 13, 15, 20]]  # one series: epochs 11-20 from the same (epoch-0 sham) networks


def panel_a(ax, bx, name=FIG, letter="a"):
    R = P_RAW[P_RAW.config == "lastblock"] if "config" in P_RAW else P_RAW
    for seg in SEGS:
        r = R[R.epoch.isin(seg)]
        P = r[r.kind == "pseudo"].groupby(["partition", "epoch"]).cost_cen.mean().unstack()
        for _, row in P.iterrows():
            ax.plot(seg, row[seg].values, color=ARM_C["grouping"], lw=0.45, alpha=0.55, zorder=2)
        rnd = r[r.kind == "random"].groupby("epoch").cost_cen.mean()
        ax.plot(seg, rnd[seg].values, color=ARM_C["random"], ls=LS["random"], lw=1.0, marker="s", ms=2.2, zorder=3)
        T = r[r.kind == "true"].pivot_table(index="seed", columns="epoch", values="cost_cen")
        for _, row in T.iterrows():
            ax.scatter(seg, row[seg].values, s=4, color=ARM_C["superclass"], alpha=0.45, lw=0, zorder=4)
        ax.plot(seg, T[seg].mean().values, color=ARM_C["superclass"], ls=LS["standard"], lw=1.4, marker="o", ms=3.0, zorder=5)
    ns.source_data(pd.DataFrame(dict(seed=R.seed, epoch=R.epoch, directions=R.kind, grouping=R.partition.astype(int),
                                     accuracy_cost_pp=R.cost_cen)), name, letter)
    for a in (ax, bx):
        a.set_xlim(0, 21)
        a.set_xticks([1, 5, 10, 15, 20])
    N = P_NET.copy()
    g = N.groupby("epoch")
    # effect mark: margin over the average grouping at epoch 10
    t10, p10 = g.true_cen.mean()[10], g.pseudo_mean_cen.mean()[10]
    ax.add_patch(FancyArrowPatch((8.9, p10), (8.9, t10), arrowstyle="<|-|>", mutation_scale=6, color=RED, lw=1.4,
                                 zorder=6, shrinkA=0, shrinkB=0))
    ax.text(9.9, p10 - 2.4, minus(f"+{t10 - p10:.1f} pp"), color=RED, fontsize=BIG, ha="right", va="top", zorder=7)
    ax.set_ylim(-1, 27)
    ax.set_yticks([0, 10, 20])
    ax.tick_params(labelbottom=False)
    ax.set_ylabel("Fine-class accuracy\ncost of removal (pp)")
    ax.set_title("Removing superclass directions\nfrom trained networks", fontsize=7)
    lab = dict(clip_on=False, fontsize=6, va="center", ha="left", linespacing=1.0)
    ax.text(21.5, g.true_cen.mean()[20] + 1.0, "superclass\ndirections", color=ARM_C["superclass"], **lab)
    ax.text(21.5, g.pseudo_mean_cen.mean()[20] - 1.2, "20 random\ngroupings", color=ARM_C["grouping"], **lab)
    ax.text(21.5, g.random_cen.mean()[20] + 2.6, "19 random\ndirections", color=ARM_C["random"], **lab)
    # margin sub-axis: superclass minus the best of the 20 groupings, per network
    N["margin_best"] = N.true_cen - N.pseudo_max_cen
    for seg in SEGS:
        n = N[N.epoch.isin(seg)]
        bx.scatter(n.epoch, n.margin_best, s=4, color=ARM_C["superclass"], alpha=0.45, lw=0, zorder=4)
        bx.plot(seg, n.groupby("epoch").margin_best.mean()[seg].values, color=ARM_C["superclass"], lw=1.2, marker="o", ms=2.6,
                zorder=5)
    bx.axhline(0, color=REF, ls="--", lw=0.6, zorder=0)
    bx.set_ylim(-1.5, 6.5)
    bx.set_yticks([0, 5])
    bx.set_xlabel("Epoch", labelpad=1.5)
    bx.set_ylabel("vs best\ngrouping")
    bx.text(21.5, N[N.epoch == 20].margin_best.mean(), "margin over\nthe best of 20", color=ARM_C["superclass"], **lab)
    # stored values
    rk = g.rank_true_of21_cen.apply(lambda s: int((s == 1).sum()))
    e13 = N[N.epoch.isin([1, 2, 3])]
    late = N[N.epoch.isin([11, 12, 13, 15, 20])].groupby("epoch")
    summ = dict(
        epochs1_3_superclass=e13.true_cen.mean(), epochs1_3_groupings_mean=e13.pseudo_mean_cen.mean(),
        epochs1_3_random=e13.random_cen.mean(), epochs1_3_margin_vs_mean=e13.D_cen.mean(),
        epochs1_3_margin_vs_best=e13.margin_best.mean(),
        epoch10_margin_vs_mean=N[N.epoch == 10].D_cen.mean(),
        epoch10_margin_vs_best=N[N.epoch == 10].margin_best.mean(),
        epochs11_20_margin_vs_mean_min=late.D_cen.mean().min(), epochs11_20_margin_vs_mean_max=late.D_cen.mean().max(),
        epochs11_20_margin_vs_best_min=late.margin_best.mean().min(),
        epochs11_20_margin_vs_best_max=late.margin_best.mean().max(),
        random_min=g.random_cen.mean().min(), random_max=g.random_cen.mean().max(),
        n_networks_margin_vs_best_positive_epochs5_20=int((N[N.epoch >= 5].margin_best > 0).sum()),
        n_networks_epochs5_20=int((N.epoch >= 5).sum()))
    per_epoch = g[["true_cen", "pseudo_mean_cen", "pseudo_max_cen", "random_cen", "D_cen"]].mean().join(
        N.groupby("epoch").margin_best.mean()).join(rk.rename("n_networks_rank1_of21")).reset_index()
    ns.source_data(per_epoch.rename(columns={"true_cen": "superclass_mean", "pseudo_mean_cen": "groupings_mean",
                                             "pseudo_max_cen": "best_grouping_mean",
                                             "random_cen": "random_directions_mean",
                                             "D_cen": "margin_vs_mean_grouping",
                                             "margin_best": "margin_vs_best_grouping"}), name, f"{letter}_summary")
    ns.source_data(pd.DataFrame([summ]), name, f"{letter}_key_values")
    print("[a] rank-1 counts", rk.to_dict())
    print("[a] " + ", ".join(f"{k} {v:.2f}" for k, v in summ.items()))


# Suppression of the superclass directions
def pv(window, arm, t0=False):
    d = V[(V.window == window) & (V.arm == arm) & (V.t0 == t0)]
    if not t0:
        d = d[d.epoch > 0]
    return d.pivot(index="seed", columns="epoch", values="val_accuracy")


def px(arm):
    return X[X.arm == arm].pivot(index="seed", columns="epoch", values="val_acc")


def sham_e10():
    """Sham branched at epoch 10 (window 11-20 sham rows, incl. its unprojected epoch-10 checkpoint)."""
    return V[(V.window == "11-20") & (V.arm == "sham")].pivot_table(index="seed", columns="epoch",
                                                                    values="val_accuracy")


CASES = [  # header, superclass arm, random arm, sham, first window epoch, install window key, tag
    ("Suppression in the last block\nepochs 1–10", lambda: pv("1-10", "coarse"), lambda: pv("1-10", "random"),
     lambda: pv("1-10", "sham"), 1, None, "last_block_1_10"),
]
ARM_STYLE = {"superclass": dict(ls=LS["between_sc"], marker="o", band=BAND_A),
             "random": dict(ls=LS["random"], marker="s", band=BAND_A * 0.6)}


def deltas():
    """Per case: dict(D=superclass/random minus sham per seed x epoch, install=..., epochs)."""
    out = []
    for header, fC, fR, fS, st, win, tag in CASES:
        C, R, S = fC(), fR(), fS()
        ep = list(range(st, 31))
        res = dict(header=header, st=st, tag=tag, ep=ep, seeds=list(C.index), D={}, install={})
        for arm, A in (("superclass", C), ("random", R)):
            res["D"][arm] = A[ep].values - S.loc[A.index, ep].values
            if win:
                I = pv(win, "coarse" if arm == "superclass" else "random", t0=True)
                res["install"][arm] = I[st - 1].values - S.loc[I.index, st - 1].values
        out.append(res)
    return out


def suppression_first3(cases=None):
    """Cost of suppressing the superclass code, first three epochs of each window, vs sham and vs random directions
    (per seed, mean, t-based 95% CI). Used by intervention_panels.panel_supp_cost."""
    rows = []
    for c in cases or deltas():
        f_sham = c["D"]["superclass"][:, :3].mean(1)
        f_rand = f_sham - c["D"]["random"][:, :3].mean(1)
        ms, los, his = tci(f_sham)
        mr, lor, hir = tci(f_rand)
        e30 = (c["D"]["superclass"][:, -1], c["D"]["random"][:, -1])
        rows.append(dict(window=c["tag"], epochs=f"{c['st']}-{c['st'] + 2}",
                         superclass_minus_sham=ms, sham_ci95_lo=los, sham_ci95_hi=his,
                         sham_n_negative=int((f_sham < 0).sum()),
                         superclass_minus_random=mr, random_ci95_lo=lor, random_ci95_hi=hir,
                         random_n_negative=int((f_rand < 0).sum()),
                         **{f"seed{s}_minus_sham": v for s, v in zip(c["seeds"], f_sham)},
                         **{f"seed{s}_minus_random": v for s, v in zip(c["seeds"], f_rand)},
                         epoch30_superclass_minus_sham=e30[0].mean(), epoch30_random_minus_sham=e30[1].mean(),
                         epoch30_superclass_minus_random=(e30[0] - e30[1]).mean(),
                         epoch30_superclass_minus_random_n_negative=int(((e30[0] - e30[1]) < 0).sum()),
                         reference=("sham branched at epoch 10" if c["st"] == 21 else "sham, same branch")))
    return pd.DataFrame(rows)


def panel_b(axs, cases, name=FIG, letter="b"):
    rows = []
    for k, (ax, c) in enumerate(zip(axs, cases)):
        st, ep = c["st"], c["ep"]
        ax.axvspan(st - 0.5, st + 9.5, color=WINDOW, lw=0, zorder=0)
        ax.axhline(0, color=ARM_C["sham"], lw=1.0, ls=LS["standard"], zorder=2)
        for arm in ("random", "superclass"):
            sty, d = ARM_STYLE[arm], c["D"][arm]
            m = d.mean(0)
            ax.fill_between(ep, d.min(0), d.max(0), color=ARM_C[arm], alpha=sty["band"], lw=0, zorder=1)
            ax.plot(ep, m, color=ARM_C[arm], ls=sty["ls"], lw=1.0, marker=sty["marker"], ms=1.8, markevery=2,
                    zorder=4 if arm == "superclass" else 3)
            for j, e in enumerate(ep):
                for i, s in enumerate(c["seeds"]):
                    rows.append(dict(panel=c["tag"], arm=arm, seed=s, epoch=e, point="trained",
                                     delta_vs_sham=d[i, j]))
            if arm in c["install"]:
                di = c["install"][arm]
                ax.plot([st - 1, st], [di.mean(), m[0]], color=ARM_C[arm], ls=sty["ls"], lw=1.0, zorder=3)
                ax.plot([st - 1], [di.mean()], ls="none", marker=sty["marker"], ms=2.8, mfc="white", mec=ARM_C[arm],
                        mew=0.8, zorder=5)
                for s, v in zip(c["seeds"], di):
                    rows.append(dict(panel=c["tag"], arm=arm, seed=s, epoch=st - 1, point="install",
                                     delta_vs_sham=v))
                print(f"[b] {c['tag']} install step {arm}: {di.mean():.2f}")
        # brackets: the first three epochs of each window
        yb, tk = 2.8, 0.7
        ax.plot([st, st, st + 2, st + 2], [yb - tk, yb, yb, yb - tk], color=ERR, lw=0.8, zorder=6)
        ax.set_xlim(-0.5, 30.8)
        ax.set_xticks([1, 10, 20, 30])
        ax.set_xlabel("Epoch", labelpad=1.5)
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=ARM_C["sham"], lw=1.0, ls=LS["standard"]),
         Line2D([], [], color=ARM_C["superclass"], lw=1.0, ls=LS["between_sc"], marker="o", ms=2.4),
         Line2D([], [], color=ARM_C["random"], lw=1.0, ls=LS["random"], marker="s", ms=2.4)]
    axs[0].legend(h, ["sham", "superclass directions", "random directions"], loc="lower right", ncol=3,
                  handlelength=2.3, columnspacing=1.0, handletextpad=0.4, borderaxespad=0.2, fontsize=6.3)
    for ax in axs:
        ax.set_ylim(-9.5, 3.6)
        ax.set_yticks([-8, -4, 0])
        ax.set_ylabel("Accuracy minus\nsham (pp)")
    for ax, c in zip(axs, cases):
        ax.set_title(c["header"].replace("\nepochs", ", epochs"), fontsize=6.5, pad=2)
    ns.source_data(pd.DataFrame(rows), name, letter)


# Where the suppression cost arises
FAC = RESULTS / "CLASS_level" / "per_class_long.csv"


F_CASES = [  # tag (as in b), kind in per_class_long, epochs (first three of the window), sham lane
    ("last_block_1_10", "lastblock", [1, 2, 3], "e00"),
]
A_WIN = {"lastblock": "win 1-3", "late10": "win 11-13", "late20": "win 21-23", "xlayer": "win 1-3"}


def factor_rows():
    """Fine accuracy = P(superclass right) x P(fine right | superclass right). The superclass of a prediction is the
    superclass of the network's predicted fine class. Per seed and epoch, pooled over classes; superclass arm minus
    reference (random directions or sham), averaged over the first three epochs of the window
    (as scripts/paper/class_level.py, part A; within-superclass accuracy taken directly in pp)."""
    df = pd.read_csv(FAC)
    rows = []

    def fac(x):
        g = x.groupby("epoch")[["n", "correct", "sc_correct"]].sum()
        return pd.DataFrame({"p_fine": g.correct / g.n, "p_sc": g.sc_correct / g.n,
                             "p_cond": g.correct / g.sc_correct})
    for tag, kind, eps, sham in F_CASES:
        for ref in ("sham", "random"):
            for s in SEEDS:
                sel = lambda k, a: df[(df.kind == k) & (df.arm == a) & (df.seed == s) & df.epoch.isin(eps)]
                a = fac(sel(kind, "coarse"))
                b = fac(sel(kind, "random") if ref == "random" else sel("sham", sham))
                ep = a.index.intersection(b.index)
                a, b = a.loc[ep], b.loc[ep]
                assert len(ep) == 3, (tag, ref, s, list(ep))
                rows.append(dict(window=tag, kind=kind, epochs=f"{eps[0]}-{eps[-1]}", reference=ref, seed=s,
                                 d_fine_accuracy_pp=100 * (a.p_fine - b.p_fine).mean(),
                                 d_superclass_accuracy_pp=100 * (a.p_sc - b.p_sc).mean(),
                                 d_within_superclass_accuracy_pp=100 * (a.p_cond - b.p_cond).mean(),
                                 dlog_superclass=np.log(a.p_sc / b.p_sc).mean(),
                                 dlog_within_superclass=np.log(a.p_cond / b.p_cond).mean()))
    R = pd.DataFrame(rows)
    # consistency with the stored factorisation (A_factorisation_per_seed.csv)
    A = pd.read_csv(RESULTS / "CLASS_level" / "A_factorisation_per_seed.csv")
    for (kind, ref), g in R.groupby(["kind", "reference"]):  # last block, epochs 1-3
        st = A[(A.kind == kind) & (A.ref == ref) & (A.window == A_WIN[kind])].sort_values("seed")
        g = g.sort_values("seed")
        assert np.allclose(st.d_sc_pp.values, g.d_superclass_accuracy_pp.values, atol=1e-6), (kind, ref)
        assert np.allclose(st.dlog_cond.values, g.dlog_within_superclass.values, atol=1e-6), (kind, ref)
        assert np.allclose(st.d_fine_pp.values, g.d_fine_accuracy_pp.values, atol=1e-6), (kind, ref)
    return R


def panel_c(ax, centres):
    """Per window, change in superclass assignment (blue) and in accuracy within the correct
    superclass (amber), vs sham (filled) and vs random directions (hatched)."""
    R = factor_rows()
    O = pd.read_csv(RESULTS / "ORACLE_within" / "oracle_long.csv")
    O = O[O.epoch.isin([1, 2, 3])].groupby(["arm", "seed"]).oracle_logits.mean()
    R["d_class_given_true_superclass_pp"] = [
        O[("supp_sc", s)] - O[("sham" if ref == "sham" else "supp_rand", s)] if tag.startswith("last") else np.nan
        for tag, ref, s in zip(R.window, R.reference, R.seed)]
    rng = np.random.default_rng(6)
    meas = (("d_superclass_accuracy_pp", SSI, -8.6), ("d_class_given_true_superclass_pp", CSI, 8.6))
    out, lows, highs = [], [], []
    for (tag, kind, eps, _), xc in zip(F_CASES, centres):
        for col, hue, dx in meas:
            for ref, off in (("sham", -3.6), ("random", 3.6)):
                v = R[(R.window == tag) & (R.reference == ref)].sort_values("seed")[col].values
                x = xc + dx + off
                (m, lo, hi), = bars(ax, [x], [v], hue, ref == "random", rng, width=6.4, dot_s=6)
                if m < 0:
                    y, va = min(lo, v.min()) - 0.5, "top"
                else:
                    y, va = max(hi, v.max()) + 0.5, "bottom"
                ax.text(x, y, minus(f"{m:+.1f}"), fontsize=6.5, color=ERR, ha="center", va=va)
                lows.append(min(lo, v.min()))
                highs.append(max(hi, v.max()))
                out.append(dict(window=tag, epochs=f"{eps[0]}-{eps[-1]}", measure=col, reference=ref, mean=m,
                                ci95_lo=lo, ci95_hi=hi, n_below_zero=int((v < 0).sum()), n=len(v),
                                seeds=" ".join(f"{z:.2f}" for z in v)))
                print(f"[c] {tag:17s} {col:32s} vs {ref:6s}: {m:+.2f} ({lo:+.2f}, {hi:+.2f}); "
                      f"{(v < 0).sum()}/5 below 0; seeds {np.round(v, 2).tolist()}")
    ax.axhline(0, color=REF, ls="--", lw=0.6, zorder=0)
    ax.set_ylim(min(lows) - 3.0, max(highs) + 2.5)
    ax.set_yticks([-10, -5, 0, 5])
    ticks, labels = [], []
    for xc in centres:
        ticks += [xc + meas[0][2], xc + meas[1][2]]
        labels += ["superclass\nassignment", "class given\ntrue superclass"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=6)
    for t, (col, hue, dx) in zip(ax.get_xticklabels(), meas * len(centres)):
        t.set_color(hue if hue == SSI else "#B45309")
    ax.tick_params(axis="x", length=0, pad=2)
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("Change, first\n3 epochs (pp)")
    h = [Patch(facecolor="#9CA3AF", alpha=BAR_A, lw=0, label="vs sham"),
         Patch(facecolor="white", edgecolor="#6B7280", hatch="//////", lw=0.8, label="vs random directions")]
    ax.legend(handles=h, loc="lower right", bbox_to_anchor=(1.0, 1.04), ncol=2, handlelength=1.3,
              handleheight=0.9, borderaxespad=0.0, columnspacing=1.2)
    ax.set_title("Where the cost arises", fontsize=7, loc="left", x=0.0, pad=14)
    ns.source_data(R, FIG, "c")
    ns.source_data(pd.DataFrame(out), FIG, "c_summary")


# Rebuilding of the superclass directions during suppression
REGEN = RESULTS / "REGENERATION"
D_WINDOWS = (("1-10", 1, "Last block, epochs 1–10"),)


def panel_d(axs, name=FIG, letter="d"):
    """Superclass share of the classifier input (between-superclass scatter / total scatter of the avgpool
    activations the classifier receives, i.e. after the projection) relative to sham, per window; superclass-code vs
    random-direction suppression; mean and seed range (5 seeds), window shaded, window end dashed."""
    import json
    C = pd.read_csv(REGEN / "contrasts_vs_sham_per_seed_epoch.csv", dtype={"window": str})
    K = json.load(open(REGEN / "key_numbers.json"))["in_window_vs_sham"]
    rows = []
    for k, (ax, (win, st, title)) in enumerate(zip(axs, D_WINDOWS)):
        end = st + 9
        ax.axvspan(st - 0.5, end + 0.5, color=WINDOW, lw=0, zorder=0)
        ax.axhline(1, color=ARM_C["sham"], lw=1.0, ls=LS["standard"], zorder=2)
        for arm, key in (("random", "random"), ("superclass", "coarse")):
            sty = ARM_STYLE[arm]
            P = C[(C.window == win) & (C.arm == key)].pivot(index="seed", columns="epoch",
                                                              values="sc_share_ratio_to_sham").loc[SEEDS]
            assert P.notna().all().all() and P.shape[0] == 5, (win, arm)
            ep, v = P.columns.values, P.values
            m = v.mean(0)
            ax.fill_between(ep, v.min(0), v.max(0), color=ARM_C[arm], alpha=sty["band"], lw=0, zorder=1)
            ax.plot(ep, m, color=ARM_C[arm], ls=sty["ls"], lw=1.0, marker=sty["marker"], ms=1.8,
                    zorder=4 if arm == "superclass" else 3)
            for e, mm_, lo_, hi_, col in zip(ep, m, v.min(0), v.max(0), v.T):
                rows.append(dict(window=win, arm=arm, epoch=int(e), in_window=bool(st <= e <= end),
                                 mean_ratio_to_sham=mm_, seed_min=lo_, seed_max=hi_,
                                 **{f"seed_{s}": x for s, x in zip(SEEDS, col)}))
            if arm == "superclass":
                for e, which in ((st, "first_epoch"), (end, "last_epoch")):
                    ref = K[win]["coarse"][which]["sc_share_ratio_to_sham"]
                    assert abs(P[e].mean() - ref["mean"]) < 1e-3, (win, e, P[e].mean(), ref["mean"])
                s0, s1 = P[st].mean(), P[end].mean()
                ax.text(st, s0 + 0.12, f"{s0:.2f}", fontsize=6, color=ARM_C["superclass"], ha="center", va="bottom", zorder=6)
                ye, va = {"1-10": (s1 + 0.15, "bottom"), "11-20": (s1 - 0.06, "top"), "21-30": (0.5, "bottom")}[win]
                ax.text(end - 0.5, ye, f"{s1:.2f}", fontsize=6, color=ARM_C["superclass"], ha="right", va=va, zorder=6)
                post = [e for e in ep if e > end]
                print(f"[d] {win}: superclass directions / sham, first epoch {s0:.3f} (range {P[st].min():.3f}-"
                      f"{P[st].max():.3f}), last epoch {s1:.3f} (range {P[end].min():.3f}-{P[end].max():.3f})"
                      + (f"; first epoch after release ({post[0]}) {P[post[0]].mean():.3f} (range "
                         f"{P[post[0]].min():.3f}-{P[post[0]].max():.3f}); epoch 30 {P[30].mean():.3f}" if post else ""))
            else:
                print(f"[d] {win}: random directions / sham, epoch {st} {P[st].mean():.3f}, epoch {end} "
                      f"{P[end].mean():.3f}, epoch 30 {P[30].mean():.3f}"
                      + (f", epoch {end + 1} {P[end + 1].mean():.3f}" if end + 1 in P.columns else ""))
        ax.axvline(end + 0.5, color=REF, ls="--", lw=0.6, zorder=1)
        ax.set_xlim(-0.5, 30.8)
        ax.set_xticks([1, 10, 20, 30])
        ax.set_xlabel("Epoch", labelpad=1.5)
        ax.set_ylim(0, 2.25)
        ax.set_yticks([0, 1, 2])
        ax.set_title("Superclass share of the classifier input", fontsize=7, pad=3)
        if k:
            ax.tick_params(labelleft=False)
    axs[0].set_ylabel("Superclass share,\nratio to sham")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=ARM_C["sham"], lw=1.0, ls=LS["standard"]),
         Line2D([], [], color=ARM_C["superclass"], lw=1.0, ls=LS["between_sc"], marker="o", ms=2.4),
         Line2D([], [], color=ARM_C["random"], lw=1.0, ls=LS["random"], marker="s", ms=2.4)]
    axs[-1].legend(h, ["sham", "superclass directions", "random directions"], loc="upper right", ncol=3,
                   handlelength=2.3, columnspacing=1.0, handletextpad=0.4, borderaxespad=0.2, fontsize=6.3)
    ns.source_data(pd.DataFrame(rows), name, letter)


# Specificity: random-grouping directions, dose-response
SPEC = RESULTS / "RANDGROUP_control"


def panel_e(ax, name=FIG, letter="e"):
    """First-three-epoch change against sham for the superclass code, a random-grouping code and random directions
    projected out of the last block during training: fine-class accuracy (amber) and superclass assignment (blue)
    (randgroup_specificity.py, from routing_decomposition.json)."""
    A = pd.read_csv(SPEC / "specificity_arms.csv")
    rng = np.random.default_rng(8)
    arms = (("superclass", "superclass\ndirections", BAR_A, False),
            ("randgroup", "random-grouping\ndirections", 0.45, False),
            ("random", "random\ndirections", BAR_A, True))
    meas = (("d_fine_pp", ACC_HUE, -0.21), ("d_assignment_pp", SSI, 0.21))
    out, lows = [], []
    for i, (arm, lab, alpha, hatch) in enumerate(arms):
        g = A[A.arm == arm].sort_values("seed")
        for col, hue, dx in meas:
            v = g[col].values
            m, lo, hi = tci(v)
            x = i + dx
            if hatch:
                ax.bar(x, m, width=0.38, color=hue, alpha=0.25, lw=0, zorder=1)
                ax.bar(x, m, width=0.38, facecolor="none", edgecolor=hue, lw=0.8, hatch="//////", zorder=1)
            else:
                ax.bar(x, m, width=0.38, color=hue, alpha=alpha, lw=0, zorder=1)
            ax.scatter(x + rng.uniform(-0.07, 0.07, len(v)), v, s=6, color=DOT, lw=0, zorder=3)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=ERR, elinewidth=0.9, capsize=2.2,
                        capthick=0.9, zorder=4)
            y, va = (min(lo, v.min()) - 0.4, "top") if m < 0 else (max(hi, v.max()) + 0.4, "bottom")
            ax.text(x, y, minus(f"{m:+.1f}"), fontsize=6.3, color=ERR, ha="center", va=va)
            lows.append(min(lo, v.min()))
            out.append(dict(arm=arm, measure=col, mean=m, ci95_lo=lo, ci95_hi=hi, n_below_zero=int((v < 0).sum()),
                            coarse_removed_share=float(g.coarse_removed_share.mean()),
                            **{f"seed_{s}": x_ for s, x_ in zip(g.seed, v)}))
    ax.axhline(0, color=REF, ls="--", lw=0.6, zorder=0)
    ax.set_xticks(range(3))
    ax.set_xticklabels([a[1] for a in arms], fontsize=6.3)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(min(lows) - 1.6, 3.4)
    ax.set_yticks([-10, -5, 0])
    ax.set_ylabel("Change vs sham,\nfirst 3 epochs (pp)")
    sh = A.groupby("arm").coarse_removed_share.mean()
    ax.set_xticklabels([f"{a[1]}\n(share {sh[a[0]]:.2f})" for a in arms], fontsize=6.1)
    from matplotlib.patches import Patch as _P
    h = [_P(facecolor=ACC_HUE, alpha=BAR_A, lw=0, label="fine-class accuracy"),
         _P(facecolor=SSI, alpha=BAR_A, lw=0, label="superclass assignment")]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.3, frameon=False,
              handlelength=1.2, borderaxespad=0.1, columnspacing=1.0)
    ax.set_title("Removing directions in training: superclass vs random grouping", fontsize=7, pad=14)
    for r in out:
        print(f"[e] {r['arm']:10s} {r['measure']:16s} {r['mean']:+.2f} ({r['ci95_lo']:+.2f}, {r['ci95_hi']:+.2f}) "
              f"{r['n_below_zero']}/5 below 0; coarse share {r['coarse_removed_share']:.2f}")
    ns.source_data(pd.DataFrame(out), name, letter)


def panel_f(ax, name=FIG, letter="f"):
    """Within each trained network and epoch: Spearman rho over the 20 random groupings between the accuracy cost of
    removing a grouping's code and the share of the superclass code's coarse structure that code holds
    (randgroup_specificity.py; costs PSEUDO_superclass, shares coarse_overlap.py)."""
    import json
    S = json.load(open(SPEC / "specificity_summary.json"))["dose_response"]
    R = pd.DataFrame(S["per_case"])
    eps = sorted(R.epoch.unique())
    rng = np.random.default_rng(9)
    for k, e in enumerate(eps):
        g = R[R.epoch == e]
        ax.scatter(k + rng.uniform(-0.12, 0.12, len(g)), g.rho, s=9, color=ARM_C["grouping"], lw=0, zorder=3)
        ax.plot([k - 0.25, k + 0.25], [g.rho.median()] * 2, color=ERR, lw=1.2, zorder=4)
    ax.axhline(0, color=REF, ls="--", lw=0.6, zorder=0)
    ax.set_xticks(range(len(eps)))
    ax.set_xticklabels([str(e) for e in eps])
    ax.set_xlim(-0.6, len(eps) - 0.4)
    ax.set_ylim(-0.3, 1.3)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("Epoch", labelpad=1.5)
    ax.set_ylabel("Spearman ρ, cost vs\ncoarse share (20 groupings)")
    ax.text(len(eps) - 0.45, 1.28, f"ρ > 0 in {S['n_rho_positive']} of {S['n_cases']}\nmedian {S['median_rho']:.2f}",
            fontsize=6.0, color=ERR, ha="right", va="top", linespacing=1.0)
    ax.set_title("Cost rises with the\ncoarse structure held", fontsize=7, pad=3)
    ns.source_data(R, name, letter)
    ns.source_data(pd.read_csv(SPEC / "dose_response.csv"), name, f"{letter}_groupings")
    print(f"[f] dose-response rho>0 {S['n_rho_positive']}/{S['n_cases']}, median {S['median_rho']:.3f}")

GBF = RESULTS / "GRADBLOCK" / "final_long.csv"


def panel_g(ax, name=FIG, letter="g"):
    """Fully matched dose control of the acquisition blocking (Extended Data Fig. 7c): learning signal blocked at layer3 to the pooled
    layer only, superclass directions vs the jointly matched block (same rank; learning-signal share within 0.01 over
    epochs 1-3); epochs behind at the end of the block (epoch 10) in superclass assignment of a fresh 100-way read-out
    (blue) and in the network's fine-class accuracy (amber) (effect_units.blocking_lags on GRADJOINT/long.csv)."""
    import effect_units as V
    G = V.blocking_lags(pairs=(("gradsc_l3", "gradjoint_l3"),))
    pr = V.lag_proportion(G, blk="gradsc_l3", ctl="gradjoint_l3")
    rng = np.random.default_rng(10)
    out = []
    for i, (key, hue) in enumerate((("lag_superclass_end", SSI), ("lag_fine_end", ACC_HUE))):
        r = pr[key]
        m, (lo, hi), v = r["mean"], r["ci"], np.array(r["per_seed"], float)
        ax.bar(i, m, width=0.62, color=hue, alpha=BAR_A, lw=0, zorder=1)
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=6, color=DOT, lw=0, zorder=3)
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=ERR, elinewidth=0.9, capsize=2.2,
                    capthick=0.9, zorder=4)
        ax.text(i, max(hi, v.max()) + 0.3, f"{m:.1f}", fontsize=6.3, color=ERR, ha="center", va="bottom")
        out.append(dict(contrast="superclass - jointly matched, layer3 to pooled layer", epoch=10,
                        measure=key.replace("lag_", "").replace("_end", "") + " epochs behind", mean=m, ci95_lo=lo,
                        ci95_hi=hi, n_above_zero=int((v > 0).sum()), **{f"seed_{s}": x_ for s, x_ in zip(SEEDS, v)}))
    sl = pr["slope_fine_on_superclass"]
    out.append(dict(contrast="superclass - jointly matched, layer3 to pooled layer", epoch="1-10",
                    measure="fine lag per epoch of superclass lag", mean=sl["mean"], ci95_lo=sl["ci"][0],
                    ci95_hi=sl["ci"][1], **{f"seed_{s}": x_ for s, x_ in zip(SEEDS, sl["per_seed"])}))
    ax.axhline(0, color=REF, ls="--", lw=0.6, zorder=0)
    ax.set_xticks([0, 1])
    ax.tick_params(axis="x", length=0, pad=2)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 7.2)
    ax.set_yticks([0, 2, 4, 6])
    ax.set_ylabel("Epochs behind the matched\nblock, end of block")
    ax.set_title("Blocking at layer3 to\nthe pooled layer, matched", fontsize=7, pad=3)
    for r in out:
        print(f"[h] {r['measure']:40s} {r['mean']:.2f} ({r['ci95_lo']:.2f}, {r['ci95_hi']:.2f})")
    ns.source_data(pd.DataFrame(out), name, letter)


# Cross-check against the raw lanes (when the lane data are available)
def crosscheck():
    if not RUNS_EXTENDED.exists():
        print("[check] lane data not found; raw-lane cross-check skipped")
        return
    n, worst = 0, 0.0
    lanes = {("1-10", "sham"): "critical_sham/lanes/sham_e00_seed{s}",
             ("1-10", "coarse"): "prevention/lanes/lesion_scaffold_e00_seed{s}_rp65_Llayer4-avgpool"}
    for (w, arm), pat in lanes.items():
        P = pv(w, arm)
        for s in P.index:
            f = RUNS_EXTENDED / pat.format(s=s)
            vc = val_curve(f) if f.exists() else None
            if vc is None:
                continue
            for e in P.columns:
                if e in vc:
                    worst = max(worst, abs(vc[e] - P.loc[s, e]))
                    n += 1
    print(f"[check] {n} raw-lane values vs stored curves; max |diff| = {worst:.3g}")
    assert worst < 0.01, "stored curves disagree with the raw lanes"


def grouped_bars(ax, groups, measures, title, ylab, yticks, top=1.2, rng_seed=11):
    """x groups (label, {measure: per-seed values}); one bar per measure (colour), mean, seed dots, 95% CI."""
    rng = np.random.default_rng(rng_seed)
    k = len(measures)
    wid = 0.72 / k
    lows, highs, out = [], [], []
    for i, (glab, vals) in enumerate(groups):
        for j, (mkey, mlab, hue) in enumerate(measures):
            v = np.asarray(vals[mkey], float)
            x = i + (j - (k - 1) / 2) * wid
            (m, lo, hi), = bars(ax, [x], [v], hue, False, rng, width=wid * 0.92, dot_s=6)
            y, va = (min(lo, v.min()) - 0.3, "top") if m < 0 else (max(hi, v.max()) + 0.3, "bottom")
            ax.text(x, y, minus(f"{m:+.1f}"), fontsize=6.3, color=ERR, ha="center", va=va)
            lows.append(min(lo, v.min()) - 1.0)
            highs.append(max(hi, v.max()) + 1.0)
            out.append(dict(group=glab.replace("\n", " "), measure=mlab, mean=m, ci95_lo=lo, ci95_hi=hi,
                            n_below_zero=int((v < 0).sum()), n=len(v), seeds=" ".join(f"{z:.3f}" for z in v)))
    ax.axhline(0, color=REF, ls="--", lw=0.6, zorder=0)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=6.2)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.set_ylim(min(min(lows), -0.5), max(max(highs), 0) + top)
    ax.set_yticks(yticks)
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=7, pad=3)
    h = [Patch(facecolor=hue, alpha=BAR_A, lw=0, label=mlab) for _, mlab, hue in measures]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=6.0, frameon=False,
              handlelength=1.2, handleheight=0.9, borderaxespad=0.1, labelspacing=0.2)
    return pd.DataFrame(out)


def panel_fresh(ax, name=FIG, letter="d"):
    """Fresh 100-way ridge read-out of the unprojected pooled activations of the suppressed networks (epochs 1 and 3;
    RANDGROUP_control/fresh_readout.csv, scripts/paper/fresh_readout.py): fine-class read-out accuracy (amber) and
    superclass assignment (blue), superclass arm minus sham and minus random directions."""
    F = pd.read_csv(RESULTS / "RANDGROUP_control" / "fresh_readout.csv")
    M = F[F.epoch.isin([1, 3])].groupby(["arm", "seed"])[["fine_readout", "superclass_assignment"]].mean()
    groups = [(lab, {c: (M.loc["superclass"][c] - M.loc[ref][c]).loc[SEEDS].values
                     for c in ("fine_readout", "superclass_assignment")})
              for ref, lab in (("sham", "vs sham"), ("random", "vs random\ndirections"))]
    out = grouped_bars(ax, groups, (("fine_readout", "fine-class read-out", ACC_HUE),
                                    ("superclass_assignment", "superclass assignment", SSI)),
                       "The information stays: fresh\nread-out without projection", "Change, epochs 1, 3 (pp)",
                       [-2, 0, 2], top=3.2)
    print("[fresh] " + "; ".join(f"{r.group} {r.measure}: {r.mean:+.2f} [{r.ci95_lo:+.2f}, {r.ci95_hi:+.2f}]"
                                 for r in out.itertuples()))
    ns.source_data(out, name, letter)


def panel_rg_block(ax, name=FIG, letter="f"):
    """Acquisition blocking of a random grouping's directions vs the superclass directions, epochs 1-3, minus
    energy-matched blocking: network fine-class validation accuracy (amber; forward pass unchanged) and superclass
    assignment of the fresh ridge read-out (blue) (GRADBLOCK/final_long.csv)."""
    D = pd.read_csv(GBF)
    M = D[D.epoch.isin([1, 2, 3])].groupby(["arm", "seed"])[["val_accuracy", "fresh_assignment"]].mean()
    groups = [(lab, {c: (M.loc[arm][c] - M.loc["gradenergy"][c]).loc[SEEDS].values
                     for c in ("val_accuracy", "fresh_assignment")})
              for arm, lab in (("gradsc", "superclass\ndirections"), ("gradrg", "random-grouping\ndirections"))]
    out = grouped_bars(ax, groups, (("val_accuracy", "fine-class accuracy", ACC_HUE),
                                    ("fresh_assignment", "superclass assignment", SSI)),
                       "Blocking a random grouping", "Minus energy-matched,\nepochs 1–3 (pp)", [-6, -4, -2, 0],
                       top=4.0)
    print("[rg-block] " + "; ".join(f"{r.group} {r.measure}: {r.mean:+.2f} [{r.ci95_lo:+.2f}, {r.ci95_hi:+.2f}] "
                                    f"{r.n_below_zero}/5" for r in out.itertuples()))
    ns.source_data(out, name, letter)



def panel_dose(ax, name=FIG, letter="o"):
    """Per-superclass dose-response of acquisition blocking (GRADBLOCK/superclass_gradblock.csv, producer
    scripts/paper/gradblock_superclass.py): x = share of a superclass's fine-class learning signal that lies in the
    blocked superclass directions; y = change in fresh-read-out fine-class accuracy against sham, epochs 1-3, mean over
    five seeds; filled blue, superclass block; open grey, random-direction block at the same x."""
    d = pd.read_csv(RESULTS / "GRADBLOCK" / "superclass_gradblock.csv")
    d = d[d.epoch.isin([1, 2, 3])]
    m = d.groupby("superclass")[["blocked_share", "fresh_fine_gradsc_minus_sham", "fresh_fine_gradrand_minus_sham"]].mean()
    x = 100 * m.blocked_share
    rs, ps = stats.spearmanr(m.blocked_share, m.fresh_fine_gradsc_minus_sham)
    rr, pr = stats.spearmanr(m.blocked_share, m.fresh_fine_gradrand_minus_sham)
    ax.axhline(0, color=REF, lw=0.6, ls="--", zorder=0)
    ax.scatter(x, m.fresh_fine_gradrand_minus_sham, s=12, facecolor="white", edgecolor="#9CA3AF", lw=0.7, zorder=2,
               label="random-direction block")
    ax.scatter(x, m.fresh_fine_gradsc_minus_sham, s=14, color=ns.INDEX["SSI"], lw=0, zorder=3,
               label="superclass block")
    b = np.polyfit(x, m.fresh_fine_gradsc_minus_sham, 1)
    xx = np.linspace(x.min(), x.max(), 10)
    ax.plot(xx, np.polyval(b, xx), color=ns.INDEX["SSI"], lw=0.9, zorder=2)
    ax.text(1.03, 0.62, f"superclass block:\nρ = {rs:.2f} (P = {ps:.3f})\n\nrandom block:\nρ = {rr:.2f} (P = {pr:.2f})",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.3, color=ERR, linespacing=1.15)
    ax.set_xlabel("Learning signal blocked (% of the superclass’s)", labelpad=1.5)
    ax.set_ylabel("Fine-class change vs\nsham, epochs 1–3 (pp)")
    ax.set_title("Superclasses that lose more learning signal fall further behind", fontsize=7, pad=3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.02), fontsize=6, frameon=False, handletextpad=0.3,
              borderaxespad=0.2, labelspacing=0.2)
    ax.set_xticks([20, 25, 30, 35, 40])
    rows = m.reset_index().rename(columns={"blocked_share": "blocked_share",
                                           "fresh_fine_gradsc_minus_sham": "superclass_block_minus_sham_pp",
                                           "fresh_fine_gradrand_minus_sham": "random_block_minus_sham_pp"})
    rows["spearman_superclass"], rows["p_superclass"], rows["spearman_random"], rows["p_random"] = rs, ps, rr, pr
    ns.source_data(rows, name, letter)
    print(f"o dose-response: superclass rho {rs:.3f} p {ps:.4f}; random rho {rr:.3f} p {pr:.3f}")


def letter(ax, k, x_mm, y_mm, ax_x, ax_top, w, h):
    """Panel letter at an absolute position (mm from the top-left corner)."""
    ns.panel(ax, k, x=(x_mm - ax_x) / w, y=1 + (ax_top - y_mm) / h)


def main():
    import intervention_panels
    ns.apply()
    crosscheck()
    fig = plt.figure(figsize=(W_MM * ns.MM, H_MM * ns.MM))
    # row 1, the network uses the superclass directions (suppression, last block, epochs 1-10): a | b | c
    T1, H1 = 14, 28
    a = mm_axes(fig, 16, T1, 26, H1)
    intervention_panels.panel_supp_cost(a, name=FIG, letter="a")
    b = [mm_axes(fig, 58, T1, 62, H1)]
    panel_b(b, deltas(), name=FIG, letter="b")
    lg = b[0].get_legend()
    hs, ls_ = lg.legend_handles, [t.get_text() for t in lg.get_texts()]
    lg.remove()
    b[0].legend(hs, ls_, loc="lower right", ncol=1, handlelength=2.3, handletextpad=0.4, borderaxespad=0.2,
                fontsize=6.0, labelspacing=0.15, frameon=False)
    c = mm_axes(fig, 138, T1, 40, H1)
    intervention_panels.panel_supp_where(c, name=FIG, letter="c")
    letter(a, "a", 1, T1 - 10, 16, T1, 26, H1)
    letter(b[0], "b", 45, T1 - 10, 58, T1, 62, H1)
    letter(c, "c", 124, T1 - 10, 138, T1, 40, H1)
    # row 2, the representation keeps the information: d (fresh read-out, unprojected) | e (rebuilding)
    T2, H2 = 58, 26
    d = mm_axes(fig, 16, T2, 44, H2)
    panel_fresh(d, name=FIG, letter="d")
    e = [mm_axes(fig, 84, T2, 94, H2)]
    panel_d(e, name=FIG, letter="e")
    letter(d, "d", 1, T2 - 10, 16, T2, 44, H2)
    letter(e[0], "e", 70, T2 - 10, 84, T2, 94, H2)
    # row 3, withholding only the learning signal: f (where the deficit lies) | g (random grouping) | h (stage-matched)
    # | i (learning signal of misrouted images)
    T3, H3 = 103, 28
    D = intervention_panels.gb_long()
    # f: superclass assignment (top) and fine-class accuracy (bottom), every arm
    fa = mm_axes(fig, 15, T3 - 2, 34, 14)
    fb = mm_axes(fig, 15, T3 + 14, 34, 14)
    arms = [("sham", "#9CA3AF", "-", 0.8), ("gradjoint", "#374151", "-", 1.1), ("gradrand", "#374151", ":", 1.1),
            ("gradrandfresh", "#374151", "-.", 0.8), ("gradsc", "#374151", "--", 1.5)]
    rowsf = []
    for ax_, col_, hue, lab_, yl in ((fa, "fresh_assignment", SSI, "Superclass\nassignment (%)", (15, 72)),
                                     (fb, "val_accuracy", ACC_HUE, "Fine-class\naccuracy (%)", (5, 62))):
        ax_.axvspan(0.5, 10.5, color="#F3F4F6", lw=0, zorder=0)
        for arm, c_, ls_, lw_ in arms:
            P = D[D.arm == arm].pivot(index="seed", columns="epoch", values=col_).dropna(axis=1)
            col = c_ if arm == "sham" else hue
            ax_.plot(P.columns, P.values.mean(0), color=col, ls=ls_, lw=lw_, zorder=3)
            rowsf += [dict(measure=col_, arm=arm, epoch=int(e), mean=v) for e, v in zip(P.columns, P.values.mean(0))]
        ax_.set_xlim(0.5, 30.5)
        ax_.set_ylim(*yl)
        ax_.set_ylabel(lab_, fontsize=6.3, labelpad=1)
        ax_.tick_params(labelsize=6)
    fa.tick_params(labelbottom=False)
    fa.set_xticks([1, 10, 20, 30])
    fb.set_xticks([1, 10, 20, 30])
    fb.set_xlabel("Epoch", labelpad=1)
    fa.set_title("")
    from matplotlib.lines import Line2D as _L2
    fa.legend(handles=[_L2([], [], color="#374151", ls="--", lw=1.4, label="superclass"),
                       _L2([], [], color="#374151", ls="-", lw=1.1, label="jointly matched"),
                       _L2([], [], color="#374151", ls=":", lw=1.1, label="random"),
                       _L2([], [], color="#374151", ls="-.", lw=0.8, label="random, redrawn"),
                       _L2([], [], color="#9CA3AF", ls="-", lw=0.8, label="sham")],
              loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=3, fontsize=6, frameon=False, handlelength=1.2,
              labelspacing=0.1, columnspacing=0.5, handletextpad=0.3, borderaxespad=0.1)
    ns.source_data(pd.DataFrame(rowsf), FIG, "f")
    f = fa
    g = mm_axes(fig, 62, T3, 38, H3)
    panel_rg_block(g, name=FIG, letter="g")
    h = mm_axes(fig, 116, T3, 20, H3)
    panel_g(h, name=FIG, letter="h")
    h.set_xticklabels(["super-\nclass", "class"], fontsize=6.0)
    for t, col in zip(h.get_xticklabels(), (SSI, "#B45309")):
        t.set_color(col)
    h.set_title("Blocking at layer3 to\npooled layer, matched", fontsize=7, pad=3)
    i = mm_axes(fig, 152, T3, 26, H3)
    intervention_panels.panel_sr_i(i, name=FIG, letter="i")
    i.set_xticklabels(["superclass", "random\ngrouping"], fontsize=6.0)
    i.set_title("Misrouted images keep\nthe learning signal", fontsize=7, pad=3)
    letter(f, "f", 1, T3 - 13, 15, T3 - 2, 34, 14)
    letter(g, "g", 48, T3 - 11, 62, T3, 38, H3)
    letter(h, "h", 102, T3 - 11, 116, T3, 20, H3)
    letter(i, "i", 138, T3 - 11, 152, T3, 26, H3)
    # row 4, the coarse structure the directions hold: j (trained networks) | k (random groupings) | l (dose)
    T4, H4 = 151, 26
    j = mm_axes(fig, 19, T4, 32, H4)
    jm = mm_axes(fig, 19, T4 + 29, 32, 9)
    panel_a(j, jm, name=FIG, letter="j")
    k = mm_axes(fig, 86, T4, 56, H4)
    panel_e(k, name=FIG, letter="k")
    l_ = mm_axes(fig, 156, T4, 22, H4)
    panel_f(l_, name=FIG, letter="l")
    letter(j, "j", 1, T4 - 11, 19, T4, 32, H4)
    letter(k, "k", 72, T4 - 11, 86, T4, 56, H4)
    letter(l_, "l", 142, T4 - 11, 156, T4, 22, H4)
    # row 5, removing one superclass's superclass-selective features, three architectures: m
    T5, SUP = 208, 15.5
    m_ = mm_axes(fig, 17, T5, 68, SUP)
    intervention_panels.panel_removal(m_, name=FIG, letter="m")
    leg = m_.get_legend()
    handles, labels = leg.legend_handles, [t.get_text() for t in leg.get_texts()]
    leg.remove()
    m_.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.03, 1.0), handlelength=1.3, handleheight=0.9,
              borderaxespad=0.1, labelspacing=0.3, fontsize=6.5, frameon=False)
    top = max(p.get_height() for p in m_.patches if p.get_height() > 0)
    m_.set_ylim(-2, top + 8)
    m_.set_title("Removing one superclass’s superclass-selective features", fontsize=7, pad=3)
    letter(m_, "m", 1, T5 - 8, 17, T5, 68, SUP)
    # n: learning signal shared within superclasses under each block
    n_ = mm_axes(fig, 152, T5 - 7, 26, 22.5)
    intervention_panels.panel_f_sharing(n_, name=FIG, letter="n")
    n_.set_title("Blocking keeps the\nlearning signal shared", fontsize=7, pad=3)
    n_.set_ylabel("Learning signal shared\nwithin superclass", color=intervention_panels.SSI_C, fontsize=6.5)
    n_.set_ylim(-0.005, 0.23)
    n_.set_yticks([0, 0.06, 0.12])
    n_.legend(loc="upper left", fontsize=6, frameon=False, handlelength=1.8, borderaxespad=0.1, labelspacing=0.05)
    letter(n_, "n", 138, T5 - 15, 152, T5 - 7, 26, 22.5)
    # row 6, per-superclass dose-response of acquisition blocking: o
    T6, H6 = 242, 22
    o_ = mm_axes(fig, 20, T6, 70, H6)
    panel_dose(o_)
    letter(o_, "o", 1, T6 - 8, 20, T6, 70, H6)
    for fpath in (ns.FINAL_OUT / "source_data").glob(f"{FIG}_*.csv"):
        if fpath.stem.removeprefix(f"{FIG}_")[0] not in "abcdefghijklmno":
            fpath.unlink()
    out = ns.save_final(fig, FIG, W_MM, ed=True)
    print("saved", out)


if __name__ == "__main__":
    import os as _os
    if not _os.environ.get("ES_FIG_OUT"):
        raise SystemExit("suppression_panels.py saves under the name ed8 (Extended Data Fig. 8 is drawn by interventions/ed8.py); "
                         "set ES_FIG_OUT to a separate output folder to draw this stand-alone composite.")
    main()
