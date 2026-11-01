#!/usr/bin/env python3
"""Asymmetry between coarse and fine learning under blocking, with RDT measures on both levels, and its direction
in time.

A  Epochs behind no block at epoch 10, per block and seed, with one instrument (RDT) on both levels.
     coarse   cumulative superclass events (Ab-E, all layers, common thresholds; as Fig. 5c); check: superclass-selective
              features (n_sc)
     fine     class features confined to their superclass (n_cl_confined; rdt_pace.census definition);
              also all class-selective features (n_cl), cumulative class events (Di-E), fine-class accuracy
   Arms: gradsc, gradrg, gradjoint, gradenergy, gradrand against sham. Epochs behind = rdt_pace.behind (running
   maximum of the control, linear interpolation, first crossing).
B  Per-epoch lag time courses (epochs 1-15) for gradsc, gradjoint, gradenergy: coarse (cumulative superclass events)
   and fine (fine-class accuracy; confined class features). Per seed: first epoch with lag > 1, and the lead-lag
   cross-correlation of per-epoch lag increments over epochs 2-10 (k > 0 = coarse leads by k epochs).
C1 Label noise within superclasses against clean training: RDT superclass measures (rdt_noise_contrasts.csv and
   RDT_pace/noise_long.csv), fine-class accuracy lag at epoch 10 (PLACEMENT_curve).
C2 Release surge of superclass events at epoch 11 (cross-checked against rdt_completion_long.csv).

Data: lineage_lib cache (gradenergy and gradrand built with lineage/extract.py), RDT_completion, PLACEMENT_curve.
Writes analysis_outputs/paper/ASYMMETRY_RDT/{census_long.csv, long.csv, summary.json} and, with --figure, a panel.
Usage: python scripts/paper/asymmetry_rdt.py [--figure]
"""
import argparse
import json
import os
import sys
from pathlib import Path
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lineage"))
import lineage_lib as X  # noqa: E402
from rdt_pace import behind, census  # noqa: E402

RESULTS = X.REPO / "analysis_outputs/paper"
OUT = RESULTS / "ASYMMETRY_RDT"
SEEDS = X.SEEDS
BLOCKS = ["gradsc", "gradrg", "gradjoint", "gradenergy", "gradrand"]
LABEL = {"gradsc": "coarse block", "gradrg": "random grouping", "gradjoint": "matched block",
         "gradenergy": "energy-matched", "gradrand": "random directions", "sham": "no block"}
MEASURES = {"cum_AbE": "cumulative superclass events (RDT)", "n_sc": "superclass-selective features (RDT)",
            "n_cl_confined": "confined class features (RDT)", "n_cl": "class-selective features (RDT)",
            "cum_DiE": "cumulative class events (RDT)", "fine_acc": "fine-class accuracy"}
EPS = np.arange(16)


# data
def census_table():
    f = OUT / "census_long.csv"
    if f.exists():
        return pd.read_csv(f)
    rows = []
    for s in SEEDS:
        t = X.common_thresholds(s)
        for arm in ["sham"] + BLOCKS:
            cache = X.Cache(arm, s)
            for e in X.EPOCHS:
                n_sc, n_cl, n_conf, _ = census(cache, e, t)
                rows.append({"arm": arm, "seed": s, "epoch": e, "n_sc": n_sc, "n_cl": n_cl, "n_cl_confined": n_conf})
            print("census", arm, s, rows[-1], flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(f, index=False)
    return D


def events_table():
    R = pd.read_csv(RESULTS / "RDT_completion/rdt_completion_long.csv")
    q = R[(R.kind == "events") & (R.layer == "all") & (R.tset == "common") & R.measure.isin(["Ab-E", "Di-E"])]
    rows = []
    for (arm, s, m), g in q.groupby(["arm", "seed", "measure"]):
        v = g.set_index("epoch").value.reindex(range(1, 16)).fillna(0).values
        rows += [{"arm": arm, "seed": int(s), "epoch": e, f"cum_{m.replace('-', '')}": c, f"ev_{m.replace('-', '')}": d}
                 for e, c, d in zip(EPS, np.r_[0, np.cumsum(v)], np.r_[0, v])]
    E = pd.DataFrame(rows)
    return E.groupby(["arm", "seed", "epoch"]).first().reset_index()


def accuracy_table():
    P = pd.read_csv(RESULTS / "PLACEMENT_curve/long.csv")
    P = P[P.family == "block"][["arm", "seed", "epoch", "fine", "placement"]]
    return P.rename(columns={"fine": "fine_acc", "placement": "sc_acc"})


def build():
    D = census_table().merge(events_table(), on=["arm", "seed", "epoch"], how="left")
    D = D.merge(accuracy_table(), on=["arm", "seed", "epoch"], how="left")
    return D[D.arm.isin(["sham"] + BLOCKS)].sort_values(["arm", "seed", "epoch"]).reset_index(drop=True)


def validate(D):
    """Census against RDT_pace/long.csv (same code, 4 arms) and the epoch-10 coarse lag against Fig. 5c."""
    P = pd.read_csv(RESULTS / "RDT_pace/long.csv")
    m = P.merge(D, on=["arm", "seed", "epoch"], suffixes=("_pace", ""))
    out = {"census_rows_compared": len(m)}
    for c in ("n_sc", "n_cl", "n_cl_confined"):
        out[f"{c}_mismatches"] = int((m[f"{c}_pace"] != m[c]).sum())
    out["cum_AbE_mismatches"] = int((m["cum_AbE_pace"] != m["cum_AbE"]).sum())
    F = pd.read_csv(RESULTS.parents[1] / "reproduce/out/source_data/fig5_c.csv")
    F = F[F.seed != "mean"].copy(); F["seed"] = F.seed.astype(int)
    name = {v: k for k, v in LABEL.items()}
    diffs = []
    for _, r in F.iterrows():
        arm = name[r.manipulation]
        a, c = (D[(D.arm == x) & (D.seed == r.seed)].sort_values("epoch") for x in (arm, "sham"))
        diffs.append(abs(behind(a.cum_AbE.values, c.cum_AbE.values)[10] - r.coarse_lag_epochs))
        diffs.append(abs(behind(a.fine_acc.values, c.fine_acc.values)[10] - r.fine_lag_epochs))
    out["fig5c_max_abs_diff_epochs"] = float(np.nanmax(diffs))
    return out


def lag_table(D):
    rows = []
    for arm in BLOCKS:
        for s in SEEDS:
            a = D[(D.arm == arm) & (D.seed == s)].sort_values("epoch")
            c = D[(D.arm == "sham") & (D.seed == s)].sort_values("epoch")
            for m in MEASURES:
                for e, v in zip(EPS, behind(a[m].values, c[m].values)):
                    rows.append({"analysis": "lag", "arm": arm, "arm_label": LABEL[arm], "seed": s, "measure": m,
                                 "measure_label": MEASURES[m], "epoch": int(e), "epochs_behind_no_block": v,
                                 "value_arm": float(a[m].values[e]), "value_no_block": float(c[m].values[e])})
    return pd.DataFrame(rows)


# A
def analysis_a(Lg):
    res, tab = {}, []
    e10 = Lg[Lg.epoch == 10]
    for arm in BLOCKS:
        res[arm] = {}
        g = e10[e10.arm == arm].set_index(["measure", "seed"]).epochs_behind_no_block
        for m in MEASURES:
            r = X.tci(g.loc[m].reindex(SEEDS).values)
            r["per_seed"] = {str(s): round(float(g.loc[(m, s)]), 3) for s in SEEDS}
            res[arm][m] = r
        # crossing within arm: fine (confined) - coarse (events), and fine accuracy - coarse
        for fm in ("n_cl_confined", "fine_acc", "n_cl"):
            d = g.loc[fm].reindex(SEEDS).values - g.loc["cum_AbE"].reindex(SEEDS).values
            res[arm][f"{fm} - cum_AbE"] = X.tci(d)
        # level at epoch 10 as a share of no block (per seed), same unit on both levels
        for m in ("cum_AbE", "n_cl_confined", "n_sc", "fine_acc"):
            v = e10[(e10.arm == arm) & (e10.measure == m)].set_index("seed").reindex(SEEDS)
            res[arm][f"pct_of_no_block_e10|{m}"] = X.tci(100 * v.value_arm.values / v.value_no_block.values)
        tab.append({"arm": LABEL[arm], **{m: res[arm][m]["mean"] for m in MEASURES}})
    # between-arm contrast of the fine RDT lag: coarse block vs fine-aimed blocks
    for a, b in (("gradsc", "gradjoint"), ("gradsc", "gradenergy"), ("gradrg", "gradjoint"),
                 ("gradjoint", "gradrand"), ("gradenergy", "gradrand")):
        for m in ("n_cl_confined", "cum_AbE", "fine_acc"):
            d = (e10[(e10.arm == a) & (e10.measure == m)].set_index("seed").epochs_behind_no_block.reindex(SEEDS).values
                 - e10[(e10.arm == b) & (e10.measure == m)].set_index("seed").epochs_behind_no_block.reindex(SEEDS).values)
            res.setdefault("contrasts", {})[f"{a} - {b}|{m}"] = X.tci(d)
    return res, pd.DataFrame(tab)


# B
def first_over(y, eps, thr=1.0):
    for e, v in zip(eps, y):
        if np.isfinite(v) and v > thr:
            return int(e)
    return None


def xcorr(a, b, ks=(-2, -1, 0, 1, 2)):
    """corr(a[t], b[t + k]); k > 0: a leads b by k epochs."""
    out = {}
    for k in ks:
        x, y = (a[:len(a) - k], b[k:]) if k >= 0 else (a[-k:], b[:len(b) + k])
        ok = np.isfinite(x) & np.isfinite(y)
        out[k] = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() >= 4 and x[ok].std() > 0 and y[ok].std() > 0 \
            else float("nan")
    return out


def analysis_b(Lg):
    res = {}
    for arm in ("gradsc", "gradjoint", "gradenergy"):
        r = {"per_seed": {}}
        for s in SEEDS:
            g = Lg[(Lg.arm == arm) & (Lg.seed == s)].pivot(index="epoch", columns="measure",
                                                             values="epochs_behind_no_block")
            ps = {}
            for m in ("cum_AbE", "fine_acc", "n_cl_confined"):
                ps[f"first_epoch_lag_gt1|{m}"] = first_over(g[m].loc[1:15].values, range(1, 16))
                ps[f"lag|{m}"] = [round(float(v), 3) for v in g[m].loc[1:15].values]
            for fm in ("fine_acc", "n_cl_confined"):
                dc = np.diff(g["cum_AbE"].loc[1:10].values)        # increments over epochs 2-10
                df = np.diff(g[fm].loc[1:10].values)
                xc = xcorr(dc, df)
                ps[f"xcorr_increments|cum_AbE->{fm}"] = {str(k): round(v, 3) for k, v in xc.items()}
                fin = {k: v for k, v in xc.items() if np.isfinite(v)}
                ps[f"xcorr_peak_k|cum_AbE->{fm}"] = max(fin, key=fin.get) if fin else None
            r["per_seed"][str(s)] = ps
        for fm in ("fine_acc", "n_cl_confined"):
            fc = [r["per_seed"][str(s)]["first_epoch_lag_gt1|cum_AbE"] for s in SEEDS]
            ff = [r["per_seed"][str(s)][f"first_epoch_lag_gt1|{fm}"] for s in SEEDS]
            r[f"first_gt1|coarse_vs_{fm}"] = {
                "coarse": fc, "fine": ff,
                "n_coarse_first": sum(1 for a, b in zip(fc, ff) if a is not None and (b is None or a < b)),
                "n_same_epoch": sum(1 for a, b in zip(fc, ff) if a is not None and a == b),
                "n_fine_first": sum(1 for a, b in zip(fc, ff) if b is not None and (a is None or b < a)),
                "n_neither": sum(1 for a, b in zip(fc, ff) if a is None and b is None)}
            pk = [r["per_seed"][str(s)][f"xcorr_peak_k|cum_AbE->{fm}"] for s in SEEDS]
            r[f"xcorr_peak_k|cum_AbE->{fm}"] = {"per_seed": pk, "n_k_pos": sum(1 for k in pk if k is not None and k > 0),
                                                 "n_k_0": sum(1 for k in pk if k == 0),
                                                 "n_k_neg": sum(1 for k in pk if k is not None and k < 0)}
        # growth of each lag: least-squares slope (epochs of lag per epoch) over epochs 4-10, per seed
        for m in ("cum_AbE", "fine_acc", "n_cl_confined"):
            sl = []
            for s in SEEDS:
                y = np.array(r["per_seed"][str(s)][f"lag|{m}"][3:10], float)   # epochs 4-10
                x = np.arange(4, 11)
                ok = np.isfinite(y)
                sl.append(float(np.polyfit(x[ok], y[ok], 1)[0]))
            r[f"slope_epochs_4_10|{m}"] = X.tci(sl)
        for fm in ("fine_acc", "n_cl_confined"):
            d = [r["per_seed"][str(s)]["lag|cum_AbE"][1] - r["per_seed"][str(s)][f"lag|{fm}"][1] for s in SEEDS]
            r[f"epoch2 lag cum_AbE - {fm}"] = X.tci(d)
        # mean lag courses
        m = Lg[(Lg.arm == arm) & Lg.epoch.between(1, 15)].groupby(["measure", "epoch"]).epochs_behind_no_block.mean()
        r["mean_lag"] = {mm: {int(e): round(float(v), 3) for e, v in m.loc[mm].items()}
                         for mm in ("cum_AbE", "n_sc", "fine_acc", "n_cl_confined", "n_cl")}
        res[arm] = r
    return res


# C1 / C2
def check_c1():
    C = pd.read_csv(RESULTS / "RDT_completion/noise/rdt_noise_contrasts.csv")
    out = {}
    for contrast in ("within - clean", "between - within"):
        for kind, win, m in (("events", "early (0-10)", "Ab-E"), ("events", "early (0-10)", "Di-E"),
                             ("census", "early (5-10)", "pct_sc"), ("events", "all (0-30)", "Ab-E"),
                             ("census", "all (5-30)", "pct_sc")):
            r = C[(C.tset == "common") & (C.kind == kind) & (C.window == win) & (C.measure == m) &
                  (C.contrast == contrast)].iloc[0]
            out[f"{contrast}|{m}|{win}"] = {"mean": r["mean"], "ci95": json.loads(r.ci95), "n_pos": int(r.n_pos),
                                            "n_neg": int(r.n_neg), "pct_change_mean": r.pct_change_mean,
                                            "mean_arm": r.mean_a, "mean_ref": r.mean_b,
                                            "per_seed": json.loads(r.per_seed), "p_two_sided": r.p_two_sided}
    # superclass-selective feature counts (census layers, common thresholds): RDT_pace/noise_long.csv and
    # rdt_noise_long.csv (n_sc, all layers), epochs 5 and 10
    N = pd.read_csv(RESULTS / "RDT_pace/noise_long.csv")
    L = pd.read_csv(RESULTS / "RDT_completion/noise/rdt_noise_long.csv")
    L = L[(L.kind == "census") & (L.tset == "common") & (L.measure == "n_sc")]
    for a, b in (("within", "clean"), ("between", "within")):
        for src, T, col in (("RDT_pace census layers", N, "n_sc"), ("rdt_noise_long", L, "value")):
            for ep in ((5,), (10,), (5, 10)):
                va = T[(T.arm == a) & T.epoch.isin(ep)].groupby("seed")[col].mean().reindex(SEEDS)
                vb = T[(T.arm == b) & T.epoch.isin(ep)].groupby("seed")[col].mean().reindex(SEEDS)
                r = X.tci((va - vb).values)
                r.update(mean_arm=float(va.mean()), mean_ref=float(vb.mean()))
                out[f"{a} - {b}|n_sc ({src})|epochs {ep}"] = r
        for ep in ((5,), (10,)):
            va = N[(N.arm == a) & N.epoch.isin(ep)].groupby("seed").n_cl_confined.mean().reindex(SEEDS)
            vb = N[(N.arm == b) & N.epoch.isin(ep)].groupby("seed").n_cl_confined.mean().reindex(SEEDS)
            out[f"{a} - {b}|n_cl_confined (RDT_pace)|epochs {ep}"] = {**X.tci((va - vb).values),
                                                                     "mean_arm": float(va.mean()),
                                                                     "mean_ref": float(vb.mean())}
    # fine-class accuracy lag at epoch 10 (PLACEMENT_curve noise family, epochs 0, 5, ..., 30)
    P = pd.read_csv(RESULTS / "PLACEMENT_curve/long.csv")
    N7 = [0, 5, 10, 15, 20, 25, 30]

    def lag(arm, ctrl, m):
        v = []
        for s in SEEDS:
            a = P[(P.family == "noise") & (P.arm == arm) & (P.seed == s)].set_index("epoch").loc[N7, m].values
            c = np.maximum.accumulate(P[(P.family == "noise") & (P.arm == ctrl) & (P.seed == s)]
                                      .set_index("epoch").loc[N7, m].values)
            v.append(10 - float(np.interp(a[2], c, N7)) if a[2] <= c[-1] else np.nan)
        return v

    for a, b in (("within", "clean"), ("between", "within")):
        for m, name in (("fine", "fine-class accuracy"), ("placement", "superclass accuracy")):
            out[f"{a} - {b}|{name} epochs behind at epoch 10 (PLACEMENT_curve)"] = X.tci(lag(a, b, m))
    # per-epoch validation curves (epochs 0-30) for the same lag, from the lanes (read-only)
    try:
        sys.path.insert(0, str(HERE / "figures"))
        import noise_lib as B  # noqa: E402
        ssd = Path(RDT_DATA + "/extended/experiments")
        lanes = {"clean": "standard_resnet18", "within": "within_sc_resnet18_throughout",
                 "between": "between_sc_resnet18_throughout"}
        for a, b in (("within", "clean"), ("between", "within")):
            v = []
            for s in SEEDS:
                ya = B._val_curve(ssd / f"static_noise_s{s}/lanes/{lanes[a]}")
                yb = B._val_curve(ssd / f"static_noise_s{s}/lanes/{lanes[b]}")
                v.append(behind(ya[:16], yb[:16])[10])
            out[f"{a} - {b}|fine-class accuracy epochs behind at epoch 10 (per-epoch val curves, rdt_pace.behind)"] = \
                X.tci(v)
    except Exception as ex:  # noqa: BLE001
        out["per_epoch_val_curves_error"] = repr(ex)
    return out


def check_c2(D):
    H = pd.read_csv(RESULTS.parents[1] / "reproduce/out/source_data/ed7_h.csv")
    out = {}
    for arm in ("gradsc", "gradjoint"):
        for lev, proc in (("superclass", "Ab-E"), ("class", "Di-E")):
            h = H[(H.arm == arm) & (H.process == proc)]
            if not len(h):
                continue
            r11 = h[h.epoch == 11].set_index("seed")
            r10 = h[h.epoch == 10].set_index("seed")
            out[f"{arm}|{proc}|ed7_h"] = {
                "events_e10": r10.events.reindex(SEEDS).tolist(), "events_e11": r11.events.reindex(SEEDS).tolist(),
                "ratio_e11_to_e10": (r11.events / r10.events).reindex(SEEDS).round(3).tolist(),
                "ratio_e11_to_mean_e6_10": r11.ratio_to_block_level.reindex(SEEDS).round(3).tolist(),
                "onset_epoch": r11.onset_epoch.reindex(SEEDS).tolist()}
        # cross-check from rdt_completion_long (all layers, common thresholds)
        for proc in ("AbE", "DiE"):
            r10, r11, r610 = [], [], []
            for s in SEEDS:
                g = D[(D.arm == arm) & (D.seed == s)].set_index("epoch")[f"ev_{proc}"]
                r10.append(float(g[11] / g[10])); r610.append(float(g[11] / g.loc[6:10].mean()))
            out[f"{arm}|{proc}|rdt_completion_long"] = {"ratio_e11_to_e10": np.round(r10, 3).tolist(),
                                                        "ratio_e11_to_mean_e6_10": np.round(r610, 3).tolist(),
                                                        "range_e11_to_mean_e6_10": [round(min(r610), 2),
                                                                                    round(max(r610), 2)]}
    for s in SEEDS:   # sham for reference
        g = D[(D.arm == "sham") & (D.seed == s)].set_index("epoch")["ev_AbE"]
        out.setdefault("sham|AbE|ratio_e11_to_mean_e6_10", []).append(round(float(g[11] / g.loc[6:10].mean()), 3))
    return out


# figure
def figure(Lg):
    sys.path.insert(0, str(HERE / "figures"))
    import nmi_style as ns  # noqa: E402
    import matplotlib.pyplot as plt  # noqa: E402
    from matplotlib.lines import Line2D  # noqa: E402
    ns.apply()
    BLUE, AMBER, GREYS = ns.BLUE, ns.AMBER, ns.GREYS
    W, H = 180, 58
    fig = plt.figure(figsize=(W * ns.MM, H * ns.MM))
    mm = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
    a = mm(14, 13, 70, 36)
    e10 = Lg[Lg.epoch == 10]
    arms = ["gradsc", "gradrg", "gradjoint", "gradenergy", "gradrand"]
    gap, step = 0.5, 1.25
    tr = a.get_xaxis_transform()
    for k, arm in enumerate(arms):
        x0 = k * step
        c = e10[(e10.arm == arm) & (e10.measure == "cum_AbE")].set_index("seed").epochs_behind_no_block.reindex(SEEDS)
        f = e10[(e10.arm == arm) & (e10.measure == "n_cl_confined")].set_index("seed").epochs_behind_no_block.reindex(SEEDS)
        for cy, fy in zip(c, f):
            a.plot([x0, x0 + gap], [cy, fy], color=GREYS[2], lw=0.5, alpha=0.8, zorder=2)
        a.plot([x0, x0 + gap], [c.mean(), f.mean()], color=GREYS[1], lw=1.6, zorder=3, solid_capstyle="round")
        a.scatter([x0], [c.mean()], s=16, color=BLUE[1], edgecolor="white", lw=0.5, zorder=4)
        a.scatter([x0 + gap], [f.mean()], s=16, color=AMBER[1], edgecolor="white", lw=0.5, zorder=4)
        a.text(x0 + gap / 2, -0.04, LABEL[arm].replace(" ", "\n", 1).replace("-", "-\n", 1), fontsize=6, ha="center", va="top",
               color=GREYS[0], linespacing=0.95, transform=tr)
    a.axhline(0, color="#D1D5DB", lw=0.5, zorder=0)
    a.set_xlim(-0.25, (len(arms) - 1) * step + gap + 0.25); a.set_xticks([])
    a.spines["bottom"].set_visible(False)
    a.set_ylabel("Epochs behind no block,\nat epoch 10")
    a.legend(handles=[Line2D([], [], marker="o", ls="none", ms=4, color=BLUE[1], label="superclass events"),
                      Line2D([], [], marker="o", ls="none", ms=4, color=AMBER[1], label="confined class features")],
             fontsize=6, frameon=False, loc="upper right", handletextpad=0.1, handlelength=1.0, labelspacing=0.3)
    sty = {"gradsc": (0, "-", 1.7), "gradjoint": (1, "--", 1.4), "gradenergy": (2, ":", 1.5)}
    for j, (m, ramp, lab) in enumerate((("cum_AbE", BLUE, "Superclass events,\nepochs behind no block"),
                                        ("n_cl_confined", AMBER, "Confined class features,\nepochs behind no block"))):
        ax = mm(98 + j * 47, 13, 31, 36)
        for arm, (ci, ls, lw) in sty.items():
            Y = np.array([Lg[(Lg.arm == arm) & (Lg.seed == s) & (Lg.measure == m)].sort_values("epoch")
                          .epochs_behind_no_block.values for s in SEEDS])
            for y in Y:
                ax.plot(EPS, y, color=ramp[ci], lw=0.45, alpha=0.3, zorder=2)
            ax.plot(EPS, Y.mean(0), color=ramp[ci], lw=lw, ls=ls, label=LABEL[arm], zorder=3,
                    dash_capstyle="round" if ls == ":" else "butt")
        ax.axvspan(0.5, 10.5, color="#F3F4F6", zorder=0, lw=0)
        ax.axhline(0, color=GREYS[2], lw=0.6, zorder=1)
        ax.set_xlim(1, 15); ax.set_xticks([1, 5, 10, 15]); ax.set_xlabel("Epoch", labelpad=1.5)
        ax.set_ylim(-3, 12.5); ax.set_yticks([0, 5, 10])
        ax.set_ylabel(lab)
        if j == 0:
            ax.legend(fontsize=6, frameon=False, loc="upper left", handlelength=2.2, borderaxespad=0.1,
                      labelspacing=0.25)
    for letter, x in (("a", 0.5), ("b", 86), ("c", 133)):
        fig.text(x / W, (H - 1.5) / H, letter, fontsize=8, fontweight="bold", va="top", ha="left")
    ns.check_text_sizes(fig)
    p = OUT / "asymmetry_rdt.png"
    fig.savefig(p, dpi=300)
    print("saved", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    D = build()
    Lg = lag_table(D)
    res = {"validation": validate(D)}
    res["A"], tab = analysis_a(Lg)
    res["B"] = analysis_b(Lg)
    res["C1"] = check_c1()
    res["C2"] = check_c2(D)
    Lg.to_csv(OUT / "long.csv", index=False)
    json.dump(res, open(OUT / "summary.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(res["validation"]))
    print(tab.round(2).to_string(index=False))
    if args.figure:
        figure(Lg)


if __name__ == "__main__":
    main()
