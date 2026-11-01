"""Shared loaders and drawing helpers for the label-noise figures (Extended Data Fig. 8, Supplementary Fig. 3) and
other builders.

Data (all read-only):
  fixed-label experiment (labels corrupted once and kept fixed; 'static noise'):
    * superclass probe        analysis_outputs/paper/STATIC_probe/static_probe.csv (super20, epochs 0, 5, ..., 30)
    * fine-class accuracy     raw lanes static_noise_s{seed}/lanes/{standard_resnet18, *_throughout}
                              (devtrain_metrics.json; epoch 0 from devtrain_snapshots.json)
    * exclusive SAE census    analysis_outputs/paper/SAE_outcomes/cache/EN (avgpool SAE, common thresholds of the clean
                              run of each seed): superclass-selective = SSI above AND CSI below threshold;
                              class-selective = CSI above threshold
    * transfer                analysis_outputs/paper/TRANSFER/T1/networks.csv (ANCOVA)
  labels redrawn every epoch (7 conditions):
    * probe, accuracy         analysis_outputs/paper/NOISE_scaffold/per_lane_metrics.csv (epochs 0-30)
    * process fractions       analysis_outputs/paper/NOISE_scaffold/die_fraction_per_lane.csv,
                              die_trajectory_per_lane.csv (without the final, duplicated 6->7 transition)
    * alive SAE features      sae_results.json['feature_landscape'] of each lane (data drive), last real
                              checkpoint (6 = epoch 30; checkpoint 7 re-captures the same network)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nmi_style as ns  # noqa: E402
from fig_common import RESULTS, SEEDS, RUNS_EXTENDED, lane_done  # noqa: E402

A4 = RESULTS / "NOISE_scaffold"
NOISE_LANES = Path(RDT_DATA + "/core/experiments/"
                   "0c28799c-ec9d-4e45-ac30-302dff810a8d/lanes")
EXCLUDED_LANE = ("within_sc_p01", 137)  # stored SAE decompositions of checkpoints 4-7 do not match the run

DOT = "#9CA3AF"
ERR = "#374151"
ZERO = "#6B7280"
WINDOW = "#F3F4F6"
MINUS = "−"


def mstr(s):
    return s.replace("-", MINUS)


def fmt(v, nd=1, sign=True):
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s.replace("-", MINUS)


def t_ci(x, axis=0):
    """Mean and t-based 95% CI along axis (n - 1 df)."""
    x = np.asarray(x, float)
    n = x.shape[axis]
    m = x.mean(axis)
    h = stats.t.ppf(0.975, n - 1) * x.std(axis, ddof=1) / math.sqrt(n)
    return m, m - h, m + h


def sem(x, axis=0):
    x = np.asarray(x, float)
    return x.std(axis, ddof=1) / math.sqrt(x.shape[axis])


def paired_summary(d):
    d = np.asarray(d, float)
    m, lo, hi = t_ci(d)
    return dict(mean=m, ci_lo=lo, ci_hi=hi, n=len(d), n_neg=int((d < 0).sum()), n_pos=int((d > 0).sum()),
                p=float(stats.ttest_1samp(d, 0).pvalue))


# fixed labels
STATIC_ARMS = ("clean", "within_sc", "between_sc")


def static_lanes(s):
    base = RUNS_EXTENDED / f"static_noise_s{s}" / "lanes"
    return {"clean": base / "standard_resnet18", "between_sc": base / "between_sc_resnet18_throughout",
            "within_sc": base / "within_sc_resnet18_throughout"}


def _val_curve(lane):
    mm = json.loads((lane / "dev_snapshots/metrics/devtrain_metrics.json").read_text())
    c = {int(e["epoch"]): float(e["val_accuracy"]) for e in mm}
    snaps = json.loads((lane / "dev_snapshots/metrics/devtrain_snapshots.json").read_text())
    a0 = [float(z["val_accuracy"]) for z in snaps if z.get("is_t0_baseline") or z.get("epoch") == 0][0]
    return np.array([a0] + [c[e] for e in range(1, 31)])


def static_accuracy():
    """{arm: seeds x 31 array (%)}, epochs 0-30; every lane must be complete."""
    for s in SEEDS:
        for ln in static_lanes(s).values():
            assert lane_done(ln), f"incomplete lane {ln}"
    return {a: np.array([_val_curve(static_lanes(s)[a]) for s in SEEDS]) for a in STATIC_ARMS}


def static_probe():
    """{arm: DataFrame seeds x epochs (%)}, superclass probe (20-way) on the pooled penultimate layer."""
    d = pd.read_csv(RESULTS / "STATIC_probe" / "static_probe.csv")
    out = {}
    for a in STATIC_ARMS:
        p = d[d.arm == a].pivot(index="seed", columns="epoch", values="super20") * 100
        out[a] = p.loc[SEEDS]
    return out


def cross(y, t=30.0):
    """First (linearly interpolated) epoch at which curve y (epochs 0..) reaches t."""
    y = np.asarray(y, float)
    i = int(np.argmax(y >= t))
    assert y[i] >= t and i > 0
    return i - 1 + (t - y[i - 1]) / (y[i] - y[i - 1])


def fixed_label_effects(P, A, tag="fixed labels"):
    """Between - within effects of the fixed-label experiment (paired over seeds, t-based 95% CI).
    Used by routing/noise_routing_panels.py (Extended Data Fig. 8)."""
    Dp = P["between_sc"][30].values - P["within_sc"][30].values
    Dacc = A["between_sc"] - A["within_sc"]
    D010 = np.array([np.trapezoid(r[:11], dx=1.0) / 10 for r in Dacc])
    D30 = Dacc[:, 30]
    delay = np.array([cross(A["between_sc"][i]) - cross(A["within_sc"][i]) for i in range(len(SEEDS))])
    out = {}
    for key, unit, v in (("probe_epoch30", "pp", Dp), ("fine_acc_epochs0_10", "pp", D010),
                         ("fine_acc_epoch30", "pp", D30), ("delay_to_30pct_accuracy", "epochs", delay)):
        m, lo, hi = t_ci(v)
        out[key] = dict(measure=key, unit=unit, mean=m, ci_lo=lo, ci_hi=hi, n_seeds_below_zero=int((v < 0).sum()),
                        **{f"seed_{s}": v[i] for i, s in enumerate(SEEDS)})
        print(f"[{tag}] {key}: {m:+.2f} [{lo:+.2f}, {hi:+.2f}] {unit}, {int((v < 0).sum())}/5 below zero")
    return out


def exclusive_census_EN():
    """Exclusive census of the fixed-label experiment from the SAE_outcomes cache (common thresholds, avgpool SAE).
    Rows: arm (clean/within/between), seed, epoch, n_alive, superclass_selective (SSI > thr and CSI <= thr),
    superclass_selective_inclusive (SSI > thr), class_selective (CSI > thr); shares in % of alive features."""
    sys.path.insert(0, str(RESULTS.parents[1] / "scripts" / "paper"))
    from sae_outcomes import SITE, exp_spec, lane_key, load_null, passing_lanes
    spec = exp_spec("EN")
    T = json.loads((spec["out"] / "thresholds_EN.json").read_text())
    cdir = spec["out"] / "cache" / "EN"
    rows = []
    for ln in passing_lanes("EN", spec):
        t = T["common"][str(ln["seed"])]["thresholds"]
        nl = load_null(cdir, lane_key(ln))
        z = np.load(cdir / f"{lane_key(ln)}_sel.npz")
        for c in nl["ckpts"]:
            pre = f"{c['label']}|{SITE}"
            alive = z[f"{pre}|cnt"] >= nl["alive_min"]
            hs = z[f"{pre}|feature_ssi"] > t["ssi_abstraction_thresh"]
            hc = z[f"{pre}|feature_csi"] > t["csi_differentiation_thresh"]
            na = max(int(alive.sum()), 1)
            rows.append(dict(arm=ln["arm"], seed=int(ln["seed"]), epoch=int(c["epoch"]), n_alive=int(alive.sum()),
                             superclass_selective=100 * (alive & hs & ~hc).sum() / na,
                             superclass_selective_inclusive=100 * (alive & hs).sum() / na,
                             class_selective=100 * (alive & hc).sum() / na))
    df = pd.DataFrame(rows).sort_values(["arm", "seed", "epoch"]).reset_index(drop=True)
    n = df[df.epoch >= 5].groupby("arm").seed.nunique()
    assert (n == 5).all(), f"exclusive census: seeds per arm {n.to_dict()}"
    return df


CENSUS_LAYERS = ("layer1", "layer2", "layer3", "avgpool")   # all SAE layers except ResNet layer4 (= avgpool)


def exclusive_census_stored(exp="EN", layers=CENSUS_LAYERS):
    """Exclusive census with each run's stored pipeline thresholds, pooled over the census layers.
    Task-general = SAI > t_sai;
    superclass-selective = SSI > t_ssi & CSI <= t_csi & not task-general; class-selective = CSI > t_csi & not
    task-general; superclass_selective_inclusive = SSI > t_ssi & not task-general. Shares in % of alive features
    (active on >= 1% of images)."""
    sys.path.insert(0, str(RESULTS.parents[1] / "scripts" / "paper"))
    from sae_outcomes import exp_spec, lane_key, load_null, passing_lanes
    spec = exp_spec(exp)
    T = json.loads((spec["out"] / f"thresholds_{exp}.json").read_text())
    cdir = spec["out"] / "cache" / exp
    rows = []
    for ln in passing_lanes(exp, spec):
        t = T["stored"][lane_key(ln)]["thresholds"]
        nl = load_null(cdir, lane_key(ln))
        z = np.load(cdir / f"{lane_key(ln)}_sel.npz")
        for c in nl["ckpts"]:
            n = tg = ab = abi = di = 0
            for lay in layers:
                pre = f"{c['label']}|{lay}"
                al = z[f"{pre}|cnt"] >= nl["alive_min"]
                sai = z[f"{pre}|feature_sai"] > t["sai_task_general_thresh"]
                ssi = z[f"{pre}|feature_ssi"] > t["ssi_abstraction_thresh"]
                csi = np.nan_to_num(z[f"{pre}|feature_csi"], nan=-1) > t["csi_differentiation_thresh"]
                n += al.sum()
                tg += (al & sai).sum()
                ab += (al & ssi & ~csi & ~sai).sum()
                abi += (al & ssi & ~sai).sum()
                di += (al & csi & ~sai).sum()
            rows.append(dict(arm=ln["arm"], seed=int(ln["seed"]), epoch=float(c["epoch"]), n_alive=int(n),
                             task_general=100 * tg / n, superclass_selective=100 * ab / n,
                             superclass_selective_inclusive=100 * abi / n, class_selective=100 * di / n,
                             thr_sai=t["sai_task_general_thresh"], thr_ssi=t["ssi_abstraction_thresh"],
                             thr_csi=t["csi_differentiation_thresh"]))
    return pd.DataFrame(rows).sort_values(["arm", "seed", "epoch"]).reset_index(drop=True)


# labels redrawn every epoch
REDRAWN = ["standard", "within_sc_p01", "between_sc_p01", "random_p01", "within_sc_p03", "between_sc_p03",
           "random_p03"]


def redrawn_metric(measure):
    """per_lane_metrics.csv pivot: {condition: DataFrame seeds x epochs} (x100 for probe/accuracy)."""
    pl = pd.read_csv(A4 / "per_lane_metrics.csv")
    pl = pl[pl.milestone <= 6]  # milestone 7 re-captures milestone 6
    out = {}
    for c in REDRAWN:
        p = pl[pl.condition == c].pivot(index="seed", columns="epoch", values=measure) * 100
        out[c] = p.loc[SEEDS]
    return out


def process_fractions():
    """Per-lane Di-E and Ab-E fractions (% of Tg-E + Ab-E + Di-E events, five layers) over the real transitions
    0->1 ... 5->6 (the duplicated 6->7 transition compares identical networks and is left out)."""
    d = pd.read_csv(A4 / "die_fraction_per_lane.csv")
    v = "excl. 6->7 null transition"
    out = d[["lane_id", "condition", "seed"]].copy()
    out["di_frac"] = 100 * d[f"di_frac | {v}"]
    out["ab_frac"] = 100 * d[f"ab_frac | {v}"]
    out["paper_di_frac_incl_6to7"] = 100 * d["di_frac | paper (all 7 transitions, 5 layers)"]
    out["excluded"] = [(c, s) == EXCLUDED_LANE for c, s in zip(out.condition, out.seed)]
    return out


def die_trajectories():
    """Per-lane Di-E fraction per real transition (all five layers)."""
    t = pd.read_csv(A4 / "die_trajectory_per_lane.csv")
    t = t[(t.layers == "all layers") & (t.transition != "6->7")].copy()
    t["di_frac"] *= 100
    t["t_index"] = t.transition.str.split("->").str[0].astype(int)
    return t


def alive_features(cache_csv):
    """Alive SAE features per layer at checkpoints 6 and 7 of the 35 lanes, from sae_results.json on the data
    drive. When the drive is not available the stored Source Data table (cache_csv) is used."""
    reg = pd.read_csv(A4 / "die_fraction_per_lane.csv")[["lane_id", "condition", "seed"]]
    if not NOISE_LANES.exists():
        print(f"[alive] data drive not available; using stored table {cache_csv}")
        return pd.read_csv(cache_csv)
    rows = []
    for _, r in reg.iterrows():
        fl = json.loads((NOISE_LANES / r.lane_id / "sae_analysis" / "sae_results.json").read_text())["feature_landscape"]
        for ck in ("6", "7"):
            for lay, v in fl[ck].items():
                rows.append(dict(lane_id=r.lane_id, condition=r.condition, seed=int(r.seed), checkpoint=int(ck),
                                 epoch=30, layer=lay, n_alive=int(v["n_alive"])))
    return pd.DataFrame(rows)


# transfer
def transfer():
    """Fixed-label networks' transfer to Tiny ImageNet (full) and the ANCOVA adjustment (transfer ~ in-task
    accuracy + condition, all networks with a fine-class head)."""
    import statsmodels.formula.api as smf
    df = pd.read_csv(RESULTS / "TRANSFER" / "T1" / "networks.csv")
    sn = df[df.group == "static_noise"].pivot(index="seed", columns="arm", values="transfer_full").loc[SEEDS]
    ia = df[df.group == "static_noise"].pivot(index="seed", columns="arm", values="intask_acc").loc[SEEDS]
    d = df[(df["head"] == "fine100") & (df.group != "randinit")].copy()
    d = d[~((d.group == "curriculum") & (d.arm == "scratch_E30"))]  # identical checkpoints to static-noise clean
    d["condition"] = d.group + ":" + d.arm
    ref = "static_noise:clean"
    m = smf.ols(f"transfer_full ~ intask_acc + C(condition, Treatment('{ref}'))", data=d).fit()
    L = np.zeros(len(m.params))
    for c, w in (("static_noise:between", 1.0), ("static_noise:within", -1.0)):
        L[list(m.params.index).index(f"C(condition, Treatment('{ref}'))[T.{c}]")] = w
    t = m.t_test(L)
    adj = dict(estimate=float(np.asarray(t.effect).ravel()[0]),
               ci_lo=float(np.asarray(t.conf_int()).ravel()[0]), ci_hi=float(np.asarray(t.conf_int()).ravel()[1]),
               p=float(np.asarray(t.pvalue).ravel()[0]), n_networks=int(len(d)),
               slope_intask=float(m.params["intask_acc"]))
    return sn, ia, adj


# drawing
def contrast_inset(ax, rect, D, col, xticklabels, title, ylim, xlabel=None, seed=0):
    """Contrast bars as an inset: bar = mean (hue of the panel, alpha 0.8), grey seed dots,
    dark 95% CI with caps, dashed zero line. D: seeds x bars."""
    ia = ax.inset_axes(rect)
    m, lo, hi = t_ci(D)
    xs = np.arange(D.shape[1])
    rng = np.random.default_rng(seed)
    ia.bar(xs, m, width=0.62, color=col, alpha=ns.BAR_ALPHA_CORE, lw=0, zorder=1)
    for j in xs:
        ia.scatter(j + rng.uniform(-0.17, 0.17, D.shape[0]), D[:, j], s=2.2, color=DOT, lw=0, zorder=3)
    ia.errorbar(xs, m, yerr=[m - lo, hi - m], fmt="none", ecolor=ERR, elinewidth=0.7, capsize=1.4,
                capthick=0.7, zorder=4)
    ia.axhline(0, color=ZERO, lw=0.5, ls="--", zorder=0)
    ia.set_ylim(*ylim)
    ia.set_xlim(-0.6, len(xs) - 0.4)
    ia.set_xticks(xs)
    ia.set_xticklabels(xticklabels)
    ia.tick_params(labelsize=6, length=1.5, pad=1, width=0.5)
    for s in ("top", "right"):
        ia.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ia.spines[s].set_linewidth(0.5)
    ia.set_title(title, fontsize=6, pad=2)
    if xlabel:
        ia.set_xlabel(xlabel, fontsize=6, labelpad=1)
    ia.patch.set_alpha(0.0)
    return ia, m, lo, hi


def traj(ax, x, Y, col, ls, band="sem", lw=1.0, marker="o", ms=2.4, markevery=None, label=None, zorder=3,
         band_alpha=None, alpha=1.0, mfc=None):
    """Mean line + band (s.e.m. or 95% CI) over seeds. Y: seeds x len(x)."""
    Y = np.asarray(Y, float)
    m = Y.mean(0)
    h = sem(Y) * (stats.t.ppf(0.975, Y.shape[0] - 1) if band == "ci" else 1.0)
    ax.fill_between(x, m - h, m + h, color=col, alpha=ns.BAND_ALPHA_CORE if band_alpha is None else band_alpha,
                    lw=0, zorder=zorder - 1)
    ax.plot(x, m, color=col, ls=ls, lw=lw, marker=marker, ms=ms, markevery=markevery, label=label, zorder=zorder,
            alpha=alpha, mec=col, mfc=col if mfc is None else mfc)
    return m, h
