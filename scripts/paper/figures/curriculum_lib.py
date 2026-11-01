"""Shared loaders and drawing helpers for the curriculum figures (interventions/fig6_panels.py, interventions/ed9.py).

Data loaders reuse curriculum_sweep_panels (sweep table, curriculum lanes, probe table). Nothing in this module writes files;
the figure scripts store their values with ns.source_data.

Encodings:
  sweep (ED 9a): ns.SWEEP gradient + markers by pre-training epochs;
  controlled arms, curves (ED 9): #374151, scratch solid, true superclasses dashed, random grouping dotted;
  controlled arms, bars: scratch open, random grouping hatched, true superclasses filled (alpha 0.8).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nmi_style as ns  # noqa: E402
import curriculum_sweep_panels as f6  # noqa: E402
from fig_common import RESULTS, SEEDS, RUNS_EXTENDED, val_curve  # noqa: E402

CUR = RUNS_EXTENDED / "curriculum" / "lanes"
DARK = ns.REF_DARK            # #374151
BAND = "#6B7280"              # grey trajectory band (alpha 0.2)
BAND_ALPHA = ns.BAND_ALPHA_CORE
BAR_ALPHA = ns.BAR_ALPHA_CORE
HATCH_ALPHA = 0.45
HATCH = "////"
DOT = "#9CA3AF"
ERR = "#374151"
REFLINE = "#6B7280"
WINDOW = "#F3F4F6"
HELD_OUT = (256, 314)

# controlled arms: (label, line style, bar style)
ARMS = ("scratch", "sc_rand", "sc_true")
ARM_LABEL = {"scratch": "Scratch", "sc_rand": "Random grouping", "sc_true": "True superclasses"}
ARM_LS = {"scratch": ns.COND_LS["standard"], "sc_true": ns.COND_LS["between_sc"], "sc_rand": ns.COND_LS["random"]}
ARM_LW = {"scratch": 0.9, "sc_true": 1.0, "sc_rand": 1.0}

# sweep conditions -> nominal pre-training epochs
COND = f6.COND
KNOM = f6.KNOM


def sweep_style(cond):
    """(colour, marker) of a sweep condition."""
    return ns.SWEEP[KNOM[cond]]


def sweep_label(cond):
    k = KNOM[cond]
    return "None" if k == 0 else f"{k}"


# stats
def tci(x):
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / math.sqrt(n)
    return m, m - h, m + h


def paired(x, name):
    """Paired summary row (mean, 95% t-CI, t, p, sign counts) for per-seed differences x."""
    x = np.asarray(x, float)
    m, lo, hi = tci(x)
    t, p = stats.ttest_1samp(x, 0.0)
    return dict(contrast=name, n=len(x), mean=m, ci_lo=lo, ci_hi=hi, sd=x.std(ddof=1), t=t, p_two_sided=p,
                n_pos=int((x > 0).sum()), n_neg=int((x < 0).sum()))


def sem(Y, axis=0):
    Y = np.asarray(Y, float)
    return np.nanstd(Y, axis, ddof=1) / np.sqrt(np.sum(~np.isnan(Y), axis))


def fmt_signed(v, nd=1):
    s = f"{abs(v):.{nd}f}"
    return ("+" if v > 0 else "−" if v < 0 else "") + s


# controlled experiment (curriculum)
def lane_path(name, s):
    return CUR / f"{name}_seed{s}"


def curve(name, s, key="val_accuracy"):
    c = val_curve(lane_path(name, s), key)
    if c is None:
        raise FileNotFoundError(f"curriculum lane missing: {lane_path(name, s)}")
    return c


def fine_val(arm, s, K=20, n_fine=30):
    """Fine-class validation accuracy by fine-class epoch 1..n_fine (arm in ARMS)."""
    name = {"scratch": f"scratch_E{n_fine}", "sc_true": f"sc_true_K{K}", "sc_rand": f"sc_rand_K{K}"}[arm]
    c = curve(name, s)
    off = 0 if arm == "scratch" else K
    return np.array([c[off + e] for e in range(1, n_fine + 1)])


def gap_curve(arm, s):
    """Train - validation accuracy (pp) for every logged epoch of the K20 / E30 lanes."""
    name = {"scratch": "scratch_E30", "sc_true": "sc_true_K20", "sc_rand": "sc_rand_K20"}[arm]
    tr, va = curve(name, s, "train_accuracy"), curve(name, s)
    ep = sorted(va)
    return np.array(ep), np.array([tr[e] - va[e] for e in ep])


def first_epoch(y, t):
    """First fine-class epoch (1-based) at which validation accuracy >= t; NaN if never."""
    hit = np.nonzero(np.asarray(y) >= t)[0]
    return float(hit[0] + 1) if len(hit) else np.nan


# sweep
def sweep_table():
    return f6.sweep_table()


def sweep_curve(cond):
    _, sw = f6.sweep_table()
    return val_curve(f6.SWEEP / sw.loc[cond, "lane_id"])


def std30():
    s30 = pd.read_csv(RESULTS / "CURRICULUM_sweep" / "tables" / "A_std30_summary.csv").set_index("metric")
    tc, _ = f6.sweep_table()
    per = tc[tc.condition == "standard30"][["lane", "final_val_acc", "final_gap_pp"]].copy()
    per["seed"] = per.lane.str.extract(r"seed(\d+)").astype(int)
    return s30, per


# events (common thresholds)
def events_ec():
    d = pd.read_csv(RESULTS / "SAE_outcomes" / "events_EC.csv")
    d = d[(d.tset == "common") & (d.epoch_a != d.epoch_b)].copy()
    d["phase"] = np.where((d.arm != "scratch_E30") & (d.epoch_b <= 20), "pre", "fine")
    return d


def events_stored():
    """Curriculum events with each run's stored pipeline thresholds, all five SAE layers, raw counts
    (sae_results.json process_intensity), no same-data null; duplicate final milestone dropped. A transition of a
    curriculum arm is 'pre' if its end epoch <= 20, else 'fine' (the 20->21 switch transition counts as fine-class).
    Source: THRESHOLD_impact/curriculum_all_layers_stored_transitions.csv.
    Columns: arm, seed, transition, epoch_a, epoch_b, phase, tg_h, ab_h, di_h."""
    return pd.read_csv(RESULTS / "THRESHOLD_impact" / "curriculum_all_layers_stored_transitions.csv")


EV_STORED_JSON = RESULTS / "THRESHOLD_impact" / "curriculum_all_layers_stored.json"


def events_common():
    """Curriculum events under one common threshold set per seed (the scratch arm's alive-only permutation-null 95th
    percentiles, applied to all three arms), last block (layer4 + pooled layer), raw counts
    (SAE_outcomes/events_EC.csv, tset 'common', columns *_h_real). Same phase rule as
    events_stored(): a transition of a curriculum arm is 'pre' if its end epoch <= 20, else 'fine'.
    Columns: arm, seed, transition, epoch_a, epoch_b, phase, tg_h, ab_h, di_h."""
    e = pd.read_csv(RESULTS / "SAE_outcomes" / "events_EC.csv")
    e = e[e.tset == "common"].copy()
    e = e[e.epoch_b > e.epoch_a]
    e["phase"] = np.where((e.arm != "scratch_E30") & (e.epoch_b <= 20), "pre", "fine")
    e = e.rename(columns={"tg_h_real": "tg_h", "ab_h_real": "ab_h", "di_h_real": "di_h"})
    return e[["arm", "seed", "transition", "epoch_a", "epoch_b", "phase", "tg_h", "ab_h", "di_h"]]


# drawing
def arm_bar(ax, x, vals, arm, color=DARK, width=0.6, dots=True, seed=0, err="sem", dot_color=None, zorder=1):
    """Bar by arm: scratch open, random hatched, true filled. Returns (mean, lo, hi)."""
    vals = np.asarray(vals, float)
    m = vals.mean()
    if arm == "scratch":
        ax.bar(x, m, width=width, facecolor="white", edgecolor=color, lw=0.9, zorder=zorder)
    elif arm == "sc_rand":
        ax.bar(x, m, width=width, facecolor=color, edgecolor=color, alpha=HATCH_ALPHA, lw=0.9, hatch=HATCH,
               zorder=zorder)
    else:
        ax.bar(x, m, width=width, facecolor=color, edgecolor=color, alpha=BAR_ALPHA, lw=0.9, zorder=zorder)
    if dots:
        rng = np.random.default_rng(seed)
        ax.scatter(x + rng.uniform(-0.2, 0.2, len(vals)) * width, vals, s=5, color=dot_color or ERR, lw=0,
                   zorder=zorder + 2)
    if err is None:
        return m, m, m
    if err == "sem":
        e = vals.std(ddof=1) / math.sqrt(len(vals))
        lo, hi = m - e, m + e
    else:
        _, lo, hi = tci(vals)
    ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=ERR, elinewidth=0.8, capsize=2, capthick=0.8,
                zorder=zorder + 3)
    return m, lo, hi


def band_line(ax, x, Y, ls, color=DARK, lw=1.0, label=None, band=True, marker=None, ms=2.5, zorder=3):
    Y = np.asarray(Y, float)
    m = np.nanmean(Y, 0)
    if band and Y.shape[0] > 1:
        se = sem(Y)
        ax.fill_between(x, m - se, m + se, color=BAND if color == DARK else color, alpha=BAND_ALPHA, lw=0,
                        zorder=zorder - 1)
    ax.plot(x, m, color=color, ls=ls, lw=lw, label=label, marker=marker, ms=ms, zorder=zorder)
    return m


# events by process
PCOL = {"tg": ns.PROC["Tg-E"], "ab": ns.PROC["Ab-E"], "di": ns.PROC["Di-E"]}
PLAB = {"tg": "Tg-E", "ab": "Ab-E", "di": "Di-E"}
ARMS_EC = (("scratch_E30", "scratch"), ("sc_rand_K20", "sc_rand"), ("sc_true_K20", "sc_true"))


def events_by_process(ax, label_pct=("sc_true",), pct_fs=6.5, source=None, ylim=230):
    """Events per process during fine-class training (30 epochs): scratch open, random hatched, true filled
    (process hues). Default source events_stored(): stored thresholds, all five SAE layers, raw counts.
    Returns (rows, stats) for the Source Data."""
    from matplotlib.patches import Patch
    d = (source or events_stored)()
    f = d[d.phase == "fine"].groupby(["arm", "seed"])[[f"{p}_h" for p in PCOL]].sum() / 1e3
    rows, stat = [], []
    w = 0.26
    for j, p in enumerate(PCOL):
        col = f"{p}_h"
        base = f.loc["scratch_E30"][col]
        for k, (lane, arm) in enumerate(ARMS_EC):
            v = f.loc[lane][col]
            x = j + (k - 1) * w
            arm_bar(ax, x, v.values, arm, color=PCOL[p], width=w * 0.9, err=None, seed=10 * j + k)
            rows += [dict(process=PLAB[p], arm=ARM_LABEL[arm], seed=s, events_k=e) for s, e in v.items()]
            if arm != "scratch":
                ch = (v.values / base.loc[v.index].values - 1) * 100
                diff = v.values - base.loc[v.index].values
                st = paired(diff, f"{ARM_LABEL[arm]} - scratch, {PLAB[p]} (k events)")
                st.update(mean_pct_change=ch.mean(), n_lower=int((diff < 0).sum()))
                stat.append(st)
                if arm in label_pct:
                    ax.text(x, max(v.max(), v.mean()) + 5, f"{ch.mean():.0f}%".replace("-", "−"), ha="center",
                            va="bottom", fontsize=pct_fs, color=ERR, rotation=90 if len(label_pct) > 1 else 0)
    ax.set_xticks(range(3))
    ax.set_xticklabels([PLAB[p] for p in PCOL])
    for t, p in zip(ax.get_xticklabels(), PCOL):
        t.set_color(PCOL[p])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, ylim)
    ax.set_ylabel("Events in fine-class\ntraining (×10³)")
    return rows, stat
