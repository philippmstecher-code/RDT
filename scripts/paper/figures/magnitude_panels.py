#!/usr/bin/env python3
"""Selectivity indices are independent of activation magnitude (Supplementary Fig. 4d-f; extended/supp4.py reuses
this module).

9 core CIFAR-100 lanes (ResNet-18, ViT-Small, CCT-7 x seeds 42/137/256), three layers per architecture, mid-to-late
checkpoints. Selectivity and thresholds from each lane's sae_analysis/sae_results.json ('selectivity',
'metadata.adaptive_thresholds'), activation matrices H from the lane folders. The script asserts the reference
feature count (75,165) and category sizes (CORE_N).

Per feature (n_act >= 1 on the 10,000 SAE images): conditional magnitude = mean |h| over images with |h| > 1e-6;
sparsity = fraction of images with |h| > 1e-6. Categories (may overlap): High-SSI = SSI > t_ssi, High-CSI = CSI >
t_csi, High-SAI = SAI > t_sai (each lane's adaptive thresholds), Below threshold = none of the three.
Statistics: Kruskal-Wallis over the three High groups; pairwise Mann-Whitney, r = 1 - 2U/(n1 n2); Spearman of each
index with conditional magnitude over features with index > 0.
a  conditional magnitude by category (violin + box, log axis); b  sparsity by category; c  SSI-CSI scatter
   (6,000-feature random subsample, seed 42) coloured by log10 conditional magnitude, dashed lines at the first
   lane's adaptive SSI / CSI thresholds.

Usage: ES_FIG_OUT=<dir> python scripts/paper/figures/magnitude_panels.py
(the first run reads ~20 GB of activations from the lane folders and caches the per-feature table in
analysis_outputs/paper/figures/)
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nmi_style as ns  # noqa: E402

NAME = "ed7"
W_MM, H_MM = 180, 65
CACHE = ns.OUT / "ed7_magnitude_records.csv.gz"
EXP_DIR = Path(RDT_DATA + "/core/experiments")
CORE_EXPERIMENT = "abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb"
LANES = {
    "ResNet18 seed42": "7db28a9a-f436-4d8a-a6b4-37e69fbf54eb",
    "ResNet18 seed137": "a0bcbcba-94a9-4009-bcb6-d725b47588ee",
    "ResNet18 seed256": "935c13ae-a026-4eae-a2b2-0306879b2e8c",
    "ViTSmall seed42": "10d31151-68a2-4491-836d-3ead8e00a7ad",
    "ViTSmall seed137": "909d7f68-55e1-4cd9-8c7b-68d5793798d3",
    "ViTSmall seed256": "abad8975-de55-4e7e-b95d-05067f0eed90",
    "CCT7 seed42": "4bebc329-c26a-4eb4-b18c-7586e513f49e",
    "CCT7 seed137": "edf8275e-c99f-4436-a13d-4ed50f072a66",
    "CCT7 seed256": "445dd4d4-770a-4f33-b82b-c0bc7baed9a7",
}
ARCH_LAYERS = {
    "ResNet18": ["layer3", "layer4", "avgpool"],
    "ViTSmall": ["encoder.layers.9", "encoder.layers.11", "encoder.ln"],
    "CCT7": ["transformer.layers.5", "transformer.layers.6", "norm"],
}
SSI_COLOR, CSI_COLOR, SAI_COLOR, UNSPEC_COLOR = "#3B82F6", "#F59E0B", "#8B5CF6", "#D1D5DB"
CAT_ORDER = ["High-SSI", "High-CSI", "High-SAI", "Below threshold"]
CAT_COLOR = {"High-SSI": SSI_COLOR, "High-CSI": CSI_COLOR, "High-SAI": SAI_COLOR, "Below threshold": UNSPEC_COLOR}
CAT_LABEL = {"High-SSI": "High-SSI", "High-CSI": "High-CSI", "High-SAI": "High-SAI", "Below threshold": "Below\nthreshold"}
CORE_N = {"total": 75165, "High-SSI": 7404, "High-CSI": 11387, "High-SAI": 1259, "Below threshold": 58365}


def collect():
    if CACHE.exists():
        return pd.read_csv(CACHE)
    parts = []
    for lane_name, uuid in LANES.items():
        lane = EXP_DIR / CORE_EXPERIMENT / "lanes" / uuid / "sae_analysis"
        data = json.loads((lane / "sae_results.json").read_text())
        sel, th = data["selectivity"], data["metadata"]["adaptive_thresholds"]
        arch, seed = lane_name.split()
        cps = sorted(sel.keys(), key=lambda x: (x != "terminal", int(x) if x != "terminal" else 999))
        for cp in cps[len(cps) // 3:]:  # mid-to-late; 'terminal' sorts first and is never selected
            for layer in ARCH_LAYERS[arch]:
                if layer not in sel.get(cp, {}):
                    continue
                ls = sel[cp][layer]
                ssi, csi, sai = (np.array(ls[k], float) for k in ("feature_ssi", "feature_csi", "feature_sai"))
                hp = lane / "activations" / cp / f"{layer.replace('.', '_')}.npy"
                if not hp.exists():
                    continue
                H = np.abs(np.load(hp).astype(np.float64))
                if H.shape[1] != len(ssi):
                    continue
                act = H > 1e-6
                n_act = act.sum(0)
                keep = n_act >= 1
                mag = np.where(keep, (H * act).sum(0) / np.maximum(n_act, 1), np.nan)
                parts.append(pd.DataFrame(dict(
                    lane=lane_name, arch=arch, seed=int(seed.replace("seed", "")), checkpoint=cp, layer=layer,
                    feature=np.arange(len(ssi))[keep], cond_mag=mag[keep], sparsity=n_act[keep] / H.shape[0],
                    ssi=ssi[keep], csi=csi[keep], sai=sai[keep], ssi_t=th["ssi_adaptive_thresh"],
                    csi_t=th["csi_adaptive_thresh"], sai_t=th["sai_adaptive_thresh"])))
        print(f"  {lane_name}: {sum(len(p) for p in parts):,} features so far", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(CACHE, index=False)
    return df


def categories(df):
    hs, hc, ha = df.ssi > df.ssi_t, df.csi > df.csi_t, df.sai > df.sai_t
    return {"High-SSI": hs, "High-CSI": hc, "High-SAI": ha, "Below threshold": ~(hs | hc | ha)}


def statistics(df, cats):
    rows = []
    for k in CAT_ORDER:
        m, s = df.cond_mag[cats[k]].values, df.sparsity[cats[k]].values
        rows.append(dict(test="summary", group=k, n=len(m), cond_mag_mean=m.mean(), cond_mag_sd=m.std(),
                         cond_mag_median=np.median(m), sparsity_mean=s.mean(), sparsity_sd=s.std(),
                         sparsity_median=np.median(s)))
    groups = [(k, df.cond_mag[cats[k]].values) for k in ("High-SSI", "High-CSI", "High-SAI")]
    h, p = stats.kruskal(*[g for _, g in groups])
    rows.append(dict(test="Kruskal-Wallis, conditional magnitude", group="High-SSI, High-CSI, High-SAI", statistic=h, p=p))
    for i in range(3):
        for j in range(i + 1, 3):
            u, p = stats.mannwhitneyu(groups[i][1], groups[j][1], alternative="two-sided")
            r = 1 - 2 * u / (len(groups[i][1]) * len(groups[j][1]))
            rows.append(dict(test="Mann-Whitney, conditional magnitude", group=f"{groups[i][0]} vs {groups[j][0]}",
                             statistic=u, p=p, effect_size_r=r))
    for nm, key in (("SSI", "ssi"), ("CSI", "csi"), ("SAI", "sai")):
        sub = df[df[key] > 0]
        rho, p = stats.spearmanr(sub[key], sub.cond_mag)
        rows.append(dict(test="Spearman, index vs conditional magnitude (index > 0)", group=nm, n=len(sub),
                         statistic=rho, p=p))
    return pd.DataFrame(rows)


def violin_box(ax, data, colors, labels, ylabel):
    pos = np.arange(1, len(data) + 1)
    parts = ax.violinplot(data, positions=pos, showmeans=False, showmedians=False, showextrema=False, widths=0.7)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_edgecolor(c)
        pc.set_alpha(0.25)
    bp = ax.boxplot(data, positions=pos, widths=0.18, showfliers=False, patch_artist=True, zorder=3,
                    medianprops=dict(color="black", linewidth=1), whiskerprops=dict(linewidth=0.5),
                    capprops=dict(linewidth=0.5))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_edgecolor(c)
        patch.set_alpha(0.65)
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ticks = [10.0 ** e for e in range(-6, 5) if lo <= 10.0 ** e <= hi * 1.0001]
    from matplotlib.ticker import FixedLocator, NullFormatter
    ax.yaxis.set_major_locator(FixedLocator(ticks))   # plain decimal labels: mathtext exponents print below 6 pt
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.4f}".rstrip("0").rstrip(".")))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel(ylabel, fontsize=6.5)
    ax.tick_params(labelsize=6)
    ax.set_ylim(lo, hi * 5)   # head room for the n labels above the violins
    for i, d in enumerate(data):
        ax.text(i + 1, hi * 4.6, f"n =\n{len(d):,}", ha="center", va="top", fontsize=6, color="#6B7280",
                linespacing=1.0)


def main():
    df = collect()
    cats = categories(df)
    n = {k: int(v.sum()) for k, v in cats.items()}
    n["total"] = len(df)
    print("counts", n)
    assert n == CORE_N, f"reference counts not reproduced: {n} vs {CORE_N}"
    st = statistics(df, cats)
    print(st.to_string())
    mw = st[(st.test.str.startswith("Mann")) & (st.group == "High-SSI vs High-CSI")].effect_size_r.iloc[0]
    rho = st[st.test.str.startswith("Spearman")].statistic.abs().max()
    print(f"Mann-Whitney r (High-SSI vs High-CSI) = {mw:.3f}; max |rho| = {rho:.4f}")

    ns.apply()
    fig = plt.figure(figsize=(W_MM * ns.MM, H_MM * ns.MM))
    gs = fig.add_gridspec(1, 3, wspace=0.55, left=0.085, right=0.95, bottom=0.17, top=0.93)
    present = [k for k in CAT_ORDER if n[k]]
    colors, labels = [CAT_COLOR[k] for k in present], [CAT_LABEL[k] for k in present]
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))
    violin_box(ax_a, [df.cond_mag[cats[k]].values for k in present], colors, labels,
               "Conditional magnitude\n(mean |act.| when active)")
    violin_box(ax_b, [df.sparsity[cats[k]].values for k in present], colors, labels,
               "Activation sparsity\n(frac. samples active)")
    alive = df[(df.ssi > 0) | (df.csi > 0)]
    rng = np.random.default_rng(42)
    idx = rng.choice(len(alive), 6000, replace=False) if len(alive) > 6000 else np.arange(len(alive))
    sub = alive.iloc[idx]
    log_mag = np.log10(sub.cond_mag.values + 1e-8)
    sc = ax_c.scatter(sub.ssi, sub.csi, c=log_mag, s=1.5, alpha=0.45, cmap="viridis", rasterized=True, linewidths=0)
    cb = plt.colorbar(sc, ax=ax_c, shrink=0.75, pad=0.03, aspect=20)
    cb.set_label("log10 (cond. mag.)", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.5)
    t0 = df.iloc[0]
    ax_c.axvline(t0.ssi_t, color="#9CA3AF", ls="--", lw=0.6, zorder=1)
    ax_c.axhline(t0.csi_t, color="#9CA3AF", ls="--", lw=0.6, zorder=1)
    ax_c.set_xlabel("SSI (superclass selectivity)", fontsize=6.5)
    ax_c.set_ylabel("CSI (class selectivity)", fontsize=6.5)
    ax_c.set_xlim(-0.02, 1.02)
    ax_c.set_ylim(-0.02, 1.02)
    ax_c.tick_params(labelsize=6)
    for ax, k in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c")):
        x0, _, w, _ = ax.get_position().bounds
        left_mm = 1.0 if ax is ax_a else x0 * W_MM - 13.0
        ns.panel(ax, k, x=(left_mm - x0 * W_MM) / (w * W_MM), y=1.04)

    cat_cols = pd.DataFrame({f"is_{k.replace(' ', '_').replace('-', '_')}": v for k, v in cats.items()})
    ns.source_data(pd.concat([df, cat_cols], axis=1), NAME, "a_b")
    ns.source_data(sub.assign(log10_cond_mag=log_mag), NAME, "c")
    ns.source_data(st, NAME, "stats")
    print("saved", ns.save_final(fig, NAME, W_MM, ed=True))


if __name__ == "__main__":
    import os as _os
    if not _os.environ.get("ES_FIG_OUT"):
        raise SystemExit("magnitude_panels.py draws a three-panel overview; Supplementary Fig. 4d-f is drawn by "
                         "extended/supp4.py. Set ES_FIG_OUT to an output folder to draw the overview.")
    main()
