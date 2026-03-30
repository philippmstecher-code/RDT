#!/usr/bin/env python3
"""ED Fig 10: Selectivity indices are not confounded by activation magnitude.

Three panels:
  (a) Conditional activation magnitude by selectivity type (violin + box)
  (b) Activation sparsity by selectivity type (violin + box)
  (c) SSI–CSI scatter coloured by log₁₀ conditional magnitude

Uses H matrices from external drive + selectivity/threshold data from raw_lanes JSONs.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Universal style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'Helvetica',
    'font.size': 8,
    'axes.labelsize': 8,
    'text.color': 'black',
    'axes.titlesize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})

FIG_W = 180 / 25.4   # 180 mm → inches
FIG_H = 65 / 25.4    # compact single-row height

# ── Process colors (consistent with all figures) ────────────────────────────
SSI_COLOR = '#3B82F6'   # blue  (Ab-E)
CSI_COLOR = '#F59E0B'   # amber (Di-E)
SAI_COLOR = '#8B5CF6'   # purple (Tg-E)
UNSPEC_COLOR = '#D1D5DB' # light grey

CATEGORY_STYLE = {
    'High-SSI': {'color': SSI_COLOR, 'label': 'High-SSI'},
    'High-CSI': {'color': CSI_COLOR, 'label': 'High-CSI'},
    'High-SAI': {'color': SAI_COLOR, 'label': 'High-SAI'},
    'Below threshold': {'color': UNSPEC_COLOR, 'label': 'Below\nthreshold'},
}
CAT_ORDER = ['High-SSI', 'High-CSI', 'High-SAI', 'Below threshold']

# ── Paths ────────────────────────────────────────────────────────────────────
PRECOMPUTED = ROOT / "data" / "activation_magnitude_selectivity.json"


# ═════════════════════════════════════════════════════════════════════════════
# Data collection — loads from pre-computed JSON
# ═════════════════════════════════════════════════════════════════════════════

def collect_data():
    """Load per-feature activation magnitude and selectivity data."""
    if not PRECOMPUTED.exists():
        print(f"ERROR: {PRECOMPUTED} not found.")
        print("This file contains pre-extracted per-feature activation data for ED Fig. 8.")
        import sys; sys.exit(1)

    with open(PRECOMPUTED) as f:
        blob = json.load(f)

    raw = blob["records"]
    # Convert to the format expected by classify()
    records = []
    for r in raw:
        records.append({
            "cond_mag": r["cond_magnitude"],
            "sparsity": r["sparsity"],
            "ssi": r["ssi"],
            "csi": r["csi"],
            "sai": r["sai"],
            "ssi_t": r["ssi_thresh"],
            "csi_t": r["csi_thresh"],
            "sai_t": r["sai_thresh"],
        })

    print(f"  Loaded {len(records):,} features from {PRECOMPUTED.name}")
    return records


def classify(records):
    cats = {k: {"mag": [], "spar": []} for k in CAT_ORDER}
    for r in records:
        h_ssi = r["ssi"] > r["ssi_t"]
        h_csi = r["csi"] > r["csi_t"]
        h_sai = r["sai"] > r["sai_t"]
        if h_ssi:
            cats["High-SSI"]["mag"].append(r["cond_mag"])
            cats["High-SSI"]["spar"].append(r["sparsity"])
        if h_csi:
            cats["High-CSI"]["mag"].append(r["cond_mag"])
            cats["High-CSI"]["spar"].append(r["sparsity"])
        if h_sai:
            cats["High-SAI"]["mag"].append(r["cond_mag"])
            cats["High-SAI"]["spar"].append(r["sparsity"])
        if not (h_ssi or h_csi or h_sai):
            cats["Below threshold"]["mag"].append(r["cond_mag"])
            cats["Below threshold"]["spar"].append(r["sparsity"])
    return cats


# ═════════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════════

def _violin_box(ax, data_lists, colors, labels, ylabel, log_scale=True):
    """Matched violin + overlaid boxplot — shared style."""
    positions = np.arange(1, len(data_lists) + 1)

    parts = ax.violinplot(data_lists, positions=positions, showmeans=False,
                          showmedians=False, showextrema=False, widths=0.7)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_edgecolor(c)
        pc.set_alpha(0.25)

    bp = ax.boxplot(data_lists, positions=positions, widths=0.18,
                    showfliers=False, patch_artist=True, zorder=3,
                    medianprops=dict(color='black', linewidth=1),
                    whiskerprops=dict(linewidth=0.5),
                    capprops=dict(linewidth=0.5))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_edgecolor(c)
        patch.set_alpha(0.65)

    if log_scale:
        ax.set_yscale('log')

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # n annotations at top
    ymax = ax.get_ylim()[1]
    for i, d in enumerate(data_lists):
        n = len(d)
        ax.text(i + 1, ymax * 0.65 if log_scale else ymax * 0.97,
                f'n={n:,}', ha='center', va='top',
                fontsize=5, color='#6B7280')


def build_figure(cats, records):
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = GridSpec(1, 3, figure=fig, wspace=0.42,
                  left=0.07, right=0.97, bottom=0.18, top=0.88)

    present = [k for k in CAT_ORDER if cats[k]["mag"]]
    colors = [CATEGORY_STYLE[k]['color'] for k in present]
    labels = [CATEGORY_STYLE[k]['label'] for k in present]

    # ── (a) Conditional magnitude ────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    _violin_box(ax_a,
                [cats[k]["mag"] for k in present],
                colors, labels,
                'Conditional magnitude\n(mean |act.| when active)')
    ax_a.set_title('a', loc='left', fontweight='bold', fontsize=9)

    # ── (b) Activation sparsity ──────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    _violin_box(ax_b,
                [cats[k]["spar"] for k in present],
                colors, labels,
                'Activation sparsity\n(frac. samples active)')
    ax_b.set_title('b', loc='left', fontweight='bold', fontsize=9)

    # ── (c) SSI–CSI scatter coloured by magnitude ────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])

    alive = [r for r in records if r["ssi"] > 0 or r["csi"] > 0]
    ssi_v = np.array([r["ssi"] for r in alive])
    csi_v = np.array([r["csi"] for r in alive])
    mag_v = np.array([r["cond_mag"] for r in alive])

    # Subsample for visual clarity
    rng = np.random.default_rng(42)
    if len(ssi_v) > 6000:
        idx = rng.choice(len(ssi_v), 6000, replace=False)
        ssi_v, csi_v, mag_v = ssi_v[idx], csi_v[idx], mag_v[idx]

    log_mag = np.log10(mag_v + 1e-8)
    sc = ax_c.scatter(ssi_v, csi_v, c=log_mag, s=1.5, alpha=0.45,
                      cmap='viridis', rasterized=True, linewidths=0)
    cb = plt.colorbar(sc, ax=ax_c, shrink=0.75, pad=0.03, aspect=20)
    cb.set_label('log$_{10}$(cond. mag.)', fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    cb.outline.set_linewidth(0.5)

    # Adaptive threshold lines
    if records:
        ssi_t = records[0]["ssi_t"]
        csi_t = records[0]["csi_t"]
        ax_c.axvline(ssi_t, color='#9CA3AF', ls='--', lw=0.6, zorder=1)
        ax_c.axhline(csi_t, color='#9CA3AF', ls='--', lw=0.6, zorder=1)

    ax_c.set_xlabel('SSI (superclass selectivity)')
    ax_c.set_ylabel('CSI (class selectivity)')
    ax_c.set_xlim(-0.02, 1.02)
    ax_c.set_ylim(-0.02, 1.02)
    ax_c.set_title('c', loc='left', fontweight='bold', fontsize=9)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)

    # ── Save ─────────────────────────────────────────────────────────────────
    for ext in ('pdf', 'png'):
        out_path = OUT / f"ed_fig_magnitude_selectivity.{ext}"
        fig.savefig(out_path, bbox_inches='tight')
        print(f"Saved: {out_path}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# Statistics
# ═════════════════════════════════════════════════════════════════════════════

def report_stats(cats, records):
    from scipy import stats

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    for k in CAT_ORDER:
        if not cats[k]["mag"]:
            continue
        m = np.array(cats[k]["mag"])
        s = np.array(cats[k]["spar"])
        print(f"\n{k}  (n = {len(m):,})")
        print(f"  Cond. magnitude: {m.mean():.3f} +/- {m.std():.3f}  (median {np.median(m):.3f})")
        print(f"  Sparsity:        {s.mean():.4f} +/- {s.std():.4f}  (median {np.median(s):.4f})")

    groups, names = [], []
    for k in ["High-SSI", "High-CSI", "High-SAI"]:
        if cats[k]["mag"]:
            groups.append(cats[k]["mag"])
            names.append(k)

    if len(groups) >= 2:
        H_stat, p_kw = stats.kruskal(*groups)
        print(f"\nKruskal-Wallis ({', '.join(names)}): H = {H_stat:.1f}, p = {p_kw:.2e}")
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                u, p_mw = stats.mannwhitneyu(groups[i], groups[j], alternative='two-sided')
                r_eff = 1 - (2 * u) / (len(groups[i]) * len(groups[j]))
                print(f"  {names[i]} vs {names[j]}: U = {u:.0f}, p = {p_mw:.2e}, r = {r_eff:.3f}")

    print("\nSpearman (selectivity vs cond. magnitude):")
    for idx_name, key in [("SSI", "ssi"), ("CSI", "csi"), ("SAI", "sai")]:
        vals = [(r[key], r["cond_mag"]) for r in records if r[key] > 0]
        if len(vals) > 10:
            x, y = zip(*vals)
            rho, p = stats.spearmanr(x, y)
            print(f"  {idx_name}: rho = {rho:.3f}, p = {p:.2e}, n = {len(vals):,}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Collecting feature data across 9 core lanes …")
    records = collect_data()
    print(f"\nTotal features: {len(records):,}")

    cats = classify(records)
    report_stats(cats, records)
    build_figure(cats, records)
