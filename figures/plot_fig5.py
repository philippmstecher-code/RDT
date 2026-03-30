#!/usr/bin/env python3
"""Figure 5: Feature Survival and Tg-E Longevity (F5) — 4 panels.

Panel (a): Kaplan-Meier-style step survival curves for Tg-E vs Non-Tg
Panel (b): Spearman correlation bars with 95% CI (unchanged)
Panel (c): KM curves for 3 representative lanes (one per architecture)
Panel (d): Scaffold census — stacked SAI/SSI/CSI feature counts across training
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "consolidated_findings.json"
TG_DATA = ROOT / "data" / "feature_survival_tg_expanded.json"
CENSUS = ROOT / "data" / "cumulative_scaffold_census.json"
OUT = ROOT / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

with open(DATA) as f:
    data = json.load(f)

with open(TG_DATA) as f:
    tg_expanded = json.load(f)

with open(CENSUS) as f:
    census_data = json.load(f)

# ── Universal style ──
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

HYPO_COLORS = {
    'Tg-E': '#8B5CF6', 'Ab-E': '#3B82F6', 'Di-E': '#F59E0B',
    'As-E': '#0891B2', 'De-E': '#EF4444',
}
ARCH_COLORS = {'ResNet-18': '#2C5F8A', 'ViT-Small': '#2A9D8F', 'CCT-7': '#E76F51'}

surv = data.get('feature_survival', {})
tg_surv = data.get('tg_survival', {})


def approximate_km_curve(survival_rate, mean_lifespan, max_lifespan, n):
    """Approximate a parametric step survival curve (exponential model) from summary statistics.

    Uses an exponential decay model calibrated so that:
    - S(max_lifespan) = survival_rate (the fraction surviving to terminal)
    - The curve shape follows an exponential hazard model.

    Returns (times, survival_probs) suitable for step plotting.
    """
    T = max(max_lifespan, 1)
    times = np.arange(0, T + 1)

    if survival_rate <= 0:
        survival_rate = 0.001
    if survival_rate >= 1.0:
        survival_rate = 0.999

    # Exponential model: S(t) = exp(-lambda * t)
    # S(T) = survival_rate => lambda = -ln(survival_rate) / T
    lam = -np.log(survival_rate) / T
    surv_probs = np.exp(-lam * times)

    # Start at 1.0
    surv_probs[0] = 1.0

    return times, surv_probs


fig = plt.figure(figsize=(180/25.4, 180/25.4))
gs = GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.40,
              height_ratios=[1, 1.3])

# ── Panel (a): Construction/refinement lifespan ratio per lane ──
ax_a = fig.add_subplot(gs[0, 0])
ax_a.text(-0.15, 1.05, 'a', transform=ax_a.transAxes, fontsize=8,
          fontweight='bold', va='top', ha='left')

# Compute mean lifespan ratio: (SAI + SSI) / CSI per lane
_standard_keys = [k for k in tg_expanded
                  if 'CIFAR100-seed' in k and '200ep' not in k and '8x' not in k]

# Sort by architecture then seed
_arch_order = {'ResNet18': 0, 'ViTSmall': 1, 'CCT7': 2}
_seed_order = {'seed42': 0, 'seed137': 1, 'seed256': 2}
def _sort_key_a(l):
    parts = l.split('-')
    return (_arch_order.get(parts[0], 99), _seed_order.get(parts[-1], 99))

_standard_keys = sorted(_standard_keys, key=_sort_key_a)

ratios = []
colors = []
labels_a = []
for k in _standard_keys:
    sai_c = tg_expanded[k]['cohorts']['tg']  # high-SAI features
    ssi_c = tg_expanded[k]['cohorts']['high_ssi']
    csi_c = tg_expanded[k]['cohorts']['high_csi']
    # Weighted mean lifespan of SAI + SSI cohorts
    combined_ml = (sai_c['mean_lifespan'] * sai_c['n'] + ssi_c['mean_lifespan'] * ssi_c['n']) / max(sai_c['n'] + ssi_c['n'], 1)
    ratio = combined_ml / max(csi_c['mean_lifespan'], 0.001)
    ratios.append(ratio)
    # Color by architecture
    if k.startswith('ResNet'):
        colors.append(ARCH_COLORS['ResNet-18'])
    elif k.startswith('ViT'):
        colors.append(ARCH_COLORS['ViT-Small'])
    else:
        colors.append(ARCH_COLORS['CCT-7'])
    # Short label
    name = k.replace('ResNet18', 'R18').replace('ViTSmall', 'ViT').replace('CCT7', 'CCT')
    name = name.replace('-CIFAR100', '').replace('-seed', '-')
    labels_a.append(name)

x_a = np.arange(len(ratios))
bars = ax_a.bar(x_a, ratios, color=colors, alpha=0.8, width=0.7)

# Reference line at 1.0
ax_a.axhline(1.0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

# Value labels on bars
for i, (bar, val) in enumerate(zip(bars, ratios)):
    ax_a.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
              f'{val:.2f}', ha='center', va='bottom', fontsize=6)

ax_a.set_xticks(x_a)
ax_a.set_xticklabels(labels_a, rotation=45, ha='right', fontsize=7)
ax_a.set_ylabel('Mean lifespan ratio')
ax_a.set_title('Construction / refinement feature lifespan', fontsize=8)
ax_a.set_ylim(0, max(ratios) * 1.2)

# Architecture group separators
ax_a.axvline(2.5, color='#D1D5DB', linewidth=0.5, linestyle='--')
ax_a.axvline(5.5, color='#D1D5DB', linewidth=0.5, linestyle='--')


# ── Panel (b): Spearman correlation bars with 95% CI ──
ax_b = fig.add_subplot(gs[0, 1])
ax_b.text(-0.15, 1.05, 'b', transform=ax_b.transAxes, fontsize=8,
          fontweight='bold', va='top', ha='left')

if surv:
    # Filter to 9 standard CIFAR-100 runs, sort by architecture then seed
    standard_labels = [l for l in surv.keys()
                       if 'CIFAR100' in l and '200ep' not in l and '8x' not in l]

    _arch_order = {'ResNet18': 0, 'ViTSmall': 1, 'CCT7': 2}
    _seed_order = {'seed42': 0, 'seed137': 1, 'seed256': 2}
    def _sort_key(l):
        parts = l.split('-')
        return (_arch_order.get(parts[0], 99), _seed_order.get(parts[-1], 99))

    labels_sorted = sorted(standard_labels, key=_sort_key)
    ssi_corrs = [surv[l]['ssi_survival_corr'] for l in labels_sorted]
    csi_corrs = [surv[l]['csi_survival_corr'] for l in labels_sorted]
    ssi_ps = [surv[l].get('ssi_survival_p', 1.0) for l in labels_sorted]
    csi_ps = [surv[l].get('csi_survival_p', 1.0) for l in labels_sorted]

    x_bars = np.arange(len(labels_sorted))
    w = 0.35

    # Approximate 95% CI for Spearman rho using Fisher z-transform
    def rho_ci(rho, n, alpha=0.05):
        from math import atanh, tanh, sqrt
        z = atanh(rho)
        se = 1 / sqrt(n - 3) if n > 3 else 0.1
        z_crit = 1.96
        lo = tanh(z - z_crit * se)
        hi = tanh(z + z_crit * se)
        return rho - lo, hi - rho

    ssi_err_lo = []
    ssi_err_hi = []
    csi_err_lo = []
    csi_err_hi = []
    for l in labels_sorted:
        n_tracked = surv[l].get('n_total_tracked', 1000)
        lo, hi = rho_ci(surv[l]['ssi_survival_corr'], n_tracked)
        ssi_err_lo.append(lo)
        ssi_err_hi.append(hi)
        lo, hi = rho_ci(surv[l]['csi_survival_corr'], n_tracked)
        csi_err_lo.append(lo)
        csi_err_hi.append(hi)

    ax_b.bar(x_bars - w/2, ssi_corrs, w, color=HYPO_COLORS['Ab-E'], alpha=0.7, label='SSI-lifespan',
             yerr=[ssi_err_lo, ssi_err_hi], ecolor='#374151', capsize=2, error_kw={'linewidth': 0.5})
    ax_b.bar(x_bars + w/2, csi_corrs, w, color=HYPO_COLORS['Di-E'], alpha=0.7, label='CSI-lifespan',
             yerr=[csi_err_lo, csi_err_hi], ecolor='#374151', capsize=2, error_kw={'linewidth': 0.5})

    # Asterisk-style p-value annotations
    def p_to_stars(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return 'n.s.'

    for i, (sp, cp) in enumerate(zip(ssi_ps, csi_ps)):
        y_ssi = max(ssi_corrs[i], 0) + ssi_err_hi[i] + 0.005
        ax_b.text(i - w/2, y_ssi, p_to_stars(sp),
                  ha='center', va='bottom', fontsize=7, color='black')

        if csi_corrs[i] >= 0:
            y_csi = csi_corrs[i] + csi_err_hi[i] + 0.005
            va_csi = 'bottom'
        else:
            y_csi = csi_corrs[i] - csi_err_lo[i] - 0.005
            va_csi = 'top'
        ax_b.text(i + w/2, y_csi, p_to_stars(cp),
                  ha='center', va=va_csi, fontsize=7, color='black')

    # Adjust y-limits to give space for annotations
    ax_b.set_ylim(ax_b.get_ylim()[0] - 0.03, ax_b.get_ylim()[1] + 0.08)

    ax_b.axhline(0, color='black', linewidth=0.3)
    def abbreviate_lane(l):
        name = l
        name = name.replace('ResNet18', 'R18')
        name = name.replace('ViTSmall', 'ViT')
        name = name.replace('CCT7', 'CCT')
        name = name.replace('-CIFAR100', '')
        name = name.replace('-seed', '-')
        return name

    ax_b.set_xticks(x_bars)
    ax_b.set_xticklabels([abbreviate_lane(l) for l in labels_sorted],
                          rotation=45, ha='right', fontsize=7)
    ax_b.set_ylabel('Spearman rho')
    ax_b.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7, fontsize=7, loc='upper left')

    # Architecture group separators
    ax_b.axvline(2.5, color='#D1D5DB', linewidth=0.5, linestyle='--')
    ax_b.axvline(5.5, color='#D1D5DB', linewidth=0.5, linestyle='--')

# ── Panel (c): Developmental persistence per cohort per architecture ──
ax_c = fig.add_subplot(gs[1, 0])
ax_c.text(-0.15, 1.05, 'c', transform=ax_c.transAxes, fontsize=8,
          fontweight='bold', va='top', ha='left')

# Load or compute developmental persistence data
DEV_PERSIST_FILE = ROOT / "data" / "developmental_persistence.json"

if DEV_PERSIST_FILE.exists():
    with open(DEV_PERSIST_FILE) as _f:
        dev_persist_data = json.load(_f)
else:
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "scripts"))
    DATA_DIR_C = ROOT / "data"

    # Load per-lane adaptive thresholds from consolidated_findings.json
    _cf_lanes = data.get('lanes', {})

    _ALL_LANES = [
        ("abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb", "7db28a9a-f436-4d8a-a6b4-37e69fbf54eb", "ResNet18-CIFAR100-seed42"),
        ("abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb", "a0bcbcba-94a9-4009-bcb6-d725b47588ee", "ResNet18-CIFAR100-seed137"),
        ("abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb", "935c13ae-a026-4eae-a2b2-0306879b2e8c", "ResNet18-CIFAR100-seed256"),
        ("abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb", "10d31151-68a2-4491-836d-3ead8e00a7ad", "ViTSmall-CIFAR100-seed42"),
        ("abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb", "909d7f68-55e1-4cd9-8c7b-68d5793798d3", "ViTSmall-CIFAR100-seed137"),
        ("abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb", "abad8975-de55-4e7e-b95d-05067f0eed90", "ViTSmall-CIFAR100-seed256"),
        ("abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb", "4bebc329-c26a-4eb4-b18c-7586e513f49e", "CCT7-CIFAR100-seed42"),
        ("abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb", "edf8275e-c99f-4436-a13d-4ed50f072a66", "CCT7-CIFAR100-seed137"),
        ("abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb", "445dd4d4-770a-4f33-b82b-c0bc7baed9a7", "CCT7-CIFAR100-seed256"),
    ]
    dev_persist_data = {}

    for exp_id, lane_id, label in _ALL_LANES:
        print(f"  Computing dev persistence for {label}...", flush=True)
        sae_dir = DATA_DIR_C / "experiments" / exp_id / "lanes" / lane_id / "sae_analysis"
        with open(sae_dir / "sae_results.json") as _f:
            sae_data = json.load(_f)

        layers = sae_data['metadata']['layers']
        ckpts = sae_data['metadata']['checkpoint_labels']
        sel = sae_data.get('selectivity', {})
        n_trans = len(ckpts) - 1

        def _sl_c(layer):
            return layer.replace('/', '_').replace('.', '_')

        tdir = sae_dir / "transitions"
        recs = []

        for layer in layers:
            fmaps = [dict() for _ in range(n_trans)]
            for t in range(n_trans):
                tf = tdir / f"{ckpts[t]}_to_{ckpts[t+1]}" / f"{_sl_c(layer)}.json"
                if not tf.exists():
                    continue
                with open(tf) as _f:
                    samples = json.load(_f)
                lc = {}
                for s in samples:
                    for e in s.get('stable', []):
                        a, b = e.get('fid_a'), e.get('fid_b')
                        if a is not None and b is not None:
                            lc[(a, b)] = lc.get((a, b), 0) + 1
                best = {}
                for (a, b), c in lc.items():
                    if a not in best or c > lc.get((a, best[a]), 0):
                        best[a] = b
                fmaps[t] = best

            for bi in range(len(ckpts)):
                sc = sel.get(str(ckpts[bi]), sel.get(ckpts[bi], {})).get(layer, {})
                ssi_l = sc.get('feature_ssi', [])
                csi_l = sc.get('feature_csi', [])
                sai_l = sc.get('feature_sai', [])
                if not ssi_l:
                    continue
                alive = set(fmaps[bi].keys()) if bi < n_trans else set()
                for fid in range(len(ssi_l)):
                    if ssi_l[fid] > 0 or (fid < len(csi_l) and csi_l[fid] > 0):
                        alive.add(fid)
                for fid in alive:
                    cur = fid
                    ls = 0
                    for t in range(bi, n_trans):
                        if cur in fmaps[t]:
                            ls += 1
                            cur = fmaps[t][cur]
                        else:
                            break
                    survived = (bi + ls) >= n_trans
                    recs.append({
                        'survived': survived,
                        'dev_persistence': ls / n_trans,
                        'sai': sai_l[fid] if fid < len(sai_l) else 0.0,
                        'ssi': ssi_l[fid] if fid < len(ssi_l) else 0.0,
                        'csi': csi_l[fid] if fid < len(csi_l) else 0.0,
                    })

        terminal = [r for r in recs if r['survived']]

        # Use per-lane adaptive thresholds from null permutation test
        _lane_at = _cf_lanes.get(label, {}).get('metadata', {}).get('adaptive_thresholds', {})
        if _lane_at:
            ssi_thresh = _lane_at['ssi_adaptive_thresh']
            csi_thresh = _lane_at['csi_adaptive_thresh']
            sai_thresh = _lane_at['sai_adaptive_thresh']
        else:
            ssi_thresh = float(np.median([r['ssi'] for r in recs]))
            csi_thresh = float(np.median([r['csi'] for r in recs]))
            sai_thresh = 0.9

        tg_t = [r for r in terminal if r['sai'] > sai_thresh]
        ab_t = [r for r in terminal if r['sai'] <= sai_thresh and r['ssi'] >= ssi_thresh]
        di_t = [r for r in terminal if r['sai'] <= sai_thresh and r['csi'] >= csi_thresh and r['ssi'] < ssi_thresh]

        dev_persist_data[label] = {
            'tg_dp': float(np.mean([r['dev_persistence'] for r in tg_t])) if tg_t else 0,
            'ab_dp': float(np.mean([r['dev_persistence'] for r in ab_t])) if ab_t else 0,
            'di_dp': float(np.mean([r['dev_persistence'] for r in di_t])) if di_t else 0,
            'tg_n': len(tg_t), 'ab_n': len(ab_t), 'di_n': len(di_t),
            'ssi_threshold': ssi_thresh, 'csi_threshold': csi_thresh, 'sai_threshold': sai_thresh,
            'threshold_source': 'adaptive' if _lane_at else 'median',
        }

    with open(DEV_PERSIST_FILE, 'w') as _f:
        json.dump(dev_persist_data, _f, indent=2)
    print(f"  Saved {DEV_PERSIST_FILE}")

# Plot grouped bars: 3 cohorts × 3 architectures
arch_lane_map_c = {
    'ResNet-18': [k for k in dev_persist_data if k.startswith('ResNet18-CIFAR100-seed')],
    'ViT-Small': [k for k in dev_persist_data if k.startswith('ViTSmall-CIFAR100-seed')],
    'CCT-7': [k for k in dev_persist_data if k.startswith('CCT7-CIFAR100-seed')],
}
arch_names_c = ['ResNet-18', 'ViT-Small', 'CCT-7']
x_pos = np.arange(len(arch_names_c))
w = 0.22

tg_means_c, ab_means_c, di_means_c = [], [], []
tg_sems_c, ab_sems_c, di_sems_c = [], [], []
tg_seeds_c, ab_seeds_c, di_seeds_c = [], [], []

for arch in arch_names_c:
    lanes = arch_lane_map_c[arch]
    tg_vals = [dev_persist_data[l]['tg_dp'] for l in lanes]
    ab_vals = [dev_persist_data[l]['ab_dp'] for l in lanes]
    di_vals = [dev_persist_data[l]['di_dp'] for l in lanes]
    tg_means_c.append(np.mean(tg_vals))
    ab_means_c.append(np.mean(ab_vals))
    di_means_c.append(np.mean(di_vals))
    tg_sems_c.append(np.std(tg_vals) / np.sqrt(len(tg_vals)) if len(tg_vals) > 1 else 0)
    ab_sems_c.append(np.std(ab_vals) / np.sqrt(len(ab_vals)) if len(ab_vals) > 1 else 0)
    di_sems_c.append(np.std(di_vals) / np.sqrt(len(di_vals)) if len(di_vals) > 1 else 0)
    tg_seeds_c.append(tg_vals)
    ab_seeds_c.append(ab_vals)
    di_seeds_c.append(di_vals)

ax_c.bar(x_pos - w, tg_means_c, w, yerr=tg_sems_c,
         color=HYPO_COLORS['Tg-E'], alpha=0.7, label='SAI',
         ecolor='#374151', capsize=3, error_kw={'linewidth': 0.5})
ax_c.bar(x_pos, ab_means_c, w, yerr=ab_sems_c,
         color=HYPO_COLORS['Ab-E'], alpha=0.7, label='SSI',
         ecolor='#374151', capsize=3, error_kw={'linewidth': 0.5})
ax_c.bar(x_pos + w, di_means_c, w, yerr=di_sems_c,
         color=HYPO_COLORS['Di-E'], alpha=0.7, label='CSI',
         ecolor='#374151', capsize=3, error_kw={'linewidth': 0.5})

# Overlay individual seed points
for i in range(len(arch_names_c)):
    for v in tg_seeds_c[i]:
        ax_c.scatter(i - w, v, color=HYPO_COLORS['Tg-E'], s=12, zorder=5,
                     edgecolors='white', linewidths=0.3)
    for v in ab_seeds_c[i]:
        ax_c.scatter(i, v, color=HYPO_COLORS['Ab-E'], s=12, zorder=5,
                     edgecolors='white', linewidths=0.3)
    for v in di_seeds_c[i]:
        ax_c.scatter(i + w, v, color=HYPO_COLORS['Di-E'], s=12, zorder=5,
                     edgecolors='white', linewidths=0.3)

# Annotate Tg/Di ratio
for i in range(len(arch_names_c)):
    if di_means_c[i] > 0:
        ratio = tg_means_c[i] / di_means_c[i]
        y_top = max(tg_means_c[i] + tg_sems_c[i],
                    ab_means_c[i] + ab_sems_c[i],
                    di_means_c[i] + di_sems_c[i]) + 0.008
        ax_c.text(i, y_top, f'SAI/CSI={ratio:.1f}×', ha='center', va='bottom',
                  fontsize=6.5, fontweight='bold', color='black')

ax_c.set_xticks(x_pos)
ax_c.set_xticklabels(arch_names_c, fontsize=7)
ax_c.set_ylabel('Developmental persistence')
y_max_c = max(max(tg_means_c), max(ab_means_c), max(di_means_c))
ax_c.set_ylim(0, y_max_c * 1.5)
ax_c.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7,
            fontsize=7, loc='upper right')
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)

# ── Panel (d): Scaffold census — stacked SAI/SSI/CSI counts across training ──
INDEX_COLORS = {'SAI': '#8B5CF6', 'SSI': '#3B82F6', 'CSI': '#F59E0B'}
STACK_ORDER = [('sai', 'SAI'), ('ssi', 'SSI'), ('csi', 'CSI')]

ARCH_CENSUS = {
    'ResNet-18': [l for l in census_data if l.startswith('ResNet18-CIFAR100-seed')],
    'ViT-Small': [l for l in census_data if l.startswith('ViTSmall-CIFAR100-seed')],
    'CCT-7': [l for l in census_data if l.startswith('CCT7-CIFAR100-seed')],
}

gs_d = gs[1, 1].subgridspec(3, 1, hspace=0.55)

for row_idx, (arch, lanes) in enumerate(ARCH_CENSUS.items()):
    ax_d = fig.add_subplot(gs_d[row_idx])
    if row_idx == 0:
        ax_d.text(-0.15, 1.15, 'd', transform=ax_d.transAxes, fontsize=8,
                  fontweight='bold', va='top', ha='left')

    # Collect each seed's snapshot data (seeds within an arch have same checkpoint count)
    seed_interp = {m: [] for m, _ in STACK_ORDER}
    ckpt_labels_d = None
    for label in lanes:
        if label not in census_data:
            continue
        r = census_data[label]
        if ckpt_labels_d is None:
            ckpt_labels_d = [str(c) if c != 'terminal' else '10'
                             for c in r['checkpoint_labels']]
        for metric, _ in STACK_ORDER:
            snap = np.array(r['snapshot'][metric], dtype=float)
            seed_interp[metric].append(snap)

    n_ckpts_d = len(ckpt_labels_d)
    x_d = np.arange(n_ckpts_d)

    # Mean across 3 seeds
    means = {m: np.array(seed_interp[m]).mean(axis=0) for m, _ in STACK_ORDER}

    # Stacked area
    bottom = np.zeros(n_ckpts_d)
    for metric, _ in STACK_ORDER:
        color = INDEX_COLORS[metric.upper()]
        ax_d.fill_between(x_d, bottom, bottom + means[metric],
                          color=color, alpha=0.70, linewidth=0)
        ax_d.plot(x_d, bottom + means[metric], color='white', linewidth=0.6)
        bottom = bottom + means[metric]

    ax_d.set_title(arch, fontsize=7, fontweight='bold', pad=2,
                   color='black')
    ax_d.tick_params(axis='y', labelsize=6)
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)
    ax_d.set_xlim(-0.5, n_ckpts_d - 0.5)

    if row_idx == 1:
        ax_d.set_ylabel('Features above\nadaptive threshold', fontsize=7)
    if row_idx == 2:
        ax_d.set_xticks(x_d)
        ax_d.set_xticklabels(ckpt_labels_d, fontsize=6)
        ax_d.set_xlabel('Training checkpoint', fontsize=7)
    else:
        ax_d.set_xticks(x_d)
        ax_d.set_xticklabels([])

    if row_idx == 0:
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=INDEX_COLORS[m.upper()], alpha=0.70, label=lbl)
                   for m, lbl in STACK_ORDER]
        ax_d.legend(handles=handles, frameon=True, facecolor='white', edgecolor='none',
                    framealpha=0.7, fontsize=6, loc='upper left',
                    bbox_to_anchor=(0.0, 1.50), ncol=3, columnspacing=0.4)

# ── Remove top/right spines on panels a, b, c (d subplots handled above) ──
for ax in [ax_a, ax_b, ax_c]:
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

fig.subplots_adjust(left=0.10, right=0.96, top=0.95, bottom=0.10,
                    hspace=0.50, wspace=0.40)
plt.savefig(OUT / 'fig5.pdf', bbox_inches='tight', dpi=600)
plt.savefig(OUT / 'fig5.png', bbox_inches='tight', dpi=600)
plt.close()
print("Saved fig5.pdf and fig5.png")
