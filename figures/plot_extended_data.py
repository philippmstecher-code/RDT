#!/usr/bin/env python3
"""Extended Data Figures ED1–ED10.

Generates all 10 Extended Data figures from consolidated_findings.json and
auxiliary data files, matching the main figure style (600 dpi, Helvetica,
180 mm max width, ≥7 pt font).

ED1:  Within-checkpoint SAE control (false-positive validation)
ED2:  Null baseline calibration (permutation test)
ED3:  Feature matching validation & threshold sensitivity
ED4:  Reconstruction quality across checkpoints and layers
ED5:  Layer-wise stability heatmaps (3 architectures)
ED6:  Architecture-specific temporal profiles (all seeds)
ED7:  Robustness: expansion factor (4× vs 8×) & training duration (50 vs 200 ep)
ED8:  Selectivity index distributions across training
ED9:  Per-superclass process breakdowns (20 CIFAR-100 superclasses)
ED10: Training dynamics & dataset statistics
ED11: Label noise dose-response & compensatory dynamics
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from pathlib import Path
import sys

# Add scripts dir to path for epoch_labels
sys.path.insert(0, str(Path(__file__).resolve().parent))
from epoch_labels import format_epoch_ticks

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA = DATA_DIR / "consolidated_findings.json"
OUT = ROOT / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

with open(DATA) as f:
    data = json.load(f)

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
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})

# ── Consistent color palette ──
HYPO_COLORS = {
    'Tg-E': '#8B5CF6', 'Ab-E': '#3B82F6', 'Di-E': '#F59E0B',
    'As-E': '#0891B2', 'De-E': '#EF4444',
}
ARCH_COLORS = {'ResNet-18': '#2C5F8A', 'ViT-Small': '#2A9D8F', 'CCT-7': '#E76F51'}
INDEX_COLORS = {'SSI': '#3B82F6', 'CSI': '#F59E0B', 'SAI': '#8B5CF6'}
LAYER_CMAP = plt.cm.viridis

FIG_W = 180 / 25.4   # 180 mm in inches
FIG_H_SINGLE = 90 / 25.4   # ~90 mm for 2-panel rows
FIG_H_TALL = 170 / 25.4    # ~170 mm for multi-row figures


def abbreviate_layer(name):
    """Shorten long layer names for readability in heatmaps."""
    name = name.replace('encoder.layers.', 'enc.')
    name = name.replace('encoder.ln', 'enc.ln')
    name = name.replace('transformer.layers.', 'tf.')
    name = name.replace('norm', 'norm')
    return name


def short_transition(t):
    """Shorten transition labels like '10->terminal' to '10->T'."""
    return t.replace('->terminal', '->T').replace('->', '-')


def get_iteration_ticks(lane_key, mode='checkpoint'):
    """Get iteration values (in thousands) for checkpoint or transition axes.

    Args:
        lane_key: key into data['lanes']
        mode: 'checkpoint' for per-checkpoint axes, 'transition' for per-transition axes

    Returns:
        iter_vals: list of iteration values in raw units
        iter_k: list of iteration values in thousands (for labelling)
    """
    ei = data['lanes'][lane_key]['epoch_info']
    wu = ei['checkpoint_weight_updates']
    if mode == 'transition':
        # Transitions go from checkpoint i to checkpoint i+1;
        # use the destination checkpoint's iteration count
        iter_vals = wu[1:]  # skip the initial checkpoint
    else:
        iter_vals = wu
    iter_k = [v / 1000.0 for v in iter_vals]
    return iter_vals, iter_k


def set_iteration_xaxis(ax, iter_k, every_n=2):
    """Set x-axis ticks and label using iteration values in thousands.

    Args:
        ax: matplotlib axes
        iter_k: list of iteration values in thousands
        every_n: show every Nth tick label
    """
    x = np.arange(len(iter_k))
    tick_pos = list(range(0, len(iter_k), every_n))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f'{iter_k[i]:.1f}' if iter_k[i] < 10 else f'{iter_k[i]:.0f}'
                        for i in tick_pos], fontsize=7)
    ax.set_xlabel('Iterations (\u00d710\u00b3)')

# ── Architecture configs ──
ARCH_CONFIGS = {
    'ResNet-18': {
        'lanes': ['ResNet18-CIFAR100-seed42', 'ResNet18-CIFAR100-seed137', 'ResNet18-CIFAR100-seed256'],
        'primary': 'ResNet18-CIFAR100-seed42',
        'layer_order': ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool'],
        'layer_labels': ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool'],
    },
    'ViT-Small': {
        'lanes': ['ViTSmall-CIFAR100-seed42', 'ViTSmall-CIFAR100-seed137', 'ViTSmall-CIFAR100-seed256'],
        'primary': 'ViTSmall-CIFAR100-seed42',
    },
    'CCT-7': {
        'lanes': ['CCT7-CIFAR100-seed42', 'CCT7-CIFAR100-seed137', 'CCT7-CIFAR100-seed256'],
        'primary': 'CCT7-CIFAR100-seed42',
    },
}

def get_layer_order(lane_key):
    """Get layer order for a lane, sorted by depth."""
    meta = data['lanes'][lane_key]['metadata']
    layers = list(meta['layers'])

    def layer_sort_key(name):
        """Sort layers by numeric index, with special layers last."""
        import re
        m = re.search(r'\.(\d+)$', name)
        if m:
            return (0, int(m.group(1)))
        # Non-numbered layers (avgpool, norm, encoder.ln) go last
        return (1, 0)

    layers.sort(key=layer_sort_key)
    return layers

def save_fig(fig, name):
    """Save figure as both PNG and PDF."""
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f"{name}.{ext}", dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    print(f"  Saved {name}.png and {name}.pdf")
    plt.close(fig)


def add_grid(ax):
    """Disable gridlines for clean journal figures."""
    ax.grid(False)


# ══════════════════════════════════════════════════════════════════════════════
# ED1: Methodology Validation (within-checkpoint control + feature matching)
# ══════════════════════════════════════════════════════════════════════════════
def plot_ed1_merged():
    print("Plotting ED1: Methodology validation...")
    fig = plt.figure(figsize=(FIG_W, FIG_H_TALL))
    gs = GridSpec(2, 3, figure=fig, wspace=0.4, hspace=0.45,
                  left=0.08, right=0.96, bottom=0.08, top=0.94)

    lane = data['lanes']['ResNet18-CIFAR100-seed42']
    wcc = lane['within_checkpoint_control']['per_checkpoint']
    layers = get_layer_order('ResNet18-CIFAR100-seed42')

    # Panel (a): False birth & death rates per checkpoint (averaged over layers)
    ax_a = fig.add_subplot(gs[0, 0])
    checkpoints = [entry['checkpoint'] for entry in wcc]
    false_births = []
    false_deaths = []
    for entry in wcc:
        fb = np.mean([l['false_birth_rate'] for l in entry['per_layer']])
        fd = np.mean([l['false_death_rate'] for l in entry['per_layer']])
        false_births.append(fb * 100)
        false_deaths.append(fd * 100)

    x = np.arange(len(checkpoints))
    w = 0.35
    ax_a.bar(x - w/2, false_births, w, color=HYPO_COLORS['Ab-E'], alpha=0.8, label='False birth', edgecolor='white', linewidth=0.3)
    ax_a.bar(x + w/2, false_deaths, w, color=HYPO_COLORS['De-E'], alpha=0.8, label='False death', edgecolor='white', linewidth=0.3)
    ax_a.set_xlabel('Checkpoint (epoch)')
    ax_a.set_ylabel('False-positive rate (%)')
    # Show all checkpoint ticks
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([str(c) for c in checkpoints],
                          fontsize=7, rotation=30, ha='right')
    ax_a.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7, fontsize=7)
    ax_a.set_title('False-positive rates', fontsize=8)
    ax_a.axhline(2, color='grey', ls='--', lw=0.5, alpha=0.5)
    ax_a.axhline(10, color='grey', ls='--', lw=0.5, alpha=0.5)
    ax_a.text(0.98, 0.95, 'Mean birth <2%, mean death <10%', transform=ax_a.transAxes,
              fontsize=6.5, ha='right', va='top', color='black', style='italic')
    add_grid(ax_a)
    ax_a.text(-0.22, 1.08, 'a', transform=ax_a.transAxes, fontsize=8, fontweight='bold', va='top')

    # Panel (b): Stable rate per layer (averaged over checkpoints)
    ax_b = fig.add_subplot(gs[0, 1])
    layer_stable = {l: [] for l in layers}
    for entry in wcc:
        for ldata in entry['per_layer']:
            if ldata['layer'] in layer_stable:
                layer_stable[ldata['layer']].append(ldata['stable_rate'])

    layer_means = [np.mean(layer_stable[l]) * 100 for l in layers]
    layer_stds = [np.std(layer_stable[l]) * 100 for l in layers]
    colors_layers = [LAYER_CMAP(i / (len(layers) - 1)) for i in range(len(layers))]

    ax_b.barh(range(len(layers)), layer_means, xerr=layer_stds, color=colors_layers,
              edgecolor='white', linewidth=0.3, capsize=2)
    ax_b.set_yticks(range(len(layers)))
    ax_b.set_yticklabels(layers)
    ax_b.set_xlabel('Stable rate (%)')
    ax_b.set_title('Within-checkpoint stability', fontsize=8)
    add_grid(ax_b)
    ax_b.text(-0.22, 1.08, 'b', transform=ax_b.transAxes, fontsize=8, fontweight='bold', va='top')

    # Panel (c): Mean match correlation per layer per checkpoint (heatmap)
    ax_c = fig.add_subplot(gs[0, 2])
    corr_matrix = np.zeros((len(layers), len(wcc)))
    for ci, entry in enumerate(wcc):
        for ldata in entry['per_layer']:
            if ldata['layer'] in layers:
                li = layers.index(ldata['layer'])
                corr_matrix[li, ci] = ldata['mean_match_corr']

    im = ax_c.imshow(corr_matrix, aspect='auto', cmap='YlGnBu', vmin=0.5, vmax=1.0)
    ax_c.set_xticks(range(0, len(wcc), 2))
    ax_c.set_xticklabels([wcc[i]['checkpoint'] for i in range(0, len(wcc), 2)])
    ax_c.set_yticks(range(len(layers)))
    ax_c.set_yticklabels(layers)
    ax_c.set_xlabel('Checkpoint (epoch)')
    ax_c.set_title('Mean match correlation', fontsize=8)
    plt.colorbar(im, ax=ax_c, shrink=0.8, label='Pearson r')
    ax_c.text(-0.22, 1.08, 'c', transform=ax_c.transAxes, fontsize=8, fontweight='bold', va='top')

    # ── Row 2: Feature matching validation (formerly ED3) ──
    fm = lane['feature_matching']

    # Panel (d): Born/died/stable/transformed across transitions
    ax_d = fig.add_subplot(gs[1, 0])
    transitions = sorted(fm.keys(), key=lambda t: int(t.split('->')[0]))
    stable_pct = [fm[t]['stable_pct'] * 100 for t in transitions]
    born_pct = [fm[t]['born_pct'] * 100 for t in transitions]
    died_pct = [fm[t]['died_pct'] * 100 for t in transitions]
    trans_pct = [100 - s - b - d for s, b, d in zip(stable_pct, born_pct, died_pct)]

    x_t = np.arange(len(transitions))
    ax_d.stackplot(x_t, stable_pct, trans_pct, born_pct, died_pct,
                   labels=['Stable', 'Transformed', 'Born', 'Died'],
                   colors=[HYPO_COLORS['As-E'], '#94A3B8', HYPO_COLORS['Ab-E'], HYPO_COLORS['De-E']],
                   alpha=0.8)
    ax_d.set_xlabel('Training transition')
    ax_d.set_ylabel('Fraction (%)')
    ax_d.set_xticks(x_t[::3])
    ax_d.set_xticklabels([short_transition(transitions[i]) for i in range(0, len(transitions), 3)],
                          fontsize=7, rotation=30, ha='right')
    ax_d.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7, fontsize=6, loc='center right')
    ax_d.set_title('Feature dynamics decomposition', fontsize=8)
    add_grid(ax_d)
    ax_d.text(-0.22, 1.08, 'd', transform=ax_d.transAxes, fontsize=8, fontweight='bold', va='top')

    # Panel (e): Per-layer born counts across transitions
    ax_e = fig.add_subplot(gs[1, 1])
    colors_layers2 = [LAYER_CMAP(i / (len(layers) - 1)) for i in range(len(layers))]
    for li, layer_name in enumerate(layers):
        born_counts = []
        for t in transitions:
            pl = fm[t].get('per_layer', {})
            if layer_name in pl:
                born_counts.append(pl[layer_name]['n_born'])
            else:
                born_counts.append(0)
        ax_e.plot(x_t, born_counts, 'o-', color=colors_layers2[li], markersize=3, lw=1,
                  label=layer_name, alpha=0.8)
    ax_e.set_xlabel('Training transition')
    ax_e.set_ylabel('Born feature count')
    ax_e.set_xticks(x_t[::3])
    ax_e.set_xticklabels([short_transition(transitions[i]) for i in range(0, len(transitions), 3)],
                          fontsize=7, rotation=30, ha='right')
    ax_e.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7, fontsize=6)
    ax_e.set_title('Per-layer feature births', fontsize=8)
    add_grid(ax_e)
    ax_e.text(-0.22, 1.08, 'e', transform=ax_e.transAxes, fontsize=8, fontweight='bold', va='top')

    # Panel (f): Ab-E/Di-E ratio trajectory
    ax_f = fig.add_subplot(gs[1, 2])
    ratios = lane['abh_dih_ratios']
    ratio_vals = [r['ratio'] for r in ratios]
    ratio_vals_clipped = [min(r, 60) for r in ratio_vals]

    ax_f.plot(range(len(ratio_vals_clipped)), ratio_vals_clipped, 'o-', color=HYPO_COLORS['Ab-E'],
              markersize=4, lw=1.0, label='Ab-E/Di-E ratio')
    ax_f.axhline(1, color='grey', ls='--', lw=0.8, alpha=0.5)
    ax_f.set_xlabel('Training transition')
    ax_f.set_ylabel('Ab-E / Di-E ratio')
    ax_f.set_xticks(range(0, len(ratio_vals), 2))
    if ratio_vals[0] > 60:
        ax_f.annotate(f'{ratio_vals[0]:.1f}', xy=(0, 60), fontsize=7,
                      ha='center', va='bottom', color='black')
    ax_f.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7, fontsize=7)
    ax_f.set_title('Ratio trajectory (ResNet-18)', fontsize=8)
    add_grid(ax_f)
    ax_f.text(-0.22, 1.08, 'f', transform=ax_f.transAxes, fontsize=8, fontweight='bold', va='top')

    save_fig(fig, 'ed_fig1')


def plot_ed1():
    """ED1: Methodology validation (within-checkpoint control + feature matching)."""
    plot_ed1_merged()


# ══════════════════════════════════════════════════════════════════════════════
# ED2: Null Baseline Calibration
# ══════════════════════════════════════════════════════════════════════════════
def plot_ed2():
    print("Plotting ED2: Null baseline calibration...")
    fig = plt.figure(figsize=(FIG_W, FIG_H_SINGLE))
    gs = GridSpec(1, 3, figure=fig, wspace=0.4, left=0.08, right=0.96, bottom=0.18, top=0.88)

    lane = data['lanes']['ResNet18-CIFAR100-seed42']
    nb = lane['null_baseline']['per_checkpoint']
    meta = lane['metadata']

    # Panel (a): Observed vs null SSI across checkpoints
    ax_a = fig.add_subplot(gs[0, 0])
    ckpts = [entry['checkpoint'] for entry in nb]
    obs_ssi = [entry['observed_ssi'] for entry in nb]
    null_ssi = [entry['null_ssi'] for entry in nb]

    ax_a.fill_between(range(len(ckpts)), null_ssi, alpha=0.3, color='#94A3B8', label='Null SSI (mean)')
    ax_a.plot(range(len(ckpts)), obs_ssi, 'o-', color=INDEX_COLORS['SSI'], markersize=3, lw=1.0, label='Observed SSI')
    ax_a.plot(range(len(ckpts)), null_ssi, 's--', color='#64748B', markersize=2, lw=0.8)
    ax_a.set_xlabel('Checkpoint (epoch)')
    ax_a.set_ylabel('Mean SSI')
    ax_a.set_xticks(range(0, len(ckpts), 2))
    ax_a.set_xticklabels([ckpts[i] for i in range(0, len(ckpts), 2)])
    ax_a.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7, fontsize=7)
    ax_a.set_title('Observed vs null SSI', fontsize=8)
    ax_a.text(0.98, 0.05, 'n = 1,000 permutations', transform=ax_a.transAxes,
              fontsize=6.5, ha='right', va='bottom', color='black', style='italic')
    add_grid(ax_a)
    ax_a.text(-0.22, 1.08, 'a', transform=ax_a.transAxes, fontsize=8, fontweight='bold', va='top')

    # Panel (b): Adaptive thresholds
    ax_b = fig.add_subplot(gs[0, 1])
    thresh_names = ['SSI', 'CSI', 'SAI']
    adaptive = [meta['adaptive_thresholds'].get('ssi_adaptive_thresh', 0.3),
                meta['adaptive_thresholds'].get('csi_adaptive_thresh', 0.4),
                meta['adaptive_thresholds'].get('sai_adaptive_thresh', 0.9)]
    floors = [0.1, 0.15, 0.5]
    x = np.arange(len(thresh_names))
    w = 0.35
    ax_b.bar(x - w/2, adaptive, w, color=[INDEX_COLORS[n] for n in thresh_names],
             alpha=0.8, label='Adaptive (p95 null)', edgecolor='white', linewidth=0.3)
    ax_b.bar(x + w/2, floors, w, color=[INDEX_COLORS[n] for n in thresh_names],
             alpha=0.3, label='Floor', edgecolor='grey', linewidth=0.3)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(thresh_names)
    ax_b.set_ylabel('Threshold value')
    ax_b.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7, fontsize=7)
    ax_b.set_title('Adaptive vs floor thresholds', fontsize=8)
    add_grid(ax_b)
    ax_b.text(-0.22, 1.08, 'b', transform=ax_b.transAxes, fontsize=8, fontweight='bold', va='top')

    # Panel (c): P-values across checkpoints
    ax_c = fig.add_subplot(gs[0, 2])
    pvals = [entry['p_value'] for entry in nb]
    # Replace 0 with a small value for log scale
    pvals_plot = [max(p, 1e-4) for p in pvals]
    ax_c.semilogy(range(len(ckpts)), pvals_plot, 'o-', color=INDEX_COLORS['SSI'], markersize=4, lw=1.0)
    ax_c.axhline(0.05, color='black', ls='--', lw=0.8, alpha=0.6, label='p = 0.05')
    ax_c.axhline(0.001, color='black', ls=':', lw=0.5, alpha=0.4, label='p = 0.001')
    ax_c.set_xlabel('Checkpoint (epoch)')
    ax_c.set_ylabel('p-value (permutation)')
    ax_c.set_xticks(range(0, len(ckpts), 2))
    ax_c.set_xticklabels([ckpts[i] for i in range(0, len(ckpts), 2)])
    ax_c.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7, fontsize=7)
    ax_c.set_title('Null baseline p-values', fontsize=8)
    ax_c.set_ylim(5e-5, 1.5)
    ax_c.text(-0.22, 1.08, 'c', transform=ax_c.transAxes, fontsize=8, fontweight='bold', va='top')

    save_fig(fig, 'ed_fig2')


# ══════════════════════════════════════════════════════════════════════════════
# ED3: Feature Matching Validation & Threshold Sensitivity
# ══════════════════════════════════════════════════════════════════════════════
def plot_ed3():
    print("Plotting ED3: Feature matching & threshold sensitivity...")
    fig = plt.figure(figsize=(FIG_W, FIG_H_SINGLE))
    gs = GridSpec(1, 3, figure=fig, wspace=0.4, left=0.08, right=0.96, bottom=0.18, top=0.88)

    lane = data['lanes']['ResNet18-CIFAR100-seed42']
    fm = lane['feature_matching']

    # Panel (a): Born/died/stable/transformed across transitions
    ax_a = fig.add_subplot(gs[0, 0])
    transitions = sorted(fm.keys(), key=lambda t: int(t.split('->')[0]))
    stable_pct = [fm[t]['stable_pct'] * 100 for t in transitions]
    born_pct = [fm[t]['born_pct'] * 100 for t in transitions]
    died_pct = [fm[t]['died_pct'] * 100 for t in transitions]
    # transformed is what's left
    trans_pct = [100 - s - b - d for s, b, d in zip(stable_pct, born_pct, died_pct)]

    x = np.arange(len(transitions))
    ax_a.stackplot(x, stable_pct, trans_pct, born_pct, died_pct,
                   labels=['Stable', 'Transformed', 'Born', 'Died'],
                   colors=[HYPO_COLORS['As-E'], '#94A3B8', HYPO_COLORS['Ab-E'], HYPO_COLORS['De-E']],
                   alpha=0.8)
    ax_a.set_xlabel('Training transition')
    ax_a.set_ylabel('Fraction (%)')
    ax_a.set_xticks(x[::3])
    ax_a.set_xticklabels([short_transition(transitions[i]) for i in range(0, len(transitions), 3)],
                          fontsize=7, rotation=30, ha='right')
    ax_a.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7, fontsize=7, loc='center right')
    ax_a.set_title('Feature dynamics decomposition', fontsize=8)
    add_grid(ax_a)
    ax_a.text(-0.22, 1.08, 'a', transform=ax_a.transAxes, fontsize=8, fontweight='bold', va='top')

    # Panel (b): Per-layer born counts across transitions
    ax_b = fig.add_subplot(gs[0, 1])
    layers = get_layer_order('ResNet18-CIFAR100-seed42')
    colors_layers = [LAYER_CMAP(i / (len(layers) - 1)) for i in range(len(layers))]
    for li, layer in enumerate(layers):
        born_counts = []
        for t in transitions:
            pl = fm[t].get('per_layer', {})
            if layer in pl:
                born_counts.append(pl[layer]['n_born'])
            else:
                born_counts.append(0)
        ax_b.plot(x, born_counts, 'o-', color=colors_layers[li], markersize=3, lw=1,
                  label=layer, alpha=0.8)
    ax_b.set_xlabel('Training transition')
    ax_b.set_ylabel('Born feature count')
    ax_b.set_xticks(x[::3])
    ax_b.set_xticklabels([short_transition(transitions[i]) for i in range(0, len(transitions), 3)],
                          fontsize=7, rotation=30, ha='right')
    ax_b.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7, fontsize=7)
    ax_b.set_title('Per-layer feature births', fontsize=8)
    add_grid(ax_b)
    ax_b.text(-0.22, 1.08, 'b', transform=ax_b.transAxes, fontsize=8, fontweight='bold', va='top')

    # Panel (c): Ab-E/Di-E ratio sensitivity to matching threshold
    # We simulate by showing the ratio trajectory and stable/born/died at different implied thresholds
    ax_c = fig.add_subplot(gs[0, 2])
    ratios = lane['abh_dih_ratios']
    ratio_vals = [r['ratio'] for r in ratios]
    ratio_vals_clipped = [min(r, 60) for r in ratio_vals]  # clip for display

    ax_c.plot(range(len(ratio_vals_clipped)), ratio_vals_clipped, 'o-', color=HYPO_COLORS['Ab-E'],
              markersize=4, lw=1.0, label='Ab-E/Di-E ratio')
    ax_c.axhline(1, color='grey', ls='--', lw=0.8, alpha=0.5)
    ax_c.set_xlabel('Training transition')
    ax_c.set_ylabel('Ab-E / Di-E ratio')
    ax_c.set_xticks(range(0, len(ratio_vals), 2))

    # Add annotation for first value
    if ratio_vals[0] > 60:
        ax_c.annotate(f'{ratio_vals[0]:.1f}', xy=(0, 60), fontsize=7,
                      ha='center', va='bottom', color='black')

    ax_c.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7, fontsize=7)
    ax_c.set_title('Ratio trajectory (ResNet-18)', fontsize=8)
    add_grid(ax_c)
    ax_c.text(-0.22, 1.08, 'c', transform=ax_c.transAxes, fontsize=8, fontweight='bold', va='top')

    save_fig(fig, 'ed_fig3')


# ══════════════════════════════════════════════════════════════════════════════
# ED4: Reconstruction Quality
# ══════════════════════════════════════════════════════════════════════════════
def plot_ed4():
    print("Plotting ED4: Reconstruction quality...")
    fig = plt.figure(figsize=(FIG_W, FIG_H_SINGLE * 1.15))
    gs = GridSpec(1, 3, figure=fig, wspace=0.55, left=0.08, right=0.96, bottom=0.15, top=0.88)

    # Panel (a): Cosine similarity heatmap (ResNet-18 seed 42)
    ax_a = fig.add_subplot(gs[0, 0])
    lane = data['lanes']['ResNet18-CIFAR100-seed42']
    rq = lane['reconstruction_quality']
    layers = get_layer_order('ResNet18-CIFAR100-seed42')
    checkpoints = sorted(rq.keys(), key=lambda x: int(x) if x.isdigit() else 999)

    cos_matrix = np.zeros((len(layers), len(checkpoints)))
    for ci, ckpt in enumerate(checkpoints):
        for li, layer in enumerate(layers):
            if layer in rq[ckpt]:
                cos_matrix[li, ci] = rq[ckpt][layer]['cosine_sim']

    im = ax_a.imshow(cos_matrix, aspect='auto', cmap='RdYlBu', vmin=0.98, vmax=1.0)
    ax_a.set_xticks(range(0, len(checkpoints), 2))
    ax_a.set_xticklabels([checkpoints[i] for i in range(0, len(checkpoints), 2)])
    ax_a.set_yticks(range(len(layers)))
    ax_a.set_yticklabels(layers)
    ax_a.set_xlabel('Checkpoint (epoch)')
    ax_a.set_title('Cosine similarity\n(ResNet-18)', fontsize=8)
    plt.colorbar(im, ax=ax_a, shrink=0.7, label='cos(x, x̂)')
    ax_a.text(-0.25, 1.08, 'a', transform=ax_a.transAxes, fontsize=8, fontweight='bold', va='top')

    # Panel (b): MSE heatmap (ResNet-18 seed 42)
    ax_b = fig.add_subplot(gs[0, 1])
    mse_matrix = np.zeros((len(layers), len(checkpoints)))
    for ci, ckpt in enumerate(checkpoints):
        for li, layer in enumerate(layers):
            if layer in rq[ckpt]:
                mse_matrix[li, ci] = rq[ckpt][layer]['mse']

    im2 = ax_b.imshow(mse_matrix, aspect='auto', cmap='YlOrRd')
    ax_b.set_xticks(range(0, len(checkpoints), 2))
    ax_b.set_xticklabels([checkpoints[i] for i in range(0, len(checkpoints), 2)])
    ax_b.set_yticks(range(len(layers)))
    ax_b.set_yticklabels(layers)
    ax_b.set_xlabel('Checkpoint (epoch)')
    ax_b.set_title('Reconstruction MSE\n(ResNet-18)', fontsize=8)
    plt.colorbar(im2, ax=ax_b, shrink=0.7, label='MSE')
    ax_b.text(-0.25, 1.08, 'b', transform=ax_b.transAxes, fontsize=8, fontweight='bold', va='top')

    # Panel (c): Cross-architecture cosine similarity comparison
    ax_c = fig.add_subplot(gs[0, 2])
    for arch_name, arch_conf in ARCH_CONFIGS.items():
        primary = arch_conf['primary']
        if primary not in data['lanes']:
            continue
        lane_d = data['lanes'][primary]
        rq_d = lane_d['reconstruction_quality']
        ckpts_d = sorted(rq_d.keys(), key=lambda x: int(x) if x.isdigit() else 999)
        layers_d = get_layer_order(primary)
        # Average cosine sim across layers per checkpoint
        avg_cos = []
        for ckpt in ckpts_d:
            vals = [rq_d[ckpt][l]['cosine_sim'] for l in layers_d if l in rq_d[ckpt]]
            avg_cos.append(np.mean(vals) if vals else 0)
        ax_c.plot(range(len(avg_cos)), avg_cos, 'o-', color=ARCH_COLORS[arch_name],
                  markersize=3, lw=1.0, label=arch_name)

    ax_c.set_xlabel('Checkpoint (epoch)')
    ax_c.set_ylabel('Mean cosine similarity')
    ax_c.set_ylim(0.97, 1.001)
    ax_c.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7, fontsize=7)
    ax_c.set_title('Cross-architecture\nreconstruction', fontsize=8)
    add_grid(ax_c)
    ax_c.text(-0.25, 1.08, 'c', transform=ax_c.transAxes, fontsize=8, fontweight='bold', va='top')

    save_fig(fig, 'ed_fig4')


# ══════════════════════════════════════════════════════════════════════════════
# ED5: Layer-wise Stability Heatmaps (3 architectures)
# ══════════════════════════════════════════════════════════════════════════════
def plot_ed5():
    print("Plotting ED5: Layer-wise stability heatmaps...")
    fig = plt.figure(figsize=(FIG_W, FIG_H_TALL))
    gs = GridSpec(3, 3, figure=fig, wspace=0.45, hspace=0.55,
                  left=0.14, right=0.92, bottom=0.06, top=0.94)

    metrics = ['stable_rate', 'mean_ssi', 'mean_csi']
    metric_labels = ['Feature stability rate', 'Mean SSI', 'Mean CSI']
    metric_cmaps = ['YlGnBu', 'Blues', 'Oranges']
    arch_names = ['ResNet-18', 'ViT-Small', 'CCT-7']
    panel_labels = 'abcdefghi'

    for row, arch_name in enumerate(arch_names):
        primary = ARCH_CONFIGS[arch_name]['primary']
        if primary not in data['lanes']:
            continue
        lane_d = data['lanes'][primary]
        layers = get_layer_order(primary)
        layer_labels = [abbreviate_layer(l) for l in layers]
        fm = lane_d['feature_matching']
        fl = lane_d['feature_landscape']

        for col, (metric, mlabel, cmap) in enumerate(zip(metrics, metric_labels, metric_cmaps)):
            ax = fig.add_subplot(gs[row, col])

            if metric == 'stable_rate':
                transitions = sorted(fm.keys(), key=lambda t: int(t.split('->')[0]))
                mat = np.zeros((len(layers), len(transitions)))
                for ti, t in enumerate(transitions):
                    pl = fm[t].get('per_layer', {})
                    for li, layer in enumerate(layers):
                        if layer in pl:
                            total = pl[layer]['n_stable'] + pl[layer]['n_born'] + pl[layer]['n_died'] + pl[layer]['n_transformed']
                            mat[li, ti] = pl[layer]['n_stable'] / max(total, 1)
                im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=0, vmax=1)
                x_labels = [short_transition(t) for t in transitions]
            else:
                checkpoints = sorted(fl.keys(), key=lambda x: int(x) if x.isdigit() else 999)
                mat = np.zeros((len(layers), len(checkpoints)))
                for ci, ckpt in enumerate(checkpoints):
                    for li, layer in enumerate(layers):
                        if layer in fl[ckpt]:
                            mat[li, ci] = fl[ckpt][layer].get(metric, 0)
                vmax = 0.5 if metric == 'mean_ssi' else 0.8
                im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=0, vmax=vmax)
                x_labels = checkpoints

            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels(layer_labels, fontsize=7)
            ax.set_xticks(range(0, len(x_labels), 3))
            ax.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), 3)], fontsize=6.5)

            if row == 2:
                ax.set_xlabel('Checkpoint (epoch)' if metric != 'stable_rate' else 'Training transition', fontsize=7)
            if col == 0:
                ax.set_ylabel(arch_name, fontsize=8, fontweight='bold')

            ax.set_title(mlabel if row == 0 else '', fontsize=7)
            plt.colorbar(im, ax=ax, shrink=0.6, pad=0.03)

            pidx = row * 3 + col
            ax.text(-0.30, 1.12, panel_labels[pidx], transform=ax.transAxes,
                    fontsize=8, fontweight='bold', va='top')

    save_fig(fig, 'ed_fig5')


# ══════════════════════════════════════════════════════════════════════════════
# ED6: Architecture-Specific Temporal Profiles (All Seeds)
# ══════════════════════════════════════════════════════════════════════════════
def plot_ed6():
    print("Plotting ED6: Architecture-specific temporal profiles...")
    fig = plt.figure(figsize=(FIG_W, FIG_H_TALL))
    gs = GridSpec(3, 3, figure=fig, wspace=0.40, hspace=0.50,
                  left=0.12, right=0.96, bottom=0.06, top=0.92)

    arch_names = ['ResNet-18', 'ViT-Small', 'CCT-7']
    processes = ['ab_h', 'di_h', 'tg_h']
    proc_labels = ['Ab-E events', 'Di-E events', 'Tg-E events']
    proc_colors = [HYPO_COLORS['Ab-E'], HYPO_COLORS['Di-E'], HYPO_COLORS['Tg-E']]
    panel_labels = 'abcdefghi'

    for row, arch_name in enumerate(arch_names):
        lanes = ARCH_CONFIGS[arch_name]['lanes']

        for col, (proc, plabel, pcolor) in enumerate(zip(processes, proc_labels, proc_colors)):
            ax = fig.add_subplot(gs[row, col])

            for lane_key in lanes:
                if lane_key not in data['lanes']:
                    continue
                lane_d = data['lanes'][lane_key]
                pi = lane_d['process_intensity']
                seed = lane_d['metadata']['seed']
                wu_gaps = lane_d['epoch_info']['transition_weight_update_gaps']

                # Normalize: events per 1k weight updates
                vals = []
                for i, entry in enumerate(pi):
                    gap = wu_gaps[i] if i < len(wu_gaps) else wu_gaps[-1]
                    vals.append(entry[proc] / gap * 1000 if gap > 0 else 0)
                ax.plot(range(len(vals)), vals, 'o-', markersize=3, lw=0.8,
                        alpha=0.7, label=f'seed {seed}')

            # Transition index x-axis
            if row == 2:
                ax.set_xlabel('Training transition', fontsize=7)
            else:
                ax.set_xlabel('')
            if col == 0:
                ax.set_ylabel(f'{arch_name}\nEvents / 1k updates', fontsize=7, fontweight='bold')
            if row == 0:
                ax.set_title(f'{plabel} (n = 3 seeds)', fontsize=8)

            ax.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7, fontsize=6, loc='best')
            add_grid(ax)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(
                lambda x, _: f'{x/1000:.0f}k' if x >= 1000 else f'{x:.0f}'))

            pidx = row * 3 + col
            ax.text(-0.28, 1.12, panel_labels[pidx], transform=ax.transAxes,
                    fontsize=8, fontweight='bold', va='top')

    save_fig(fig, 'ed_fig6')


# ══════════════════════════════════════════════════════════════════════════════
# ED7: Robustness — Expansion Factor & Training Duration
# ══════════════════════════════════════════════════════════════════════════════
def plot_ed7():
    print("Plotting ED7: Expansion factor & training duration...")
    fig = plt.figure(figsize=(FIG_W, FIG_H_TALL * 0.7))
    gs = GridSpec(2, 3, figure=fig, wspace=0.4, hspace=0.5,
                  left=0.08, right=0.96, bottom=0.10, top=0.92)

    # Row 1: 4× vs 8× expansion
    lane_4x = data['lanes']['ResNet18-CIFAR100-seed42']
    lane_8x = data['lanes']['ResNet18-CIFAR100-8x']
    panel_labels = 'abcdef'

    # (a) Ab-E/Di-E ratio overlay
    ax = fig.add_subplot(gs[0, 0])
    ratios_4x = [min(r['ratio'], 60) for r in lane_4x['abh_dih_ratios']]
    ratios_8x = [min(r['ratio'], 60) for r in lane_8x['abh_dih_ratios']]
    ax.plot(range(len(ratios_4x)), ratios_4x, 'o-', color=ARCH_COLORS['ResNet-18'],
            markersize=3, lw=1.0, label='4× expansion')
    ax.plot(range(len(ratios_8x)), ratios_8x, 's--', color='#F472B6',
            markersize=3, lw=1.0, label='8× expansion')
    ax.axhline(1, color='grey', ls='--', lw=0.5)
    ax.set_ylabel('Ab-E / Di-E ratio')
    ax.set_xlabel('Training transition')
    ax.set_title('Expansion factor: ratio', fontsize=8)
    ax.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7, fontsize=7)
    add_grid(ax)
    ax.text(-0.22, 1.10, 'a', transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')

    # (b) Process intensity overlay
    ax = fig.add_subplot(gs[0, 1])
    proc_labels_map = {'ab_h': 'Ab-E', 'di_h': 'Di-E'}
    for proc, color in [('ab_h', HYPO_COLORS['Ab-E']), ('di_h', HYPO_COLORS['Di-E'])]:
        plabel = proc_labels_map[proc]
        vals_4x = [e[proc] for e in lane_4x['process_intensity']]
        vals_8x = [e[proc] for e in lane_8x['process_intensity']]
        ax.plot(range(len(vals_4x)), vals_4x, 'o-', color=color, markersize=3, lw=1, alpha=0.7,
                label=f'{plabel} 4\u00d7')
        ax.plot(range(len(vals_8x)), vals_8x, 's--', color=color, markersize=3, lw=1, alpha=0.5,
                label=f'{plabel} 8\u00d7')
    ax.set_ylabel('Event count')
    ax.set_xlabel('Training transition')
    ax.set_title('Expansion factor: processes', fontsize=8)
    ax.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7, fontsize=6)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f'{x/1000:.0f}k' if x >= 1000 else f'{x:.0f}'))
    add_grid(ax)
    ax.text(-0.22, 1.10, 'b', transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')

    # (c) Churn rate overlay
    ax = fig.add_subplot(gs[0, 2])
    churn_4x = [e['churn'] * 100 for e in lane_4x['process_intensity']]
    churn_8x = [e['churn'] * 100 for e in lane_8x['process_intensity']]
    ax.plot(range(len(churn_4x)), churn_4x, 'o-', color=ARCH_COLORS['ResNet-18'],
            markersize=3, lw=1.0, label='4× expansion')
    ax.plot(range(len(churn_8x)), churn_8x, 's--', color='#F472B6',
            markersize=3, lw=1.0, label='8× expansion')
    ax.set_ylabel('Churn rate (%)')
    ax.set_xlabel('Training transition')
    ax.set_title('Expansion factor: churn', fontsize=8)
    ax.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7, fontsize=7)
    add_grid(ax)
    ax.text(-0.22, 1.10, 'c', transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')

    # Row 2: 50 ep vs 200 ep
    lane_200 = data['lanes']['ResNet18-CIFAR100-200ep']

    # (d) Ab-E/Di-E ratio
    ax = fig.add_subplot(gs[1, 0])
    ratios_200 = [min(r['ratio'], 60) for r in lane_200['abh_dih_ratios']]
    ax.plot(range(len(ratios_4x)), ratios_4x, 'o-', color=ARCH_COLORS['ResNet-18'],
            markersize=3, lw=1.0, label='50 epochs')
    ax.plot(range(len(ratios_200)), ratios_200, 's--', color='#7C3AED',
            markersize=3, lw=1.0, label='200 epochs')
    ax.axhline(1, color='grey', ls='--', lw=0.5)
    ax.set_ylabel('Ab-E / Di-E ratio')
    ax.set_xlabel('Training transition')
    ax.set_title('Training duration: ratio', fontsize=8)
    ax.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7, fontsize=7)
    add_grid(ax)
    ax.text(-0.22, 1.10, 'd', transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')

    # (e) SSI/CSI trajectories
    ax = fig.add_subplot(gs[1, 1])
    se_50 = lane_4x['selectivity_evolution']
    se_200 = lane_200['selectivity_evolution']
    ssi_50 = [e['mean_ssi'] for e in se_50]
    csi_50 = [e['mean_csi'] for e in se_50]
    ssi_200 = [e['mean_ssi'] for e in se_200]
    csi_200 = [e['mean_csi'] for e in se_200]
    ax.plot(range(len(ssi_50)), ssi_50, 'o-', color=INDEX_COLORS['SSI'], markersize=3, lw=1, label='SSI 50ep')
    ax.plot(range(len(csi_50)), csi_50, 'o-', color=INDEX_COLORS['CSI'], markersize=3, lw=1, label='CSI 50ep')
    ax.plot(range(len(ssi_200)), ssi_200, 's--', color=INDEX_COLORS['SSI'], markersize=3, lw=1, alpha=0.6, label='SSI 200ep')
    ax.plot(range(len(csi_200)), csi_200, 's--', color=INDEX_COLORS['CSI'], markersize=3, lw=1, alpha=0.6, label='CSI 200ep')
    ax.set_ylabel('Mean index value')
    ax.set_xlabel('Checkpoint (epoch)')
    ax.set_title('Training duration: selectivity', fontsize=8)
    ax.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7, fontsize=6)
    add_grid(ax)
    ax.text(-0.22, 1.10, 'e', transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')

    # (f) Churn rate
    ax = fig.add_subplot(gs[1, 2])
    churn_200 = [e['churn'] * 100 for e in lane_200['process_intensity']]
    ax.plot(range(len(churn_4x)), churn_4x, 'o-', color=ARCH_COLORS['ResNet-18'],
            markersize=3, lw=1.0, label='50 epochs')
    ax.plot(range(len(churn_200)), churn_200, 's--', color='#7C3AED',
            markersize=3, lw=1.0, label='200 epochs')
    ax.set_ylabel('Churn rate (%)')
    ax.set_xlabel('Training transition')
    ax.set_title('Training duration: churn', fontsize=8)
    ax.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7, fontsize=7)
    add_grid(ax)
    ax.text(-0.22, 1.10, 'f', transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')

    save_fig(fig, 'ed_fig7')


# ══════════════════════════════════════════════════════════════════════════════
# ED8: Selectivity Index Distributions Across Training
# ══════════════════════════════════════════════════════════════════════════════
def plot_ed8():
    print("Plotting ED8: Selectivity index distributions...")
    fig = plt.figure(figsize=(FIG_W, FIG_H_TALL * 0.7))
    gs = GridSpec(2, 3, figure=fig, wspace=0.4, hspace=0.5,
                  left=0.08, right=0.96, bottom=0.10, top=0.92)

    lane = data['lanes']['ResNet18-CIFAR100-seed42']
    fl = lane['feature_landscape']
    checkpoints = sorted(fl.keys(), key=lambda x: int(x) if x.isdigit() else 999)
    layers = get_layer_order('ResNet18-CIFAR100-seed42')
    _, iter_k_ckpt = get_iteration_ticks('ResNet18-CIFAR100-seed42', mode='checkpoint')
    panel_labels = 'abcdef'

    colors_layers = [LAYER_CMAP(i / (len(layers) - 1)) for i in range(len(layers))]

    def _layer_style(layer):
        """Return (linestyle, linewidth, alpha, markersize) — de-emphasise avgpool."""
        if layer == 'avgpool':
            return '--', 0.7, 0.4, 2
        return '-', 1, 0.8, 3

    # Row 1: SSI, CSI, SAI evolution per layer
    for col, (metric, mlabel, color) in enumerate([
        ('mean_ssi', 'Mean SSI', INDEX_COLORS['SSI']),
        ('mean_csi', 'Mean CSI', INDEX_COLORS['CSI']),
        ('mean_sai', 'Mean SAI', INDEX_COLORS['SAI']),
    ]):
        ax = fig.add_subplot(gs[0, col])
        for li, layer in enumerate(layers):
            vals = [fl[ckpt][layer].get(metric, 0) for ckpt in checkpoints if layer in fl[ckpt]]
            ls, lw, alpha, ms = _layer_style(layer)
            ax.plot(range(len(vals)), vals, marker='o', linestyle=ls, color=colors_layers[li],
                    markersize=ms, lw=lw, label=layer, alpha=alpha)
        set_iteration_xaxis(ax, iter_k_ckpt[:len(checkpoints)], every_n=2)
        ax.set_ylabel(mlabel)
        ax.set_title(f'{mlabel} per layer', fontsize=8)
        ax.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7, fontsize=6)
        add_grid(ax)
        ax.text(-0.22, 1.10, panel_labels[col], transform=ax.transAxes,
                fontsize=8, fontweight='bold', va='top')

    # Row 2: n_alive, n_high_ssi, n_high_sai per layer
    for col, (metric, mlabel) in enumerate([
        ('n_alive', 'Alive features'),
        ('n_high_ssi', 'High-SSI features'),
        ('n_high_sai', 'Task-general features'),
    ]):
        ax = fig.add_subplot(gs[1, col])
        for li, layer in enumerate(layers):
            vals = [fl[ckpt][layer].get(metric, 0) for ckpt in checkpoints if layer in fl[ckpt]]
            ls, lw, alpha, ms = _layer_style(layer)
            ax.plot(range(len(vals)), vals, marker='o', linestyle=ls, color=colors_layers[li],
                    markersize=ms, lw=lw, label=layer, alpha=alpha)
        set_iteration_xaxis(ax, iter_k_ckpt[:len(checkpoints)], every_n=2)
        ax.set_ylabel('Feature count')
        ax.set_title(mlabel, fontsize=8)
        ax.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7, fontsize=6)
        add_grid(ax)
        ax.text(-0.22, 1.10, panel_labels[col + 3], transform=ax.transAxes,
                fontsize=8, fontweight='bold', va='top')

    save_fig(fig, 'ed_fig8')


# ══════════════════════════════════════════════════════════════════════════════
# ED9: Per-Superclass Process Breakdowns (20 CIFAR-100 superclasses)
# ══════════════════════════════════════════════════════════════════════════════
def plot_ed9():
    print("Plotting ED9: Cross-architecture superclass process invariance...")

    # Two groups at natural scales:
    #   (a) Dominant pair: As-E vs De-E
    #   (b) Minor processes: Tg-E vs Ab-E vs Di-E
    # Column definitions — each is (list_of_keys_to_sum, label)
    group_a = [(['as_h'], 'As-E'), (['de_h'], 'De-E')]
    group_b = [(['tg_h'], 'Tg-E'), (['ab_h'], 'Ab-E'), (['di_h'], 'Di-E')]

    # Identify CIFAR-100 standard lanes per architecture
    arch_lanes = {}
    for label in data['lanes']:
        meta = data['lanes'][label]['metadata']
        if meta['dataset'] != 'CIFAR-100' or meta['epochs'] != 50 or meta['expansion'] != '4x':
            continue
        arch_lanes.setdefault(meta['architecture'], []).append(label)

    arch_names = ['ResNet-18', 'ViT-Small', 'CCT-7']
    superclasses = sorted(data['lanes'][arch_lanes['ResNet-18'][0]]['superclass_summary'].keys())
    sc_labels = [sc.replace('_', ' ') for sc in superclasses]
    sc_labels = [s if len(s) <= 18 else ' '.join(s.split()[:2]) + '...' for s in sc_labels]
    n_sc = len(superclasses)

    def _build_matrices(proc_group):
        """Return (mean_matrix, sd_matrix) for a group of process columns.
        Each column is defined as (list_of_keys_to_sum, label)."""
        n_p = len(proc_group)
        arch_fracs = {}
        for arch in arch_names:
            fracs = np.zeros((n_sc, n_p))
            for si, sc in enumerate(superclasses):
                for pi, (proc_keys, _) in enumerate(proc_group):
                    # Sum the listed process keys for each lane, then mean across seeds
                    lane_sums = []
                    for lb in arch_lanes[arch]:
                        pf = data['lanes'][lb]['superclass_summary'][sc]['process_fractions']
                        lane_sums.append(sum(pf.get(k, 0) for k in proc_keys) * 100)
                    fracs[si, pi] = np.mean(lane_sums)
            arch_fracs[arch] = fracs

        mean_mat = np.zeros((n_sc, n_p))
        sd_mat = np.zeros((n_sc, n_p))
        for si in range(n_sc):
            for pi in range(n_p):
                arch_vals = [arch_fracs[a][si, pi] for a in arch_names]
                mean_mat[si, pi] = np.mean(arch_vals)
                sd_mat[si, pi] = np.std(arch_vals)
        return mean_mat, sd_mat

    mean_a, sd_a = _build_matrices(group_a)
    mean_b, sd_b = _build_matrices(group_b)

    # ── Figure: 2×2 grid — top row = means, bottom row = SDs ──
    # Use a wider layout with explicit colorbar axes to avoid overlap
    fig = plt.figure(figsize=(FIG_W, FIG_H_TALL * 0.85))

    # Manual axes placement for precise control
    # Columns: left (2 procs) narrower, right (3 procs) wider
    # Each column gets its own colorbar
    row_h = 0.38
    row_gap = 0.08
    top = 0.93
    left_margin = 0.17
    col_gap = 0.12
    cb_w = 0.012
    cb_pad = 0.008

    # Column widths proportional to process count
    avail_w = 0.95 - left_margin - 2 * (cb_w + cb_pad) - col_gap
    w_a = avail_w * 2 / 5
    w_b = avail_w * 3 / 5

    # Positions
    x_a = left_margin
    x_cb_a = x_a + w_a + cb_pad
    x_b = x_cb_a + cb_w + col_gap
    x_cb_b = x_b + w_b + cb_pad

    y_top = top - row_h
    y_bot = y_top - row_gap - row_h

    def _draw_heatmap(ax, matrix, proc_group, cmap, vmin, vmax, show_ylabels, title, panel_label, cb_ax):
        n_p = len(proc_group)
        im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                        interpolation='nearest')
        ax.set_xticks(range(n_p))
        ax.set_xticklabels([lbl for _, lbl in proc_group], fontsize=7, fontweight='medium')
        ax.set_yticks(range(n_sc))
        if show_ylabels:
            ax.set_yticklabels(sc_labels, fontsize=6)
        else:
            ax.set_yticklabels([])
        ax.set_title(title, fontsize=7.5, pad=6)
        ax.text(-0.04 if not show_ylabels else -0.30, 1.04, panel_label,
                transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')
        # Thin cell borders
        ax.set_xticks([x - 0.5 for x in range(1, n_p)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, n_sc)], minor=True)
        ax.tick_params(which='minor', length=0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
            spine.set_color('#9CA3AF')
        # Annotate cells
        thresh = vmin + (vmax - vmin) * 0.45
        for si in range(n_sc):
            for pi in range(n_p):
                val = matrix[si, pi]
                color = 'white' if val > thresh else '#374151'
                ax.text(pi, si, f'{val:.1f}', ha='center', va='center',
                        fontsize=5, color=color, fontweight='medium')
        # Colorbar
        cb = fig.colorbar(im, cax=cb_ax, orientation='vertical')
        cb.ax.tick_params(labelsize=6, width=0.4, length=2)
        cb.outline.set_linewidth(0.4)
        return im

    # Create axes
    ax_a = fig.add_axes([x_a, y_top, w_a, row_h])
    cb_ax_a_top = fig.add_axes([x_cb_a, y_top, cb_w, row_h])
    ax_b = fig.add_axes([x_b, y_top, w_b, row_h])
    cb_ax_b_top = fig.add_axes([x_cb_b, y_top, cb_w, row_h])

    ax_c = fig.add_axes([x_a, y_bot, w_a, row_h])
    cb_ax_c = fig.add_axes([x_cb_a, y_bot, cb_w, row_h])
    ax_d = fig.add_axes([x_b, y_bot, w_b, row_h])
    cb_ax_d = fig.add_axes([x_cb_b, y_bot, cb_w, row_h])

    # Row 1: Mean fractions
    _draw_heatmap(ax_a, mean_a, group_a,
                  'YlOrRd', 0, 65, True,
                  'Dominant processes (%)', 'a', cb_ax_a_top)
    _draw_heatmap(ax_b, mean_b, group_b,
                  'YlOrRd', 0, 12, False,
                  'Minor processes (%)', 'b', cb_ax_b_top)

    # Row 2: Cross-architecture SD
    sd_vmax = max(4, np.max(np.concatenate([sd_a.ravel(), sd_b.ravel()])) * 1.2)
    _draw_heatmap(ax_c, sd_a, group_a,
                  'Blues', 0, sd_vmax, True,
                  'Cross-architecture SD (pp)', 'c', cb_ax_c)
    _draw_heatmap(ax_d, sd_b, group_b,
                  'Blues', 0, sd_vmax, False,
                  'Cross-architecture SD (pp)', 'd', cb_ax_d)

    save_fig(fig, 'ed_fig9')


# ══════════════════════════════════════════════════════════════════════════════
# ED10: Training Dynamics & Independent Init Control
# ══════════════════════════════════════════════════════════════════════════════
def plot_ed10():
    print("Plotting ED10: Independent init control...")
    fig = plt.figure(figsize=(FIG_W * 0.45, FIG_H_SINGLE * 1.15))
    fig.subplots_adjust(left=0.18, right=0.94, bottom=0.15, top=0.88)

    # Single panel: Independent init control — Ab-E/Di-E ratio
    ax = fig.add_subplot(111)
    _, iter_k_trans = get_iteration_ticks('ResNet18-CIFAR100-seed42', mode='transition')
    # Load independent init from raw lanes
    init_ctrl_path = DATA_DIR / "raw_lanes" / "ResNet18_CIFAR100_independent_init_control.json"
    if init_ctrl_path.exists():
        with open(init_ctrl_path) as f:
            init_ctrl = json.load(f)
        # Extract abh_dih_ratios if available
        if 'abh_dih_ratios' in init_ctrl:
            ratios_ctrl = [min(r['ratio'], 300) for r in init_ctrl['abh_dih_ratios']]
        elif 'process_intensity' in init_ctrl:
            ratios_ctrl = []
            for e in init_ctrl['process_intensity']:
                ab = e.get('ab_h', 0)
                di = e.get('di_h', 1)
                ratios_ctrl.append(min(ab / max(di, 1), 300))
        else:
            ratios_ctrl = []

        lane_std = data['lanes']['ResNet18-CIFAR100-seed42']
        ratios_std = [min(r['ratio'], 300) for r in lane_std['abh_dih_ratios']]

        ax.plot(range(len(ratios_std)), ratios_std, 'o-', color=ARCH_COLORS['ResNet-18'],
                markersize=3, lw=1.0, label='Shared init')
        ax.plot(range(len(ratios_ctrl)), ratios_ctrl, 's--', color='#DC2626',
                markersize=3, lw=1.0, label='Independent init')
        ax.axhline(1, color='grey', ls='--', lw=0.5)
        ax.set_yscale('log')
        ax.set_ylabel('Ab-E / Di-E ratio (log)')
        n_trans = len(iter_k_trans)
        tick_pos = list(range(0, n_trans, 2))
        ax.set_xticks(tick_pos)
        trans_labels = []
        for i in tick_pos:
            ik = iter_k_trans[i]
            iter_str = f'{ik:.1f}k' if ik < 10 else f'{ik:.0f}k'
            trans_labels.append(f'T{i}\n({iter_str})')
        ax.set_xticklabels(trans_labels, fontsize=6)
        ax.set_xlabel('Transition (iterations)')
        ax.set_title('Independent init control', fontsize=8)
        ax.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7, fontsize=7)
        ax.text(0.98, 0.05, 'Same ordering reproduced', transform=ax.transAxes,
                fontsize=6.5, ha='right', va='bottom', color='black', style='italic')
    else:
        ax.text(0.5, 0.5, 'Data not available', transform=ax.transAxes,
                ha='center', va='center', fontsize=8, color='black')
        ax.set_title('Independent init control', fontsize=8)
    add_grid(ax)

    save_fig(fig, 'ed_fig10')


# ══════════════════════════════════════════════════════════════════════════════
# ED11: Label Noise Dose-Response & Compensatory Dynamics
# ══════════════════════════════════════════════════════════════════════════════
def plot_ed11():
    print("Plotting ED11: Label noise dose-response & compensatory dynamics (multi-seed)...")

    # Load targeted label noise summary (multi-seed format)
    noise_path = DATA_DIR / "targeted_label_noise_summary_5seeds.json"
    if not noise_path.exists():
        print("  WARNING: targeted_label_noise_summary.json not found, skipping ED11")
        return
    with open(noise_path) as f:
        noise_data = json.load(f)

    conditions = noise_data['conditions']
    seeds = [str(s) for s in noise_data.get('seeds', [42, 137, 256])]

    def _get_per_seed_vals(cond_key, extract_fn):
        """Extract a value from each seed's data for a condition."""
        vals = []
        per_seed = conditions[cond_key]['per_seed']
        for s in seeds:
            if s in per_seed:
                v = extract_fn(per_seed[s])
                if v is not None:
                    vals.append(v)
        return vals

    def _mean_sem(vals):
        if len(vals) == 0:
            return 0, 0
        m = sum(vals) / len(vals)
        if len(vals) >= 2:
            import statistics
            sem = statistics.stdev(vals) / len(vals) ** 0.5
        else:
            sem = 0
        return m, sem

    # Define condition groups
    noise_types = ['within_sc', 'between_sc', 'random']
    noise_type_labels = ['Within-SC', 'Between-SC', 'Random']
    noise_type_colors = ['#6366F1', '#F59E0B', '#EF4444']
    doses = ['p01', 'p03']

    fig = plt.figure(figsize=(180/25.4, 70/25.4))
    gs = GridSpec(1, 2, figure=fig, wspace=0.40, left=0.10, right=0.95, bottom=0.15, top=0.88)

    # ── Panel (a): Compensatory proliferation (terminal alive features, multi-seed) ──
    ax_b = fig.add_subplot(gs[0, 0])

    cond_keys_ordered = ['standard', 'within_sc_p01', 'between_sc_p01', 'random_p01',
                         'within_sc_p03', 'between_sc_p03', 'random_p03']
    cond_labels = ['Std', 'W-1%', 'B-1%', 'R-1%', 'W-3%', 'B-3%', 'R-3%']
    cond_colors = ['#64748B',
                   '#6366F1', '#F59E0B', '#EF4444',
                   '#6366F1', '#F59E0B', '#EF4444']
    cond_alphas = [0.9, 0.6, 0.6, 0.6, 0.9, 0.9, 0.9]

    def _terminal_alive(seed_data):
        """Extract terminal n_alive_mean from selectivity_evolution."""
        se = seed_data.get('selectivity_evolution', {})
        if not se:
            return None
        last_key = max(se.keys(), key=lambda k: int(k))
        return se[last_key].get('n_alive_mean', None)

    alive_means = []
    alive_sems = []
    for ck in cond_keys_ordered:
        if ck in conditions:
            vals = _get_per_seed_vals(ck, _terminal_alive)
            m, s = _mean_sem(vals)
            alive_means.append(m)
            alive_sems.append(s)
        else:
            alive_means.append(0)
            alive_sems.append(0)

    bars = ax_b.bar(range(len(alive_means)), alive_means, yerr=alive_sems,
                    color=cond_colors, edgecolor='white', linewidth=0.3,
                    capsize=2, error_kw={'lw': 0.8})
    for i, bar in enumerate(bars):
        bar.set_alpha(cond_alphas[i])
        if i in [1, 2, 3]:
            bar.set_hatch('//')

    ax_b.set_xticks(range(len(cond_labels)))
    ax_b.set_xticklabels(cond_labels, fontsize=6, rotation=30, ha='right')
    ax_b.set_ylabel('Terminal alive features (mean ± s.e.m.)')
    ax_b.set_title('Compensatory proliferation', fontsize=8)
    ax_b.axhline(alive_means[0], color='grey', ls='--', lw=0.5, alpha=0.5)
    add_grid(ax_b)
    ax_b.text(-0.25, 1.08, 'a', transform=ax_b.transAxes, fontsize=8, fontweight='bold', va='top')

    # ── Panel (b): Di-E fraction trajectories (multi-seed mean ± SEM) ──
    ax_c = fig.add_subplot(gs[0, 1])

    trajectory_conds = ['standard', 'between_sc_p03', 'random_p03']
    trajectory_labels = ['Standard', 'Between-SC 3%', 'Random 3%']
    trajectory_colors = ['#64748B', '#F59E0B', '#EF4444']
    trajectory_styles = ['-', '--', ':']

    for cond_key, clabel, ccolor, cstyle in zip(
            trajectory_conds, trajectory_labels, trajectory_colors, trajectory_styles):
        if cond_key not in conditions:
            continue

        # Gather di_frac per transition across seeds
        trans_keys = None
        per_seed_fracs = {}
        for s in seeds:
            sd = conditions[cond_key]['per_seed'].get(s, {})
            pe = sd.get('process_events_per_transition', {})
            if not pe:
                continue
            sorted_trans = sorted(pe.keys(), key=lambda t: int(t.split('->')[0]))
            if trans_keys is None:
                trans_keys = sorted_trans
            per_seed_fracs[s] = [pe[t].get('di_frac', 0) for t in sorted_trans]

        if trans_keys is None:
            continue

        n_trans = len(trans_keys)
        means = []
        sems = []
        for ti in range(n_trans):
            vals = [per_seed_fracs[s][ti] for s in per_seed_fracs if ti < len(per_seed_fracs[s])]
            m, se = _mean_sem(vals)
            means.append(m * 100)
            sems.append(se * 100)

        x = list(range(n_trans))
        ax_c.errorbar(x, means, yerr=sems, fmt='o' + cstyle, color=ccolor,
                      markersize=3, lw=1.0, label=clabel, alpha=0.85, capsize=2)
        ax_c.fill_between(x, [m - s for m, s in zip(means, sems)],
                          [m + s for m, s in zip(means, sems)],
                          color=ccolor, alpha=0.12)

    ax_c.set_xlabel('Transition index')
    ax_c.set_ylabel('Di-E fraction per transition (%)')
    ax_c.set_title('Di-E developmental trajectories', fontsize=8)
    ax_c.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7,
                fontsize=6.5)
    add_grid(ax_c)
    ax_c.text(-0.25, 1.08, 'b', transform=ax_c.transAxes, fontsize=8, fontweight='bold', va='top')

    save_fig(fig, 'ed_fig11')


# ══════════════════════════════════════════════════════════════════════════════
# NEW ED8: Early Process Signature Predicts Terminal Accuracy
# ══════════════════════════════════════════════════════════════════════════════
def plot_ed_predict():
    """ED Fig 8: Predictive accuracy — Ab-H/Di-H ratio predicts terminal
    accuracy and generalization gap across all architectures."""
    print("Plotting NEW ED8: Predictive accuracy...")
    from scipy.stats import spearmanr
    from matplotlib.lines import Line2D
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    import pandas as pd

    # Load per-lane scatter data from cache
    scatter_path = DATA_DIR / "predict_scatter_cache.json"
    if not scatter_path.exists():
        print("  SKIP: predict_scatter_cache.json not found (run predict_terminal_accuracy.py first)")
        return

    with open(scatter_path) as f:
        scatter_data = json.load(f)

    lanes_data = [{
        "arch": d["architecture"],
        "cond": d["condition"],
        "val_acc": d["val_accuracy"],
        "overfit_gap": d["overfit_gap"],
        "ratio": d["t0_ab_di_ratio"],
        "ab_frac": d["t0_ab_frac"],
        "di_frac": d["t0_di_frac"],
    } for d in scatter_data]

    # Condition colors for scatter
    COND_COLORS = {
        "standard": '#2196F3', "noise_standard": '#4CAF50',
        "curriculum_standard": '#00BCD4', "curriculum_switch": '#8BC34A',
        "noise_between_sc_p01": '#FF9800', "noise_between_sc_p03": '#F44336',
        "noise_within_sc_p01": '#9C27B0', "noise_within_sc_p03": '#E91E63',
        "noise_random_p01": '#795548', "noise_random_p03": '#607D8B',
        "200_epoch": '#3F51B5', "8x_SAE": '#FF5722', "independent_init": '#FFC107',
    }
    # Condition group labels for legend
    COND_LEGEND = {
        "standard": "Standard", "noise_standard": "Noise ctrl",
        "curriculum_standard": "Curriculum ctrl", "curriculum_switch": "Curriculum switch",
        "noise_between_sc_p01": "Between-SC p=0.1", "noise_between_sc_p03": "Between-SC p=0.3",
        "noise_within_sc_p01": "Within-SC p=0.1", "noise_within_sc_p03": "Within-SC p=0.3",
        "noise_random_p01": "Random p=0.1", "noise_random_p03": "Random p=0.3",
        "200_epoch": "200 epochs", "8x_SAE": "8x SAE", "independent_init": "Indep. init",
    }

    # ResNet-18 only for panels (a) and (b)
    resnet = [d for d in lanes_data if d["arch"] == "ResNet-18"]

    fig = plt.figure(figsize=(FIG_W, FIG_H_SINGLE + 0.9))
    gs = GridSpec(1, 3, figure=fig, wspace=0.45, left=0.08, right=0.96,
                  bottom=0.22, top=0.85)

    # ── Panel (a): Ab-E/Di-E ratio vs terminal accuracy (ResNet-18) ──
    ax_a = fig.add_subplot(gs[0, 0])
    ratios_r = [d["ratio"] for d in resnet]
    log_ratios_r = [np.log1p(d["ratio"]) for d in resnet]
    accs_r = [d["val_acc"] for d in resnet]
    for d in resnet:
        c_color = COND_COLORS.get(d["cond"], "#999999")
        ax_a.scatter(np.log1p(d["ratio"]), d["val_acc"], c=c_color, s=25, alpha=0.75,
                     edgecolors='white', linewidth=0.3, zorder=3)
    rho_acc, p_acc = spearmanr(ratios_r, accs_r)
    z = np.polyfit(log_ratios_r, accs_r, 1)
    x_line = np.linspace(min(log_ratios_r), max(log_ratios_r), 100)
    ax_a.plot(x_line, np.poly1d(z)(x_line), color='black', lw=0.8, ls='--', alpha=0.5)
    ax_a.set_xlabel('log(1 + Ab-E/Di-E ratio)')
    ax_a.set_ylabel('Terminal val accuracy (%)')
    p_exp = int(np.floor(np.log10(p_acc))) if p_acc > 0 else -10
    ax_a.set_title(f'ResNet-18 (N = {len(resnet)}), ' + r'$\rho$' + f' = {rho_acc:.2f}', fontsize=7)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    add_grid(ax_a)
    ax_a.text(-0.22, 1.08, 'a', transform=ax_a.transAxes, fontsize=8, fontweight='bold', va='top')

    # ── Panel (b): Ab-E/Di-E ratio vs overfit gap (ResNet-18) ──
    ax_b = fig.add_subplot(gs[0, 1])
    gaps_r = [d["overfit_gap"] for d in resnet]
    for d in resnet:
        c_color = COND_COLORS.get(d["cond"], "#999999")
        ax_b.scatter(np.log1p(d["ratio"]), d["overfit_gap"], c=c_color, s=25, alpha=0.75,
                     edgecolors='white', linewidth=0.3, zorder=3)
    rho_gap, p_gap = spearmanr(ratios_r, gaps_r)
    z2 = np.polyfit(log_ratios_r, gaps_r, 1)
    ax_b.plot(x_line, np.poly1d(z2)(x_line), color='black', lw=0.8, ls='--', alpha=0.5)
    ax_b.set_xlabel('log(1 + Ab-E/Di-E ratio)')
    ax_b.set_ylabel('Overfit gap (train - val acc, pp)')
    ax_b.set_title(f'ResNet-18 (N = {len(resnet)}), ' + r'$\rho$' + f' = {rho_gap:.2f}', fontsize=7)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    add_grid(ax_b)
    ax_b.text(-0.22, 1.08, 'b', transform=ax_b.transAxes, fontsize=8, fontweight='bold', va='top')

    # ── Panel (c): LOO-CV R² comparison bar chart (ResNet-18 only, N=48) ──
    ax_c = fig.add_subplot(gs[0, 2])
    df_r = pd.DataFrame(resnet)
    y_r = df_r['val_acc'].values
    loo = LeaveOneOut()

    def _loo_r2(X, y_vec):
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        yp = np.zeros_like(y_vec, dtype=float)
        for tr, te in loo.split(Xs):
            m = RidgeCV(alphas=np.logspace(-3, 3, 20))
            m.fit(Xs[tr], y_vec[tr])
            yp[te] = m.predict(Xs[te])
        return r2_score(y_vec, yp)

    proc_feat = df_r[['ratio', 'ab_frac', 'di_frac']].values
    r2_proc = _loo_r2(proc_feat, y_r)

    # Overfit gap prediction
    gap_r = df_r['overfit_gap'].values
    r2_gap = _loo_r2(proc_feat, gap_r)

    model_labels = ["Val accuracy", "Overfit gap"]
    r2_vals = [r2_proc, r2_gap]
    bar_colors = [HYPO_COLORS['Ab-E'], HYPO_COLORS['Di-E']]

    bars = ax_c.bar(range(len(model_labels)), r2_vals, color=bar_colors, alpha=0.8,
                    edgecolor='white', linewidth=0.3, width=0.5)
    ax_c.set_xticks(range(len(model_labels)))
    ax_c.set_xticklabels(model_labels, fontsize=6.5)
    ax_c.set_ylabel('LOO-CV R' + r'$^2$')
    ax_c.set_title(f'Process-only prediction\n(ResNet-18, N = {len(resnet)})', fontsize=7)
    ax_c.set_ylim(0, max(r2_vals) * 1.25)
    for bar, val in zip(bars, r2_vals):
        ax_c.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                  f'{val:.3f}', ha='center', va='bottom', fontsize=6.5)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    add_grid(ax_c)
    ax_c.text(-0.22, 1.08, 'c', transform=ax_c.transAxes, fontsize=8, fontweight='bold', va='top')

    # ── Legend (conditions, below figure) ──
    legend_elements = []
    used_conds = sorted(set(d["cond"] for d in resnet))
    for cond in used_conds:
        if cond in COND_COLORS:
            legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=COND_COLORS[cond], markersize=4.5,
                                          label=COND_LEGEND.get(cond, cond)))
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=min(5, len(legend_elements)),
               fontsize=5, bbox_to_anchor=(0.35, 0.02), frameon=False,
               handletextpad=0.2, columnspacing=0.8)

    save_fig(fig, 'ed_fig8_predict')


# ══════════════════════════════════════════════════════════════════════════════
# NEW ED9: Granger Causality — Ab-H(t) → Di-H(t+1) Per Superclass
# ══════════════════════════════════════════════════════════════════════════════
def plot_ed_granger():
    """ED Fig 9: Granger causality — per-superclass coefficients, scatter,
    and cross-correlation function."""
    print("Plotting NEW ED9: Granger causality...")
    from scipy.stats import spearmanr, pearsonr

    granger_path = DATA_DIR / "granger_causality_results.json"
    series_path = DATA_DIR / "superclass_transition_series.json"
    if not granger_path.exists() or not series_path.exists():
        print("  SKIP: Granger data files not found")
        return

    with open(granger_path) as f:
        granger = json.load(f)

    import pandas as pd
    df = pd.read_json(series_path)

    fig = plt.figure(figsize=(FIG_W, FIG_H_SINGLE + 0.6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.5, left=0.10, right=0.96, bottom=0.18, top=0.85)

    # ── Panel (a): Scatter of Ab-H(t) vs Di-H(t+1) ──
    ax_a = fig.add_subplot(gs[0, 0])
    rows = []
    for (lane, sc_idx), group in df.groupby(["lane", "superclass_idx"]):
        group = group.sort_values("transition")
        ab_h = group["ab_h"].values
        di_h = group["di_h"].values
        for i in range(len(ab_h) - 1):
            rows.append({"ab_h_t": ab_h[i], "di_h_t1": di_h[i + 1]})

    scatter_data_df = pd.DataFrame(rows)
    x_s = np.log1p(scatter_data_df["ab_h_t"])
    y_s = np.log1p(scatter_data_df["di_h_t1"])
    ax_a.scatter(x_s, y_s, alpha=0.10, s=5, c=HYPO_COLORS['Ab-E'], edgecolors='none', rasterized=True)

    z = np.polyfit(x_s, y_s, 1)
    x_line = np.linspace(x_s.min(), x_s.max(), 100)
    ax_a.plot(x_line, np.poly1d(z)(x_line), color='black', lw=0.8, ls='--', alpha=0.6)
    rho, pval = spearmanr(scatter_data_df["ab_h_t"], scatter_data_df["di_h_t1"])
    ax_a.set_xlabel('log(1 + Ab-E count at t)')
    ax_a.set_ylabel('log(1 + Di-E count at t+1)')
    ax_a.set_title(f'Ab-E(t) vs Di-E(t+1), ' + r'$\rho$' + f' = {rho:.2f}', fontsize=7)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    add_grid(ax_a)
    ax_a.text(-0.25, 1.08, 'a', transform=ax_a.transAxes, fontsize=8, fontweight='bold', va='top')

    # ── Panel (b): Per-superclass Granger coefficients ──
    ax_b = fig.add_subplot(gs[0, 1])
    sc_results = granger["per_superclass"]
    sc_names = sorted(sc_results.keys(), key=lambda s: sc_results[s]["coef"])
    coefs = [sc_results[sc]["coef"] for sc in sc_names]
    sig_flags = [sc_results[sc]["sig"] != "" for sc in sc_names]
    colors = [HYPO_COLORS['Ab-E'] if s else '#94A3B8' for s in sig_flags]
    y_pos = range(len(sc_names))
    ax_b.barh(y_pos, coefs, color=colors, height=0.7, edgecolor='white', linewidth=0.3)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([sc.replace("_", " ").replace("large ", "lg. ")
                          for sc in sc_names], fontsize=5.5)
    ax_b.axvline(0, color='black', linewidth=0.4)
    ax_b.set_xlabel(r'Ab-E(t) coefficient ($\beta$)')
    ax_b.set_title('Per-superclass Granger\n(20/20 sig. after Bonferroni)', fontsize=7)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    add_grid(ax_b)
    ax_b.text(-0.35, 1.08, 'b', transform=ax_b.transAxes, fontsize=8, fontweight='bold', va='top')

    # ── Panel (c): Cross-correlation function ──
    ax_c = fig.add_subplot(gs[0, 2])
    lag_results = {}
    for lag in [-3, -2, -1, 0, 1, 2, 3]:
        corrs = []
        for (lane, sc_idx), group in df.groupby(["lane", "superclass_idx"]):
            group = group.sort_values("transition")
            ab = group["ab_h"].values
            di = group["di_h"].values
            n = len(ab)
            if abs(lag) >= n:
                continue
            if lag > 0:
                r, _ = pearsonr(ab[:n - lag], di[lag:])
            elif lag < 0:
                r, _ = pearsonr(ab[-lag:], di[:n + lag])
            else:
                r, _ = pearsonr(ab, di)
            if not np.isnan(r):
                corrs.append(r)
        lag_results[lag] = (np.mean(corrs), np.std(corrs) / np.sqrt(len(corrs)))

    lags = sorted(lag_results.keys())
    means = [lag_results[l][0] for l in lags]
    sems = [lag_results[l][1] for l in lags]
    bar_colors_ccf = [HYPO_COLORS['Ab-E'] if l >= 0 else '#94A3B8' for l in lags]
    ax_c.bar(lags, means, yerr=sems, color=bar_colors_ccf, alpha=0.75, capsize=2, width=0.6,
             edgecolor='white', linewidth=0.3, error_kw={'lw': 0.6})
    ax_c.axhline(0, color='black', linewidth=0.4)
    # Highlight peak
    peak_lag = lags[np.argmax(means)]
    peak_val = max(means)
    ax_c.annotate(f'peak at lag +{peak_lag}' if peak_lag > 0 else f'peak at lag {peak_lag}',
                  xy=(peak_lag, peak_val),
                  xytext=(peak_lag - 1.5, peak_val + 0.06), fontsize=6, style='italic',
                  arrowprops=dict(arrowstyle='->', color='black', lw=0.5))
    ax_c.set_xlabel('Lag (Ab-E leads Di-E  ' + r'$\rightarrow$' + ')')
    ax_c.set_ylabel('Mean Pearson r')
    ax_c.set_title('Cross-correlation Ab-E vs Di-E', fontsize=7)
    ax_c.set_xticks(lags)
    ax_c.set_xticklabels([str(l) for l in lags])
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    add_grid(ax_c)
    ax_c.text(-0.22, 1.08, 'c', transform=ax_c.transAxes, fontsize=8, fontweight='bold', va='top')

    # Summary text below figure
    pooled = granger["pooled"]
    fig.text(0.5, 0.03,
             f'Pooled Granger F = {pooled["granger_f"]:.1f}, P < 10' + r'$^{-6}$'
             + f', ' + r'$\Delta$' + f'R' + r'$^2$' + f' = {pooled["delta_r2"]:.3f}, '
             + r'$\beta$' + f' = {pooled["ab_h_coef"]:.2f}  |  Permutation P < 0.001',
             ha='center', fontsize=6.5, color='#475569')

    save_fig(fig, 'ed_fig9_granger')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("Generating Extended Data Figures (9 figures + 1 table = 10 items)")
    print("=" * 60)

    plot_ed1()     # ED Fig 1: Methodology validation
    plot_ed2()     # ED Fig 2: Null baseline calibration
    plot_ed5()     # ED Fig 3: Layer-wise stability heatmaps
    plot_ed6()     # ED Fig 4: Architecture-specific temporal profiles
    plot_ed7()     # ED Fig 5: Robustness (expansion + duration)
    plot_ed10()    # ED Fig 6: Independent init + crystallization
    plot_ed11()    # ED Fig 7: Label noise dose-response
    plot_ed_predict()   # ED Fig 8: Predictive accuracy
    plot_ed_granger()   # ED Fig 9: Granger causality

    print("\n" + "=" * 60)
    print("All Extended Data figures generated successfully!")
    print("=" * 60)
    print("\nFinal 10 Extended Data items:")
    print("  ED Table 1 → ed_table1          (overview of 55 runs)")
    print("  ED Fig 1   → ed_fig1            (methodology validation)")
    print("  ED Fig 2   → ed_fig2            (null baseline calibration)")
    print("  ED Fig 3   → ed_fig5            (layer-wise stability)")
    print("  ED Fig 4   → ed_fig6            (architecture temporal profiles)")
    print("  ED Fig 5   → ed_fig7            (robustness: expansion + duration)")
    print("  ED Fig 6   → ed_fig10           (independent init + crystallization)")
    print("  ED Fig 7   → ed_fig11           (label noise dose-response)")
    print("  ED Fig 8   → ed_fig8_predict    (predictive accuracy)")
    print("  ED Fig 9   → ed_fig9_granger    (Granger causality)")
    print(f"\nOutput directory: {OUT}")
