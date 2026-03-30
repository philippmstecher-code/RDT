#!/usr/bin/env python3
"""Figure 2: Feature turnover concentrates in an early reorganisation window — 2 panels."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
# epoch_labels no longer used — plain transition indices on x-axis

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "consolidated_findings.json"
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
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})

# ── Consistent color palette ──
HYPO_COLORS = {
    'Tg-E': '#8B5CF6', 'Ab-E': '#3B82F6', 'Di-E': '#F59E0B',
    'As-E': '#0891B2', 'De-E': '#EF4444',
}
ARCH_COLORS = {'ResNet-18': '#2C5F8A', 'ViT-Small': '#2A9D8F', 'CCT-7': '#E76F51'}
N_TR = 9  # number of transitions to show

# Identify CIFAR-100 standard lanes
cifar100_lanes = [l for l in data['lanes']
                  if data['lanes'][l]['metadata']['dataset'] == 'CIFAR-100'
                  and data['lanes'][l]['metadata']['epochs'] == 50
                  and data['lanes'][l]['metadata']['expansion'] == '4x']
primary = 'ResNet18-CIFAR100-seed42'

fig = plt.figure(figsize=(180/25.4, 70/25.4))
gs = GridSpec(1, 2, figure=fig, wspace=0.4)

# ── Panel (a): Feature churn — architecture means + faint individual lanes ──
ax_a = fig.add_subplot(gs[0, 0])
ax_a.text(-0.15, 1.05, 'a', transform=ax_a.transAxes, fontsize=8,
          fontweight='bold', va='top', ha='left')

# Plot all individual lanes faintly (9 standard CIFAR-100 runs only)
arch_churn = {}
for label in cifar100_lanes:
    lane = data['lanes'][label]
    pi = lane['process_intensity']
    wu_gaps = data['lanes'][label]['epoch_info'].get('transition_weight_update_gaps', [])
    if wu_gaps and len(wu_gaps) >= len(pi):
        churn = [p['churn'] * 100 / max(wu / 1000, 0.001) for p, wu in zip(pi, wu_gaps)]
    else:
        churn = [p['churn'] * 100 for p in pi]
    arch = lane['metadata']['architecture']
    color = ARCH_COLORS.get(arch, '#888888')
    ax_a.plot(range(min(len(churn), N_TR)), churn[:N_TR], color=color, linewidth=0.4, alpha=0.15)
    if arch not in arch_churn:
        arch_churn[arch] = []
    arch_churn[arch].append(churn)

# Architecture-level means
for arch, color in ARCH_COLORS.items():
    if arch in arch_churn:
        max_l = max(len(v) for v in arch_churn[arch])
        padded = np.array([v[:N_TR] + [v[min(N_TR-1, len(v)-1)]] * max(0, N_TR - len(v)) for v in arch_churn[arch]])
        mean = padded.mean(axis=0)
        ax_a.plot(range(N_TR), mean, color=color, linewidth=1.0, label=arch)

# Critical learning period background
ax_a.axvspan(-0.5, 2.5, color='#FDE68A', alpha=0.2, zorder=0)
ax_a.text(1.0, 0.98, 'Critical period', fontsize=6, fontstyle='italic', color='#92400E',
          transform=ax_a.get_xaxis_transform(), ha='center', va='top')

ax_a.set_xlabel('Training transition')
ax_a.set_ylabel('Churn rate (% per 1k updates)')
ax_a.set_xticks(range(N_TR))
ax_a.set_xticklabels([str(i) for i in range(N_TR)])
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
ax_a.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7, fontsize=7, loc='upper left',
            bbox_to_anchor=(0.0, 1.22), ncol=3, columnspacing=0.6)

# ── Panel (b): Born/Died stacked bars — averaged across all 9 CIFAR-100 lanes ──
ax_b = fig.add_subplot(gs[0, 1])
ax_b.text(-0.15, 1.05, 'b', transform=ax_b.transAxes, fontsize=8,
          fontweight='bold', va='top', ha='left')

# Collect born/died normalised per 1k updates
all_born = []
all_died = []

for label in cifar100_lanes:
    fm = data['lanes'][label]['feature_matching']
    transitions_fm = sorted(fm.keys(), key=lambda x: int(x.split('->')[0]))
    wu_gaps = data['lanes'][label]['epoch_info'].get('transition_weight_update_gaps', [])
    born_raw = [fm[t]['n_born'] for t in transitions_fm]
    died_raw = [fm[t]['n_died'] for t in transitions_fm]
    if wu_gaps and len(wu_gaps) >= len(born_raw):
        born = [b / max(wu / 1000, 0.001) for b, wu in zip(born_raw, wu_gaps)]
        died = [d / max(wu / 1000, 0.001) for d, wu in zip(died_raw, wu_gaps)]
    else:
        born = born_raw
        died = died_raw
    all_born.append(born)
    all_died.append(died)

# Pad to N_TR transitions and average
all_born = np.array([v[:N_TR] + [v[min(N_TR-1, len(v)-1)]] * max(0, N_TR - len(v)) for v in all_born])
all_died = np.array([v[:N_TR] + [v[min(N_TR-1, len(v)-1)]] * max(0, N_TR - len(v)) for v in all_died])

mean_born = all_born.mean(axis=0)
mean_died = all_died.mean(axis=0)

x_fm = np.arange(N_TR)
w = 0.8
bottom = np.zeros(N_TR)

for vals, color, label in [
    (mean_born, HYPO_COLORS['As-E'], 'Born'),
    (mean_died, HYPO_COLORS['De-E'], 'Died'),
]:
    ax_b.bar(x_fm, vals, w, bottom=bottom, color=color, label=label, alpha=0.8)
    bottom += vals

# Critical learning period background
ax_b.axvspan(-0.5, 2.5, color='#FDE68A', alpha=0.2, zorder=0)

ax_b.set_xticks(x_fm)
ax_b.set_xticklabels([str(i) for i in range(N_TR)])
ax_b.set_xlabel('Training transition')
ax_b.set_ylabel('Feature events per 1k updates')
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7, fontsize=7, loc='upper left',
            bbox_to_anchor=(0.0, 1.22), ncol=2, columnspacing=0.6)

fig.subplots_adjust(left=0.10, right=0.95, top=0.88, bottom=0.15, wspace=0.40)
plt.savefig(OUT / 'fig2.pdf', bbox_inches='tight', dpi=600)
plt.savefig(OUT / 'fig2.png', bbox_inches='tight', dpi=600)
plt.close()
print("Saved fig2.pdf and fig2.png")
