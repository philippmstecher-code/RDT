#!/usr/bin/env python3
"""Figure 4: Consistency across architectures, seeds, and datasets (F4) — 4 panels.

Updated to include TinyImageNet cross-dataset validation.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pathlib import Path
from epoch_labels import format_epoch_ticks

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
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})

HYPO_COLORS = {
    'Tg-E': '#8B5CF6', 'Ab-E': '#3B82F6', 'Di-E': '#F59E0B',
    'As-E': '#0891B2', 'De-E': '#EF4444',
}
ARCH_COLORS = {'ResNet-18': '#2C5F8A', 'ViT-Small': '#2A9D8F', 'CCT-7': '#E76F51'}
TINYIMAGENET_COLOR = '#F472B6'  # rose/salmon — unified across all panels

cifar100_lanes = [l for l in data['lanes']
                  if data['lanes'][l]['metadata']['dataset'] == 'CIFAR-100'
                  and data['lanes'][l]['metadata']['epochs'] == 50
                  and data['lanes'][l]['metadata']['expansion'] == '4x']

tinyimagenet_lanes = [l for l in data['lanes']
                      if data['lanes'][l]['metadata']['dataset'] == 'TinyImageNet']

cross = data['cross_lane_statistics']

fig = plt.figure(figsize=(180/25.4, 140/25.4))
gs = GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.50,
              width_ratios=[1, 1.3])
# Panel (c) will span full width — see gs[1, :] below

# ── Panel (a): Initial vs final construction/refinement ratio (log scale) ──
ax_a = fig.add_subplot(gs[0, 0])
ax_a.text(-0.15, 1.05, 'a', transform=ax_a.transAxes, fontsize=8,
          fontweight='bold', va='top', ha='left')

def _cr_ratio(r):
    """(Tg-E + Ab-E) / Di-E from a ratio record."""
    return (r.get('tg_h', 0) + r['ab_h']) / max(r['di_h'], 1)

# Collect initial and final ratios per architecture (CIFAR-100 standard only)
arch_init = {}
arch_final = {}
for label in cifar100_lanes:
    lane = data['lanes'][label]
    arch = lane['metadata']['architecture']
    ratios = lane['abh_dih_ratios']
    if not ratios:
        continue
    arch_init.setdefault(arch, []).append(_cr_ratio(ratios[0]))
    arch_final.setdefault(arch, []).append(_cr_ratio(ratios[-1]))

# TinyImageNet
tin_init, tin_final = [], []
for label in tinyimagenet_lanes:
    ratios = data['lanes'][label]['abh_dih_ratios']
    if ratios:
        tin_init.append(_cr_ratio(ratios[0]))
        tin_final.append(_cr_ratio(ratios[-1]))

group_names = list(ARCH_COLORS.keys()) + ['ResNet-18\n(TinyImageNet)']
group_colors = list(ARCH_COLORS.values()) + [TINYIMAGENET_COLOR]
init_vals = [arch_init.get(a, [1.0]) for a in ARCH_COLORS.keys()] + [tin_init or [1.0]]
final_vals = [arch_final.get(a, [1.0]) for a in ARCH_COLORS.keys()] + [tin_final or [1.0]]

x_a = np.arange(len(group_names))
bar_w = 0.3

# Grouped bars: initial (light/hatched) and final (solid)
for i in range(len(group_names)):
    init_mean = np.mean(init_vals[i])
    final_mean = np.mean(final_vals[i])
    ax_a.bar(i - bar_w / 2, init_mean, bar_w, color=group_colors[i], alpha=0.35,
             edgecolor=group_colors[i], linewidth=0.6)
    ax_a.bar(i + bar_w / 2, final_mean, bar_w, color=group_colors[i], alpha=0.85)


ax_a.set_yscale('log')
ax_a.axhline(1, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
ax_a.set_xticks(x_a)
ax_a.set_xticklabels(group_names, fontsize=7)
ax_a.set_ylabel('(Tg-E + Ab-E) / Di-E ratio (log)')
ax_a.set_title('Construction / refinement ratio', fontsize=8)

# Legend
from matplotlib.patches import Patch
legend_patches = [Patch(facecolor='#9CA3AF', alpha=0.35, edgecolor='#9CA3AF', label='Initial'),
                  Patch(facecolor='#9CA3AF', alpha=0.85, label='Final')]
ax_a.legend(handles=legend_patches, frameon=True, facecolor='white', edgecolor='none',
            framealpha=0.7, fontsize=7)

# ── Panel (b): Superclass invariance — dot/strip plot ──
ax_b = fig.add_subplot(gs[0, 1])
ax_b.text(-0.15, 1.05, 'b', transform=ax_b.transAxes, fontsize=8,
          fontweight='bold', va='top', ha='left')

proc_list = [('as_h', 'As-E', HYPO_COLORS['As-E']),
             ('de_h', 'De-E', HYPO_COLORS['De-E']),
             ('ab_h', 'Ab-E', HYPO_COLORS['Ab-E']),
             ('di_h', 'Di-E', HYPO_COLORS['Di-E']),
             ('tg_h', 'Tg-E', HYPO_COLORS['Tg-E'])]

inv_cifar = cross.get('f4_superclass_invariance', {})
inv_tin = cross.get('f4_superclass_invariance_tinyimagenet', {})

# Compute Tg-E per-superclass invariance on the fly (not in pre-computed cross stats)
def _compute_tg_invariance(lanes_dict, lane_keys):
    sc_vals = {}
    for label in lane_keys:
        ss = lanes_dict[label].get('superclass_summary', {})
        for sc, info in ss.items():
            frac = info.get('process_fractions', {}).get('tg_h', 0)
            sc_vals.setdefault(sc, []).append(frac)
    return {sc: np.mean(vals) for sc, vals in sc_vals.items()}

tg_cifar_sc = _compute_tg_invariance(data['lanes'], cifar100_lanes)
tg_tin_sc = _compute_tg_invariance(data['lanes'], tinyimagenet_lanes)

if inv_cifar or inv_tin:
    proc_names = [lbl for _, lbl, _ in proc_list]
    x_b = np.arange(len(proc_names))
    rng = np.random.default_rng(42)
    jitter_w = 0.12  # horizontal jitter half-width

    cifar_color = '#374151'   # dark gray
    tin_color = TINYIMAGENET_COLOR

    for i, (proc_key, proc_label, _) in enumerate(proc_list):
        # CIFAR-100 per-superclass dots
        if proc_key == 'tg_h':
            cifar_vals = np.array([v * 100 for v in tg_cifar_sc.values()])
        else:
            cifar_sc = inv_cifar.get(proc_key, {}).get('per_superclass_mean', {})
            cifar_vals = np.array([v * 100 for v in cifar_sc.values()])
        if len(cifar_vals):
            jx = rng.uniform(-jitter_w, jitter_w, size=len(cifar_vals))
            ax_b.scatter(i - 0.15 + jx, cifar_vals, s=12, color=cifar_color,
                         alpha=0.45, edgecolors='none', zorder=4)
            ax_b.plot([i - 0.15 - 0.10, i - 0.15 + 0.10],
                      [cifar_vals.mean(), cifar_vals.mean()],
                      color=cifar_color, linewidth=1.0, zorder=5)

        # TinyImageNet per-superclass dots
        if proc_key == 'tg_h':
            tin_vals = np.array([v * 100 for v in tg_tin_sc.values()])
        else:
            tin_sc = inv_tin.get(proc_key, {}).get('per_superclass_mean', {})
            tin_vals = np.array([v * 100 for v in tin_sc.values()])
        if len(tin_vals):
            jx = rng.uniform(-jitter_w, jitter_w, size=len(tin_vals))
            ax_b.scatter(i + 0.15 + jx, tin_vals, s=12, color=tin_color,
                         alpha=0.45, edgecolors='none', zorder=4)
            ax_b.plot([i + 0.15 - 0.10, i + 0.15 + 0.10],
                      [tin_vals.mean(), tin_vals.mean()],
                      color=tin_color, linewidth=1.0, zorder=5)

    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(proc_names)
    ax_b.set_ylabel('Process fraction (%)')
    ax_b.set_title('Superclass invariance', fontsize=8)
    for spine in ['top', 'right']:
        ax_b.spines[spine].set_visible(False)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cifar_color,
               markersize=5, alpha=0.7, label='CIFAR-100 (20 sc)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=tin_color,
               markersize=5, alpha=0.7, label='TinyImageNet (27 sc)'),
    ]
    ax_b.legend(handles=legend_elements, frameon=True, facecolor='white',
                edgecolor='none', framealpha=0.7, fontsize=7, loc='upper right')

# ── Panel (c): SSI trajectories — CIFAR-100 multi-seed + TinyImageNet overlay ──
ax_c = fig.add_subplot(gs[1, 0])
ax_c.text(-0.15, 1.05, 'c', transform=ax_c.transAxes, fontsize=8,
          fontweight='bold', va='top', ha='left')

# CIFAR-100 seed overlay
seed_lanes = ['ResNet18-CIFAR100-seed42', 'ResNet18-CIFAR100-seed137', 'ResNet18-CIFAR100-seed256']
styles = ['-', '--', ':']
N_CP = 10  # show checkpoints 0–9
all_ssi = []
for label, ls in zip(seed_lanes, styles):
    if label in data['lanes']:
        ssi = [se['mean_ssi'] for se in data['lanes'][label]['selectivity_evolution']][:N_CP]
        ax_c.plot(range(len(ssi)), ssi, linestyle=ls, linewidth=1, color=ARCH_COLORS['ResNet-18'],
                  label=f"CIFAR-100 s{data['lanes'][label]['metadata']['seed']}")
        all_ssi.append(ssi)

# Agreement zone for CIFAR-100 seeds
if len(all_ssi) > 1:
    max_l = max(len(s) for s in all_ssi)
    padded = np.array([s + [s[-1]] * (max_l - len(s)) for s in all_ssi])
    lo = padded.min(axis=0)
    hi = padded.max(axis=0)
    ax_c.fill_between(range(max_l), lo, hi, alpha=0.15, color=ARCH_COLORS['ResNet-18'])

# TinyImageNet SSI trajectory
for label in tinyimagenet_lanes:
    ssi = [se['mean_ssi'] for se in data['lanes'][label]['selectivity_evolution']][:N_CP]
    ax_c.plot(range(len(ssi)), ssi, linestyle='-', linewidth=1.0, color=TINYIMAGENET_COLOR,
              label='TinyImageNet', marker='o', markersize=2)

ax_c.set_xlabel('Checkpoint (epoch)')
cp_epochs = data['lanes']['ResNet18-CIFAR100-seed42']['epoch_info']['checkpoint_epochs'][:N_CP]
format_epoch_ticks(ax_c, cp_epochs, every_n=2)
ax_c.set_ylabel('Mean SSI')
ax_c.set_title('Cross-dataset SSI trajectories', fontsize=8)
ax_c.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7, fontsize=7, loc='lower left')

# ── Panel (d): CSI trajectories — CIFAR-100 multi-seed + TinyImageNet overlay ──
ax_d = fig.add_subplot(gs[1, 1])
ax_d.text(-0.15, 1.05, 'd', transform=ax_d.transAxes, fontsize=8,
          fontweight='bold', va='top', ha='left')

all_csi = []
for label, ls in zip(seed_lanes, styles):
    if label in data['lanes']:
        csi = [se['mean_csi'] for se in data['lanes'][label]['selectivity_evolution']][:N_CP]
        ax_d.plot(range(len(csi)), csi, linestyle=ls, linewidth=1, color=ARCH_COLORS['ResNet-18'],
                  label=f"CIFAR-100 s{data['lanes'][label]['metadata']['seed']}")
        all_csi.append(csi)

# Agreement zone for CIFAR-100 seeds
if len(all_csi) > 1:
    max_l = max(len(s) for s in all_csi)
    padded = np.array([s + [s[-1]] * (max_l - len(s)) for s in all_csi])
    lo = padded.min(axis=0)
    hi = padded.max(axis=0)
    ax_d.fill_between(range(max_l), lo, hi, alpha=0.15, color=ARCH_COLORS['ResNet-18'])

# TinyImageNet CSI trajectory
for label in tinyimagenet_lanes:
    csi = [se['mean_csi'] for se in data['lanes'][label]['selectivity_evolution']][:N_CP]
    ax_d.plot(range(len(csi)), csi, linestyle='-', linewidth=1.0, color=TINYIMAGENET_COLOR,
              label='TinyImageNet', marker='o', markersize=2)

ax_d.set_xlabel('Checkpoint (epoch)')
cp_epochs_d = data['lanes']['ResNet18-CIFAR100-seed42']['epoch_info']['checkpoint_epochs'][:N_CP]
format_epoch_ticks(ax_d, cp_epochs_d, every_n=2)
ax_d.set_ylabel('Mean CSI')
ax_d.set_title('Cross-dataset CSI trajectories', fontsize=8)
ax_d.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7, fontsize=7, loc='lower left')

# ── Remove top/right spines on all panels ──
for ax in [ax_a, ax_c, ax_d]:
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.08, hspace=0.55, wspace=0.45)
plt.savefig(OUT / 'fig4.pdf', bbox_inches='tight', dpi=600)
plt.savefig(OUT / 'fig4.png', bbox_inches='tight', dpi=600)
plt.close()
print("Saved fig4.pdf and fig4.png")
