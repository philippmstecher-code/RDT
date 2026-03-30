#!/usr/bin/env python3
"""Figure 6: Curriculum Switch-Point Sweep — 4 panels, 7 conditions.

Panel (a): Validation accuracy trajectories (7 conditions) with switch markers
Panel (b): Di-E dose-response (scatter+line, x = SC epochs)
Panel (c): Process composition (7 stacked bars: Ab-E, Tg-E, Di-E fractions)
Panel (d): Regularisation benefit (overfit gap + stable features vs SC duration)

Includes all seven switch-point conditions (e05–e30).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

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

# ── Process colors (consistent with all figures) ────────────────────────────
HYPO_COLORS = {
    'Tg-E': '#8B5CF6', 'Ab-E': '#3B82F6', 'Di-E': '#F59E0B',
    'As-E': '#0891B2', 'De-E': '#EF4444',
}
ABH_COLOR = HYPO_COLORS['Ab-E']
DIH_COLOR = HYPO_COLORS['Di-E']
TGH_COLOR = HYPO_COLORS['Tg-E']

# ── Condition colors: sequential palette by SC duration ──────────────────────
COND_COLORS = {
    'standard':   '#3B82F6',  # blue
    'switch_e05': '#6366F1',  # indigo
    'switch_e10': '#8B5CF6',  # violet
    'switch_e15': '#A855F7',  # purple
    'switch_e20': '#D946EF',  # fuchsia
    'switch_e25': '#F59E0B',  # amber
    'switch_e30': '#DC2626',  # red
}
COND_LABELS = {
    'standard':   'Standard (0 SC)',
    'switch_e05': 'Switch e05',
    'switch_e10': 'Switch e10',
    'switch_e15': 'Switch e15',
    'switch_e20': 'Switch e20',
    'switch_e25': 'Switch e25',
    'switch_e30': 'Switch e30',
}
COND_MARKERS = {
    'standard': 'o', 'switch_e05': 's', 'switch_e10': 'D',
    'switch_e15': 'P', 'switch_e20': 'X',
    'switch_e25': '^', 'switch_e30': 'v',
}
CONDITIONS = ['standard', 'switch_e05', 'switch_e10', 'switch_e15',
              'switch_e20', 'switch_e25', 'switch_e30']
SC_EPOCHS = [0, 5, 10, 15, 20, 25, 30]

# ── Data (all from RESULTS_curriculum_switch.md) ─────────────────────────────

# Panel (a): Accuracy trajectories
# Fine-class phase validation accuracy (100-class task)
FINE_TRAJECTORIES = {
    'standard':   {'epochs': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                   'acc':    [13.0, 29.8, 35.5, 43.9, 44.5, 49.3, 51.5, 54.2, 57.1, 58.8, 59.1]},
    'switch_e05': {'epochs': [10, 15, 20, 25, 30, 35, 40, 45, 50],
                   'acc':    [27.4, 35.5, 41.6, 42.7, 47.4, 51.1, 54.3, 56.7, 57.9]},
    'switch_e10': {'epochs': [15, 20, 25, 30, 35, 40, 45, 50],
                   'acc':    [30.7, 39.2, 42.6, 48.0, 48.9, 54.7, 57.2, 58.2]},
    'switch_e15': {'epochs': [20, 25, 30, 35, 40, 45, 50],
                   'acc':    [33.5, 40.4, 43.4, 48.0, 51.8, 55.4, 57.1]},
    'switch_e20': {'epochs': [25, 30, 35, 40, 45, 50],
                   'acc':    [38.4, 42.9, 46.4, 51.9, 55.7, 57.9]},
    'switch_e25': {'epochs': [30, 35, 40, 45, 50],
                   'acc':    [39.5, 44.8, 49.4, 55.1, 58.2]},
    'switch_e30': {'epochs': [35, 40, 45, 50],
                   'acc':    [40.1, 47.6, 52.5, 58.2]},
}
# Superclass phase (20-class task) — shown as dashed
SC_TRAJECTORIES = {
    'switch_e05': {'epochs': [1, 5],                     'acc': [14.5, 31.2]},
    'switch_e10': {'epochs': [1, 5, 10],                 'acc': [19.3, 36.2, 46.4]},
    'switch_e15': {'epochs': [15],                       'acc': [43.9]},
    'switch_e20': {'epochs': [1, 5, 10, 15, 20],        'acc': [17.8, 34.5, 44.1, 53.0, 58.4]},
    'switch_e25': {'epochs': [1, 5, 10, 15, 20, 25],    'acc': [16.2, 34.7, 42.3, 50.7, 57.8, 62.2]},
    'switch_e30': {'epochs': [1, 5, 10, 15, 20, 25, 30],'acc': [15.9, 34.1, 41.5, 50.8, 57.2, 61.7, 64.5]},
}

# Panel (b): Di-E events (total across all transitions, layers, classes)
# Order: standard, e05, e10, e15, e20, e25, e30
DIH_EVENTS = [358816, 99974, 96357, 53872, 91430, 55315, 77578]

# Panel (c): Process fractions (Ab-H + Tg-H + Di-H only)
ABH_EVENTS = [72727, 28744, 34326, 9214, 41686, 30872, 25424]
TGH_EVENTS = [218519, 175589, 288343, 285110, 247141, 244427, 207218]

# Panel (d): Train vs Val accuracy
TRAIN_ACC = [96.19, 90.89, 89.83, 84.91, 83.06, 79.83, 76.69]
VAL_ACC   = [59.13, 57.86, 58.21, 57.07, 57.85, 58.24, 58.21]
OVERFIT_GAP = [t - v for t, v in zip(TRAIN_ACC, VAL_ACC)]


# ── Build figure ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(183/25.4, 160/25.4))  # 183mm × 160mm (slightly taller for 7 conditions)
gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.35,
              left=0.08, right=0.95, top=0.95, bottom=0.07)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Panel (a): Accuracy trajectories
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax_a = fig.add_subplot(gs[0, 0])

for cond in CONDITIONS:
    color = COND_COLORS[cond]
    marker = COND_MARKERS[cond]
    # Fine-class phase (solid)
    ft = FINE_TRAJECTORIES[cond]
    ax_a.plot(ft['epochs'], ft['acc'], color=color, linewidth=1.0,
              marker=marker, markersize=2.5, label=COND_LABELS[cond], zorder=3)
    # SC phase (dashed, lighter)
    if cond in SC_TRAJECTORIES:
        st = SC_TRAJECTORIES[cond]
        ax_a.plot(st['epochs'], st['acc'], color=color, linewidth=0.8,
                  linestyle='--', alpha=0.5, zorder=2)
        # Connect SC end to fine start with dotted line (the drop)
        switch_epoch = SC_EPOCHS[CONDITIONS.index(cond)]
        ax_a.plot([st['epochs'][-1], ft['epochs'][0]],
                  [st['acc'][-1], ft['acc'][0]],
                  color=color, linewidth=0.6, linestyle=':', alpha=0.4, zorder=2)

# Convergence band annotation
ax_a.axhspan(57.0, 59.5, alpha=0.08, color='grey', zorder=0)
ax_a.annotate('57–59%', xy=(3, 58.3), fontsize=6, color='black', ha='left', va='center')


ax_a.set_xlabel('Epoch')
ax_a.set_ylabel('Validation accuracy (%)')
ax_a.set_xlim(0, 52)
ax_a.set_ylim(10, 68)
ax_a.legend(loc='lower right', frameon=True, facecolor='white',
            edgecolor='none', framealpha=0.8, fontsize=5.5)
ax_a.text(-0.15, 1.05, 'a', transform=ax_a.transAxes, fontsize=8,
          fontweight='bold', va='top', ha='left')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Panel (b): Di-E dose-response
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax_b = fig.add_subplot(gs[0, 1])

for i, (sc, dih, cond) in enumerate(zip(SC_EPOCHS, DIH_EVENTS, CONDITIONS)):
    ax_b.scatter(sc, dih / 1000, color=COND_COLORS[cond], marker=COND_MARKERS[cond],
                 s=40, zorder=4, edgecolors='white', linewidths=0.5)

# Connect with line
ax_b.plot(SC_EPOCHS, [d / 1000 for d in DIH_EVENTS], color='grey', linewidth=0.8,
          linestyle='-', alpha=0.5, zorder=2)

# Annotate 6.7× between standard and e25
ax_b.annotate('', xy=(25, DIH_EVENTS[5] / 1000), xytext=(0, DIH_EVENTS[0] / 1000),
              arrowprops=dict(arrowstyle='<->', color=DIH_COLOR, lw=1.0,
                              connectionstyle='arc3,rad=-0.2'))
ax_b.annotate('6.7×', xy=(10, 220), fontsize=8, fontweight='bold',
              color='black', ha='center', va='center',
              bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                        edgecolor='none', alpha=0.9))

# Horizontal reference line at standard level
ax_b.axhline(y=DIH_EVENTS[0] / 1000, color=COND_COLORS['standard'],
             linestyle=':', linewidth=0.5, alpha=0.4)

ax_b.set_xlabel('Superclass pre-training epochs')
ax_b.set_ylabel('Total Di-E events (×10³)')
ax_b.set_xlim(-3, 33)
ax_b.set_xticks(SC_EPOCHS)
ax_b.text(-0.15, 1.05, 'b', transform=ax_b.transAxes, fontsize=8,
          fontweight='bold', va='top', ha='left')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Panel (c): Process composition (stacked bars)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax_c = fig.add_subplot(gs[1, 0])

x_bars = np.arange(len(CONDITIONS))
width = 0.6

# Compute fractions (Ab + Tg + Di = 100%)
total_3 = [a + t + d for a, t, d in zip(ABH_EVENTS, TGH_EVENTS, DIH_EVENTS)]
tg_fracs = [t / s * 100 for t, s in zip(TGH_EVENTS, total_3)]
ab_fracs = [a / s * 100 for a, s in zip(ABH_EVENTS, total_3)]
di_fracs = [d / s * 100 for d, s in zip(DIH_EVENTS, total_3)]

# Stack: Tg-E (bottom), Ab-E (middle), Di-E (top)
bars_tg = ax_c.bar(x_bars, tg_fracs, width, color=TGH_COLOR,
                    edgecolor='white', linewidth=0.5, label='Tg-E')
bars_ab = ax_c.bar(x_bars, ab_fracs, width, bottom=tg_fracs,
                    color=ABH_COLOR, edgecolor='white', linewidth=0.5, label='Ab-E')
bars_di = ax_c.bar(x_bars, di_fracs, width,
                    bottom=[t + a for t, a in zip(tg_fracs, ab_fracs)],
                    color=DIH_COLOR, edgecolor='white', linewidth=0.5, label='Di-E')

# Label Di-E percentage on each bar
for i, di_f in enumerate(di_fracs):
    y_pos = tg_fracs[i] + ab_fracs[i] + di_f / 2
    if di_f > 8:
        ax_c.text(x_bars[i], y_pos, f'{di_f:.1f}%', ha='center', va='center',
                  fontsize=5, color='white', fontweight='bold')
    else:
        ax_c.text(x_bars[i], tg_fracs[i] + ab_fracs[i] + di_f + 1,
                  f'{di_f:.1f}%', ha='center', va='bottom',
                  fontsize=5, color='black', fontweight='bold')

cond_short = ['Std', 'e05', 'e10', 'e15', 'e20', 'e25', 'e30']
ax_c.set_xticks(x_bars)
ax_c.set_xticklabels(cond_short)
ax_c.set_ylabel('Process fraction (%)\n(Ab-E + Tg-E + Di-E)')
ax_c.set_ylim(0, 108)
ax_c.legend(loc='upper right', frameon=True, facecolor='white',
            edgecolor='none', framealpha=0.8, ncol=3, fontsize=6)
ax_c.text(-0.15, 1.05, 'c', transform=ax_c.transAxes, fontsize=8,
          fontweight='bold', va='top', ha='left')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Panel (d): Train vs Val accuracy with generalization gap
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax_d = fig.add_subplot(gs[1, 1])

x_d = np.arange(len(CONDITIONS))
w_d = 0.35

# Train bars (lighter) and val bars (solid)
bars_train = ax_d.bar(x_d - w_d/2, TRAIN_ACC, w_d, color='#9CA3AF', alpha=0.5,
                       label='Train acc', edgecolor='white', linewidth=0.5)
bars_val = ax_d.bar(x_d + w_d/2, VAL_ACC, w_d,
                     color=[COND_COLORS[c] for c in CONDITIONS], alpha=0.85,
                     label='Val acc', edgecolor='white', linewidth=0.5)

# Draw red bracket showing the gap for each condition
for i in range(len(CONDITIONS)):
    gap = OVERFIT_GAP[i]
    mid_x = x_d[i]
    ax_d.plot([mid_x, mid_x], [VAL_ACC[i], TRAIN_ACC[i]],
              color='#EF4444', linewidth=1.2, zorder=5)
    # Small horizontal caps
    cap_w = 0.08
    ax_d.plot([mid_x - cap_w, mid_x + cap_w], [VAL_ACC[i], VAL_ACC[i]],
              color='#EF4444', linewidth=1.0, zorder=5)
    ax_d.plot([mid_x - cap_w, mid_x + cap_w], [TRAIN_ACC[i], TRAIN_ACC[i]],
              color='#EF4444', linewidth=1.0, zorder=5)
    # Gap label
    ax_d.text(mid_x + 0.3, (VAL_ACC[i] + TRAIN_ACC[i]) / 2,
              f'{gap:.0f}', fontsize=5.5, color='#EF4444', fontweight='bold',
              ha='left', va='center')

cond_short = ['Std', 'e05', 'e10', 'e15', 'e20', 'e25', 'e30']
ax_d.set_xticks(x_d)
ax_d.set_xticklabels(cond_short)
ax_d.set_ylabel('Accuracy (%)')
ax_d.set_ylim(0, 105)

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Patch(facecolor='#9CA3AF', alpha=0.5, label='Train'),
    Patch(facecolor='#6366F1', alpha=0.85, label='Val'),
    Line2D([0], [0], color='#EF4444', linewidth=1.2, label='Gap'),
]
ax_d.legend(handles=legend_elements, loc='upper right', frameon=True,
            facecolor='white', edgecolor='none', framealpha=0.8, fontsize=6)

ax_d.text(-0.15, 1.05, 'd', transform=ax_d.transAxes, fontsize=8,
          fontweight='bold', va='top', ha='left')

# ── Remove top/right spines on all panels ────────────────────────────────────
for ax in [ax_a, ax_b, ax_c, ax_d]:
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

# ── Save ─────────────────────────────────────────────────────────────────────
for fmt in ['png', 'pdf']:
    fig.savefig(OUT / f'fig8.{fmt}', dpi=600, bbox_inches='tight')
    print(f"Saved: {OUT / f'fig8.{fmt}'}")

plt.close()
print("Done.")
