#!/usr/bin/env python3
"""Figure: Targeted Label Noise Experiment — 4 panels (multi-seed).

Panel (a): Experimental design schematic (concrete CIFAR-100 classes)
Panel (b): Grouped bars — all 7 conditions, Di-E and Ab-E side by side
Panel (c): Dissociation index — between-SC vs random selective effect on Di-E vs Ab-E
Panel (d): Developmental trajectories — Construction and Refinement overlaying 3 conditions
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "targeted_label_noise_summary_5seeds.json"

# ── Universal style (matches other figures) ──
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

# Colors — consistent with all figures
ABH_COLOR = '#3B82F6'   # blue
DIH_COLOR = '#F59E0B'   # amber
TGH_COLOR = '#8B5CF6'   # purple

# ── Load multi-seed data ──
with open(DATA_PATH) as f:
    data = json.load(f)

SEEDS = [str(s) for s in data["seeds"]]
CONDITION_KEYS = ['standard', 'within_sc_p01', 'within_sc_p03',
                  'between_sc_p01', 'between_sc_p03', 'random_p01', 'random_p03']

condition_labels = ['Std', 'W .1', 'W .3', 'B .1', 'B .3', 'R .1', 'R .3']
condition_full = ['Standard', 'Within p=0.1', 'Within p=0.3',
                  'Between p=0.1', 'Between p=0.3', 'Random p=0.1', 'Random p=0.3']

# Extract summary stats for panel (b)
ab_frac_mean = []
ab_frac_sem = []
di_frac_mean = []
di_frac_sem = []
tg_frac_mean = []
tg_frac_sem = []

for cond in CONDITION_KEYS:
    s = data["conditions"][cond]["summary"]
    ab_frac_mean.append(s["ab_frac_mean"] * 100)
    ab_frac_sem.append(s["ab_frac_sem"] * 100)
    di_frac_mean.append(s["di_frac_mean"] * 100)
    di_frac_sem.append(s["di_frac_sem"] * 100)
    tg_frac_mean.append(s["tg_frac_mean"] * 100)
    tg_frac_sem.append(s["tg_frac_sem"] * 100)

# Panel (c): Dose-response — Δ fraction from standard (in pp) for between_sc
std_summary = data["conditions"]["standard"]["summary"]
std_di_frac = std_summary["di_frac_mean"] * 100  # standard Di-H fraction %
std_ab_frac = std_summary["ab_frac_mean"] * 100

# Per-seed Di-H and Ab-H fraction for standard and between_sc conditions
std_di_per_seed = [data["conditions"]["standard"]["per_seed"][s]["process_fractions"]["di_frac"] * 100 for s in SEEDS]
std_ab_per_seed = [data["conditions"]["standard"]["per_seed"][s]["process_fractions"]["ab_frac"] * 100 for s in SEEDS]

dose_labels = ['p=0.1', 'p=0.3']
dose_conds = ['between_sc_p01', 'between_sc_p03']

# Compute Δ per seed, then mean ± SEM
di_delta_mean = []
di_delta_sem = []
di_delta_seeds = []
ab_delta_mean = []
ab_delta_sem = []
ab_delta_seeds = []

for cond in dose_conds:
    di_per_seed = [data["conditions"][cond]["per_seed"][s]["process_fractions"]["di_frac"] * 100 for s in SEEDS]
    ab_per_seed = [data["conditions"][cond]["per_seed"][s]["process_fractions"]["ab_frac"] * 100 for s in SEEDS]
    di_deltas = [di_per_seed[i] - std_di_per_seed[i] for i in range(len(SEEDS))]
    ab_deltas = [ab_per_seed[i] - std_ab_per_seed[i] for i in range(len(SEEDS))]
    di_delta_mean.append(np.mean(di_deltas))
    di_delta_sem.append(np.std(di_deltas, ddof=1) / np.sqrt(len(SEEDS)))
    di_delta_seeds.append(di_deltas)
    ab_delta_mean.append(np.mean(ab_deltas))
    ab_delta_sem.append(np.std(ab_deltas, ddof=1) / np.sqrt(len(SEEDS)))
    ab_delta_seeds.append(ab_deltas)

# Panel (d): Trajectories — compute mean ± SEM of Di-H fraction per transition
transitions = ['0->1', '1->2', '2->3', '3->4', '4->5', '5->6', '6->7']
transition_labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7']

traj_conditions = [
    ('Standard', 'standard'),
    ('Between-superclass p=0.3', 'between_sc_p03'),
    ('Random p=0.3', 'random_p03'),
]


def get_traj_fracs(cond_key, frac_key):
    """Get per-seed trajectory of a fraction across transitions. Returns (mean, sem) arrays."""
    all_seeds = []
    for s in SEEDS:
        seed_vals = []
        for t in transitions:
            seed_vals.append(data["conditions"][cond_key]["per_seed"][s]["process_events_per_transition"][t][frac_key] * 100)
        all_seeds.append(seed_vals)
    all_seeds = np.array(all_seeds)  # (n_seeds, n_transitions)
    mean = np.mean(all_seeds, axis=0)
    sem = np.std(all_seeds, axis=0, ddof=1) / np.sqrt(len(SEEDS))
    return mean, sem


# ── Build figure ──
fig = plt.figure(figsize=(180/25.4, 170/25.4))

# Layout: row 0 = (a) full width, row 1 = (b) full width, row 2 = (c) + (d)
gs = GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.32,
              left=0.08, right=0.96, top=0.97, bottom=0.06,
              height_ratios=[1.0, 1.0, 1.0])

# ═══════════════════════════════════════════════════════════════════════════
# Panel (a): Noise-type schematic — concrete CIFAR-100 classes
# ═══════════════════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(gs[0, :])
ax_a.set_xlim(0, 16)
ax_a.set_ylim(-0.5, 11)
ax_a.axis('off')

# -- Design constants --
SC_A_TINT = '#EFF6FF'
SC_B_TINT = '#FFFBEB'
HIGHLIGHT_WITHIN = '#3B82F6'
HIGHLIGHT_BETWEEN = '#DC2626'
HIGHLIGHT_RANDOM = '#6B7280'
ARROW_COLOR = '#4B5563'

# Unified arrow style for all rows
ARROW_KW = dict(arrowstyle='->,head_length=0.4,head_width=0.25',
                color=ARROW_COLOR, lw=1.0, zorder=7)

# -- Class names --
SC_A_NAME = 'Aquatic mammals'
SC_A_CLASSES = ['beaver', 'dolphin', 'otter']
SC_B_NAME = 'Flowers'
SC_B_CLASSES = ['orchid', 'rose', 'tulip']

# -- Row geometry --
row_configs = [
    {
        'label': 'Within-\nsuperclass',
        'y_center': 9.0,
        'description': 'beaver -> dolphin or otter\n(same superclass, p = 0.1, 0.3)',
        'desc_color': '#166534',
        'highlight': 'within',
    },
    {
        'label': 'Between-\nsuperclass',
        'y_center': 5.5,
        'description': 'beaver -> orchid, rose, or tulip\n(different superclass only, p = 0.1, 0.3)',
        'desc_color': '#DC2626',
        'highlight': 'between',
    },
    {
        'label': 'Random',
        'y_center': 2.0,
        'description': 'beaver -> any of 99 other classes\n(no constraint, p = 0.1, 0.3)',
        'desc_color': '#6B7280',
        'highlight': 'random',
    },
]

row_h = 2.5
sc_w = 3.0
gap = 0.6
left_x = 2.3
mid_x = left_x + sc_w + gap / 2
right_x = left_x + sc_w + gap

for cfg in row_configs:
    yc = cfg['y_center']
    y_bot = yc - row_h / 2
    y_top = yc + row_h / 2

    # -- Row label (left) --
    ax_a.text(0.1, yc, cfg['label'], fontsize=7.5, fontweight='bold',
              va='center', ha='left', color='#111827')

    # -- Superclass A region --
    rect_a = FancyBboxPatch((left_x, y_bot), sc_w, row_h,
                             boxstyle='round,pad=0.12',
                             facecolor=SC_A_TINT, edgecolor='#BFDBFE',
                             linewidth=0.6, zorder=1)
    ax_a.add_patch(rect_a)

    # -- Superclass B region --
    rect_b = FancyBboxPatch((right_x, y_bot), sc_w, row_h,
                             boxstyle='round,pad=0.12',
                             facecolor=SC_B_TINT, edgecolor='#FDE68A',
                             linewidth=0.6, zorder=1)
    ax_a.add_patch(rect_b)

    # -- Superclass labels (inside box, top third) --
    sc_label_y = yc + 0.55
    ax_a.text(left_x + sc_w / 2, sc_label_y, SC_A_NAME, fontsize=6.5,
              ha='center', va='center', color='black', fontweight='bold', zorder=5)
    ax_a.text(right_x + sc_w / 2, sc_label_y, SC_B_NAME, fontsize=6.5,
              ha='center', va='center', color='black', fontweight='bold', zorder=5)

    # -- Class name positions (bottom third of box) --
    a_cx = [left_x + 0.5, left_x + sc_w / 2, left_x + sc_w - 0.5]
    b_cx = [right_x + 0.5, right_x + sc_w / 2, right_x + sc_w - 0.5]
    class_y = yc - 0.6

    # -- Draw class labels --
    for cx, name in zip(a_cx, SC_A_CLASSES):
        ax_a.text(cx, class_y, name, fontsize=6.5, ha='center', va='center',
                  color='black', zorder=5)
    for cx, name in zip(b_cx, SC_B_CLASSES):
        ax_a.text(cx, class_y, name, fontsize=6.5, ha='center', va='center',
                  color='black', zorder=5)

    # -- Highlight boxes + arrows --
    hl_lw = 1.0
    hl_style = 'round,pad=0.1'
    # Place arrows below class names to avoid overlapping text
    ar_y_below = class_y - 0.45

    if cfg['highlight'] == 'within':
        # Separate box around each superclass
        for bx in [left_x, right_x]:
            hl = FancyBboxPatch((bx + 0.06, y_bot + 0.06),
                                 sc_w - 0.12, row_h - 0.12,
                                 boxstyle=hl_style,
                                 facecolor='none', edgecolor=HIGHLIGHT_WITHIN,
                                 linewidth=hl_lw, linestyle='--', zorder=6)
            ax_a.add_patch(hl)
        # beaver -> dolphin (arc below class names)
        ax_a.annotate('', xy=(a_cx[1], ar_y_below),
                      xytext=(a_cx[0], ar_y_below),
                      arrowprops=dict(**ARROW_KW, connectionstyle='arc3,rad=0.35'))
        # beaver -> otter (wider arc below)
        ax_a.annotate('', xy=(a_cx[2], ar_y_below - 0.15),
                      xytext=(a_cx[0], ar_y_below - 0.15),
                      arrowprops=dict(**ARROW_KW, connectionstyle='arc3,rad=0.25'))

    elif cfg['highlight'] == 'between':
        # Single box spanning both
        hl = FancyBboxPatch((left_x + 0.06, y_bot + 0.06),
                             right_x + sc_w - left_x - 0.12, row_h - 0.12,
                             boxstyle=hl_style,
                             facecolor='none', edgecolor=HIGHLIGHT_BETWEEN,
                             linewidth=hl_lw, linestyle='--', zorder=6)
        ax_a.add_patch(hl)
        # Red boundary line
        ax_a.plot([mid_x, mid_x], [y_bot + 0.35, y_top - 0.35],
                  color=HIGHLIGHT_BETWEEN, linewidth=1.0, zorder=4)
        # beaver -> orchid (arc below class names)
        ax_a.annotate('', xy=(b_cx[0], ar_y_below),
                      xytext=(a_cx[0], ar_y_below),
                      arrowprops=dict(**ARROW_KW, connectionstyle='arc3,rad=0.12'))
        # beaver -> tulip (wider arc below)
        ax_a.annotate('', xy=(b_cx[2], ar_y_below - 0.15),
                      xytext=(a_cx[0], ar_y_below - 0.15),
                      arrowprops=dict(**ARROW_KW, connectionstyle='arc3,rad=0.08'))

    elif cfg['highlight'] == 'random':
        # Single box spanning both (dotted)
        hl = FancyBboxPatch((left_x + 0.06, y_bot + 0.06),
                             right_x + sc_w - left_x - 0.12, row_h - 0.12,
                             boxstyle=hl_style,
                             facecolor='none', edgecolor=HIGHLIGHT_RANDOM,
                             linewidth=hl_lw, linestyle=':', zorder=6)
        ax_a.add_patch(hl)
        # beaver -> dolphin (short arc below)
        ax_a.annotate('', xy=(a_cx[1], ar_y_below),
                      xytext=(a_cx[0], ar_y_below),
                      arrowprops=dict(**ARROW_KW, connectionstyle='arc3,rad=0.35'))
        # beaver -> rose (long arc below, across boxes)
        ax_a.annotate('', xy=(b_cx[1], ar_y_below - 0.1),
                      xytext=(a_cx[0], ar_y_below - 0.1),
                      arrowprops=dict(**ARROW_KW, connectionstyle='arc3,rad=0.08'))
        # tulip -> otter (reverse arc below, matching curvature)
        ax_a.annotate('', xy=(a_cx[2], ar_y_below - 0.3),
                      xytext=(b_cx[2], ar_y_below - 0.3),
                      arrowprops=dict(**ARROW_KW, connectionstyle='arc3,rad=-0.15'))

    # -- Description sentence (right side, well-padded) --
    ax_a.text(right_x + sc_w + 0.6, yc, cfg['description'],
              fontsize=7, va='center', ha='left',
              color='black', fontstyle='italic',
              linespacing=1.4)

# -- Panel label --
ax_a.text(-0.03, 1.03, 'a', transform=ax_a.transAxes, fontsize=8,
          fontweight='bold', va='top', ha='left')

# ═══════════════════════════════════════════════════════════════════════════
# Panel (b): Grouped bars — all 7 conditions showing Di-E and Ab-E fractions
# ═══════════════════════════════════════════════════════════════════════════
ax_b = fig.add_subplot(gs[1, :])
ax_b.text(-0.04, 1.12, 'b', transform=ax_b.transAxes, fontsize=8,
          fontweight='bold', va='top', ha='left')

x_b = np.arange(len(CONDITION_KEYS))
w_b = 0.35

bars_di = ax_b.bar(x_b - w_b/2, di_frac_mean, w_b, yerr=di_frac_sem,
                   color=DIH_COLOR, alpha=0.8, label='Di-E (differentiation)',
                   ecolor='#374151', capsize=3, error_kw={'linewidth': 0.5})
bars_ab = ax_b.bar(x_b + w_b/2, ab_frac_mean, w_b, yerr=ab_frac_sem,
                   color=ABH_COLOR, alpha=0.8, label='Ab-E (abstraction)',
                   ecolor='#374151', capsize=3, error_kw={'linewidth': 0.5})

# Reference line at standard Di-E level
ax_b.axhline(di_frac_mean[0], color=DIH_COLOR, linestyle=':', linewidth=0.5, alpha=0.4)
ax_b.axhline(ab_frac_mean[0], color=ABH_COLOR, linestyle=':', linewidth=0.5, alpha=0.4)

# Condition group separators and background shading
ax_b.axvspan(-0.5, 0.5, color='#F3F4F6', alpha=0.4, zorder=0)  # standard
ax_b.axvline(0.5, color='#E5E7EB', linewidth=0.5, linestyle='-')
ax_b.axvline(2.5, color='#E5E7EB', linewidth=0.5, linestyle='-')
ax_b.axvline(4.5, color='#E5E7EB', linewidth=0.5, linestyle='-')

# Group labels at top
for x_pos, label in [(0, 'Standard'), (1.5, 'Within-SC'), (3.5, 'Between-SC'), (5.5, 'Random')]:
    ax_b.text(x_pos, max(di_frac_mean) * 1.12, label, ha='center', va='bottom',
              fontsize=6.5, fontweight='bold', color='#374151')

ax_b.set_xticks(x_b)
ax_b.set_xticklabels(['Std', 'p=0.1', 'p=0.3', 'p=0.1', 'p=0.3', 'p=0.1', 'p=0.3'], fontsize=7)
ax_b.set_ylabel('Process fraction (%)')
ax_b.set_ylim(0, max(di_frac_mean) * 1.25)
ax_b.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7,
            fontsize=7, loc='upper right', ncol=1,
            bbox_to_anchor=(1.0, 1.22))
for spine in ['top', 'right']:
    ax_b.spines[spine].set_visible(False)

# Highlight the Between-SC Di-E gap with a red bracket
# Between-SC p=0.3 is index 4; standard Di-E is the reference
bsc_p03_idx = 4  # between_sc_p03
gap_x = bsc_p03_idx - w_b/2  # center of the Di-E bar at between_sc_p03
gap_top = di_frac_mean[0]  # standard level
gap_bot = di_frac_mean[bsc_p03_idx]
gap_pp = gap_top - gap_bot

# Red vertical line showing the drop — offset slightly left to avoid bar overlap
arrow_x = bsc_p03_idx - w_b + 0.0
ax_b.annotate('', xy=(arrow_x, gap_bot), xytext=(arrow_x, gap_top),
              arrowprops=dict(arrowstyle='<->', color='#DC2626', lw=1.5, shrinkA=1, shrinkB=1))
ax_b.text(arrow_x - 0.15, (gap_top + gap_bot) / 2, f'\u2212{gap_pp:.0f} pp',
          fontsize=7, fontweight='bold', color='#DC2626', ha='right', va='center',
          bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))


# ═══════════════════════════════════════════════════════════════════════════
# Panel (c): Dissociation index — between-SC vs random selective effect
# ═══════════════════════════════════════════════════════════════════════════
ax_c = fig.add_subplot(gs[2, 0])
ax_c.text(-0.18, 1.08, 'c', transform=ax_c.transAxes, fontsize=8,
          fontweight='bold', va='top', ha='left')

# Compute per-seed: (Δ_between - Δ_random) for Di-E and Ab-E at p=0.3
std_di_per_seed = [data["conditions"]["standard"]["per_seed"][s]["process_fractions"]["di_frac"] * 100 for s in SEEDS]
std_ab_per_seed = [data["conditions"]["standard"]["per_seed"][s]["process_fractions"]["ab_frac"] * 100 for s in SEEDS]

dissoc_di = []  # between-SC effect minus random effect on Di-E
dissoc_ab = []  # between-SC effect minus random effect on Ab-E
for si, s in enumerate(SEEDS):
    bsc_di = data["conditions"]["between_sc_p03"]["per_seed"][s]["process_fractions"]["di_frac"] * 100
    rnd_di = data["conditions"]["random_p03"]["per_seed"][s]["process_fractions"]["di_frac"] * 100
    bsc_ab = data["conditions"]["between_sc_p03"]["per_seed"][s]["process_fractions"]["ab_frac"] * 100
    rnd_ab = data["conditions"]["random_p03"]["per_seed"][s]["process_fractions"]["ab_frac"] * 100
    # Δ from standard
    delta_bsc_di = bsc_di - std_di_per_seed[si]
    delta_rnd_di = rnd_di - std_di_per_seed[si]
    delta_bsc_ab = bsc_ab - std_ab_per_seed[si]
    delta_rnd_ab = rnd_ab - std_ab_per_seed[si]
    dissoc_di.append(delta_bsc_di - delta_rnd_di)  # negative = between hurts Di-E more
    dissoc_ab.append(delta_bsc_ab - delta_rnd_ab)  # ~0 = no selective effect on Ab-E

dissoc_di = np.array(dissoc_di)
dissoc_ab = np.array(dissoc_ab)

x_c = np.array([0, 1])
bar_w_c = 0.5

ax_c.bar(0, dissoc_di.mean(), bar_w_c, yerr=dissoc_di.std(ddof=1)/np.sqrt(len(SEEDS)),
         color=DIH_COLOR, alpha=0.8, ecolor='#374151', capsize=4, error_kw={'linewidth': 0.5})
ax_c.bar(1, dissoc_ab.mean(), bar_w_c, yerr=dissoc_ab.std(ddof=1)/np.sqrt(len(SEEDS)),
         color=ABH_COLOR, alpha=0.8, ecolor='#374151', capsize=4, error_kw={'linewidth': 0.5})

# Individual seed dots
rng = np.random.default_rng(42)
for vals, x_pos in [(dissoc_di, 0), (dissoc_ab, 1)]:
    jitter = rng.uniform(-0.08, 0.08, size=len(vals))
    ax_c.scatter(x_pos + jitter, vals, s=18, color='#374151', alpha=0.5,
                 edgecolors='white', linewidths=0.3, zorder=5)

ax_c.axhline(0, color='#9CA3AF', linestyle='--', linewidth=0.5, zorder=1)
ax_c.set_xticks(x_c)
ax_c.set_xticklabels(['Di-E\n(differentiation)', 'Ab-E\n(abstraction)'], fontsize=7)
ax_c.set_ylabel('\u0394 between-SC \u2212 \u0394 random (pp)')
ax_c.set_title('Superclass-boundary\nselectivity (p = 0.3)', fontsize=8, fontweight='bold', pad=4)
for spine in ['top', 'right']:
    ax_c.spines[spine].set_visible(False)

# Annotate significance
ax_c.text(0, dissoc_di.mean() - abs(dissoc_di.mean()) * 0.25, f'{dissoc_di.mean():.1f} pp',
          ha='center', va='top', fontsize=7, fontweight='bold', color='white')
ax_c.text(1, max(0.3, dissoc_ab.mean() + 0.5), f'{dissoc_ab.mean():+.1f} pp',
          ha='center', va='bottom', fontsize=7, color='#374151')

# ═══════════════════════════════════════════════════════════════════════════
# Panel (d): Developmental trajectories — 2 subplots overlaying 3 conditions
# ═══════════════════════════════════════════════════════════════════════════
gs_d = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2, 1], hspace=0.35)

x_trans = np.arange(len(transitions))

# Pre-compute trajectories: Construction = Tg-E + Ab-E, Refinement = Di-E
CONSTRUCT_COLOR = '#6366F1'  # indigo
REFINE_COLOR = DIH_COLOR     # amber

COND_STYLES = {
    'standard':       {'linestyle': '-',  'linewidth': 1.5, 'label': 'Standard'},
    'between_sc_p03': {'linestyle': '--', 'linewidth': 1.2, 'label': 'Between-SC p=0.3'},
    'random_p03':     {'linestyle': ':',  'linewidth': 1.2, 'label': 'Random p=0.3'},
}

all_traj_data = {}
for title, cond_key in traj_conditions:
    ab_mean, ab_sem = get_traj_fracs(cond_key, 'ab_frac')
    di_mean, di_sem = get_traj_fracs(cond_key, 'di_frac')
    tg_mean, tg_sem = get_traj_fracs(cond_key, 'tg_frac')
    construct_mean = tg_mean + ab_mean
    construct_sem = np.sqrt(tg_sem**2 + ab_sem**2)
    all_traj_data[cond_key] = {
        'Construction': (construct_mean, construct_sem),
        'Refinement': (di_mean, di_sem),
    }

y_max = max(
    np.max(vals[0] + vals[1])
    for cond_data in all_traj_data.values()
    for vals in cond_data.values()
) * 1.15

# Top: Construction trajectories
ax_d_top = fig.add_subplot(gs_d[0])
ax_d_top.text(-0.15, 1.15, 'd', transform=ax_d_top.transAxes, fontsize=8,
              fontweight='bold', va='top', ha='left')

for title, cond_key in traj_conditions:
    style = COND_STYLES[cond_key]
    mean, sem = all_traj_data[cond_key]['Construction']
    ax_d_top.plot(x_trans, mean, linestyle=style['linestyle'], linewidth=style['linewidth'],
                  color=CONSTRUCT_COLOR, label=style['label'], marker='o', markersize=2.5, zorder=3)
    ax_d_top.fill_between(x_trans, mean - sem, mean + sem, color=CONSTRUCT_COLOR, alpha=0.1, zorder=2)

ax_d_top.set_ylim(0, y_max)
ax_d_top.set_xlim(-0.3, 6.3)
ax_d_top.set_title('Construction (Tg-E + Ab-E)', fontsize=7, fontweight='bold', pad=2)
ax_d_top.set_xticks(x_trans)
ax_d_top.set_xticklabels([])
ax_d_top.set_ylabel('Fraction (%)', fontsize=7)
ax_d_top.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7,
                fontsize=6, loc='upper right')
for spine in ['top', 'right']:
    ax_d_top.spines[spine].set_visible(False)

# Bottom: Refinement trajectories
ax_d_bot = fig.add_subplot(gs_d[1])

for title, cond_key in traj_conditions:
    style = COND_STYLES[cond_key]
    mean, sem = all_traj_data[cond_key]['Refinement']
    ax_d_bot.plot(x_trans, mean, linestyle=style['linestyle'], linewidth=style['linewidth'],
                  color=REFINE_COLOR, label=style['label'], marker='s', markersize=2.5, zorder=3)
    ax_d_bot.fill_between(x_trans, mean - sem, mean + sem, color=REFINE_COLOR, alpha=0.1, zorder=2)

ax_d_bot.set_ylim(0, y_max)
ax_d_bot.set_xlim(-0.3, 6.3)
ax_d_bot.set_title('Refinement (Di-E)', fontsize=7, fontweight='bold', pad=2)
ax_d_bot.set_xticks(x_trans)
ax_d_bot.set_xticklabels(transition_labels, fontsize=6)
ax_d_bot.set_xlabel('Training transition', fontsize=7)
ax_d_bot.set_ylabel('Fraction (%)', fontsize=7)
ax_d_bot.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7,
                fontsize=6, loc='upper right')
for spine in ['top', 'right']:
    ax_d_bot.spines[spine].set_visible(False)

# ── Save ──
for fmt in ['png', 'pdf']:
    fig.savefig(OUT / f'fig5_label_noise.{fmt}', dpi=600, bbox_inches='tight')
    print(f"Saved: {OUT / f'fig5_label_noise.{fmt}'}")

plt.close()
print("Done.")
