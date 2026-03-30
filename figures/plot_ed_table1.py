#!/usr/bin/env python3
"""Extended Data Table 1: Overview of all 39 training runs.

Generates a clean table figure.
Includes standard replication, ablations, cross-dataset, curriculum,
and label noise conditions with final validation accuracy.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Universal style ──
plt.rcParams.update({
    'font.family': 'Helvetica',
    'font.size': 7,
})

# ── Data: all 39 training runs ──
# Columns: Run#, Architecture, Params(M), Dataset, Seed, Epochs, Ckpts, SAE Exp., Val Acc(%), Condition, Purpose/Section
columns = [
    'Run', 'Architecture', 'Params\n(M)', 'Dataset', 'Seed',
    'Epochs', 'Ckpts', 'SAE\nExp.', 'Val Acc\n(%)', 'Condition', 'Purpose / Section'
]

# Accuracy notes:
# - Standard 9 runs: not stored in raw lane files; from Methods
#   ResNet-18 ~59% (seed 42), ViT-Small ~62% (seeds 42, 137, 256),
#   CCT-7 ~62-65%. Use '—' where exact val not in data files.
# - Curriculum: from curriculum_switch_summary.json val_acc_terminal
# - Label noise 21: from seed-specific raw lane files (terminal_val_accuracy)
# - 200ep, 8x, indep-init, TinyImageNet: not in data files.

rows = [
    # ── GROUP 1: Standard replication (9 runs) ──
    # ResNet-18, 3 seeds
    ['1',  'ResNet-18',  '11.2', 'CIFAR-100', '42',  '50', '12', '4\u00d7', '\u2014', 'Standard', 'Three-phase ordering (Figs 1\u20134)'],
    ['2',  'ResNet-18',  '11.2', 'CIFAR-100', '137', '50', '12', '4\u00d7', '\u2014', 'Standard', 'Three-phase ordering (Figs 1\u20134)'],
    ['3',  'ResNet-18',  '11.2', 'CIFAR-100', '256', '50', '12', '4\u00d7', '\u2014', 'Standard', 'Three-phase ordering (Figs 1\u20134)'],
    # ViT-Small, 3 seeds
    ['4',  'ViT-Small',  '22.0', 'CIFAR-100', '42',  '50', '12', '4\u00d7', '\u2014', 'Standard', 'Three-phase ordering (Figs 1\u20134)'],
    ['5',  'ViT-Small',  '22.0', 'CIFAR-100', '137', '50', '12', '4\u00d7', '\u2014', 'Standard', 'Three-phase ordering (Figs 1\u20134)'],
    ['6',  'ViT-Small',  '22.0', 'CIFAR-100', '256', '50', '12', '4\u00d7', '\u2014', 'Standard', 'Three-phase ordering (Figs 1\u20134)'],
    # CCT-7, 3 seeds
    ['7',  'CCT-7',      '3.7',  'CIFAR-100', '42',  '50', '12', '4\u00d7', '\u2014', 'Standard', 'Three-phase ordering (Figs 1\u20134)'],
    ['8',  'CCT-7',      '3.7',  'CIFAR-100', '137', '50', '12', '4\u00d7', '\u2014', 'Standard', 'Three-phase ordering (Figs 1\u20134)'],
    ['9',  'CCT-7',      '3.7',  'CIFAR-100', '256', '50', '12', '4\u00d7', '\u2014', 'Standard', 'Three-phase ordering (Figs 1\u20134)'],

    # ── GROUP 2: Ablation & control (3 runs) ──
    ['10', 'ResNet-18',  '11.2', 'CIFAR-100', '42',  '200', '12', '4\u00d7', '\u2014', '200 epochs',       'Extended training (ED Fig 6)'],
    ['11', 'ResNet-18',  '11.2', 'CIFAR-100', '42',  '50',  '12', '8\u00d7', '\u2014', '8\u00d7 expansion', 'SAE capacity robustness (ED Fig 6)'],
    ['12', 'ResNet-18',  '11.2', 'CIFAR-100', '42',  '50',  '12', '4\u00d7', '\u2014', 'Indep. init',      'Artefact control (ED Fig 9)'],

    # ── GROUP 3: Cross-dataset (1 run) ──
    ['13', 'ResNet-18',  '11.2', 'Tiny\nImageNet', '42', '50', '10', '4\u00d7', '\u2014', 'Cross-dataset', 'Cross-dataset generality (Fig 3)'],

    # ── GROUP 4: Curriculum sweep (5 runs) ──
    # Accuracies from curriculum_switch_summary.json: val_acc_terminal
    ['14', 'ResNet-18',  '11.2', 'CIFAR-100', '42', '50', '12',      '4\u00d7', '59.1', 'Standard',     'Scaffolding dose\u2013response (Fig 6)'],
    ['15', 'ResNet-18',  '11.2', 'CIFAR-100', '42', '50', '12\u201314', '4\u00d7', '57.9', 'Switch e05',   'Scaffolding dose\u2013response (Fig 6)'],
    ['16', 'ResNet-18',  '11.2', 'CIFAR-100', '42', '50', '12\u201314', '4\u00d7', '58.2', 'Switch e10',   'Scaffolding dose\u2013response (Fig 6)'],
    ['17', 'ResNet-18',  '11.2', 'CIFAR-100', '42', '50', '12\u201314', '4\u00d7', '58.2', 'Switch e25',   'Scaffolding dose\u2013response (Fig 6)'],
    ['18', 'ResNet-18',  '11.2', 'CIFAR-100', '42', '50', '12\u201314', '4\u00d7', '58.2', 'Switch e30',   'Scaffolding dose\u2013response (Fig 6)'],

    # ── GROUP 5: Label noise (35 runs = 7 conditions × 5 seeds) ──
    # standard (no noise)
    ['19', 'ResNet-18',  '11.2', 'CIFAR-100', '42',  '30', '8', '4\u00d7', '58.5', 'No noise',           'Causal scaffolding (Fig 5)'],
    ['20', 'ResNet-18',  '11.2', 'CIFAR-100', '137', '30', '8', '4\u00d7', '57.8', 'No noise',           'Causal scaffolding (Fig 5)'],
    ['21', 'ResNet-18',  '11.2', 'CIFAR-100', '256', '30', '8', '4\u00d7', '58.2', 'No noise',           'Causal scaffolding (Fig 5)'],
    ['22', 'ResNet-18',  '11.2', 'CIFAR-100', '7',   '30', '8', '4\u00d7', '\u2014', 'No noise',           'Causal scaffolding (Fig 5)'],
    ['23', 'ResNet-18',  '11.2', 'CIFAR-100', '314', '30', '8', '4\u00d7', '\u2014', 'No noise',           'Causal scaffolding (Fig 5)'],
    # within_sc_p01
    ['24', 'ResNet-18',  '11.2', 'CIFAR-100', '42',  '30', '8', '4\u00d7', '57.4', 'Within-SC p=0.1',   'Causal scaffolding (Fig 5)'],
    ['25', 'ResNet-18',  '11.2', 'CIFAR-100', '137', '30', '8', '4\u00d7', '57.4', 'Within-SC p=0.1',   'Causal scaffolding (Fig 5)'],
    ['26', 'ResNet-18',  '11.2', 'CIFAR-100', '256', '30', '8', '4\u00d7', '57.7', 'Within-SC p=0.1',   'Causal scaffolding (Fig 5)'],
    ['27', 'ResNet-18',  '11.2', 'CIFAR-100', '7',   '30', '8', '4\u00d7', '\u2014', 'Within-SC p=0.1',   'Causal scaffolding (Fig 5)'],
    ['28', 'ResNet-18',  '11.2', 'CIFAR-100', '314', '30', '8', '4\u00d7', '\u2014', 'Within-SC p=0.1',   'Causal scaffolding (Fig 5)'],
    # within_sc_p03
    ['29', 'ResNet-18',  '11.2', 'CIFAR-100', '42',  '30', '8', '4\u00d7', '55.7', 'Within-SC p=0.3',   'Causal scaffolding (Fig 5)'],
    ['30', 'ResNet-18',  '11.2', 'CIFAR-100', '137', '30', '8', '4\u00d7', '55.5', 'Within-SC p=0.3',   'Causal scaffolding (Fig 5)'],
    ['31', 'ResNet-18',  '11.2', 'CIFAR-100', '256', '30', '8', '4\u00d7', '55.8', 'Within-SC p=0.3',   'Causal scaffolding (Fig 5)'],
    ['32', 'ResNet-18',  '11.2', 'CIFAR-100', '7',   '30', '8', '4\u00d7', '\u2014', 'Within-SC p=0.3',   'Causal scaffolding (Fig 5)'],
    ['33', 'ResNet-18',  '11.2', 'CIFAR-100', '314', '30', '8', '4\u00d7', '\u2014', 'Within-SC p=0.3',   'Causal scaffolding (Fig 5)'],
    # between_sc_p01
    ['34', 'ResNet-18',  '11.2', 'CIFAR-100', '42',  '30', '8', '4\u00d7', '57.0', 'Between-SC p=0.1',  'Causal scaffolding (Fig 5)'],
    ['35', 'ResNet-18',  '11.2', 'CIFAR-100', '137', '30', '8', '4\u00d7', '57.7', 'Between-SC p=0.1',  'Causal scaffolding (Fig 5)'],
    ['36', 'ResNet-18',  '11.2', 'CIFAR-100', '256', '30', '8', '4\u00d7', '57.3', 'Between-SC p=0.1',  'Causal scaffolding (Fig 5)'],
    ['37', 'ResNet-18',  '11.2', 'CIFAR-100', '7',   '30', '8', '4\u00d7', '\u2014', 'Between-SC p=0.1',  'Causal scaffolding (Fig 5)'],
    ['38', 'ResNet-18',  '11.2', 'CIFAR-100', '314', '30', '8', '4\u00d7', '\u2014', 'Between-SC p=0.1',  'Causal scaffolding (Fig 5)'],
    # between_sc_p03
    ['39', 'ResNet-18',  '11.2', 'CIFAR-100', '42',  '30', '8', '4\u00d7', '54.0', 'Between-SC p=0.3',  'Causal scaffolding (Fig 5)'],
    ['40', 'ResNet-18',  '11.2', 'CIFAR-100', '137', '30', '8', '4\u00d7', '54.3', 'Between-SC p=0.3',  'Causal scaffolding (Fig 5)'],
    ['41', 'ResNet-18',  '11.2', 'CIFAR-100', '256', '30', '8', '4\u00d7', '54.5', 'Between-SC p=0.3',  'Causal scaffolding (Fig 5)'],
    ['42', 'ResNet-18',  '11.2', 'CIFAR-100', '7',   '30', '8', '4\u00d7', '\u2014', 'Between-SC p=0.3',  'Causal scaffolding (Fig 5)'],
    ['43', 'ResNet-18',  '11.2', 'CIFAR-100', '314', '30', '8', '4\u00d7', '\u2014', 'Between-SC p=0.3',  'Causal scaffolding (Fig 5)'],
    # random_p01
    ['44', 'ResNet-18',  '11.2', 'CIFAR-100', '42',  '30', '8', '4\u00d7', '57.7', 'Random p=0.1',      'Causal scaffolding (Fig 5)'],
    ['45', 'ResNet-18',  '11.2', 'CIFAR-100', '137', '30', '8', '4\u00d7', '57.2', 'Random p=0.1',      'Causal scaffolding (Fig 5)'],
    ['46', 'ResNet-18',  '11.2', 'CIFAR-100', '256', '30', '8', '4\u00d7', '57.2', 'Random p=0.1',      'Causal scaffolding (Fig 5)'],
    ['47', 'ResNet-18',  '11.2', 'CIFAR-100', '7',   '30', '8', '4\u00d7', '\u2014', 'Random p=0.1',      'Causal scaffolding (Fig 5)'],
    ['48', 'ResNet-18',  '11.2', 'CIFAR-100', '314', '30', '8', '4\u00d7', '\u2014', 'Random p=0.1',      'Causal scaffolding (Fig 5)'],
    # random_p03
    ['49', 'ResNet-18',  '11.2', 'CIFAR-100', '42',  '30', '8', '4\u00d7', '54.8', 'Random p=0.3',      'Causal scaffolding (Fig 5)'],
    ['50', 'ResNet-18',  '11.2', 'CIFAR-100', '137', '30', '8', '4\u00d7', '54.4', 'Random p=0.3',      'Causal scaffolding (Fig 5)'],
    ['51', 'ResNet-18',  '11.2', 'CIFAR-100', '256', '30', '8', '4\u00d7', '52.9', 'Random p=0.3',      'Causal scaffolding (Fig 5)'],
    ['52', 'ResNet-18',  '11.2', 'CIFAR-100', '7',   '30', '8', '4\u00d7', '\u2014', 'Random p=0.3',      'Causal scaffolding (Fig 5)'],
    ['53', 'ResNet-18',  '11.2', 'CIFAR-100', '314', '30', '8', '4\u00d7', '\u2014', 'Random p=0.3',      'Causal scaffolding (Fig 5)'],
]

n_rows = len(rows)

# ── Group definitions for visual styling ──
# Each group: (label, start_row, end_row, color)
GROUPS = [
    ('Standard replication',          0,  8, '#E8F4FD'),   # Light blue
    ('Ablation & control',            9, 11, '#FFF3E0'),   # Light orange
    ('Cross-dataset',                12, 12, '#E8F5E9'),   # Light green
    ('Curriculum sweep',             13, 17, '#FCE4EC'),   # Light pink
    ('Label noise: no noise',        18, 22, '#F3E8FF'),   # Light purple (5 seeds each)
    ('Label noise: within-SC p=0.1', 23, 27, '#EDE7F6'),
    ('Label noise: within-SC p=0.3', 28, 32, '#F3E8FF'),
    ('Label noise: between-SC p=0.1',33, 37, '#EDE7F6'),
    ('Label noise: between-SC p=0.3',38, 42, '#F3E8FF'),
    ('Label noise: random p=0.1',    43, 47, '#EDE7F6'),
    ('Label noise: random p=0.3',    48, 52, '#F3E8FF'),
]

# Map each row index to its group color
row_colors = ['white'] * n_rows
for _, start, end, color in GROUPS:
    for i in range(start, end + 1):
        row_colors[i] = color

# Group boundaries (0-based row index of last row in group) for thicker separators
group_boundaries = [8, 11, 12, 17]  # After standard, ablation, cross-dataset, curriculum

# ── Create figure ──
fig_width = 190 / 25.4  # 190mm -> inches
row_height = 0.20
fig_height = (n_rows + 3) * row_height  # header + rows + padding

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.axis('off')

# Column widths (proportional) — 11 columns
col_widths = [
    0.030,  # Run
    0.065,  # Architecture
    0.035,  # Params
    0.055,  # Dataset
    0.030,  # Seed
    0.035,  # Epochs
    0.030,  # Ckpts
    0.030,  # SAE Exp.
    0.040,  # Val Acc
    0.095,  # Condition
    0.195,  # Purpose/Section
]

table = ax.table(
    cellText=rows,
    colLabels=columns,
    cellLoc='center',
    colWidths=col_widths,
    loc='center',
)

table.auto_set_font_size(False)
table.set_fontsize(5.0)

# Style header
for j in range(len(columns)):
    cell = table[0, j]
    cell.set_facecolor('#2C3E50')
    cell.set_text_props(color='white', fontweight='bold', fontsize=5.5)
    cell.set_edgecolor('#CCCCCC')
    cell.set_height(0.030)

# Style data rows
for i in range(1, n_rows + 1):
    row_idx = i - 1  # 0-based
    for j in range(len(columns)):
        cell = table[i, j]
        cell.set_edgecolor('#CCCCCC')
        cell.set_height(0.019)
        cell.set_facecolor(row_colors[row_idx])

    # Thicker border at group boundaries
    if row_idx in group_boundaries:
        for j in range(len(columns)):
            cell = table[i, j]
            cell.set_edgecolor('#888888')

# Left-align the Purpose/Section column (last column, index 10)
for i in range(n_rows + 1):  # includes header
    cell = table[i, 10]
    cell._loc = 'left'
    cell.PAD = 0.02

# Left-align the Condition column (index 9)
for i in range(1, n_rows + 1):
    cell = table[i, 9]
    cell._loc = 'left'
    cell.PAD = 0.02

# ── Title ──
plt.title(
    f'Extended Data Table 1 | Overview of all {n_rows} training runs',
    fontsize=9, fontweight='bold', pad=12, loc='left'
)

# ── Footnotes ──
footnote_text = (
    'Runs 1\u20139: 3 architectures \u00d7 3 seeds. '
    'All ViT runs use ViT-Small (22.0M). '
    'Runs 10\u201312: ablation and artefact controls. '
    'Run 13: cross-dataset (Tiny ImageNet, 200 classes, 27 superclasses). '
    'Runs 14\u201318: curriculum sweep (warm-restart LR). '
    'Runs 19\u201353: label noise (7 conditions \u00d7 5 seeds, 30 epochs). '
    '\u2014 = accuracy not stored in data package.'
)
fig.text(
    0.02, 0.005, footnote_text,
    fontsize=4.5, color='#555555', ha='left', va='bottom',
    wrap=True
)

plt.tight_layout(rect=[0, 0.025, 1, 1])

# Save both PNG and PDF
for fmt in ['png', 'pdf']:
    fig.savefig(str(OUT / f'ed_table1.{fmt}'), dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {OUT / f'ed_table1.{fmt}'}")

plt.close()
print(f"\nTotal runs: {n_rows}")
