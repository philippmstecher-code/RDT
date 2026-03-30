#!/usr/bin/env python3
"""Figure 1: Method and Feature Lifecycle — detailed FDT flowchart + selectivity + Ab-E/Di-E ratio.

Layout:
  Row 1 (full width): Panel (a) — detailed 5-row FDT pipeline flowchart with real example
  Row 2 (two columns): Panel (b) — selectivity evolution, Panel (c) — Ab-E/Di-E ratio
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Arc, Rectangle
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
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
    'text.color': 'black',
    'axes.labelsize': 8,
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
INDEX_COLORS = {'SSI': '#3B82F6', 'CSI': '#F59E0B', 'SAI': '#8B5CF6'}

# Lifecycle state colours
BORN_COLOR = '#0891B2'     # green
STABLE_COLOR = '#3B82F6'   # blue
DIED_COLOR = '#EF4444'     # red

# ── Figure layout: panels (b) and (c) only ──
# Panel (a) will be composited from the vector PDF afterwards.
fig_w = 180 / 25.4   # 180 mm → inches
# Height for b+c panels only (panel a added via PDF compositing)
bc_h = 70 / 25.4
fig = plt.figure(figsize=(fig_w, bc_h))

gs = gridspec.GridSpec(1, 2, figure=fig,
                       width_ratios=[1, 1],
                       left=0.09, right=0.97, bottom=0.14, top=0.92,
                       wspace=0.30)

ax_b = fig.add_subplot(gs[0, 0])
ax_c = fig.add_subplot(gs[0, 1])


# ══════════════════════════════════════════════════════════════════════════════
# Panel (b): Stacked area — process composition per transition
# ══════════════════════════════════════════════════════════════════════════════
ax_b.text(-0.15, 1.05, 'b', transform=ax_b.transAxes, fontsize=8,
          fontweight='bold', va='top', ha='left')

primary = 'ResNet18-CIFAR100-seed42'

cifar_lanes = [k for k in sorted(data['lanes'].keys())
               if data['lanes'][k]['metadata']['dataset'] == 'CIFAR-100'
               and data['lanes'][k]['metadata']['epochs'] == 50
               and data['lanes'][k]['metadata']['expansion'] == '4x']
n_trans = min(len(data['lanes'][ln]['abh_dih_ratios']) for ln in cifar_lanes)

tg_all = np.zeros((len(cifar_lanes), n_trans))
nontg_all = np.zeros((len(cifar_lanes), n_trans))
di_all = np.zeros((len(cifar_lanes), n_trans))

for i, ln in enumerate(cifar_lanes):
    ratios = data['lanes'][ln]['abh_dih_ratios']
    for j in range(n_trans):
        r = ratios[j]
        total = r['ab_h'] + r['tg_h'] + r['di_h']
        if total > 0:
            tg_all[i, j] = r['tg_h'] / total * 100
            nontg_all[i, j] = r['ab_h'] / total * 100
            di_all[i, j] = r['di_h'] / total * 100

x = np.arange(n_trans)
n_lanes = len(cifar_lanes)

tg_mean = tg_all.mean(0)
nontg_mean = nontg_all.mean(0)
di_mean = di_all.mean(0)

ax_b.fill_between(x, 0, tg_mean, color=HYPO_COLORS['Tg-E'],
                   alpha=0.70, label='Tg-E', zorder=2)
ax_b.fill_between(x, tg_mean, tg_mean + nontg_mean, color=HYPO_COLORS['Ab-E'],
                   alpha=0.70, label='Ab-E', zorder=2)
ax_b.fill_between(x, tg_mean + nontg_mean, tg_mean + nontg_mean + di_mean,
                   color=HYPO_COLORS['Di-E'], alpha=0.70,
                   label='Di-E', zorder=2)

ax_b.plot(x, tg_mean, color='white', linewidth=0.6, zorder=3)
ax_b.plot(x, tg_mean + nontg_mean, color='white', linewidth=0.6, zorder=3)

total_ab_mean = tg_mean + nontg_mean
cross_x_ab_di = None
for j in range(1, n_trans):
    if di_mean[j] > total_ab_mean[j]:
        if total_ab_mean[j-1] >= di_mean[j-1]:
            gap_prev = total_ab_mean[j-1] - di_mean[j-1]
            gap_curr = di_mean[j] - total_ab_mean[j]
            cross_x_ab_di = (j - 1) + gap_prev / (gap_prev + gap_curr)
        else:
            cross_x_ab_di = j - 0.5
        # annotation suppressed
        break

cross_x_tg_ab = None
for j in range(1, n_trans):
    if nontg_mean[j] > tg_mean[j]:
        if tg_mean[j-1] >= nontg_mean[j-1]:
            gap_prev = tg_mean[j-1] - nontg_mean[j-1]
            gap_curr = nontg_mean[j] - tg_mean[j]
            cross_x_tg_ab = (j - 1) + gap_prev / (gap_prev + gap_curr)
        else:
            cross_x_tg_ab = j - 0.5
        ax_b.axvline(cross_x_tg_ab, color='#374151', linewidth=0.7, linestyle=':',
                      alpha=0.6, zorder=4)
        break

x_min = -0.5
x_max = n_trans - 0.5
phase_i_end = 2.0
phase_iii_start = 5.0

ax_b.axvspan(x_min, phase_i_end, alpha=0.06, color=HYPO_COLORS['Tg-E'], zorder=0)
ax_b.axvspan(phase_i_end, phase_iii_start, alpha=0.06, color=HYPO_COLORS['Ab-E'], zorder=0)
ax_b.axvspan(phase_iii_start, x_max, alpha=0.06, color=HYPO_COLORS['Di-E'], zorder=0)

label_y = 105
ax_b.text((x_min + phase_i_end) / 2, label_y, 'Phase I\nTask-general',
          ha='center', va='bottom', fontsize=6, color='black',
          fontweight='bold', linespacing=1.1, zorder=6)
ax_b.text((phase_i_end + phase_iii_start) / 2, label_y, 'Phase II\nAbstraction',
          ha='center', va='bottom', fontsize=6, color='black',
          fontweight='bold', linespacing=1.1, zorder=6)
ax_b.text((phase_iii_start + x_max) / 2, label_y, 'Phase III\nDifferentiation',
          ha='center', va='bottom', fontsize=6, color='black',
          fontweight='bold', linespacing=1.1, zorder=6)

ax_b.set_xticks(range(n_trans))
ax_b.set_xticklabels([str(i) for i in range(n_trans)])
ax_b.set_xlim(-0.5, n_trans - 0.5)
ax_b.set_xlabel('Training transition')
ax_b.set_ylabel('Process composition (%)')
ax_b.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7, loc='center right', fontsize=6, handlelength=1.5)
ax_b.text(0.98, 0.45, f'n = {n_lanes} runs', fontsize=6,
          fontstyle='italic', color='#6B7280', transform=ax_b.transAxes,
          ha='right', va='top')
ax_b.set_ylim(0, 118)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)


# ══════════════════════════════════════════════════════════════════════════════
# Panel (c): C/R ratio (construction/refinement) — median + IQR + architecture means
# ══════════════════════════════════════════════════════════════════════════════
ax_c.text(-0.15, 1.05, 'c', transform=ax_c.transAxes, fontsize=8,
          fontweight='bold', va='top', ha='left')

# Use only the 9 standard CIFAR-100 runs (3 arch × 3 seeds, 50 epochs, 4× expansion)
standard_lanes = [k for k in sorted(data['lanes'].keys())
                  if data['lanes'][k]['metadata']['dataset'] == 'CIFAR-100'
                  and data['lanes'][k]['metadata']['epochs'] == 50
                  and data['lanes'][k]['metadata']['expansion'] == '4x']

arch_ratios = {}
all_ratios = []
for label in standard_lanes:
    lane = data['lanes'][label]
    arch = lane['metadata']['architecture']
    ratios = lane['abh_dih_ratios']
    # C/R ratio = (Ab-E + Tg-E) / Di-E
    vals = [(r['ab_h'] + r['tg_h']) / r['di_h'] if r['di_h'] > 0 else float('inf') for r in ratios]
    vals = [v if np.isfinite(v) else 1000 for v in vals]
    all_ratios.append(vals)
    if arch not in arch_ratios:
        arch_ratios[arch] = []
    arch_ratios[arch].append(vals)

max_len = max(len(v) for v in all_ratios)
padded = np.array([v + [v[-1]] * (max_len - len(v)) for v in all_ratios])

median = np.median(padded, axis=0)
q25 = np.percentile(padded, 25, axis=0)
q75 = np.percentile(padded, 75, axis=0)
x = np.arange(max_len)

ax_c.set_xlim(-0.5, max_len - 0.5)
ax_c.fill_between(x, q25, q75, alpha=0.12, color='#6B7280', label='IQR', zorder=2)
ax_c.semilogy(x, median, color='#374151', linewidth=1.0, label='_nolegend_', zorder=3)

ARCH_LINESTYLES = {'ResNet-18': '-', 'ViT-Small': '--', 'CCT-7': '-.'}
for arch, color in ARCH_COLORS.items():
    if arch in arch_ratios:
        arch_vals = arch_ratios[arch]
        arch_max = max(len(v) for v in arch_vals)
        arch_padded = np.array([v + [v[-1]] * (arch_max - len(v)) for v in arch_vals])
        arch_mean = np.mean(arch_padded, axis=0)
        ax_c.semilogy(range(arch_max), arch_mean, color=color, linewidth=1.0,
                      linestyle=ARCH_LINESTYLES.get(arch, '-'),
                      label=arch, alpha=0.8, zorder=3)

ax_c.axhline(1, color='black', linestyle='--', linewidth=0.5, alpha=0.4, label='Ratio = 1', zorder=3)
ax_c.set_xticks(range(n_trans))
ax_c.set_xticklabels([str(i) for i in range(n_trans)])
ax_c.set_xlabel('Training transition')
ax_c.set_ylabel('C/R ratio')
ax_c.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.7, fontsize=6, loc='upper right')
ax_c.set_ylim(0.01, None)
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)


# ── Save panels (b)+(c) as intermediate PDF ──
bc_pdf = OUT / '_fig1_bc.pdf'
plt.savefig(bc_pdf, dpi=600)
plt.close()

# ── Composite: stack Pipeline_overview.pdf (panel a) on top of b+c ──
panel_a_path = OUT / 'Figure1_pipeline_v3  -  Repariert.pdf'
if not panel_a_path.exists():
    print(f"Saved _fig1_bc.pdf (data panels b+c)")
    print(f"SKIP compositing: panel (a) schematic PDF not found at {panel_a_path}")
    print("Place the pipeline schematic PDF there to generate the full composite figure.")
    import sys; sys.exit(0)

from pypdf import PdfReader, PdfWriter, Transformation, PageObject

panel_a_reader = PdfReader(panel_a_path)
bc_reader = PdfReader(bc_pdf)

page_a = panel_a_reader.pages[0]
page_bc = bc_reader.pages[0]

# Get dimensions (in PDF points, 1 pt = 1/72 inch)
a_w = float(page_a.mediabox.width)
a_h = float(page_a.mediabox.height)
bc_w = float(page_bc.mediabox.width)
bc_h = float(page_bc.mediabox.height)

# Scale panel (a) to match the width of panels (b)+(c)
scale = bc_w / a_w
scaled_a_h = a_h * scale

# Add padding between panel (a) and panels (b)+(c)
padding = 10  # points

# Create combined page
total_h = scaled_a_h + padding + bc_h
combined = PageObject.create_blank_page(width=bc_w, height=total_h)

# Place panels (b)+(c) at the bottom
combined.merge_page(page_bc)

# Place panel (a) at the top, scaled to full width
op = Transformation().scale(scale, scale).translate(0, bc_h + padding)
combined.merge_transformed_page(page_a, op)

# Add "a" label as a vector overlay
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.units import mm
import io

label_buf = io.BytesIO()
c = rl_canvas.Canvas(label_buf, pagesize=(bc_w, total_h))
c.setFont("Helvetica", 8)
# Position: left margin, above panel (a) content
c.drawString(2, total_h - 9, "a")
c.save()
label_buf.seek(0)
label_page = PdfReader(label_buf).pages[0]
combined.merge_page(label_page)

writer = PdfWriter()
writer.add_page(combined)
writer.write(str(OUT / 'fig1.pdf'))
print(f"Saved fig1.pdf (vector composite: {bc_w:.0f}×{total_h:.0f} pt)")

# Also render a high-quality PNG
import subprocess
subprocess.run([
    'sips', '-s', 'format', 'png', '--resampleWidth', '7200',
    str(OUT / 'fig1.pdf'), '--out', str(OUT / 'fig1.png')
], check=True)
print("Saved fig1.png (from vector PDF)")

# Clean up intermediate
bc_pdf.unlink()

