"""Shared style for all figures: colours, fonts, panel letters, export and Source Data tables.

Colour conventions (consistent across all figures of the paper):
  processes  Tg-E purple, Ab-E blue, Di-E amber, As-E cyan, De-E red
  indices    SAI purple, SSI blue, CSI amber (same hues as Tg-E/Ab-E/Di-E)
  archs      ResNet-18 steel blue, ViT-Small teal, CCT-7 coral; Tiny ImageNet pink
  fills/bars use the base colour at alpha FILL_ALPHA
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROC = {"Tg-E": "#8B5CF6", "Ab-E": "#3B82F6", "Di-E": "#F59E0B", "As-E": "#0891B2", "De-E": "#EF4444"}
INDEX = {"SAI": "#8B5CF6", "SSI": "#3B82F6", "CSI": "#F59E0B"}
ARCH = {"ResNet-18": "#2C5F8A", "ViT-Small": "#2A9D8F", "CCT-7": "#E76F51"}
ARCH_LS = {"ResNet-18": "-", "ViT-Small": "--", "CCT-7": "-."}
TINY = "#F472B6"
NEUTRAL = "#6B7280"
ACCENT_RED = PROC["De-E"]  # palette red; only for annotations, never for conditions
FILL_ALPHA = 0.75
# Label-noise / intervention conditions: line styles for conditions, colours for processes
COND_LS = {"standard": "-", "between_sc": "--", "random": ":", "within_sc": "-."}
# Semantic code for interventions (same hues as the hierarchy levels): blue = superclass-level
# manipulation (Ab-E/SSI colour), amber = class-level manipulation (Di-E/CSI colour), grey = controls,
# dark grey = reference/clean.
REF_DARK = "#374151"
COND_COLOR = {"standard": REF_DARK, "between_sc": PROC["Ab-E"], "within_sc": PROC["Di-E"], "random": NEUTRAL}
ARM_COLOR = {"sc_true": PROC["Ab-E"], "sc_rand": NEUTRAL, "scratch": REF_DARK,
             "scaffold": PROC["Ab-E"], "random": NEUTRAL, "sham": REF_DARK}

MM = 1 / 25.4
FULL_W = 180 * MM      # double-column width (Nature)
HALF_W = 88 * MM

OUT = Path(__file__).resolve().parents[3] / "analysis_outputs" / "paper" / "figures"


def apply():
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],  # Helvetica.ttc has no bold face for matplotlib
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "axes.titleweight": "normal",
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "lines.linewidth": 1.1, "lines.markersize": 3,
        "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.dpi": 600, "figure.dpi": 150,
    })


def panel(ax, letter, x=-0.14, y=1.06):
    """Lower-case bold panel letter, top-left."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top", ha="left")


# Final-figure export
BAND_ALPHA_CORE = 0.2   # trajectory bands
BAR_ALPHA_CORE = 0.8    # bars
SLATE = "#64748B"            # null / random-group reference
INDIGO = "#6366F1"
RED_600 = "#DC2626"
# Curriculum sweep, keyed by pre-training epochs: kept outside the process hues. 0 = scratch dark grey; 20 fuchsia
# (= true-superclass arm); 30 red; 5-15 teal -> pink, 25 raspberry.
SWEEP = {0: ("#374151", "o"), 5: ("#0F766E", "s"), 10: ("#14B8A6", "D"), 15: ("#F472B6", "P"),
         20: ("#D946EF", "X"), 25: ("#BE185D", "^"), 30: ("#DC2626", "v")}
# Intervention arms, one palette for every figure: colour = arm, never one of the process hues. Superclass directions
# take the true-superclass fuchsia, random-grouping directions slate.
ARM = {"superclass": "#D946EF", "random_grouping": "#64748B", "random_directions": "#374151",
       "energy": "#16A34A", "joint": "#0F766E", "random_redrawn": "#374151", "sham": "#9CA3AF"}
# Label-noise conditions, where conditions are coloured: between-superclass red, within-superclass teal, random light
# grey, clean dark grey.
NOISE = {"clean": "#374151", "between": "#DC2626", "within": "#0D9488", "random": "#9CA3AF"}
# Level colour rule. Colour encodes the level measured: coarse / superclass measures blue, fine / class measures amber, task-general
# purple, references (scratch, clean) grey. Conditions are told apart within the hue by shade, line style, markers
# and direct labels: pre-built on the true superclasses darkest + solid; random grouping mid + dashed;
# superclass-blind task light + dotted; scratch thin grey solid. Ramps run dark -> light.
BLUE = ["#1E40AF", "#3B82F6", "#60A5FA", "#93C5FD"]
AMBER = ["#92400E", "#D97706", "#F59E0B", "#FCD34D"]
PURPLE = ["#5B21B6", "#8B5CF6", "#A78BFA", "#C4B5FD"]
GREYS = ["#374151", "#6B7280", "#9CA3AF", "#E5E7EB"]
# text-safe label colours (direct labels, annotations)
BLUE_TXT = ("#1E40AF", "#2563EB")
AMBER_TXT = ("#92400E", "#B45309")
GREY_TXT = "#6B7280"
# condition -> (ramp index, line style); scratch is drawn GREYS[1] thin solid
COND_SHADE = {"sc_true": (0, "-"), "sc_rand": (1, "--"), "sc_blind": (2, ":")}
SCRATCH_STYLE = dict(color=GREYS[1], ls="-", lw=0.9)
FINAL_OUT = Path(os.environ["ES_FIG_OUT"]) if os.environ.get("ES_FIG_OUT") else OUT / "final"  # output override
MIN_PT = 6.0


def check_text_sizes(fig, min_pt=MIN_PT):
    """Return [(text, size)] for every visible text element below min_pt (ticks included)."""
    from matplotlib.text import Text
    bad = []
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip() and t.get_fontsize() < min_pt - 1e-6:
            bad.append((t.get_text()[:40], round(t.get_fontsize(), 2)))
    return bad


def save_final(fig, name, width_mm, ed=False, strict=True):
    """Save a final figure at its exact print width (no tight cropping).

    - width_mm must be 180 (full) or 88 (half); the figure canvas is checked against it.
    - every visible text must be >= 6 pt (strict=True raises, else prints).
    - writes PDF + PNG (600 dpi) to FINAL_OUT, plus a 600-dpi LZW TIFF for Extended Data figures.
    """
    import re
    from matplotlib.text import Text
    w_in, _ = fig.get_size_inches()
    if abs(w_in * 25.4 - width_mm) > 0.5:
        raise ValueError(f"{name}: canvas width {w_in * 25.4:.1f} mm != {width_mm} mm")
    # Only panel letters are bold; every other text is regular weight.
    from matplotlib.text import Text as _Tw
    for t in fig.findobj(_Tw):
        txt = t.get_text().strip()
        is_panel_letter = len(txt) == 1 and txt.isalpha() and txt.islower() and t.get_fontsize() >= 7.9
        if not is_panel_letter and t.get_fontweight() not in ("normal", 400, "regular"):
            t.set_fontweight("normal")
    # Nature figure guide: panel letters 8 pt bold, all other text 5-7 pt (>= 6 pt kept as our floor).
    from matplotlib.text import Text as _T
    big = [(t.get_text()[:30], round(t.get_fontsize(), 1)) for t in fig.findobj(_T)
           if t.get_visible() and t.get_text().strip() and t.get_fontsize() > 7.01
           and not (len(t.get_text().strip()) == 1 and t.get_fontsize() <= 8.01)]
    if big:
        print(f"NOTE {name}: {len(big)} text elements > 7 pt (Nature: 5-7 pt except 8-pt panel letters): {big[:8]}")
    bad = check_text_sizes(fig)
    if bad:
        msg = f"{name}: {len(bad)} text elements < {MIN_PT} pt: {bad[:8]}"
        if strict:
            raise ValueError(msg)
        print("WARNING", msg)
    FINAL_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FINAL_OUT / f"{name}.pdf")
    fig.savefig(FINAL_OUT / f"{name}.png", dpi=600)
    if ed:
        fig.savefig(FINAL_OUT / f"{name}.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return FINAL_OUT / f"{name}.png"


def source_data(df, fig_name, panel):
    """Write the data behind one panel to FINAL_OUT/source_data/<fig>_<panel>.csv (Source Data)."""
    d = FINAL_OUT / "source_data"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{fig_name}_{panel}.csv"
    df.to_csv(path, index=False)
    return path


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    return OUT / f"{name}.png"


def rdt_tag(ax, x=0.98, y=0.98, ha="right", va="top"):
    """Panel tag for RDT-derived panels; disabled (returns None)."""
    return None
