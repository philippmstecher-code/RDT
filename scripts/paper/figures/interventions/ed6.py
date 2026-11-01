#!/usr/bin/env python3
"""Extended Data Fig. 6: nesting of class-selective features in the coarse subspace, and use of the coarse directions.

  a, b  nesting (fig4.draw_d_nest, fig4.draw_d_ablation)
  c-h   use of the coarse directions, shifted down by DY mm: c from routing/routing_panels.py, d and f-h from
        suppression_panels.py, e from intervention_panels.py
  i     removal of one superclass's selective features in three architectures (intervention_panels.panel_removal)

Colour = level measured: c, h (fine-class accuracy) in ambers, g (superclass share) in blues; superclass directions
darkest solid, random directions light dotted, sham thin grey, the 20 random groupings grey (null). The shared suppression_panels.py
panel colours are overridden here only (level colours, context manager below). Source Data: ed6_<letter>.csv.

Usage: [ES_FIG_OUT=<dir>] python scripts/paper/figures/interventions/ed6.py
"""
import contextlib
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "routing"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
import nmi_style as ns  # noqa: E402
import routing_panels as ROUT  # noqa: E402  (routing/routing_panels.py: routing panels)
import fig4  # noqa: E402
from fig_common import ERR, WINDOW  # noqa: E402

FIG = "ed6"
DY = 50                      # the routing layout of routing/routing_panels.py moves down by this much
W_MM, H_MM = 180, ROUT.H_MM + DY
A = ns.ARM
LS = ns.COND_LS
Canvas, key_strip, SSI, CSI, strip_axes = ROUT.Canvas, ROUT.key_strip, ROUT.SSI, ROUT.CSI, ROUT.strip_axes
SHAM = ns.GREYS[2]           # reference line
SHAM_LW = 0.8
DARK = ns.GREYS[0]           # key line styles; h margin arrow


@contextlib.contextmanager
def suppression_level(ramp):
    """Temporarily recolour the arms of the shared suppression_panels.py panels by level: superclass directions ramp[0] solid,
    random directions ramp[2] dotted, random groupings grey (null), sham thin grey; effect arrow dark grey."""
    import suppression_panels
    saved = (suppression_panels.ARM_C, suppression_panels.ARM_STYLE, suppression_panels.RED)
    suppression_panels.ARM_C = {"superclass": ramp[0], "random": ramp[2], "grouping": A["random_grouping"], "sham": SHAM}
    suppression_panels.ARM_STYLE = {"superclass": dict(ls="-", marker="o", band=suppression_panels.BAND_A),
                     "random": dict(ls=":", marker="s", band=suppression_panels.BAND_A * 0.6)}
    suppression_panels.RED = DARK
    try:
        yield
    finally:
        suppression_panels.ARM_C, suppression_panels.ARM_STYLE, suppression_panels.RED = saved


def thin_sham(ax):
    for ln in ax.get_lines():
        if ln.get_color() == SHAM and ln.get_linestyle() == "-":
            ln.set_linewidth(SHAM_LW)


def main():
    import suppression_panels
    import intervention_panels
    ns.apply()
    ROUT.FIG = FIG               # panel functions write Source Data under this figure name
    for f in (ns.FINAL_OUT / "source_data").glob(f"{FIG}_*.csv"):
        f.unlink()
    cv = Canvas(W_MM, H_MM)

    # a, b: nesting
    T0, H0 = 13, 26
    xa, wa = 30, 34
    a = cv.ax(xa, T0, wa, H0)
    sd_n, _ = fig4.draw_d_nest(a)
    a.set_title("")
    cv.letter("a", 14, T0 - 9)
    cv.title("Class features form within their\nown superclass’s subspace", 17.5, T0 - 8.5)
    xb, wb = 98, 64
    b = cv.ax(xb, T0, wb, H0)
    sd_a, _ = fig4.draw_d_ablation(b)
    b.set_title("")
    cv.letter("b", 80, T0 - 9)
    cv.title("Few class features are assembled\nfrom coarse features", 83.5, T0 - 8.5)

    # key strip for c-h
    key_strip(cv, [
        [("line", "superclass directions", dict(c=DARK, ls="-", lw=1.0, marker="o")),
         ("line", "random directions", dict(c=DARK, ls=":", lw=1.0, marker="s")),
         ("line", "random groupings", dict(c=A["random_grouping"], lw=0.6)),
         ("line", "sham", dict(c=SHAM, lw=SHAM_LW)),
         ("shade", "suppression, epochs 1–10", dict(c=WINDOW, ec="#D1D5DB"))],
        [("patch", "superclass measures", dict(c=SSI)),
         ("patch", "class measures (e, class given the true superclass)", dict(c=CSI)),
         ("hatch", "control (d, random directions)", dict(c="#6B7280"))],
    ], y_mm=2.5 + DY, x0_mm=3)

    # c, d
    T1, H1 = 22 + DY, 32
    xc, wc = 16, 60
    with suppression_level(ns.AMBER):
        c_ax, _ = ROUT.panel_a(cv, xc, T1, wc, H1)   # Source Data written under letter "a"; renamed below
    thin_sham(c_ax)
    cv.letter("c", 1, T1 - 9)
    cv.title("Fine-class accuracy falls while the superclass\ndirections are suppressed and recovers on release",
             4.5, T1 - 8.5)
    xd, wd = 122, 56
    d = cv.ax(xd, T1, wd, H1)
    suppression_panels.panel_e(d, name=FIG, letter="d")
    strip_axes(d)
    d.set_xticklabels([t.get_text() for t in d.get_xticklabels()], fontsize=6.2, linespacing=1.0)
    cv.letter("d", 104, T1 - 9)
    cv.title("The cost follows the coarse structure\nthe removed directions hold", 107.5, T1 - 8.5)

    # e, f, g
    T2, H2 = 77 + DY, 29
    xe, we = 16, 40
    e = cv.ax(xe, T2, we, H2)
    intervention_panels.panel_supp_where(e, name=FIG, letter="e")
    strip_axes(e)
    e.set_ylim(e.get_ylim()[0], 1.2)
    cv.letter("e", 1, T2 - 9)
    cv.title("Superclass accuracy loses\nmore than class choice", 4.5, T2 - 8.5)
    xf, wf = 79, 34
    f = cv.ax(xf, T2, wf, H2)
    suppression_panels.panel_fresh(f, name=FIG, letter="f")
    strip_axes(f)
    f.set_ylim(-2.6, 2.6)
    cv.letter("f", 64, T2 - 9)
    cv.title("The information stays\nin the representation", 67.5, T2 - 8.5)
    xg, wg = 134, 44
    g = cv.ax(xg, T2, wg, H2)
    with suppression_level(ns.BLUE):
        suppression_panels.panel_d([g], name=FIG, letter="g")
    strip_axes(g, keep=("0.",))
    thin_sham(g)
    for ln in list(g.get_lines()):           # window-end dashed line: the shading already marks the window
        if ln.get_linestyle() == "--" and len(set(ln.get_xdata())) == 1:
            ln.remove()
    for t_ in g.texts:                       # first in-window value: clear of the y-axis spine
        if t_.get_position()[0] == 1:
            t_.set_position((0.0, t_.get_position()[1]))
            t_.set_ha("left")
    g.set_title("")
    cv.letter("g", 120, T2 - 9)
    cv.title("The network rebuilds the superclass\nshare while it is suppressed", 123.5, T2 - 8.5)

    # h
    T3 = 128 + DY
    xh, wh = 18, 100
    h = cv.ax(xh, T3, wh, 21)
    hm = cv.ax(xh, T3 + 23, wh, 8)
    with suppression_level(ns.AMBER):
        suppression_panels.panel_a(h, hm, name=FIG, letter="h")
    strip_axes(h, keep=("pp",))
    strip_axes(hm)
    h.set_title("")
    S = json.load(open(suppression_panels.SPEC / "specificity_summary.json"))["dose_response"]
    ns.source_data(pd.DataFrame(S["per_case"]), FIG, "h_dose_response")
    ns.source_data(pd.read_csv(suppression_panels.SPEC / "dose_response.csv"), FIG, "h_dose_response_groupings")
    cv.letter("h", 1, T3 - 9)
    cv.title("From epoch 5, trained networks rely on the superclass\ndirections more than on any random grouping",
             4.5, T3 - 8.5)

    # i: removal of one superclass's selective features, three architectures
    xi, wi = 136, 42
    i_ = cv.ax(xi, T3, wi, 31)
    intervention_panels.panel_removal(i_, name=FIG, letter="i")
    i_.set_title("")
    if i_.get_legend():
        for tx in i_.get_legend().get_texts():
            tx.set_fontsize(6)
            tx.set_text(tx.get_text().replace(" (S)", "").replace("Superclass-", "superclass-").replace("Matched", "matched"))
    i_.set_xticklabels([t_.get_text() for t_ in i_.get_xticklabels()], fontsize=6)
    cv.letter("i", 124, T3 - 9)
    cv.title("Removing one superclass's features\nmisplaces its images", 127.5, T3 - 8.5)

    # ROUT.panel_a wrote its Source Data under letter a: rename to c, then write a, b
    sd = ns.FINAL_OUT / "source_data"
    for old in sorted(sd.glob(f"{FIG}_a*.csv")):
        old.rename(sd / old.name.replace(f"{FIG}_a", f"{FIG}_c", 1))
    ns.source_data(sd_n, FIG, "a")
    ns.source_data(sd_a, FIG, "b")
    out = ns.save_final(cv.fig, FIG, W_MM, ed=True)
    print("saved", out)


if __name__ == "__main__":
    main()
