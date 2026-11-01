#!/usr/bin/env python3
"""Routing panels of Extended Data Fig. 6 (drawn there as panels c-h by interventions/ed6.py): networks use the
superclass directions in the forward pass to route images to their superclass. Run alone, it draws them lettered a-f.

  a  suppression of the superclass directions in the last block, epochs 1-10: fine-class accuracy minus sham over
     epochs 1-30; right, epochs 1-3 against sham and against random directions
  b  the same for three direction sets (superclass, random grouping, random), assignment and fine-class
  c  where the cost lies: superclass assignment and class given the true superclass
  d  fresh read-out of the unprojected activations: the information stays
  e  superclass share of the classifier input relative to sham
  f  removal from trained networks: superclass directions vs 20 random groupings and random directions; the dose
     statistic over the groupings goes to Source Data only (ed6_f_dose_response*.csv)
Panel functions come from suppression_panels.py and intervention_panels.py and are restyled here (per-panel legends, titles and bar values removed;
one shared key strip, helpers from extended/blocking_routing_panels.py). Source Data are written under this figure's name and letters.
Usage: python scripts/paper/figures/routing/routing_panels.py  (ES_FIG_OUT=<dir> redirects the output)
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
import nmi_style as ns  # noqa: E402
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location("blocking_routing_panels", HERE.parent / "extended" / "blocking_routing_panels.py")   # not figures/granger_panels.py
_E5 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_E5)
CSI, SSI, Canvas, key_strip = _E5.CSI, _E5.SSI, _E5.Canvas, _E5.key_strip
from fig_common import ERR, WINDOW  # noqa: E402

FIG = "ed6"
W_MM, H_MM = 180, 170
A = ns.ARM
LS = ns.COND_LS


def strip_axes(ax, keep=()):
    """Remove the per-panel legend, title and texts (except those whose text contains one of `keep`)."""
    if ax.get_legend():
        ax.get_legend().remove()
    ax.set_title("")
    for t in list(ax.texts):
        if not any(k in t.get_text() for k in keep):
            t.remove()


# a: suppression time course with the epochs 1-3 summary
def panel_a(cv, x, y, w, h, ws=17, gap=6):
    """Time course in the left axes; epochs 1-3 summary in a narrow axes on its right, same y-scale."""
    import suppression_panels
    import intervention_panels
    ax = cv.ax(x, y, w, h)
    cases = suppression_panels.deltas()
    suppression_panels.panel_b([ax], cases, name=FIG, letter="a")
    strip_axes(ax)
    ax.set_ylabel("Fine-class accuracy\nminus sham (pp)")
    ax.text(3.6, 2.8, "epochs 1\u20133", fontsize=6.0, color=ERR, ha="left", va="center")
    ylim = (-9.8, 4.8)
    ax.set_ylim(*ylim)
    sx = x + w + gap
    sm = cv.ax(sx, y, ws, h)
    intervention_panels.panel_supp_cost(sm, name=FIG, letter="a_summary")
    sm.set_title("")
    sm.set_ylabel("")
    sm.set_ylim(*ylim)
    sm.set_yticks([-8, -4, 0])
    sm.tick_params(axis="y", labelleft=False)
    sm.set_xticklabels(["vs\nsham", "vs random\ndirections"], fontsize=6.0, linespacing=0.95)
    for t in sm.texts:
        t.set_fontsize(6.3)
    sm.text(0.5, 1.0, "epochs 1\u20133", transform=sm.transAxes, fontsize=6.0, color=ERR, ha="center", va="top")
    inset = sm
    S = suppression_panels.suppression_first3().set_index("window").loc["last_block_1_10"]
    print(f"[a] vs sham {S.superclass_minus_sham:+.2f} ({S.sham_n_negative}/5 < 0); vs random "
          f"{S.superclass_minus_random:+.2f} ({S.random_n_negative}/5 < 0)")
    return ax, inset


def main():
    import suppression_panels
    import intervention_panels
    ns.apply()
    for f in (ns.FINAL_OUT / "source_data").glob(f"{FIG}_*.csv"):
        f.unlink()
    cv = Canvas(W_MM, H_MM)
    key_strip(cv, [
        [("line", "superclass directions", dict(c=A["superclass"], ls=LS["between_sc"], lw=1.0, marker="o")),
         ("line", "random directions", dict(c=A["random_directions"], ls=LS["random"], lw=1.0, marker="s")),
         ("line", "random groupings", dict(c=A["random_grouping"], lw=0.6)),
         ("line", "sham", dict(c=A["sham"], lw=1.0)),
         ("shade", "suppression, epochs 1–10", dict(c=WINDOW, ec="#D1D5DB"))],
        [("patch", "superclass assignment", dict(c=SSI)),
         ("patch", "fine-class accuracy (c, class given the true superclass)", dict(c=CSI)),
         ("hatch", "control (b, random directions)", dict(c="#6B7280"))],
    ], y_mm=2.5, x0_mm=3)

    # row 1: a | b
    T1, H1 = 22, 32
    xa, wa = 16, 60
    panel_a(cv, xa, T1, wa, H1)
    cv.letter("a", 1, T1 - 9)
    cv.title("Fine-class accuracy falls while the superclass\ndirections are suppressed and recovers on release",
             4.5, T1 - 8.5)
    xb, wb = 122, 56
    b = cv.ax(xb, T1, wb, H1)
    suppression_panels.panel_e(b, name=FIG, letter="b")
    strip_axes(b)
    b.set_xticklabels([t.get_text() for t in b.get_xticklabels()], fontsize=6.2, linespacing=1.0)
    cv.letter("b", 104, T1 - 9)
    cv.title("The cost follows the coarse structure\nthe removed directions hold", 107.5, T1 - 8.5)

    # row 2: c | d | e
    T2, H2 = 77, 29
    xc, wc = 16, 40
    c = cv.ax(xc, T2, wc, H2)
    intervention_panels.panel_supp_where(c, name=FIG, letter="c")
    strip_axes(c)
    c.set_ylim(c.get_ylim()[0], 1.2)
    cv.letter("c", 1, T2 - 9)
    cv.title("The cost lies in\nsuperclass assignment", 4.5, T2 - 8.5)
    xd, wd = 79, 34
    d = cv.ax(xd, T2, wd, H2)
    suppression_panels.panel_fresh(d, name=FIG, letter="d")
    strip_axes(d)
    d.set_ylim(-2.6, 2.6)
    cv.letter("d", 64, T2 - 9)
    cv.title("The information stays\nin the representation", 67.5, T2 - 8.5)
    xe, we = 134, 44
    e = cv.ax(xe, T2, we, H2)
    suppression_panels.panel_d([e], name=FIG, letter="e")
    strip_axes(e, keep=("0.",))
    e.set_title("")
    cv.letter("e", 120, T2 - 9)
    cv.title("The network rebuilds the superclass\nshare while it is suppressed", 123.5, T2 - 8.5)

    # row 3: f | g
    T3 = 128
    xf, wf = 18, 100
    f = cv.ax(xf, T3, wf, 21)
    fm = cv.ax(xf, T3 + 23, wf, 8)
    suppression_panels.panel_a(f, fm, name=FIG, letter="f")
    strip_axes(f, keep=("pp",))
    strip_axes(fm)
    f.set_title("")
    # dose statistic over the 20 groupings: Source Data only
    S = json.load(open(suppression_panels.SPEC / "specificity_summary.json"))["dose_response"]
    ns.source_data(pd.DataFrame(S["per_case"]), FIG, "f_dose_response")
    ns.source_data(pd.read_csv(suppression_panels.SPEC / "dose_response.csv"), FIG, "f_dose_response_groupings")
    print(f"[f] dose-response rho>0 {S['n_rho_positive']}/{S['n_cases']}, median {S['median_rho']:.3f}")
    cv.letter("f", 1, T3 - 9)
    cv.title("From epoch 5, trained networks rely on the superclass\ndirections more than on any random grouping",
             4.5, T3 - 8.5)
    out = ns.save_final(cv.fig, FIG, W_MM, ed=True)
    print("saved", out)


if __name__ == "__main__":
    main()
