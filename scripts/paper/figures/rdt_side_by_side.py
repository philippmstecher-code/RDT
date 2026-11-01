#!/usr/bin/env python3
"""RDT and SAE-free read-outs of one intervention side by side, as % change (intervention vs control) per seed:
between- vs within-superclass label noise. Also imported by interventions/ed8.py and routing/noise_routing_panels.py.

RDT (common threshold set per seed = clean run's stored thresholds; scripts/paper/noise_rdt_analysis.py,
analysis_outputs/paper/RDT_noise/per_lane):
  superclass-selective features   % of alive features, census layers (layer1-3, pooled), epoch 30
  Ab-E events                      all layers, transitions 0->30 summed
  Di-E events                      all layers, transitions 0->30 summed
  class-selective features         % of alive features, census layers, epoch 30
SAE-free (noise_lib.static_probe / static_accuracy):
  superclass probe, epoch 30; fine-class accuracy, epoch 30
Bars = mean over seeds of the per-seed % change; dots = seeds; error bars = t-based 95% CI.
Usage: python scripts/paper/figures/rdt_side_by_side.py [out.png]
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nmi_style as ns  # noqa: E402
import noise_lib as L  # noqa: E402
from fig_common import RESULTS  # noqa: E402

SEEDS = [7, 42, 137, 256, 314]
BASE = RESULTS / "RDT_noise" / "per_lane"
CENSUS = ("layer1", "layer2", "layer3", "avgpool")
SC, CL = ns.INDEX["SSI"], ns.INDEX["CSI"]


def lane(arm, s):
    return json.load(open(BASE / f"{arm}_seed{s}.json"))


def census(arm, s, cat, lab="6"):
    c = lane(arm, s)["census"][lab]["common"]
    return 100 * sum(c[x][cat] for x in CENSUS) / max(sum(c[x]["n_alive"] for x in CENSUS), 1)


def events(arm, s, p):
    r = lane(arm, s)
    return sum(r["events"][k]["common"][x][p] for k in r["events"] for x in r["layers"])


def rel(b, w):
    b, w = np.asarray(b, float), np.asarray(w, float)
    return 100 * (b / w - 1)


def measures():
    P, A = L.static_probe(), L.static_accuracy()
    return [
        ("superclass-\nselective\nfeatures", "rdt", SC,
         rel([census("between", s, "n_sc") for s in SEEDS], [census("within", s, "n_sc") for s in SEEDS])),
        ("superclass\nevents\n(Ab-E)", "rdt", SC,
         rel([events("between", s, "ab_h") for s in SEEDS], [events("within", s, "ab_h") for s in SEEDS])),
        ("class\nevents\n(Di-E)", "rdt", CL,
         rel([events("between", s, "di_h") for s in SEEDS], [events("within", s, "di_h") for s in SEEDS])),
        ("class-\nselective\nfeatures", "rdt", CL,
         rel([census("between", s, "n_cls") for s in SEEDS], [census("within", s, "n_cls") for s in SEEDS])),
        ("superclass\nprobe", "free", SC, rel(P["between_sc"][30].values, P["within_sc"][30].values)),
        ("fine-class\naccuracy", "free", CL, rel(A["between_sc"][:, 30], A["within_sc"][:, 30])),
    ]


def panel(ax):
    rng = np.random.default_rng(5)
    M = measures()
    xs = [0, 1, 2, 3, 4.6, 5.6]
    for x, (lab, kind, col, v) in zip(xs, M):
        m, lo, hi = L.t_ci(v)
        if kind == "rdt":
            ax.bar(x, m, width=0.62, color=col, alpha=ns.BAR_ALPHA_CORE, lw=0, zorder=1)
        else:
            ax.bar(x, m, width=0.62, facecolor="white", edgecolor=col, lw=1.0, zorder=1)
        ax.scatter(x + rng.uniform(-0.13, 0.13, len(v)), v, s=7, color=L.DOT, lw=0, zorder=3)
        ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=L.ERR, elinewidth=0.9, capsize=2.2,
                    capthick=0.9, zorder=4)
        y = (max(hi, v.max()) + 4, "bottom") if m > 0 else (min(lo, v.min()) - 4, "top")
        ax.text(x, y[0], f"{m:+.0f}%".replace("-", "−"), fontsize=6.3, color=L.ERR, ha="center", va=y[1])
        print(f"{lab.replace(chr(10), ' '):34s} {kind:4s} {m:+6.1f}% [{lo:+.1f}, {hi:+.1f}]  seeds "
              f"{int((np.sign(v) == np.sign(m)).sum())}/5")
    ax.axhline(0, color="#6B7280", ls="--", lw=0.6, zorder=0)
    ax.axvline(3.8, color="#D1D5DB", lw=0.6, zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([m[0] for m in M], fontsize=6.2, linespacing=1.0)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(-0.6, 6.2)
    ax.set_ylim(-60, 145)
    ax.set_yticks([-50, 0, 50, 100])
    ax.set_ylabel("Between- vs within-superclass\nnoise (% change)")
    ax.text(1.5, 143, "RDT (features and events)", fontsize=6.5, color=L.ERR, ha="center", va="top")
    ax.text(5.1, 143, "without SAEs", fontsize=6.5, color=L.ERR, ha="center", va="top")
    ax.set_title("Between- vs within-superclass noise: RDT and SAE-free read-outs", fontsize=7, pad=3)


def main():
    ns.apply()
    W, H = 92, 52
    fig = plt.figure(figsize=(W * ns.MM, H * ns.MM))
    ax = fig.add_axes([16 / W, 13 / H, 74 / W, 33 / H])
    panel(ax)
    out = sys.argv[1] if len(sys.argv) > 1 else "rdt_side_by_side.png"
    fig.savefig(out, dpi=300)
    print("saved", out)


if __name__ == "__main__":
    main()
