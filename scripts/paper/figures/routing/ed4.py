#!/usr/bin/env python3
"""Extended Data Fig. 4: the drawing and data of architecture_spread.py, saved under the Extended Data name with the
.tif output. Panels a, b; Source Data ed4_a.csv, ed4_b.csv, ed4_arch_means.csv, ed4_per_run.csv.
Usage: python scripts/paper/figures/routing/ed4.py  (ES_FIG_OUT=<dir> redirects the output)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import nmi_style as ns  # noqa: E402
import architecture_spread  # noqa: E402

architecture_spread.FIG = "ed4"
_save = ns.save_final
ns.save_final = lambda fig, name, width_mm, ed=False, strict=True: _save(fig, name, width_mm, ed=True, strict=strict)

if __name__ == "__main__":
    architecture_spread.main()
