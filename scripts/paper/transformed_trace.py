#!/usr/bin/env python3
"""Run scripts/trace_transformed_features.py unchanged with three settings applied from outside:

  1. lanes: the 9 core CIFAR-100 lanes (its LANES list omits the three seed-256 lanes and includes three
     ablation lanes without activations);
  2. outputs: redirected to analysis_outputs/paper/TRANSFORMED/trace/;
  3. terminal transition: 'terminal' is dropped from metadata.checkpoint_labels when sae_results.json is read, so
     only real milestone transitions are analysed ('<k>->terminal' compares the final model with itself and is
     reported by transformed_analysis.py).

Usage: transformed_trace.py [lane_prefix,...]
"""
from __future__ import annotations

import json as _json
import sys
import types
from pathlib import Path
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "packages" / "ml" / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import trace_transformed_features as T  # noqa: E402

DATA = Path(RDT_DATA + "/core")
CORE = DATA / "experiments" / "abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb" / "lanes"
OUT = REPO / "analysis_outputs" / "paper" / "TRANSFORMED" / "trace"
CORE_LANES = [
    ("7db28a9a-f436-4d8a-a6b4-37e69fbf54eb", "ResNet18-seed42"),
    ("a0bcbcba-94a9-4009-bcb6-d725b47588ee", "ResNet18-seed137"),
    ("935c13ae-a026-4eae-a2b2-0306879b2e8c", "ResNet18-seed256"),
    ("10d31151-68a2-4491-836d-3ead8e00a7ad", "ViTSmall-seed42"),
    ("909d7f68-55e1-4cd9-8c7b-68d5793798d3", "ViTSmall-seed137"),
    ("abad8975-de55-4e7e-b95d-05067f0eed90", "ViTSmall-seed256"),
    ("4bebc329-c26a-4eb4-b18c-7586e513f49e", "CCT7-seed42"),
    ("edf8275e-c99f-4436-a13d-4ed50f072a66", "CCT7-seed137"),
    ("445dd4d4-770a-4f33-b82b-c0bc7baed9a7", "CCT7-seed256"),
]


def _loads_without_terminal(s, *a, **k):
    d = _json.loads(s, *a, **k)
    if isinstance(d, dict) and "metadata" in d and "checkpoint_labels" in d["metadata"]:
        d["metadata"]["checkpoint_labels"] = [x for x in d["metadata"]["checkpoint_labels"] if x != "terminal"]
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    T.OUT_DIR = OUT
    T.OUT_MD = OUT / "transformed_trace.md"
    T.json = types.SimpleNamespace(loads=_loads_without_terminal, dumps=_json.dumps)
    results = []
    sel = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    for lane_id, label in CORE_LANES:
        if sel and not any(lane_id.startswith(s) for s in sel):
            continue
        dst = OUT / f"{label}.json"
        if dst.exists():
            results.append(_json.loads(dst.read_text())); continue
        print(f"== {label}", flush=True)
        r = T.analyze_lane(label, CORE / lane_id / "sae_analysis")
        dst.write_text(_json.dumps(r, indent=2))
        results.append(r)
    T.write_report(results)
    print(T.OUT_MD.read_text())


if __name__ == "__main__":
    main()
