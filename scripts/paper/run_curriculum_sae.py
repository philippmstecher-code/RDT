#!/usr/bin/env python3
"""SAE / RDT analysis of the finished curriculum lanes (sc_true, sc_rand, scratch).

Uses the published pipeline (saeanalysis.analyze_sae_features via
run_noise_curriculum_experiment.run_sae) with the template sae_policy, exactly as for the sweep.
Lanes are processed seed by seed (42, 137, 7, 256, 314) so partial results are usable; a lane is
skipped if it has no .lane_done (training unfinished) or already carries .sae_done.

Usage: python -u scripts/paper/run_curriculum_sae.py --template-json <template.json>
       [--experiment-id curriculum] [--arms "sc_true_K20_seed{},..."]
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

WT = Path(RDT_DATA + "/checkouts/lesion")
spec = importlib.util.spec_from_file_location("ncr", WT / "scripts" / "run_noise_curriculum_experiment.py")
ncr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ncr)

SEEDS = [42, 137, 7, 256, 314]
ARMS = ["sc_true_K20_seed{}", "sc_rand_K20_seed{}", "scratch_E30_seed{}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-id", default="curriculum")
    ap.add_argument("--template-json", required=True)
    ap.add_argument("--arms", default=",".join(ARMS), help="lane-name patterns, comma-separated ({} = seed)")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    policy = json.loads(Path(args.template_json).read_text()).get("sae_policy", {})
    lanes_dir = ncr.DATA_DIR / "experiments" / args.experiment_id / "lanes"
    classes = ncr.canonical_classes(args.experiment_id)
    for s in SEEDS:
        for a in arms:
            lane = lanes_dir / a.format(s)
            if not (lane / ".lane_done").exists():
                print(f"[{lane.name}] SAE_SKIP training not finished", flush=True)
                continue
            if (lane / ".sae_done").exists():
                print(f"[{lane.name}] SAE_SKIP already done", flush=True)
                continue
            t0 = time.time()
            print(f"[{lane.name}] SAE_START", flush=True)
            ncr.run_sae(lane, classes, policy)
            (lane / ".sae_done").touch()
            print(f"[{lane.name}] SAE_DONE mins={(time.time() - t0) / 60:.1f}", flush=True)


if __name__ == "__main__":
    main()
