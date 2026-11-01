#!/usr/bin/env python3
"""SAE / RDT analysis of clean per-epoch runs (critical_sham/sham_e00_seed{S}: standard training with a rank-0
projector, checkpoint every epoch), epochs 0-15; seeds 7, 42, 137 by default.

For each seed, milestones 0..15 (epochs 0..15) are copied (activations, predictions; no weights) into
/data/rdt/extended/experiments/sham_perepoch/lanes/sham_e00_seed{S} with a filtered
devtrain_snapshots.json, and the unchanged pipeline is run there with the template sae_policy, exactly as
run_sae_lanes.py (run_noise_curriculum_experiment.run_sae; PYTHONHASHSEED=0 and DEVTRAIN_MPS_MAX_BATCH=128 set by the
caller). The original lanes are not modified.
Usage: PYTHONHASHSEED=0 DEVTRAIN_MPS_MAX_BATCH=128 python -u scripts/paper/run_sham_perepoch_sae.py [SEEDS]
       (SEEDS comma-separated, e.g. 256,314)
"""
import importlib.util
import json
import sys
import shutil
import time
from pathlib import Path
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

WT = Path(RDT_DATA + "/checkouts/lesion")
spec = importlib.util.spec_from_file_location("ncr", WT / "scripts" / "run_noise_curriculum_experiment.py")
ncr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ncr)

SRC = Path(RDT_DATA + "/extended/experiments/critical_sham/lanes")
DST = Path(RDT_DATA + "/extended/experiments/sham_perepoch/lanes")
TPL = Path(RDT_DATA + "/core/experiments/noise_curriculum_arch/_config/template.json")
SEEDS = [7, 42, 137]
MAX_EPOCH = 15


def make_lane(s):
    src, dst = SRC / f"sham_e00_seed{s}", DST / f"sham_e00_seed{s}"
    snaps = json.load(open(src / "dev_snapshots" / "metrics" / "devtrain_snapshots.json"))
    keep = [x for x in snaps if x.get("snapshot_type") != "final" and x["weight_updates"] <= MAX_EPOCH * 391]
    assert [x["milestone_index"] for x in keep] == list(range(MAX_EPOCH + 1)), [x["milestone_index"] for x in keep]
    for x in keep:
        m = f"milestone_{x['milestone_index']}"
        (dst / "dev_snapshots" / m).mkdir(parents=True, exist_ok=True)
        for f in ("individual_multilayer_activations.pt", "sample_predictions.pt"):
            if not (dst / "dev_snapshots" / m / f).exists():
                shutil.copy2(src / "dev_snapshots" / m / f, dst / "dev_snapshots" / m / f)
    (dst / "dev_snapshots" / "metrics").mkdir(parents=True, exist_ok=True)
    json.dump(keep, open(dst / "dev_snapshots" / "metrics" / "devtrain_snapshots.json", "w"), indent=1)
    (dst / "SOURCE.txt").write_text(f"copy of {src} milestones 0..{MAX_EPOCH} (epochs 0..{MAX_EPOCH}); "
                                    "activations and predictions only\n")
    return dst


def main():
    if len(sys.argv) > 1:   # optional comma-separated seeds
        SEEDS[:] = [int(x) for x in sys.argv[1].split(",")]
    policy = json.loads(TPL.read_text()).get("sae_policy", {})
    classes = ncr.canonical_classes("curriculum")
    for s in SEEDS:
        lane = make_lane(s)
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
