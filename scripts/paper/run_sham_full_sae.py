#!/usr/bin/env python3
"""SAE / RDT analysis of the sham runs on the full suppression checkpoint grid, as the comparator for the
scaffold-shaping analyses of the epochs 1-10 last-block suppression.

Sham = critical_sham/sham_e00_seed{S} (rank-0 projector branched at epoch 0), checkpoints identical to the
suppression arms: install (t0), epochs 1-15, 20, 25, 30 and the duplicated final checkpoint (milestones 0-19).
For each seed, all milestones are copied (activations and predictions; no weights) into
/data/rdt/extended/experiments/sham_fullsae/lanes/sham_e00_seed{S} with the unfiltered
devtrain_snapshots.json, and the unchanged pipeline is run there with the template sae_policy, exactly as
run_sham_perepoch_sae.py and the suppression arms (expansion 4, k 32, 100 images per class, 1,000 permutations).
The original lanes are not modified.
Usage: PYTHONHASHSEED=0 DEVTRAIN_MPS_MAX_BATCH=128 python -u scripts/paper/run_sham_full_sae.py SEED [SEED ...]
"""
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

WT = Path(RDT_DATA + "/checkouts/lesion")
spec = importlib.util.spec_from_file_location("ncr", WT / "scripts" / "run_noise_curriculum_experiment.py")
ncr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ncr)

SRC = Path(RDT_DATA + "/extended/experiments/critical_sham/lanes")
DST = Path(RDT_DATA + "/extended/experiments/sham_fullsae/lanes")
TPL = Path(RDT_DATA + "/core/experiments/noise_curriculum_arch/_config/template.json")


def make_lane(s):
    src, dst = SRC / f"sham_e00_seed{s}", DST / f"sham_e00_seed{s}"
    snaps = json.load(open(src / "dev_snapshots" / "metrics" / "devtrain_snapshots.json"))
    assert [x["milestone_index"] for x in snaps] == list(range(len(snaps))), [x["milestone_index"] for x in snaps]
    for x in snaps:
        m = f"milestone_{x['milestone_index']}"
        (dst / "dev_snapshots" / m).mkdir(parents=True, exist_ok=True)
        for f in ("individual_multilayer_activations.pt", "sample_predictions.pt"):
            if not (dst / "dev_snapshots" / m / f).exists():
                shutil.copy2(src / "dev_snapshots" / m / f, dst / "dev_snapshots" / m / f)
    (dst / "dev_snapshots" / "metrics").mkdir(parents=True, exist_ok=True)
    json.dump(snaps, open(dst / "dev_snapshots" / "metrics" / "devtrain_snapshots.json", "w"), indent=1)
    (dst / "SOURCE.txt").write_text(f"copy of {src} milestones 0..{len(snaps) - 1} (all checkpoints); "
                                    "activations and predictions only\n")
    return dst


def main():
    seeds = [int(s) for s in sys.argv[1:]]
    policy = json.loads(TPL.read_text()).get("sae_policy", {})
    classes = ncr.canonical_classes("curriculum")
    for s in seeds:
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
