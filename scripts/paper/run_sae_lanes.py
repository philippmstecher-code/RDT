#!/usr/bin/env python3
"""SAE / RDT analysis for the disruption experiments: static label noise (en: static_noise_s{seed}, clean /
between_sc / within_sc; matched: the similarity-matched arm) and suppression from birth (sup: coarse code / random).

Same SAE pipeline and template sae_policy as run_curriculum_sae.py. Lanes are processed seed by seed; a lane is
skipped if it already carries .sae_done. The seed-42 suppression lanes are read from the core experiments directory:
their snapshots (activations, predictions and metrics) are copied to prevention_s42_copy in the extended experiments
directory and analysed there.

Usage: python -u scripts/paper/run_sae_lanes.py --template-json <template.json> [--only en|sup|matched]
"""
import argparse
import importlib.util
import json
import shutil
import time
from pathlib import Path
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

WT = Path(RDT_DATA + "/checkouts/lesion")
spec = importlib.util.spec_from_file_location("ncr", WT / "scripts" / "run_noise_curriculum_experiment.py")
ncr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ncr)

RUNS_EXTENDED = Path(RDT_DATA + "/extended/experiments")
RUNS_CORE = Path(RDT_DATA + "/core/experiments")
SEEDS = [42, 137, 7, 256, 314]
EN_ARMS = ["standard_resnet18", "between_sc_resnet18_throughout", "within_sc_resnet18_throughout"]
SUP_ARMS = ["lesion_scaffold_e00_seed{}_rp65_Llayer4-avgpool", "lesion_random_variance_e00_seed{}_rp65_Llayer4-avgpool"]


def extended_copy_of(src: Path, dst: Path) -> Path:
    """Copy the snapshots (activations, predictions, metrics; no model weights) of lane src to dst."""
    if (dst / ".copied").exists():
        return dst
    for p in (src / "dev_snapshots").rglob("*"):
        if p.is_dir() or p.name.startswith("._") or p.name in ("checkpoint.pt", "training_state.pt"):
            continue
        q = dst / p.relative_to(src)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
    (dst / ".copied").touch()
    return dst


def lanes(only):
    out = []
    for s in SEEDS:
        if only in (None, "en"):
            out += [RUNS_EXTENDED / f"static_noise_s{s}" / "lanes" / a for a in EN_ARMS]
        if only == "matched":   # similarity-matched arm
            out.append(RUNS_EXTENDED / f"static_noise_s{s}" / "lanes" / "between_sc_matched_resnet18_throughout")
        if only in (None, "sup"):
            for a in SUP_ARMS:
                name = a.format(s)
                if s == 42:
                    out.append(extended_copy_of(RUNS_CORE / "scaffold_lesion" / "lanes" / name,
                                           RUNS_EXTENDED / "prevention_s42_copy" / "lanes" / name))
                else:
                    out.append(RUNS_EXTENDED / "prevention" / "lanes" / name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template-json", required=True)
    ap.add_argument("--only", choices=["en", "sup", "matched"], default=None)
    args = ap.parse_args()
    policy = json.loads(Path(args.template_json).read_text()).get("sae_policy", {})
    classes = ncr.canonical_classes("curriculum")  # canonical CIFAR-100 class order, identical for all runs
    for lane in lanes(args.only):
        if not (lane / "dev_snapshots" / "metrics" / "devtrain_snapshots.json").exists():
            print(f"[{lane.name}] SAE_SKIP no snapshots", flush=True)
            continue
        if (lane / ".sae_done").exists():
            print(f"[{lane.name}] SAE_SKIP already done", flush=True)
            continue
        t0 = time.time()
        print(f"[{lane.parent.parent.name}/{lane.name}] SAE_START", flush=True)
        ncr.run_sae(lane, classes, policy)
        (lane / ".sae_done").touch()
        print(f"[{lane.parent.parent.name}/{lane.name}] SAE_DONE mins={(time.time() - t0) / 60:.1f}", flush=True)


if __name__ == "__main__":
    main()
