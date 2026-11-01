#!/usr/bin/env python3
"""
Reclassify completed SAE analysis lanes with mutually exclusive Ab-H and Tg-H.

saeanalysis.py counts a feature that meets the Tg-H (task-general) threshold in
tg_h only, never also in ab_h. This script re-runs `reclassify_from_saved()` on
every lane that has the required artifacts, so that stored results follow the
same rule, and regenerates sae_results.json and the per-sample transition JSONs.

Usage:
    python scripts/reclassify_all_lanes.py [--data-root /data/rdt/core]
                                           [--yes]
                                           [--dry-run]

Flags:
    --data-root   Root directory containing experiments/  (default: /data/rdt/core)
    --yes         Skip interactive confirmation
    --dry-run     Discover and list eligible lanes without running anything
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

# Ensure the ML package is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
ML_SRC = REPO_ROOT / "packages" / "ml" / "src"
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))


# Helpers

def _has_required_artifacts(sae_dir: Path) -> Tuple[bool, List[str]]:
    """Check that a lane's sae_analysis/ dir has everything reclassify needs."""
    missing: List[str] = []

    results_file = sae_dir / "sae_results.json"
    if not results_file.exists():
        missing.append("sae_results.json")

    act_dir = sae_dir / "activations"
    if not act_dir.exists() or not any(act_dir.rglob("*.npy")):
        missing.append("activations/*.npy")

    sae_wt_dir = sae_dir / "saes"
    if not sae_wt_dir.exists() or not any(sae_wt_dir.rglob("*.pt")):
        missing.append("saes/*.pt")

    # dev_snapshots with multilayer activations (needed by _phase1_load)
    lane_dir = sae_dir.parent
    snap_dir = lane_dir / "dev_snapshots"
    if not snap_dir.exists():
        missing.append("dev_snapshots/")
    else:
        has_any_snapshot = any(snap_dir.rglob("individual_multilayer_activations.pt"))
        if not has_any_snapshot:
            missing.append("dev_snapshots/*/individual_multilayer_activations.pt")

    return (len(missing) == 0, missing)


def _load_metadata(sae_dir: Path) -> Optional[Dict[str, Any]]:
    """Load the metadata block from an existing sae_results.json."""
    results_file = sae_dir / "sae_results.json"
    try:
        with open(results_file) as f:
            data = json.load(f)
        return data.get("metadata", {})
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARNING: could not read {results_file}: {e}")
        return None


def discover_lanes(data_root: Path) -> List[Dict[str, Any]]:
    """Walk data_root/experiments/ and find all lanes with completed SAE analysis."""
    experiments_dir = data_root / "experiments"
    if not experiments_dir.exists():
        print(f"ERROR: experiments directory not found at {experiments_dir}")
        sys.exit(1)

    eligible: List[Dict[str, Any]] = []

    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        lanes_dir = exp_dir / "lanes"
        if not lanes_dir.exists():
            continue

        for lane_dir in sorted(lanes_dir.iterdir()):
            if not lane_dir.is_dir():
                continue
            sae_dir = lane_dir / "sae_analysis"
            if not sae_dir.exists():
                continue

            ok, missing = _has_required_artifacts(sae_dir)
            if not ok:
                print(f"  SKIP {exp_dir.name[:8]}/{lane_dir.name[:8]}: missing {', '.join(missing)}")
                continue

            meta = _load_metadata(sae_dir)
            if meta is None:
                print(f"  SKIP {exp_dir.name[:8]}/{lane_dir.name[:8]}: unreadable metadata")
                continue

            eligible.append({
                "lane_dir": lane_dir,
                "sae_dir": sae_dir,
                "experiment_id": exp_dir.name,
                "lane_id": lane_dir.name,
                "metadata": meta,
            })

    return eligible


# Reconstruct parameters from embedded metadata

def _build_params(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the arguments for reclassify_from_saved() from the metadata
    that is already embedded in sae_results.json.
    """
    selected_classes: List[str] = meta["selected_classes"]
    expansion_factor: int = meta.get("expansion_factor", 4)
    k_sparse: int = meta.get("k_sparse", 32)
    analysis_thresholds: Optional[Dict[str, float]] = meta.get("analysis_thresholds")

    # Infer dataset_id from class name prefix (e.g. "cifar100_apple" -> "cifar100")
    dataset_id = "cifar100"  # default
    if selected_classes:
        first = selected_classes[0]
        if first.startswith("cifar100_"):
            dataset_id = "cifar100"
        elif first.startswith("cifar10_"):
            dataset_id = "cifar10"
        elif first.startswith("imagenet_"):
            dataset_id = "imagenet"

    # Reconstruct a minimal experiment_config with the keys that
    # reclassify_from_saved actually reads:
    #   - experiment_config.get("sae_policy", {})  -> expansion_factor, k_sparse, null_permutations
    #   - experiment_config.get("dataset_id", "cifar100")
    experiment_config: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "sae_policy": {
            "expansion_factor": expansion_factor,
            "k_sparse": k_sparse,
            "null_permutations": 100,
        },
    }

    return {
        "selected_classes": selected_classes,
        "experiment_config": experiment_config,
        "expansion_factor": expansion_factor,
        "k_sparse": k_sparse,
        "analysis_thresholds": analysis_thresholds,
    }


# Verification: process counts before vs after

def _read_process_totals(sae_dir: Path) -> Dict[str, int]:
    """Sum ab_h and tg_h from process_intensity in sae_results.json."""
    results_file = sae_dir / "sae_results.json"
    with open(results_file) as f:
        data = json.load(f)
    totals: Dict[str, int] = {"ab_h": 0, "tg_h": 0, "di_h": 0, "as_h": 0, "de_h": 0}
    for entry in data.get("process_intensity", []):
        for key in totals:
            totals[key] += entry.get(key, 0)
    return totals


# Main

def _progress_callback(pct: float, msg: str) -> None:
    print(f"    [{pct:5.1f}%] {msg}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-run SAE process classification on all completed lanes "
                    "(Ab-H and Tg-H mutually exclusive)."
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path(RDT_DATA + "/core"),
        help="Root directory containing experiments/ (default: /data/rdt/core)",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip interactive confirmation prompt",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only discover and list eligible lanes; do not reclassify",
    )
    parser.add_argument(
        "--lanes", type=str, default=None,
        help="Comma-separated lane numbers to process (e.g. '1,3,5'). "
             "Use --dry-run first to see numbering. If omitted, all lanes are processed.",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("  Mutually exclusive Ab-H / Tg-H: batch reclassification")
    print("=" * 72)
    print()
    print(f"Data root: {args.data_root}")
    print()

    # Discover
    print("Discovering eligible lanes...")
    lanes = discover_lanes(args.data_root)
    print()

    if not lanes:
        print("No eligible lanes found. Nothing to do.")
        return

    # Display lanes
    print(f"Found {len(lanes)} eligible lane(s):\n")
    print(f"  {'#':<4} {'Experiment':<40} {'Lane':<40} {'Chkpts':>6}")
    print(f"  {'—'*4} {'—'*40} {'—'*40} {'—'*6}")
    for i, lane_info in enumerate(lanes, 1):
        n_ckpts = len(lane_info["metadata"].get("checkpoint_labels", []))
        print(f"  {i:<4} {lane_info['experiment_id'][:38]:<40} "
              f"{lane_info['lane_id'][:38]:<40} {n_ckpts:>6}")

    # Counts before reclassification
    print()
    print("Current process totals (BEFORE reclassification):")
    print(f"  {'#':<4} {'ab_h':>10} {'tg_h':>10} {'di_h':>10} {'as_h':>10} {'de_h':>10}")
    print(f"  {'—'*4} {'—'*10} {'—'*10} {'—'*10} {'—'*10} {'—'*10}")
    old_totals: List[Dict[str, int]] = []
    for i, lane_info in enumerate(lanes, 1):
        totals = _read_process_totals(lane_info["sae_dir"])
        old_totals.append(totals)
        print(f"  {i:<4} {totals['ab_h']:>10,} {totals['tg_h']:>10,} "
              f"{totals['di_h']:>10,} {totals['as_h']:>10,} {totals['de_h']:>10,}")

    print()

    if args.dry_run:
        print("DRY RUN: no changes made.")
        if args.lanes is None:
            print("Use --lanes 1,2,3 to select specific lanes (numbers from table above).")
        return

    # Select lanes
    if args.lanes is not None:
        lane_input = args.lanes
    elif not args.yes:
        print("Enter lane numbers to process (e.g. '1,3,5' or 'all'), then press Enter:")
        lane_input = input("> ").strip()
    else:
        lane_input = "all"

    if lane_input.lower() == "all" or lane_input == "":
        print(f"Processing all {len(lanes)} lane(s).")
    else:
        try:
            selected_indices = [int(x.strip()) for x in lane_input.split(",")]
        except ValueError:
            print(f"ERROR: expected comma-separated integers or 'all', got: {lane_input}")
            sys.exit(1)
        invalid = [x for x in selected_indices if x < 1 or x > len(lanes)]
        if invalid:
            print(f"ERROR: lane numbers out of range (1-{len(lanes)}): {invalid}")
            sys.exit(1)
        # Filter both lanes and old_totals to the selected subset
        selected_set = set(selected_indices)
        lanes = [lane for i, lane in enumerate(lanes, 1) if i in selected_set]
        old_totals = [t for i, t in enumerate(old_totals, 1) if i in selected_set]
        print(f"Selected {len(lanes)} lane(s): {', '.join(str(x) for x in selected_indices)}")

    print()

    # Confirm
    if not args.yes:
        print("This will re-run process classification (phases 4b–9) on the selected lane(s).")
        print("Existing sae_results.json and transition JSONs will be OVERWRITTEN.")
        resp = input("Proceed? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            print("Aborted.")
            return

    # Import reclassify (deferred so --dry-run / --help don't need torch)
    print()
    print("Loading saeanalysis module...")
    from saeanalysis import reclassify_from_saved

    # Run reclassification
    failed: List[Tuple[int, str]] = []
    for i, lane_info in enumerate(lanes, 1):
        lane_dir = lane_info["lane_dir"]
        meta = lane_info["metadata"]
        params = _build_params(meta)

        print()
        print(f"{'─' * 72}")
        print(f"[{i}/{len(lanes)}] Reclassifying: {lane_info['experiment_id'][:12]}/.../{lane_info['lane_id'][:12]}")
        print(f"         Lane: {lane_dir}")
        print(f"         Classes: {meta.get('n_classes', '?')}, "
              f"Checkpoints: {len(meta.get('checkpoint_labels', []))}, "
              f"Layers: {len(meta.get('layers', []))}")
        print()

        t0 = time.time()
        try:
            results = reclassify_from_saved(
                lane_dir=str(lane_dir),
                selected_classes=params["selected_classes"],
                experiment_config=params["experiment_config"],
                progress_callback=_progress_callback,
                expansion_factor=params["expansion_factor"],
                k_sparse=params["k_sparse"],
                analysis_thresholds=params["analysis_thresholds"],
            )
            elapsed = time.time() - t0

            if "error" in results:
                print(f"    ERROR from reclassify_from_saved: {results['error']}")
                failed.append((i, str(results["error"])))
                continue

            print(f"    Done in {elapsed:.1f}s")

            new_totals = _read_process_totals(lane_info["sae_dir"])
            old = old_totals[i - 1]
            print()
            print(f"    {'metric':<12} {'before':>12} {'after':>12} {'delta':>12}")
            print(f"    {'——————':<12} {'——————':>12} {'——————':>12} {'——————':>12}")
            for key in ("ab_h", "tg_h", "di_h", "as_h", "de_h"):
                delta = new_totals[key] - old[key]
                sign = "+" if delta > 0 else ""
                print(f"    {key:<12} {old[key]:>12,} {new_totals[key]:>12,} {sign}{delta:>11,}")

            # ab_h should decrease (Tg-H events are removed from it)
            if new_totals["ab_h"] >= old["ab_h"] and old["tg_h"] > 0:
                print()
                print("    WARNING: ab_h did not decrease; check that saeanalysis.py counts Tg-H features in tg_h only")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"    FAILED after {elapsed:.1f}s: {e}")
            import traceback
            traceback.print_exc()
            failed.append((i, str(e)))

    # Summary
    print()
    print("=" * 72)
    print(f"  Reclassification complete: {len(lanes) - len(failed)}/{len(lanes)} succeeded")
    if failed:
        print(f"  FAILED ({len(failed)}):")
        for idx, err in failed:
            print(f"    Lane #{idx}: {err}")
    print("=" * 72)



if __name__ == "__main__":
    main()
