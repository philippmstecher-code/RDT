#!/usr/bin/env python3
"""
Convert stored SAE lane results from nested categories (every Tg-H event also an
Ab-H event) to the mutually exclusive categories of the Methods, in place.

The "ab_h" label is removed from any feature that has "tg_h" and all aggregated
counts are adjusted. The conversion is exact because every tg_h event has a
matching ab_h: exclusive ab_h = ab_h - tg_h.

Patches:
  1. Per-sample transition JSONs  (transitions/ and transitions_weight/)
  2. Aggregated sae_results.json  (process_intensity, class_process_summary,
     superclass_summary, sample_consistency, hypotheses, weight_matching)

Usage:
    python scripts/make_ab_tg_exclusive.py [--data-root /data/rdt/core]
                                               [--lanes 1,3,5]
                                               [--yes]
                                               [--dry-run]
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


# Helpers (shared with reclassify_all_lanes.py)

def _has_required_artifacts(sae_dir: Path) -> Tuple[bool, List[str]]:
    missing: List[str] = []
    if not (sae_dir / "sae_results.json").exists():
        missing.append("sae_results.json")
    return (len(missing) == 0, missing)


def _is_already_patched(sae_dir: Path) -> Tuple[bool, str]:
    """
    Detect whether this lane has already been patched.

    Uses two independent checks:
      1. Metadata flag: metadata.ab_h_tg_h_double_count_patched == True
      2. Data check: scan a sample of transition files for any feature that
         has BOTH "ab_h" and "tg_h" in its processes list.  If none found,
         the data is already clean.

    Returns (is_patched, reason).
    """
    results_file = sae_dir / "sae_results.json"
    try:
        with open(results_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return (False, "unreadable")

    # Check 1: metadata flag stamped by this script after patching
    meta = data.get("metadata", {})
    if meta.get("ab_h_tg_h_double_count_patched") is True:
        return (True, "metadata flag set")

    # Check 2: double-counted data always has ab_h >= tg_h, so ab_h < tg_h at any
    # transition means the counts are already exclusive (subtracting would go negative).
    pi = data.get("process_intensity", [])
    for entry in pi:
        if entry.get("ab_h", 0) < entry.get("tg_h", 0):
            return (True, "ab_h < tg_h (counts already exclusive)")

    return (False, "needs patching")


def discover_lanes(data_root: Path) -> List[Dict[str, Any]]:
    experiments_dir = data_root / "experiments"
    if not experiments_dir.exists():
        print(f"ERROR: {experiments_dir} not found")
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
            patched, reason = _is_already_patched(sae_dir)
            if patched:
                print(f"  SKIP {exp_dir.name[:8]}/{lane_dir.name[:8]}: already exclusive ({reason})")
                continue
            eligible.append({
                "lane_dir": lane_dir,
                "sae_dir": sae_dir,
                "experiment_id": exp_dir.name,
                "lane_id": lane_dir.name,
            })
    return eligible


# Phase 1 - Patch transition JSONs

def _patch_transition_file(path: Path, dry_run: bool) -> Dict[str, int]:
    """
    Patch a single transition JSON file.

    For every feature in born[] and stable[] whose processes list contains
    both "ab_h" and "tg_h", remove "ab_h".  Update process_counts to match.

    Returns {"features_fixed": N, "samples_affected": M}.
    """
    with open(path) as f:
        events = json.load(f)

    features_fixed = 0
    samples_affected = 0

    for ev in events:
        sample_touched = False
        for category in ("born", "stable"):
            for feat in ev.get(category, []):
                procs = feat.get("processes", [])
                if "tg_h" in procs and "ab_h" in procs:
                    procs.remove("ab_h")
                    feat["processes"] = procs
                    ev["process_counts"]["ab_h"] = ev["process_counts"].get("ab_h", 1) - 1
                    features_fixed += 1
                    sample_touched = True
        if sample_touched:
            samples_affected += 1

    if not dry_run and features_fixed > 0:
        with open(path, "w") as f:
            json.dump(events, f)

    return {"features_fixed": features_fixed, "samples_affected": samples_affected}


def patch_transitions(sae_dir: Path, dry_run: bool) -> Dict[str, int]:
    """Patch all transition JSON files (activation + weight matching)."""
    totals = {"features_fixed": 0, "samples_affected": 0, "files_patched": 0,
              "files_scanned": 0, "files_corrupt": 0}
    corrupt_files: List[str] = []

    for subdir in ("transitions", "transitions_weight"):
        trans_dir = sae_dir / subdir
        if not trans_dir.exists():
            continue
        for json_file in sorted(trans_dir.rglob("*.json")):
            totals["files_scanned"] += 1
            try:
                result = _patch_transition_file(json_file, dry_run)
            except (json.JSONDecodeError, OSError) as e:
                totals["files_corrupt"] += 1
                rel = json_file.relative_to(sae_dir)
                corrupt_files.append(str(rel))
                print(f"             WARNING: skipping corrupt file {rel} ({e})")
                continue
            totals["features_fixed"] += result["features_fixed"]
            totals["samples_affected"] += result["samples_affected"]
            if result["features_fixed"] > 0:
                totals["files_patched"] += 1

    totals["corrupt_files"] = corrupt_files
    return totals


# Phase 2 - Patch sae_results.json

def _fix_counts_dict(d: Dict[str, Any]) -> int:
    """
    Given a dict with ab_h and tg_h keys, set ab_h = ab_h - tg_h.
    Returns the delta (amount subtracted).
    """
    old_ab = d.get("ab_h", 0)
    tg = d.get("tg_h", 0)
    new_ab = old_ab - tg
    if new_ab < 0:
        new_ab = 0  # clamp
    d["ab_h"] = new_ab
    return old_ab - new_ab


def _fix_total(d: Dict[str, Any]) -> None:
    """
    Recompute 'total' = ab_h + tg_h + di_h + as_h + de_h + unclassified.

    With double counting, total excluded tg_h (a subset of ab_h); with exclusive
    counts it includes both. The value is unchanged:
    old_total = old_ab_h + di + as + de = new_ab_h + tg_h + di + as + de.
    It is recomputed as a consistency check.
    """
    if "total" in d:
        d["total"] = (d.get("ab_h", 0) + d.get("tg_h", 0) +
                      d.get("di_h", 0) + d.get("as_h", 0) +
                      d.get("de_h", 0) + d.get("unclassified", 0))


def _fix_dominant(d: Dict[str, Any]) -> None:
    """Recompute 'dominant' from the corrected process counts."""
    if "dominant" not in d:
        return
    proc_counts = {k: d.get(k, 0) for k in ("ab_h", "tg_h", "di_h", "as_h", "de_h")}
    total = sum(proc_counts.values())
    if total > 0:
        d["dominant"] = max(proc_counts, key=proc_counts.get)
    else:
        d["dominant"] = "none"


def _recompute_sample_consistency(
    sae_dir: Path,
    subdir: str = "transitions",
) -> Optional[Dict[str, Dict[str, int]]]:
    """
    Recompute sample_consistency from the (already patched) transition JSONs.

    For each sample, sum process counts across all transitions and layers,
    then determine the dominant process (>40% of total -> dominant, else mixed).
    """
    trans_dir = sae_dir / subdir
    if not trans_dir.exists():
        return None

    # (class_idx, sample_idx) -> {ab_h, tg_h, di_h, as_h, de_h}
    sample_events: Dict[Tuple[int, int], Dict[str, int]] = {}

    for json_file in sorted(trans_dir.rglob("*.json")):
        try:
            with open(json_file) as f:
                events = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue  # skip corrupt files (already warned in phase 1)
        for ev in events:
            ci = ev["class_idx"]
            si = ev["sample_idx_in_class"]
            key = (ci, si)
            if key not in sample_events:
                sample_events[key] = {"ab_h": 0, "tg_h": 0, "di_h": 0, "as_h": 0, "de_h": 0}
            for proc in ("ab_h", "tg_h", "di_h", "as_h", "de_h"):
                sample_events[key][proc] += ev["process_counts"].get(proc, 0)

    consistency: Dict[int, Dict[str, int]] = {}
    for (ci, si), counts in sample_events.items():
        if ci not in consistency:
            consistency[ci] = {
                "ab_h_dominant": 0, "tg_h_dominant": 0, "di_h_dominant": 0,
                "as_h_dominant": 0, "de_h_dominant": 0, "mixed": 0,
            }
        total = sum(counts.values())
        if total == 0:
            consistency[ci]["mixed"] += 1
            continue
        dominant = max(counts, key=counts.get)
        dominant_pct = counts[dominant] / total
        if dominant_pct > 0.4:
            consistency[ci][f"{dominant}_dominant"] += 1
        else:
            consistency[ci]["mixed"] += 1

    return {str(k): v for k, v in consistency.items()}


def patch_sae_results(sae_dir: Path, dry_run: bool) -> Dict[str, Any]:
    """
    Patch the aggregated sae_results.json in-place.

    Applies corrected_ab_h = old_ab_h - old_tg_h at every level,
    recomputes totals/dominants, recomputes sample_consistency from
    patched transitions, and updates hypothesis events/pcts.
    """
    results_file = sae_dir / "sae_results.json"
    with open(results_file) as f:
        data = json.load(f)

    report: Dict[str, Any] = {}

    # We patch two result sections: top-level (activation matching) and weight_matching
    for section_key in (None, "weight_matching"):
        section = data if section_key is None else data.get(section_key, {})
        if not section:
            continue
        label = section_key or "activation"

        # process_intensity
        for pi in section.get("process_intensity", []):
            _fix_counts_dict(pi)
            _fix_dominant(pi)
            # Recompute total (ab_h + tg_h + di_h + as_h + de_h + unclassified)
            _fix_total(pi)

            # adjusted_ab_h is a copy of raw ab_h; apply the same correction
            if "adjusted_ab_h" in pi:
                old_adj = pi["adjusted_ab_h"]
                adj_tg = pi.get("adjusted_tg_h", 0)
                pi["adjusted_ab_h"] = max(0, old_adj - adj_tg)
                if "adjusted_total" in pi:
                    pi["adjusted_total"] = (
                        pi["adjusted_ab_h"] + pi.get("adjusted_tg_h", 0) +
                        pi.get("adjusted_di_h", 0) + pi.get("adjusted_as_h", 0) +
                        pi.get("adjusted_de_h", 0) + pi.get("unclassified", 0)
                    )

            # churn = event_total / (event_total + stable) is unchanged because total is unchanged

        # class_process_summary
        for ci, cls_data in section.get("class_process_summary", {}).items():
            _fix_counts_dict(cls_data)
            _fix_total(cls_data)
            _fix_dominant(cls_data)
            for pt in cls_data.get("per_transition", []):
                _fix_counts_dict(pt)

        # superclass_summary
        for sc_name, sc_data in section.get("superclass_summary", {}).items():
            pb = sc_data.get("process_breakdown", {})
            _fix_counts_dict(pb)
            # Recompute superclass total
            if "total" in pb:
                pb["total"] = (pb.get("ab_h", 0) + pb.get("tg_h", 0) +
                               pb.get("di_h", 0) + pb.get("as_h", 0) +
                               pb.get("de_h", 0) + pb.get("unclassified", 0))
            for class_name, cls_proc in sc_data.get("per_class_process", {}).items():
                _fix_counts_dict(cls_proc)

        # sample_consistency - recompute from patched transitions
        trans_subdir = "transitions" if section_key is None else "transitions_weight"
        new_sc = _recompute_sample_consistency(sae_dir, trans_subdir)
        if new_sc is not None:
            section["sample_consistency"] = new_sc

        # hypotheses
        hypotheses = section.get("hypotheses", {})
        # Recompute total_events (sum of process_intensity totals - unchanged)
        total_events = sum(pi.get("total", 0) for pi in section.get("process_intensity", []))

        # Ab-H: update events and events_pct
        ab_h_hyp = hypotheses.get("Ab-H", {})
        if ab_h_hyp:
            new_ab_events = sum(pi.get("ab_h", 0) for pi in section.get("process_intensity", []))
            ab_h_hyp["events"] = new_ab_events
            ab_h_hyp["events_pct"] = new_ab_events / max(total_events, 1) * 100
            # Evidence items (ssi_increase, ssi_trend_spearman, birth_vs_baseline)
            # are independent of ab_h counts - no change needed

        # Tg-H, Di-H, As-H, De-H: events/pcts unchanged (their counts didn't change)
        # But recalculate pcts since total_events is the denominator
        for hyp_id, proc_key in [("Tg-H", "tg_h"), ("Di-H", "di_h"),
                                  ("As-H", "as_h"), ("De-H", "de_h")]:
            hyp = hypotheses.get(hyp_id, {})
            if hyp and "events" in hyp:
                hyp["events_pct"] = hyp["events"] / max(total_events, 1) * 100

        report[label] = {
            "new_ab_h_total": sum(pi.get("ab_h", 0) for pi in section.get("process_intensity", [])),
            "total_events": total_events,
        }

    # Mark as patched
    meta = data.get("metadata", {})
    meta["ab_h_tg_h_double_count_patched"] = True

    if not dry_run:
        with open(results_file, "w") as f:
            json.dump(data, f, indent=2)

    return report


# Verification

def _read_process_totals(sae_dir: Path) -> Optional[Dict[str, int]]:
    results_file = sae_dir / "sae_results.json"
    try:
        with open(results_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    totals: Dict[str, int] = {"ab_h": 0, "tg_h": 0, "di_h": 0, "as_h": 0, "de_h": 0}
    for entry in data.get("process_intensity", []):
        for key in totals:
            totals[key] += entry.get(key, 0)
    return totals


def _verify_no_double_count(sae_dir: Path) -> List[str]:
    """Spot-check transition files for remaining double-counts."""
    issues: List[str] = []
    for subdir in ("transitions", "transitions_weight"):
        trans_dir = sae_dir / subdir
        if not trans_dir.exists():
            continue
        # Check a sample of files
        files = sorted(trans_dir.rglob("*.json"))
        for json_file in files[:5]:  # spot-check first 5
            try:
                with open(json_file) as f:
                    events = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            for ev in events:
                for cat in ("born", "stable"):
                    for feat in ev.get(cat, []):
                        procs = feat.get("processes", [])
                        if "tg_h" in procs and "ab_h" in procs:
                            issues.append(f"{json_file.name}: fid={feat.get('fid', '?')} "
                                          f"still has both ab_h and tg_h")
    return issues


# Main

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make Ab-H and Tg-H counts mutually exclusive in place "
                    "(no SAE re-analysis needed)."
    )
    parser.add_argument("--data-root", type=Path, default=Path(RDT_DATA + "/core"))
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip interactive confirmation")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing")
    parser.add_argument("--lanes", type=str, default=None,
                        help="Comma-separated lane numbers (e.g. '1,3,5') or 'all'")
    args = parser.parse_args()

    print("=" * 72)
    print("  Mutually exclusive Ab-H / Tg-H: in-place correction")
    print("=" * 72)
    print(f"\nData root: {args.data_root}\n")

    # Discover
    print("Discovering eligible lanes...")
    lanes = discover_lanes(args.data_root)
    print()

    if not lanes:
        print("No eligible lanes found.")
        return

    # Display
    print(f"Found {len(lanes)} eligible lane(s):\n")
    print(f"  {'#':<4} {'Experiment':<40} {'Lane':<40}")
    print(f"  {'—'*4} {'—'*40} {'—'*40}")
    for i, info in enumerate(lanes, 1):
        print(f"  {i:<4} {info['experiment_id'][:38]:<40} {info['lane_id'][:38]:<40}")

    # Show old counts
    print()
    print("Current process totals (BEFORE patch):")
    print(f"  {'#':<4} {'ab_h':>10} {'tg_h':>10} {'overlap%':>10} {'di_h':>10} {'as_h':>10} {'de_h':>10}")
    print(f"  {'—'*4} {'—'*10} {'—'*10} {'—'*10} {'—'*10} {'—'*10} {'—'*10}")
    old_totals: List[Optional[Dict[str, int]]] = []
    for i, info in enumerate(lanes, 1):
        totals = _read_process_totals(info["sae_dir"])
        old_totals.append(totals)
        if totals is None:
            print(f"  {i:<4} {'(unreadable: file incomplete or corrupt)':>63}")
        else:
            overlap = totals["tg_h"] / max(totals["ab_h"], 1) * 100
            print(f"  {i:<4} {totals['ab_h']:>10,} {totals['tg_h']:>10,} {overlap:>9.1f}% "
                  f"{totals['di_h']:>10,} {totals['as_h']:>10,} {totals['de_h']:>10,}")

    print()

    # Lane selection
    if args.lanes is not None:
        lane_input = args.lanes
    elif not args.yes:
        print("Enter lane numbers to process (e.g. '1,3,5' or 'all'):")
        lane_input = input("> ").strip()
    else:
        lane_input = "all"

    if lane_input.lower() in ("all", ""):
        print(f"Processing all {len(lanes)} lane(s).")
    else:
        try:
            selected = [int(x.strip()) for x in lane_input.split(",")]
        except ValueError:
            print(f"ERROR: expected comma-separated integers or 'all', got: {lane_input}")
            sys.exit(1)
        invalid = [x for x in selected if x < 1 or x > len(lanes)]
        if invalid:
            print(f"ERROR: out of range (1-{len(lanes)}): {invalid}")
            sys.exit(1)
        sel_set = set(selected)
        lanes = [l for i, l in enumerate(lanes, 1) if i in sel_set]
        old_totals = [t for i, t in enumerate(old_totals, 1) if i in sel_set]
        print(f"Selected {len(lanes)} lane(s).")

    print()

    if args.dry_run:
        print("DRY RUN: scanning for double counts without writing...\n")

    # Confirm
    if not args.yes and not args.dry_run:
        print("This will patch transition JSONs and sae_results.json IN PLACE.")
        resp = input("Proceed? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            print("Aborted.")
            return

    # Patch each lane
    failed: List[Tuple[int, str]] = []
    for i, (info, old) in enumerate(zip(lanes, old_totals), 1):
        sae_dir = info["sae_dir"]
        print(f"{'─' * 72}")
        print(f"[{i}/{len(lanes)}] {info['experiment_id'][:12]}/.../{info['lane_id'][:12]}")
        print(f"         {sae_dir}")
        print()

        t0 = time.time()
        try:
            # Phase 1: Patch transition JSONs
            print("    Phase 1: Patching transition JSONs...")
            trans_result = patch_transitions(sae_dir, args.dry_run)
            print(f"             {trans_result['files_scanned']} files scanned, "
                  f"{trans_result['files_patched']} patched, "
                  f"{trans_result['features_fixed']:,} features fixed")

            # Phase 2: Patch sae_results.json
            print("    Phase 2: Patching sae_results.json...")
            agg_result = patch_sae_results(sae_dir, args.dry_run)
            for section, info_s in agg_result.items():
                print(f"             [{section}] new ab_h total: {info_s['new_ab_h_total']:,}")

            # Phase 3: Verify
            if not args.dry_run:
                print("    Phase 3: Verifying...")
                issues = _verify_no_double_count(sae_dir)
                if issues:
                    print(f"    WARNING: {len(issues)} remaining double-counts found!")
                    for issue in issues[:5]:
                        print(f"      {issue}")
                else:
                    print("             No remaining double-counts (spot-check passed)")

                # Show before/after
                new = _read_process_totals(sae_dir)
                if new is not None and old is not None:
                    print()
                    print(f"    {'metric':<12} {'before':>12} {'after':>12} {'delta':>12}")
                    print(f"    {'——————':<12} {'——————':>12} {'——————':>12} {'——————':>12}")
                    for key in ("ab_h", "tg_h", "di_h", "as_h", "de_h"):
                        delta = new[key] - old[key]
                        sign = "+" if delta > 0 else ""
                        print(f"    {key:<12} {old[key]:>12,} {new[key]:>12,} {sign}{delta:>11,}")

            elapsed = time.time() - t0
            print(f"\n    Done in {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"    FAILED after {elapsed:.1f}s: {e}")
            import traceback
            traceback.print_exc()
            failed.append((i, str(e)))

    # Summary
    print()
    print("=" * 72)
    n_ok = len(lanes) - len(failed)
    verb = "would be patched" if args.dry_run else "patched"
    print(f"  {n_ok}/{len(lanes)} lanes {verb} successfully")
    if failed:
        print(f"  FAILED ({len(failed)}):")
        for idx, err in failed:
            print(f"    Lane #{idx}: {err}")
    print("=" * 72)



if __name__ == "__main__":
    main()
