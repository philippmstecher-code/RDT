#!/usr/bin/env python3
"""
SAE Expansion Factor Comparison Orchestrator.

Creates a new experiment with 3 lanes (different SAE expansion factors)
that share DEVTRAIN data from an existing ResNet-18 run via symlinks,
then runs SAEANALYSIS sequentially for each.

Usage:
    python packages/ml/src/run_sae_expansion_comparison.py
    python packages/ml/src/run_sae_expansion_comparison.py --expansion-factors 4 8 16
    python packages/ml/src/run_sae_expansion_comparison.py --source-experiment <uuid> --source-lane <uuid>
"""
import sys
import os
import gc
import json
import uuid
import shutil
import sqlite3
import argparse
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────
DEFAULT_SOURCE_EXPERIMENT = "ce93d3f6-282b-4811-84f8-be6c589c0500"
DEFAULT_SOURCE_LANE = "c645c39e-e21f-4cff-853f-967bf69ee857"
DEFAULT_EXPANSION_FACTORS = [4, 8, 16]

SCRIPT_DIR = Path(__file__).resolve().parent          # runpodScripts
PROJECT_ROOT = SCRIPT_DIR.parent                      # project root
ML_SRC_DIR = PROJECT_ROOT / "packages" / "ml" / "src"
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "rcx.db"
EXPERIMENTS_DIR = DATA_DIR / "experiments"

# Persistent workspace volume — larger storage that survives pod restarts.
# Lane data is archived here after completion to free space on the working volume.
WORKSPACE_VOLUME = Path(os.environ.get("RUNPOD_WORKSPACE_VOLUME", "/workspace"))

STOVE_TYPES = [
    ("NETINIT",       1),
    ("DEVTRAIN",      2),
    ("TERMREP",       3),
    ("SNAPANALYSIS",  4),
    ("SAEANALYSIS",   4),
    ("CAUSAL",        5),
]

# Directories to symlink from source lane (read-only data)
SYMLINK_DIRS = [
    "dev_snapshots",
    "weights",
    "terminal_snapshot",
    "metrics",
    "snapshots",
    "snapshot_analysis",
    "terminal_analysis",
]


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def db_connect():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def verify_source(conn, experiment_id, lane_id):
    """Verify source experiment, lane, and DEVTRAIN data exist."""
    row = conn.execute(
        "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
    ).fetchone()
    if not row:
        sys.exit(f"Source experiment {experiment_id} not found in DB")

    lane_row = conn.execute(
        "SELECT * FROM lanes WHERE id = ? AND experiment_id = ?",
        (lane_id, experiment_id),
    ).fetchone()
    if not lane_row:
        sys.exit(f"Source lane {lane_id} not found in experiment {experiment_id}")

    lane_path = EXPERIMENTS_DIR / experiment_id / "lanes" / lane_id
    snapshots = lane_path / "dev_snapshots"
    if not snapshots.exists():
        sys.exit(f"dev_snapshots not found at {snapshots}")

    milestones = sorted(p.name for p in snapshots.iterdir() if p.name.startswith("milestone_"))
    if not milestones:
        sys.exit(f"No milestone directories in {snapshots}")

    netinit_row = conn.execute(
        "SELECT configuration FROM stoves WHERE lane_id = ? AND stove_type = 'NETINIT'",
        (lane_id,),
    ).fetchone()
    if not netinit_row or not netinit_row["configuration"]:
        sys.exit("NETINIT configuration not found for source lane")

    print(f"Source experiment: {row['name']}")
    print(f"Source lane:       {lane_id}")
    print(f"Dataset:           {row['dataset_id']}")
    print(f"Milestones:        {len(milestones)} ({milestones[0]}..{milestones[-1]})")

    return {
        "experiment": dict(row),
        "lane_path": lane_path,
        "netinit_config": json.loads(netinit_row["configuration"]),
        "selected_classes": json.loads(row["selected_classes"]),
        "dataset_id": row["dataset_id"],
    }


def create_experiment(conn, source, expansion_factors):
    """Create the new experiment in the DB."""
    exp_id = str(uuid.uuid4())
    name = f"SAE Expansion Comparison — {', '.join(str(ef) + 'x' for ef in expansion_factors)}"
    ts = now_iso()

    conn.execute(
        "INSERT INTO experiments (id, name, description, status, dataset_id, selected_classes, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            exp_id,
            name,
            f"Expansion factor comparison using source {source['experiment']['id']}",
            "running",
            source["dataset_id"],
            json.dumps(source["selected_classes"]),
            ts,
            ts,
        ),
    )

    # Create experiment directory + symlink datasets
    exp_dir = EXPERIMENTS_DIR / exp_id
    (exp_dir / "lanes").mkdir(parents=True, exist_ok=True)

    source_datasets = EXPERIMENTS_DIR / source["experiment"]["id"] / "datasets"
    if source_datasets.exists():
        (exp_dir / "datasets").symlink_to(source_datasets)

    print(f"\nCreated experiment: {exp_id}")
    print(f"  Name: {name}")
    return exp_id


def create_lane_with_stoves(conn, experiment_id, lane_name, order_index, netinit_config, expansion_factor):
    """Create a lane + all 6 stoves in the DB."""
    lane_id = str(uuid.uuid4())
    ts = now_iso()

    conn.execute(
        "INSERT INTO lanes (id, experiment_id, name, order_index, created_at) VALUES (?, ?, ?, ?, ?)",
        (lane_id, experiment_id, lane_name, order_index, ts),
    )

    # Modify NETINIT config for this expansion factor
    config = json.loads(json.dumps(netinit_config))  # deep copy
    config["sae_policy"]["expansion_factor"] = expansion_factor

    for stove_type, stove_order in STOVE_TYPES:
        stove_id = str(uuid.uuid4())
        if stove_type == "NETINIT":
            status, cfg = "completed", json.dumps(config)
        elif stove_type == "DEVTRAIN":
            status, cfg = "completed", json.dumps({"snapshots_captured": 11})
        elif stove_type in ("TERMREP", "SNAPANALYSIS"):
            status, cfg = "completed", None
        else:
            status, cfg = "not_started", None

        conn.execute(
            "INSERT INTO stoves (id, lane_id, stove_type, status, progress, configuration, order_index, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (stove_id, lane_id, stove_type, status, 0.0, cfg, stove_order, ts, ts),
        )

    print(f"  Lane {lane_name}: {lane_id}")
    return lane_id, config


def setup_lane_filesystem(experiment_id, lane_id, source_lane_path):
    """Create lane directory, symlink source data, create sae_analysis dir."""
    lane_dir = EXPERIMENTS_DIR / experiment_id / "lanes" / lane_id
    lane_dir.mkdir(parents=True, exist_ok=True)

    for dirname in SYMLINK_DIRS:
        src = source_lane_path / dirname
        dst = lane_dir / dirname
        if src.exists() and not dst.exists():
            dst.symlink_to(src)

    # sae_analysis must be a real directory (SAEANALYSIS writes here)
    (lane_dir / "sae_analysis").mkdir(exist_ok=True)

    return lane_dir


def update_stove(conn, lane_id, stove_type, status, progress=None):
    """Update a stove's status and optionally progress."""
    ts = now_iso()
    if progress is not None:
        conn.execute(
            "UPDATE stoves SET status = ?, progress = ?, updated_at = ? WHERE lane_id = ? AND stove_type = ?",
            (status, progress, ts, lane_id, stove_type),
        )
    else:
        conn.execute(
            "UPDATE stoves SET status = ?, updated_at = ? WHERE lane_id = ? AND stove_type = ?",
            (status, ts, lane_id, stove_type),
        )
    conn.commit()


def run_saeanalysis(conn, lane_id, lane_dir, selected_classes, netinit_config, dataset_id):
    """Run SAEANALYSIS for a single lane. Returns True on success."""
    ef = netinit_config["sae_policy"]["expansion_factor"]

    experiment_config = {
        "dataset_id": dataset_id,
        "sae_policy": netinit_config["sae_policy"],
    }

    update_stove(conn, lane_id, "SAEANALYSIS", "running", 0.0)

    def progress_cb(pct, msg):
        print(f"  [{pct:5.1f}%] {msg}")
        # Update DB progress periodically (every ~5%)
        if int(pct) % 5 == 0:
            update_stove(conn, lane_id, "SAEANALYSIS", "running", pct / 100.0)

    try:
        from saeanalysis import analyze_sae_features

        result = analyze_sae_features(
            lane_dir=str(lane_dir),
            selected_classes=selected_classes,
            experiment_config=experiment_config,
            progress_callback=progress_cb,
        )

        if "error" in result:
            print(f"  SAEANALYSIS returned error: {result['error']}")
            update_stove(conn, lane_id, "SAEANALYSIS", "error", 0.0)
            return False

        results_path = Path(lane_dir) / "sae_analysis" / "sae_results.json"
        print(f"  Results: {results_path} (exists: {results_path.exists()})")

        update_stove(conn, lane_id, "SAEANALYSIS", "completed", 1.0)
        return True

    except Exception:
        traceback.print_exc()
        update_stove(conn, lane_id, "SAEANALYSIS", "error", 0.0)
        return False


def archive_lane_to_workspace(experiment_id, lane_id):
    """Copy completed lane data to the workspace volume and replace with symlink.

    After a lane finishes (all stoves completed), the lane directory can be
    large (checkpoints, snapshots, SAE results).  This function:
      1. Copies the lane directory to WORKSPACE_VOLUME/<experiment_id>/lanes/<lane_id>
      2. Removes the original lane directory
      3. Creates a symlink from the original path → workspace copy

    This frees space on the (small) working volume while keeping all file
    paths valid through the symlink.
    """
    lane_dir = EXPERIMENTS_DIR / experiment_id / "lanes" / lane_id
    archive_dir = WORKSPACE_VOLUME / "experiments" / experiment_id / "lanes" / lane_id

    if not lane_dir.exists():
        print(f"  [archive] Lane dir does not exist: {lane_dir}")
        return False

    if lane_dir.is_symlink():
        print(f"  [archive] Already archived (symlink): {lane_dir} -> {lane_dir.resolve()}")
        return True

    if not WORKSPACE_VOLUME.exists():
        print(f"  [archive] Workspace volume not mounted: {WORKSPACE_VOLUME}")
        print(f"  [archive] Skipping archive — set RUNPOD_WORKSPACE_VOLUME if path differs")
        return False

    print(f"  [archive] Copying lane data to workspace volume...")
    print(f"    src:  {lane_dir}")
    print(f"    dest: {archive_dir}")

    archive_dir.parent.mkdir(parents=True, exist_ok=True)

    # Use copytree to preserve symlinks within the lane (e.g. shared DEVTRAIN data)
    if archive_dir.exists():
        print(f"  [archive] Destination already exists, removing stale copy...")
        shutil.rmtree(archive_dir)

    shutil.copytree(lane_dir, archive_dir, symlinks=True)

    # Verify the copy succeeded by checking key files exist
    if not archive_dir.exists():
        print(f"  [archive] ERROR: Copy failed — archive dir not created")
        return False

    src_count = sum(1 for _ in lane_dir.rglob("*") if _.is_file())
    dst_count = sum(1 for _ in archive_dir.rglob("*") if _.is_file())
    if dst_count < src_count:
        print(f"  [archive] WARNING: File count mismatch (src={src_count}, dst={dst_count})")
        print(f"  [archive] Keeping original — not creating symlink")
        return False

    print(f"  [archive] Verified: {dst_count} files copied")

    # Remove original and create symlink
    shutil.rmtree(lane_dir)
    lane_dir.symlink_to(archive_dir)

    print(f"  [archive] Symlink created: {lane_dir} -> {archive_dir}")
    print(f"  [archive] Lane archived successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description="SAE Expansion Factor Comparison")
    parser.add_argument(
        "--expansion-factors", type=int, nargs="+", default=DEFAULT_EXPANSION_FACTORS,
        help="Expansion factors to compare (default: 4 8 16)",
    )
    parser.add_argument(
        "--source-experiment", default=DEFAULT_SOURCE_EXPERIMENT,
        help="Source experiment UUID",
    )
    parser.add_argument(
        "--source-lane", default=DEFAULT_SOURCE_LANE,
        help="Source lane UUID",
    )
    args = parser.parse_args()

    # Setup ML imports
    sys.path.insert(0, str(ML_SRC_DIR))

    print("=" * 60)
    print("SAE Expansion Factor Comparison")
    print(f"Factors: {args.expansion_factors}")
    print("=" * 60)

    conn = db_connect()

    # 1. Verify source
    source = verify_source(conn, args.source_experiment, args.source_lane)

    # 2. Create experiment
    experiment_id = create_experiment(conn, source, args.expansion_factors)

    # 3. Create lanes + stoves + filesystem
    lanes = []
    for i, ef in enumerate(args.expansion_factors):
        lane_id, config = create_lane_with_stoves(
            conn, experiment_id, f"Expansion {ef}x", i, source["netinit_config"], ef
        )
        lane_dir = setup_lane_filesystem(experiment_id, lane_id, source["lane_path"])
        lanes.append((ef, lane_id, lane_dir, config))

    conn.commit()
    print(f"\nDB records created. Starting SAEANALYSIS runs...\n")

    # 4. Run SAEANALYSIS sequentially
    results = {}
    for ef, lane_id, lane_dir, config in lanes:
        print("=" * 60)
        print(f"SAEANALYSIS — Expansion Factor {ef}x")
        print(f"Lane: {lane_id}")
        print(f"SAE policy: {json.dumps(config['sae_policy'], indent=2)}")
        print("=" * 60)

        success = run_saeanalysis(
            conn, lane_id, lane_dir,
            source["selected_classes"], config, source["dataset_id"],
        )
        results[ef] = success

        if success:
            # Archive completed lane to workspace volume
            print(f"\nArchiving lane {ef}x to workspace volume...")
            if not archive_lane_to_workspace(experiment_id, lane_id):
                print(f"WARNING: Archive failed for {ef}x — lane data remains on working volume")
        else:
            print(f"\nWARNING: {ef}x failed, continuing to next...\n")

        # Memory cleanup between runs
        gc.collect()
        try:
            import torch
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # 5. Update experiment status
    all_ok = all(results.values())
    conn.execute(
        "UPDATE experiments SET status = ?, updated_at = ? WHERE id = ?",
        ("completed" if all_ok else "error", now_iso(), experiment_id),
    )
    conn.commit()
    conn.close()

    # 6. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for ef, ok in results.items():
        print(f"  {ef}x: {'COMPLETED' if ok else 'FAILED'}")
    print(f"\nExperiment ID: {experiment_id}")
    print(f"View in frontend to compare results across lanes.")
    print("=" * 60)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
