#!/usr/bin/env python3
"""
Independent SAE Initialization Experiment.

Creates a new experiment + lane that reuses DEVTRAIN data from the
ResNet18-seed256 lane but runs SAEANALYSIS with shared_init_seed=None,
so each checkpoint×layer SAE gets its own independent random initialization.

This tests whether the shared-init assumption matters for process detection.

Usage:
    # Step 1: Create experiment, lane, symlinks (setup only)
    python runpodScripts/run_independent_init_sae.py --setup

    # Step 2: Run SAE analysis (after reviewing setup)
    python runpodScripts/run_independent_init_sae.py --run
"""
import sys
import json
import uuid
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ML_SRC_DIR = PROJECT_ROOT / "packages" / "ml" / "src"
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "rcx.db"
EXPERIMENTS_DIR = DATA_DIR / "experiments"

# ── Source lane (ResNet18-seed256, completed all stoves) ──────────────
SOURCE_EXPERIMENT_ID = "aaaee715-5b96-4b46-9ffe-0c0bdb5f6e3f"
SOURCE_LANE_ID = "935c13ae-a026-4eae-a2b2-0306879b2e8c"

# ── New experiment identifiers (stable across reruns) ─────────────────
NEW_EXPERIMENT_ID = "b1c2d3e4-5f6a-7b8c-9d0e-f1a2b3c4d5e6"
NEW_LANE_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def db_connect():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def get_source_config(conn):
    """Read NETINIT config and selected_classes from the source lane."""
    exp = conn.execute(
        "SELECT * FROM experiments WHERE id = ?", (SOURCE_EXPERIMENT_ID,)
    ).fetchone()
    if not exp:
        sys.exit(f"Source experiment {SOURCE_EXPERIMENT_ID} not found")

    stove = conn.execute(
        "SELECT configuration FROM stoves WHERE lane_id = ? AND stove_type = 'NETINIT'",
        (SOURCE_LANE_ID,),
    ).fetchone()
    if not stove:
        sys.exit(f"NETINIT stove not found for source lane {SOURCE_LANE_ID}")

    netinit_config = json.loads(stove["configuration"])
    selected_classes = json.loads(exp["selected_classes"])
    return netinit_config, selected_classes, exp["dataset_id"]


def setup(conn):
    """Create experiment, lane, stoves, and symlinks."""
    print("=" * 60)
    print("SETUP: Independent SAE Init Experiment")
    print("=" * 60)

    # Get source config
    netinit_config, selected_classes, dataset_id = get_source_config(conn)

    # Modify SAE policy: remove shared_init_seed
    netinit_config_independent = json.loads(json.dumps(netinit_config))
    netinit_config_independent["sae_policy"]["shared_init_seed"] = None
    print(f"\nSource SAE policy: shared_init_seed={netinit_config['sae_policy'].get('shared_init_seed')}")
    print(f"New SAE policy:    shared_init_seed={netinit_config_independent['sae_policy']['shared_init_seed']}")

    # Check if experiment already exists
    existing = conn.execute(
        "SELECT id FROM experiments WHERE id = ?", (NEW_EXPERIMENT_ID,)
    ).fetchone()
    if existing:
        print(f"\nExperiment already exists: {NEW_EXPERIMENT_ID}")
        print("Skipping DB creation. Checking symlinks...")
    else:
        # Create experiment
        ts = now_iso()
        conn.execute(
            "INSERT INTO experiments (id, name, dataset_id, selected_classes, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                NEW_EXPERIMENT_ID,
                "IndependentInit-ResNet18-seed256",
                dataset_id,
                json.dumps(selected_classes),
                ts, ts,
            ),
        )

        # Create lane
        conn.execute(
            "INSERT INTO lanes (id, experiment_id, name, order_index, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                NEW_LANE_ID,
                NEW_EXPERIMENT_ID,
                "ResNet18-100class-10ms-seed256-indepSAE",
                0,
                ts,
            ),
        )

        # Create stoves — mark NETINIT through SNAPANALYSIS as completed (reused via symlinks)
        stove_defs = [
            ("NETINIT", "completed", 1.0, json.dumps(netinit_config_independent), 0),
            ("DEVTRAIN", "completed", 1.0, None, 1),
            ("TERMREP", "completed", 1.0, None, 2),
            ("SNAPANALYSIS", "completed", 1.0, None, 3),
            ("SAEANALYSIS", "not_started", 0.0, None, 4),
        ]
        for stove_type, status, progress, config, order in stove_defs:
            stove_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO stoves (id, lane_id, stove_type, status, progress, configuration, order_index, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (stove_id, NEW_LANE_ID, stove_type, status, progress, config, order, ts, ts),
            )

        conn.commit()
        print(f"\nCreated experiment: IndependentInit-ResNet18-seed256 ({NEW_EXPERIMENT_ID})")
        print(f"Created lane:       ResNet18-100class-10ms-seed256-indepSAE ({NEW_LANE_ID})")

    # Create lane directory with symlinks
    source_lane_dir = EXPERIMENTS_DIR / SOURCE_EXPERIMENT_ID / "lanes" / SOURCE_LANE_ID
    new_lane_dir = EXPERIMENTS_DIR / NEW_EXPERIMENT_ID / "lanes" / NEW_LANE_ID

    # Resolve source (may itself be a symlink to /workspace)
    source_resolved = source_lane_dir.resolve()
    print(f"\nSource lane dir: {source_lane_dir}")
    print(f"  Resolves to:   {source_resolved}")
    print(f"New lane dir:    {new_lane_dir}")

    new_lane_dir.mkdir(parents=True, exist_ok=True)

    # Symlink DEVTRAIN-related directories
    dirs_to_symlink = [
        "dev_snapshots",
        "terminal_analysis",
        "terminal_snapshot",
        "snapshots",
        "weights",
        "metrics",
        "snapshot_analysis",
    ]

    for dirname in dirs_to_symlink:
        src = source_resolved / dirname
        dst = new_lane_dir / dirname
        if not src.exists():
            print(f"  SKIP (not found): {dirname}")
            continue
        if dst.exists() or dst.is_symlink():
            print(f"  EXISTS: {dirname} -> {dst.resolve()}")
            continue
        dst.symlink_to(src)
        print(f"  LINKED: {dirname} -> {src}")

    # Ensure sae_analysis dir exists (fresh, not symlinked)
    sae_dir = new_lane_dir / "sae_analysis"
    if sae_dir.exists() and not sae_dir.is_symlink():
        print(f"\n  sae_analysis/ already exists (will be used for independent SAE results)")
    else:
        sae_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Created fresh: sae_analysis/")

    # Also symlink datasets if needed
    datasets_src = (EXPERIMENTS_DIR / SOURCE_EXPERIMENT_ID / "datasets").resolve()
    datasets_dst = EXPERIMENTS_DIR / NEW_EXPERIMENT_ID / "datasets"
    if datasets_src.exists() and not datasets_dst.exists():
        datasets_dst.symlink_to(datasets_src)
        print(f"  LINKED: datasets -> {datasets_src}")

    print(f"\n{'=' * 60}")
    print("SETUP COMPLETE")
    print(f"{'=' * 60}")
    print(f"\nNew lane directory: {new_lane_dir}")
    print(f"SAE analysis output: {sae_dir}")
    print(f"\nTo run SAE analysis:")
    print(f"  python runpodScripts/run_independent_init_sae.py --run")


def run_sae(conn):
    """Run SAEANALYSIS with independent init on the new lane."""
    print("=" * 60)
    print("RUNNING: Independent SAE Init Analysis")
    print("=" * 60)

    sys.path.insert(0, str(ML_SRC_DIR))
    from saeanalysis import analyze_sae_features

    # Get config
    netinit_config, selected_classes, dataset_id = get_source_config(conn)

    # Override shared_init_seed to None for independent init
    experiment_config = json.loads(json.dumps(netinit_config))
    experiment_config["sae_policy"]["shared_init_seed"] = None
    experiment_config["dataset_id"] = dataset_id

    sae_policy = experiment_config["sae_policy"]
    print(f"\nSAE Policy:")
    print(f"  expansion_factor:  {sae_policy['expansion_factor']}")
    print(f"  k_sparse:          {sae_policy['k_sparse']}")
    print(f"  n_steps:           {sae_policy['n_steps']}")
    print(f"  shared_init_seed:  {sae_policy['shared_init_seed']}  <-- INDEPENDENT")
    print(f"  null_permutations: {sae_policy['null_permutations']}")

    new_lane_dir = EXPERIMENTS_DIR / NEW_EXPERIMENT_ID / "lanes" / NEW_LANE_ID
    if not new_lane_dir.exists():
        sys.exit(f"Lane directory not found: {new_lane_dir}\nRun --setup first.")

    # Verify symlinks are in place
    required_symlinks = ["dev_snapshots", "terminal_analysis"]
    for dirname in required_symlinks:
        p = new_lane_dir / dirname
        if not p.exists():
            sys.exit(f"Required directory not found: {p}\nRun --setup first.")

    # Update stove status
    ts = now_iso()
    conn.execute(
        "UPDATE stoves SET status = 'running', progress = 0.0, updated_at = ? "
        "WHERE lane_id = ? AND stove_type = 'SAEANALYSIS'",
        (ts, NEW_LANE_ID),
    )
    conn.commit()

    try:
        results = analyze_sae_features(
            lane_dir=str(new_lane_dir),
            selected_classes=selected_classes,
            experiment_config=experiment_config,
            expansion_factor=sae_policy["expansion_factor"],
            k_sparse=sae_policy["k_sparse"],
            n_steps=sae_policy["n_steps"],
        )

        ts = now_iso()
        conn.execute(
            "UPDATE stoves SET status = 'completed', progress = 1.0, updated_at = ? "
            "WHERE lane_id = ? AND stove_type = 'SAEANALYSIS'",
            (ts, NEW_LANE_ID),
        )
        conn.commit()

        print(f"\n{'=' * 60}")
        print("SAEANALYSIS COMPLETE (Independent Init)")
        print(f"{'=' * 60}")

        # Print summary
        if "error" in results:
            print(f"ERROR: {results['error']}")
        else:
            summary = results.get("global_summary", {})
            print(f"Checkpoints analyzed: {summary.get('num_checkpoints', '?')}")
            print(f"Layers analyzed:      {summary.get('num_layers', '?')}")
            print(f"Total features:       {summary.get('total_features', '?')}")

            process_counts = summary.get("process_counts", {})
            if process_counts:
                print(f"\nProcess counts:")
                for proc, count in sorted(process_counts.items()):
                    print(f"  {proc}: {count}")

        sae_results_path = new_lane_dir / "sae_analysis" / "sae_results.json"
        print(f"\nResults: {sae_results_path}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        ts = now_iso()
        conn.execute(
            "UPDATE stoves SET status = 'error', progress = 0.0, updated_at = ? "
            "WHERE lane_id = ? AND stove_type = 'SAEANALYSIS'",
            (ts, NEW_LANE_ID),
        )
        conn.commit()
        sys.exit(1)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Independent SAE Init Experiment")
    parser.add_argument("--setup", action="store_true", help="Create experiment, lane, and symlinks")
    parser.add_argument("--run", action="store_true", help="Run SAEANALYSIS with independent init")
    args = parser.parse_args()

    if not args.setup and not args.run:
        parser.print_help()
        print("\nUse --setup first, then --run")
        sys.exit(0)

    conn = db_connect()

    try:
        if args.setup:
            setup(conn)
        if args.run:
            run_sae(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
