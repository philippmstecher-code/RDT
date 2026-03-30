#!/usr/bin/env python3
"""
Seed 256 Pipeline Orchestrator — 3 architectures (ResNet18, ViT-Small, CCT-7).

Runs the full stove pipeline (NETINIT → DEVTRAIN → TERMREP → SNAPANALYSIS + SAEANALYSIS)
for each architecture sequentially. Uses the backend API for all operations.

Supports resume: completed stoves are skipped automatically.

Usage:
    python runpodScripts/run_seed256.py
    python runpodScripts/run_seed256.py --dry-run
    python runpodScripts/run_seed256.py --lane resnet18
"""
import os
import sys
import json
import time
import shutil
import argparse
from datetime import datetime, timezone
from pathlib import Path

from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# ── Constants ──────────────────────────────────────────────────────────
API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
EXPERIMENT_ID = "aaaee715-5b96-4b46-9ffe-0c0bdb5f6e3f"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = DATA_DIR / "experiments"
WORKSPACE_VOLUME = Path(os.environ.get("RUNPOD_WORKSPACE_VOLUME", "/workspace"))

# Pipeline order. SNAPANALYSIS and SAEANALYSIS run in parallel (same order_index).
PIPELINE = ["NETINIT", "DEVTRAIN", "TERMREP", "SNAPANALYSIS", "SAEANALYSIS"]

# ── Lane Configurations ───────────────────────────────────────────────
BASE_CONFIG = {
    "initialization_method": "kaiming_normal",
    "random_seed": 256,
    "batch_size": 128,
    "training_policy": {
        "epochs": 50,
        "accuracy_threshold": 70,
    },
    "snapshot_policy": {
        "milestone_count": 10,
        "distribution_scheme": "uniform",
        "samples_per_class": 100,
        "terminal_capture": "both",
    },
    "transform_config": {
        "normalize_mean": [0.5071, 0.4867, 0.4408],
        "normalize_std": [0.2675, 0.2565, 0.2761],
        "augmentation_enabled": True,
        "random_crop_padding": 4,
        "horizontal_flip": True,
    },
    "validation_policy": {
        "frequency_mode": "percentage",
        "frequency_value": 5,
        "min_batch_interval": 5,
        "validation_set_fraction": 1,
    },
    "memory_policy": {
        "disk_based_linear_probes": True,
        "disk_based_multilayer": True,
        "linear_probe_batch_size": 32,
        "activation_extraction_enabled": True,
        "activation_extraction_interval": 50,
    },
    "logging_policy": {
        "batch_log_frequency": 10,
        "batch_log_include_activations": False,
        "metrics_flush_interval": 10,
        "metrics_keep_recent": 50,
    },
    "sae_policy": {
        "expansion_factor": 4,
        "k_sparse": 32,
        "n_steps": 5000,
        "shared_init_seed": 42,
        "null_permutations": 1000,
    },
    "representation_methods": [
        "mean_activations",
        "individual_activations",
        "multilayer_activations",
    ],
}

LANE_CONFIGS = {
    "resnet18": {
        "name": "ResNet18-100class-10ms-seed256",
        "network_type": "resnet18",
        "optimizer": {
            "type": "sgd",
            "learning_rate": 0.1,
            "weight_decay": 0.0005,
            "momentum": 0.9,
            "nesterov": True,
        },
    },
    "vit_small": {
        "name": "VitSmall-100class-10ms-seed256",
        "network_type": "vit_small",
        "optimizer": {
            "type": "adamw",
            "learning_rate": 0.0005,
            "weight_decay": 0.03,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
        },
        "transform_config": {
            "resize": 224,
        },
    },
    "cct_7": {
        "name": "CCT7-100class-10ms-seed256",
        "network_type": "cct_7",
        "optimizer": {
            "type": "adamw",
            "learning_rate": 0.0005,
            "weight_decay": 0.03,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
        },
    },
}

LANE_ORDER = ["resnet18", "vit_small", "cct_7"]


# ── Helpers ────────────────────────────────────────────────────────────
def api(method, path, json_body=None):
    """Make an API request using stdlib. Returns parsed JSON or None on failure."""
    url = f"{API_BASE}{path}"
    headers = {"Content-Type": "application/json"} if json_body is not None else {}
    data = json.dumps(json_body).encode() if json_body is not None else None
    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except HTTPError as e:
        body = e.read().decode()[:500]
        print(f"API error {e.code}: {body}")
        return None
    except (URLError, OSError):
        sys.exit(f"Cannot connect to backend at {API_BASE} — is it running?")


def now_ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def build_netinit_config(lane_key):
    """Merge base config with lane-specific overrides."""
    overrides = LANE_CONFIGS[lane_key]
    config = json.loads(json.dumps(BASE_CONFIG))  # deep copy
    config["network_type"] = overrides["network_type"]
    config["optimizer"] = overrides["optimizer"]
    if "transform_config" in overrides:
        config["transform_config"].update(overrides["transform_config"])
    return config


# ── Lane Discovery / Creation ─────────────────────────────────────────
def get_existing_lanes():
    """Fetch lanes already created for this experiment."""
    data = api("GET", f"/api/lanes/experiment/{EXPERIMENT_ID}")
    if not data:
        return {}
    lanes = data if isinstance(data, list) else data.get("lanes", data.get("data", []))
    return {l["name"]: l for l in lanes}


def get_stoves_for_lane(lane_id):
    """Return dict of stove_type -> stove for a lane."""
    data = api("GET", f"/api/stoves/lane/{lane_id}")
    if not data:
        return {}
    stoves = data if isinstance(data, list) else data.get("stoves", data.get("data", []))
    return {s["stove_type"]: s for s in stoves}


def create_lane(lane_key):
    """Create a lane via the API. Returns the lane dict."""
    lane_name = LANE_CONFIGS[lane_key]["name"]
    data = api("POST", "/api/lanes", {
        "experiment_id": EXPERIMENT_ID,
        "name": lane_name,
    })
    if not data:
        sys.exit(f"Failed to create lane {lane_name}")
    lane_id = data.get("id") or data.get("lane_id")
    print(f"  Created lane: {lane_name} ({lane_id})")
    return data


def configure_netinit(stove_id, lane_key):
    """Configure a NETINIT stove with the correct hyperparameters."""
    config = build_netinit_config(lane_key)
    data = api("PATCH", f"/api/stoves/{stove_id}/configuration", {"configuration": config})
    if data is None:
        sys.exit(f"Failed to configure NETINIT stove {stove_id}")
    print(f"  Configured NETINIT: {LANE_CONFIGS[lane_key]['network_type']} seed=256")
    return data


# ── Stove Execution ───────────────────────────────────────────────────
def start_stove(stove_type, stove_id):
    """Start a stove via the appropriate API endpoint."""
    endpoints = {
        "NETINIT":       ("POST", f"/api/netinit/initialize/{stove_id}"),
        "DEVTRAIN":      ("POST", "/api/devtrain/start"),
        "TERMREP":       ("POST", "/api/termrep/start"),
        "SNAPANALYSIS":  ("POST", "/api/snapanalysis/start"),
        "SAEANALYSIS":   ("POST", "/api/saeanalysis-stove/start"),
    }
    method, path = endpoints[stove_type]
    body = {"stove_id": stove_id} if stove_type != "NETINIT" else None
    return api(method, path, body)


def poll_stove(stove_id, stove_type, poll_interval=30, timeout=36000):
    """Poll a stove until completed or error. Returns final status."""
    start = time.time()
    last_progress = -1

    while time.time() - start < timeout:
        data = api("GET", f"/api/stoves/{stove_id}")
        if not data:
            print(f"  [{now_ts()}] WARNING: Could not fetch stove status, retrying...")
            time.sleep(poll_interval)
            continue

        status = data.get("status", "unknown")
        progress = data.get("progress", 0.0)
        pct = progress * 100 if isinstance(progress, (int, float)) else 0

        if status == "completed":
            print(f"  [{now_ts()}] {stove_type}: COMPLETED")
            return "completed"
        if status == "error":
            print(f"  [{now_ts()}] {stove_type}: ERROR")
            return "error"

        # Only log when progress changes meaningfully
        if int(pct) != last_progress:
            print(f"  [{now_ts()}] {stove_type}: {status} ({pct:.1f}%)")
            last_progress = int(pct)

        # Adaptive polling: faster for quick stoves, slower for training
        if stove_type == "DEVTRAIN":
            time.sleep(max(poll_interval, 60))
        elif stove_type in ("NETINIT",):
            time.sleep(5)
        else:
            time.sleep(poll_interval)

    print(f"  [{now_ts()}] {stove_type}: TIMEOUT after {timeout}s")
    return "timeout"


def archive_lane_to_workspace(experiment_id, lane_id):
    """Copy completed lane data to workspace volume, replace with symlink."""
    lane_dir = EXPERIMENTS_DIR / experiment_id / "lanes" / lane_id
    archive_dir = WORKSPACE_VOLUME / "experiments" / experiment_id / "lanes" / lane_id

    if not lane_dir.exists():
        print(f"  [archive] Lane dir does not exist: {lane_dir}")
        return False
    if lane_dir.is_symlink():
        print(f"  [archive] Already archived: {lane_dir} -> {lane_dir.resolve()}")
        return True
    if not WORKSPACE_VOLUME.exists():
        print(f"  [archive] Workspace volume not mounted: {WORKSPACE_VOLUME}")
        return False

    print(f"  [archive] Copying to {archive_dir}...")
    archive_dir.parent.mkdir(parents=True, exist_ok=True)

    if archive_dir.exists():
        shutil.rmtree(archive_dir)

    shutil.copytree(lane_dir, archive_dir, symlinks=True)

    if not archive_dir.exists():
        print(f"  [archive] ERROR: Copy failed")
        return False

    src_count = sum(1 for p in lane_dir.rglob("*") if p.is_file())
    dst_count = sum(1 for p in archive_dir.rglob("*") if p.is_file())
    if dst_count < src_count:
        print(f"  [archive] WARNING: File count mismatch (src={src_count}, dst={dst_count})")
        return False

    print(f"  [archive] Verified: {dst_count} files")
    shutil.rmtree(lane_dir)
    lane_dir.symlink_to(archive_dir)
    print(f"  [archive] Symlink created: {lane_dir} -> {archive_dir}")
    return True


# ── Pipeline Runner ───────────────────────────────────────────────────
def run_lane_pipeline(lane_key, dry_run=False):
    """Run the full stove pipeline for a single lane."""
    lane_name = LANE_CONFIGS[lane_key]["name"]

    print(f"\n{'=' * 60}")
    print(f"LANE: {lane_name}")
    print(f"{'=' * 60}")

    # Find or create lane
    existing = get_existing_lanes()
    if lane_name in existing:
        lane = existing[lane_name]
        lane_id = lane["id"]
        print(f"  Found existing lane: {lane_id}")
    else:
        if dry_run:
            print(f"  [DRY RUN] Would create lane: {lane_name}")
            return True
        lane = create_lane(lane_key)
        lane_id = lane.get("id") or lane.get("lane_id")

    # Get stoves
    stoves = get_stoves_for_lane(lane_id)
    if not stoves:
        sys.exit(f"No stoves found for lane {lane_id}")

    # Configure NETINIT if not yet configured/completed
    netinit = stoves.get("NETINIT")
    if netinit and netinit["status"] not in ("completed", "running"):
        if dry_run:
            print(f"  [DRY RUN] Would configure NETINIT")
        else:
            configure_netinit(netinit["id"], lane_key)

    # Run sequential stages: NETINIT → DEVTRAIN → TERMREP
    for stove_type in ["NETINIT", "DEVTRAIN", "TERMREP"]:
        stove = stoves.get(stove_type)
        if not stove:
            sys.exit(f"Missing {stove_type} stove for lane {lane_id}")

        if stove["status"] == "completed":
            print(f"  {stove_type}: already completed, skipping")
            continue

        if dry_run:
            print(f"  [DRY RUN] Would run {stove_type} ({stove['id']})")
            continue

        # If already running, just poll — don't try to start again
        if stove["status"] == "running":
            print(f"\n  {stove_type} already running, resuming poll...")
        else:
            print(f"\n  Starting {stove_type}...")
            result = start_stove(stove_type, stove["id"])
            if result is None:
                print(f"  ERROR: Failed to start {stove_type}")
                return False

        status = poll_stove(stove["id"], stove_type)
        if status != "completed":
            print(f"  PIPELINE FAILED at {stove_type} (status={status})")
            return False

    # Run parallel stages: SNAPANALYSIS + SAEANALYSIS
    # Re-fetch stoves to get current status after sequential stages
    stoves = get_stoves_for_lane(lane_id)
    parallel_stoves = {}
    for stove_type in ["SNAPANALYSIS", "SAEANALYSIS"]:
        stove = stoves.get(stove_type)
        if not stove:
            sys.exit(f"Missing {stove_type} stove for lane {lane_id}")

        if stove["status"] == "completed":
            print(f"  {stove_type}: already completed, skipping")
            continue

        if dry_run:
            print(f"  [DRY RUN] Would run {stove_type} ({stove['id']})")
            continue

        # If already running, just poll — don't try to start again
        if stove["status"] == "running":
            print(f"\n  {stove_type} already running, resuming poll...")
        else:
            print(f"\n  Starting {stove_type}...")
            result = start_stove(stove_type, stove["id"])
            if result is None:
                print(f"  ERROR: Failed to start {stove_type}")
                return False
        parallel_stoves[stove_type] = stove

    if dry_run:
        return True

    # Poll parallel stoves until both complete
    for stove_type, stove in parallel_stoves.items():
        status = poll_stove(stove["id"], stove_type)
        if status != "completed":
            print(f"  PIPELINE FAILED at {stove_type} (status={status})")
            return False

    print(f"\n  Lane {lane_name}: ALL STOVES COMPLETED")

    # Archive to workspace volume
    if not archive_lane_to_workspace(EXPERIMENT_ID, lane_id):
        print(f"  WARNING: Archive failed — lane data remains on working volume")

    return True


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Seed 256 Pipeline Orchestrator — 3 architectures",
    )
    parser.add_argument(
        "--lane", choices=list(LANE_CONFIGS.keys()),
        help="Run only a specific lane (default: all three sequentially)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without executing",
    )
    args = parser.parse_args()

    lanes_to_run = [args.lane] if args.lane else LANE_ORDER

    print("=" * 60)
    print("Seed 256 Pipeline Orchestrator")
    print(f"Experiment: {EXPERIMENT_ID}")
    print(f"Lanes:      {', '.join(LANE_CONFIGS[k]['name'] for k in lanes_to_run)}")
    print(f"Pipeline:   {' → '.join(PIPELINE)}")
    print(f"Started:    {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    # Verify backend is reachable by fetching the experiment
    exp = api("GET", f"/api/experiments/{EXPERIMENT_ID}")
    if exp is None:
        sys.exit(
            f"Backend not reachable or experiment {EXPERIMENT_ID} not found. "
            "Ensure the backend is running and the experiment exists."
        )
    print(f"Experiment: {exp.get('name', 'unknown')}\n")

    results = {}
    for lane_key in lanes_to_run:
        success = run_lane_pipeline(lane_key, dry_run=args.dry_run)
        results[lane_key] = success
        if not success and not args.dry_run:
            print(f"\nLane {lane_key} FAILED — stopping orchestrator")
            break

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    for key, ok in results.items():
        status = "COMPLETED" if ok else "FAILED"
        print(f"  {LANE_CONFIGS[key]['name']}: {status}")
    print(f"\nFinished: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
