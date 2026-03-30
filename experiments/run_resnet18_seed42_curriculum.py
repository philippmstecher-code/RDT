#!/usr/bin/env python3
"""
ResNet-18 Seed-42 Curriculum Experiment — Two Lanes on CIFAR-100.

Both lanes use the standard pipeline (NETINIT → DEVTRAIN → TERMREP → SNAPANALYSIS
+ SAEANALYSIS) via the backend API. Causal intervention is excluded.

Lane A ("Standard"): ResNet-18 trained on 100 fine classes for 50 epochs,
    optimised for best accuracy.

Lane B ("Superclass-Curriculum"): ResNet-18 trained on 20 superclass labels
    for epochs 1–25 (coarse reward via curriculum_policy), then switches to
    100 fine-class labels for epochs 26–50.

Both lanes use weight-update-based milestones to capture snapshots at T=0
and every 5 epochs (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 = 11 snapshots).

Usage:
    python runpodScripts/run_resnet18_seed42_curriculum.py
    python runpodScripts/run_resnet18_seed42_curriculum.py --dry-run
    python runpodScripts/run_resnet18_seed42_curriculum.py --lane standard
    python runpodScripts/run_resnet18_seed42_curriculum.py --lane curriculum
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# ── Constants ──────────────────────────────────────────────────────────
API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = DATA_DIR / "experiments"

EXPERIMENT_NAME = "ResNet18-seed42-curriculum-experiment"
EXPERIMENT_DESCRIPTION = (
    "Two-lane experiment: (A) Standard 100-class ResNet-18 optimised for accuracy, "
    "(B) Superclass-curriculum ResNet-18 trained on 20 superclasses for 25 epochs "
    "then fine-tuned on 100 classes for 25 epochs. Snapshots every 5 epochs."
)
DATASET_ID = "cifar100"

PIPELINE = ["NETINIT", "DEVTRAIN", "TERMREP", "SNAPANALYSIS", "SAEANALYSIS"]

# All 100 CIFAR-100 fine classes (prefixed with cifar100_, matching existing experiments)
CIFAR100_CLASSES = [
    "cifar100_apple", "cifar100_aquarium_fish", "cifar100_baby", "cifar100_bear",
    "cifar100_beaver", "cifar100_bed", "cifar100_bee", "cifar100_beetle",
    "cifar100_bicycle", "cifar100_bottle", "cifar100_bowl", "cifar100_boy",
    "cifar100_bridge", "cifar100_bus", "cifar100_butterfly", "cifar100_camel",
    "cifar100_can", "cifar100_castle", "cifar100_caterpillar", "cifar100_cattle",
    "cifar100_chair", "cifar100_chimpanzee", "cifar100_clock", "cifar100_cloud",
    "cifar100_cockroach", "cifar100_couch", "cifar100_crab", "cifar100_crocodile",
    "cifar100_cup", "cifar100_dinosaur", "cifar100_dolphin", "cifar100_elephant",
    "cifar100_flatfish", "cifar100_forest", "cifar100_fox", "cifar100_girl",
    "cifar100_hamster", "cifar100_house", "cifar100_kangaroo", "cifar100_keyboard",
    "cifar100_lamp", "cifar100_lawn_mower", "cifar100_leopard", "cifar100_lion",
    "cifar100_lizard", "cifar100_lobster", "cifar100_man", "cifar100_maple_tree",
    "cifar100_motorcycle", "cifar100_mountain", "cifar100_mouse", "cifar100_mushroom",
    "cifar100_oak_tree", "cifar100_orange", "cifar100_orchid", "cifar100_otter",
    "cifar100_palm_tree", "cifar100_pear", "cifar100_pickup_truck", "cifar100_pine_tree",
    "cifar100_plain", "cifar100_plate", "cifar100_poppy", "cifar100_porcupine",
    "cifar100_possum", "cifar100_rabbit", "cifar100_raccoon", "cifar100_ray",
    "cifar100_road", "cifar100_rocket", "cifar100_rose", "cifar100_sea",
    "cifar100_seal", "cifar100_shark", "cifar100_shrew", "cifar100_skunk",
    "cifar100_skyscraper", "cifar100_snail", "cifar100_snake", "cifar100_spider",
    "cifar100_squirrel", "cifar100_streetcar", "cifar100_sunflower",
    "cifar100_sweet_pepper", "cifar100_table", "cifar100_tank", "cifar100_telephone",
    "cifar100_television", "cifar100_tiger", "cifar100_tractor", "cifar100_train",
    "cifar100_trout", "cifar100_tulip", "cifar100_turtle", "cifar100_wardrobe",
    "cifar100_whale", "cifar100_willow_tree", "cifar100_wolf", "cifar100_woman",
    "cifar100_worm",
]

# ── Shared Config ─────────────────────────────────────────────────────
BASE_CONFIG = {
    "network_type": "resnet18",
    "initialization_method": "kaiming_normal",
    "random_seed": 42,
    "batch_size": 128,
    "training_policy": {
        "epochs": 50,
        "accuracy_threshold": 80,
    },
    "snapshot_policy": {
        "milestone_count": 10,
        "milestone_type": "weight_updates",
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

# ── Lane Configurations ───────────────────────────────────────────────
LANE_CONFIGS = {
    "standard": {
        "name": "ResNet18-100class-seed42-optimised",
        "optimizer": {
            "type": "sgd",
            "learning_rate": 0.1,
            "weight_decay": 5e-4,
            "momentum": 0.9,
            "nesterov": True,
        },
        # No curriculum — standard 100-class training throughout
    },
    "curriculum": {
        "name": "ResNet18-superclass-curriculum-seed42",
        "optimizer": {
            "type": "sgd",
            "learning_rate": 0.1,
            "weight_decay": 5e-4,
            "momentum": 0.9,
            "nesterov": True,
        },
        "curriculum_policy": {
            "enabled": True,
            "phases": [
                {
                    "start_epoch": 1,
                    "end_epoch": 25,
                    "label_mode": "superclass",
                    "learning_rate": 0.1,
                },
                {
                    "start_epoch": 26,
                    "end_epoch": 50,
                    "label_mode": "fine",
                    "learning_rate": 0.01,
                },
            ],
        },
    },
}

LANE_ORDER = ["standard", "curriculum"]


# ── API Helpers ───────────────────────────────────────────────────────
def api(method, path, json_body=None, timeout=30):
    url = f"{API_BASE}{path}"
    headers = {"Content-Type": "application/json"} if json_body is not None else {}
    data = json.dumps(json_body).encode() if json_body is not None else None
    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except HTTPError as e:
        body = e.read().decode()[:500]
        print(f"API error {e.code}: {body}")
        return None
    except (URLError, OSError):
        sys.exit(f"Cannot connect to backend at {API_BASE} — is it running?")


def now_ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


# ── Experiment Setup ──────────────────────────────────────────────────
def find_or_create_experiment():
    experiments = api("GET", "/api/experiments")
    if experiments:
        for exp in experiments:
            if exp.get("name") == EXPERIMENT_NAME:
                print(f"  Found existing experiment: {exp['id']}")
                return exp

    print(f"  Creating experiment: {EXPERIMENT_NAME}")
    exp = api("POST", "/api/experiments", {
        "name": EXPERIMENT_NAME,
        "dataset_id": DATASET_ID,
        "selected_classes": CIFAR100_CLASSES,
        "description": EXPERIMENT_DESCRIPTION,
    })
    if not exp:
        sys.exit("Failed to create experiment")
    print(f"  Created experiment: {exp['id']}")
    return exp


def ensure_dataset_downloaded(experiment_id):
    status_resp = api("GET", f"/api/experiments/{experiment_id}/datasets/status")
    if status_resp and status_resp.get("download_status") == "completed":
        print("  Dataset: already downloaded")
        return

    print("  Triggering dataset download...")
    result = api("POST", f"/api/experiments/{experiment_id}/datasets/download")
    if result is None:
        sys.exit("Failed to trigger dataset download")

    while True:
        time.sleep(10)
        status_resp = api("GET", f"/api/experiments/{experiment_id}/datasets/status")
        if not status_resp:
            print(f"  [{now_ts()}] WARNING: Could not fetch download status, retrying...")
            continue
        dl_status = status_resp.get("download_status", "unknown")
        progress = status_resp.get("progress", 0)
        if dl_status == "completed":
            print(f"  [{now_ts()}] Dataset download: COMPLETED")
            return
        if dl_status == "error":
            sys.exit(f"Dataset download failed: {status_resp.get('error_message', 'unknown')}")
        print(f"  [{now_ts()}] Dataset download: {dl_status} ({progress * 100:.0f}%)")


# ── Lane / Stove Helpers ─────────────────────────────────────────────
def get_existing_lanes(experiment_id):
    data = api("GET", f"/api/lanes/experiment/{experiment_id}")
    if not data:
        return {}
    lanes = data if isinstance(data, list) else data.get("lanes", data.get("data", []))
    return {l["name"]: l for l in lanes}


def get_stoves_for_lane(lane_id):
    data = api("GET", f"/api/stoves/lane/{lane_id}")
    if not data:
        return {}
    stoves = data if isinstance(data, list) else data.get("stoves", data.get("data", []))
    return {s["stove_type"]: s for s in stoves}


def create_lane(experiment_id, lane_key):
    lane_name = LANE_CONFIGS[lane_key]["name"]
    data = api("POST", "/api/lanes", {
        "experiment_id": experiment_id,
        "name": lane_name,
    })
    if not data:
        sys.exit(f"Failed to create lane {lane_name}")
    print(f"  Created lane: {lane_name} ({data.get('id')})")
    return data


def build_netinit_config(lane_key):
    """Merge base config with lane-specific overrides."""
    config = json.loads(json.dumps(BASE_CONFIG))  # deep copy
    overrides = LANE_CONFIGS[lane_key]
    config["optimizer"] = overrides["optimizer"]
    # Add curriculum_policy if present
    if "curriculum_policy" in overrides:
        config["curriculum_policy"] = overrides["curriculum_policy"]
    return config


def configure_netinit(stove_id, lane_key):
    config = build_netinit_config(lane_key)
    data = api("PATCH", f"/api/stoves/{stove_id}/configuration", {"configuration": config})
    if data is None:
        sys.exit(f"Failed to configure NETINIT stove {stove_id}")
    has_curriculum = "curriculum_policy" in LANE_CONFIGS[lane_key]
    print(f"  Configured NETINIT: resnet18 seed=42 ({lane_key})"
          f"{' [curriculum enabled]' if has_curriculum else ''}")
    return data


def start_stove(stove_type, stove_id):
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
        if int(pct) != last_progress:
            print(f"  [{now_ts()}] {stove_type}: {status} ({pct:.1f}%)")
            last_progress = int(pct)
        if stove_type == "DEVTRAIN":
            time.sleep(max(poll_interval, 60))
        elif stove_type == "NETINIT":
            time.sleep(5)
        else:
            time.sleep(poll_interval)
    print(f"  [{now_ts()}] {stove_type}: TIMEOUT after {timeout}s")
    return "timeout"


# ── Lane Setup ────────────────────────────────────────────────────────
def setup_lane(experiment_id, lane_key, dry_run=False):
    """Create lane and configure NETINIT. Returns (lane_id, stoves) or None."""
    lane_name = LANE_CONFIGS[lane_key]["name"]

    existing = get_existing_lanes(experiment_id)
    if lane_name in existing:
        lane = existing[lane_name]
        lane_id = lane["id"]
        print(f"  [{lane_key}] Found existing lane: {lane_id}")
    else:
        if dry_run:
            print(f"  [{lane_key}] [DRY RUN] Would create lane: {lane_name}")
            return None
        lane = create_lane(experiment_id, lane_key)
        lane_id = lane.get("id") or lane.get("lane_id")

    stoves = get_stoves_for_lane(lane_id)
    if not stoves:
        sys.exit(f"No stoves found for lane {lane_id}")

    # Configure NETINIT if needed
    netinit = stoves.get("NETINIT")
    if netinit and netinit["status"] not in ("completed", "running"):
        if not dry_run:
            configure_netinit(netinit["id"], lane_key)

    return lane_id, stoves


def run_stove_if_needed(lane_key, stove_type, stoves, dry_run=False):
    """Start a stove if not already completed/running. Returns True on success."""
    stove = stoves.get(stove_type)
    if not stove:
        sys.exit(f"Missing {stove_type} stove for {lane_key}")

    if stove["status"] == "completed":
        print(f"  [{lane_key}] {stove_type}: already completed, skipping")
        return True

    if dry_run:
        print(f"  [{lane_key}] [DRY RUN] Would run {stove_type}")
        return True

    if stove["status"] != "running":
        print(f"  [{lane_key}] Starting {stove_type}...")
        result = start_stove(stove_type, stove["id"])
        if result is None:
            print(f"  [{lane_key}] ERROR: Failed to start {stove_type}")
            return False
    else:
        print(f"  [{lane_key}] {stove_type} already running, resuming poll...")

    return stove


def poll_multiple_stoves(stove_map, poll_interval=30, timeout=36000):
    """Poll multiple stoves concurrently. stove_map: {label: (stove_id, stove_type)}.
    Returns dict of label -> final_status."""
    start = time.time()
    remaining = dict(stove_map)
    results = {}
    last_progress = {label: -1 for label in remaining}

    while remaining and (time.time() - start < timeout):
        for label, (stove_id, stove_type) in list(remaining.items()):
            data = api("GET", f"/api/stoves/{stove_id}")
            if not data:
                continue
            status = data.get("status", "unknown")
            progress = data.get("progress", 0.0)
            pct = progress * 100 if isinstance(progress, (int, float)) else 0

            if status == "completed":
                print(f"  [{now_ts()}] {label}/{stove_type}: COMPLETED")
                results[label] = "completed"
                del remaining[label]
            elif status == "error":
                print(f"  [{now_ts()}] {label}/{stove_type}: ERROR")
                results[label] = "error"
                del remaining[label]
            elif int(pct) != last_progress[label]:
                print(f"  [{now_ts()}] {label}/{stove_type}: {status} ({pct:.1f}%)")
                last_progress[label] = int(pct)

        if remaining:
            time.sleep(poll_interval if len(remaining) == 1 else max(poll_interval, 60))

    # Timeout remaining
    for label in remaining:
        results[label] = "timeout"
    return results


# ── Pipeline Runner (parallel-aware) ─────────────────────────────────
def run_pipeline_parallel(experiment_id, lane_keys, dry_run=False):
    """Run the full pipeline for multiple lanes, training them in parallel."""
    # Step 1: Setup all lanes and run NETINIT
    lane_info = {}  # lane_key -> (lane_id, stoves)
    for lane_key in lane_keys:
        print(f"\n[SETUP] {LANE_CONFIGS[lane_key]['name']}")
        result = setup_lane(experiment_id, lane_key, dry_run)
        if result is None:
            continue
        lane_info[lane_key] = result

    if dry_run:
        return {k: True for k in lane_keys}

    # Step 2: Run NETINIT for all lanes (fast, sequential is fine)
    for lane_key, (lane_id, stoves) in lane_info.items():
        stove = run_stove_if_needed(lane_key, "NETINIT", stoves, dry_run)
        if stove is True:
            continue
        if stove is False:
            return {lane_key: False}
        status = poll_stove(stove["id"], "NETINIT")
        if status != "completed":
            return {lane_key: False}

    # Step 3: Start DEVTRAIN for ALL lanes concurrently
    print(f"\n{'=' * 60}")
    print(f"PARALLEL DEVTRAIN — {len(lane_info)} lanes")
    print(f"{'=' * 60}")
    devtrain_polls = {}
    for lane_key, (lane_id, stoves) in lane_info.items():
        # Re-fetch stoves to get current status
        stoves = get_stoves_for_lane(lane_id)
        lane_info[lane_key] = (lane_id, stoves)
        stove = run_stove_if_needed(lane_key, "DEVTRAIN", stoves, dry_run)
        if stove is True:
            continue
        if stove is False:
            return {lane_key: False}
        devtrain_polls[lane_key] = (stove["id"], "DEVTRAIN")

    if devtrain_polls:
        results = poll_multiple_stoves(devtrain_polls, poll_interval=60)
        for lane_key, status in results.items():
            if status != "completed":
                print(f"  [{lane_key}] DEVTRAIN FAILED: {status}")
                return {lane_key: False}

    # Step 4: Run TERMREP for all lanes (sequential — fast)
    for lane_key, (lane_id, stoves) in lane_info.items():
        stoves = get_stoves_for_lane(lane_id)
        lane_info[lane_key] = (lane_id, stoves)
        stove = run_stove_if_needed(lane_key, "TERMREP", stoves, dry_run)
        if stove is True:
            continue
        if stove is False:
            return {lane_key: False}
        status = poll_stove(stove["id"], "TERMREP")
        if status != "completed":
            return {lane_key: False}

    # Step 5: Run SNAPANALYSIS + SAEANALYSIS for all lanes (all concurrently)
    print(f"\n{'=' * 60}")
    print(f"PARALLEL ANALYSIS — SNAPANALYSIS + SAEANALYSIS × {len(lane_info)} lanes")
    print(f"{'=' * 60}")
    analysis_polls = {}
    for lane_key, (lane_id, stoves) in lane_info.items():
        stoves = get_stoves_for_lane(lane_id)
        lane_info[lane_key] = (lane_id, stoves)
        for stove_type in ["SNAPANALYSIS", "SAEANALYSIS"]:
            stove = run_stove_if_needed(lane_key, stove_type, stoves, dry_run)
            if stove is True:
                continue
            if stove is False:
                return {lane_key: False}
            label = f"{lane_key}"
            analysis_polls[f"{lane_key}/{stove_type}"] = (stove["id"], stove_type)

    if analysis_polls:
        results = poll_multiple_stoves(analysis_polls, poll_interval=30)
        for label, status in results.items():
            if status != "completed":
                lane_key = label.split("/")[0]
                print(f"  [{label}] FAILED: {status}")
                return {lane_key: False}

    return {k: True for k in lane_keys}


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ResNet-18 Seed-42 Curriculum Experiment",
    )
    parser.add_argument(
        "--lane", choices=["standard", "curriculum"],
        help="Run only a specific lane (default: both in parallel)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without executing",
    )
    args = parser.parse_args()

    lanes_to_run = [args.lane] if args.lane else LANE_ORDER

    print("=" * 60)
    print("ResNet-18 Seed-42 Curriculum Experiment")
    print(f"Lanes: {', '.join(LANE_CONFIGS[k]['name'] for k in lanes_to_run)}")
    print(f"Pipeline: {' → '.join(PIPELINE)} (no causal)")
    print(f"Mode: {'parallel' if len(lanes_to_run) > 1 else 'single lane'}")
    print(f"Snapshots: T=0, then every 5 epochs")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    # Find or create experiment
    print("\n[EXPERIMENT SETUP]")
    experiment = find_or_create_experiment()
    experiment_id = experiment["id"]

    # Ensure dataset is downloaded
    print("\n[DATASET]")
    ensure_dataset_downloaded(experiment_id)

    # Run lanes in parallel
    results = run_pipeline_parallel(experiment_id, lanes_to_run, dry_run=args.dry_run)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    for key in lanes_to_run:
        ok = results.get(key, False)
        status = "COMPLETED" if ok else "FAILED"
        print(f"  {LANE_CONFIGS[key]['name']}: {status}")
    print(f"\nFinished: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    sys.exit(0 if all(results.get(k, False) for k in lanes_to_run) else 1)


if __name__ == "__main__":
    main()
