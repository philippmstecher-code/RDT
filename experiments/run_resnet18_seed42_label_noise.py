#!/usr/bin/env python3
"""
ResNet-18 Seed-42 Targeted Label Noise Experiment.

Creates a new experiment with 7 lanes testing how structured label noise
affects Ab-H (superclass-selective features) and Di-H (fine-class differentiation):

  standard:        No noise (baseline)
  within_sc_p01:   p=0.1 flip within same superclass
  within_sc_p03:   p=0.3 flip within same superclass
  between_sc_p01:  p=0.1 flip to different superclass
  between_sc_p03:  p=0.3 flip to different superclass
  random_p01:      p=0.1 flip to any class
  random_p03:      p=0.3 flip to any class

Pipeline per lane: NETINIT → DEVTRAIN → TERMREP → SNAPANALYSIS → SAEANALYSIS.

Usage:
    python runpodScripts/run_resnet18_seed42_label_noise.py
    python runpodScripts/run_resnet18_seed42_label_noise.py --dry-run
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

EXPERIMENT_NAME = "ResNet18-seed42-label-noise-30ep"

DATASET_ID = "cifar100"
TOTAL_EPOCHS = 30
BATCHES_PER_EPOCH = 391  # CIFAR-100 with batch_size=128
SNAPSHOT_INTERVAL = 5

PIPELINE = ["NETINIT", "DEVTRAIN", "SAEANALYSIS"]

# All 100 CIFAR-100 fine classes
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

# ── Lane definitions ───────────────────────────────────────────────────
LANES = {
    "standard": {"noise_type": None, "noise_prob": 0.0},
    # p=0.3 first (stronger signal)
    "within_sc_p03": {"noise_type": "within_sc", "noise_prob": 0.3},
    "between_sc_p03": {"noise_type": "between_sc", "noise_prob": 0.3},
    "random_p03": {"noise_type": "random", "noise_prob": 0.3},
    # p=0.1 (dose-response)
    "within_sc_p01": {"noise_type": "within_sc", "noise_prob": 0.1},
    "between_sc_p01": {"noise_type": "between_sc", "noise_prob": 0.1},
    "random_p01": {"noise_type": "random", "noise_prob": 0.1},
}


# ── NETINIT config builder ────────────────────────────────────────────
def build_netinit_config(noise_type=None, noise_prob=0.0):
    milestones = [
        ep * BATCHES_PER_EPOCH
        for ep in range(SNAPSHOT_INTERVAL, TOTAL_EPOCHS + 1, SNAPSHOT_INTERVAL)
    ]

    config = {
        "network_type": "resnet18",
        "initialization_method": "kaiming_normal",
        "random_seed": 42,
        "batch_size": 128,
        "training_policy": {
            "epochs": TOTAL_EPOCHS,
            "accuracy_threshold": None,
        },
        "snapshot_policy": {
            "milestone_count": len(milestones),
            "milestone_type": "weight_updates",
            "milestone_weight_updates": milestones,
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
            "individual_multilayer_activations",
        ],
        "optimizer": {
            "type": "sgd",
            "learning_rate": 0.1,
            "weight_decay": 5e-4,
            "momentum": 0.9,
            "nesterov": True,
        },
    }

    if noise_type is not None:
        config["noise_policy"] = {
            "enabled": True,
            "noise_type": noise_type,
            "noise_prob": noise_prob,
        }

    return config


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


# ── Stove Helpers ────────────────────────────────────────────────────
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


def poll_stove(stove_id, stove_type, poll_interval=30, timeout=36000):
    start = time.time()
    last_progress = -1
    while time.time() - start < timeout:
        data = api("GET", f"/api/stoves/{stove_id}")
        if not data:
            time.sleep(poll_interval)
            continue
        st = data.get("status", "unknown")
        progress = data.get("progress", 0.0)
        pct = progress * 100 if isinstance(progress, (int, float)) else 0
        if st == "completed":
            print(f"  [{now_ts()}] {stove_type}: COMPLETED")
            return "completed"
        if st == "error":
            print(f"  [{now_ts()}] {stove_type}: ERROR")
            return "error"
        if int(pct) != last_progress:
            print(f"  [{now_ts()}] {stove_type}: {st} ({pct:.1f}%)")
            last_progress = int(pct)
        if stove_type == "DEVTRAIN":
            time.sleep(max(poll_interval, 60))
        elif stove_type == "NETINIT":
            time.sleep(5)
        else:
            time.sleep(poll_interval)
    print(f"  [{now_ts()}] {stove_type}: TIMEOUT after {timeout}s")
    return "timeout"


def poll_multiple_stoves(stove_map, poll_interval=30, timeout=36000):
    """Poll multiple stoves concurrently. stove_map: {label: (stove_id, stove_type)}."""
    start = time.time()
    remaining = dict(stove_map)
    results = {}
    last_progress = {label: -1 for label in remaining}

    while remaining and (time.time() - start < timeout):
        for label, (stove_id, stove_type) in list(remaining.items()):
            data = api("GET", f"/api/stoves/{stove_id}")
            if not data:
                continue
            st = data.get("status", "unknown")
            progress = data.get("progress", 0.0)
            pct = progress * 100 if isinstance(progress, (int, float)) else 0

            if st == "completed":
                print(f"  [{now_ts()}] {label}/{stove_type}: COMPLETED")
                results[label] = "completed"
                del remaining[label]
            elif st == "error":
                print(f"  [{now_ts()}] {label}/{stove_type}: ERROR")
                results[label] = "error"
                del remaining[label]
            elif int(pct) != last_progress[label]:
                print(f"  [{now_ts()}] {label}/{stove_type}: {st} ({pct:.1f}%)")
                last_progress[label] = int(pct)

        if remaining:
            time.sleep(poll_interval if len(remaining) == 1 else max(poll_interval, 60))

    for label in remaining:
        results[label] = "timeout"
    return results


def start_stove(stove_type, stove_id):
    endpoints = {
        "NETINIT":      ("POST", f"/api/netinit/initialize/{stove_id}"),
        "DEVTRAIN":     ("POST", "/api/devtrain/start"),
        "TERMREP":      ("POST", "/api/termrep/start"),
        "SNAPANALYSIS": ("POST", "/api/snapanalysis/start"),
        "SAEANALYSIS":  ("POST", "/api/saeanalysis-stove/start"),
    }
    method, path = endpoints[stove_type]
    body = {"stove_id": stove_id} if stove_type != "NETINIT" else None
    return api(method, path, body)


def run_lane_pipeline(lane_name, lane_id):
    """Run the full pipeline for a single lane."""
    print(f"\n{'─' * 50}")
    print(f"LANE: {lane_name}")
    print(f"{'─' * 50}")

    stoves = get_stoves_for_lane(lane_id)
    if not stoves:
        print(f"  ERROR: No stoves found for lane {lane_id}")
        return False

    # ── NETINIT
    netinit = stoves.get("NETINIT")
    if netinit and netinit["status"] != "completed":
        noise_cfg = LANES[lane_name]
        config = build_netinit_config(noise_cfg["noise_type"], noise_cfg["noise_prob"])
        result = api("PATCH", f"/api/stoves/{netinit['id']}/configuration", {"configuration": config})
        if result is None:
            print(f"  ERROR: Failed to configure NETINIT")
            return False
        noise_desc = f"type={noise_cfg['noise_type']}, p={noise_cfg['noise_prob']}" if noise_cfg["noise_type"] else "none"
        print(f"  [NETINIT] Configured (noise: {noise_desc})")

        start_stove("NETINIT", netinit["id"])
        st = poll_stove(netinit["id"], "NETINIT")
        if st != "completed":
            print(f"  ERROR: NETINIT failed: {st}")
            return False
    else:
        print(f"  [NETINIT] Already completed")

    # ── DEVTRAIN
    stoves = get_stoves_for_lane(lane_id)
    devtrain = stoves.get("DEVTRAIN")
    if devtrain["status"] != "completed":
        print(f"  [DEVTRAIN] Starting ({TOTAL_EPOCHS} epochs)...")
        start_stove("DEVTRAIN", devtrain["id"])
        st = poll_stove(devtrain["id"], "DEVTRAIN", poll_interval=60)
        if st != "completed":
            print(f"  ERROR: DEVTRAIN failed: {st}")
            return False
    else:
        print(f"  [DEVTRAIN] Already completed")

    # ── SAEANALYSIS
    stoves = get_stoves_for_lane(lane_id)
    saeanalysis = stoves.get("SAEANALYSIS")
    if saeanalysis and saeanalysis["status"] != "completed":
        print(f"  [SAEANALYSIS] Starting...")
        start_stove("SAEANALYSIS", saeanalysis["id"])
        st = poll_stove(saeanalysis["id"], "SAEANALYSIS")
        if st != "completed":
            print(f"  ERROR: SAEANALYSIS failed: {st}")
            return False
    elif saeanalysis:
        print(f"  [SAEANALYSIS] Already completed")

    print(f"  [DONE] {lane_name} pipeline complete")
    return True


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Run targeted label noise experiment on ResNet-18 Seed-42",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    args = parser.parse_args()

    print("=" * 60)
    print("ResNet-18 Seed-42 — TARGETED LABEL NOISE EXPERIMENT")
    print(f"Epochs: {TOTAL_EPOCHS}")
    print(f"Lanes ({len(LANES)}):")
    for name, cfg in LANES.items():
        noise_desc = f"type={cfg['noise_type']}, p={cfg['noise_prob']}" if cfg["noise_type"] else "no noise"
        print(f"  {name:<20} {noise_desc}")
    print(f"Pipeline: {' → '.join(PIPELINE)}")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] NETINIT configs:")
        for name, cfg in LANES.items():
            config = build_netinit_config(cfg["noise_type"], cfg["noise_prob"])
            print(f"\n--- {name} ---")
            print(json.dumps(config, indent=2))
        return

    # ── Create experiment ─────────────────────────────────────────────
    # Check if experiment already exists by listing experiments
    experiments = api("GET", "/api/experiments")
    experiment_id = None
    if experiments:
        exp_list = experiments if isinstance(experiments, list) else experiments.get("experiments", experiments.get("data", []))
        for exp in exp_list:
            if exp.get("name") == EXPERIMENT_NAME:
                experiment_id = exp["id"]
                print(f"\n[EXPERIMENT] Found existing: {experiment_id}")
                break

    if experiment_id is None:
        print(f"\n[EXPERIMENT] Creating: {EXPERIMENT_NAME}")
        exp_data = api("POST", "/api/experiments", {
            "name": EXPERIMENT_NAME,
            "dataset_id": DATASET_ID,
            "selected_classes": CIFAR100_CLASSES,
        })
        if not exp_data:
            sys.exit("Failed to create experiment")
        experiment_id = exp_data.get("id") or exp_data.get("experiment_id")
        print(f"  Created experiment: {experiment_id}")

    # ── Ensure dataset is downloaded ─────────────────────────────────
    ds_status = api("GET", f"/api/experiments/{experiment_id}/datasets/status")
    if not ds_status or ds_status.get("download_status") != "completed":
        print("[DATASET] Triggering download...")
        api("POST", f"/api/experiments/{experiment_id}/datasets/download")
        for _ in range(120):
            time.sleep(5)
            ds_status = api("GET", f"/api/experiments/{experiment_id}/datasets/status")
            if ds_status and ds_status.get("download_status") == "completed":
                break
            pct = ds_status.get("progress", 0) if ds_status else 0
            print(f"  [{now_ts()}] Downloading dataset: {pct:.0f}%")
        else:
            sys.exit("Dataset download timed out")
        print("[DATASET] Download complete")

    # ── Create lanes ──────────────────────────────────────────────────
    existing_lanes = get_existing_lanes(experiment_id)
    lane_ids = {}

    for lane_name in LANES:
        full_name = f"ResNet18-noise-{lane_name}-seed42"
        if full_name in existing_lanes:
            lane_ids[lane_name] = existing_lanes[full_name]["id"]
            print(f"[LANE] Found existing: {full_name}")
        else:
            print(f"[LANE] Creating: {full_name}")
            lane_data = api("POST", "/api/lanes", {
                "experiment_id": experiment_id,
                "name": full_name,
            })
            if not lane_data:
                sys.exit(f"Failed to create lane {full_name}")
            lane_ids[lane_name] = lane_data.get("id") or lane_data.get("lane_id")
            print(f"  Created lane: {lane_ids[lane_name]}")

    # ── Run pipelines sequentially ────────────────────────────────────
    results = {}
    for lane_name in LANES:
        success = run_lane_pipeline(lane_name, lane_ids[lane_name])
        results[lane_name] = "COMPLETED" if success else "FAILED"

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("EXPERIMENT COMPLETE")
    print(f"Experiment: {EXPERIMENT_NAME} ({experiment_id})")
    for lane_name, status in results.items():
        print(f"  {lane_name:<20} {status}")
    print(f"Finished: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    if any(s == "FAILED" for s in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
