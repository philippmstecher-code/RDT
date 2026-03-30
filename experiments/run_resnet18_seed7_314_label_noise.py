#!/usr/bin/env python3
"""
ResNet-18 Targeted Label Noise — multi-seed replication (seeds 7, 314).

Runs 7 conditions per seed (14 lanes total) sequentially:
  standard:         no noise (clean baseline)
  within_sc_p03:    p=0.3 flip within same superclass
  between_sc_p03:   p=0.3 flip to different superclass (key Ab-H→Di-H test)
  random_p03:       p=0.3 flip to any class (difficulty-matched control)
  within_sc_p01:    p=0.1 flip within same superclass
  between_sc_p01:   p=0.1 flip to different superclass (dose-response)
  random_p01:       p=0.1 flip to any class

Each seed gets its own experiment. Pipeline per lane: NETINIT → DEVTRAIN → SAEANALYSIS.

Usage:
    python runpodScripts/run_resnet18_seed7_314_label_noise.py
    python runpodScripts/run_resnet18_seed7_314_label_noise.py --dry-run
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

SEEDS = [7, 314]

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

# ── Lane definitions (same 7 conditions per seed) ─────────────────────
NOISE_CONDITIONS = {
    "standard":       {"noise_type": None,         "noise_prob": 0.0},
    "within_sc_p03":  {"noise_type": "within_sc",  "noise_prob": 0.3},
    "between_sc_p03": {"noise_type": "between_sc", "noise_prob": 0.3},
    "random_p03":     {"noise_type": "random",     "noise_prob": 0.3},
    "within_sc_p01":  {"noise_type": "within_sc",  "noise_prob": 0.1},
    "between_sc_p01": {"noise_type": "between_sc", "noise_prob": 0.1},
    "random_p01":     {"noise_type": "random",     "noise_prob": 0.1},
}


# ── NETINIT config builder ────────────────────────────────────────────
def build_netinit_config(seed, noise_type=None, noise_prob=0.0):
    milestones = [
        ep * BATCHES_PER_EPOCH
        for ep in range(SNAPSHOT_INTERVAL, TOTAL_EPOCHS + 1, SNAPSHOT_INTERVAL)
    ]

    config = {
        "network_type": "resnet18",
        "initialization_method": "kaiming_normal",
        "random_seed": seed,
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
            "frequency_value": 17,
            "min_batch_interval": 5,
            "validation_set_fraction": 1,
        },
        "memory_policy": {
            "disk_based_linear_probes": False,
            "disk_based_multilayer": False,
            "linear_probe_batch_size": 32,
            "activation_extraction_enabled": False,
            "activation_extraction_interval": 50,
        },
        "logging_policy": {
            "batch_log_frequency": 391,
            "batch_log_include_activations": False,
            "metrics_flush_interval": 391,
            "metrics_keep_recent": 50,
        },
        "sae_policy": {
            "expansion_factor": 4,
            "k_sparse": 32,
            "n_steps": 5000,
            "shared_init_seed": seed,
            "null_permutations": 100,
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


def run_lane_pipeline(lane_name, lane_id, seed):
    """Run the full pipeline for a single lane."""
    print(f"\n{'─' * 50}")
    print(f"LANE: {lane_name} (seed {seed})")
    print(f"{'─' * 50}")

    stoves = get_stoves_for_lane(lane_id)
    if not stoves:
        print(f"  ERROR: No stoves found for lane {lane_id}")
        return False

    # ── NETINIT
    netinit = stoves.get("NETINIT")
    if netinit and netinit["status"] != "completed":
        noise_cfg = NOISE_CONDITIONS[lane_name]
        config = build_netinit_config(seed, noise_cfg["noise_type"], noise_cfg["noise_prob"])
        result = api("PATCH", f"/api/stoves/{netinit['id']}/configuration", {"configuration": config})
        if result is None:
            print(f"  ERROR: Failed to configure NETINIT")
            return False
        noise_desc = f"type={noise_cfg['noise_type']}, p={noise_cfg['noise_prob']}"
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

    print(f"  [DONE] {lane_name} (seed {seed}) pipeline complete")
    return True


def setup_experiment(seed):
    """Find or create experiment for a given seed, return experiment_id."""
    experiment_name = f"ResNet18-seed{seed}-label-noise-30ep"

    experiments = api("GET", "/api/experiments")
    experiment_id = None
    if experiments:
        exp_list = experiments if isinstance(experiments, list) else experiments.get("experiments", experiments.get("data", []))
        for exp in exp_list:
            if exp.get("name") == experiment_name:
                experiment_id = exp["id"]
                print(f"\n[EXPERIMENT] Found existing: {experiment_name} ({experiment_id})")
                break

    if experiment_id is None:
        print(f"\n[EXPERIMENT] Creating: {experiment_name}")
        exp_data = api("POST", "/api/experiments", {
            "name": experiment_name,
            "dataset_id": DATASET_ID,
            "selected_classes": CIFAR100_CLASSES,
        })
        if not exp_data:
            sys.exit(f"Failed to create experiment {experiment_name}")
        experiment_id = exp_data.get("id") or exp_data.get("experiment_id")
        print(f"  Created experiment: {experiment_id}")

    # ── Ensure dataset is downloaded
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

    return experiment_id


def setup_lanes(experiment_id, seed):
    """Create lanes for a given seed, return {lane_name: lane_id}."""
    existing_lanes = get_existing_lanes(experiment_id)
    lane_ids = {}

    for lane_name in NOISE_CONDITIONS:
        full_name = f"ResNet18-noise-{lane_name}-seed{seed}"
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

    return lane_ids


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Run targeted label noise replication with seeds 7 & 314",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    args = parser.parse_args()

    total_lanes = len(SEEDS) * len(NOISE_CONDITIONS)
    print("=" * 60)
    print("ResNet-18 — TARGETED LABEL NOISE MULTI-SEED REPLICATION")
    print(f"Seeds: {SEEDS}")
    print(f"Epochs: {TOTAL_EPOCHS}")
    print(f"Conditions per seed ({len(NOISE_CONDITIONS)}):")
    for name, cfg in NOISE_CONDITIONS.items():
        print(f"  {name:<20} type={cfg['noise_type']}, p={cfg['noise_prob']}")
    print(f"Total lanes: {total_lanes}")
    print(f"Pipeline: {' → '.join(PIPELINE)}")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] NETINIT configs:")
        for seed in SEEDS:
            for name, cfg in NOISE_CONDITIONS.items():
                config = build_netinit_config(seed, cfg["noise_type"], cfg["noise_prob"])
                print(f"\n--- seed {seed} / {name} ---")
                print(json.dumps(config, indent=2))
        return

    # ── Run all seeds sequentially ───────────────────────────────────
    all_results = {}
    for seed in SEEDS:
        print(f"\n{'═' * 60}")
        print(f"  SEED {seed}")
        print(f"{'═' * 60}")

        experiment_id = setup_experiment(seed)
        lane_ids = setup_lanes(experiment_id, seed)

        for lane_name in NOISE_CONDITIONS:
            key = f"seed{seed}/{lane_name}"
            success = run_lane_pipeline(lane_name, lane_ids[lane_name], seed)
            all_results[key] = "COMPLETED" if success else "FAILED"

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("MULTI-SEED REPLICATION COMPLETE")
    for key, status in all_results.items():
        print(f"  {key:<40} {status}")
    print(f"Finished: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    if any(s == "FAILED" for s in all_results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
