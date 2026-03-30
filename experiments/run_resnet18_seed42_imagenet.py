#!/usr/bin/env python3
"""
ResNet-18 Seed 42 — Tiny ImageNet End-to-End Pipeline Orchestrator.

Runs the full stove pipeline (NETINIT → DEVTRAIN → TERMREP → SNAPANALYSIS + SAEANALYSIS)
for a single ResNet-18 lane on the Tiny ImageNet dataset with optimized NETINIT configuration.

Supports resume: completed stoves are skipped automatically.

Usage:
    python runpodScripts/run_resnet18_seed42_imagenet.py
    python runpodScripts/run_resnet18_seed42_imagenet.py --dry-run
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

EXPERIMENT_NAME = "ResNet18-TinyImageNet-seed42"
EXPERIMENT_DESCRIPTION = (
    "ResNet-18 trained on Tiny ImageNet (200 classes) with seed 42, "
    "80 epochs, optimized NETINIT (Kaiming Normal + SGD with cosine-friendly LR)."
)
DATASET_ID = "tiny_imagenet"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = DATA_DIR / "experiments"
WORKSPACE_VOLUME = Path(os.environ.get("RUNPOD_WORKSPACE_VOLUME", "/workspace"))

# Pipeline order. SNAPANALYSIS and SAEANALYSIS run in parallel (same order_index).
PIPELINE = ["NETINIT", "DEVTRAIN", "TERMREP", "SNAPANALYSIS", "SAEANALYSIS"]

# ── All 200 Tiny ImageNet classes (prefixed with dataset ID) ──────────
TINY_IMAGENET_CLASSES = [
    # Dogs
    "tiny_imagenet_Chihuahua", "tiny_imagenet_Yorkshire_terrier",
    "tiny_imagenet_golden_retriever", "tiny_imagenet_Labrador_retriever",
    "tiny_imagenet_German_shepherd", "tiny_imagenet_standard_poodle",
    # Cats
    "tiny_imagenet_tabby", "tiny_imagenet_Egyptian_cat", "tiny_imagenet_Persian_cat",
    # Large mammals
    "tiny_imagenet_brown_bear", "tiny_imagenet_cougar", "tiny_imagenet_lion",
    "tiny_imagenet_hog", "tiny_imagenet_ox", "tiny_imagenet_bison",
    "tiny_imagenet_bighorn", "tiny_imagenet_gazelle", "tiny_imagenet_Arabian_camel",
    "tiny_imagenet_African_elephant",
    # Primates
    "tiny_imagenet_baboon", "tiny_imagenet_chimpanzee", "tiny_imagenet_orangutan",
    # Small mammals
    "tiny_imagenet_guinea_pig", "tiny_imagenet_lesser_panda", "tiny_imagenet_koala",
    # Birds
    "tiny_imagenet_goose", "tiny_imagenet_albatross", "tiny_imagenet_king_penguin",
    "tiny_imagenet_black_stork",
    # Reptiles & amphibians
    "tiny_imagenet_bullfrog", "tiny_imagenet_tailed_frog",
    "tiny_imagenet_European_fire_salamander", "tiny_imagenet_American_alligator",
    "tiny_imagenet_boa_constrictor",
    # Aquatic creatures
    "tiny_imagenet_goldfish", "tiny_imagenet_jellyfish", "tiny_imagenet_brain_coral",
    "tiny_imagenet_sea_slug", "tiny_imagenet_sea_cucumber", "tiny_imagenet_dugong",
    "tiny_imagenet_American_lobster", "tiny_imagenet_spiny_lobster",
    # Insects
    "tiny_imagenet_monarch", "tiny_imagenet_sulphur_butterfly", "tiny_imagenet_ladybug",
    "tiny_imagenet_dragonfly", "tiny_imagenet_bee", "tiny_imagenet_cockroach",
    "tiny_imagenet_mantis", "tiny_imagenet_fly", "tiny_imagenet_grasshopper",
    "tiny_imagenet_walking_stick",
    # Arachnids & myriapods
    "tiny_imagenet_tarantula", "tiny_imagenet_black_widow", "tiny_imagenet_scorpion",
    "tiny_imagenet_centipede",
    # Invertebrates
    "tiny_imagenet_snail", "tiny_imagenet_slug", "tiny_imagenet_trilobite",
    # Vehicles
    "tiny_imagenet_school_bus", "tiny_imagenet_sports_car", "tiny_imagenet_moving_van",
    "tiny_imagenet_bullet_train", "tiny_imagenet_trolleybus", "tiny_imagenet_freight_car",
    "tiny_imagenet_go-kart", "tiny_imagenet_police_van", "tiny_imagenet_limousine",
    "tiny_imagenet_convertible", "tiny_imagenet_beach_wagon", "tiny_imagenet_tractor",
    "tiny_imagenet_jinrikisha", "tiny_imagenet_gondola", "tiny_imagenet_lifeboat",
    # Clothing
    "tiny_imagenet_academic_gown", "tiny_imagenet_apron", "tiny_imagenet_bikini",
    "tiny_imagenet_bow_tie", "tiny_imagenet_cardigan", "tiny_imagenet_fur_coat",
    "tiny_imagenet_kimono", "tiny_imagenet_military_uniform", "tiny_imagenet_miniskirt",
    "tiny_imagenet_poncho", "tiny_imagenet_sandal", "tiny_imagenet_sock",
    "tiny_imagenet_sombrero", "tiny_imagenet_sunglasses", "tiny_imagenet_swimming_trunks",
    "tiny_imagenet_vestment",
    # Food
    "tiny_imagenet_espresso", "tiny_imagenet_pizza", "tiny_imagenet_potpie",
    "tiny_imagenet_ice_cream", "tiny_imagenet_pretzel", "tiny_imagenet_guacamole",
    "tiny_imagenet_ice_lolly", "tiny_imagenet_mashed_potato", "tiny_imagenet_meat_loaf",
    "tiny_imagenet_lemon", "tiny_imagenet_banana", "tiny_imagenet_orange",
    "tiny_imagenet_bell_pepper", "tiny_imagenet_pomegranate", "tiny_imagenet_mushroom",
    "tiny_imagenet_cauliflower", "tiny_imagenet_acorn",
    # Kitchen & tableware
    "tiny_imagenet_frying_pan", "tiny_imagenet_wok", "tiny_imagenet_plate",
    "tiny_imagenet_wooden_spoon", "tiny_imagenet_teapot",
    # Containers
    "tiny_imagenet_beer_bottle", "tiny_imagenet_pop_bottle", "tiny_imagenet_water_jug",
    "tiny_imagenet_barrel", "tiny_imagenet_pill_bottle", "tiny_imagenet_beaker",
    "tiny_imagenet_bucket",
    # Furniture
    "tiny_imagenet_rocking_chair", "tiny_imagenet_dining_table", "tiny_imagenet_desk",
    "tiny_imagenet_chest",
    # Household
    "tiny_imagenet_bathtub", "tiny_imagenet_refrigerator", "tiny_imagenet_lampshade",
    "tiny_imagenet_plunger", "tiny_imagenet_broom", "tiny_imagenet_candle",
    "tiny_imagenet_torch", "tiny_imagenet_bannister", "tiny_imagenet_teddy",
    # Electronics
    "tiny_imagenet_computer_keyboard", "tiny_imagenet_remote_control", "tiny_imagenet_iPod",
    "tiny_imagenet_CD_player", "tiny_imagenet_cash_machine", "tiny_imagenet_pay-phone",
    "tiny_imagenet_space_heater",
    # Tools & equipment
    "tiny_imagenet_nail", "tiny_imagenet_chain", "tiny_imagenet_pole", "tiny_imagenet_crane",
    "tiny_imagenet_potter's_wheel", "tiny_imagenet_sewing_machine",
    "tiny_imagenet_lawn_mower", "tiny_imagenet_reel",
    # Instruments & measures
    "tiny_imagenet_abacus", "tiny_imagenet_binoculars", "tiny_imagenet_hourglass",
    "tiny_imagenet_magnetic_compass", "tiny_imagenet_stopwatch",
    # Musical instruments
    "tiny_imagenet_organ", "tiny_imagenet_oboe", "tiny_imagenet_drumstick",
    "tiny_imagenet_brass",
    # Sports & recreation
    "tiny_imagenet_volleyball", "tiny_imagenet_basketball", "tiny_imagenet_rugby_ball",
    "tiny_imagenet_punching_bag", "tiny_imagenet_dumbbell", "tiny_imagenet_scoreboard",
    "tiny_imagenet_snorkel",
    # Structures
    "tiny_imagenet_barn", "tiny_imagenet_dam", "tiny_imagenet_triumphal_arch",
    "tiny_imagenet_viaduct", "tiny_imagenet_steel_arch_bridge",
    "tiny_imagenet_suspension_bridge", "tiny_imagenet_picket_fence", "tiny_imagenet_thatch",
    "tiny_imagenet_cliff_dwelling", "tiny_imagenet_obelisk", "tiny_imagenet_altar",
    "tiny_imagenet_beacon", "tiny_imagenet_water_tower", "tiny_imagenet_birdhouse",
    "tiny_imagenet_maypole", "tiny_imagenet_flagpole", "tiny_imagenet_fountain",
    "tiny_imagenet_barbershop", "tiny_imagenet_confectionery", "tiny_imagenet_butcher_shop",
    "tiny_imagenet_turnstile", "tiny_imagenet_parking_meter",
    # Landscapes
    "tiny_imagenet_cliff", "tiny_imagenet_coral_reef", "tiny_imagenet_seashore",
    "tiny_imagenet_lakeside", "tiny_imagenet_alp",
    # Accessories & gear
    "tiny_imagenet_backpack", "tiny_imagenet_Christmas_stocking", "tiny_imagenet_gasmask",
    "tiny_imagenet_neck_brace", "tiny_imagenet_umbrella", "tiny_imagenet_syringe",
    # Miscellaneous
    "tiny_imagenet_spider_web", "tiny_imagenet_comic_book", "tiny_imagenet_projectile",
    "tiny_imagenet_cannon",
]

# ── Optimized NETINIT Configuration for Tiny ImageNet ─────────────────
NETINIT_CONFIG = {
    "network_type": "resnet18",
    "initialization_method": "kaiming_normal",
    "random_seed": 42,
    "batch_size": 128,
    "optimizer": {
        "type": "sgd",
        "learning_rate": 0.1,
        "weight_decay": 5e-4,
        "momentum": 0.9,
        "nesterov": True,
    },
    "training_policy": {
        "epochs": 100,
        "accuracy_threshold": 75,
    },
    "snapshot_policy": {
        "milestone_count": 10,
        "distribution_scheme": "uniform",
        "samples_per_class": 100,
        "terminal_capture": "both",
    },
    "transform_config": {
        # ImageNet normalization stats (standard for Tiny ImageNet)
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "resize": 64,  # Tiny ImageNet native resolution (64x64)
        "augmentation_enabled": True,
        "random_crop_padding": 8,
        "horizontal_flip": True,
        "color_jitter_brightness": 0.2,
        "color_jitter_contrast": 0.2,
        "color_jitter_saturation": 0.2,
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

LANE_NAME = "ResNet18-200class-10ms-seed42"


# ── Helpers ────────────────────────────────────────────────────────────
def api(method, path, json_body=None, timeout=30):
    """Make an API request using stdlib. Returns parsed JSON or None on failure."""
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
    """Find existing experiment by name or create a new one. Returns experiment dict."""
    # List all experiments and look for ours
    experiments = api("GET", "/api/experiments")
    if experiments:
        for exp in experiments:
            if exp.get("name") == EXPERIMENT_NAME:
                print(f"  Found existing experiment: {exp['id']}")
                return exp

    # Create new experiment with all 200 Tiny ImageNet classes
    print(f"  Creating experiment: {EXPERIMENT_NAME}")
    exp = api("POST", "/api/experiments", {
        "name": EXPERIMENT_NAME,
        "dataset_id": DATASET_ID,
        "selected_classes": TINY_IMAGENET_CLASSES,
        "description": EXPERIMENT_DESCRIPTION,
    })
    if not exp:
        sys.exit("Failed to create experiment")
    print(f"  Created experiment: {exp['id']}")
    return exp


def ensure_dataset_downloaded(experiment_id):
    """Check dataset download status and trigger if needed. Polls until complete."""
    status_resp = api("GET", f"/api/experiments/{experiment_id}/datasets/status")

    if status_resp and status_resp.get("download_status") == "completed":
        print("  Dataset: already downloaded")
        return

    # Trigger download
    print("  Triggering dataset download...")
    result = api("POST", f"/api/experiments/{experiment_id}/datasets/download")
    if result is None:
        sys.exit("Failed to trigger dataset download")

    # Poll until complete
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
            error_msg = status_resp.get("error_message", "unknown error")
            sys.exit(f"Dataset download failed: {error_msg}")

        print(f"  [{now_ts()}] Dataset download: {dl_status} ({progress * 100:.0f}%)")


# ── Lane Discovery / Creation ─────────────────────────────────────────
def get_existing_lanes(experiment_id):
    """Fetch lanes already created for this experiment."""
    data = api("GET", f"/api/lanes/experiment/{experiment_id}")
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


def create_lane(experiment_id):
    """Create a lane via the API. Returns the lane dict."""
    data = api("POST", "/api/lanes", {
        "experiment_id": experiment_id,
        "name": LANE_NAME,
    })
    if not data:
        sys.exit(f"Failed to create lane {LANE_NAME}")
    lane_id = data.get("id") or data.get("lane_id")
    print(f"  Created lane: {LANE_NAME} ({lane_id})")
    return data


def configure_netinit(stove_id):
    """Configure a NETINIT stove with the optimized hyperparameters."""
    data = api("PATCH", f"/api/stoves/{stove_id}/configuration", {"configuration": NETINIT_CONFIG})
    if data is None:
        sys.exit(f"Failed to configure NETINIT stove {stove_id}")
    print(f"  Configured NETINIT: resnet18, seed=42, epochs=80, Tiny ImageNet")
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


def poll_stove(stove_id, stove_type, poll_interval=30, timeout=72000):
    """Poll a stove until completed or error. Returns final status."""
    start = time.time()
    last_progress = -1

    while time.time() - start < timeout:
        data = api("GET", f"/api/stoves/{stove_id}")
        if not data:
            print(f"  [{now_ts()}] WARNING: Could not fetch stove status, retrying...")
            time.sleep(poll_interval)
            continue

        stove_status = data.get("status", "unknown")
        progress = data.get("progress", 0.0)
        pct = progress * 100 if isinstance(progress, (int, float)) else 0

        if stove_status == "completed":
            print(f"  [{now_ts()}] {stove_type}: COMPLETED")
            return "completed"
        if stove_status == "error":
            print(f"  [{now_ts()}] {stove_type}: ERROR")
            return "error"

        # Only log when progress changes meaningfully
        if int(pct) != last_progress:
            print(f"  [{now_ts()}] {stove_type}: {stove_status} ({pct:.1f}%)")
            last_progress = int(pct)

        # Adaptive polling: faster for quick stoves, slower for training
        if stove_type == "DEVTRAIN":
            time.sleep(max(poll_interval, 60))
        elif stove_type == "NETINIT":
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
def run_pipeline(dry_run=False):
    """Run the full end-to-end pipeline."""

    print(f"\n{'=' * 60}")
    print("SETUP: Experiment & Dataset")
    print(f"{'=' * 60}")

    # Step 1: Find or create experiment
    experiment = find_or_create_experiment()
    experiment_id = experiment["id"]

    # Step 2: Ensure dataset is downloaded
    if dry_run:
        print("  [DRY RUN] Would ensure dataset is downloaded")
    else:
        ensure_dataset_downloaded(experiment_id)

    # Step 3: Find or create lane
    print(f"\n{'=' * 60}")
    print(f"LANE: {LANE_NAME}")
    print(f"{'=' * 60}")

    existing = get_existing_lanes(experiment_id)
    if LANE_NAME in existing:
        lane = existing[LANE_NAME]
        lane_id = lane["id"]
        print(f"  Found existing lane: {lane_id}")
    else:
        if dry_run:
            print(f"  [DRY RUN] Would create lane: {LANE_NAME}")
            return True
        lane = create_lane(experiment_id)
        lane_id = lane.get("id") or lane.get("lane_id")

    # Get stoves
    stoves = get_stoves_for_lane(lane_id)
    if not stoves:
        sys.exit(f"No stoves found for lane {lane_id}")

    # Configure NETINIT if not yet configured/completed
    netinit = stoves.get("NETINIT")
    if netinit and netinit["status"] not in ("completed", "running"):
        if dry_run:
            print("  [DRY RUN] Would configure NETINIT")
        else:
            configure_netinit(netinit["id"])

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

        final_status = poll_stove(stove["id"], stove_type)
        if final_status != "completed":
            print(f"  PIPELINE FAILED at {stove_type} (status={final_status})")
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
        final_status = poll_stove(stove["id"], stove_type)
        if final_status != "completed":
            print(f"  PIPELINE FAILED at {stove_type} (status={final_status})")
            return False

    print(f"\n  Lane {LANE_NAME}: ALL STOVES COMPLETED")

    # Archive to workspace volume
    if not archive_lane_to_workspace(experiment_id, lane_id):
        print("  WARNING: Archive failed — lane data remains on working volume")

    return True


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ResNet-18 Seed 42 — Tiny ImageNet End-to-End Pipeline",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without executing",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ResNet-18 Seed 42 — Tiny ImageNet Pipeline")
    print("=" * 60)
    print(f"  Dataset:       Tiny ImageNet (200 classes)")
    print(f"  Network:       ResNet-18")
    print(f"  Seed:          42")
    print(f"  Epochs:        100")
    print(f"  Optimizer:     SGD (lr=0.1, wd=5e-4, momentum=0.9, nesterov)")
    print(f"  Scheduler:     CosineAnnealingLR (built-in)")
    print(f"  Init:          Kaiming Normal (optimized NETINIT)")
    print(f"  Augmentation:  RandomCrop(64,pad=8) + HFlip + ColorJitter")
    print(f"  Batch size:    128")
    print(f"  Milestones:    10 uniform snapshots")
    print(f"  Pipeline:      {' -> '.join(PIPELINE)}")
    print(f"  Started:       {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    # Verify backend is reachable
    health = api("GET", "/api/experiments")
    if health is None:
        sys.exit(
            "Backend not reachable. Ensure the backend is running at "
            f"{API_BASE} before starting the pipeline."
        )

    success = run_pipeline(dry_run=args.dry_run)

    print(f"\n{'=' * 60}")
    print("RESULT")
    print("=" * 60)
    result_status = "COMPLETED" if success else "FAILED"
    print(f"  {LANE_NAME}: {result_status}")
    print(f"\nFinished: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
