#!/usr/bin/env python3
"""
Standalone pipeline runner for training reproduction.

Replaces the FastAPI backend with a simple CLI that runs the 3-stage
pipeline: NETINIT -> DEVTRAIN -> SAEANALYSIS.

Usage:
    python run_pipeline.py --config configs/standard/resnet18_cifar100_seed42.json
    python run_pipeline.py --config configs/standard/resnet18_cifar100_seed42.json --output-dir ./output --dry-run
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Make src/ importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
# PydanticLikeNamespace — SimpleNamespace with .model_dump() / .get()
# ---------------------------------------------------------------------------
class PydanticLikeNamespace(SimpleNamespace):
    """Drop-in replacement for Pydantic models used by train_with_snapshots.

    Provides attribute access (config.network_type) plus the .model_dump()
    and .get() methods that the training code expects.
    """

    def model_dump(self):
        """Return a plain dict, recursively converting nested namespaces."""
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, PydanticLikeNamespace):
                out[k] = v.model_dump()
            elif isinstance(v, list):
                out[k] = [
                    item.model_dump() if isinstance(item, PydanticLikeNamespace) else item
                    for item in v
                ]
            else:
                out[k] = v
        return out

    def get(self, key, default=None):
        return getattr(self, key, default)


def dict_to_namespace(d):
    """Recursively convert a dict (or None) to PydanticLikeNamespace."""
    if d is None:
        return None
    if not isinstance(d, dict):
        return d
    return PydanticLikeNamespace(
        **{k: dict_to_namespace(v) for k, v in d.items()}
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DATASET_NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "tiny_imagenet": 200,
}

DATASET_ALL_CLASSES = {
    # Will be populated at runtime by loading the actual dataset metadata
}


def _resolve_num_classes(dataset_id: str) -> int:
    n = DATASET_NUM_CLASSES.get(dataset_id)
    if n is None:
        raise ValueError(
            f"Unknown dataset_id '{dataset_id}'. "
            f"Supported: {list(DATASET_NUM_CLASSES.keys())}"
        )
    return n


def _resolve_selected_classes(dataset_id: str) -> list:
    """Return the full class name list for the dataset.

    For CIFAR datasets we use torchvision to get canonical class names.
    """
    import torchvision

    if dataset_id == "cifar10":
        ds = torchvision.datasets.CIFAR10(root="/tmp/cifar10_meta", train=True, download=True)
        return list(ds.classes)
    elif dataset_id == "cifar100":
        ds = torchvision.datasets.CIFAR100(root="/tmp/cifar100_meta", train=True, download=True)
        return list(ds.classes)
    elif dataset_id == "tiny_imagenet":
        # Tiny ImageNet class list must come from the downloaded dataset
        raise NotImplementedError(
            "Tiny ImageNet class resolution requires a pre-downloaded dataset. "
            "Please add a 'selected_classes' key to your config JSON."
        )
    else:
        raise ValueError(f"Cannot auto-resolve classes for dataset '{dataset_id}'")


def _log_stage(stage: str, msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{stage}] {msg}")


def _log_separator(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Config loading & validation
# ---------------------------------------------------------------------------
REQUIRED_KEYS = [
    "experiment_name",
    "network_type",
    "dataset_id",
    "random_seed",
    "training_policy",
    "snapshot_policy",
    "optimizer",
]


def load_config(config_path: str) -> dict:
    """Load and validate the JSON config file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        cfg = json.load(f)

    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    return cfg


def _enrich_config(cfg: dict) -> dict:
    """Add defaults that the training code expects but the JSON may omit."""
    cfg.setdefault("batch_size", 128)
    cfg.setdefault("representation_methods", ["mean_activations"])
    cfg.setdefault("sae_policy", None)
    cfg.setdefault("curriculum_policy", None)
    cfg.setdefault("noise_policy", None)
    cfg.setdefault("transform_config", None)

    # snapshot_policy defaults
    sp = cfg.get("snapshot_policy", {}) or {}
    sp.setdefault("terminal_capture", "at_threshold")
    sp.setdefault("milestone_type", "accuracy")
    sp.setdefault("milestone_accuracies", None)
    sp.setdefault("milestone_weight_updates", None)
    cfg["snapshot_policy"] = sp

    # training_policy defaults
    tp = cfg.get("training_policy", {}) or {}
    tp.setdefault("epochs", 100)
    tp.setdefault("accuracy_threshold", None)
    cfg["training_policy"] = tp

    return cfg


# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------
def create_directory_structure(output_dir: str, experiment_name: str) -> dict:
    """Create the lane directory structure and return path dict."""
    base = Path(output_dir) / experiment_name
    dirs = {
        "lane_dir": base,
        "weights_dir": base / "weights",
        "snapshots_dir": base / "dev_snapshots",
        "metrics_dir": base / "metrics",
        "sae_analysis_dir": base / "sae_analysis",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return {k: str(v) for k, v in dirs.items()}


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
def stage_netinit(config_ns, dirs: dict, num_classes: int) -> str:
    """NETINIT: create network and save initial weights."""
    _log_separator("STAGE 1 / 3 : NETINIT")

    from models import create_network
    from initialization import initialize_weights, save_weights

    _log_stage("NETINIT", f"Creating {config_ns.network_type} with {num_classes} classes")
    model = create_network(
        network_type=config_ns.network_type,
        num_classes=num_classes,
    )

    _log_stage("NETINIT", "Initializing weights (kaiming_normal)")
    initialize_weights(model, method="kaiming_normal", seed=config_ns.random_seed)

    weights_path = str(Path(dirs["weights_dir"]) / "initial_weights.pt")
    save_weights(model, weights_path)
    _log_stage("NETINIT", f"Saved initial weights -> {weights_path}")

    del model
    return weights_path


def stage_devtrain(
    config_ns,
    dirs: dict,
    weights_path: str,
    dataset_id: str,
    selected_classes: list,
    experiment_name: str,
) -> dict:
    """DEVTRAIN: train with periodic snapshots."""
    _log_separator("STAGE 2 / 3 : DEVTRAIN")

    from devtrain import train_with_snapshots

    _log_stage("DEVTRAIN", f"Training on {dataset_id} ({len(selected_classes)} classes)")
    _log_stage("DEVTRAIN", f"Snapshots -> {dirs['snapshots_dir']}")

    result = train_with_snapshots(
        initial_weights_path=weights_path,
        netinit_config=config_ns,
        dataset_id=dataset_id,
        selected_classes=selected_classes,
        snapshots_base_dir=dirs["snapshots_dir"],
        metrics_dir=dirs["metrics_dir"],
        experiment_id=experiment_name,
    )

    _log_stage("DEVTRAIN", f"Training complete — {result.get('total_snapshots', '?')} snapshots captured")
    return result


def stage_saeanalysis(
    config_dict: dict,
    dirs: dict,
    selected_classes: list,
) -> dict:
    """SAEANALYSIS: per-checkpoint SAE feature decomposition."""
    _log_separator("STAGE 3 / 3 : SAEANALYSIS")

    from saeanalysis import analyze_sae_features

    sae_policy = config_dict.get("sae_policy") or {}

    def progress_cb(pct, msg):
        _log_stage("SAE", f"[{pct:5.1f}%] {msg}")

    _log_stage("SAE", f"Starting SAE analysis on {dirs['lane_dir']}")

    result = analyze_sae_features(
        lane_dir=dirs["lane_dir"],
        selected_classes=selected_classes,
        experiment_config=config_dict,
        progress_callback=progress_cb,
        expansion_factor=sae_policy.get("expansion_factor", 4),
        k_sparse=sae_policy.get("k_sparse", 32),
        n_steps=sae_policy.get("n_steps", 10_000),
    )

    _log_stage("SAE", "SAE analysis complete")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run the 3-stage training reproduction pipeline (NETINIT -> DEVTRAIN -> SAEANALYSIS).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --config configs/standard/resnet18_cifar100_seed42.json
  python run_pipeline.py --config configs/standard/resnet18_cifar100_seed42.json --output-dir ./output
  python run_pipeline.py --config configs/standard/resnet18_cifar100_seed42.json --dry-run
        """,
    )
    parser.add_argument(
        "--config", required=True, help="Path to JSON config file"
    )
    parser.add_argument(
        "--output-dir", default="./output", help="Base output directory (default: ./output)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print plan without running",
    )
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────
    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    cfg = _enrich_config(cfg)

    experiment_name = cfg["experiment_name"]
    dataset_id = cfg["dataset_id"]
    num_classes = _resolve_num_classes(dataset_id)

    # Resolve selected classes — prefer config value, else auto-resolve
    if "selected_classes" in cfg and cfg["selected_classes"]:
        selected_classes = cfg["selected_classes"]
    else:
        selected_classes = _resolve_selected_classes(dataset_id)

    config_ns = dict_to_namespace(cfg)

    # ── Dry-run ──────────────────────────────────────────────────────
    if args.dry_run:
        _log_separator("DRY RUN — Pipeline Plan")
        print(f"  Experiment : {experiment_name}")
        print(f"  Network    : {cfg['network_type']}")
        print(f"  Dataset    : {dataset_id} ({num_classes} classes)")
        print(f"  Seed       : {cfg['random_seed']}")
        print(f"  Epochs     : {cfg['training_policy']['epochs']}")
        print(f"  Milestones : {cfg['snapshot_policy']['milestone_count']}")
        print(f"  Optimizer  : {cfg['optimizer']['type']} (lr={cfg['optimizer']['learning_rate']})")
        print(f"  Output dir : {Path(args.output_dir).resolve() / experiment_name}")
        sae = cfg.get("sae_policy") or {}
        if sae:
            print(f"  SAE policy : expansion={sae.get('expansion_factor', 4)}, "
                  f"k={sae.get('k_sparse', 32)}, steps={sae.get('n_steps', 10000)}")
        else:
            print("  SAE policy : (none — SAEANALYSIS will be skipped)")
        print(f"\n  Stages:")
        print(f"    1. NETINIT     — create {cfg['network_type']}, kaiming_normal init")
        print(f"    2. DEVTRAIN    — train with {cfg['snapshot_policy']['milestone_count']} snapshot milestones")
        if sae:
            print(f"    3. SAEANALYSIS — per-checkpoint SAE feature decomposition")
        else:
            print(f"    3. SAEANALYSIS — SKIPPED (no sae_policy)")
        print(f"\n  Config validated successfully. Remove --dry-run to execute.")
        sys.exit(0)

    # ── Create directories ───────────────────────────────────────────
    dirs = create_directory_structure(args.output_dir, experiment_name)

    # Save a copy of the config into the lane directory
    config_copy_path = Path(dirs["lane_dir"]) / "pipeline_config.json"
    with open(config_copy_path, "w") as f:
        json.dump(cfg, f, indent=2)

    t0 = time.time()
    _log_separator(f"Pipeline: {experiment_name}")
    _log_stage("PIPELINE", f"Output -> {dirs['lane_dir']}")

    # ── Stage 1: NETINIT ─────────────────────────────────────────────
    try:
        weights_path = stage_netinit(config_ns, dirs, num_classes)
    except Exception as exc:
        _log_stage("NETINIT", f"FAILED: {exc}")
        traceback.print_exc()
        sys.exit(1)

    # ── Stage 2: DEVTRAIN ────────────────────────────────────────────
    try:
        devtrain_result = stage_devtrain(
            config_ns, dirs, weights_path, dataset_id, selected_classes, experiment_name,
        )
    except Exception as exc:
        _log_stage("DEVTRAIN", f"FAILED: {exc}")
        traceback.print_exc()
        sys.exit(1)

    # ── Stage 3: SAEANALYSIS ─────────────────────────────────────────
    sae_policy = cfg.get("sae_policy")
    if sae_policy:
        try:
            sae_result = stage_saeanalysis(cfg, dirs, selected_classes)
        except Exception as exc:
            _log_stage("SAE", f"FAILED: {exc}")
            traceback.print_exc()
            sys.exit(1)
    else:
        _log_stage("SAE", "Skipped — no sae_policy in config")
        sae_result = None

    # ── Summary ──────────────────────────────────────────────────────
    elapsed = time.time() - t0
    _log_separator("Pipeline Complete")
    print(f"  Experiment  : {experiment_name}")
    print(f"  Duration    : {elapsed / 60:.1f} minutes ({elapsed:.0f}s)")
    print(f"  Snapshots   : {devtrain_result.get('total_snapshots', 'N/A')}")
    print(f"  Final epoch : {devtrain_result.get('final_epoch', 'N/A')}")
    if sae_result:
        print(f"  SAE results : {dirs['lane_dir']}/sae_analysis/sae_results.json")
    print(f"  Output dir  : {dirs['lane_dir']}")


if __name__ == "__main__":
    main()
