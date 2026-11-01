#!/usr/bin/env python3
"""Run the RDT pipeline for one training run from a JSON configuration.

Three stages, with the code that produced the paper's runs (packages/ml/src):
  1. NETINIT      create the network and its initial weights (Kaiming normal, fixed seed)
  2. DEVTRAIN     train, capturing checkpoints at uniformly spaced milestones with per-sample activations
                  (devtrain.train_with_snapshots)
  3. SAEANALYSIS  per checkpoint and layer, train a top-k sparse autoencoder, match features across checkpoints,
                  compute SSI/CSI/SAI against permutation nulls and classify developmental events
                  (saeanalysis.analyze_sae_features)

The dataset is downloaded on first use into data/experiments/<experiment_name>/datasets/, where the training code
reads it. Results go to <output-dir>/<experiment_name>/ (sae_analysis/sae_results.json holds the event counts).

Usage:
  python run/run_pipeline.py --config run/configs/demo/demo_cifar100_4superclasses.json
  python run/run_pipeline.py --config run/configs/standard/resnet18_cifar100_seed42.json [--output-dir output] [--dry-run]
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packages" / "ml" / "src"))
sys.path.insert(0, str(REPO / "packages" / "backend" / "src"))

DATASETS = ("cifar10", "cifar100", "tiny_imagenet")


def log(stage, msg):
    print(f"[{time.strftime('%H:%M:%S')}] [{stage}] {msg}", flush=True)


def banner(title):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}\n", flush=True)


def load_config(path):
    """The run's JSON configuration, and the same configuration as the job system's model (defaults filled in)."""
    from netinit_models import NetInitConfiguration
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    cfg = json.loads(path.read_text())
    for k in ("experiment_name", "dataset_id"):
        if k not in cfg:
            raise ValueError(f"config is missing {k}")
    if cfg["dataset_id"] not in DATASETS:
        raise ValueError(f"dataset_id must be one of {DATASETS}")
    return cfg, NetInitConfiguration(**cfg)


def prepare_dataset(cfg):
    """Put the dataset where the training code reads it and return the class names of the run."""
    root = REPO / "data" / "experiments" / cfg["experiment_name"] / "datasets" / cfg["dataset_id"]
    if cfg["dataset_id"] == "tiny_imagenet":
        if not (root / "train").is_dir():
            sys.exit(f"Tiny ImageNet is not downloaded automatically: unpack tiny-imagenet-200 so that {root}/train "
                     f"and {root}/val hold one folder per class (val reorganised from val_annotations.txt).")
        return cfg["selected_classes"]
    import torchvision
    cls = torchvision.datasets.CIFAR100 if cfg["dataset_id"] == "cifar100" else torchvision.datasets.CIFAR10
    for train in (True, False):
        ds = cls(root=str(root), train=train, download=True)
    classes = list(ds.classes)
    if cfg.get("superclasses"):  # a subset of CIFAR-100 by superclass (demo)
        from cifar100_hierarchy import CIFAR100_SUPERCLASSES
        return [c for c in classes if any(c in CIFAR100_SUPERCLASSES[s] for s in cfg["superclasses"])]
    return cfg.get("selected_classes") or classes


def main():
    ap = argparse.ArgumentParser(description="RDT pipeline: NETINIT -> DEVTRAIN -> SAEANALYSIS for one run.")
    ap.add_argument("--config", required=True, help="JSON configuration (run/configs/)")
    ap.add_argument("--output-dir", default=str(REPO / "output"), help="base output directory (default: output/)")
    ap.add_argument("--dry-run", action="store_true", help="validate the configuration and print the plan")
    args = ap.parse_args()

    try:
        cfg, ns = load_config(args.config)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        sys.exit(f"ERROR: {exc}")
    name, sae = cfg["experiment_name"], cfg.get("sae_policy")
    lane = Path(args.output_dir).resolve() / name

    if args.dry_run:
        banner("Plan")
        print(f"  Run        : {name}\n  Network    : {cfg['network_type']}\n  Dataset    : {cfg['dataset_id']}"
              f"{' (superclasses ' + ', '.join(cfg['superclasses']) + ')' if cfg.get('superclasses') else ''}\n"
              f"  Seed       : {cfg['random_seed']}\n  Epochs     : {cfg['training_policy']['epochs']}\n"
              f"  Milestones : {cfg['snapshot_policy']['milestone_count']}\n"
              f"  SAE        : {'expansion %s, k = %s, %s steps' % (sae.get('expansion_factor', 4), sae.get('k_sparse', 32), sae.get('n_steps', 10000)) if sae else 'none (SAEANALYSIS skipped)'}\n"
              f"  Output     : {lane}")
        return

    dirs = {k: lane / v for k, v in [("lane", ""), ("weights", "weights"), ("snapshots", "dev_snapshots"),
                                     ("metrics", "metrics"), ("sae", "sae_analysis")]}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    (lane / "pipeline_config.json").write_text(json.dumps(cfg, indent=2))
    t0 = time.time()

    if cfg.get("mps_max_batch"):  # the training code caps the batch on Apple MPS (default 16) unless told otherwise
        import os
        os.environ.setdefault("DEVTRAIN_MPS_MAX_BATCH", str(cfg["mps_max_batch"]))
    log("DATA", f"preparing {cfg['dataset_id']}")
    selected = prepare_dataset(cfg)
    log("DATA", f"{len(selected)} classes")

    try:
        banner("Stage 1/3: NETINIT")
        from initialization import initialize_weights, save_weights
        from models import create_network
        model = create_network(network_type=cfg["network_type"], num_classes=len(selected))
        initialize_weights(model, method=cfg.get("initialization_method", "kaiming_normal"), seed=cfg["random_seed"])
        weights = str(dirs["weights"] / "initial_weights.pt")
        save_weights(model, weights)
        log("NETINIT", f"{cfg['network_type']}, {sum(p.numel() for p in model.parameters()):,} parameters")
        del model

        banner("Stage 2/3: DEVTRAIN")
        from devtrain import train_with_snapshots
        res = train_with_snapshots(initial_weights_path=weights, netinit_config=ns, dataset_id=cfg["dataset_id"],
                                   selected_classes=selected, snapshots_base_dir=str(dirs["snapshots"]),
                                   metrics_dir=str(dirs["metrics"]), experiment_id=name)
        log("DEVTRAIN", f"{res.get('total_snapshots', '?')} checkpoints")

        if sae:
            banner("Stage 3/3: SAEANALYSIS")
            from saeanalysis import analyze_sae_features
            analyze_sae_features(lane_dir=str(lane), selected_classes=selected, experiment_config=cfg,
                                 expansion_factor=sae.get("expansion_factor", 4), k_sparse=sae.get("k_sparse", 32),
                                 n_steps=sae.get("n_steps", 10_000))
    except Exception as exc:
        traceback.print_exc()
        sys.exit(f"FAILED: {exc}")

    banner("Done")
    print(f"  {name}: {(time.time() - t0) / 60:.1f} min; results in {lane}")
    if sae:
        summarize(lane / "sae_analysis" / "sae_results.json")


def summarize(path):
    """Print the developmental events per transition (Tg-E, Ab-E, Di-E) from sae_results.json."""
    r = json.loads(path.read_text())
    rows = r.get("process_intensity") or []
    if not rows:
        print(f"  event counts: see {path}")
        return
    meta = r.get("metadata", {})
    thr = meta.get("analysis_thresholds") or {}
    print(f"\n  checkpoints: {', '.join(map(str, meta.get('checkpoint_labels', [])))}")
    if thr:
        print(f"  thresholds: SSI {thr.get('ssi_abstraction_thresh', float('nan')):.3f}, "
              f"CSI {thr.get('csi_differentiation_thresh', float('nan')):.3f}, "
              f"SAI {thr.get('sai_task_general_thresh', float('nan')):.3f}")
    print("  developmental events per transition, summed over layers (Methods): Tg-E task-general,")
    print("  Ab-E superclass-level, Di-E class-level")
    print("\n  transition      Tg-E    Ab-E    Di-E")
    for t in rows:
        g = lambda k: t.get(k, t.get(k.lower(), float("nan")))
        print(f"  {str(t.get('transition', t.get('label', ''))):14s} {g('tg_h'):7.0f} {g('ab_h'):7.0f} {g('di_h'):7.0f}")


if __name__ == "__main__":
    main()
