#!/usr/bin/env python3
"""
Orchestrator for the label-noise curriculum experiment.

Per lane it runs the full pipeline end to end, without the backend or database:
    NETINIT (shared per-seed initial weights)
      -> DEVTRAIN (train + every-5-epoch snapshots, with the noise-curriculum policy)
      -> SAEANALYSIS (per-checkpoint SAEs -> sae_results.json)

Defaults: fixed between-superclass noise, p=0.3, applied for epochs [0, K) and
then switched off, sweeping the window length K (as in the superclass curriculum
sweep). Conditions per (arm, seed):
    standard (no noise) | switch_e05 | e10 | e15 | e20 | e25 | throughout
Outputs are written under <repo>/data/experiments/<EXPERIMENT_ID>/ (`data` can be
a symlink to external storage).

  * Idempotent and resumable: every lane and stage writes a marker; re-running the
    same command skips finished work.
  * Continue on failure (default): a failed lane is recorded (FAILED.txt and the
    status log) and the run moves on. Use --abort-on-fail to stop instead.
  * Append-only status log <experiment>/ORCHESTRATOR_STATUS.jsonl and a summary
    printed at the end. Run under `python -u` for unbuffered output.

Usage:
    python -u scripts/run_noise_curriculum_experiment.py \
        --experiment-id noise_curriculum \
        --arms between_sc \
        --seeds 42,137,256 \
        --switch-epochs 5,10,15,20,25 \
        --epochs 30 --noise-prob 0.3 2>&1 | tee -a noise_curriculum.out

Add `within_sc` to --arms for the identification contrast arm.
Add --plan-only to print the plan without training.
"""
import argparse
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

REPO = Path(__file__).resolve().parent.parent
ML_SRC = REPO / "packages" / "ml" / "src"
BE_SRC = REPO / "packages" / "backend" / "src"
# ML_SRC must precede BE_SRC on sys.path so the bare module name
# `models` resolves to packages/ml/src/models.py (which defines create_network and
# the network classes the pipeline trains), not the backend's pydantic-schema
# models.py. Inserting BE_SRC last leaves ML_SRC at index 0.
sys.path.insert(0, str(BE_SRC))
sys.path.insert(0, str(ML_SRC))

# get_dataset() uses <repo>/data/experiments/<exp>/datasets as the dataset root;
# <repo>/data can be a symlink to external storage.
DATA_DIR = REPO / "data"

# Reference template (an existing noise lane), used to inherit the
# architecture, optimizer, snapshot and SAE policy of the comparable
# 30-epoch, every-5-epoch noise lanes. Read from the database; embedded fallback below.
REF_DB_CANDIDATES = ["rcx_RUNPOD.db", "rcx.db", "rcx_BACKUP.db"]
REF_LANE = "2cc9ed75-ab1f-444f-a59b-5cc9ac6c8f9f"   # standard / seed137 noise lane
REF_EXP = "0c28799c-ec9d-4e45-ac30-302dff810a8d"     # has datasets/cifar100

# Fallback if no database is reachable (matches the reference lane).
EMBEDDED_TEMPLATE = {
    "network_type": "resnet18",
    "initialization_method": "kaiming_normal",
    "random_seed": 42,
    "batch_size": 128,
    "training_policy": {"epochs": 30, "accuracy_threshold": None},
    "snapshot_policy": {
        "milestone_count": 6, "milestone_type": "weight_updates",
        "milestone_weight_updates": [1955, 3910, 5865, 7820, 9775, 11730],
        "distribution_scheme": "uniform", "samples_per_class": 100,
        "terminal_capture": "both", "milestone_accuracies": [50.0],
    },
    "optimizer": {"type": "sgd", "learning_rate": 0.1, "weight_decay": 0.0005,
                  "momentum": 0.9, "nesterov": True},
    "sae_policy": {"expansion_factor": 4, "k_sparse": 32, "n_steps": 5000,
                   "shared_init_seed": 42, "null_permutations": 1000},
    "representation_methods": ["individual_multilayer_activations"],
    "validation_policy": {}, "memory_policy": {}, "logging_policy": {},
    "transform_config": None,
}


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_status(exp_dir, event, **kw):
    """Append one JSON line to the status log and print a readable line."""
    rec = {"ts": now(), "event": event, **kw}
    with open(exp_dir / "ORCHESTRATOR_STATUS.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    extra = " ".join(f"{k}={v}" for k, v in kw.items())
    print(f"[{rec['ts']}] {event} {extra}", flush=True)


# Template, dataset and classes
def load_template(args):
    """Resolve the NETINIT config template from --template-json, the database, or the embedded fallback (in that order)."""
    if args.template_json:
        return json.loads(Path(args.template_json).read_text()), f"json:{args.template_json}"
    import sqlite3
    for db_name in ([args.ref_db] if args.ref_db else REF_DB_CANDIDATES):
        db = DATA_DIR / db_name
        if not db.exists():
            continue
        try:
            con = sqlite3.connect(str(db))
            row = con.execute(
                "SELECT configuration FROM stoves WHERE lane_id=? AND stove_type='NETINIT'",
                (args.ref_lane,)).fetchone()
            con.close()
            if row and row[0]:
                return json.loads(row[0]), f"db:{db_name}:{args.ref_lane}"
        except Exception as e:
            print(f"  (template DB {db_name} unreadable: {e})", flush=True)
    print("  WARNING: no DB template found: using EMBEDDED_TEMPLATE fallback.", flush=True)
    return dict(EMBEDDED_TEMPLATE), "embedded"


def ensure_dataset(exp_id, dataset="cifar100"):
    """Ensure the dataset is present under this experiment's datasets dir.
    Tries: existing dir -> symlink from another experiment -> torchvision download."""
    if dataset == "tiny_imagenet":
        # TinyImageNet (downloaded and val-restructured once into datasets_cache).
        ds_dir = DATA_DIR / "experiments" / exp_id / "datasets" / "tiny_imagenet"
        # Prefer a local copy: TinyImageNet is 110k small JPEGs, and reading them
        # from external storage per batch is slow.
        local_cache = Path(RDT_DATA + "/tiny_imagenet_cache/tiny-imagenet-200")
        cache = local_cache if (local_cache / "train").exists() else \
            DATA_DIR / "datasets_cache" / "tiny-imagenet-200"
        ds_dir.mkdir(parents=True, exist_ok=True)
        if not (ds_dir / "train").exists():
            os.symlink(cache / "train", ds_dir / "train")
        if not (ds_dir / "val").exists():
            os.symlink(cache / "val_sorted", ds_dir / "val")
        return
    ds_dir = DATA_DIR / "experiments" / exp_id / "datasets" / "cifar100"
    if (ds_dir / "cifar-100-python").exists():
        return
    ds_dir.mkdir(parents=True, exist_ok=True)
    # reuse an existing copy if present
    for cand in (DATA_DIR / "experiments").glob("*/datasets/cifar100/cifar-100-python"):
        print(f"  linking CIFAR-100 from existing copy: {cand.parent}", flush=True)
        if (ds_dir / "cifar-100-python").exists():
            break
        os.symlink(cand, ds_dir / "cifar-100-python")
        return
    # otherwise download once
    print("  downloading CIFAR-100 via torchvision (one-time)...", flush=True)
    import torchvision
    torchvision.datasets.CIFAR100(root=str(ds_dir), train=True, download=True)
    torchvision.datasets.CIFAR100(root=str(ds_dir), train=False, download=True)


def canonical_classes(exp_id, dataset="cifar100"):
    """Fine-class names in the dataset's native label order (matches get_dataset)."""
    if dataset == "tiny_imagenet":
        # ImageFolder labels classes by sorted wnid directory names; get_dataset then
        # renames each to its readable label but keeps the index. Mirror that order
        # so selected_classes[i] == the class with dataset label i.
        from tiny_imagenet_hierarchy import WNID_TO_CLASS
        train_dir = DATA_DIR / "experiments" / exp_id / "datasets" / "tiny_imagenet" / "train"
        wnids = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
        return [WNID_TO_CLASS.get(w, w) for w in wnids]
    import torchvision
    ds = torchvision.datasets.CIFAR100(
        root=str(DATA_DIR / "experiments" / exp_id / "datasets" / "cifar100"),
        train=False, download=False)
    return list(ds.classes)


# Config helpers
def build_lane_config(template, epochs, seed, noise_policy_dict, arch, dataset="cifar100"):
    """Copy the template into a namespace with this lane's overrides."""
    from run_devtrain_direct import dict_to_namespace, add_missing_defaults
    cfg = json.loads(json.dumps(template))
    cfg["training_policy"]["epochs"] = epochs
    # weight-update snapshots do not need accuracy milestones, but
    # add_missing_defaults computes them from accuracy_threshold if absent.
    if cfg["training_policy"].get("accuracy_threshold") is None:
        cfg["training_policy"]["accuracy_threshold"] = 100.0
    cfg["snapshot_policy"].setdefault("milestone_accuracies", [50.0])
    cfg["random_seed"] = seed
    # Architecture sweep: each lane fixes one architecture (CNN, transformer,
    # conv-transformer hybrid). All three run at 32x32 (resnet18, cct_7
    # by design; vit_tiny via image_size=32/patch=4) so transform_config stays None.
    cfg["network_type"] = arch
    cfg["curriculum_policy"] = None
    cfg["noise_policy"] = noise_policy_dict
    # Skip the per-class-mean REPMETRICS extraction (multilayer_activations.pt):
    # it is not used by this experiment's downstream analysis (which reads only
    # sample_predictions.pt and sae_results.json) and writes ~50k small files per milestone. The SAE input
    # individual_multilayer_activations.pt is always extracted by devtrain.
    cfg["representation_methods"] = []
    # TinyImageNet is 64px and the template's transform_config has no resize, so
    # the config path would default RandomCrop to 224 and leave val at 64px
    # (incompatible with ViT at image_size=32). A 32px resize makes train and val
    # 32x32 so every CIFAR architecture config trains unchanged.
    if dataset == "tiny_imagenet":
        tc = cfg.get("transform_config") or {}
        tc = dict(tc); tc["resize"] = 32
        cfg["transform_config"] = tc
    # Epoch-align the weight-update snapshot milestones to the switch epochs
    # (5,10,15,20,25,30). The template milestones [1955,3910,...,11730]
    # assume batch 128 (50000/128 ~ 391 batches/epoch, i.e. every 5
    # epochs). On MPS devtrain caps the effective batch at 16, giving 3125
    # batches/epoch, so those counts would fall at epochs 0.6-3.75.
    # Milestones are recomputed from the effective batch size so they fall on
    # the switch epochs on any device (on CUDA the effective batch stays 128).
    import math, os, torch
    # train-set size per dataset (drives batches/epoch -> milestone alignment)
    n_train = 100000 if dataset == "tiny_imagenet" else 50000  # TIN 200x500, CIFAR 100x500
    template_bs = cfg.get("batch_size", 128)
    device = ('cuda' if torch.cuda.is_available()
              else 'mps' if torch.backends.mps.is_available() else 'cpu')
    # Mirror devtrain's MPS batch cap (same DEVTRAIN_MPS_MAX_BATCH override) so the
    # effective batch, and hence batches/epoch and the milestone weight-update counts,
    # match training.
    mps_max_batch = int(os.environ.get("DEVTRAIN_MPS_MAX_BATCH", "16"))
    eff_bs = mps_max_batch if (device == 'mps' and template_bs > mps_max_batch) else template_bs
    batches_per_epoch = math.ceil(n_train / eff_bs)
    switch_epochs = [e for e in (5, 10, 15, 20, 25, 30) if e <= epochs]
    cfg["snapshot_policy"]["milestone_weight_updates"] = [
        e * batches_per_epoch for e in switch_epochs]
    cfg["snapshot_policy"]["milestone_count"] = len(switch_epochs)
    ns = dict_to_namespace(cfg)
    add_missing_defaults(ns, cfg)
    return ns


def lane_plan(args):
    """Return list of (lane_id, arch, noise_policy_dict).

    Sweeps architectures (CNN, transformer, hybrid) at a single fixed seed, the
    cross-architecture counterpart of the per-seed sweep. The noise manipulation
    (fixed between-superclass corruption, switched off at K) is identical across
    architectures; only network_type varies.
    """
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    switch = [int(k) for k in args.switch_epochs.split(",") if k.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seed = args.seed
    plan = []
    # Grouped by architecture so each architecture's full set (clean baseline + all
    # noise lanes) runs contiguously, in the order given by --archs. This lets the
    # cheaper architectures finish before the most expensive one (e.g. cct_7 on
    # MPS) starts; list the slowest architecture last in --archs.
    for a in archs:
        # standard (clean) baseline for this architecture
        plan.append((f"standard_{a}", a, {"enabled": False}))
        # noise lanes for this architecture
        for arm in arms:
            for k in switch:
                plan.append((f"{arm}_{a}_e{k:02d}", a, {
                    "enabled": True, "noise_type": arm, "noise_prob": args.noise_prob,
                    "fixed": True, "noise_seed": seed, "noise_until_epoch": k}))
            pol = {"enabled": True, "noise_type": arm, "noise_prob": args.noise_prob,
                   "fixed": True, "noise_seed": seed, "noise_until_epoch": None}
            if arm == "between_sc_matched":  # similarity-matched targets
                pol["matched_targets_json"] = args.matched_targets_json
            plan.append((f"{arm}_{a}_throughout", a, pol))
    return plan


# Stages
def ensure_seed_init(exp_dir, arch, seed, template, num_classes=100):
    """Create (once) a shared initial-weights file per (architecture, seed); reuse
    across all noise conditions for that architecture."""
    init_path = exp_dir / "_inits" / f"{arch}_seed{seed}.pt"
    if init_path.exists():
        return init_path
    init_path.parent.mkdir(parents=True, exist_ok=True)
    from models import create_network
    from initialization import initialize_weights, save_weights
    model = create_network(network_type=arch, num_classes=num_classes)
    initialize_weights(model, method=template["initialization_method"], seed=seed)
    save_weights(model, str(init_path))
    return init_path


def run_devtrain(lane_dir, cfg, init_path, selected_classes, exp_id, dataset_id="cifar100"):
    from devtrain import train_with_snapshots
    snaps = lane_dir / "dev_snapshots"
    metrics = snaps / "metrics"
    snaps.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    return train_with_snapshots(
        initial_weights_path=str(init_path), netinit_config=cfg,
        dataset_id=dataset_id, selected_classes=selected_classes,
        snapshots_base_dir=str(snaps), metrics_dir=str(metrics),
        experiment_id=exp_id, stove_id=None)


def run_sae(lane_dir, selected_classes, sae_policy):
    from saeanalysis import analyze_sae_features
    return analyze_sae_features(
        lane_dir=str(lane_dir), selected_classes=selected_classes,
        experiment_config={"dataset_id": "cifar100", "sae_policy": sae_policy},
        progress_callback=lambda p, m: print(f"    sae [{p:5.1f}%] {m}", flush=True),
        expansion_factor=sae_policy.get("expansion_factor", 4),
        k_sparse=sae_policy.get("k_sparse", 32),
        n_steps=sae_policy.get("n_steps", 5000))


def main():
    ap = argparse.ArgumentParser(description="Label-noise curriculum orchestrator")
    ap.add_argument("--experiment-id", default="noise_curriculum")
    ap.add_argument("--arms", default="between_sc",
                    help="comma list: between_sc[,within_sc]")
    ap.add_argument("--archs", default="resnet18,vit_tiny,cct_7",
                    help="comma-separated architectures to sweep (CNN/transformer/hybrid)")
    ap.add_argument("--seed", type=int, default=42,
                    help="single seed shared by all architectures")
    ap.add_argument("--switch-epochs", default="5,10,15,20,25")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--noise-prob", type=float, default=0.3)
    ap.add_argument("--matched-targets-json", default=None,
                    help="between_sc_matched: per-class flip targets")
    ap.add_argument("--dataset", default="cifar100",
                    help="cifar100 | tiny_imagenet (TIN resized to 32x32, 200 classes)")
    ap.add_argument("--skip-sae", action="store_true", help="train only, no SAE analysis")
    ap.add_argument("--abort-on-fail", action="store_true")
    ap.add_argument("--plan-only", action="store_true", help="print the lane plan and exit")
    ap.add_argument("--template-json", default=None)
    ap.add_argument("--ref-db", default=None)
    ap.add_argument("--ref-lane", default=REF_LANE)
    args = ap.parse_args()

    exp_id = args.experiment_id
    exp_dir = DATA_DIR / "experiments" / exp_id
    plan = lane_plan(args)

    print("=" * 72)
    print(f"LABEL-NOISE CURRICULUM ORCHESTRATOR  exp={exp_id}")
    print(f"arms={args.arms} archs={args.archs} seed={args.seed} switch={args.switch_epochs} "
          f"epochs={args.epochs} p={args.noise_prob}")
    print(f"{len(plan)} lanes -> {exp_dir}")
    print("=" * 72, flush=True)
    for lane_id, arch, npol in plan:
        u = npol.get("noise_until_epoch", "-") if npol.get("enabled") else "clean"
        print(f"  {lane_id:<32} arch={arch:<10} noise={'%s p=%s until=%s' % (npol.get('noise_type'), args.noise_prob, u) if npol.get('enabled') else 'OFF'}")
    if args.plan_only:
        print("\n--plan-only: exiting without running.", flush=True)
        return

    # the data root must resolve (e.g. external storage behind the `data` symlink)
    if not DATA_DIR.exists():
        sys.exit(f"FATAL: {DATA_DIR} does not exist (check the `data` symlink).")
    exp_dir.mkdir(parents=True, exist_ok=True)

    template, tsrc = load_template(args)
    log_status(exp_dir, "RUN_START", n_lanes=len(plan), template=tsrc,
               epochs=args.epochs, arms=args.arms, archs=args.archs, seed=args.seed)
    # save the resolved template for provenance
    (exp_dir / "_config").mkdir(exist_ok=True)
    (exp_dir / "_config" / "template.json").write_text(json.dumps(template, indent=2))
    (exp_dir / "_config" / "run_args.json").write_text(json.dumps(vars(args), indent=2))

    ensure_dataset(exp_id, args.dataset)
    selected_classes = canonical_classes(exp_id, args.dataset)
    expected_n = 200 if args.dataset == "tiny_imagenet" else 100
    assert len(selected_classes) == expected_n, \
        f"expected {expected_n} classes for {args.dataset}, got {len(selected_classes)}"

    done, skipped, failed = [], [], []
    for i, (lane_id, arch, npol) in enumerate(plan, 1):
        lane_dir = exp_dir / "lanes" / lane_id
        lane_dir.mkdir(parents=True, exist_ok=True)
        tag = f"[{i}/{len(plan)}] {lane_id}"
        if (lane_dir / ".lane_done").exists():
            skipped.append(lane_id)
            print(f"{tag}: already complete: skip", flush=True)
            continue
        try:
            t0 = time.time()
            init_path = ensure_seed_init(exp_dir, arch, args.seed, template,
                                         num_classes=len(selected_classes))
            cfg = build_lane_config(template, args.epochs, args.seed, npol, arch, args.dataset)

            if not (lane_dir / ".devtrain_done").exists():
                log_status(exp_dir, "DEVTRAIN_START", lane=lane_id)
                run_devtrain(lane_dir, cfg, init_path, selected_classes, exp_id, args.dataset)
                (lane_dir / ".devtrain_done").touch()
                log_status(exp_dir, "DEVTRAIN_DONE", lane=lane_id,
                           mins=round((time.time() - t0) / 60, 1))
            else:
                print(f"{tag}: devtrain already done: skip to SAE", flush=True)

            if not args.skip_sae and not (lane_dir / ".sae_done").exists():
                t1 = time.time()
                log_status(exp_dir, "SAE_START", lane=lane_id)
                run_sae(lane_dir, selected_classes, template.get("sae_policy", {}))
                if not (lane_dir / "sae_analysis" / "sae_results.json").exists():
                    raise RuntimeError("SAE finished but sae_results.json missing")
                (lane_dir / ".sae_done").touch()
                log_status(exp_dir, "SAE_DONE", lane=lane_id,
                           mins=round((time.time() - t1) / 60, 1))

            (lane_dir / ".lane_done").touch()
            done.append(lane_id)
            log_status(exp_dir, "LANE_DONE", lane=lane_id,
                       mins=round((time.time() - t0) / 60, 1))
        except KeyboardInterrupt:
            log_status(exp_dir, "INTERRUPTED", lane=lane_id)
            print("\nInterrupted: re-run the same command to resume.", flush=True)
            raise
        except Exception as e:
            failed.append(lane_id)
            (lane_dir / "FAILED.txt").write_text(f"{now()}\n{e}\n\n{traceback.format_exc()}")
            log_status(exp_dir, "LANE_FAILED", lane=lane_id, error=str(e)[:200])
            print(f"{tag}: FAILED: {e} (see {lane_dir/'FAILED.txt'})", flush=True)
            traceback.print_exc()
            if args.abort_on_fail:
                log_status(exp_dir, "RUN_ABORTED", failed=lane_id)
                sys.exit(1)

    log_status(exp_dir, "RUN_END", done=len(done), skipped=len(skipped), failed=len(failed))
    print("\n" + "=" * 72)
    print(f"SUMMARY  done={len(done)} skipped(prev)={len(skipped)} failed={len(failed)}")
    if failed:
        print("FAILED lanes (re-run the same command to retry them):")
        for l in failed:
            print(f"  - {l}")
    else:
        print("All lanes complete.")
    print("=" * 72, flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
