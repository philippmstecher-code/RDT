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
        --epochs 30 --noise-prob 0.3 2>&1 | tee -a logs/noise_curriculum/noise_curriculum.out

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

REPO = Path(__file__).resolve().parent.parent
ML_SRC = REPO / "packages" / "ml" / "src"
BE_SRC = REPO / "packages" / "backend" / "src"
sys.path.insert(0, str(ML_SRC))
sys.path.insert(0, str(BE_SRC))

# get_dataset() uses <repo>/data/experiments/<exp>/datasets as the dataset root;
# <repo>/data can be a symlink to external storage.
DATA_DIR = REPO / "data"

# Reference template (an existing noise lane), used to inherit the
# architecture, optimizer, snapshot and SAE policy of the comparable
# 30-epoch, every-5-epoch noise lanes. Read from the database; embedded fallback below.
REF_DB_CANDIDATES = ["rcx_RUNPOD.db", "rcx.db"]
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
    print("  WARNING: no DB template found; using the EMBEDDED_TEMPLATE fallback.", flush=True)
    return dict(EMBEDDED_TEMPLATE), "embedded"


def ensure_dataset(exp_id):
    """Ensure CIFAR-100 is present under this experiment's datasets dir.
    Tries the existing dir, then a symlink from another experiment, then a torchvision download."""
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


def canonical_classes(exp_id):
    """100 CIFAR-100 fine-class names in torchvision's native order (matches get_dataset)."""
    import torchvision
    ds = torchvision.datasets.CIFAR100(
        root=str(DATA_DIR / "experiments" / exp_id / "datasets" / "cifar100"),
        train=False, download=False)
    return list(ds.classes)


# Config helpers
def build_lane_config(template, epochs, seed, noise_policy_dict):
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
    cfg["curriculum_policy"] = None
    cfg["noise_policy"] = noise_policy_dict
    ns = dict_to_namespace(cfg)
    add_missing_defaults(ns, cfg)
    return ns


def lane_plan(args):
    """Return list of (lane_id, seed, noise_policy_dict)."""
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    switch = [int(k) for k in args.switch_epochs.split(",") if k.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    plan = []
    # standard (clean) baseline, once per seed and independent of the arm
    for s in seeds:
        plan.append((f"standard_seed{s}", s, {"enabled": False}))
    for arm in arms:
        for s in seeds:
            for k in switch:
                plan.append((f"{arm}_seed{s}_e{k:02d}", s, {
                    "enabled": True, "noise_type": arm, "noise_prob": args.noise_prob,
                    "fixed": True, "noise_seed": s, "noise_until_epoch": k}))
            plan.append((f"{arm}_seed{s}_throughout", s, {
                "enabled": True, "noise_type": arm, "noise_prob": args.noise_prob,
                "fixed": True, "noise_seed": s, "noise_until_epoch": None}))
    return plan


# Stages
def ensure_seed_init(exp_dir, seed, template):
    """Create (once) a shared initial-weights file per seed; reuse across conditions."""
    init_path = exp_dir / "_inits" / f"seed{seed}.pt"
    if init_path.exists():
        return init_path
    init_path.parent.mkdir(parents=True, exist_ok=True)
    from models import create_network
    from initialization import initialize_weights, save_weights
    model = create_network(network_type=template["network_type"], num_classes=100)
    initialize_weights(model, method=template["initialization_method"], seed=seed)
    save_weights(model, str(init_path))
    return init_path


def run_devtrain(lane_dir, cfg, init_path, selected_classes, exp_id):
    from devtrain import train_with_snapshots
    snaps = lane_dir / "dev_snapshots"
    metrics = snaps / "metrics"
    snaps.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    return train_with_snapshots(
        initial_weights_path=str(init_path), netinit_config=cfg,
        dataset_id="cifar100", selected_classes=selected_classes,
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
    ap.add_argument("--seeds", default="42,137,256")
    ap.add_argument("--switch-epochs", default="5,10,15,20,25")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--noise-prob", type=float, default=0.3)
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
    print(f"arms={args.arms} seeds={args.seeds} switch={args.switch_epochs} "
          f"epochs={args.epochs} p={args.noise_prob}")
    print(f"{len(plan)} lanes -> {exp_dir}")
    print("=" * 72, flush=True)
    for lane_id, seed, npol in plan:
        u = npol.get("noise_until_epoch", "-") if npol.get("enabled") else "clean"
        print(f"  {lane_id:<28} seed={seed:<4} noise={'%s p=%s until=%s' % (npol.get('noise_type'), args.noise_prob, u) if npol.get('enabled') else 'OFF'}")
    if args.plan_only:
        print("\n--plan-only: exiting without running.", flush=True)
        return

    # the data root must resolve (e.g. external storage behind the `data` symlink)
    if not DATA_DIR.exists():
        sys.exit(f"FATAL: {DATA_DIR} does not resolve. Mount the external drive / fix the `data` symlink.")
    exp_dir.mkdir(parents=True, exist_ok=True)

    template, tsrc = load_template(args)
    log_status(exp_dir, "RUN_START", n_lanes=len(plan), template=tsrc,
               epochs=args.epochs, arms=args.arms, seeds=args.seeds)
    # save the resolved template for provenance
    (exp_dir / "_config").mkdir(exist_ok=True)
    (exp_dir / "_config" / "template.json").write_text(json.dumps(template, indent=2))
    (exp_dir / "_config" / "run_args.json").write_text(json.dumps(vars(args), indent=2))

    ensure_dataset(exp_id)
    selected_classes = canonical_classes(exp_id)
    assert len(selected_classes) == 100, f"expected 100 classes, got {len(selected_classes)}"

    done, skipped, failed = [], [], []
    for i, (lane_id, seed, npol) in enumerate(plan, 1):
        lane_dir = exp_dir / "lanes" / lane_id
        lane_dir.mkdir(parents=True, exist_ok=True)
        tag = f"[{i}/{len(plan)}] {lane_id}"
        if (lane_dir / ".lane_done").exists():
            skipped.append(lane_id)
            print(f"{tag}: already complete, skip", flush=True)
            continue
        try:
            t0 = time.time()
            init_path = ensure_seed_init(exp_dir, seed, template)
            cfg = build_lane_config(template, args.epochs, seed, npol)

            if not (lane_dir / ".devtrain_done").exists():
                log_status(exp_dir, "DEVTRAIN_START", lane=lane_id)
                run_devtrain(lane_dir, cfg, init_path, selected_classes, exp_id)
                (lane_dir / ".devtrain_done").touch()
                log_status(exp_dir, "DEVTRAIN_DONE", lane=lane_id,
                           mins=round((time.time() - t0) / 60, 1))
            else:
                print(f"{tag}: devtrain already done, skip to SAE", flush=True)

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
            print("\nInterrupted; re-run the same command to resume.", flush=True)
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
