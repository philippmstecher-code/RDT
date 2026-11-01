#!/usr/bin/env python3
"""Task-irrelevant superclasses (ResNet-18, CIFAR-100).

One lane per seed: the recipe of the clean 30-epoch runs (template: SGD 0.1, nesterov,
wd 5e-4, cosine, batch 128), but every training AND validation label is the index of
the image's class within its CIFAR-100 superclass (alphabetical order of fine-class
names, 0-4). The head has 5 outputs; each output pools one class from every
superclass, so superclass identity carries no label information.

Implementation: a single curriculum phase (epochs 1..E, label_mode
'within_superclass_index') in devtrain of the task-irrelevant training code (WT; variants/task_irrelevant). The phase machinery swaps in a fresh 5-way head at epoch 1
(the backbone keeps the shared seed init) and builds a fresh optimizer + cosine over
the phase (= the whole run). Snapshots (t0 + every `--every` epochs) capture
activations on the FINE-labelled validation loader (curriculum_swap), so the SAE /
RDT analysis computes selectivity against the true fine classes and superclasses.

Data go to that checkout's `data` directory.
Usage: python scripts/paper/run_taskirr.py --seed S --template-json T [--epochs 30] [--every 5]
       [--with-sae]
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

WT = Path(RDT_DATA + "/checkouts/task_irrelevant")
spec = importlib.util.spec_from_file_location("ncr", WT / "scripts" / "run_noise_curriculum_experiment.py")
ncr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ncr)  # sets up sys.path for packages/ml/src from the checkout

PRODUCTION_EXP = "taskirr"
LABEL_MODE = "within_superclass_index"


def build_cfg(template, seed, epochs, curriculum, every, bpe):
    from run_devtrain_direct import dict_to_namespace, add_missing_defaults
    cfg = json.loads(json.dumps(template))
    cfg["training_policy"]["epochs"] = epochs
    if cfg["training_policy"].get("accuracy_threshold") is None:
        cfg["training_policy"]["accuracy_threshold"] = 100.0
    cfg["snapshot_policy"].setdefault("milestone_accuracies", [50.0])
    cfg["random_seed"] = seed
    cfg["network_type"] = "resnet18"
    cfg["noise_policy"] = None
    cfg["representation_methods"] = []
    cfg["curriculum_policy"] = curriculum
    eps = [e for e in range(every, epochs + 1, every)]
    cfg["snapshot_policy"]["milestone_weight_updates"] = [e * bpe for e in eps]
    cfg["snapshot_policy"]["milestone_count"] = len(eps)
    ns = dict_to_namespace(cfg)
    add_missing_defaults(ns, cfg)
    return ns, eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-id", default=PRODUCTION_EXP)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--template-json", required=True)
    ap.add_argument("--with-sae", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    # Smoke-test knobs (never with the production experiment id).
    ap.add_argument("--smoke-per-class", type=int, default=0,
                    help="SMOKE: subsample train/val to N images per fine class")
    ap.add_argument("--smoke-cpu", action="store_true", help="SMOKE: force CPU (hide MPS)")
    ap.add_argument("--smoke-sae-steps", type=int, default=0,
                    help="SMOKE: override sae_policy n_steps (and null_permutations=20)")
    args = ap.parse_args()

    smoke = bool(args.smoke_per_class or args.smoke_cpu or args.smoke_sae_steps)
    if smoke and args.experiment_id == PRODUCTION_EXP:
        sys.exit(f"smoke options cannot be used with the production experiment id {PRODUCTION_EXP}")

    import torch
    if args.smoke_cpu:
        torch.backends.mps.is_available = lambda: False
    template = json.loads(Path(args.template_json).read_text())
    sae_policy = dict(template.get("sae_policy", {}))
    if args.smoke_sae_steps:
        sae_policy.update(n_steps=args.smoke_sae_steps, null_permutations=20)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tpl_bs = template.get("batch_size", 128)
    eff_bs = min(int(os.environ.get("DEVTRAIN_MPS_MAX_BATCH", "16")), tpl_bs) if dev == "mps" else tpl_bs
    n_train = 100 * args.smoke_per_class if args.smoke_per_class else 50000
    bpe = math.ceil(n_train / eff_bs)

    exp_dir = ncr.DATA_DIR / "experiments" / args.experiment_id
    lr = template["optimizer"]["learning_rate"]
    cur = {"enabled": True, "phases": [
        {"start_epoch": 1, "end_epoch": args.epochs, "label_mode": LABEL_MODE, "learning_rate": lr}]}
    lid = f"taskirr_within_index_seed{args.seed}"
    print(f"exp={args.experiment_id} seed={args.seed} device={dev} eff_bs={eff_bs} bpe={bpe} "
          f"-> {exp_dir}")
    print(f"  {lid:<32} epochs={args.epochs} label_mode={LABEL_MODE} (5 outputs) lr={lr} "
          f"snapshots=t0+every {args.every} ep, sae={'yes' if args.with_sae else 'no'} {sae_policy}")
    if args.plan_only:
        return

    if args.smoke_per_class:
        import devtrain
        from torch.utils.data import Subset
        _orig_get_dataset = devtrain.get_dataset

        def _subset_get_dataset(*a, **kw):
            ds, names = _orig_get_dataset(*a, **kw)
            try:  # get_dataset returns RemappedDataset(Subset(CIFAR100, indices), mapping)
                base = ds.subset
                targets = [ds.mapping[base.dataset.targets[j]] for j in base.indices]
            except AttributeError:
                targets = [ds[i][1] for i in range(len(ds))]
            seen, idx = {}, []
            for i, t in enumerate(targets):
                t = int(t)
                if seen.get(t, 0) < args.smoke_per_class:
                    seen[t] = seen.get(t, 0) + 1
                    idx.append(i)
            print(f"  [SMOKE] subset {len(idx)}/{len(ds)} samples", flush=True)
            return Subset(ds, idx), names

        devtrain.get_dataset = _subset_get_dataset

    exp_dir.mkdir(parents=True, exist_ok=True)
    ncr.ensure_dataset(args.experiment_id)
    classes = ncr.canonical_classes(args.experiment_id)
    lane = exp_dir / "lanes" / lid
    lane.mkdir(parents=True, exist_ok=True)
    if (lane / ".lane_done").exists():
        print(f"{lid}: done, skip", flush=True)
        return
    t0 = time.time()
    init = ncr.ensure_seed_init(exp_dir, "resnet18", args.seed, template)
    cfg, eps = build_cfg(template, args.seed, args.epochs, cur, args.every, bpe)
    (lane / "_config").mkdir(exist_ok=True)
    (lane / "_config" / "lane.json").write_text(json.dumps(
        {"lane": lid, "epochs": args.epochs, "curriculum": cur, "milestone_epochs": eps,
         "seed": args.seed, "label_mode": LABEL_MODE, "num_outputs": 5,
         "checkout": str(WT), "sae_policy": sae_policy, "smoke": smoke,
         "pythonhashseed": os.environ.get("PYTHONHASHSEED")}, indent=2))
    print(f"[{lid}] DEVTRAIN_START milestones={eps}", flush=True)
    ncr.run_devtrain(lane, cfg, init, classes, args.experiment_id)
    if args.with_sae:
        ncr.run_sae(lane, classes, sae_policy)
    (lane / ".lane_done").touch()
    print(f"[{lid}] LANE_DONE mins={(time.time() - t0) / 60:.1f}", flush=True)


if __name__ == "__main__":
    main()
