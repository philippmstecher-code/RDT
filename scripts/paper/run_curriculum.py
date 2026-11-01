#!/usr/bin/env python3
"""Semantic-control curriculum (ResNet-18, CIFAR-100).

Arms (one seed per invocation):
  sc_true   K epochs on the TRUE 20 superclasses, then fine classes (warm restart).
  sc_rand   K epochs on a RANDOM partition of the 100 classes into 20 groups of 5
            (same coarse-task size and schedule, no semantic alignment), then fine.
  scratch   fine classes only, trained for (epochs - K) epochs with its own cosine,
            i.e. the same fine-phase schedule as the curriculum arms.
  sc_blind  K epochs on the superclass-blind five-way task (label = index of the class within
            its true superclass, as the taskirr 'within_superclass_index' runs), then fine;
            same switch and schedule as sc_true / sc_rand. Needs the training code with the
            curriculum_policy 'coarse_labels' option (variants/blind_pretraining):
            CURRICULUM_WT=/data/rdt/checkouts/blind_pretraining.
  sc_rand5  K epochs on five random groups of 20 classes (coarse_labels 'random_groups', seed 2000 + seed): the
            label count of sc_blind without cutting across the superclasses; then fine (variants/random_groups_pretraining):
            CURRICULUM_WT=/data/rdt/checkouts/random_groups_pretraining.
  sc_sup5   K epochs on five groups of four whole superclasses (coarse_labels 'superclass_groups', seed
            3000 + seed): the label count of sc_blind and sc_rand5, aligned with the hierarchy; then fine
            (variants/superclass_groups_pretraining): CURRICULUM_WT=/data/rdt/checkouts/superclass_groups_pretraining.

Runs from a checkout of the training code (CURRICULUM_WT) so that the devtrain/curriculum code of
the published runs is used; data go to that checkout's `data` directory. Activations are
captured every `--every` epochs and at the switch; SAE analysis is optional (--with-sae).
Usage: python scripts/paper/run_curriculum.py --seed S --template-json T [--K 20] [--epochs 50]
       [--arms sc_true,sc_rand,scratch] [--with-sae]
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

WT = Path(os.environ.get("CURRICULUM_WT", RDT_DATA + "/checkouts/lesion"))
spec = importlib.util.spec_from_file_location("ncr", WT / "scripts" / "run_noise_curriculum_experiment.py")
ncr = importlib.util.module_from_spec(spec)
sys.argv_backup = list(sys.argv)
spec.loader.exec_module(ncr)  # sets up sys.path for packages/ml/src from the checkout


def build_cfg(template, seed, epochs, curriculum, every, switch, bpe):
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
    eps = sorted(set(list(range(every, epochs + 1, every)) + ([switch, switch + 1] if switch else [])))
    eps = [e for e in eps if 1 <= e <= epochs]
    cfg["snapshot_policy"]["milestone_weight_updates"] = [e * bpe for e in eps]
    cfg["snapshot_policy"]["milestone_count"] = len(eps)
    ns = dict_to_namespace(cfg)
    add_missing_defaults(ns, cfg)
    return ns, eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-id", default="curriculum")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--K", type=int, default=20, help="superclass-phase epochs")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--arms", default="sc_true,sc_rand,scratch")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--template-json", required=True)
    ap.add_argument("--with-sae", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    import math
    import os
    import torch
    template = json.loads(Path(args.template_json).read_text())
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    eff_bs = int(os.environ.get("DEVTRAIN_MPS_MAX_BATCH", "16")) if dev == "mps" else template.get("batch_size", 128)
    bpe = math.ceil(50000 / min(eff_bs, template.get("batch_size", 128)))

    exp_dir = ncr.DATA_DIR / "experiments" / args.experiment_id
    lr = template["optimizer"]["learning_rate"]
    plan = []
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if arm in ("sc_true", "sc_rand", "sc_blind", "sc_rand5", "sc_sup5"):
            cur = {"enabled": True, "phases": [
                {"start_epoch": 1, "end_epoch": args.K, "label_mode": "superclass", "learning_rate": lr},
                {"start_epoch": args.K + 1, "end_epoch": args.epochs, "label_mode": "fine", "learning_rate": lr}]}
            if arm == "sc_rand":
                cur["random_partition_seed"] = 1000 + args.seed
            if arm == "sc_blind":
                cur["coarse_labels"] = "within_superclass_index"
            if arm == "sc_rand5":
                cur.update(coarse_labels="random_groups", n_coarse_groups=5, coarse_group_seed=2000 + args.seed)
            if arm == "sc_sup5":
                cur.update(coarse_labels="superclass_groups", n_coarse_groups=5, coarse_group_seed=3000 + args.seed)
            plan.append((f"{arm}_K{args.K:02d}_seed{args.seed}", cur, args.epochs, args.K))
        elif arm == "scratch":
            plan.append((f"scratch_E{args.epochs - args.K:02d}_seed{args.seed}", None, args.epochs - args.K, None))
    print(f"exp={args.experiment_id} seed={args.seed} K={args.K} bpe={bpe} -> {exp_dir} (code {WT})")
    for lid, cur, ep, sw in plan:
        print(f"  {lid:<28} epochs={ep} curriculum={'none' if cur is None else cur['phases'][0]['label_mode'] + ('-RANDOM' if 'random_partition_seed' in (cur or {}) else '')}")
    if args.plan_only:
        return

    exp_dir.mkdir(parents=True, exist_ok=True)
    ncr.ensure_dataset(args.experiment_id)
    classes = ncr.canonical_classes(args.experiment_id)
    for lid, cur, ep, sw in plan:
        lane = exp_dir / "lanes" / lid
        lane.mkdir(parents=True, exist_ok=True)
        if (lane / ".lane_done").exists():
            print(f"{lid}: done, skip", flush=True)
            continue
        t0 = time.time()
        init = ncr.ensure_seed_init(exp_dir, "resnet18", args.seed, template)
        cfg, eps = build_cfg(template, args.seed, ep, cur, args.every, sw, bpe)
        (lane / "_config").mkdir(exist_ok=True)
        (lane / "_config" / "lane.json").write_text(json.dumps(
            {"lane": lid, "epochs": ep, "curriculum": cur, "milestone_epochs": eps, "seed": args.seed}, indent=2))
        print(f"[{lid}] DEVTRAIN_START milestones={eps}", flush=True)
        ncr.run_devtrain(lane, cfg, init, classes, args.experiment_id)
        if args.with_sae:
            ncr.run_sae(lane, classes, template.get("sae_policy", {}))
        (lane / ".lane_done").touch()
        print(f"[{lid}] LANE_DONE mins={(time.time() - t0) / 60:.1f}", flush=True)


if __name__ == "__main__":
    main()
