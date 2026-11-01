#!/usr/bin/env python3
"""
Orchestrator for the mid-training scaffold lesion experiments.

Intervenes on the representation: at a branch checkpoint of a finished clean run,
select SAE features per tracked layer (scaffold = top-N high-SSI, refinement
control = high-CSI, random control = random alive), then continue training with
forward hooks that project activations orthogonally to the frozen decoder
directions of the selected features for a window of epochs, release, and train
to completion (see packages/ml/src/lesion.py).

Per lesion lane:
    SPEC      (build lesion_spec.json from the clean lane's branch-checkpoint
               SAE artifacts; exact variance matching across conditions)
      -> DEVTRAIN  (resume from the clean lane's milestone checkpoint +
                    optimizer/scheduler state; dense per-epoch snapshots through
                    the window and ~5 recovery epochs; superclass-probe
                    manipulation check on lesioned activations)
      -> SAEANALYSIS (standard per-checkpoint pipeline -> sae_results.json)

The clean lane (default
data/experiments/noise_curriculum_arch/lanes/standard_resnet18, seed 42) is
both the sham and the shared prefix: every lesion lane loads its milestone
checkpoint.pt and training_state.pt, so seed 42 needs no clean retraining.
For other seeds the orchestrator first trains an in-experiment clean_seed{S}
lane on the standard 5-epoch grid.

Default design: seed 42; branches e5..e25 x window 5; 3 conditions; count- and
variance-matched control lanes (scaffold lanes are shared between matchings
since the scaffold set defines the variance target): 25 lesion lanes in total.

Idempotent, resumable and continues on failure, with the same conventions as
run_noise_curriculum_experiment.py. Typical invocations:

    # pilot: e20 scaffold + e20 random, variance-matched
    python -u scripts/run_scaffold_lesion_experiment.py --pilot 2>&1 | tee -a scaffold_lesion.out

    # full seed-42 sweep (resumable; pilot lanes are skipped as already done)
    python -u scripts/run_scaffold_lesion_experiment.py 2>&1 | tee -a scaffold_lesion.out
"""
import argparse
import json

import numpy as np
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ML_SRC = REPO / "packages" / "ml" / "src"
BE_SRC = REPO / "packages" / "backend" / "src"
# ML_SRC must precede BE_SRC so bare `models` resolves to packages/ml/src/models.py.
sys.path.insert(0, str(BE_SRC))
sys.path.insert(0, str(ML_SRC))

DATA_DIR = REPO / "data"
DEFAULT_CLEAN_LANE = (DATA_DIR / "experiments" / "noise_curriculum_arch"
                      / "lanes" / "standard_resnet18")
TRACKED_LAYERS = ["layer1", "layer2", "layer3", "layer4", "avgpool"]
N_TRAIN = 50000  # CIFAR-100 train set

# Matches the clean lane (noise_curriculum_arch/standard_resnet18).
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
    "representation_methods": [],
    "validation_policy": {}, "memory_policy": {}, "logging_policy": {},
    "transform_config": None,
}


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_status(exp_dir, event, **kw):
    rec = {"ts": now(), "event": event, **kw}
    with open(exp_dir / "ORCHESTRATOR_STATUS.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    extra = " ".join(f"{k}={v}" for k, v in kw.items())
    print(f"[{rec['ts']}] {event} {extra}", flush=True)


# Template, dataset and classes (same conventions as the noise orchestrator)
def load_template(args):
    if args.template_json:
        return json.loads(Path(args.template_json).read_text()), f"json:{args.template_json}"
    cand = Path(args.clean_lane).parent.parent / "_config" / "template.json"
    if cand.exists():
        return json.loads(cand.read_text()), f"clean-exp:{cand}"
    print("  WARNING: no template found: using EMBEDDED_TEMPLATE fallback.", flush=True)
    return json.loads(json.dumps(EMBEDDED_TEMPLATE)), "embedded"


def ensure_dataset(exp_id):
    ds_dir = DATA_DIR / "experiments" / exp_id / "datasets" / "cifar100"
    if (ds_dir / "cifar-100-python").exists():
        return
    ds_dir.mkdir(parents=True, exist_ok=True)
    for cand in (DATA_DIR / "experiments").glob("*/datasets/cifar100/cifar-100-python"):
        print(f"  linking CIFAR-100 from existing copy: {cand.parent}", flush=True)
        if (ds_dir / "cifar-100-python").exists():
            break
        os.symlink(cand, ds_dir / "cifar-100-python")
        return
    print("  downloading CIFAR-100 via torchvision (one-time)...", flush=True)
    import torchvision
    torchvision.datasets.CIFAR100(root=str(ds_dir), train=True, download=True)
    torchvision.datasets.CIFAR100(root=str(ds_dir), train=False, download=True)


def canonical_classes(exp_id):
    import torchvision
    ds = torchvision.datasets.CIFAR100(
        root=str(DATA_DIR / "experiments" / exp_id / "datasets" / "cifar100"),
        train=False, download=False)
    return list(ds.classes)


def effective_batches_per_epoch(template):
    """Mirror devtrain's MPS batch cap so milestone WU offsets match training."""
    import torch
    template_bs = template.get("batch_size", 128)
    device = ('cuda' if torch.cuda.is_available()
              else 'mps' if torch.backends.mps.is_available() else 'cpu')
    mps_max_batch = int(os.environ.get("DEVTRAIN_MPS_MAX_BATCH", "16"))
    eff_bs = mps_max_batch if (device == 'mps' and template_bs > mps_max_batch) else template_bs
    return math.ceil(N_TRAIN / eff_bs), eff_bs


# Lane plan
def lane_plan(args):
    """Return list of lane dicts. Clean lanes (for seeds without an external
    clean lane) come first so a single sequential pass resolves dependencies.

    Scaffold lanes are shared across matchings: the scaffold set defines the
    variance target, so its count- and variance-matched selections coincide.
    """
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    branches = [int(b) for b in args.branch_epochs.split(",") if b.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    matchings = [m.strip() for m in args.matchings.split(",") if m.strip()]
    layers = (None if args.layers == "all"
              else [l.strip() for l in args.layers.split(",") if l.strip()])
    # The primary code the controls variance-match: SAI (taskgeneral) if it
    # is being run, else SSI (scaffold). A run mixes one primary code with its
    # dose-matched controls, never two primaries.
    target_code = "taskgeneral" if "taskgeneral" in conditions else "scaffold"
    primary_cond = target_code  # 'scaffold'|'taskgeneral': the matching-invariant lane
    layer_suffix = "" if args.layers == "all" else "_L" + "-".join(layers)

    plan = []
    for seed in seeds:
        if seed == args.clean_lane_seed:
            clean_dir = Path(args.clean_lane)
        else:
            clean_dir = None  # resolved to in-experiment clean lane below
            plan.append({"kind": "clean", "lane_id": f"clean_seed{seed}", "seed": seed})
        for b in branches:
            for cond in conditions:
                if cond in ("scaffold", "taskgeneral", "randgroup", "finecode", "gradsc", "gradrand", "gradenergy", "gradrg", "gradjoint", "gradrandfresh"):
                    keys = [("shared", "variance")]  # one lane, matching-invariant
                else:
                    keys = [(m, m) for m in matchings]
                if args.reproject_every and cond == "refinement":
                    continue  # escalation design: primary + dose-matched random only
                for tag, matching in keys:
                    suffix = "" if tag == "shared" else f"_{tag}"
                    lid = f"lesion_{cond}{suffix}_e{b:02d}_seed{seed}"
                    if args.reproject_every:
                        lid += f"_rp{args.reproject_every}"
                    lid += layer_suffix
                    plan.append({
                        "kind": "lesion", "lane_id": lid, "seed": seed,
                        "branch": b, "condition": cond, "matching": matching,
                        "layers": layers, "clean_dir": clean_dir,
                        "target_code": target_code,
                    })
    if args.pilot:
        rp = f"_rp{args.reproject_every}" if args.reproject_every else ""
        keep = {f"lesion_{primary_cond}_e20_seed{seeds[0]}{rp}{layer_suffix}",
                f"lesion_random_variance_e20_seed{seeds[0]}{rp}{layer_suffix}"}
        plan = [l for l in plan if l["lane_id"] in keep]
    return plan


# Config building
def dense_milestone_epochs(branch, window, total_epochs, post_release_dense):
    """Per-epoch snapshots from branch+1 through release+post_release_dense,
    plus the remaining standard 5-epoch grid points, deduped."""
    dense_end = min(branch + window + post_release_dense, total_epochs)
    epochs = set(range(branch + 1, dense_end + 1))
    epochs.update(e for e in range(5, total_epochs + 1, 5) if e > branch)
    return sorted(epochs)


def build_lesion_lane_config(template, args, lane, spec_path, bpe):
    """Continuation config: dense branch-relative WU milestones + lesion_policy."""
    from run_devtrain_direct import dict_to_namespace, add_missing_defaults
    cfg = json.loads(json.dumps(template))
    epochs = args.epochs
    b, window = lane["branch"], args.window
    cfg["training_policy"]["epochs"] = epochs
    if cfg["training_policy"].get("accuracy_threshold") is None:
        cfg["training_policy"]["accuracy_threshold"] = 100.0
    cfg["snapshot_policy"].setdefault("milestone_accuracies", [50.0])
    cfg["random_seed"] = lane["seed"]
    cfg["network_type"] = "resnet18"
    cfg["curriculum_policy"] = None
    cfg["noise_policy"] = None
    cfg["blur_policy"] = None
    # Skip per-class-mean REPMETRICS (not used downstream).
    cfg["representation_methods"] = []

    # Branch-relative weight-update milestones (resume resets the WU counter).
    mile_epochs = dense_milestone_epochs(b, window, epochs, args.post_release_dense)
    cfg["snapshot_policy"]["milestone_weight_updates"] = [
        (e - b) * bpe for e in mile_epochs]
    cfg["snapshot_policy"]["milestone_count"] = len(mile_epochs)

    cfg["lesion_policy"] = {
        "enabled": True,
        "spec_path": str(spec_path),
        # window = epochs [b+1, b+window]; hooks active iff onset <= epoch < offset
        "onset_epoch": b + 1,
        "offset_epoch": b + 1 + window,
        "layers": lane["layers"],
        "lesion_during_capture": True,
        # Probe all tracked layers, not only lesioned ones: if the forming
        # scaffold is displaced upstream (e.g. layer3) under suppression, the
        # per-layer probe curves are what reveal it.
        # SAI/task-general lanes use a 100-way task probe as the binding check;
        # the superclass probe is blind to SAI removal (see lesion.py).
        "probe_check": {"enabled": True, "every_epochs": args.probe_every,
                        "samples_per_class": 50, "layers": TRACKED_LAYERS,
                        "label_level": ("class" if lane.get("target_code") == "taskgeneral"
                                        else "superclass")},
        # Escalation: refit and remove the rebuilt linear superclass code every
        # k updates. The random condition adds dose-matched random directions
        # instead (same count the superclass refit would have added).
        "reproject_every_k_updates": args.reproject_every,
        "reproject_mode": ("random" if lane["condition"] == "random" else
                           "taskgeneral" if lane["condition"] == "taskgeneral" else
                           "randgroup" if lane["condition"] == "randgroup" else
                           "finecode" if lane["condition"] == "finecode" else
                           "random" if lane["condition"] == "gradrand" else
                           "gradenergy" if lane["condition"] == "gradenergy" else
                           "gradjoint" if lane["condition"] == "gradjoint" else
                           "gradrandfresh" if lane["condition"] == "gradrandfresh" else
                           "randgroup" if lane["condition"] == "gradrg" else "superclass"),
        "reproject_q_max": args.reproject_q_max,
        "reproject_evr_floor": args.reproject_evr_floor,
        "reproject_max_rank_frac": args.reproject_max_rank_frac,
        "reproject_rank_policy": args.reproject_rank_policy,
        "reproject_samples_per_class": 20,
    }
    ns = dict_to_namespace(cfg)
    add_missing_defaults(ns, cfg)
    return ns, mile_epochs


def build_clean_lane_config(template, args, seed, bpe):
    """Standard 5-epoch-grid config for an in-experiment clean lane."""
    from run_devtrain_direct import dict_to_namespace, add_missing_defaults
    cfg = json.loads(json.dumps(template))
    cfg["training_policy"]["epochs"] = args.epochs
    if cfg["training_policy"].get("accuracy_threshold") is None:
        cfg["training_policy"]["accuracy_threshold"] = 100.0
    cfg["snapshot_policy"].setdefault("milestone_accuracies", [50.0])
    cfg["random_seed"] = seed
    cfg["network_type"] = "resnet18"
    cfg["curriculum_policy"] = None
    cfg["noise_policy"] = None
    cfg["representation_methods"] = []
    grid = [e for e in range(5, args.epochs + 1, 5)]
    cfg["snapshot_policy"]["milestone_weight_updates"] = [e * bpe for e in grid]
    cfg["snapshot_policy"]["milestone_count"] = len(grid)
    ns = dict_to_namespace(cfg)
    add_missing_defaults(ns, cfg)
    return ns


# Stages
def ensure_seed_init(exp_dir, seed, template):
    init_path = exp_dir / "_inits" / f"resnet18_seed{seed}.pt"
    if init_path.exists():
        return init_path
    init_path.parent.mkdir(parents=True, exist_ok=True)
    from models import create_network
    from initialization import initialize_weights, save_weights
    model = create_network(network_type="resnet18", num_classes=100)
    initialize_weights(model, method=template["initialization_method"], seed=seed)
    save_weights(model, str(init_path))
    return init_path


def assert_grid_alignment(clean_dir, branch, bpe):
    """The lesion lane's batches/epoch must reproduce the clean lane's grid,
    otherwise branch-relative WU offsets land on the wrong epochs."""
    if branch == 0:
        return  # t0 branch (prevention-from-birth): trivially aligned
    policy_file = clean_dir / "dev_snapshots" / "metrics" / "devtrain_snapshot_policy.json"
    policy = json.loads(policy_file.read_text())
    clean_wus = policy.get("milestone_weight_updates", [])
    expected = branch * bpe
    if expected not in clean_wus:
        raise RuntimeError(
            f"GRID MISMATCH: branch epoch {branch} x {bpe} batches/epoch = {expected} WU, "
            f"but the clean lane's milestones are {clean_wus}. Effective batch differs "
            f"from the clean run: set DEVTRAIN_MPS_MAX_BATCH so the effective batch "
            f"matches (clean lane used {clean_wus[0] // 5 if clean_wus else '?'} batches/epoch).")


def run_spec(lane_dir, lane, args, artifacts_cache):
    """SPEC stage: build + persist lesion_spec.json from the clean lane."""
    from lesion import build_lesion_spec, load_branch_artifacts
    clean_dir = lane["clean_dir"]
    layers = lane["layers"] or TRACKED_LAYERS
    cache_key = (str(clean_dir), lane["branch"], tuple(layers))
    if cache_key not in artifacts_cache:
        artifacts_cache[cache_key] = load_branch_artifacts(
            clean_dir, lane["branch"], layers)
    if lane["condition"] == "randgroup":
        return run_spec_randgroup(lane_dir, lane, args, artifacts_cache[cache_key])
    if lane["condition"] == "finecode":
        return run_spec_finecode(lane_dir, lane, args, artifacts_cache[cache_key])
    if lane["condition"] in ("gradsc", "gradrand", "gradenergy", "gradjoint", "gradrandfresh"):
        return run_spec_gradblock(lane_dir, lane, args, artifacts_cache[cache_key])
    if lane["condition"] == "gradrg":
        # acquisition block along a fixed random grouping's code: the random-grouping spec (same grouping as the
        # suppression arm, RandomState(10000 + seed)) with the gradient hook instead of the forward projection
        spec_path, spec = run_spec_randgroup(lane_dir, lane, args, artifacts_cache[cache_key])
        spec["lesion_type"] = "gradrg"
        spec["hook_kind"] = "gradient"
        spec_path.write_text(json.dumps(spec, indent=2))
        return spec_path, spec
    spec = build_lesion_spec(
        clean_dir, lane["branch"], layers,
        lesion_type=lane["condition"], top_n=args.top_n,
        matching=lane["matching"], seed=lane["seed"],
        variance_tol=args.variance_tol,
        ssi_gate=not args.no_ssi_gate,
        target_code=lane.get("target_code", "scaffold"),
        _artifacts=artifacts_cache[cache_key])
    spec_path = lane_dir / "lesion_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    for layer, why in spec.get("excluded_layers", {}).items():
        print(f"    spec {layer}: EXCLUDED: {why['reason']}", flush=True)
    for layer, ld in spec["layers"].items():
        print(f"    spec {layer}: n={ld['actual_n']} rank={ld['subspace_rank']} "
              f"var={ld['exact_subspace_variance_fraction']:.4f} "
              f"(target {ld['target_variance_fraction']:.4f}, "
              f"feasible={ld['variance_match_feasible']})", flush=True)
    return spec_path, spec


def random_grouping(seed, n_classes=100, group_size=5):
    """One fixed random grouping per seed: RandomState(10000 + seed).permutation(n).reshape(-1, group_size), row = group;
    returns grouping[class index] = group index."""
    perm = np.random.RandomState(10000 + seed).permutation(n_classes).reshape(-1, group_size)
    grouping = [0] * n_classes
    for g, row in enumerate(perm):
        for c in row:
            grouping[int(c)] = g
    return grouping


def run_spec_randgroup(lane_dir, lane, args, artifacts):
    """SPEC stage of the random-grouping control: the superclass arm's spec structure (same branch checkpoint, layers
    and initial rank), with the initial subspace replaced by the leading between-group directions of a fixed random
    grouping of the classes, estimated on the branch checkpoint's stored activations; re-estimation then uses the
    same grouping (lesion.LesionController.reproject, mode 'randgroup')."""
    import torch
    from lesion import build_lesion_spec, superclass_code_directions
    layers = lane["layers"] or TRACKED_LAYERS
    spec = build_lesion_spec(
        lane["clean_dir"], lane["branch"], layers, lesion_type="scaffold", top_n=args.top_n,
        matching="variance", seed=lane["seed"], variance_tol=args.variance_tol,
        ssi_gate=not args.no_ssi_gate, target_code="scaffold", _artifacts=artifacts)
    grouping = random_grouping(lane["seed"])
    n_cls = len(grouping)
    for layer, ld in spec["layers"].items():
        X = artifacts["per_layer"][layer]["X"]
        n_per = X.shape[0] // n_cls
        assert n_per * n_cls == X.shape[0], (layer, X.shape)
        y = torch.tensor([grouping[c] for c in range(n_cls) for _ in range(n_per)], dtype=torch.long)
        r = int(ld["subspace_rank"])
        dirs, evr = superclass_code_directions(X, y, q_max=r, evr_floor=0.0)
        assert dirs.shape[1] == r, (layer, dirs.shape, r)
        path = lane_dir / f"randgroup_basis_{layer}.npy"
        np.save(path, dirs.numpy().astype(np.float32))
        ld["basis_path"] = str(path)
        ld["randgroup_initial_rank"] = r
        ld["randgroup_initial_between_group_evr"] = [round(float(v), 5) for v in evr[:r]]
        print(f"    spec {layer}: randgroup basis rank={r} (superclass arm initial rank {r}); "
              f"between-group EVR of these directions {sum(evr[:r]):.3f}", flush=True)
    spec["lesion_type"] = "randgroup"
    spec["grouping"] = grouping
    spec["grouping_rule"] = "np.random.RandomState(10000 + seed).permutation(100).reshape(20, 5); row = group"
    spec_path = lane_dir / "lesion_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    return spec_path, spec


def run_spec_finecode(lane_dir, lane, args, artifacts):
    """SPEC stage of the fine-code control: the superclass arm's spec structure (same branch checkpoint, layers and
    initial rank), with the initial subspace replaced by the leading directions of the within-superclass between-class
    scatter (lesion.finecode_directions), estimated on the branch checkpoint's stored activations; re-estimation then
    adds as many within-superclass class directions as the superclass refit would add (mode 'finecode')."""
    import torch
    from cifar100_hierarchy import build_superclass_map
    from lesion import build_lesion_spec, finecode_directions
    layers = lane["layers"] or TRACKED_LAYERS
    spec = build_lesion_spec(
        lane["clean_dir"], lane["branch"], layers, lesion_type="scaffold", top_n=args.top_n,
        matching="variance", seed=lane["seed"], variance_tol=args.variance_tol,
        ssi_gate=not args.no_ssi_gate, target_code="scaffold", _artifacts=artifacts)
    classes = canonical_classes(args.experiment_id)
    sc_map = build_superclass_map(classes, {n: i for i, n in enumerate(classes)})
    sc_idx = {s_: i for i, s_ in enumerate(sorted(set(sc_map.values())))}
    n_cls = len(classes)
    for layer, ld in spec["layers"].items():
        X = artifacts["per_layer"][layer]["X"]
        n_per = X.shape[0] // n_cls
        assert n_per * n_cls == X.shape[0], (layer, X.shape)
        y_cls = torch.tensor([c for c in range(n_cls) for _ in range(n_per)], dtype=torch.long)
        y_sc = torch.tensor([sc_idx[sc_map[c]] for c in range(n_cls) for _ in range(n_per)], dtype=torch.long)
        r = int(ld["subspace_rank"])
        from lesion import superclass_code_directions
        sc_full, _ = superclass_code_directions(X, y_sc, q_max=19, evr_floor=1e-6)
        dirs, evr = finecode_directions(X, y_cls, y_sc, n_dirs=r, exclude=sc_full)
        assert dirs.shape[1] == r, (layer, dirs.shape, r)
        path = lane_dir / f"finecode_basis_{layer}.npy"
        np.save(path, dirs.numpy().astype(np.float32))
        ld["basis_path"] = str(path)
        ld["finecode_initial_rank"] = r
        ld["finecode_initial_within_sc_evr"] = [round(float(v), 5) for v in evr[:r]]
        print(f"    spec {layer}: finecode basis rank={r}; within-superclass between-class EVR {sum(evr[:r]):.3f}",
              flush=True)
    spec["lesion_type"] = "finecode"
    spec_path = lane_dir / "lesion_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    return spec_path, spec


def run_spec_gradblock(lane_dir, lane, args, artifacts):
    """SPEC stage of the acquisition-blocking arms (hook_kind 'gradient': forward unchanged, gradient projected off
    the removed directions). Same branch, layers and initial rank as the superclass arm. gradsc: initial subspace =
    leading between-superclass directions of the branch checkpoint's stored activations, re-estimated as the
    superclass code (mode 'superclass'); gradrand: random orthonormal directions of the same rank, extended by as
    many random directions as the superclass refit would add (mode 'random'); gradenergy: energy-matched control,
    blocking the leading learning-signal directions outside the current superclass code, as large a share of the
    learning signal as the gradsc arm would block (mode 'gradenergy')."""
    import torch
    from cifar100_hierarchy import build_superclass_map
    from lesion import build_lesion_spec, superclass_code_directions
    layers = lane["layers"] or TRACKED_LAYERS
    spec = build_lesion_spec(
        lane["clean_dir"], lane["branch"], layers, lesion_type="scaffold", top_n=args.top_n,
        matching="variance", seed=lane["seed"], variance_tol=args.variance_tol,
        ssi_gate=not args.no_ssi_gate, target_code="scaffold", _artifacts=artifacts)
    classes = canonical_classes(args.experiment_id)
    sc_map = build_superclass_map(classes, {n: i for i, n in enumerate(classes)})
    sc_idx = {s_: i for i, s_ in enumerate(sorted(set(sc_map.values())))}
    n_cls = len(classes)
    g = torch.Generator().manual_seed(20000 + lane["seed"])
    for layer, ld in spec["layers"].items():
        X = artifacts["per_layer"][layer]["X"]
        n_per = X.shape[0] // n_cls
        assert n_per * n_cls == X.shape[0], (layer, X.shape)
        r = int(ld["subspace_rank"])
        if lane["condition"] in ("gradsc", "gradenergy", "gradjoint", "gradrandfresh"):
            # gradenergy / gradjoint: this superclass basis is the controller's shadow (never applied); see lesion._gradenergy_step
            y_sc = torch.tensor([sc_idx[sc_map[c]] for c in range(n_cls) for _ in range(n_per)], dtype=torch.long)
            dirs, _ = superclass_code_directions(X, y_sc, q_max=r, evr_floor=0.0)
        else:
            dirs, _ = torch.linalg.qr(torch.randn(X.shape[1], r, generator=g))
        assert dirs.shape[1] == r, (layer, dirs.shape, r)
        path = lane_dir / f"gradblock_basis_{layer}.npy"
        np.save(path, dirs.numpy().astype(np.float32))
        ld["basis_path"] = str(path)
        print(f"    spec {layer}: {lane['condition']} initial rank={r}", flush=True)
    spec["lesion_type"] = lane["condition"]
    spec["hook_kind"] = "gradient"
    spec_path = lane_dir / "lesion_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    return spec_path, spec


def run_lesion_devtrain(lane_dir, cfg, lane, spec, selected_classes, exp_id, n_milestones):
    """DEVTRAIN stage: resume from the clean lane's branch milestone."""
    from devtrain import train_with_snapshots
    clean_dir = lane["clean_dir"]
    m = spec["branch_ckpt_label"]
    ckpt = clean_dir / "dev_snapshots" / f"milestone_{m}" / "checkpoint.pt"
    tstate = clean_dir / "dev_snapshots" / f"milestone_{m}" / "training_state.pt"
    if not ckpt.exists() or not tstate.exists():
        raise RuntimeError(f"missing branch checkpoint/training_state under {ckpt.parent}")
    snaps = lane_dir / "dev_snapshots"
    metrics = snaps / "metrics"
    snaps.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    resume_state = {
        "checkpoint_path": str(ckpt),
        # The branch milestone is captured at the last batch of the branch epoch,
        # so the checkpoint has completed `branch` epochs -> resume at branch+1.
        # (devtrain tops up the restored scheduler to last_epoch = start_epoch-1.)
        "start_epoch": lane["branch"] + 1,
        "existing_snapshots": [],
        "next_milestone_index": 0,
        "milestones_reached": [False] * n_milestones,
        "milestones_wu_reached": [False] * n_milestones,
        "total_weight_updates": 0,
        "training_state_path": str(tstate),
        "capture_t0": True,  # milestone_0 = instantaneous lesioned footprint
    }
    return train_with_snapshots(
        initial_weights_path=str(ckpt), netinit_config=cfg,
        dataset_id="cifar100", selected_classes=selected_classes,
        snapshots_base_dir=str(snaps), metrics_dir=str(metrics),
        experiment_id=exp_id, stove_id=None, resume_state=resume_state)


def run_clean_devtrain(lane_dir, cfg, init_path, selected_classes, exp_id):
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


def clean_lane_ready(clean_dir):
    return (clean_dir / "sae_analysis" / "sae_results.json").exists()


def main():
    ap = argparse.ArgumentParser(description="Mid-training scaffold-lesion orchestrator")
    ap.add_argument("--experiment-id", default="scaffold_lesion")
    ap.add_argument("--clean-lane", default=str(DEFAULT_CLEAN_LANE),
                    help="finished clean lane used as sham + shared prefix")
    ap.add_argument("--clean-lane-seed", type=int, default=42,
                    help="seed of --clean-lane (other seeds get in-experiment clean lanes)")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--branch-epochs", default="5,10,15,20,25")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--top-n", type=int, default=32, help="scaffold features per layer")
    ap.add_argument("--conditions", default="scaffold,refinement,random",
                    help="scaffold (high-SSI) | taskgeneral (high-SAI) | refinement "
                         "(high-CSI control) | random (alive control). A run mixes ONE "
                         "primary code (scaffold|taskgeneral) with its matched controls; "
                         "controls variance-match whichever primary is present.")
    ap.add_argument("--matchings", default="count,variance",
                    help="control-lane matchings to run (scaffold lanes are shared)")
    ap.add_argument("--layers", default="all",
                    help="'all' (5 tracked layers) or comma list for single-layer arms")
    ap.add_argument("--variance-tol", type=float, default=0.10)
    ap.add_argument("--no-ssi-gate", action="store_true",
                    help="disable the adaptive-SSI-threshold gate (flat top-N at "
                         "every layer; lesions low-SSI features at layer1-3)")
    ap.add_argument("--reproject-every", type=int, default=None,
                    help="ESCALATION: refit+remove the rebuilt linear superclass "
                         "code every K weight updates during the window "
                         "(391 = once/epoch at batch 128). Random condition gets "
                         "dose-matched random directions. Drops refinement lanes.")
    ap.add_argument("--reproject-q-max", type=int, default=19)
    ap.add_argument("--reproject-evr-floor", type=float, default=0.02)
    ap.add_argument("--reproject-max-rank-frac", type=float, default=0.5)
    ap.add_argument("--reproject-rank-policy", default="accumulate",
                    choices=["accumulate", "sliding"],
                    help="'sliding' evicts the OLDEST dynamic directions at the "
                         "rank cap so suppression never saturates (use for long "
                         "windows / high-LR prevention arms)")
    ap.add_argument("--probe-every", type=int, default=1)
    ap.add_argument("--post-release-dense", type=int, default=5,
                    help="per-epoch snapshots for this many epochs after release")
    ap.add_argument("--pilot", action="store_true",
                    help="only e20 scaffold + e20 random (variance) for the first seed")
    ap.add_argument("--skip-sae", action="store_true")
    ap.add_argument("--abort-on-fail", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--template-json", default=None)
    args = ap.parse_args()

    # The clean lane trained with effective batch 128 (391 batches/epoch). Raise
    # the default MPS cap to 128 so the continuation reproduces that grid; the
    # assertion below fails on any residual mismatch.
    os.environ.setdefault("DEVTRAIN_MPS_MAX_BATCH", "128")

    exp_id = args.experiment_id
    exp_dir = DATA_DIR / "experiments" / exp_id
    plan = lane_plan(args)

    print("=" * 72)
    print(f"SCAFFOLD-LESION ORCHESTRATOR  exp={exp_id}")
    print(f"seeds={args.seeds} branches={args.branch_epochs} window={args.window} "
          f"topN={args.top_n} conditions={args.conditions} matchings={args.matchings} "
          f"layers={args.layers} pilot={args.pilot}")
    print(f"clean lane (sham + prefix): {args.clean_lane}")
    print(f"{len(plan)} lanes -> {exp_dir}")
    print("=" * 72, flush=True)
    for lane in plan:
        if lane["kind"] == "clean":
            print(f"  {lane['lane_id']:<44} clean 30-epoch run (prefix + sham)")
        else:
            print(f"  {lane['lane_id']:<44} branch=e{lane['branch']:02d} "
                  f"cond={lane['condition']:<10} matching={lane['matching']}")
    if args.plan_only:
        print("\n--plan-only: exiting without running.", flush=True)
        return

    if not DATA_DIR.exists():
        sys.exit(f"FATAL: {DATA_DIR} does not exist.")
    if not clean_lane_ready(Path(args.clean_lane)):
        sys.exit(f"FATAL: clean lane {args.clean_lane} has no sae_results.json: "
                 f"lesion specs need its SAE artifacts.")
    exp_dir.mkdir(parents=True, exist_ok=True)

    template, tsrc = load_template(args)
    bpe, eff_bs = effective_batches_per_epoch(template)
    log_status(exp_dir, "RUN_START", n_lanes=len(plan), template=tsrc,
               eff_batch=eff_bs, batches_per_epoch=bpe, pilot=args.pilot)
    (exp_dir / "_config").mkdir(exist_ok=True)
    (exp_dir / "_config" / "template.json").write_text(json.dumps(template, indent=2))
    (exp_dir / "_config" / "run_args.json").write_text(json.dumps(vars(args), indent=2))

    ensure_dataset(exp_id)
    selected_classes = canonical_classes(exp_id)
    assert len(selected_classes) == 100, f"expected 100 classes, got {len(selected_classes)}"

    artifacts_cache = {}
    done, skipped, failed = [], [], []
    for i, lane in enumerate(plan, 1):
        lane_id = lane["lane_id"]
        lane_dir = exp_dir / "lanes" / lane_id
        lane_dir.mkdir(parents=True, exist_ok=True)
        tag = f"[{i}/{len(plan)}] {lane_id}"
        if (lane_dir / ".lane_done").exists():
            skipped.append(lane_id)
            print(f"{tag}: already complete: skip", flush=True)
            continue
        try:
            t0 = time.time()
            if lane["kind"] == "clean":
                init_path = ensure_seed_init(exp_dir, lane["seed"], template)
                cfg = build_clean_lane_config(template, args, lane["seed"], bpe)
                if not (lane_dir / ".devtrain_done").exists():
                    log_status(exp_dir, "DEVTRAIN_START", lane=lane_id)
                    run_clean_devtrain(lane_dir, cfg, init_path, selected_classes, exp_id)
                    (lane_dir / ".devtrain_done").touch()
                    log_status(exp_dir, "DEVTRAIN_DONE", lane=lane_id,
                               mins=round((time.time() - t0) / 60, 1))
            else:
                # Resolve this seed's clean lane and gate on its SAE artifacts.
                clean_dir = lane["clean_dir"] or (exp_dir / "lanes" / f"clean_seed{lane['seed']}")
                lane["clean_dir"] = clean_dir
                if not clean_lane_ready(clean_dir):
                    log_status(exp_dir, "LANE_BLOCKED", lane=lane_id,
                               reason=f"clean lane {clean_dir.name} not SAE-complete")
                    failed.append(lane_id)
                    continue
                assert_grid_alignment(clean_dir, lane["branch"], bpe)

                if not (lane_dir / ".spec_done").exists():
                    log_status(exp_dir, "SPEC_START", lane=lane_id)
                    spec_path, spec = run_spec(lane_dir, lane, args, artifacts_cache)
                    (lane_dir / "_config").mkdir(exist_ok=True)
                    (lane_dir / "_config" / "lesion_args.json").write_text(json.dumps({
                        "branch": lane["branch"], "condition": lane["condition"],
                        "matching": lane["matching"], "window": args.window,
                        "top_n": args.top_n, "layers": lane["layers"],
                        "clean_lane": str(clean_dir)}, indent=2))
                    (lane_dir / ".spec_done").touch()
                    log_status(exp_dir, "SPEC_DONE", lane=lane_id)
                else:
                    spec_path = lane_dir / "lesion_spec.json"
                    spec = json.loads(spec_path.read_text())

                cfg, mile_epochs = build_lesion_lane_config(
                    template, args, lane, spec_path, bpe)
                if not (lane_dir / ".devtrain_done").exists():
                    log_status(exp_dir, "DEVTRAIN_START", lane=lane_id,
                               start_epoch=lane["branch"] + 1,
                               milestone_epochs=mile_epochs)
                    run_lesion_devtrain(lane_dir, cfg, lane, spec,
                                        selected_classes, exp_id, len(mile_epochs))
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
            print(f"{tag}: FAILED: {e} (see {lane_dir / 'FAILED.txt'})", flush=True)
            traceback.print_exc()
            if args.abort_on_fail:
                log_status(exp_dir, "RUN_ABORTED", failed=lane_id)
                sys.exit(1)

    log_status(exp_dir, "RUN_END", done=len(done), skipped=len(skipped), failed=len(failed))
    print("\n" + "=" * 72)
    print(f"SUMMARY  done={len(done)} skipped(prev)={len(skipped)} failed={len(failed)}")
    if failed:
        print("FAILED/BLOCKED lanes (re-run the same command to retry):")
        for l in failed:
            print(f"  - {l}")
    else:
        print("All lanes complete.")
    print("=" * 72, flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
