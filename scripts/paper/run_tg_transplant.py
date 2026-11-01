#!/usr/bin/env python3
"""Task-general transplant test: do learned early (task-general) features speed up the formation of coarse structure?

Network ResNet-18, CIFAR-100, seeds 7, 42, 137 (the sham_e00 runs with per-epoch SAEs). Epochs 1-15 of the standard
30-epoch cosine schedule (T_max = 30, as the sham_e00 runs; devtrain's CosineAnnealingLR is wrapped), so the existing
sham_e00 runs (critical_sham weights, sham_perepoch per-epoch RDT, epochs 0-15) are the scratch reference (arm
A, not retrained). The stem (conv1, bn1) and layer1-2 are frozen in every other arm (requires_grad False: no update, no weight
decay; BatchNorm keeps batch statistics in training mode, so its running statistics follow the fixed layers).
  A        (reference, existing run) sham_e00: the seed's initial weights, all layers trainable
  Afrz     initial weights, stem + layer1-2 frozen at initialisation
  B        stem + layer1-2 transplanted from the same seed's standard run at the transplant epoch (1: tgtransplant;
           5: tgtransplant_e5, task-general features mature in layer1-2) and frozen;
           layer3, layer4, fc from initialisation
  C        B, with the task-general subspace projected out of the layer1 and layer2 outputs throughout training
           (span of the unit decoder directions of that layer's task-general features at the transplant epoch: per-epoch SAE
           sham_perepoch saes/<epoch>, stored per-run thresholds, 'methods' rule); activation projection, fixed basis
  Crand    B, with a random subspace of the same rank and the same captured activation variance projected out
           (Gaussian directions with covariance S^beta, beta set by bisection; tg_nesting.var_matched)
Snapshots at epochs 0, 0.25, 0.5, 1, 2, 3, 5, 8, 11 and 15 (weight-update milestones, 391 updates per epoch); RDT
(per-checkpoint SAE pipeline) on every lane after all training.
Writes /data/rdt/extended/experiments/<exp>/lanes/{arm}_seed{S}/ (exp tgtransplant by default).
Usage: python -u scripts/paper/run_tg_transplant.py [--seeds 7,42,137] [--arms ...] [--skip-sae]
       [--transplant-epoch 5 --exp tgtransplant_e5]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

if os.environ.get("PYTHONHASHSEED") != "0" or os.environ.get("DEVTRAIN_MPS_MAX_BATCH") != "128":
    env = dict(os.environ, PYTHONHASHSEED="0", DEVTRAIN_MPS_MAX_BATCH="128")
    os.execve(sys.executable, [sys.executable, "-u", *sys.argv], env)

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
WT = Path(RDT_DATA + "/checkouts/random_grouping")
sys.path.insert(0, str(WT / "scripts"))
sys.path.insert(0, str(REPO / "scripts/paper"))
import run_scaffold_lesion_experiment as R  # noqa: E402
import torch  # noqa: E402
import devtrain  # noqa: E402  (on sys.path through R)
from tg_nesting import cats, orth, var_matched, var_share  # noqa: E402

E = Path(RDT_DATA + "/extended/experiments")
EXP = "tgtransplant"
TEPOCH = 1                                                   # transplant epoch (--transplant-epoch)
TPL = Path(RDT_DATA + "/core/experiments/noise_curriculum_arch/_config/template.json")
EPOCHS = 15
MILESTONE_EPOCHS = [0.25, 0.5, 1, 2, 3, 5, 8, 11]          # intermediate; epoch 0 and the final epoch are captured too
FROZEN = ("conv1.", "bn1.", "layer1.", "layer2.")
LAYERS = ["layer1", "layer2"]
ARMS = ["Afrz", "B", "C", "Crand"]
SCHEDULE_EPOCHS = 30                                         # cosine T_max of the reference runs

_orig_create_optimizer = devtrain.create_optimizer
_FREEZE = {"on": False}


def _create_optimizer(model, cfg):
    """Freeze the stem and layer1-2 before the optimizer is built (no grad, hence no update and no weight decay)."""
    if _FREEZE["on"]:
        n = 0
        for name, p in model.named_parameters():
            if name.startswith(FROZEN):
                p.requires_grad_(False)
                n += p.numel()
        print(f"[TRANSPLANT] frozen parameters: {n}", flush=True)
    return _orig_create_optimizer(model, cfg)


devtrain.create_optimizer = _create_optimizer
_orig_cosine = devtrain.CosineAnnealingLR


def _cosine(optimizer, T_max, eta_min=0.0):
    """The reference runs' schedule: cosine over 30 epochs, of which the first 15 are trained."""
    return _orig_cosine(optimizer, T_max=SCHEDULE_EPOCHS, eta_min=eta_min)


devtrain.CosineAnnealingLR = _cosine


def sham_dir(s):
    return E / f"critical_sham/lanes/sham_e00_seed{s}"


def state(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    return sd.get("model_state_dict", sd) if isinstance(sd, dict) and "model_state_dict" in sd else sd


def make_init(s, arm, ld):
    """Initial weights of the lane: the seed's initialisation, with stem + layer1-2 from the transplant epoch for B, C,
    Crand."""
    snaps = json.load(open(sham_dir(s) / "dev_snapshots/metrics/devtrain_snapshots.json"))
    snaps = snaps.get("snapshots", snaps) if isinstance(snaps, dict) else snaps
    lab = {(x.get("snapshot_type"), int(x.get("weight_updates") or 0)): x["milestone_index"] for x in snaps}
    init = state(sham_dir(s) / "dev_snapshots" / f"milestone_{lab[('t0_resume', 0)]}" / "checkpoint.pt")
    sd = {k: v.clone() for k, v in init.items()}
    if arm in ("B", "C", "Crand"):
        e1 = state(sham_dir(s) / "dev_snapshots" / f"milestone_{lab[('intermediate', 391 * TEPOCH)]}" / "checkpoint.pt")
        moved = [k for k in sd if k.startswith(FROZEN)]
        for k in moved:
            sd[k] = e1[k].clone()
        print(f"[{arm}_seed{s}] transplanted {len(moved)} tensors from epoch {TEPOCH}", flush=True)
    p = ld / "init_weights.pt"
    torch.save(sd, p)
    return p, lab[("intermediate", 391 * TEPOCH)]


def build_spec(s, arm, ld, ms1):
    """Projection spec for C (task-general subspace) and Crand (rank- and variance-matched random subspace)."""
    sd = E / f"sham_perepoch/lanes/sham_e00_seed{s}/sae_analysis"
    thr = json.load(open(sd / "sae_results.json"))["metadata"]["analysis_thresholds"]
    t = {"sai": thr["sai_task_general_thresh"], "ssi": thr["ssi_abstraction_thresh"], "csi": thr["csi_differentiation_thresh"]}
    A = torch.load(sham_dir(s) / "dev_snapshots" / f"milestone_{ms1}" / "individual_multilayer_activations.pt",
                   map_location="cpu", weights_only=False)
    layers = {}
    rng = np.random.default_rng(500 + s)
    for L in LAYERS:
        H = np.load(sd / "activations" / str(TEPOCH) / f"{L}.npy").astype(np.float64)
        tg = cats(H, t)[1]
        D = torch.load(sd / "saes" / str(TEPOCH) / f"{L}.pt", map_location="cpu", weights_only=True)["decoder.weight"].double().numpy()
        D = D / np.maximum(np.linalg.norm(D, axis=0, keepdims=True), 1e-12)
        Q = orth(D[:, tg])
        X = torch.cat([A[L][c] for c in sorted(A[L])]).double().numpy()
        X = X - X.mean(0)
        S = X.T @ X / len(X)
        v_tg = var_share(Q, S)
        if arm == "Crand":
            Qs, _ = var_matched(S, Q.shape[1], v_tg, rng, 64)
            Q = Qs[int(np.argmin([abs(var_share(q, S) - v_tg) for q in Qs]))]
        path = ld / f"transplant_basis_{L}.npy"
        np.save(path, Q.astype(np.float32))
        layers[L] = {"basis_path": str(path), "subspace_rank": int(Q.shape[1]), "n_task_general": int(tg.sum()),
                     "var_share_tg": v_tg, "var_share_removed": var_share(Q, S)}
        print(f"[{arm}_seed{s}] {L}: rank {Q.shape[1]} of {D.shape[0]}, variance removed {var_share(Q, S):.3f} "
              f"(task-general {v_tg:.3f})", flush=True)
    return {"lesion_type": f"transplant_{arm}", "hook_kind": "activation", "sae_dir": str(sd / "saes" / str(TEPOCH)),
            "seed": s, "layers": layers}


def build_cfg(template, s, spec_path, bpe):
    from run_devtrain_direct import dict_to_namespace, add_missing_defaults
    cfg = json.loads(json.dumps(template))
    cfg["training_policy"]["epochs"] = EPOCHS
    if cfg["training_policy"].get("accuracy_threshold") is None:
        cfg["training_policy"]["accuracy_threshold"] = 100.0
    cfg["snapshot_policy"].setdefault("milestone_accuracies", [50.0])
    cfg["random_seed"] = s
    cfg["network_type"] = "resnet18"
    cfg["curriculum_policy"] = None
    cfg["noise_policy"] = None
    cfg["blur_policy"] = None
    cfg["representation_methods"] = []
    cfg["snapshot_policy"]["milestone_weight_updates"] = [int(round(e * bpe)) for e in MILESTONE_EPOCHS]
    cfg["snapshot_policy"]["milestone_count"] = len(MILESTONE_EPOCHS)
    if spec_path is not None:
        cfg["lesion_policy"] = {
            "enabled": True, "spec_path": str(spec_path), "onset_epoch": 1, "offset_epoch": None, "layers": LAYERS,
            "lesion_during_capture": True,
            "probe_check": {"enabled": False}, "reproject_every_k_updates": None}
    ns = dict_to_namespace(cfg)
    add_missing_defaults(ns, cfg)
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="7,42,137")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--skip-sae", action="store_true")
    ap.add_argument("--transplant-epoch", type=int, default=1)
    ap.add_argument("--exp", default="tgtransplant")
    a = ap.parse_args()
    global TEPOCH, EXP
    TEPOCH, EXP = a.transplant_epoch, a.exp
    seeds = [int(x) for x in a.seeds.split(",")]
    arms = a.arms.split(",")
    exp_dir = E / EXP
    exp_dir.mkdir(parents=True, exist_ok=True)
    template = json.loads(TPL.read_text())
    bpe, _ = R.effective_batches_per_epoch(template)
    assert bpe == 391, bpe
    (exp_dir / "_config").mkdir(exist_ok=True)
    (exp_dir / "_config" / "run_args.json").write_text(json.dumps(
        {"transplant_epoch": TEPOCH, "epochs": EPOCHS, "schedule_epochs": SCHEDULE_EPOCHS, "milestone_epochs": MILESTONE_EPOCHS, "frozen": FROZEN, "layers": LAYERS, "arms": arms,
         "seeds": seeds}, indent=2))
    R.ensure_dataset(EXP)
    classes = R.canonical_classes(EXP)
    assert len(classes) == 100
    lanes = []
    for s in seeds:
        for arm in arms:
            lid = f"{arm}_seed{s}"
            ld = exp_dir / "lanes" / lid
            ld.mkdir(parents=True, exist_ok=True)
            lanes.append(ld)
            if (ld / ".devtrain_done").exists():
                continue
            try:
                t0 = time.time()
                init_path, ms1 = make_init(s, arm, ld)
                spec_path = None
                if arm in ("C", "Crand"):
                    spec = build_spec(s, arm, ld, ms1)
                    spec_path = ld / "lesion_spec.json"
                    spec_path.write_text(json.dumps(spec, indent=2))
                cfg = build_cfg(template, s, spec_path, bpe)
                (ld / "_config").mkdir(exist_ok=True)
                (ld / "_config" / "transplant_args.json").write_text(json.dumps(
                    {"arm": arm, "seed": s, "frozen": arm != "A", "transplant_epoch": TEPOCH if arm in ("B", "C", "Crand") else None,
                     "projection": arm if arm in ("C", "Crand") else None, "milestone_epochs": MILESTONE_EPOCHS,
                     "source_lane": str(sham_dir(s))}, indent=2))
                _FREEZE["on"] = arm != "A"
                print(f"[{lid}] train epochs 1-{EPOCHS}", flush=True)
                R.log_status(exp_dir, "DEVTRAIN_START", lane=lid)
                R.run_clean_devtrain(ld, cfg, init_path, classes, EXP)
                (ld / ".devtrain_done").touch()
                R.log_status(exp_dir, "DEVTRAIN_DONE", lane=lid, mins=round((time.time() - t0) / 60, 1))
                print(f"[{lid}] trained in {(time.time() - t0) / 60:.1f} min", flush=True)
            except Exception as e:
                import traceback
                (ld / "FAILED.txt").write_text(f"{e}\n\n{traceback.format_exc()}")
                R.log_status(exp_dir, "LANE_FAILED", lane=lid, error=str(e)[:200])
                print(f"[{lid}] FAILED: {e}", flush=True)
                traceback.print_exc()
            finally:
                _FREEZE["on"] = False
    if a.skip_sae:
        return
    for ld in lanes:
        if (ld / ".sae_done").exists() or not (ld / ".devtrain_done").exists():
            continue
        t1 = time.time()
        R.log_status(exp_dir, "SAE_START", lane=ld.name)
        R.run_sae(ld, classes, template.get("sae_policy", {}))
        if (ld / "sae_analysis" / "sae_results.json").exists():
            (ld / ".sae_done").touch()
        R.log_status(exp_dir, "SAE_DONE", lane=ld.name, mins=round((time.time() - t1) / 60, 1))
        print(f"[{ld.name}] RDT done in {(time.time() - t1) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
