#!/usr/bin/env python3
"""Per-epoch checkpoints for CCT-7 (CIFAR-100, seed 42): training plus the full SAE pipeline at every epoch.

Feature events between sparse checkpoints may be missed, especially late in training where the accuracy-spaced
intervals are long. The core CCT-7 seed-42 lane (core experiment abf16d20, lane 4bebc329) had
10 accuracy-spaced checkpoints at 0, 190, 361, 619, 1010, 1515, 2335, 3842, 12140, 19550 weight updates
(epochs 0, 0.49, 0.92, 1.58, 2.58, 3.87, 5.97, 9.83, 31.05, 50). This run repeats that training with a checkpoint at
the end of every epoch (milestone e = epoch e, 391 weight updates per epoch, epochs 0..50 = 51 checkpoints) and runs
the unchanged SAE pipeline (packages/ml/src/saeanalysis.analyze_sae_features) on all 51 checkpoints.

Configuration (identical to the core lane unless listed under "Changes")
  NETINIT config of lane 4bebc329 (from the job database, embedded verbatim below as CORE_NETINIT):
    cct_7, AdamW lr 5e-4, weight decay 0.03, betas (0.9, 0.999), eps 1e-8, batch 128, 50 epochs, cosine annealing to
    1e-6 (devtrain), augmentation RandomCrop(32, pad 4) + horizontal flip, CIFAR-100 mean/std normalisation,
    accuracy_threshold 70 with terminal_capture 'both' (never reached: the core run ended at 52.96 %),
    validation / memory / logging policies as in the core lane, samples_per_class 100 (all 10,000 test images),
    SAE policy expansion 4, top-k 32, 5,000 steps, shared_init_seed 42, 1,000 null permutations.
  Class order: the experiment's selected_classes (superclass-grouped, "cifar100_" prefix), so label i and the
    superclass map are those of the core run (checked against its sae_results.json metadata).
  Initial weights: the core lane's weights/initial.pt (== its milestone_0/checkpoint.pt), copied to _inits/.
    (Re-initialising cct_7 with kaiming_normal seed 42 in this code does NOT reproduce these weights, so the
    stored file is used.)
  Dataset: the core experiment's own CIFAR-100 copy (read-only symlink; md5 identical to the other copies).
  Activation capture: individual_multilayer_activations.pt (auto-selected layers norm, transformer.layers.0/1/3/5/6;
    100 test images per class) and sample_predictions.pt at every checkpoint, as in the core lane.
  Changes
    1. snapshot_policy: milestone_type 'weight_updates', milestones 391 * e for e = 1..49 (+ t0 and the final
       snapshot at epoch 50) instead of 10 accuracy milestones.
    2. representation_methods = [] (core lane: mean_activations, individual_activations, multilayer_activations;
       none of them is read by the SAE pipeline; devtrain extracts the SAE inputs unconditionally). Saves disk/time.
    3. Hardware: the core run trained on a CUDA GPU with 2 DataLoader workers; this run trains on Apple
       MPS with 0 workers (devtrain's MPS rule) at the configured batch 128 (DEVTRAIN_MPS_MAX_BATCH=128; without it
       devtrain would cap MPS batches at 16). Bit-exact reproduction of the core trajectory is therefore not
       possible (different kernels, different augmentation/shuffle RNG streams); everything else is identical.
    4. PYTHONHASHSEED=0 (SAE shared-init seed = shared_init_seed + hash(layer_name)); the script re-executes itself
       with it. As in run_sham_perepoch_sae.py / run_sae_lanes.py.
  Code: the ML package of the lesion variant installed at WT (packages/ml/src, see variants/lesion; its
    data/experiments resolves to DATA_EXP). It is byte-identical to this repository's packages/ml/src except a 4-line energy-matched-lesion block
    in devtrain.py that is inactive without a lesion policy. saeanalysis.py is the pipeline used for every other SAE
    analysis in scripts/paper (Ab/Tg exclusivity, CSI null rule, no separate 'terminal' checkpoint).

Resumability
  * Stage markers in the lane dir: .devtrain_done, .sae_done. Re-running the same command skips finished stages.
  * Training: if dev_snapshots/metrics/devtrain_snapshots.json exists and its last snapshot is not 'final', training
    resumes from that milestone's checkpoint.pt + training_state.pt (AdamW and cosine-schedule state) at the next
    epoch (devtrain resume_state). Milestone directories newer than the last recorded snapshot (interrupted capture)
    are removed first. devtrain re-seeds the global RNG on (re)start, so a resumed run is statistically, not
    bitwise, equivalent to an uninterrupted one; elapsed_time_seconds restarts at the resume point.
  * SAE stage: the pipeline is called unchanged, with two wrappers that do not change its arithmetic:
      - Phase 2 (SAE training) is called once per checkpoint (same RNG stream as one call over all checkpoints:
        create_sae_init restores the RNG state, so the shared init is identical) and each finished checkpoint is
        marked saes/<label>/.complete; on a re-run, completed checkpoints are loaded from their saved state dicts
        (exactly as saeanalysis.reclassify_from_saved loads them) instead of being retrained.
      - Phase 3 (encoding) is called once per checkpoint and the returned code matrices are replaced by
        copy-on-write memory maps of the .npy files Phase 3 itself writes (51 checkpoints x 6 layers x 10,000 x 1,024
        float32 = 12.5 GB would otherwise be held in RAM; numerically identical).
    --no-wrap runs the pipeline without both wrappers.

Optional --lean-monitoring (default OFF = core configuration): the core lane's validation_policy runs a full 10,000-image
  validation pass every 5 % of an epoch (20 per epoch) and memory_policy extracts activation statistics every 50
  batches. Both are monitoring only (eval mode, no gradient; in weight-update milestone mode they trigger nothing,
  since the 70 % accuracy threshold is never reached). The flag disables them (validation every 10,000 batches =
  never within an epoch; the end-of-epoch validation and all checkpoint captures are unchanged). It does not change
  weights, optimiser, schedule, data or augmentation; it only shifts the realisation of the CPU RNG stream (each
  DataLoader iterator draws a base seed), which is not reproducible across CUDA/MPS anyway. These passes take about
  half of the training wall-clock.
  * Short test run: --stop-after-epoch N stops training (devtrain cancellation flag) after the epoch-N checkpoint and runs the
    SAE stage on the checkpoints that exist (N + 1).

Outputs: /data/rdt/extended/experiments/<experiment-id>/
  _config/  (core_netinit.json, netinit_perepoch.json, config_changes.json, selected_classes.json, run_args.json)
  _inits/cct_7_seed42_core_initial.pt
  datasets/cifar100/cifar-100-python -> core experiment's dataset (read-only)
  ORCHESTRATOR_STATUS.jsonl
  lanes/cct7_seed42/{dev_snapshots/, sae_analysis/, run.log (timestamped stdout), sae_progress.jsonl}
Never writes to the source data volume (EXCLUDED; read-only source of config, weights and dataset).

Usage
  full run : python -u scripts/paper/run_perepoch_cct.py --experiment-id perepoch_cct
  test run : python -u scripts/paper/run_perepoch_cct.py --experiment-id perepoch_cct_pilot --stop-after-epoch 1
  plan only: add --plan-only
Analysis: scripts/paper/perepoch_cct_analysis.py
"""
import os
import sys
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

# SAE shared-init seeds use hash(layer_name): fix the hash seed before anything else (re-exec once).
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, list(getattr(sys, "orig_argv", [sys.executable, *sys.argv])))
# Train at the configured batch (128) on MPS; devtrain otherwise caps MPS batches at 16.
os.environ["DEVTRAIN_MPS_MAX_BATCH"] = "128"

import argparse  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

WT = Path(RDT_DATA + "/checkouts/lesion")
ML_SRC = WT / "packages" / "ml" / "src"
DATA_EXP = Path(RDT_DATA + "/extended/experiments")
EXCLUDED = Path(RDT_DATA + "/core")
CORE_EXP = EXCLUDED / "experiments" / "abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb"
CORE_LANE = CORE_EXP / "lanes" / "4bebc329-c26a-4eb4-b18c-7586e513f49e"
LANE = "cct7_seed42"

# NETINIT of lane 4bebc329 (from the job database), verbatim.
CORE_NETINIT = json.loads(
    '{"network_type": "cct_7", "initialization_method": "kaiming_normal", "training_policy": {"epochs": 50, '
    '"accuracy_threshold": 70}, "optimizer": {"type": "adamw", "learning_rate": 0.0005, "weight_decay": 0.03, '
    '"beta1": 0.9, "beta2": 0.999, "epsilon": 1e-08}, "batch_size": 128, "snapshot_policy": {"milestone_count": 10, '
    '"distribution_scheme": "uniform", "samples_per_class": 100, "terminal_capture": "both"}, '
    '"representation_methods": ["mean_activations", "individual_activations", "multilayer_activations"], '
    '"transform_config": {"normalize_mean": [0.5071, 0.4867, 0.4408], "normalize_std": [0.2675, 0.2565, 0.2761], '
    '"augmentation_enabled": true, "random_crop_padding": 4, "horizontal_flip": true}, "validation_policy": '
    '{"frequency_mode": "percentage", "frequency_value": 5, "min_batch_interval": 5, "validation_set_fraction": 1}, '
    '"memory_policy": {"disk_based_linear_probes": true, "disk_based_multilayer": true, "linear_probe_batch_size": 32, '
    '"activation_extraction_enabled": true, "activation_extraction_interval": 50}, "logging_policy": '
    '{"batch_log_frequency": 10, "batch_log_include_activations": false, "metrics_flush_interval": 10, '
    '"metrics_keep_recent": 50}, "sae_policy": {"expansion_factor": 4, "k_sparse": 32, "n_steps": 5000, '
    '"shared_init_seed": 42, "null_permutations": 1000}, "random_seed": 42}')

# experiments.selected_classes of abf16d20 (superclass-grouped order), without the "cifar100_" prefix.
SELECTED_CLASSES = ["cifar100_" + c for c in " ".join([
    "beaver dolphin otter seal whale aquarium_fish flatfish ray shark trout",
    "orchid poppy rose sunflower tulip bottle bowl can cup plate",
    "apple mushroom orange pear sweet_pepper clock keyboard lamp telephone television",
    "bed chair couch table wardrobe bee beetle butterfly caterpillar cockroach",
    "bear leopard lion tiger wolf bridge castle house road skyscraper",
    "cloud forest mountain plain sea camel cattle chimpanzee elephant kangaroo",
    "fox porcupine possum raccoon skunk crab lobster snail spider worm",
    "baby boy girl man woman crocodile dinosaur lizard snake turtle",
    "hamster mouse rabbit shrew squirrel maple_tree oak_tree palm_tree pine_tree willow_tree",
    "bicycle bus motorcycle pickup_truck train lawn_mower rocket streetcar tank tractor"]).split()]

EPOCHS = CORE_NETINIT["training_policy"]["epochs"]
N_TRAIN = 50_000
BATCHES_PER_EPOCH = math.ceil(N_TRAIN / CORE_NETINIT["batch_size"])  # 391
WU_MILESTONES = [e * BATCHES_PER_EPOCH for e in range(1, EPOCHS)]          # epochs 1..49; epoch 50 = final snapshot


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def guard(p: Path):
    """Reject any output path in the read-only source data directory."""
    rp = Path(os.path.realpath(p))
    if str(rp).startswith(str(EXCLUDED)):
        raise RuntimeError(f"read-only source data directory, not writing under {EXCLUDED}: {p} -> {rp}")
    return p


class TsTee:
    """stdout passthrough that also appends '<unix time>\\t<line>' to a log file (phase timings)."""

    def __init__(self, stream, path: Path):
        self.s, self.f, self.buf = stream, open(guard(path), "a"), ""

    def write(self, x):
        self.s.write(x)
        self.buf += x
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self.f.write(f"{time.time():.3f}\t{line}\n")
        self.f.flush()
        return len(x)

    def flush(self):
        self.s.flush()
        self.f.flush()

    def __getattr__(self, k):
        return getattr(self.s, k)


def log_status(exp_dir: Path, event: str, **kw):
    rec = {"ts": now(), "event": event, **kw}
    with open(guard(exp_dir / "ORCHESTRATOR_STATUS.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[{rec['ts']}] {event} " + " ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# configuration
def perepoch_config(lean: bool = False) -> dict:
    cfg = copy.deepcopy(CORE_NETINIT)
    if lean:  # monitoring only (see --lean-monitoring); never touches weights, optimiser, schedule or data
        cfg["validation_policy"].update({"frequency_mode": "absolute", "frequency_value": 10_000})
        cfg["memory_policy"]["activation_extraction_enabled"] = False
    sp = cfg["snapshot_policy"]
    sp["milestone_type"] = "weight_updates"
    sp["milestone_weight_updates"] = list(WU_MILESTONES)
    sp["milestone_count"] = len(WU_MILESTONES)
    cfg["representation_methods"] = []
    return cfg


CONFIG_CHANGES = {
    "snapshot_policy.milestone_type": ["accuracy (default)", "weight_updates"],
    "snapshot_policy.milestone_weight_updates": ["10 accuracy milestones (11.8 .. 64.2 % val acc)",
                                                 f"391*e, e=1..49 (+ t0 and final at epoch {EPOCHS})"],
    "snapshot_policy.milestone_count": [10, len(WU_MILESTONES)],
    "representation_methods": [CORE_NETINIT["representation_methods"],
                               "[] (SAE inputs individual_multilayer_activations + sample_predictions are extracted "
                               "unconditionally by devtrain)"],
    "hardware": ["CUDA GPU, DataLoader num_workers=2", "Apple MPS, num_workers=0 (devtrain MPS rule), batch 128 via "
                 "DEVTRAIN_MPS_MAX_BATCH=128"],
    "PYTHONHASHSEED": ["unknown (CUDA run)", "0"],
    "validation_policy / memory_policy (only with --lean-monitoring)": [
        "20 intra-epoch validation passes, activation statistics every 50 batches",
        "none within an epoch (monitoring only)"],
}


def check_selected_classes():
    """The embedded class order must equal the core lane's SAE metadata (read-only, first 400 kB only)."""
    rp = CORE_LANE / "sae_analysis" / "sae_results.json"
    if not rp.exists():
        print("  (core sae_results.json not reachable; class-order check skipped)", flush=True)
        return None
    with open(rp) as f:
        head = f.read(400_000)
    m = re.search(r'"selected_classes":\s*\[([^\]]*)\]', head)
    sub = [s.strip().strip('"') for s in m.group(1).split(",")]
    assert sub == SELECTED_CLASSES, "embedded class order differs from the core lane"
    return True


def prepare_experiment(exp_dir: Path, args) -> Path:
    guard(exp_dir)
    for d in ("_config", "_inits", "lanes", "datasets/cifar100"):
        (exp_dir / d).mkdir(parents=True, exist_ok=True)
    # dataset: the core experiment's CIFAR-100 copy, read-only through a symlink
    link = exp_dir / "datasets" / "cifar100" / "cifar-100-python"
    src = CORE_EXP / "datasets" / "cifar100" / "cifar-100-python"
    if not link.exists():
        if link.is_symlink():
            link.unlink()
        os.symlink(src, link)
    for f in ("train", "test", "meta"):
        assert (link / f).exists(), f"dataset file missing: {link / f}"
    # initial weights: the core lane's weights/initial.pt
    init = exp_dir / "_inits" / "cct_7_seed42_core_initial.pt"
    src_init = CORE_LANE / "weights" / "initial.pt"
    if not init.exists():
        shutil.copy2(src_init, guard(init))
    if src_init.exists():
        assert md5(init) == md5(src_init), "copied initial weights differ from the core lane's file"
    cfg = perepoch_config(args.lean_monitoring)
    (exp_dir / "_config" / "core_netinit.json").write_text(json.dumps(CORE_NETINIT, indent=2))
    (exp_dir / "_config" / "netinit_perepoch.json").write_text(json.dumps(cfg, indent=2))
    (exp_dir / "_config" / "config_changes.json").write_text(json.dumps(CONFIG_CHANGES, indent=2))
    (exp_dir / "_config" / "selected_classes.json").write_text(json.dumps(SELECTED_CLASSES, indent=1))
    (exp_dir / "_config" / "run_args.json").write_text(json.dumps(
        {**vars(args), "source_lane": str(CORE_LANE), "code": str(ML_SRC), "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
         "DEVTRAIN_MPS_MAX_BATCH": os.environ.get("DEVTRAIN_MPS_MAX_BATCH")}, indent=2))
    return init


def namespace(cfg: dict):
    from run_devtrain_direct import add_missing_defaults, dict_to_namespace
    ns = dict_to_namespace(json.loads(json.dumps(cfg)))
    add_missing_defaults(ns, cfg)
    return ns


# training
def resume_state_from_disk(lane_dir: Path):
    """None = fresh start; 'done' = final snapshot present; dict = devtrain resume_state."""
    snaps_dir = lane_dir / "dev_snapshots"
    sf = snaps_dir / "metrics" / "devtrain_snapshots.json"
    snaps = json.load(open(sf)) if sf.exists() else []
    if not snaps:
        if snaps_dir.exists():
            shutil.rmtree(guard(snaps_dir))  # interrupted before the t0 snapshot was recorded
        return None
    last = snaps[-1]
    if last["snapshot_type"] == "final":
        return "done"
    idx, wu, ep = last["milestone_index"], last["weight_updates"], last["epoch"]
    assert idx == ep and wu == ep * BATCHES_PER_EPOCH, f"unexpected last snapshot {last}"
    for d in snaps_dir.glob("milestone_*"):
        if int(d.name.split("_")[1]) > idx:
            shutil.rmtree(guard(d))  # capture interrupted after the last recorded snapshot
    ck = snaps_dir / f"milestone_{idx}"
    for f in ("checkpoint.pt", "training_state.pt"):
        assert (ck / f).exists(), f"cannot resume: {ck / f} missing"
    return {
        "checkpoint_path": str(ck / "checkpoint.pt"),
        "training_state_path": str(ck / "training_state.pt"),
        "start_epoch": ep + 1,  # the milestone is captured at the last batch of epoch `ep`
        "existing_snapshots": snaps,
        "next_milestone_index": idx + 1,
        "milestones_reached": [],
        "milestones_wu_reached": [w <= wu for w in WU_MILESTONES],
        "total_weight_updates": wu,
        "capture_t0": False,
    }


class StopAfterEpoch(threading.Thread):
    """For --stop-after-epoch: raise devtrain's cancellation flag once the checkpoint of epoch N is recorded."""

    def __init__(self, lane_dir: Path, epoch: int):
        super().__init__(daemon=True)
        self.metrics = lane_dir / "dev_snapshots" / "metrics"
        self.epoch, self.fired, self.halt = epoch, False, threading.Event()

    def run(self):
        sf = self.metrics / "devtrain_snapshots.json"
        while not self.halt.wait(1.0):
            try:
                snaps = json.load(open(sf))
            except Exception:
                continue
            if any(s["milestone_index"] >= self.epoch and s["snapshot_type"] != "t0_baseline" for s in snaps):
                (self.metrics / "devtrain_cancel.flag").write_text("pilot stop\n")
                self.fired = True
                return


def run_devtrain(exp_dir: Path, lane_dir: Path, init: Path, args):
    from devtrain import train_with_snapshots
    ns = namespace(perepoch_config(args.lean_monitoring))
    rs = resume_state_from_disk(lane_dir)
    if rs == "done":
        return "done"
    snaps, metrics = lane_dir / "dev_snapshots", lane_dir / "dev_snapshots" / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    if rs is not None:
        log_status(exp_dir, "DEVTRAIN_RESUME", lane=LANE, start_epoch=rs["start_epoch"],
                   next_milestone=rs["next_milestone_index"])
    stopper = None
    if args.stop_after_epoch is not None:
        stopper = StopAfterEpoch(lane_dir, args.stop_after_epoch)
        stopper.start()
    try:
        train_with_snapshots(
            initial_weights_path=rs["checkpoint_path"] if rs else str(init), netinit_config=ns,
            dataset_id="cifar100", selected_classes=SELECTED_CLASSES, snapshots_base_dir=str(snaps),
            metrics_dir=str(metrics), experiment_id=args.experiment_id, stove_id=None, resume_state=rs)
    except Exception as e:
        if stopper is not None and stopper.fired and "cancelled" in str(e).lower():
            (metrics / "devtrain_cancel.flag").unlink(missing_ok=True)
            return "stopped"
        raise
    finally:
        if stopper is not None:
            stopper.halt.set()
    return "done"


def verify_snapshots(lane_dir: Path, full: bool):
    snaps = json.load(open(lane_dir / "dev_snapshots" / "metrics" / "devtrain_snapshots.json"))
    idx = [s["milestone_index"] for s in snaps]
    assert idx == list(range(len(snaps))), idx
    for s in snaps:
        assert s["weight_updates"] == BATCHES_PER_EPOCH * s["milestone_index"], s
        assert s["epoch"] == s["milestone_index"] or s["snapshot_type"] == "t0_baseline", s
        d = lane_dir / "dev_snapshots" / f"milestone_{s['milestone_index']}"
        for f in ("checkpoint.pt", "individual_multilayer_activations.pt", "sample_predictions.pt"):
            assert (d / f).exists(), d / f
    if full:
        assert len(snaps) == EPOCHS + 1 and snaps[-1]["snapshot_type"] == "final", (len(snaps), snaps[-1])
    return len(snaps)


# SAE stage
def install_wrappers(sa):
    """Per-checkpoint resumable Phase 2 and memory-mapped Phase 3 (arithmetic unchanged; see module docstring)."""
    import numpy as np
    import torch
    orig2, orig3 = sa._phase2_train_per_checkpoint_saes, sa._phase3_encode_per_checkpoint

    def phase2(all_data, output_dir, layers, labels, expansion_factor, k_sparse, n_steps, cb, shared_init_seed=42):
        out = {}
        for lab in labels:
            d = Path(output_dir) / "saes" / lab
            if (d / ".complete").exists():
                out[lab] = {}
                for L in layers:
                    p = d / f"{sa._safe_layer_filename(L)}.pt"
                    if not p.exists():
                        continue
                    st = torch.load(p, map_location="cpu", weights_only=True)
                    sae = sa.SparseAutoencoder(st["encoder.weight"].shape[1], expansion_factor, k_sparse)
                    sae.load_state_dict(st)
                    sae.eval()
                    out[lab][L] = sae
                sa._progress(cb, 10, f"Loaded cached SAEs {lab} ({len(out[lab])} layers)")
                continue
            out.update(orig2(all_data, output_dir, layers, [lab], expansion_factor, k_sparse, n_steps, cb,
                             shared_init_seed=shared_init_seed))
            (d / ".complete").write_text(now() + "\n")
        return out

    def phase3(all_saes, all_data, labels, layers, output_dir, k_sparse, cb):
        acts, rq = {}, {}
        for lab in labels:
            a, r = orig3(all_saes, all_data, [lab], layers, output_dir, k_sparse, cb)
            acts[lab] = {L: np.load(Path(output_dir) / "activations" / lab / f"{sa._safe_layer_filename(L)}.npy",
                                    mmap_mode="c") for L in a[lab]}
            for L in a[lab]:
                assert acts[lab][L].shape == a[lab][L].shape
            rq.update(r)
            del a
        return acts, rq

    sa._phase2_train_per_checkpoint_saes, sa._phase3_encode_per_checkpoint = phase2, phase3


def run_sae(lane_dir: Path, wrap: bool):
    import saeanalysis as sa
    if wrap:
        install_wrappers(sa)
    plog = open(guard(lane_dir / "sae_progress.jsonl"), "a")

    def cb(p, m):
        plog.write(json.dumps({"t": round(time.time(), 3), "pct": round(float(p), 2), "msg": m}) + "\n")
        plog.flush()
        print(f"    sae [{p:5.1f}%] {m}", flush=True)

    pol = CORE_NETINIT["sae_policy"]
    (lane_dir / "sae_analysis").mkdir(exist_ok=True)
    return sa.analyze_sae_features(
        lane_dir=str(lane_dir), selected_classes=SELECTED_CLASSES,
        experiment_config={"dataset_id": "cifar100", "sae_policy": pol}, progress_callback=cb,
        expansion_factor=pol["expansion_factor"], k_sparse=pol["k_sparse"], n_steps=pol["n_steps"])


# main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--experiment-id", default="perepoch_cct")
    ap.add_argument("--stop-after-epoch", type=int, default=None, help="pilot: stop training after this epoch")
    ap.add_argument("--skip-sae", action="store_true")
    ap.add_argument("--no-wrap", action="store_true", help="SAE pipeline without the resume / memory-map wrappers")
    ap.add_argument("--lean-monitoring", action="store_true",
                    help="skip the 20 intra-epoch full validation passes and the periodic activation statistics "
                         "(monitoring only; ~halves epoch time; see docstring)")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    exp_dir = guard(DATA_EXP / args.experiment_id)
    lane_dir = exp_dir / "lanes" / LANE
    cfg = perepoch_config(args.lean_monitoring)
    print("=" * 100)
    print(f"PER-EPOCH CCT-7  exp={exp_dir}  lane={LANE}")
    print(f"  source lane      {CORE_LANE}")
    print(f"  code             {ML_SRC}")
    print(f"  epochs {EPOCHS}, batch {cfg['batch_size']}, {BATCHES_PER_EPOCH} updates/epoch, "
          f"checkpoints {len(WU_MILESTONES) + 2} (t0, {WU_MILESTONES[0]}..{WU_MILESTONES[-1]}, final "
          f"{EPOCHS * BATCHES_PER_EPOCH})")
    print(f"  optimizer {cfg['optimizer']}  sae_policy {cfg['sae_policy']}")
    print(f"  PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')} DEVTRAIN_MPS_MAX_BATCH={os.environ['DEVTRAIN_MPS_MAX_BATCH']}"
          f"  stop_after_epoch={args.stop_after_epoch}  lean_monitoring={args.lean_monitoring}")
    print("=" * 100, flush=True)
    if args.plan_only:
        return

    assert (WT / "data" / "experiments").resolve() == DATA_EXP.resolve(), "lesion-variant data/experiments does not resolve to DATA_EXP"
    sys.path.insert(0, str(ML_SRC))
    lane_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = TsTee(sys.stdout, lane_dir / "run.log")
    import torch
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    log_status(exp_dir, "RUN_START", lane=LANE, device=dev, stop_after_epoch=args.stop_after_epoch, pid=os.getpid())
    check_selected_classes()
    init = prepare_experiment(exp_dir, args)
    try:
        if not (lane_dir / ".devtrain_done").exists():
            t0 = time.time()
            log_status(exp_dir, "DEVTRAIN_START", lane=LANE)
            res = run_devtrain(exp_dir, lane_dir, init, args)
            n = verify_snapshots(lane_dir, full=(res == "done"))
            if res == "done":
                (lane_dir / ".devtrain_done").write_text(now() + "\n")
            log_status(exp_dir, "DEVTRAIN_" + res.upper(), lane=LANE, n_checkpoints=n,
                       mins=round((time.time() - t0) / 60, 1))
        else:
            print("devtrain already done - skip", flush=True)
        if args.skip_sae or (lane_dir / ".sae_done").exists():
            print("SAE stage skipped" + (" (done)" if (lane_dir / ".sae_done").exists() else ""), flush=True)
            return
        t1 = time.time()
        log_status(exp_dir, "SAE_START", lane=LANE, wrapped=not args.no_wrap)
        out = run_sae(lane_dir, wrap=not args.no_wrap)
        if isinstance(out, dict) and out.get("error"):
            raise RuntimeError(out["error"])
        assert (lane_dir / "sae_analysis" / "sae_results.json").exists(), "sae_results.json missing"
        (lane_dir / ".sae_done").write_text(now() + "\n")
        log_status(exp_dir, "SAE_DONE", lane=LANE, mins=round((time.time() - t1) / 60, 1))
    except KeyboardInterrupt:
        log_status(exp_dir, "INTERRUPTED", lane=LANE)
        raise
    except Exception as e:
        (lane_dir / "FAILED.txt").write_text(f"{now()}\n{e}\n\n{traceback.format_exc()}")
        log_status(exp_dir, "FAILED", lane=LANE, error=str(e)[:300])
        raise


if __name__ == "__main__":
    main()
