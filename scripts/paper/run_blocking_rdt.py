#!/usr/bin/env python3
"""RDT (per-checkpoint SAE pipeline) on the acquisition-blocking runs, epochs 0-15, with the settings of the clean
per-epoch runs (run_sham_perepoch_sae.py; sham_perepoch).

Lanes (ResNet-18, CIFAR-100, branch epoch 0, blocking window epochs 1-10, seeds 7 / 42 / 137):
  gradsc      superclass-blocked   gradblock/lanes/lesion_gradsc_e00_seed{S}_rp65_Llayer1-layer2-layer3-layer4-avgpool
  gradenergy  energy-matched       gradenergy/lanes/lesion_gradenergy_e00_seed{S}_rp65_L...
  gradrg      random-grouping      gradrg/lanes/lesion_gradrg_e00_seed{S}_rp65_L...
The forward pass of these runs is unchanged, so the stored activations (milestone_k/individual_multilayer_activations.pt,
same layers / classes / samples / shapes as the sham lanes; checked on every copied milestone) are the network's own.

Settings reused exactly from run_sham_perepoch_sae.py: the unchanged pipeline (packages/ml/src/saeanalysis.py) called
through the same entry point (lesion run_noise_curriculum_experiment.run_sae: dataset cifar100, class order
canonical_classes('curriculum')), the template sae_policy (noise_curriculum_arch/_config/template.json:
expansion 4, top-k 32, 5,000 steps, shared_init_seed 137, null_permutations 1000), default matching thresholds
(stable r >= 0.5, birth/death r < 0.2), PYTHONHASHSEED=0 (the shared SAE init seed is shared_init_seed +
hash(layer_name)) and DEVTRAIN_MPS_MAX_BATCH=128. The script re-executes itself if PYTHONHASHSEED is not 0.

As for the sham lanes, milestones 0..15 (epochs 0..15; mapped by devtrain_snapshots.json: the non-final snapshot with
weight_updates = 391 x epoch) are copied (activations and predictions only, no weights) into a new experiment directory
with a filtered devtrain_snapshots.json, and the pipeline runs there. This keeps the duplicate 'final' milestone and the
random-grouping runs' epochs 20-30 out of the analysis (the pipeline takes every milestone_* directory it finds) and
leaves the training lanes untouched:
  /data/rdt/extended/experiments/blocking_rdt/lanes/{arm}_seed{S}/sae_analysis/
Resumable per lane: a lane with .sae_done is skipped; a lane whose source run has not yet reached epoch 15 is reported
NOT_READY and skipped (--wait polls until all requested lanes are done). An interrupted lane restarts from scratch (the
pipeline is one pass over all checkpoints). Per-lane timing: {lane}/RDT_TIMING.json (every progress message, timed);
run log: blocking_rdt/RUN_STATUS.jsonl.

Test lane (--pilot): superclass-blocked seed 7, epochs 0-1 only, into experiments/blocking_rdt_pilot/lanes/gradsc_seed7.
Because the SAE minibatch RNG runs through the checkpoints in order, its checkpoints 0 and 1 are trained exactly
as in a full run of that lane. Epoch-0 activations equal the sham's (same branch checkpoint) and the shared SAE init is
the same, but SAE training on MPS is not bitwise reproducible over 5,000 steps (a re-run of the same SAE differs from
both earlier runs), so the epoch-0 codes differ slightly from the sham's; blocking_rdt_analysis.py reports this
epoch-0 difference as the SAE run-to-run noise floor.

Never writes to the data volume that holds the template (read only, as in run_sham_perepoch_sae.py).
Usage:
  python -u scripts/paper/run_blocking_rdt.py --dry-run          # plan + readiness, no copy, no GPU
  python -u scripts/paper/run_blocking_rdt.py                    # all ready lanes
  python -u scripts/paper/run_blocking_rdt.py --arms gradsc,gradenergy --seeds 7,42 --wait
  python -u scripts/paper/run_blocking_rdt.py --pilot
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

if os.environ.get("PYTHONHASHSEED") != "0" or os.environ.get("DEVTRAIN_MPS_MAX_BATCH") != "128":
    env = dict(os.environ, PYTHONHASHSEED="0", DEVTRAIN_MPS_MAX_BATCH="128")
    os.execve(sys.executable, [sys.executable, "-u", *sys.argv], env)

REPO = Path(__file__).resolve().parents[2]
WT = Path(RDT_DATA + "/checkouts/lesion")
E = Path(RDT_DATA + "/extended/experiments")
TPL = Path(RDT_DATA + "/core/experiments/noise_curriculum_arch/_config/template.json")
SHAM = E / "sham_perepoch" / "lanes"
SUFFIX = "rp65_Llayer1-layer2-layer3-layer4-avgpool"
SRC = {"gradsc": E / "gradblock" / "lanes" / ("lesion_gradsc_e00_seed{s}_" + SUFFIX),
       "gradenergy": E / "gradenergy" / "lanes" / ("lesion_gradenergy_e00_seed{s}_" + SUFFIX),
       # random directions, same number (heavy-block control for class-level feature counts)
       "gradrand": E / "gradblock" / "lanes" / ("lesion_gradrand_e00_seed{s}_" + SUFFIX),
       "gradrg": E / "gradrg" / "lanes" / ("lesion_gradrg_e00_seed{s}_" + SUFFIX),
       # stage-matched dose control: blocks at layer2 to the pooled layer only, epochs 1-10, trained to epoch 15
       "gradsc_l2": E / "gradl2" / "lanes" / "lesion_gradsc_e00_seed{s}_rp65_Llayer2-layer3-layer4-avgpool",
       "gradenergy_l2": E / "gradl2" / "lanes" / "lesion_gradenergy_e00_seed{s}_rp65_Llayer2-layer3-layer4-avgpool",
       # jointly matched block (rank and learning-signal share) and the pair confined to layer3 to the pooled layer
       "gradjoint": E / "gradjoint" / "lanes" / ("lesion_gradjoint_e00_seed{s}_" + SUFFIX),
       "gradsc_l3": E / "gradl3" / "lanes" / "lesion_gradsc_e00_seed{s}_rp65_Llayer3-layer4-avgpool",
       "gradjoint_l3": E / "gradl3" / "lanes" / "lesion_gradjoint_e00_seed{s}_rp65_Llayer3-layer4-avgpool"}
ARM_LABEL = {"gradsc": "superclass-blocked", "gradenergy": "energy-matched", "gradrg": "random-grouping",
             "gradrand": "random directions",
             "gradsc_l2": "superclass-blocked, layer2 to pooled", "gradenergy_l2": "energy-matched, layer2 to pooled",
             "gradjoint": "jointly matched", "gradsc_l3": "superclass-blocked, layer3 to pooled",
             "gradjoint_l3": "jointly matched, layer3 to pooled"}
SEEDS = [7, 42, 137]
MAX_EPOCH = 15
UPD = 391
EXPECTED = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512, "avgpool": 512}  # sham lanes' format
N_CLASSES, N_PER_CLASS = 100, 100
FILES = ("individual_multilayer_activations.pt", "sample_predictions.pt")
assert not str(E).startswith(RDT_DATA + "/core")


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(root, event, **kw):
    root.mkdir(parents=True, exist_ok=True)
    rec = {"ts": now(), "event": event, **kw}
    with open(root / "RUN_STATUS.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def src_dir(arm, s):
    return Path(str(SRC[arm]).format(s=s))


def epoch_map(src, max_epoch):
    """epoch -> source milestone index, from devtrain_snapshots.json (non-final snapshot, weight_updates = 391 x epoch).
    Returns (map, missing epochs, kept snapshot records)."""
    f = src / "dev_snapshots" / "metrics" / "devtrain_snapshots.json"
    if not f.exists():
        return {}, list(range(max_epoch + 1)), []
    snaps = json.load(open(f))
    snaps = snaps.get("snapshots", snaps) if isinstance(snaps, dict) else snaps
    m, keep = {}, []
    for x in snaps:
        if x.get("snapshot_type") == "final":
            continue
        wu = int(x["weight_updates"])
        if wu % UPD == 0 and wu // UPD <= max_epoch:
            e = wu // UPD
            md = src / "dev_snapshots" / f"milestone_{x['milestone_index']}"
            if all((md / fn).exists() for fn in FILES):
                m[e] = int(x["milestone_index"])
                keep.append(x)
    missing = [e for e in range(max_epoch + 1) if e not in m]
    return m, missing, keep


def check_format(act, pred, where):
    import torch  # noqa: F401
    assert set(act) == set(EXPECTED), (where, sorted(act))
    for L, d in EXPECTED.items():
        assert sorted(act[L]) == list(range(N_CLASSES)), (where, L)
        for c in range(N_CLASSES):
            t = act[L][c]
            assert tuple(t.shape) == (N_PER_CLASS, d), (where, L, c, tuple(t.shape))
            assert bool(t.isfinite().all()), (where, L, c, "non-finite")
    assert sorted(pred) == list(range(N_CLASSES)), (where, "predictions")


def make_lane(arm, s, dst_root, max_epoch):
    """Copy epochs 0..max_epoch into dst_root/{arm}_seed{s}; check format; filtered devtrain_snapshots.json."""
    import torch
    src, dst = src_dir(arm, s), dst_root / f"{arm}_seed{s}"
    m, missing, keep = epoch_map(src, max_epoch)
    if missing:
        return None, f"NOT_READY (epochs missing in source: {missing[:5]}{'...' if len(missing) > 5 else ''})"
    out_snaps = []
    for e in range(max_epoch + 1):
        sd, dd = src / "dev_snapshots" / f"milestone_{m[e]}", dst / "dev_snapshots" / f"milestone_{e}"
        dd.mkdir(parents=True, exist_ok=True)
        for fn in FILES:
            if not (dd / fn).exists() or (dd / fn).stat().st_size != (sd / fn).stat().st_size:
                shutil.copy2(sd / fn, dd / fn.replace(".pt", ".pt.part"))
                os.replace(dd / fn.replace(".pt", ".pt.part"), dd / fn)
        act = torch.load(dd / FILES[0], map_location="cpu", weights_only=False)
        pred = torch.load(dd / FILES[1], map_location="cpu", weights_only=False)
        check_format(act, pred, f"{arm}_seed{s}/epoch{e}")
        if e == 0:  # branch checkpoint: forward pass identical to the sham's epoch 0
            ref = torch.load(SHAM / f"sham_e00_seed{s}" / "dev_snapshots" / "milestone_0" / FILES[0],
                             map_location="cpu", weights_only=False)
            d0 = max(float((act[L][c] - ref[L][c]).abs().max()) for L in EXPECTED for c in range(N_CLASSES))
        x = next(k for k in keep if int(k["milestone_index"]) == m[e])
        out_snaps.append({**x, "milestone_index": e, "source_milestone_index": m[e],
                          "snapshot_dir": str(dd)})
    (dst / "dev_snapshots" / "metrics").mkdir(parents=True, exist_ok=True)
    json.dump(out_snaps, open(dst / "dev_snapshots" / "metrics" / "devtrain_snapshots.json", "w"), indent=1)
    (dst / "SOURCE.txt").write_text(
        f"copy of {src} epochs 0..{max_epoch} (source milestones {[m[e] for e in range(max_epoch + 1)]} -> "
        f"milestone_0..{max_epoch}); activations and predictions only; format checked against the sham lanes; "
        f"epoch-0 activations max |diff| vs sham_e00_seed{s} = {d0:.3g}\n")
    return dst, f"READY (epoch-0 max|diff| vs sham {d0:.3g})"


def load_runner():
    spec = importlib.util.spec_from_file_location("ncr", WT / "scripts" / "run_noise_curriculum_experiment.py")
    ncr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ncr)
    a, b = WT / "packages/ml/src/saeanalysis.py", REPO / "packages/ml/src/saeanalysis.py"
    assert sha(a) == sha(b), "saeanalysis.py differs between lesion and the repo"
    return ncr


def run_lane(ncr, lane, classes, policy, root):
    events, t0 = [], time.time()

    def cb(p, msg):
        events.append({"t": round(time.time() - t0, 2), "pct": round(float(p), 1), "msg": msg})
        print(f"    sae [{p:5.1f}%] {msg}", flush=True)
    from saeanalysis import analyze_sae_features
    n_ck = len([d for d in (lane / "dev_snapshots").iterdir() if d.name.startswith("milestone_")])
    log(root, "SAE_START", lane=lane.name, n_checkpoints=n_ck)
    analyze_sae_features(lane_dir=str(lane), selected_classes=classes,          # == ncr.run_sae, timed callback
                         experiment_config={"dataset_id": "cifar100", "sae_policy": policy}, progress_callback=cb,
                         expansion_factor=policy.get("expansion_factor", 4), k_sparse=policy.get("k_sparse", 32),
                         n_steps=policy.get("n_steps", 5000))
    secs = time.time() - t0
    out = lane / "sae_analysis"
    du = {d.name: sum(f.stat().st_size for f in d.rglob("*") if f.is_file() and not f.name.startswith("._"))
          for d in out.iterdir() if d.is_dir()}
    du["sae_results.json"] = (out / "sae_results.json").stat().st_size
    json.dump({"lane": lane.name, "n_checkpoints": n_ck, "seconds": secs, "bytes": du,
               "python_hash_seed": os.environ.get("PYTHONHASHSEED"), "sae_policy": policy,
               "saeanalysis_sha256_16": sha(REPO / "packages/ml/src/saeanalysis.py"), "progress": events},
              open(lane / "RDT_TIMING.json", "w"), indent=1)
    (lane / ".sae_done").touch()
    log(root, "SAE_DONE", lane=lane.name, mins=round(secs / 60, 2), gb=round(sum(du.values()) / 1e9, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="gradsc,gradenergy,gradrg")
    ap.add_argument("--seeds", default="7,42,137")
    ap.add_argument("--pilot", action="store_true", help="gradsc seed 7, epochs 0-1, into blocking_rdt_pilot")
    ap.add_argument("--pilot-arm", default="gradsc", help="arm of the pilot lane (seed 7, epochs 0-1)")
    ap.add_argument("--dry-run", action="store_true", help="readiness only; no copy, no SAE")
    ap.add_argument("--wait", action="store_true", help="poll every 10 min until all requested lanes are done")
    a = ap.parse_args()
    if a.pilot:
        plan, max_epoch, root = [(a.pilot_arm, 7)], 1, E / "blocking_rdt_pilot"
    else:
        plan = [(arm, int(s)) for arm in a.arms.split(",") for s in a.seeds.split(",")]
        max_epoch, root = MAX_EPOCH, E / "blocking_rdt"
    lanes_root = root / "lanes"
    policy = json.loads(TPL.read_text()).get("sae_policy", {})
    print(f"{len(plan)} lanes -> {lanes_root}; epochs 0..{max_epoch}; sae_policy {policy}; "
          f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}", flush=True)
    if a.dry_run:
        for arm, s in plan:
            dst = lanes_root / f"{arm}_seed{s}"
            m, missing, _ = epoch_map(src_dir(arm, s), max_epoch)
            st = "DONE" if (dst / ".sae_done").exists() else ("NOT_READY" if missing else "READY")
            print(f"  {arm:<11}{s:>4}  {ARM_LABEL[arm]:<19} {st:<9} source milestones {[m.get(e) for e in range(max_epoch + 1)]}")
        return
    ncr = load_runner()
    classes = ncr.canonical_classes("curriculum")
    assert len(classes) == N_CLASSES
    log(root, "RUN_START", plan=[f"{x}_seed{y}" for x, y in plan], max_epoch=max_epoch, sae_policy=policy)
    while True:
        pending = []
        for arm, s in plan:
            dst = lanes_root / f"{arm}_seed{s}"
            if (dst / ".sae_done").exists():
                continue
            # lane lock: two runner processes may share the plan; a lane held by a live process is skipped
            lock = dst / ".sae_running"
            if lock.exists():
                try:
                    os.kill(int(lock.read_text().strip()), 0)
                    log(root, "LOCKED", lane=dst.name, pid=lock.read_text().strip())
                    continue
                except (ValueError, ProcessLookupError):
                    lock.unlink()           # stale lock of a dead process
            dst.mkdir(parents=True, exist_ok=True)
            lock.write_text(str(os.getpid()))
            try:
                lane, status = make_lane(arm, s, lanes_root, max_epoch)
                if lane is None:
                    log(root, "NOT_READY", lane=dst.name, status=status)
                    pending.append(dst.name)
                    continue
                log(root, "LANE_READY", lane=dst.name, status=status)
                try:
                    run_lane(ncr, lane, classes, policy, root)
                except Exception as ex:  # keep going with the other lanes; the lane restarts on the next call
                    log(root, "SAE_FAILED", lane=dst.name, error=repr(ex)[-500:])
                    pending.append(dst.name)
            finally:
                lock.unlink(missing_ok=True)
        if not pending or not a.wait:
            break
        log(root, "WAITING", pending=pending)
        time.sleep(600)
    log(root, "RUN_END", pending=pending)


if __name__ == "__main__":
    main()
