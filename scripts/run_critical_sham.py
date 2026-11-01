#!/usr/bin/env python3
"""
Sham lanes for the critical-period experiment.

A sham lane is a lesion lane without the lesion: it resumes from the same clean-lane branch checkpoint
(checkpoint.pt + training_state.pt) on the same machine, with the same continuation config as the coarse /
random lanes of run_scaffold_lesion_experiment.py (template, dense branch-relative milestones, probe
schedule), but with lesion_policy.enabled = False, so no projection hooks are installed and no
re-projection runs. It is the baseline for the lasting-cost contrasts (coarse - sham, random - sham)
and removes platform and resume-path differences between the lesion lanes and the clean lanes.

Reuses the orchestrator's helpers so the config is identical except for the disabled lesion.
Writes to its own experiment directory (default critical_sham).

    python -u scripts/run_critical_sham.py --clean-lane <dir> --clean-lane-seed 7 --seeds 7 \
        --branch-epochs 0,10 --window 10 --epochs 30 --template-json <template.json> [--plan-only]
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_scaffold_lesion_experiment as orch  # noqa: E402  (sets sys.path for packages/ml/src)


def sham_config(template, branch, window, epochs, seed, bpe, post_release_dense):
    """The lesion-lane continuation config with the lesion disabled."""
    args = SimpleNamespace(
        epochs=epochs, window=window, post_release_dense=post_release_dense, probe_every=1,
        reproject_every=None, reproject_q_max=19, reproject_evr_floor=0.02,
        reproject_max_rank_frac=0.7, reproject_rank_policy="sliding")
    lane = {"branch": branch, "seed": seed, "layers": ["layer4", "avgpool"], "condition": "sham",
            "target_code": "scaffold"}
    cfg, mile_epochs = orch.build_lesion_lane_config(template, args, lane, spec_path="", bpe=bpe)
    cfg.lesion_policy.enabled = False
    return cfg, mile_epochs


def main():
    ap = argparse.ArgumentParser(description="Sham lanes for critical (no projection)")
    ap.add_argument("--experiment-id", default="critical_sham")
    ap.add_argument("--clean-lane", required=True)
    ap.add_argument("--clean-lane-seed", type=int, required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--branch-epochs", default="0,10")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--post-release-dense", type=int, default=5)
    ap.add_argument("--template-json", required=True)
    ap.add_argument("--skip-sae", action="store_true")
    ap.add_argument("--plan-only", action="store_true",
                    help="build configs and check inputs, then exit (no training, no GPU)")
    args = ap.parse_args()
    os.environ.setdefault("DEVTRAIN_MPS_MAX_BATCH", "128")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    assert seeds == [args.clean_lane_seed], "one seed per call, matching --clean-lane-seed"
    seed = seeds[0]
    branches = [int(b) for b in args.branch_epochs.split(",") if b.strip()]
    clean_dir = Path(args.clean_lane)
    exp_id = args.experiment_id
    exp_dir = orch.DATA_DIR / "experiments" / exp_id
    template = json.loads(Path(args.template_json).read_text())
    bpe, eff_bs = orch.effective_batches_per_epoch(template)
    from lesion import _load_branch_checkpoint_label

    plan = []
    for b in branches:
        orch.assert_grid_alignment(clean_dir, b, bpe)
        label = _load_branch_checkpoint_label(clean_dir, b)
        m = clean_dir / "dev_snapshots" / f"milestone_{label}"
        for f in ("checkpoint.pt", "training_state.pt"):
            if not (m / f).exists():
                raise RuntimeError(f"missing {m / f}")
        cfg, mile_epochs = sham_config(template, b, args.window, args.epochs, seed, bpe,
                                       args.post_release_dense)
        assert cfg.lesion_policy.enabled is False
        plan.append({"lane_id": f"sham_e{b:02d}_seed{seed}", "branch": b, "seed": seed,
                     "clean_dir": clean_dir, "label": label, "cfg": cfg, "mile_epochs": mile_epochs})

    print("=" * 72)
    print(f"CRITICAL SHAM  exp={exp_id} seed={seed} eff_batch={eff_bs} batches/epoch={bpe}")
    print(f"clean lane: {clean_dir}")
    for p in plan:
        print(f"  {p['lane_id']:<22} resume milestone_{p['label']} -> epoch {p['branch'] + 1}; "
              f"milestones at epochs {p['mile_epochs']}; lesion disabled")
    print("=" * 72, flush=True)
    if args.plan_only:
        print("--plan-only: exiting without running.", flush=True)
        return

    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "_config").mkdir(exist_ok=True)
    (exp_dir / "_config" / "template.json").write_text(json.dumps(template, indent=2))
    orch.ensure_dataset(exp_id)
    selected_classes = orch.canonical_classes(exp_id)
    assert len(selected_classes) == 100
    failed = []
    for p in plan:
        lane_dir = exp_dir / "lanes" / p["lane_id"]
        lane_dir.mkdir(parents=True, exist_ok=True)
        if (lane_dir / ".lane_done").exists():
            print(f"{p['lane_id']}: already complete, skip", flush=True)
            continue
        try:
            t0 = time.time()
            (lane_dir / "_config").mkdir(exist_ok=True)
            (lane_dir / "_config" / "sham_args.json").write_text(json.dumps({
                "branch": p["branch"], "condition": "sham", "window": args.window,
                "clean_lane": str(clean_dir), "branch_ckpt_label": p["label"],
                "lesion_enabled": False}, indent=2))
            if not (lane_dir / ".devtrain_done").exists():
                orch.log_status(exp_dir, "DEVTRAIN_START", lane=p["lane_id"],
                                start_epoch=p["branch"] + 1, milestone_epochs=p["mile_epochs"])
                lane = {"clean_dir": clean_dir, "branch": p["branch"]}
                orch.run_lesion_devtrain(lane_dir, p["cfg"], lane, {"branch_ckpt_label": p["label"]},
                                         selected_classes, exp_id, len(p["mile_epochs"]))
                (lane_dir / ".devtrain_done").touch()
                orch.log_status(exp_dir, "DEVTRAIN_DONE", lane=p["lane_id"],
                                mins=round((time.time() - t0) / 60, 1))
            if not args.skip_sae and not (lane_dir / ".sae_done").exists():
                t1 = time.time()
                orch.log_status(exp_dir, "SAE_START", lane=p["lane_id"])
                orch.run_sae(lane_dir, selected_classes, template.get("sae_policy", {}))
                if not (lane_dir / "sae_analysis" / "sae_results.json").exists():
                    raise RuntimeError("SAE finished but sae_results.json missing")
                (lane_dir / ".sae_done").touch()
                orch.log_status(exp_dir, "SAE_DONE", lane=p["lane_id"],
                                mins=round((time.time() - t1) / 60, 1))
            (lane_dir / ".lane_done").touch()
            orch.log_status(exp_dir, "LANE_DONE", lane=p["lane_id"],
                            mins=round((time.time() - t0) / 60, 1))
        except Exception as e:
            failed.append(p["lane_id"])
            (lane_dir / "FAILED.txt").write_text(f"{orch.now()}\n{e}\n\n{traceback.format_exc()}")
            orch.log_status(exp_dir, "LANE_FAILED", lane=p["lane_id"], error=str(e)[:200])
            traceback.print_exc()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
