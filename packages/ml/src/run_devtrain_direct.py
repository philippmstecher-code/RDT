#!/usr/bin/env python3
"""
Run DEVTRAIN for one lane without the backend.
Usage: python run_devtrain_direct.py --lane-dir <path>
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import sqlite3
import traceback
from pathlib import Path
from types import SimpleNamespace


class PydanticLikeNamespace(SimpleNamespace):
    """SimpleNamespace with model_dump() for Pydantic compatibility."""
    def model_dump(self):
        result = {}
        for k, v in self.__dict__.items():
            if callable(v):
                continue
            if isinstance(v, PydanticLikeNamespace):
                result[k] = v.model_dump()
            elif isinstance(v, list):
                result[k] = [item.model_dump() if isinstance(item, PydanticLikeNamespace) else item for item in v]
            else:
                result[k] = v
        return result

    def get(self, key, default=None):
        return getattr(self, key, default)


def dict_to_namespace(d):
    """Recursively convert dict to PydanticLikeNamespace."""
    if isinstance(d, dict):
        ns = PydanticLikeNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        return ns
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    return d


def add_missing_defaults(ns, config_dict):
    """Add missing attributes that the backend Pydantic models would provide."""
    # training_policy defaults
    if hasattr(ns, 'training_policy'):
        tp = ns.training_policy
        if not hasattr(tp, 'epochs'):
            tp.epochs = 50
        if not hasattr(tp, 'accuracy_threshold'):
            tp.accuracy_threshold = 70

    # snapshot_policy defaults
    if hasattr(ns, 'snapshot_policy'):
        sp = ns.snapshot_policy
        if not hasattr(sp, 'milestone_count'):
            sp.milestone_count = 10
        if not hasattr(sp, 'distribution_scheme'):
            sp.distribution_scheme = 'uniform'
        if not hasattr(sp, 'samples_per_class'):
            sp.samples_per_class = 100
        if not hasattr(sp, 'terminal_capture'):
            sp.terminal_capture = 'both'
        # milestone_accuracies is computed by the backend
        if not hasattr(sp, 'milestone_accuracies'):
            # Uniformly spaced milestone accuracies
            threshold = getattr(ns.training_policy, 'accuracy_threshold', 70)
            count = sp.milestone_count
            # Random accuracy baseline for 100 classes = 1%
            baseline = 1.0
            step = (threshold - baseline) / (count + 1)
            sp.milestone_accuracies = [baseline + step * (i + 1) for i in range(count)]
            print(f"  Computed milestone_accuracies: {[f'{a:.1f}' for a in sp.milestone_accuracies]}")

    # validation_policy defaults
    if hasattr(ns, 'validation_policy'):
        vp = ns.validation_policy
        if not hasattr(vp, 'frequency_mode'):
            vp.frequency_mode = 'percentage'
        if not hasattr(vp, 'frequency_value'):
            vp.frequency_value = 5
        if not hasattr(vp, 'min_batch_interval'):
            vp.min_batch_interval = 5
        if not hasattr(vp, 'validation_set_fraction'):
            vp.validation_set_fraction = 1.0
        def get_check_interval(total_batches, vp=vp):
            if vp.frequency_mode == 'percentage':
                interval = max(1, int(total_batches * vp.frequency_value / 100))
            else:
                interval = max(1, vp.frequency_value)
            return max(interval, vp.min_batch_interval)
        vp.get_check_interval = get_check_interval

    # memory_policy defaults
    if hasattr(ns, 'memory_policy'):
        mp = ns.memory_policy
        if not hasattr(mp, 'disk_based_linear_probes'):
            mp.disk_based_linear_probes = True
        if not hasattr(mp, 'disk_based_multilayer'):
            mp.disk_based_multilayer = True
        if not hasattr(mp, 'linear_probe_batch_size'):
            mp.linear_probe_batch_size = 32
        if not hasattr(mp, 'activation_extraction_enabled'):
            mp.activation_extraction_enabled = True
        if not hasattr(mp, 'activation_extraction_interval'):
            mp.activation_extraction_interval = 50

    # logging_policy defaults
    if hasattr(ns, 'logging_policy'):
        lp = ns.logging_policy
        if not hasattr(lp, 'batch_log_frequency'):
            lp.batch_log_frequency = 10
        if not hasattr(lp, 'batch_log_include_activations'):
            lp.batch_log_include_activations = False
        if not hasattr(lp, 'metrics_flush_interval'):
            lp.metrics_flush_interval = 10
        if not hasattr(lp, 'metrics_keep_recent'):
            lp.metrics_keep_recent = 50


def main():
    parser = argparse.ArgumentParser(description="Direct DEVTRAIN runner")
    parser.add_argument("--lane-dir", required=True, help="Path to lane directory")
    parser.add_argument("--clean", action="store_true", help="Clean existing snapshots first")
    args = parser.parse_args()

    lane_path = Path(args.lane_dir)

    # Load config from DB
    db_path = lane_path.parent.parent.parent.parent / "rcx.db"
    lane_id = lane_path.name
    experiment_id = lane_path.parent.parent.name

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute(
        "SELECT configuration FROM stoves WHERE lane_id = ? AND stove_type = 'NETINIT'",
        (lane_id,),
    )
    row = cursor.fetchone()
    config_dict = json.loads(row[0])

    cursor.execute(
        "SELECT dataset_id, selected_classes FROM experiments WHERE id = ?",
        (experiment_id,),
    )
    exp_row = cursor.fetchone()
    conn.close()

    dataset_id = exp_row[0]
    selected_classes = json.loads(exp_row[1])

    print(f"Lane: {lane_id}")
    print(f"Experiment: {experiment_id}")
    print(f"Dataset: {dataset_id}")
    print(f"Classes: {len(selected_classes)}")
    print(f"Network: {config_dict.get('network_type')}")
    print(f"Seed: {config_dict.get('random_seed')}")

    # Namespace that mimics the backend's Pydantic config objects
    netinit_config = dict_to_namespace(config_dict)
    add_missing_defaults(netinit_config, config_dict)

    # Paths
    initial_weights = lane_path / "weights" / "initial.pt"
    snapshots_base = lane_path / "dev_snapshots"
    metrics_dir = snapshots_base / "metrics"

    if args.clean:
        import shutil
        if snapshots_base.exists():
            print(f"\nCleaning {snapshots_base}...")
            shutil.rmtree(snapshots_base)

    snapshots_base.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nInitial weights: {initial_weights} (exists: {initial_weights.exists()})")
    print(f"Snapshots: {snapshots_base}")
    print(f"Metrics: {metrics_dir}")

    from devtrain import train_with_snapshots

    print("\n" + "=" * 60)
    print("STARTING DEVTRAIN (direct)")
    print("=" * 60 + "\n")

    try:
        result = train_with_snapshots(
            initial_weights_path=str(initial_weights),
            netinit_config=netinit_config,
            dataset_id=dataset_id,
            selected_classes=selected_classes,
            snapshots_base_dir=str(snapshots_base),
            metrics_dir=str(metrics_dir),
            experiment_id=experiment_id,
            stove_id=None,
        )

        print("\n" + "=" * 60)
        print("DEVTRAIN COMPLETE")
        print(f"Total snapshots: {result.get('total_snapshots', '?')}")
        print("=" * 60)

        # Verify training_state.pt files
        for d in sorted(snapshots_base.iterdir()):
            if d.name.startswith("milestone_"):
                ts = d / "training_state.pt"
                cp = d / "checkpoint.pt"
                print(f"  {d.name}: checkpoint={'YES' if cp.exists() else 'NO'} training_state={'YES' if ts.exists() else 'NO'}")

    except Exception as e:
        print("\n" + "=" * 60)
        print("DEVTRAIN CRASHED")
        print("=" * 60)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
