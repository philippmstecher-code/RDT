#!/usr/bin/env python3
"""
Run SAEANALYSIS for one lane without the backend.
Usage: python run_saeanalysis_direct.py --lane-dir <path>
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import sqlite3
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Direct SAEANALYSIS runner")
    parser.add_argument("--lane-dir", required=True, help="Path to lane directory")
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
    config = json.loads(row[0])

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

    experiment_config = {
        "dataset_id": dataset_id,
        "sae_policy": config.get("sae_policy", {}),
    }
    print(f"SAE policy: {json.dumps(experiment_config['sae_policy'], indent=2)}")

    from saeanalysis import analyze_sae_features

    def progress_cb(pct, msg):
        print(f"  [{pct:.0f}%] {msg}")

    print("\n" + "=" * 60)
    print("STARTING SAEANALYSIS (direct)")
    print("=" * 60 + "\n")

    try:
        result = analyze_sae_features(
            lane_dir=str(lane_path),
            selected_classes=selected_classes,
            experiment_config=experiment_config,
            progress_callback=progress_cb,
        )

        print("\n" + "=" * 60)
        print("SAEANALYSIS COMPLETE")
        print("=" * 60)

        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            results_path = lane_path / "sae_analysis" / "sae_results.json"
            print(f"Results: {results_path} (exists: {results_path.exists()})")
            if "superclass_summary" in result:
                print(f"Superclasses analyzed: {len(result['superclass_summary'])}")

    except Exception as e:
        print("\n" + "=" * 60)
        print("SAEANALYSIS CRASHED")
        print("=" * 60)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
