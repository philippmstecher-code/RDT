"""Shared helpers for the imposed-grouping curriculum lanes (sc_rand).

* exact reconstruction of the imposed random partition of the sc_rand curriculum lanes
  (replicates devtrain.py of the training code, curriculum block, read-only);
* grouping masks in the Phase-4 format used by census_compute.selectivity_from_M.
"""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

sys.dont_write_bytecode = True  # never write __pycache__ into the training code directory

WT = Path(RDT_DATA + "/checkouts/lesion")
LANES = Path(RDT_DATA + "/extended/experiments/curriculum/lanes")
CIFAR_ROOT = Path(RDT_DATA + "/extended/experiments/curriculum/datasets/cifar100")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis_outputs" / "paper" / "IMPOSED_grouping"
SEEDS = [7, 42, 137, 256, 314]
ARMS = ["sc_rand_K20", "sc_true_K20", "scratch_E30"]
SWITCH_EPOCH = 20
END_EPOCH = 50


def lane_dir(arm: str, seed: int) -> Path:
    return LANES / f"{arm}_seed{seed}"


def cifar100_superclasses() -> Dict[str, List[str]]:
    """CIFAR100_SUPERCLASSES from the training code (run in memory, no import / bytecode)."""
    return runpy.run_path(str(WT / "packages" / "ml" / "src" / "cifar100_hierarchy.py"))["CIFAR100_SUPERCLASSES"]


def canonical_classes() -> List[str]:
    """As run_noise_curriculum_experiment.canonical_classes: torchvision CIFAR100(train=False).classes."""
    import torchvision
    ds = torchvision.datasets.CIFAR100(root=str(CIFAR_ROOT), train=False, download=False)
    return list(ds.classes)


def true_label_map(selected_classes: List[str], dataset_id: str = "cifar100") -> Dict[int, int]:
    """devtrain.py: fine idx -> superclass idx (superclasses sorted by name). Also asserts that
    every name is found (devtrain silently falls back to superclass 0 otherwise)."""
    sc = cifar100_superclasses()
    names = sorted(sc.keys())
    s2i = {n: i for i, n in enumerate(names)}
    f2s = {f: s2i[s] for s, fs in sc.items() for f in fs}
    out = {}
    for fine_idx, fine_name in enumerate(selected_classes):
        bare = fine_name.split("_", 1)[1] if fine_name.startswith(f"{dataset_id}_") else fine_name
        assert bare in f2s or fine_name in f2s, f"class {fine_name} not in hierarchy (devtrain would default to 0)"
        out[fine_idx] = f2s.get(bare, f2s.get(fine_name, 0))
    return out


def imposed_label_map(selected_classes: List[str], partition_seed: int) -> Dict[int, int]:
    """devtrain.py (curriculum block, random_partition_seed):
        _rng = np.random.RandomState(int(seed))
        _true_labels = [map[i] for i in sorted(map)]
        _perm_labels = list(_rng.permutation(_true_labels))
        map = {i: int(_perm_labels[k]) for k, i in enumerate(sorted(map))}"""
    m = true_label_map(selected_classes)
    rng = np.random.RandomState(int(partition_seed))
    tl = [m[i] for i in sorted(m)]
    pl = list(rng.permutation(tl))
    return {i: int(pl[k]) for k, i in enumerate(sorted(m))}


def partition_seed_of(seed: int) -> int:
    """run_curriculum.py: curriculum['random_partition_seed'] = 1000 + seed."""
    return 1000 + seed


def label_vector(m: Dict[int, int]) -> np.ndarray:
    return np.array([m[i] for i in sorted(m)], dtype=int)


def masks_from_labels(labels: np.ndarray):
    """Phase-4 mask format: one boolean row per group (group index order), groups with >= 2 classes."""
    groups = sorted(set(labels.tolist()))
    real = np.stack([labels == g for g in groups])
    ge2 = real[real.sum(axis=1) >= 2]
    return real, ge2


def lane_json(arm: str, seed: int) -> dict:
    return json.loads((lane_dir(arm, seed) / "_config" / "lane.json").read_text())
