#!/usr/bin/env python3
"""Example images for the Fig. 6a schematic: one CIFAR-100 test image (the fourth of its class) for each of the
five classes of three superclasses (flowers, vehicles 1, large carnivores), so that the figure builder does not need
the dataset itself.
Writes analysis_outputs/paper/figures/inputs/fig6_cifar_examples.npz (images uint8 (3, 5, 32, 32, 3), superclass
and class names).
Usage: .venv/bin/python scripts/paper/fig6_cifar_examples.py [<cifar-100-python dir>]   (default $CIFAR100_DIR)
"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CIFAR = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CIFAR100_DIR", "data/cifar-100-python"))
OUT = REPO / "analysis_outputs/paper/figures/inputs/fig6_cifar_examples.npz"
SUPER = ("flowers", "vehicles_1", "large_carnivores")


def main():
    meta = pickle.load(open(CIFAR / "meta", "rb"), encoding="latin1")
    d = pickle.load(open(CIFAR / "test", "rb"), encoding="latin1")
    X = d["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    fine, coarse = np.array(d["fine_labels"]), np.array(d["coarse_labels"])
    imgs, names = [], []
    for s in SUPER:
        c = meta["coarse_label_names"].index(s)
        ks = sorted(set(fine[coarse == c]))
        imgs.append([X[np.where(fine == k)[0][3]] for k in ks])
        names.append([meta["fine_label_names"][k] for k in ks])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, images=np.array(imgs, dtype=np.uint8), superclasses=np.array(SUPER),
                        classes=np.array(names))
    print("wrote", OUT, np.array(imgs).shape)


if __name__ == "__main__":
    main()
