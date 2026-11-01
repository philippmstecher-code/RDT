#!/usr/bin/env python3
"""Split-half variability of the population superclass share share_sc = B_sc / B_cls (raw and within-class-whitened
space) over 20 random 50/50 image splits per class, for every open lane x checkpoint x layer, plus an image
bootstrap (resample images within class with replacement, then split; 20 replicates) for the pooled layer.
Writes tables/share_sc_splits.csv (mean and sd over splits; bootstrap sd).

Usage: dictfree_share_splits.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dictfree_lib import OUT, open_lanes, load_acts, superclass_of_class  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dictfree_compute import Ctx, var_parts, whiten  # noqa: E402

N_SPLIT = 20


def share(X, idxA, idxB, masks):
    MA, MB = X[idxA].mean(1), X[idxB].mean(1)
    DA, DB = MA - MA.mean(0), MB - MB.mean(0)
    bc, bs = var_parts(DA, DB, masks)
    return bs.sum() / bc.sum(), bc.sum(), bs.sum()


def splits(rng, boot=False):
    A, B = [], []
    for c in range(100):
        imgs = rng.choice(100, 100, replace=True) if boot else rng.permutation(100)
        A.append(c * 100 + imgs[:50])
        B.append(c * 100 + imgs[50:])
    return np.array(A), np.array(B)


def main():
    rows = []
    for ln in open_lanes():
        ctx = Ctx(superclass_of_class(ln["dir"]))
        for c in ln["ckpts"]:
            acts = load_acts(ln, c["label"])
            for L in ln["layers"]:
                X = acts[L]
                Xw = whiten(X)
                rng = np.random.RandomState(11)
                sp = [splits(rng) for _ in range(N_SPLIT)]
                r = {"lane": ln["key"], "arch": ln["arch"], "group": ln["group"], "seed": ln["seed"],
                     "checkpoint": c["label"], "idx": c["idx"], "tau": c["tau"], "epoch": c["epoch"],
                     "val_acc": c["val_acc"], "layer": L, "is_pooled": L == ln["pooled"]}
                for tag, Z in (("raw", X), ("wht", Xw)):
                    v = np.array([share(Z, a, b, ctx.masks) for a, b in sp])
                    r[f"{tag}_share_mean"], r[f"{tag}_share_sd"] = v[:, 0].mean(), v[:, 0].std(ddof=1)
                    r[f"{tag}_fsc_mean"] = v[:, 2].mean() / Z.var(0).sum()
                    r[f"{tag}_fwsc_mean"] = (v[:, 1] - v[:, 2]).mean() / Z.var(0).sum()
                    if L == ln["pooled"]:
                        rb = np.random.RandomState(12)
                        vb = np.array([share(Z, *splits(rb, boot=True), ctx.masks)[0] for _ in range(N_SPLIT)])
                        r[f"{tag}_share_boot_sd"] = vb.std(ddof=1)
                rows.append(r)
            print(ln["key"], c["label"], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "tables" / "share_sc_splits.csv", index=False)


if __name__ == "__main__":
    main()
