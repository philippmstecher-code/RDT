"""Exact information decomposition of the fine-label log-likelihood into coarse + within-group parts.

For a partition G of the 100 classes into 20 groups of 5:
  log p(y) = log p_G(g(y)) + log p(y | g(y))      (p_G = summed softmax; p(y|g) = restricted 5-way softmax)
  I_fine = E[log p(y)] + log 100 ;  I_G = E[log p_G(g(y))] + log 20 ;  I_wG = I_fine - I_G  (exact)
  share_G = I_G / I_fine. Under 'resolved-uncertainty equipartition' I_G/log20 = I_wG/log5 -> share = log20/log100 = 0.651.
Controls: (a) 50 random 20x5 partitions (same logits), (b) flat-confusion model: per image keep p(y), spread the rest
uniformly over the other 99 classes.
Output: infoshare_network.csv (lane x ckpt), infoshare_superclass.csv (per true superclass)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timetable_lib import CACHE, CLS_OF_IMG, OUT, SC_OF_CLASS, SC_OF_IMG, lanes  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

N_RAND = 50


def logsoftmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    return Z - np.log(np.exp(Z).sum(1, keepdims=True))


def group_info(logP, groups_of_class):
    """E-terms per image: log p_G(g(y)) (vector over images)."""
    P = np.exp(logP)
    G = np.zeros((100, 20))
    G[np.arange(100), groups_of_class] = 1
    PG = P @ G
    return np.log(np.clip(PG[np.arange(len(P)), groups_of_class[CLS_OF_IMG]], 1e-30, None))


def main():
    rng = np.random.default_rng(0)
    rand_parts = []
    for _ in range(N_RAND):
        perm = rng.permutation(100)
        g = np.empty(100, int)
        g[perm] = np.repeat(np.arange(20), 5)
        rand_parts.append(g)
    rows, srows = [], []
    for key, lane in lanes().items():
        for ti, lab in enumerate(lane["labels"]):
            f = CACHE / f"{key}_m{lab}_logits.npy"
            if not f.exists():
                continue
            logP = logsoftmax(np.load(f).astype(np.float64))
            ly = logP[np.arange(10000), CLS_OF_IMG]
            I_fine_img = ly + np.log(100)
            lsc = group_info(logP, SC_OF_CLASS)
            I_sc_img = lsc + np.log(20)
            I_fine, I_sc = I_fine_img.mean(), I_sc_img.mean()
            rs = []
            for g in rand_parts:
                rs.append((group_info(logP, g) + np.log(20)).mean() / I_fine)
            # flat-confusion model
            q = np.exp(ly)
            psc_flat = q + 4 * (1 - q) / 99
            I_sc_flat = (np.log(psc_flat) + np.log(20)).mean()
            acc = (logP.argmax(1) == CLS_OF_IMG).mean()
            rows.append(dict(lane=key, arch=lane["arch"], seed=lane["seed"], ckpt=ti, epoch=lane["epochs"][ti], acc=acc,
                             I_fine=I_fine, I_sc=I_sc, I_ws=I_fine - I_sc, share_true=I_sc / I_fine,
                             share_rand_mean=np.mean(rs), share_rand_sd=np.std(rs), share_rand_max=np.max(rs),
                             share_flat=I_sc_flat / I_fine,
                             frac_sc=I_sc / np.log(20), frac_ws=(I_fine - I_sc) / np.log(5)))
            for s in range(20):
                m = SC_OF_IMG == s
                srows.append(dict(lane=key, arch=lane["arch"], ckpt=ti, epoch=lane["epochs"][ti], sc=s,
                                  I_fine=I_fine_img[m].mean(), I_sc=I_sc_img[m].mean(),
                                  I_ws=(I_fine_img[m] - I_sc_img[m]).mean()))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "infoshare_network.csv", index=False)
    pd.DataFrame(srows).to_csv(OUT / "infoshare_superclass.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(d[["lane", "epoch", "acc", "I_fine", "share_true", "share_rand_mean", "share_rand_max", "share_flat",
                 "frac_sc", "frac_ws"]].round(3).to_string())


if __name__ == "__main__":
    main()
