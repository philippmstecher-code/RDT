"""Saxe-style mode strength vs group learning timing, true superclasses vs random 20x5 partitions.

Mode strength of group G: s_G = || W (mu_G - mu) ||, W = PCA-64 whitening of standardised training pixels
(group mean = mean of member class means; class priors equal).  Also class-level s_c and within-group s_{c|G}.
Timing per lane (group prob p_G(true group) averaged over images of the group):
  lead (final-normalised, chance baseline), final, t50 = epoch at which (p - chance)/(p_final - chance) first reaches 0.5
  (linear interpolation in log(1+epoch)).
Saxe et al. (2019) deep linear nets: learning time ~ 1/s  -> slope of log t50 on log s ~ -1.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timetable_lib import CACHE, CLS_OF_IMG, OUT, SC_OF_CLASS, lanes, load_split  # noqa: E402
from timetable_infoshare import logsoftmax  # noqa: E402
from timetable_predict import partial_spearman  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

N_RAND = 30


def whitened_class_means(k=64):
    X, y = load_split(True)
    Z = X.reshape(len(X), -1).astype(np.float32) / 255.0
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
    ev, V = np.linalg.eigh(Z.T @ Z / len(Z))
    ev, V = ev[::-1], V[:, ::-1]
    W = V[:, :k] / np.sqrt(ev[:k] + 1e-3 * ev[0])
    return np.stack([Z[y == c].mean(0) for c in range(100)]) @ W


def t_cross(ep, prog, thr=0.5):
    x = np.log1p(ep)
    for i in range(1, len(prog)):
        if prog[i] >= thr:
            if prog[i - 1] >= thr:
                return np.expm1(x[i - 1])
            f = (thr - prog[i - 1]) / (prog[i] - prog[i - 1])
            return np.expm1(x[i - 1] + f * (x[i] - x[i - 1]))
    return np.nan


def group_curves(key, lane, g):
    """(T, 20) mean prob of true group per group."""
    out = []
    G = np.zeros((100, 20))
    G[np.arange(100), g] = 1
    gi = g[CLS_OF_IMG]
    for lab in lane["labels"]:
        P = np.exp(logsoftmax(np.load(CACHE / f"{key}_m{lab}_logits.npy").astype(np.float64)))
        pg = (P @ G)[np.arange(10000), gi]
        out.append(np.bincount(gi, pg, 20) / np.bincount(gi, None, 20))
    return np.stack(out)


def timing(C, ep, b=0.05):
    A = C.mean(1)
    rel = (C - b) / (C[-1] - b)
    Ag = (A - b) / (A[-1] - b)
    lead = np.mean(rel[1:-1] - Ag[1:-1, None], 0)
    t50 = np.array([t_cross(ep, rel[:, u]) for u in range(C.shape[1])])
    return lead, C[-1], t50


def main():
    M = whitened_class_means()
    mu = M.mean(0)
    rng = np.random.default_rng(1)
    parts = [("true", SC_OF_CLASS)]
    for r in range(N_RAND):
        g = np.empty(100, int)
        g[rng.permutation(100)] = np.repeat(np.arange(20), 5)
        parts.append((f"rand{r}", g))
    rows, units = [], []
    for key, lane in lanes().items():
        ep = np.array(lane["epochs"])
        if lane["arch"] == "ViT-Small":  # last ViT checkpoint (epoch 100) is far beyond the others; keep as 'final'
            pass
        for name, g in parts:
            s = np.linalg.norm(np.stack([M[g == j].mean(0) for j in range(20)]) - mu, axis=1)
            C = group_curves(key, lane, g)
            lead, fin, t50 = timing(C, ep)
            ok = np.isfinite(t50) & (t50 > 0)
            slope = np.polyfit(np.log(s[ok]), np.log(t50[ok]), 1)[0] if ok.sum() > 5 else np.nan
            rows.append(dict(lane=key, arch=lane["arch"], partition=name, rho_lead=spearmanr(s, lead)[0],
                             rho_final=spearmanr(s, fin)[0], partial_lead=partial_spearman(s, lead, fin),
                             rho_t50=spearmanr(s[ok], t50[ok])[0], slope_logt50_logs=slope, n_t50=int(ok.sum())))
            if name == "true":
                for u in range(20):
                    units.append(dict(lane=key, arch=lane["arch"], sc=u, s=s[u], lead=lead[u], final=fin[u], t50=t50[u]))
        d = pd.DataFrame([r for r in rows if r["lane"] == key])
        t, rr = d[d.partition == "true"].iloc[0], d[d.partition != "true"]
        print(f"{key}: true rho_lead={t.rho_lead:.2f} partial={t.partial_lead:.2f} rho_t50={t.rho_t50:.2f} "
              f"slope={t.slope_logt50_logs:.2f} | random: rho_lead={rr.rho_lead.mean():.2f}+-{rr.rho_lead.std():.2f} "
              f"partial={rr.partial_lead.mean():.2f}+-{rr.partial_lead.std():.2f} slope={rr.slope_logt50_logs.mean():.2f}"
              f"+-{rr.slope_logt50_logs.std():.2f} frac_rand_partial>=true={(rr.partial_lead >= t.partial_lead).mean():.2f}",
              flush=True)
    pd.DataFrame(rows).to_csv(OUT / "saxe_partition_test.csv", index=False)
    pd.DataFrame(units).to_csv(OUT / "saxe_superclass_units.csv", index=False)


if __name__ == "__main__":
    main()
