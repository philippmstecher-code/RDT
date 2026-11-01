#!/usr/bin/env python3
"""Superclass structure in a ResNet-18 trained without labels (SimCLR, simclr_train.py) against the supervised sham runs.

Per checkpoint, on the pooled-layer (avgpool, 512-d) activations of the 10,000 CIFAR-100 test images (100 per class,
class-major):
  (1) code share: B_cls = between-class scatter, Q = rank-19 leading eigenvectors of the
      between-group scatter of a grouping (coarse_overlap.between_scatter / top); share = tr(Q' B_cls Q) / tr(B_cls).
      Grouping = the 20 CIFAR-100 superclasses (sc_code_share) and the 20 random 20 x 5 groupings
      RandomState(p).permutation(100).reshape(20, 5), p = 0..19 (coarse_overlap.grouping; rg_code_share_pNN, with
      their mean and max); sc_code_exceeds_all_rg = superclass share above all 20.
  (2) Extended Data Fig. 1g measure (dictfree_compute.Ctx + population, tag 'raw'): split-half
      unbiased between-superclass variance / between-class variance of the unwhitened activations (ed1g_share_sc),
      with its null over 100 class->superclass permutations (RandomState(42); ed1g_null_mean, ed1g_null_p95, ed1g_z).
  (3) linear probes (curriculum_probe.probe: standardised features, RidgeClassifier alpha 10, 5-fold stratified CV):
      20-way superclass (probe_super20) and 100-way fine class (probe_fine100).
Models: simclr (<root>/seed<S>/epoch_<EEE>/pooled_test.npz, epochs 0-100 as stored) and sham (critical_sham
sham_e00_seed<S>, stored avgpool activations, curriculum_probe.load; epoch 0 = milestone_0, the initialisation with zero
weight updates; epochs 1, 2, 3, 5, 10, 20, 30 = coarse_overlap.milestones). Seeds 7, 42, 137.
Writes <out>/simclr_measures.csv (long: model, seed, epoch, measure, value), <out>/simclr_summary.json (mean over seeds
per model and epoch; per-seed superclass-above-all-random flags) and a per-checkpoint cache <out>/cache/. CPU only.
Usage: python scripts/paper/simclr_analysis.py [--root DIR] [--out DIR] [--seeds 7 42 137] [--no-sham]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coarse_overlap import between_scatter, grouping, lane, milestones, top  # noqa: E402
from dictfree_compute import Ctx, population  # noqa: E402
from curriculum_probe import SUPER_OF, load, probe  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(RDT_DATA + "/extended/experiments/simclr_train")
OUT = REPO / "analysis_outputs/paper/SIMCLR"
SEEDS = [7, 42, 137]
SIMCLR_EPOCHS = (0, 1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
SHAM_EPOCHS = (0, 1, 2, 3, 5, 10, 20, 30)
N_RG = 20
Y_CANON = np.repeat(np.arange(100), 100)
GROUPS_RG = [grouping(np.random.RandomState(p)) for p in range(N_RG)]
CTX = Ctx(SUPER_OF)


def measures(X: np.ndarray, y: np.ndarray) -> dict:
    assert X.shape == (10000, 512) and np.array_equal(y, Y_CANON), "expects 100 test images per class, class-major"
    X64 = X.astype(np.float64)
    B = between_scatter(X64, y, np.arange(100))
    trB = np.trace(B)

    def share(g):
        Q = top(between_scatter(X64, y, g))
        return float(np.trace(Q.T @ B @ Q) / trB)

    sc = share(SUPER_OF)
    rg = np.array([share(g) for g in GROUPS_RG])
    o = {"sc_code_share": sc, "rg_code_share_mean": float(rg.mean()), "rg_code_share_max": float(rg.max()),
         "sc_code_exceeds_all_rg": float(sc > rg.max())}
    o.update({f"rg_code_share_p{p:02d}": float(v) for p, v in enumerate(rg)})
    p = population(X64, CTX, "raw")
    o.update({"ed1g_share_sc": p["raw_share_sc"], "ed1g_null_mean": p["raw_share_sc_null_mean"],
              "ed1g_null_p95": p["raw_share_sc_null_p95"], "ed1g_z": p["raw_share_sc_z"]})
    X32 = X.astype(np.float32)
    o["probe_super20"] = float(probe(X32, SUPER_OF[y]))
    o["probe_fine100"] = float(probe(X32, y))
    return o


def simclr_items(root: Path, seeds):
    for s in seeds:
        for e in SIMCLR_EPOCHS:
            f = root / f"seed{s}" / f"epoch_{e:03d}" / "pooled_test.npz"
            if f.exists():
                yield "simclr", s, e, f


def load_simclr(f: Path):
    z = np.load(f)
    y = z["y_fine"].astype(int)
    assert np.array_equal(SUPER_OF[y], z["y_super"].astype(int)), "superclass labels disagree with SUPER_OF"
    return z["X"], y


def sham_items(seeds, epochs):
    for s in seeds:
        ld = lane("sham", s)
        ms = milestones(ld)
        snaps = json.load(open(ld / "dev_snapshots/metrics/devtrain_snapshots.json"))
        m0 = [x for x in snaps if x["milestone_index"] == 0][0]
        assert m0["weight_updates"] == 0, (s, m0)  # milestone_0 = initialisation
        ms[0] = ld / "dev_snapshots" / "milestone_0"
        for e in epochs:
            yield "sham", s, e, ms[e]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--no-sham", action="store_true")
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    cache = a.out / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    items = list(simclr_items(a.root, a.seeds))
    if not a.no_sham:
        items += list(sham_items(a.seeds, SHAM_EPOCHS))
    rows = []
    for model, s, e, src in items:
        cf = cache / f"{model}_s{s}_e{e:03d}.json"
        if cf.exists() and not a.recompute:
            o = json.load(open(cf))
        else:
            X, y = load_simclr(src) if model == "simclr" else load(src, "avgpool")
            o = measures(X, y)
            o["_src"] = str(src)
            json.dump(o, open(cf, "w"), indent=1)
        print(model, s, e, {k: round(v, 4) for k, v in o.items()
                            if k in ("sc_code_share", "rg_code_share_mean", "rg_code_share_max", "ed1g_share_sc",
                                     "ed1g_null_p95", "probe_super20", "probe_fine100")}, flush=True)
        rows += [{"model": model, "seed": s, "epoch": e, "measure": k, "value": v}
                 for k, v in o.items() if not k.startswith("_")]
    D = pd.DataFrame(rows)
    D.to_csv(a.out / "simclr_measures.csv", index=False)

    main_m = ["sc_code_share", "rg_code_share_mean", "rg_code_share_max", "sc_code_exceeds_all_rg", "ed1g_share_sc",
              "ed1g_null_mean", "ed1g_null_p95", "ed1g_z", "probe_super20", "probe_fine100"]
    W = D[D.measure.isin(main_m)].pivot_table(index=["model", "epoch", "seed"], columns="measure", values="value")
    by = W.groupby(["model", "epoch"])
    summ = by.mean().round(4)
    summ["n_seeds"] = by.size()
    summ["n_seeds_sc_above_all_rg"] = by.sc_code_exceeds_all_rg.sum().astype(int)
    summ["n_seeds_ed1j_above_null_p95"] = by.apply(lambda g: int((g.ed1g_share_sc > g.ed1g_null_p95).sum()))
    res = {"root": str(a.root), "seeds": a.seeds,
           "by_model_epoch": summ.drop(columns="sc_code_exceeds_all_rg").reset_index().to_dict("records"),
           "per_seed": W.round(4).reset_index().to_dict("records")}
    json.dump(res, open(a.out / "simclr_summary.json", "w"), indent=1)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summ.drop(columns="sc_code_exceeds_all_rg"))


if __name__ == "__main__":
    main()
