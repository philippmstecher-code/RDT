#!/usr/bin/env python3
"""How much of the coarse (superclass-aligned) structure does a random-grouping code also contain, and how much coarse
information remains in the network when either code is removed during training?

Part 1, overlap (sham networks, critical_sham sham_e00_seed{7,42,137,256,314}, epochs 1, 2, 3, 5, 10, last block =
layer4 and the pooled layer; stored activations, 100 test images per class):
  For a grouping g of the 100 classes, B_g = between-group scatter of the activations, Q_g = its 19 leading eigenvectors
  (the code removed by the suppression arms, at the maximal per-refit rank). Superclass code Q_sc; random-grouping codes
  Q_rg for the grouping each seed used in the random-grouping arm (RandomState(10000 + seed)) and for 20 further
  random groupings (RandomState(p), p = 0..19); random orthonormal subspaces of the same rank as a null.
  captured_coarse = tr(Q_rg' B_sc Q_rg) / tr(Q_sc' B_sc Q_sc): the share of the superclass between-group variance held by
  the superclass code that a random-grouping code of the same rank also holds (1 = removes all coarse structure the
  superclass code removes; null = random subspace).
  cos2 = ||Q_sc' Q_rg||_F^2 / 19: mean squared cosine of the principal angles.
Part 2, coarse information during training: 20-way superclass linear probe (curriculum_probe.probe: ridge, 5-fold CV) on the
  stored pooled-layer activations of every checkpoint (epochs 1-15, 20, 25, 30), per arm and seed. For the suppression
  arms the activations were recorded with the projection on, i.e. as the next layers saw them.
  Arms: sham; superclass (prevention lesion_scaffold_e00); random (lesion_random_variance_e00); randgroup
  (randgroup lesion_randgroup_e00; seeds whose run is complete).
Writes analysis_outputs/paper/RANDGROUP_control/coarse_overlap.csv, coarse_probe.csv, coarse_summary.json.
Usage: python scripts/paper/coarse_overlap.py [overlap|probe|all]
"""
import json
import os
import sys
from pathlib import Path
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curriculum_probe import SUPER_OF, load, probe  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis_outputs/paper/RANDGROUP_control"
E = Path(RDT_DATA + "/extended/experiments")
FEST = Path(RDT_DATA + "/core/experiments")
SEEDS = [7, 42, 137, 256, 314]
RANK = 19
LAYERS = ("layer4", "avgpool")


def grouping(rs):
    perm = rs.permutation(100).reshape(20, 5)
    g = np.zeros(100, int)
    for k, row in enumerate(perm):
        g[row] = k
    return g


def between_scatter(X, y_cls, groups):
    lab = groups[y_cls]
    mu = X.mean(0)
    B = np.zeros((X.shape[1], X.shape[1]))
    for k in np.unique(lab):
        m = lab == k
        d = X[m].mean(0) - mu
        B += m.mean() * np.outer(d, d)
    return B


def top(B, r=RANK):
    w, v = np.linalg.eigh(B)
    return v[:, ::-1][:, :r]


def lane(arm, s):
    if arm == "sham":
        return E / "critical_sham/lanes" / f"sham_e00_seed{s}"
    if arm == "randgroup":
        return E / "randgroup/lanes" / f"lesion_randgroup_e00_seed{s}_rp65_Llayer4-avgpool"
    nm = {"superclass": "lesion_scaffold_e00_seed{}_rp65_Llayer4-avgpool",
          "random": "lesion_random_variance_e00_seed{}_rp65_Llayer4-avgpool"}[arm].format(s)
    for root in ((E / "prevention_s42_copy/lanes") if s == 42 else (E / "prevention/lanes"),
                 FEST / "scaffold_lesion/lanes"):
        if (root / nm / "dev_snapshots/milestone_1/individual_multilayer_activations.pt").exists():
            return root / nm
    return None


def milestones(ld):
    """{epoch: milestone dir} for epochs 1-15, 20, 25, 30 (intermediate checkpoints only)."""
    snaps = json.load(open(ld / "dev_snapshots/metrics/devtrain_snapshots.json"))
    out = {}
    for x in snaps:
        if x.get("snapshot_type") == "intermediate":
            out[int(x["epoch"])] = ld / "dev_snapshots" / f"milestone_{x['milestone_index']}"
    return out


def overlap():
    rows = []
    rng = np.random.default_rng(0)
    for s in SEEDS:
        ms = milestones(lane("sham", s))
        groups = {"run": grouping(np.random.RandomState(10000 + s))}
        groups.update({f"removal_{p}": grouping(np.random.RandomState(p)) for p in range(20)})
        for e in (1, 2, 3, 5, 10):
            for layer in LAYERS:
                X, y = load(ms[e], layer)
                X = X.astype(np.float64)
                Bsc = between_scatter(X, y, SUPER_OF)
                Qsc = top(Bsc)
                base = np.trace(Qsc.T @ Bsc @ Qsc)
                for gname, g in groups.items():
                    Q = top(between_scatter(X, y, g))
                    rows.append({"seed": s, "epoch": e, "layer": layer, "subspace": gname,
                                 "captured_coarse": float(np.trace(Q.T @ Bsc @ Q) / base),
                                 "cos2": float(((Qsc.T @ Q) ** 2).sum() / RANK)})
                for k in range(5):
                    Q, _ = np.linalg.qr(rng.standard_normal((X.shape[1], RANK)))
                    rows.append({"seed": s, "epoch": e, "layer": layer, "subspace": f"random_{k}",
                                 "captured_coarse": float(np.trace(Q.T @ Bsc @ Q) / base),
                                 "cos2": float(((Qsc.T @ Q) ** 2).sum() / RANK)})
            print("overlap", s, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "coarse_overlap.csv", index=False)
    return d


def probes():
    rows = []
    for arm in ("sham", "superclass", "random", "randgroup"):
        for s in SEEDS:
            ld = lane(arm, s)
            if ld is None or not (ld / "dev_snapshots/metrics/devtrain_snapshots.json").exists():
                continue
            ms = milestones(ld)
            if arm == "randgroup" and 30 not in ms:
                continue  # run not complete
            for e in sorted(ms):
                X, y = load(ms[e], "avgpool")
                rows.append({"arm": arm, "seed": s, "epoch": e, "superclass_probe": float(probe(X, SUPER_OF[y]))})
            print("probe", arm, s, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "coarse_probe.csv", index=False)
    return d


def summarize(ov, pr):
    res = {}
    if ov is not None:
        ov = ov.assign(kind=ov.subspace.str.replace(r"_\d+$", "", regex=True))
        res["overlap_mean_by_epoch_layer_kind"] = (ov.groupby(["layer", "epoch", "kind"])[["captured_coarse", "cos2"]]
                                                   .mean().round(3).reset_index().to_dict("records"))
    if pr is not None and len(pr):
        p = pr.pivot_table(index=["seed", "epoch"], columns="arm", values="superclass_probe")
        for arm in ("superclass", "random", "randgroup"):
            if arm in p:
                d = (p[arm] - p["sham"]).dropna().unstack("seed")
                res[f"probe_{arm}_minus_sham_by_epoch"] = {int(e): [round(float(v), 4) for v in d.loc[e].dropna()]
                                                           for e in d.index}
    json.dump(res, open(OUT / "coarse_summary.json", "w"), indent=1)
    return res


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    OUT.mkdir(parents=True, exist_ok=True)
    ov = overlap() if cmd in ("overlap", "all") else (pd.read_csv(OUT / "coarse_overlap.csv")
                                                      if (OUT / "coarse_overlap.csv").exists() else None)
    pr = probes() if cmd in ("probe", "all") else (pd.read_csv(OUT / "coarse_probe.csv")
                                                   if (OUT / "coarse_probe.csv").exists() else None)
    res = summarize(ov, pr)
    print(json.dumps(res, indent=1)[:4000])


if __name__ == "__main__":
    main()
