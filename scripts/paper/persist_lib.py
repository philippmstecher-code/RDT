"""Feature-lifespan records for the 9 core CIFAR-100 runs (used by persist_analyze.py, persist_200ep.py).

Builds per-feature lifespan records from the matching cache
(analysis_outputs/paper/TRANSFORMED/cache/<lane>.pkl; produced by
scripts/paper/transformed_matching_cache.py with the pipeline's own
`_phase5_feature_matching`) and the stored Phase-4 selectivity in sae_results.json.

Two constructions:

* ``records_reference`` (reference construction; reproduces the stored reference values): the
  construction of scripts/compute_feature_survival.py / backend `_compute_feature_survival`; forward map =
  stable links with >= 1 co-active sample (the per-sample transitions JSON only contains
  stable links that fire on some sample), ALL transitions incl. the identical
  '<k>->terminal' one, ALL layers (ResNet layer4 and avgpool both), records at every
  checkpoint incl. terminal, alive = in forward map or SSI>0 or CSI>0.

* ``records_real`` (real-transition construction): real transitions only (last real milestone is the right-censoring
  point), ResNet avgpool dropped (bit-identical input to layer4), alive = pipeline alive mask
  (>= 1% of samples), records at every real checkpoint that has >= 1 transition ahead.
  Chain rule ``paper`` (only stable r>=0.5 links continue a chain; transformed 0.2<=r<0.5
  links break it) or ``continuity`` (transformed links also continue the chain).

Read-only on the data directory.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

REPO = Path(__file__).resolve().parents[2]
DATA = Path(RDT_DATA + "/core")
CORE = DATA / "experiments" / "abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb" / "lanes"
CACHE = REPO / "analysis_outputs" / "paper" / "TRANSFORMED" / "cache"
CENSUS_THR = REPO / "analysis_outputs" / "paper" / "CENSUS_robust" / "tables" / "thresholds_per_run.csv"
OUT = REPO / "analysis_outputs" / "paper" / "PERSIST_robust"

CORE_LANES = [
    ("7db28a9a-f436-4d8a-a6b4-37e69fbf54eb", "ResNet-18", 42),
    ("a0bcbcba-94a9-4009-bcb6-d725b47588ee", "ResNet-18", 137),
    ("935c13ae-a026-4eae-a2b2-0306879b2e8c", "ResNet-18", 256),
    ("10d31151-68a2-4491-836d-3ead8e00a7ad", "ViT-Small", 42),
    ("909d7f68-55e1-4cd9-8c7b-68d5793798d3", "ViT-Small", 137),
    ("abad8975-de55-4e7e-b95d-05067f0eed90", "ViT-Small", 256),
    ("4bebc329-c26a-4eb4-b18c-7586e513f49e", "CCT-7", 42),
    ("edf8275e-c99f-4436-a13d-4ed50f072a66", "CCT-7", 137),
    ("445dd4d4-770a-4f33-b82b-c0bc7baed9a7", "CCT-7", 256),
]
ARCHS = ("ResNet-18", "ViT-Small", "CCT-7")
RUN_NAME = {"ResNet-18": "ResNet18-100class-10ms-seed{}", "ViT-Small": "VitSmall-100class-10ms-seed{}",
            "CCT-7": "CCT7-100class-10ms-seed{}"}
PAPER_LABEL = {"ResNet-18": "ResNet18-CIFAR100-seed{}", "ViT-Small": "ViTSmall-CIFAR100-seed{}",
               "CCT-7": "CCT7-CIFAR100-seed{}"}

REC_COLS = ("layer_i", "birth", "fid", "life", "maxp", "cens", "ssi", "csi", "sai", "chain_start")


def load_lane(lane_id: str):
    d = pickle.load(open(CACHE / f"{lane_id}.pkl", "rb"))
    res = json.loads((CORE / lane_id / "sae_analysis" / "sae_results.json").read_text())
    return d, res


def sel_arrays(selectivity, label, layer, n):
    s = selectivity.get(label, {}).get(layer, {})
    out = []
    for k in ("feature_ssi", "feature_csi", "feature_sai"):
        a = np.zeros(n, dtype=np.float64)
        v = np.asarray(s.get(k, []), dtype=np.float64)
        a[: min(n, len(v))] = v[:n]
        out.append(a)
    return out


def _n_feat(per_t, t):
    return len(per_t[t]["alive_a"])


def records_reference(d, res):
    """Reproduces the stored feature-survival records (feature_survival_tg_expanded.json) from the matching cache."""
    labels = d["labels"]
    selectivity = res["selectivity"]
    trans = [f"{a}->{b}" for a, b in zip(labels[:-1], labels[1:])]
    n_t = len(trans)
    rows = []
    for li, layer in enumerate(d["layers"]):
        per_t = d["layer_data"][layer]
        fwd = [{i: j for i, j, _, nco in per_t[t]["stable"] if nco > 0} for t in trans]
        for b, lab in enumerate(labels):
            s_raw = selectivity.get(lab, {}).get(layer, {})
            ssi_l = s_raw.get("feature_ssi", [])
            csi_l = s_raw.get("feature_csi", [])
            sai_l = s_raw.get("feature_sai", [])
            if not ssi_l:
                continue
            alive = set(fwd[b].keys()) if b < n_t else set()
            for fid in range(len(ssi_l)):
                if ssi_l[fid] > 0 or (fid < len(csi_l) and csi_l[fid] > 0):
                    alive.add(fid)
            for fid in alive:
                cur, life = fid, 0
                for k in range(b, n_t):
                    if cur in fwd[k]:
                        life += 1
                        cur = fwd[k][cur]
                    else:
                        break
                ssi = ssi_l[fid] if fid < len(ssi_l) else 0.0
                csi = csi_l[fid] if fid < len(csi_l) else 0.0
                sai = sai_l[fid] if fid < len(sai_l) else 0.0
                rows.append((li, b, fid, life, n_t - b, int(b + life >= n_t), ssi, csi, sai, 0))
    return np.array(rows, dtype=np.float64)


def records_real(d, res, rule: str, dedup: bool = True):
    """rule in {'paper', 'continuity'}; terminal transition excluded."""
    assert rule in ("paper", "continuity")
    labels = d["labels"]
    assert labels[-1] == "terminal"
    real = labels[:-1]
    trans = [f"{a}->{b}" for a, b in zip(real[:-1], real[1:])]
    n_t = len(trans)
    selectivity = res["selectivity"]
    layers = [l for l in d["layers"] if not (dedup and d["arch"] == "ResNet-18" and l == "avgpool")]
    rows = []
    for layer in layers:
        li = d["layers"].index(layer)
        per_t = d["layer_data"][layer]
        fwd = []
        for t in trans:
            m = {i: j for i, j, _, _ in per_t[t]["stable"]}
            if rule == "continuity":
                for i, j, _, _ in per_t[t]["transformed"]:
                    m[i] = j
            fwd.append(m)
        for b in range(n_t):  # checkpoints with >= 1 real transition ahead
            alive = np.where(per_t[trans[b]]["alive_a"])[0]
            ssi, csi, sai = sel_arrays(selectivity, real[b], layer, _n_feat(per_t, trans[b]))
            targets = set(fwd[b - 1].values()) if b > 0 else set()
            for fid in alive:
                cur, life = int(fid), 0
                for k in range(b, n_t):
                    if cur in fwd[k]:
                        life += 1
                        cur = fwd[k][cur]
                    else:
                        break
                rows.append((li, b, int(fid), life, n_t - b, int(b + life >= n_t),
                             ssi[fid], csi[fid], sai[fid], int(int(fid) not in targets)))
    return np.array(rows, dtype=np.float64)


def terminal_persistence(d, res, rule: str, dedup: bool = True):
    """Features alive at the last REAL checkpoint, traced backwards; returns array
    (layer_i, age/n_t, ssi, csi, sai) with selectivity at the last real checkpoint."""
    labels = d["labels"]
    real = labels[:-1]
    trans = [f"{a}->{b}" for a, b in zip(real[:-1], real[1:])]
    n_t = len(trans)
    selectivity = res["selectivity"]
    layers = [l for l in d["layers"] if not (dedup and d["arch"] == "ResNet-18" and l == "avgpool")]
    rows = []
    for layer in layers:
        li = d["layers"].index(layer)
        per_t = d["layer_data"][layer]
        back = []
        for t in trans:
            m = {j: i for i, j, _, _ in per_t[t]["stable"]}
            if rule == "continuity":
                for i, j, _, _ in per_t[t]["transformed"]:
                    m[j] = i
            back.append(m)
        last = trans[-1]
        nb = len(per_t[last]["alive_b"])
        ssi, csi, sai = sel_arrays(selectivity, real[-1], layer, nb)
        for j0 in np.where(per_t[last]["alive_b"])[0]:
            cur, age = int(j0), 0
            for k in range(n_t - 1, -1, -1):
                if cur in back[k]:
                    age += 1
                    cur = back[k][cur]
                else:
                    break
            rows.append((li, age / n_t, ssi[j0], csi[j0], sai[j0]))
    return np.array(rows, dtype=np.float64)


def terminal_persistence_reference(d, res):
    """Backward trace from the 'terminal' checkpoint through ALL transitions incl. the
    identical '<k>->terminal' one, all layers, stable links with co-activation."""
    labels = d["labels"]
    trans = [f"{a}->{b}" for a, b in zip(labels[:-1], labels[1:])]
    n_t = len(trans)
    selectivity = res["selectivity"]
    rows = []
    for li, layer in enumerate(d["layers"]):
        per_t = d["layer_data"][layer]
        back = [{j: i for i, j, _, nco in per_t[t]["stable"] if nco > 0} for t in trans]
        last = trans[-1]
        nb = len(per_t[last]["alive_b"])
        ssi, csi, sai = sel_arrays(selectivity, labels[-1], layer, nb)
        for j0 in np.where(per_t[last]["alive_b"])[0]:
            cur, age = int(j0), 0
            for k in range(n_t - 1, -1, -1):
                if cur in back[k]:
                    age += 1
                    cur = back[k][cur]
                else:
                    break
            rows.append((li, age / n_t, ssi[j0], csi[j0], sai[j0]))
    return np.array(rows, dtype=np.float64)


def cohort_labels(ssi, csi, sai, thr):
    """Mutually exclusive cohorts, priority SAI > SSI > CSI.
    0 = none, 1 = SAI, 2 = SSI, 3 = CSI."""
    c = np.zeros(len(ssi), dtype=np.int8)
    is_sai = sai > thr["sai"]
    is_ssi = (~is_sai) & (ssi > thr["ssi"])
    is_csi = (~is_sai) & (~is_ssi) & (csi > thr["csi"])
    c[is_sai] = 1
    c[is_ssi] = 2
    c[is_csi] = 3
    return c
