#!/usr/bin/env python3
"""Census robustness: compute step.

For the 9 core runs + Tiny ImageNet + independent-init run, per checkpoint and layer, on the
stored SAE codes (read-only):
  * real selectivity (SSI, CSI_local, SAI) recomputed vectorised from the class-mean |code|
    matrix and checked against the stored Phase-4 values in sae_results.json;
  * permutation nulls following saeanalysis._phase4b_adaptive_thresholds (n_perm = 100 instead
    of the 1000 used by the stored runs; same RNG seed/structure):
      - SSI null: permute class->superclass assignment (sizes preserved), all features;
      - SAI null (pipeline scheme): identical permutation; SAI is invariant to it, so the
        pipeline "null" equals the real SAI distribution -> stored as real SAI;
      - SAI null (sample scheme, diagnostic): SAI after shuffling sample->class rows;
      - CSI null: shuffle sample->class rows, CSI_local in the best real superclass; the
        pipeline keeps values > eps from best superclasses with >= 5 classes (flag stored).
  * per feature: active-sample count, alive flag (>= 1% of samples), non-zero flag.

Writes <scratch>/census_null/<lane_id>.npz (raw arrays, not a deliverable) and
analysis_outputs/paper/CENSUS_robust/per_lane/<lane_id>_meta.json (validation + metadata).

Usage: census_compute.py --scratch DIR [--workers 4] [--only PREFIX]
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sqlite3
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noise_null_lib import EXP, REPO, Lane  # noqa: E402

OUT = REPO / "analysis_outputs" / "paper" / "CENSUS_robust"
N_PERM = 100
EPS = 1e-10
ACT_EPS = 1e-6
ALIVE_FRAC = 0.01
MIN_SC_FOR_NULL = 5

CORE_EXP = "abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb"
TIN_EXP = "821eaadb-0160-48f8-9669-ec3d9185c428"
IND_EXP = "independent_sae"
GROUP = {CORE_EXP: "core", TIN_EXP: "tinyimagenet", IND_EXP: "indep_init"}
DATASET = {CORE_EXP: "cifar100", TIN_EXP: "tiny_imagenet", IND_EXP: "cifar100"}


def lane_names() -> Dict[str, str]:
    names: Dict[str, str] = {}
    for db in ("rcx.db", "rcx_RUNPOD.db"):
        p = EXP.parent / db
        if not p.exists():
            continue
        c = sqlite3.connect(f"file:{p}?immutable=1", uri=True)
        for lid, nm in c.execute("select id, name from lanes"):
            names.setdefault(lid, nm)
        c.close()
    return names


def lanes_to_run() -> List[Tuple[str, Path, str, str, str]]:
    names = lane_names()
    out = []
    for exp, grp in GROUP.items():
        for d in sorted((EXP / exp / "lanes").iterdir()):
            if not (d / "sae_analysis" / "sae_results.json").exists():
                continue
            out.append((d.name, d, DATASET[exp], grp, names.get(d.name, d.name[:8])))
    return out


def _lit(x):
    return ast.literal_eval(x) if isinstance(x, str) else x


def selectivity_from_M(M: np.ndarray, sc_masks: np.ndarray, sc_masks_ge2: np.ndarray, n_sc: int
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised Phase-4 SSI, CSI_local, SAI for rows of M (features x classes), all non-zero."""
    tot = M.sum(axis=1)
    frac = (M @ sc_masks.T.astype(np.float64)) / (tot[:, None] + EPS)
    b = frac.argmax(axis=1)
    ssi = np.maximum((frac.max(axis=1) - 1.0 / n_sc) / (1.0 - 1.0 / n_sc + EPS), 0.0)
    p = M / (tot[:, None] + EPS)
    ent = -np.sum(np.where(p > 0, p * np.log(p + EPS), 0.0), axis=1)
    sai = ent / math.log(M.shape[1])
    bm = sc_masks[b]
    bs = bm.sum(axis=1)
    Mw = np.where(bm, M, 0.0)
    wmax, wtot = Mw.max(axis=1), Mw.sum(axis=1)
    raw = np.where(wtot > EPS, wmax / (wtot + EPS), 0.0)
    base = 1.0 / np.maximum(bs, 1)
    csi = np.where((wtot > EPS) & (bs >= 2), np.maximum((raw - base) / (1.0 - base + EPS), 0.0), np.nan)
    return ssi, csi, sai


def run(args) -> str:
    lane_id, lane_dir, dataset, group, name, scratch = args
    t0 = time.time()
    lane = Lane(lane_dir, dataset_id=dataset)
    meta = lane.results["metadata"]
    ati = _lit(meta.get("adaptive_thresholds")) or {}
    thr = _lit(meta.get("analysis_thresholds")) or {}
    labels, layers = lane.labels, lane.layers
    n_cls, n_per = lane.n_classes, lane.n_per_class
    n = n_cls * n_per
    alive_min = max(1, int(np.ceil(n * ALIVE_FRAC)))

    # superclass structure (real), as in Phase 4 (column = sorted class idx)
    sc_names = sorted(lane.superclass_groups.keys())
    real_masks = []
    for sc in sc_names:
        m = np.zeros(n_cls, dtype=bool)
        m[[c for c in lane.superclass_groups[sc] if c < n_cls]] = True
        if m.any():
            real_masks.append(m)
    real_masks = np.stack(real_masks)
    n_sc = real_masks.shape[0]
    masks_ge2 = real_masks[real_masks.sum(axis=1) >= 2]
    sizes_ge2 = masks_ge2.sum(axis=1)

    # RNG structure as the pipeline: class->superclass permutations first, then CSI row perms
    rng = np.random.RandomState(42)
    all_cls = sorted(lane.superclass_map.keys())
    sc_sizes = [len(lane.superclass_groups[sc]) for sc in sc_names]
    perm_masks = []
    for _ in range(N_PERM):
        perm = rng.permutation(all_cls)
        rows, off = [], 0
        for sz in sc_sizes:
            m = np.zeros(n_cls, dtype=bool)
            m[[c for c in perm[off:off + sz] if c < n_cls]] = True
            off += sz
            if m.any():
                rows.append(m)
        perm_masks.append(np.stack(rows).astype(np.float64))

    raw: Dict[str, np.ndarray] = {}
    val = {"ssi_maxabs": 0.0, "csi_maxabs": 0.0, "sai_maxabs": 0.0}
    for lab in labels:
        for L in layers:
            H = lane.load_codes(lab, L)
            assert H.shape[0] == n, (H.shape, n)
            Habs = np.abs(H)
            cnt = (Habs > ACT_EPS).sum(axis=0)
            M_all = Habs.reshape(n_cls, n_per, -1).mean(axis=1).T.astype(np.float64)  # (F, n_cls)
            nz = np.where(M_all.sum(axis=1) > EPS)[0]
            M = M_all[nz]
            X = Habs[:, nz].astype(np.float32)
            ssi, csi, sai = selectivity_from_M(M, real_masks, masks_ge2, n_sc)
            st = lane.results["selectivity"][lab][L]
            s_ssi = np.asarray(st["feature_ssi"], dtype=np.float64)[nz]
            s_csi = np.asarray(st["feature_csi"], dtype=np.float64)[nz]
            s_sai = np.asarray(st["feature_sai"], dtype=np.float64)[nz]
            val["ssi_maxabs"] = max(val["ssi_maxabs"], float(np.max(np.abs(ssi - s_ssi), initial=0)))
            ok = np.isfinite(csi)
            val["csi_maxabs"] = max(val["csi_maxabs"], float(np.max(np.abs(csi[ok] - s_csi[ok]), initial=0)))
            val["sai_maxabs"] = max(val["sai_maxabs"], float(np.max(np.abs(sai - s_sai), initial=0)))

            # SSI null (class -> superclass permutation)
            tot = M.sum(axis=1)
            ssi_null = np.empty((N_PERM, nz.size), dtype=np.float32)
            for p, pm in enumerate(perm_masks):
                k = pm.shape[0]
                fr = (M @ pm.T) / (tot[:, None] + EPS)
                ssi_null[p] = np.clip((fr.max(axis=1) - 1.0 / k) / (1.0 - 1.0 / k + EPS), 0.0, 1.0)

            # CSI null + sample-permutation SAI null (shuffle sample -> class rows)
            csi_null = np.zeros((N_PERM, nz.size), dtype=np.float32)
            csi_keep = np.zeros((N_PERM, nz.size), dtype=bool)
            sai_snull = np.zeros((N_PERM, nz.size), dtype=np.float32)
            logn = math.log(n_cls)
            for p in range(N_PERM):
                perm = rng.permutation(n)
                Mp = X[perm].reshape(n_cls, n_per, -1).mean(axis=1).T  # (f, n_cls)
                tp = Mp.sum(axis=1)
                fr = (Mp @ masks_ge2.T.astype(np.float32)) / (tp[:, None] + EPS)
                b = fr.argmax(axis=1)
                bm, bs = masks_ge2[b], sizes_ge2[b]
                Mw = np.where(bm, Mp, 0.0)
                wmax, wtot = Mw.max(axis=1), Mw.sum(axis=1)
                r = np.where(wtot > EPS, wmax / (wtot + EPS), 0.0)
                base = 1.0 / bs
                v = np.where(tp > EPS, np.maximum((r - base) / (1.0 - base + EPS), 0.0), 0.0)
                csi_null[p] = v
                csi_keep[p] = (v > EPS) & (bs >= MIN_SC_FOR_NULL)
                pr = Mp / (tp[:, None] + EPS)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ent = -np.sum(np.where(pr > 0, pr * np.log(pr + EPS), 0.0), axis=1)
                sai_snull[p] = ent / logn

            key = f"{lab}|{L}"
            raw[f"{key}|nz"] = nz.astype(np.int32)
            raw[f"{key}|cnt"] = cnt[nz].astype(np.int32)
            raw[f"{key}|ssi"] = s_ssi.astype(np.float32)
            raw[f"{key}|csi"] = s_csi.astype(np.float32)
            raw[f"{key}|sai"] = s_sai.astype(np.float32)
            raw[f"{key}|ssi_null"] = ssi_null.astype(np.float16)
            raw[f"{key}|csi_null"] = csi_null.astype(np.float16)
            raw[f"{key}|csi_keep"] = np.packbits(csi_keep, axis=1)
            raw[f"{key}|sai_snull"] = sai_snull.astype(np.float16)
            raw[f"{key}|n_feat"] = np.array([H.shape[1]], dtype=np.int32)
    scratch.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(scratch / f"{lane_id}.npz", **raw)
    meta_out = {"lane_id": lane_id, "name": name, "group": group, "dataset": dataset,
                "lane_dir": str(lane_dir), "labels": labels, "layers": layers, "n_classes": n_cls,
                "n_per_class": n_per, "alive_min": alive_min, "n_perm": N_PERM,
                "stored_thresholds": {k: float(thr[k]) for k in ("ssi_abstraction_thresh",
                                                                 "sai_task_general_thresh",
                                                                 "csi_differentiation_thresh")},
                "stored_ati": {k: v for k, v in ati.items() if isinstance(v, (int, float))},
                "validation_selectivity_maxabs": val, "seconds": time.time() - t0}
    d = OUT / "per_lane"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{lane_id}_meta.json", "w") as f:
        json.dump(meta_out, f, indent=1)
    return f"{name} ({lane_id[:8]}): sel maxabs {val} ({time.time() - t0:.0f}s)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default="")
    ap.add_argument("--scratch", required=True)
    a = ap.parse_args()
    jobs = [(lid, d, ds, g, nm, Path(a.scratch)) for lid, d, ds, g, nm in lanes_to_run()
            if not a.only or lid.startswith(a.only)]
    for j in jobs:
        print(j[3], j[4], j[0], flush=True)
    with Pool(a.workers) as pool:
        for msg in pool.imap_unordered(run, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
