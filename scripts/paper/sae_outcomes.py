#!/usr/bin/env python3
"""SAE measures as outcomes for the curriculum (EC), static-noise (EN) and suppression (SUP) experiments, plus
threshold calibration sets.

Subcommands (run in this order; each reads the previous step's outputs):
  integrity  <exp>   stored SAE reconstructs stored inputs (cosine within 0.01 of stored) and stored
                     codes == CPU re-encoding up to top-k near-ties, every lane/checkpoint/layer.
  thresholds <exp>   alive-only permutation nulls (SSI: class->superclass permutation; CSI: sample->class
                     permutation; as saeanalysis._phase4b_adaptive_thresholds, n_perm = 100), pooled over real
                     checkpoints and all layers of a lane. Common SSI/CSI thresholds per seed = reference lane
                     p95; SAI fixed 0.9; run-specific thresholds stored; delta.
  census     <exp>   SSI/CSI/SAI census (fraction of alive features above threshold), avgpool (+ layer4).
  events     <exp>   Tg-E/Ab-E/Di-E at the duplicate sites (layer4 + avgpool) per real transition:
                     real, same-data null (layer4<->avgpool SAEs at the later checkpoint, both directions)
                     and excess (clipped at 0); common / run-specific / stored thresholds.
  test       <exp>   paired outcomes over seeds: true vs random grouping (EC), across vs within superclasses (EN),
                     superclass vs random suppression (SUP) (test_curriculum, test_noise, test_suppression).
  all        <exp>   all of the above.

Experiments: EC, EN, SUP, PT (ImageNet-pretrained vs scratch); RN, SW (threshold calibration); TESTC, TESTN, TESTH (format tests on the
label-noise lanes, written to SAE_outcomes/_format_test).
Reuses noise_null_lib (Lane, Phase-5 `match`, Phase-6 `classify_counts`) and census_compute (`selectivity_from_M`, null
construction); the CSI null keep rule (CSI > eps, best superclass >= 5 classes) and the integrity check (CPU re-encoding
of the stored inputs, 'intensity_rule_ok' = >= 99.9% of samples with identical active sets) are implemented here.
Read-only on all data; CPU only. Writes under analysis_outputs/paper/SAE_outcomes/.

Usage: PYTHONHASHSEED=0 python sae_outcomes.py <subcommand> <exp> [--workers 3] [--force]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from noise_null_lib import PROCS, REPO, Lane, classify_counts, match  # noqa: E402
from census_compute import selectivity_from_M  # noqa: E402

sys.path.insert(0, str(REPO / "packages" / "ml" / "src"))
from sae import SparseAutoencoder  # noqa: E402

torch.set_num_threads(2)
DEVICE = torch.device("cpu")

OUT = REPO / "analysis_outputs" / "paper" / "SAE_outcomes"
RUNS_EXTENDED = Path(RDT_DATA + "/extended/experiments")
LOCAL_NOISE = REPO / "data" / "experiments" / "0c28799c-ec9d-4e45-ac30-302dff810a8d" / "lanes"
NOISE_REGISTRY = REPO / "analysis_outputs" / "paper" / "INTENSITY" / "lanes_noise_curriculum.csv"
SEEDS = [7, 42, 137, 256, 314]
DUP = ("layer4", "avgpool")
SITE = "avgpool"
UPDATES_PER_EPOCH = 391

# fixed settings
N_PERM = 100
EPS = 1e-10
ACT_EPS = 1e-6
ALIVE_FRAC = 0.01
MIN_SC_FOR_NULL = 5
NULL_PCT = 95.0
SAI_FIXED = 0.9
COS_TOL = 0.01          # reconstruction cosine within 0.01 of stored
VAL_TOL = 1e-3          # code values on samples with identical active sets: |diff| <= VAL_TOL * (1 + |v|)
TIE_TOL = 1e-3          # near-tie: gap between k-th and (k+1)-th pre-activation <= TIE_TOL * (1 + |v_k|)
TSETS = ("common", "run", "stored")
EV = {"Tg": "tg_h", "Ab": "ab_h", "Di": "di_h"}


# Experiment registry

def _noise_registry() -> Dict[tuple, Path]:
    df = pd.read_csv(NOISE_REGISTRY)
    df = df[df.group == "noise"]
    return {(r.condition, int(r.seed)): Path(r.lane_dir) for r in df.itertuples()}


def exp_spec(exp: str) -> Dict[str, Any]:
    """lanes: list of {arm, seed, dir}; ref: reference arm for common thresholds."""
    lanes: List[Dict[str, Any]] = []
    if exp == "EC":
        arms = ["sc_true_K20", "sc_rand_K20", "scratch_E30",
                "sc_true_K10", "sc_rand_K10", "scratch_E40", "sc_true_K30", "sc_rand_K30", "scratch_E20"]
        for a in arms:
            for s in SEEDS:
                lanes.append({"arm": a, "seed": s, "dir": RUNS_EXTENDED / "curriculum" / "lanes" / f"{a}_seed{s}"})
        return {"lanes": lanes, "ref": "scratch_E30", "test": "curriculum_test", "out": OUT,
                "required_arms": ["sc_true_K20", "sc_rand_K20", "scratch_E30"]}
    if exp == "EN":
        m = {"clean": "standard_resnet18", "between": "between_sc_resnet18_throughout",
             "within": "within_sc_resnet18_throughout"}
        for a, nm in m.items():
            for s in SEEDS:
                lanes.append({"arm": a, "seed": s, "dir": RUNS_EXTENDED / f"static_noise_s{s}" / "lanes" / nm})
        return {"lanes": lanes, "ref": "clean", "test": "noise_test", "out": OUT, "required_arms": list(m)}
    if exp == "SUP":
        m = {"coarse": "lesion_scaffold_e00_seed{}_rp65_Llayer4-avgpool",
             "random": "lesion_random_variance_e00_seed{}_rp65_Llayer4-avgpool"}
        for a, nm in m.items():
            for s in SEEDS:
                root = RUNS_EXTENDED / ("prevention_s42_copy" if s == 42 else "prevention") / "lanes"
                lanes.append({"arm": a, "seed": s, "dir": root / nm.format(s)})
        return {"lanes": lanes, "ref": "random", "test": "suppression_test", "out": OUT, "required_arms": list(m)}
    if exp == "PT":      # ImageNet-pretrained vs scratch fine-tuning at lr 0.01
        for a in ("pretrained_ft_lr01", "scratch_lr01"):
            for s in (7, 42, 137):
                lanes.append({"arm": a, "seed": s, "dir": RUNS_EXTENDED / "pretrained" / "lanes" / f"{a}_seed{s}"})
        return {"lanes": lanes, "ref": "scratch_lr01", "test": "pretrained_test", "out": OUT,
                "required_arms": ["pretrained_ft_lr01", "scratch_lr01"]}
    reg = _noise_registry()
    if exp == "RN":      # label noise redrawn every epoch (ED Fig. 8), all 7 conditions; threshold calibration
        for (c, s), d in sorted(reg.items()):
            lanes.append({"arm": c, "seed": s, "dir": d})
        return {"lanes": lanes, "ref": "standard", "test": "calibration", "out": OUT,
                "required_arms": sorted({c for c, _ in reg})}
    if exp == "SW":      # curriculum sweep, one lane per condition (ED Fig. 9a); threshold calibration
        df = pd.read_csv(NOISE_REGISTRY)
        for r in df[df.group == "curriculum"].itertuples():
            lanes.append({"arm": r.condition, "seed": int(r.seed), "dir": Path(r.lane_dir)})
        return {"lanes": lanes, "ref": "standard50", "test": "calibration", "out": OUT,
                "required_arms": sorted(df[df.group == "curriculum"].condition)}
    ft = OUT / "_format_test"
    if exp == "TESTN":   # format test of the static-noise test path (within = p01 so that seed 137 exercises exclusion)
        m = {"clean": "standard", "between": "between_sc_p03", "within": "within_sc_p01"}
        for a, c in m.items():
            for s in SEEDS:
                lanes.append({"arm": a, "seed": s, "dir": reg[(c, s)]})
        return {"lanes": lanes, "ref": "clean", "test": "noise_test", "out": ft, "required_arms": list(m)}
    if exp == "TESTC":   # format test of the curriculum test path (arm names mapped onto label-noise lanes)
        m = {"sc_true_K20": "between_sc_p03", "sc_rand_K20": "random_p03", "scratch_E30": "standard"}
        for a, c in m.items():
            for s in SEEDS:
                lanes.append({"arm": a, "seed": s, "dir": reg[(c, s)]})
        return {"lanes": lanes, "ref": "scratch_E30", "test": "curriculum_test", "out": ft,
                "required_arms": list(m)}
    if exp == "TESTH":   # format test of the suppression test path (coarse = between p03, random = standard)
        m = {"coarse": "between_sc_p03", "random": "standard"}
        for a, c in m.items():
            for s in SEEDS:
                lanes.append({"arm": a, "seed": s, "dir": reg[(c, s)]})
        return {"lanes": lanes, "ref": "random", "test": "suppression_test", "out": ft, "required_arms": list(m),
                "window_end": 10}
    raise SystemExit(f"unknown experiment {exp}")


def lane_key(ln: Dict[str, Any]) -> str:
    return f"{ln['arm']}__s{ln['seed']}"


def is_ready(d: Path) -> bool:
    if (d / ".sae_done").exists():
        return True
    res, prog = d / "sae_analysis" / "sae_results.json", d / "sae_analysis" / "saeanalysis_progress.json"
    if res.exists() and prog.exists():
        try:
            return int(json.loads(prog.read_text()).get("progress_pct", 0)) >= 100
        except Exception:
            return False
    return False


def ready_lanes(spec) -> List[Dict[str, Any]]:
    return [ln for ln in spec["lanes"] if is_ready(Path(ln["dir"]))]


def checkpoint_info(lane_dir: Path, labels: List[str]) -> List[Dict[str, Any]]:
    """epoch = weight_updates / 391 (robust to the t0 'epoch' field); the last label is flagged as the
    duplicate final checkpoint if it is a 'final' snapshot with the same weight_updates as its predecessor."""
    snaps = {int(s["milestone_index"]): s for s in
             json.load(open(lane_dir / "dev_snapshots" / "metrics" / "devtrain_snapshots.json"))}
    out = []
    for i, lab in enumerate(labels):
        s = snaps[int(lab)]
        upd = int(s["weight_updates"])
        ep = upd / UPDATES_PER_EPOCH
        ep = int(round(ep)) if abs(ep - round(ep)) < 1e-6 else round(ep, 3)
        dup = (i == len(labels) - 1 and i > 0 and s.get("snapshot_type") == "final"
               and upd == int(snaps[int(labels[i - 1])]["weight_updates"]))
        out.append({"label": lab, "epoch": ep, "updates": upd, "snapshot_type": s.get("snapshot_type"),
                    "val_accuracy": s.get("val_accuracy"), "dup_final": bool(dup)})
    return out


def real_ckpts(ci: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [c for c in ci if not c["dup_final"]]


def _lit(x):
    import ast
    return ast.literal_eval(x) if isinstance(x, str) else x


def sc_masks(lane: Lane):
    n_cls = lane.n_classes
    sc_names = sorted(lane.superclass_groups.keys())
    rows = []
    for sc in sc_names:
        m = np.zeros(n_cls, dtype=bool)
        m[[c for c in lane.superclass_groups[sc] if c < n_cls]] = True
        if m.any():
            rows.append(m)
    real = np.stack(rows)
    ge2 = real[real.sum(axis=1) >= 2]
    return sc_names, real, ge2


def run_pool(fn, jobs, workers):
    out = []
    if workers <= 1:
        for j in jobs:
            out.append(fn(j))
            print(out[-1].get("msg", ""), flush=True)
        return out
    import multiprocessing as mp
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        for r in ex.map(fn, jobs):
            out.append(r)
            print(r.get("msg", ""), flush=True)
    return out


def cache_dir(spec, exp) -> Path:
    d = spec["out"] / "cache" / exp
    d.mkdir(parents=True, exist_ok=True)
    return d


# 1. integrity

def _integrity_lane(job) -> Dict[str, Any]:
    ln = job["lane"]
    d = Path(ln["dir"])
    t0 = time.time()
    try:
        lane = Lane(d)
        k = int(lane.results["metadata"]["k_sparse"])
        rq = lane.results.get("reconstruction_quality") or {}
        ci = checkpoint_info(d, lane.labels)
        _, real_m, ge2 = sc_masks(lane)
        n_sc = real_m.shape[0]
        rows = []
        for c in ci:
            lab = c["label"]
            acts = torch.load(d / "dev_snapshots" / f"milestone_{lab}" / "individual_multilayer_activations.pt",
                              map_location="cpu", weights_only=False)
            for L in lane.layers:
                row = {"arm": ln["arm"], "seed": ln["seed"], "lane_dir": str(d), "checkpoint": lab,
                       "epoch": c["epoch"], "dup_final": c["dup_final"], "layer": L}
                dl = acts[L]
                keys = sorted(dl.keys())
                assert keys == list(range(lane.n_classes)), "class keys"
                X = torch.cat([dl[ci_].float().reshape(dl[ci_].shape[0], -1) for ci_ in keys])
                sd = torch.load(d / "sae_analysis" / "saes" / lab / f"{L}.pt", map_location="cpu")
                d_in, d_hid = sd["encoder.weight"].shape[1], sd["encoder.weight"].shape[0]
                sae = SparseAutoencoder(d_in, d_hid // d_in, k_sparse=k)
                sae.load_state_dict(sd)
                sae.eval()
                with torch.no_grad():
                    pre = sae.encoder(X)
                    H_own = sae.encode(X)
                    cos = float(torch.nn.functional.cosine_similarity(X, sae.decode(H_own), dim=1).mean())
                H_own_np = H_own.numpy()
                H_st = np.load(lane.act_path(lab, L))
                row["own_recon_cos"] = cos
                row["stored_recon_cos"] = float((rq.get(lab, {}).get(L, {}) or {}).get("cosine_sim", np.nan))
                row["cos_ok"] = bool(np.isfinite(row["stored_recon_cos"])
                                     and abs(cos - row["stored_recon_cos"]) <= COS_TOL)
                same_shape = H_st.shape == H_own_np.shape
                row["shape_ok"] = bool(same_shape)
                if same_shape:
                    same = ((H_st != 0) == (H_own_np != 0)).all(axis=1)
                    row["frac_same_active_set"] = float(same.mean())
                    row["n_mismatch_samples"] = int((~same).sum())
                    if same.any():
                        a, b = H_st[same], H_own_np[same]
                        rel = np.abs(a - b) / (1.0 + np.abs(b))
                        row["max_rel_diff_same_set"] = float(rel.max())
                    else:
                        row["max_rel_diff_same_set"] = float("nan")
                    mm = np.where(~same)[0]
                    if mm.size:
                        top = torch.topk(pre[torch.as_tensor(mm)], k + 1, dim=1).values.numpy()
                        gap = (top[:, k - 1] - top[:, k]) / (1.0 + np.abs(top[:, k - 1]))
                        row["max_tie_gap_mismatch"] = float(gap.max())
                        row["n_mismatch_not_near_tie"] = int((gap > TIE_TOL).sum())
                    else:
                        row["max_tie_gap_mismatch"] = 0.0
                        row["n_mismatch_not_near_tie"] = 0
                    row["codes_ok"] = bool(row["n_mismatch_not_near_tie"] == 0
                                           and (not np.isfinite(row["max_rel_diff_same_set"])
                                                or row["max_rel_diff_same_set"] <= VAL_TOL))
                    row["intensity_rule_ok"] = bool(row["frac_same_active_set"] >= 0.999)
                    # stored Phase-4 selectivity vs recomputation from stored codes (diagnostic)
                    Habs = np.abs(H_st)
                    M_all = Habs.reshape(lane.n_classes, lane.n_per_class, -1).mean(axis=1).T.astype(np.float64)
                    nz = np.where(M_all.sum(axis=1) > EPS)[0]
                    ssi, csi, sai = selectivity_from_M(M_all[nz], real_m, ge2, n_sc)
                    st = lane.results["selectivity"][lab][L]
                    dmax = 0.0
                    for v, key in ((ssi, "feature_ssi"), (csi, "feature_csi"), (sai, "feature_sai")):
                        s_ = np.asarray(st[key], dtype=np.float64)[nz]
                        ok = np.isfinite(v)
                        dmax = max(dmax, float(np.max(np.abs(v[ok] - s_[ok]), initial=0.0)))
                    row["sel_recompute_maxabs"] = dmax
                else:
                    row["codes_ok"] = False
                    row["intensity_rule_ok"] = False
                row["pass"] = bool(row["cos_ok"] and row["codes_ok"])
                rows.append(row)
            del acts
        n_fail = sum(not r["pass"] for r in rows)
        return {"rows": rows, "msg": f"[integrity] {lane_key(ln)}: {len(rows)} ckpt-layers, {n_fail} fail "
                                     f"({time.time() - t0:.0f}s)"}
    except Exception:
        return {"rows": [{"arm": ln["arm"], "seed": ln["seed"], "lane_dir": str(d), "pass": False,
                          "error": traceback.format_exc()[-500:]}],
                "msg": f"[integrity] {lane_key(ln)}: ERROR {traceback.format_exc()[-300:]}"}


def cmd_integrity(exp, spec, workers):
    lanes = ready_lanes(spec)
    print(f"{exp}: {len(lanes)} ready lanes of {len(spec['lanes'])}", flush=True)
    res = run_pool(_integrity_lane, [{"lane": ln} for ln in lanes], workers)
    rows = [r for x in res for r in x["rows"]]
    df = pd.DataFrame(rows)
    spec["out"].mkdir(parents=True, exist_ok=True)
    df.to_csv(spec["out"] / f"integrity_{exp}.csv", index=False)
    lane_pass = df.groupby(["arm", "seed"])["pass"].all().reset_index()
    failing = lane_pass[~lane_pass["pass"]]
    summ = {"experiment": exp, "n_lanes_checked": int(len(lane_pass)),
            "n_ckpt_layers": int(len(df)), "n_ckpt_layer_fail": int((~df["pass"]).sum()),
            "excluded_lanes": [f"{r.arm}__s{r.seed}" for r in failing.itertuples()],
            "not_ready_lanes": [lane_key(ln) for ln in spec["lanes"] if ln not in lanes],
            "tolerances": {"cos": COS_TOL, "val_rel": VAL_TOL, "tie_rel": TIE_TOL}}
    if "own_recon_cos" in df:
        ok = df[df["pass"]]
        summ.update({
            "max_abs_cos_diff_pass": float((ok.own_recon_cos - ok.stored_recon_cos).abs().max()) if len(ok) else None,
            "min_frac_same_active_set_pass": float(ok.frac_same_active_set.min()) if len(ok) else None,
            "max_rel_diff_same_set_pass": float(ok.max_rel_diff_same_set.max()) if len(ok) else None,
            "total_mismatch_samples_pass": int(ok.n_mismatch_samples.sum()) if len(ok) else None,
            "max_tie_gap_mismatch_pass": float(ok.max_tie_gap_mismatch.max()) if len(ok) else None,
            "max_sel_recompute_maxabs": float(df.sel_recompute_maxabs.max()) if "sel_recompute_maxabs" in df else None,
            "fail_rows": df[~df["pass"]][[c for c in ("arm", "seed", "checkpoint", "layer", "own_recon_cos",
                                                       "stored_recon_cos", "frac_same_active_set",
                                                       "n_mismatch_not_near_tie", "error") if c in df]]
            .to_dict("records")})
    json.dump(summ, open(spec["out"] / f"integrity_{exp}_summary.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in summ.items() if k != "fail_rows"}, indent=1, default=str))


def passing_lanes(exp, spec) -> List[Dict[str, Any]]:
    f = spec["out"] / f"integrity_{exp}.csv"
    if not f.exists():
        raise SystemExit(f"run 'integrity {exp}' first")
    df = pd.read_csv(f)
    lp = df.groupby(["arm", "seed"])["pass"].all()
    return [ln for ln in spec["lanes"] if lp.get((ln["arm"], ln["seed"]), False)]


# 2. permutation nulls (alive-only) + per-lane cache of selectivity / alive counts

def _pstats(v: np.ndarray) -> Dict[str, float]:
    v = v[np.isfinite(v)]
    if not v.size:
        return {"p95": float("nan"), "mean": float("nan"), "std": float("nan"), "n": 0}
    return {"p95": float(np.percentile(v, NULL_PCT)), "mean": float(v.mean()), "std": float(v.std()),
            "n": int(v.size)}


def _null_lane(job) -> Dict[str, Any]:
    ln, cdir = job["lane"], Path(job["cache"])
    d = Path(ln["dir"])
    key = lane_key(ln)
    fjson, fnpz = cdir / f"{key}_null.json", cdir / f"{key}_sel.npz"
    if fjson.exists() and fnpz.exists() and not job.get("force"):
        return {"msg": f"[null] {key}: cached"}
    t0 = time.time()
    lane = Lane(d)
    ci = real_ckpts(checkpoint_info(d, lane.labels))
    n_cls, n_per = lane.n_classes, lane.n_per_class
    n = n_cls * n_per
    alive_min = max(1, int(np.ceil(n * ALIVE_FRAC)))
    sc_names, real_m, ge2 = sc_masks(lane)
    sizes_ge2 = ge2.sum(axis=1)
    # RNG structure as the pipeline: class->superclass permutations first, then CSI row permutations
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

    pooled = {"ssi": [], "csi": [], "sai_real": []}
    by_layer: Dict[str, Dict[str, list]] = {L: {"ssi": [], "csi": []} for L in lane.layers}
    per_ck: Dict[str, Any] = {}
    raw: Dict[str, np.ndarray] = {}
    sel_all = lane.results["selectivity"]
    for c in ci:
        lab = c["label"]
        per_ck[lab] = {}
        for L in lane.layers:
            H = lane.load_codes(lab, L)
            assert H.shape[0] == n
            Habs = np.abs(H)
            cnt = (Habs > ACT_EPS).sum(axis=0)
            alive = cnt >= alive_min
            M_all = Habs.reshape(n_cls, n_per, -1).mean(axis=1).T.astype(np.float64)
            tot = M_all.sum(axis=1)
            idx = np.where(alive & (tot > EPS))[0]
            M = M_all[idx]
            t_ = tot[idx]
            # SSI null (pipeline: positive values of active features), alive only
            ssi_v = []
            for pm in perm_masks:
                kk = pm.shape[0]
                fr = (M @ pm.T) / (t_[:, None] + EPS)
                v = np.clip((fr.max(axis=1) - 1.0 / kk) / (1.0 - 1.0 / kk + EPS), 0.0, 1.0)
                ssi_v.append(v[v > 0])
            ssi_v = np.concatenate(ssi_v) if ssi_v else np.zeros(0)
            # CSI null (sample->class permutation, CSI_local; keep > eps and best SC >= 5 classes), alive only
            X = Habs[:, idx].astype(np.float32)
            csi_v = []
            for _ in range(N_PERM):
                perm = rng.permutation(n)
                if not idx.size:
                    continue
                Mp = X[perm].reshape(n_cls, n_per, -1).mean(axis=1).T
                tp = Mp.sum(axis=1)
                fr = (Mp @ ge2.T.astype(np.float32)) / (tp[:, None] + EPS)
                b = fr.argmax(axis=1)
                bm, bs = ge2[b], sizes_ge2[b]
                Mw = np.where(bm, Mp, 0.0)
                wmax, wtot = Mw.max(axis=1), Mw.sum(axis=1)
                r = np.where(wtot > EPS, wmax / (wtot + EPS), 0.0)
                base = 1.0 / bs
                v = np.where(tp > EPS, np.maximum((r - base) / (1.0 - base + EPS), 0.0), 0.0)
                keep = (v > EPS) & (bs >= MIN_SC_FOR_NULL)
                csi_v.append(v[keep])
            csi_v = np.concatenate(csi_v) if csi_v else np.zeros(0)
            st = sel_all[lab][L]
            sai = np.asarray(st["feature_sai"], dtype=np.float64)
            sai_real = sai[idx]
            sai_real = sai_real[sai_real > 0]
            pooled["ssi"].append(ssi_v)
            pooled["csi"].append(csi_v)
            pooled["sai_real"].append(sai_real)
            by_layer[L]["ssi"].append(ssi_v)
            by_layer[L]["csi"].append(csi_v)
            per_ck[lab][L] = {"ssi_p95": _pstats(ssi_v)["p95"], "csi_p95": _pstats(csi_v)["p95"],
                              "n_alive": int(alive.sum())}
            pre = f"{lab}|{L}"
            raw[f"{pre}|cnt"] = cnt.astype(np.int32)
            for kname in ("feature_ssi", "feature_csi", "feature_sai"):
                raw[f"{pre}|{kname}"] = np.asarray(st[kname], dtype=np.float32)
    ati = _lit(lane.results["metadata"].get("adaptive_thresholds")) or {}
    thr = _lit(lane.results["metadata"].get("analysis_thresholds")) or {}
    out = {"lane": key, "arm": ln["arm"], "seed": ln["seed"], "lane_dir": str(d), "labels_real": [c["label"] for c in ci],
           "ckpts": ci, "layers": lane.layers, "n_perm": N_PERM, "alive_min": alive_min,
           "pooled": {k: _pstats(np.concatenate(v) if v else np.zeros(0)) for k, v in pooled.items()},
           "by_layer": {L: {k: _pstats(np.concatenate(v) if v else np.zeros(0)) for k, v in dd.items()}
                        for L, dd in by_layer.items()},
           "per_ckpt": per_ck,
           "stored": {"thresholds": {k: float(thr[k]) for k in ("ssi_abstraction_thresh", "sai_task_general_thresh",
                                                               "csi_differentiation_thresh") if k in thr},
                      "ati": {k: float(v) for k, v in ati.items() if isinstance(v, (int, float))}},
           "seconds": time.time() - t0}
    np.savez_compressed(fnpz, **raw)
    json.dump(out, open(fjson, "w"), default=str)
    return {"msg": f"[null] {key}: SSI p95 {out['pooled']['ssi']['p95']:.4f} CSI p95 {out['pooled']['csi']['p95']:.4f} "
                   f"({time.time() - t0:.0f}s)"}


def tset_from_null(nl: Dict[str, Any]) -> Dict[str, Any]:
    """Threshold set (Phase-6 format) from an alive-only null: SSI/CSI = p95; SAI = fixed 0.9.
    z-scale for stable-pair increases: null mean/std (SSI, CSI); SAI: mean/std of real positive SAI of
    alive features (the pipeline's SAI 'null' equals the real SAI distribution)."""
    p = nl["pooled"]
    return {"thresholds": {"ssi_abstraction_thresh": p["ssi"]["p95"], "csi_differentiation_thresh": p["csi"]["p95"],
                           "sai_task_general_thresh": SAI_FIXED},
            "ati": {"ssi_null_mean": p["ssi"]["mean"], "ssi_null_std": p["ssi"]["std"],
                    "csi_null_mean": p["csi"]["mean"], "csi_null_std": p["csi"]["std"],
                    "sai_null_mean": p["sai_real"]["mean"], "sai_null_std": p["sai_real"]["std"]}}


def load_null(cdir: Path, key: str) -> Dict[str, Any]:
    return json.load(open(cdir / f"{key}_null.json"))


def census_one(z, lab: str, L: str, thr: Dict[str, float], alive_min: int) -> Dict[str, float]:
    cnt = z[f"{lab}|{L}|cnt"]
    alive = cnt >= alive_min
    ssi = z[f"{lab}|{L}|feature_ssi"].astype(np.float64)
    csi = z[f"{lab}|{L}|feature_csi"].astype(np.float64)
    sai = z[f"{lab}|{L}|feature_sai"].astype(np.float64)
    na = int(alive.sum())
    n_ssi = int((alive & (ssi > thr["ssi_abstraction_thresh"])).sum())
    n_csi = int((alive & (csi > thr["csi_differentiation_thresh"])).sum())
    n_sai = int((alive & (sai > thr["sai_task_general_thresh"])).sum())
    f = (lambda x: x / na if na else float("nan"))
    return {"n_alive": na, "n_ssi": n_ssi, "n_csi": n_csi, "n_sai": n_sai,
            "ssi": f(n_ssi), "csi": f(n_csi), "sai": f(n_sai)}


def cmd_thresholds(exp, spec, workers, force=False):
    lanes = passing_lanes(exp, spec)
    cdir = cache_dir(spec, exp)
    run_pool(_null_lane, [{"lane": ln, "cache": str(cdir), "force": force} for ln in lanes], workers)
    rows, common, run, stored = [], {}, {}, {}
    for ln in lanes:
        nl = load_null(cdir, lane_key(ln))
        run[lane_key(ln)] = tset_from_null(nl)
        stored[lane_key(ln)] = nl["stored"]
        if ln["arm"] == spec["ref"]:
            common[str(ln["seed"])] = tset_from_null(nl)
        rows.append({"lane": lane_key(ln), "arm": ln["arm"], "seed": ln["seed"], "is_reference": ln["arm"] == spec["ref"],
                     "ssi_p95_alive": nl["pooled"]["ssi"]["p95"], "csi_p95_alive": nl["pooled"]["csi"]["p95"],
                     "ssi_null_mean": nl["pooled"]["ssi"]["mean"], "ssi_null_std": nl["pooled"]["ssi"]["std"],
                     "csi_null_mean": nl["pooled"]["csi"]["mean"], "csi_null_std": nl["pooled"]["csi"]["std"],
                     "n_ssi_null": nl["pooled"]["ssi"]["n"], "n_csi_null": nl["pooled"]["csi"]["n"],
                     **{f"ssi_p95_{L}": nl["by_layer"][L]["ssi"]["p95"] for L in nl["layers"]},
                     **{f"csi_p95_{L}": nl["by_layer"][L]["csi"]["p95"] for L in nl["layers"]},
                     "stored_ssi_thresh": nl["stored"]["thresholds"].get("ssi_abstraction_thresh"),
                     "stored_csi_thresh": nl["stored"]["thresholds"].get("csi_differentiation_thresh"),
                     "stored_sai_thresh": nl["stored"]["thresholds"].get("sai_task_general_thresh")})
    # delta: p95 over seeds x real checkpoints of reference lanes of |census(avgpool) - census(layer4)|
    drows = []
    for ln in lanes:
        if ln["arm"] != spec["ref"]:
            continue
        nl = load_null(cdir, lane_key(ln))
        z = np.load(cdir / f"{lane_key(ln)}_sel.npz")
        thr = common[str(ln["seed"])]["thresholds"]
        for c in nl["ckpts"]:
            a = census_one(z, c["label"], "avgpool", thr, nl["alive_min"])
            b = census_one(z, c["label"], "layer4", thr, nl["alive_min"])
            drows.append({"seed": ln["seed"], "checkpoint": c["label"], "epoch": c["epoch"],
                          **{f"{m}_avgpool": a[m] for m in ("ssi", "csi", "sai")},
                          **{f"{m}_layer4": b[m] for m in ("ssi", "csi", "sai")},
                          **{f"abs_diff_{m}": abs(a[m] - b[m]) for m in ("ssi", "csi", "sai")}})
    dd = pd.DataFrame(drows)
    delta = {m: float(np.percentile(dd[f"abs_diff_{m}"].dropna(), 95)) if len(dd) else float("nan")
             for m in ("ssi", "csi", "sai")}
    out = {"experiment": exp, "reference_arm": spec["ref"], "n_perm": N_PERM, "percentile": NULL_PCT,
           "sai_fixed": SAI_FIXED, "pooling": "all real checkpoints (duplicate final dropped) x all layers; "
                                             "alive features (>= 1% of samples) only",
           "common": common, "run": run, "stored": stored, "delta": delta,
           "delta_n_values": int(len(dd)),
           "seeds_without_reference": [s for s in SEEDS if str(s) not in common]}
    json.dump(out, open(spec["out"] / f"thresholds_{exp}.json", "w"), indent=1)
    pd.DataFrame(rows).to_csv(spec["out"] / f"thresholds_{exp}.csv", index=False)
    dd.to_csv(spec["out"] / f"delta_{exp}.csv", index=False)
    print(json.dumps({"common": {s: v["thresholds"] for s, v in common.items()}, "delta": delta}, indent=1))


def load_thresholds(exp, spec) -> Dict[str, Any]:
    f = spec["out"] / f"thresholds_{exp}.json"
    if not f.exists():
        raise SystemExit(f"run 'thresholds {exp}' first")
    return json.load(open(f))


def tset_for(T, tset: str, ln) -> Optional[Dict[str, Any]]:
    if tset == "common":
        return T["common"].get(str(ln["seed"]))
    if tset == "run":
        return T["run"].get(lane_key(ln))
    st = T["stored"].get(lane_key(ln))
    return {"thresholds": st["thresholds"], "ati": st["ati"]} if st else None


# 3. census

def cmd_census(exp, spec):
    T = load_thresholds(exp, spec)
    cdir = cache_dir(spec, exp)
    rows = []
    for ln in passing_lanes(exp, spec):
        key = lane_key(ln)
        if not (cdir / f"{key}_sel.npz").exists():
            continue
        nl = load_null(cdir, key)
        z = np.load(cdir / f"{key}_sel.npz")
        for ts in TSETS:
            t = tset_for(T, ts, ln)
            if t is None:
                continue
            for c in nl["ckpts"]:
                for L in ("avgpool", "layer4"):
                    r = census_one(z, c["label"], L, t["thresholds"], nl["alive_min"])
                    rows.append({"lane": key, "arm": ln["arm"], "seed": ln["seed"], "tset": ts, "layer": L,
                                 "checkpoint": c["label"], "epoch": c["epoch"], **r,
                                 "thr_ssi": t["thresholds"]["ssi_abstraction_thresh"],
                                 "thr_csi": t["thresholds"]["csi_differentiation_thresh"],
                                 "thr_sai": t["thresholds"]["sai_task_general_thresh"]})
    pd.DataFrame(rows).to_csv(spec["out"] / f"census_{exp}.csv", index=False)
    print(f"census_{exp}.csv: {len(rows)} rows")


# 4. events

def _md_slim(md):
    return {k: md[k] for k in ("stable", "born", "died", "alive_a", "alive_b", "n_stable", "n_born", "n_died")}


def _events_lane(job) -> Dict[str, Any]:
    ln, cdir, T = job["lane"], Path(job["cache"]), job["T"]
    key = lane_key(ln)
    d = Path(ln["dir"])
    t0 = time.time()
    try:
        lane = Lane(d)
        nl = load_null(cdir, key)
        ci = nl["ckpts"]
        labs = [c["label"] for c in ci]
        ep = {c["label"]: c["epoch"] for c in ci}
        sel = lane.results["selectivity"]
        fm = cdir / f"{key}_match.pkl"
        mcache = pickle.load(open(fm, "rb")) if fm.exists() else {}
        codes = {lab: {L: lane.load_codes(lab, L) for L in DUP} for lab in labs}
        extra_pairs = job.get("extra_pairs", [])
        need = []
        for lab in labs:
            need += [("null", lab, "layer4->avgpool"), ("null", lab, "avgpool->layer4")]
        for a, b in zip(labs[:-1], labs[1:]):
            need += [("real", f"{a}->{b}", L) for L in DUP]
        for a, b, L in extra_pairs:
            if a in codes and b in codes:
                need.append(("real", f"{a}->{b}", L))
        for nk in need:
            if nk in mcache:
                continue
            kind, x, y = nk
            if kind == "null":
                A, B = y.split("->")
                mcache[nk] = _md_slim(match(codes[x][A], codes[x][B]))
            else:
                a, b = x.split("->")
                mcache[nk] = _md_slim(match(codes[a][y], codes[b][y]))
        pickle.dump(mcache, open(fm, "wb"))
        rows = []
        tsets = {ts: tset_for(T, ts, ln) for ts in TSETS}

        def cc(ts, Ha, Hb, md, sa_, sb_):
            t = tsets[ts]
            return classify_counts(Ha, Hb, md, sa_, sb_, thresholds=t["thresholds"], ati=t["ati"])
        null = {}
        for lab in labs:
            for ts, t in tsets.items():
                if t is None:
                    continue
                n1 = cc(ts, codes[lab]["layer4"], codes[lab]["avgpool"], mcache[("null", lab, "layer4->avgpool")],
                        sel[lab]["layer4"], sel[lab]["avgpool"])
                n2 = cc(ts, codes[lab]["avgpool"], codes[lab]["layer4"], mcache[("null", lab, "avgpool->layer4")],
                        sel[lab]["avgpool"], sel[lab]["layer4"])
                null[(lab, ts)] = (n1, n2)
        for a, b in zip(labs[:-1], labs[1:]):
            k = f"{a}->{b}"
            for ts, t in tsets.items():
                if t is None:
                    continue
                R = {L: cc(ts, codes[a][L], codes[b][L], mcache[("real", k, L)], sel[a][L], sel[b][L]) for L in DUP}
                nb1, nb2 = null[(b, ts)]
                na1, na2 = null[(a, ts)]
                row = {"lane": key, "arm": ln["arm"], "seed": ln["seed"], "tset": ts, "transition": k,
                       "epoch_a": ep[a], "epoch_b": ep[b]}
                for p in PROCS:
                    real = R["layer4"][p] + R["avgpool"][p]
                    nul = nb1[p] + nb2[p]                       # both directions at the later checkpoint
                    nul_ab = 0.5 * (na1[p] + na2[p]) + 0.5 * (nb1[p] + nb2[p])  # averaged over both checkpoints (as in noise_null_analyze)
                    row.update({f"{p}_real_layer4": R["layer4"][p], f"{p}_real_avgpool": R["avgpool"][p],
                                f"{p}_real": real, f"{p}_null_l4ap": nb1[p], f"{p}_null_apl4": nb2[p],
                                f"{p}_null": nul, f"{p}_excess": real - nul, f"{p}_excess_clip": max(real - nul, 0),
                                f"{p}_null_ab": nul_ab, f"{p}_excess_clip_ab": max(real - nul_ab, 0.0)})
                rows.append(row)
        # extra pairs for the suppression test: features at epoch 30 matched to features before the window
        extra = []
        for a, b, L in extra_pairs:
            if a not in codes or b not in codes:
                continue
            md = mcache[("real", f"{a}->{b}", L)]
            for ts, t in tsets.items():
                if t is None:
                    continue
                cnt = (np.abs(codes[b][L]) > ACT_EPS).sum(axis=0)
                alive = cnt >= nl["alive_min"]
                ssi = np.asarray(sel[b][L]["feature_ssi"], dtype=np.float64)
                ssi_feats = set(np.where(alive & (ssi > t["thresholds"]["ssi_abstraction_thresh"]))[0].tolist())
                matched_b = {int(s[1]) for s in md["stable"]}
                n_m = len(ssi_feats & matched_b)
                extra.append({"lane": key, "arm": ln["arm"], "seed": ln["seed"], "tset": ts, "pair": f"{a}->{b}",
                              "layer": L, "n_ssi_b": len(ssi_feats), "n_ssi_b_matched_r05": n_m,
                              "frac_matched": n_m / len(ssi_feats) if ssi_feats else float("nan")})
        return {"rows": rows, "extra": extra, "msg": f"[events] {key}: {len(rows)} rows ({time.time() - t0:.0f}s)"}
    except Exception:
        return {"rows": [], "extra": [], "msg": f"[events] {key}: ERROR {traceback.format_exc()[-800:]}"}


def cmd_events(exp, spec, workers):
    T = load_thresholds(exp, spec)
    cdir = cache_dir(spec, exp)
    jobs = []
    for ln in passing_lanes(exp, spec):
        if not (cdir / f"{lane_key(ln)}_null.json").exists():
            continue
        extra = []
        if spec["test"] == "suppression_test":
            nl = load_null(cdir, lane_key(ln))
            labs = [c["label"] for c in nl["ckpts"]]
            extra = [(labs[0], labs[-1], SITE)]   # before the window (milestone 0) -> last real checkpoint
        jobs.append({"lane": ln, "cache": str(cdir), "T": T, "extra_pairs": extra})
    res = run_pool(_events_lane, jobs, workers)
    pd.DataFrame([r for x in res for r in x["rows"]]).to_csv(spec["out"] / f"events_{exp}.csv", index=False)
    ex = [r for x in res for r in x["extra"]]
    if ex:
        pd.DataFrame(ex).to_csv(spec["out"] / f"matched_prewindow_{exp}.csv", index=False)
    print(f"events_{exp}.csv written")


# 5. tests

def paired(vals: Dict[int, float], name: str = "") -> Dict[str, Any]:
    seeds = sorted(s for s, v in vals.items() if v is not None and np.isfinite(v))
    d = np.array([vals[s] for s in seeds], dtype=np.float64)
    n = len(d)
    out: Dict[str, Any] = {"name": name, "n": n, "per_seed": {str(s): float(vals[s]) for s in seeds}}
    if n == 0:
        return out
    m = float(d.mean())
    out.update({"mean": m, "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()), "n_zero": int((d == 0).sum())})
    if n >= 2:
        sd = float(d.std(ddof=1))
        se = sd / math.sqrt(n)
        tq = float(stats.t.ppf(0.975, n - 1))
        out.update({"sd": sd, "se": se, "ci95": [m - tq * se, m + tq * se], "df": n - 1})
        if se > 0:
            tt = m / se
            out.update({"t": tt, "p_two_sided": float(2 * stats.t.sf(abs(tt), n - 1))})
        else:
            out.update({"t": float("nan"), "p_two_sided": float("nan")})
    return out


def verdict_dir(r: Dict[str, Any], sign: int, min_count: int = 4) -> str:
    if r.get("n", 0) < 2 or "ci95" not in r:
        return "NOT SUPPORTED (insufficient seeds)"
    lo, hi = r["ci95"]
    ok_mean = r["mean"] * sign > 0
    ok_ci = lo > 0 if sign > 0 else hi < 0
    ok_cnt = (r["n_pos"] if sign > 0 else r["n_neg"]) >= min_count
    v = "SUPPORTED" if (ok_mean and ok_ci and ok_cnt) else "NOT SUPPORTED"
    r["criteria"] = {"mean_direction": ok_mean, "ci_excludes_0": ok_ci, f"count_ge_{min_count}": ok_cnt}
    return v


def _census_val(C, arm, seed, epoch, metric="ssi", tset="common", layer=SITE):
    q = C[(C.arm == arm) & (C.seed == seed) & (C.tset == tset) & (C.layer == layer) & (C.epoch == epoch)]
    return float(q[metric].iloc[0]) if len(q) == 1 else None


def _diff(f, a1, a2, seeds):
    out = {}
    for s in seeds:
        x, y = f(a1, s), f(a2, s)
        out[s] = None if x is None or y is None else x - y
    return out


def _events_sum(E, arm, seed, tset, proc, window, col="excess_clip"):
    q = E[(E.arm == arm) & (E.seed == seed) & (E.tset == tset)]
    if not len(q):
        return None
    lo, hi = window
    q = q[(q.epoch_a >= lo) & (q.epoch_b <= hi)]
    return float(q[f"{proc}_{col}"].sum())


def _per_epoch_table(C, arms, metric, tset):
    rows = []
    for arm in arms:
        q = C[(C.arm == arm) & (C.tset == tset) & (C.layer == SITE)]
        for e, g in q.groupby("epoch"):
            v = g[metric].values
            n = len(v)
            se = v.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
            tq = stats.t.ppf(0.975, n - 1) if n > 1 else float("nan")
            rows.append({"arm": arm, "epoch": e, "metric": metric, "tset": tset, "n": n, "mean": float(v.mean()),
                         "ci_lo": float(v.mean() - tq * se), "ci_hi": float(v.mean() + tq * se),
                         **{f"s{s}": float(x) for s, x in zip(g.seed, v)}})
    return rows


def _paired_by_epoch(C, a1, a2, metric, tset, seeds):
    rows = []
    eps = sorted(set(C[(C.arm == a1)].epoch) & set(C[(C.arm == a2)].epoch))
    for e in eps:
        r = paired(_diff(lambda a, s: _census_val(C, a, s, e, metric, tset), a1, a2, seeds), f"{a1}-{a2}@{e}")
        rows.append({"contrast": f"{a1} - {a2}", "epoch": e, "metric": metric, "tset": tset, "n": r["n"],
                     "mean": r.get("mean"), "ci_lo": r.get("ci95", [None, None])[0],
                     "ci_hi": r.get("ci95", [None, None])[1], "p_two_sided": r.get("p_two_sided"),
                     "n_pos": r.get("n_pos"), "n_neg": r.get("n_neg"), **{f"s{k}": v for k, v in r["per_seed"].items()}})
    return rows


def test_curriculum(exp, spec, C, E, T) -> Dict[str, Any]:
    seeds = SEEDS
    res: Dict[str, Any] = {"hypothesis": "curriculum_test", "experiment": exp}
    tables: Dict[str, list] = {}
    for K in (20, 10, 30):
        tr, rd = f"sc_true_K{K}", f"sc_rand_K{K}"
        if not ((C.arm == tr).any() and (C.arm == rd).any()):
            res[f"K{K}"] = "lanes not available"
            continue
        blk: Dict[str, Any] = {}
        for ts in TSETS:
            prim = paired(_diff(lambda a, s: _census_val(C, a, s, K, "ssi", ts), tr, rd, seeds),
                          f"SSI census @epoch {K}, {tr} - {rd}")
            prim["verdict"] = verdict_dir(prim, +1)
            csi = paired(_diff(lambda a, s: _census_val(C, a, s, K, "csi", ts), tr, rd, seeds),
                         f"CSI census @epoch {K} (switch), {tr} - {rd}")
            ab = paired(_diff(lambda a, s: _events_sum(E, a, s, ts, "ab_h", (0, K)), tr, rd, seeds),
                        f"excess Ab-E summed over superclass phase (transitions epoch 0..{K}), {tr} - {rd}")
            ab_ab = paired(_diff(lambda a, s: _events_sum(E, a, s, ts, "ab_h", (0, K), "excess_clip_ab"), tr, rd,
                                 seeds), "same, null = N(a)+N(b) convention (sensitivity)")
            parts = {"primary_ssi_census_switch": prim, "csi_census_switch": csi,
                     "excess_AbE_superclass_phase": ab, "excess_AbE_superclass_phase_nullab": ab_ab}
            # arm-level descriptives of the event sums
            for arm in (tr, rd):
                for proc in ("ab_h", "di_h", "tg_h"):
                    v = {s: _events_sum(E, arm, s, ts, proc, (0, K)) for s in seeds}
                    parts[f"arm_{arm}_excess_{proc}_superclass_phase"] = paired(v, f"{arm} {proc} excess sum")
                    vr = {s: _events_sum(E, arm, s, ts, proc, (0, K), "real") for s in seeds}
                    parts[f"arm_{arm}_real_{proc}_superclass_phase"] = paired(vr, f"{arm} {proc} real sum")
                    vn = {s: _events_sum(E, arm, s, ts, proc, (0, K), "null") for s in seeds}
                    parts[f"arm_{arm}_null_{proc}_superclass_phase"] = paired(vn, f"{arm} {proc} null sum")
            blk[ts] = parts
        res[f"K{K}"] = blk
        if K == 20:
            arms = [a for a in ("sc_true_K20", "sc_rand_K20", "scratch_E30") if (C.arm == a).any()]
            for ts in TSETS:
                tables.setdefault("census_by_epoch", []).extend(
                    _per_epoch_table(C, arms, "ssi", ts) + _per_epoch_table(C, arms, "csi", ts)
                    + _per_epoch_table(C, arms, "sai", ts))
                tables.setdefault("paired_by_epoch", []).extend(
                    _paired_by_epoch(C, tr, rd, "ssi", ts, seeds) + _paired_by_epoch(C, tr, rd, "csi", ts, seeds))
    p = res["K20"]["common"]["primary_ssi_census_switch"] if isinstance(res.get("K20"), dict) else {}
    res["primary"] = p
    res["verdict"] = p.get("verdict")
    return res, tables


def test_noise(exp, spec, C, E, T) -> Dict[str, Any]:
    seeds = SEEDS
    res: Dict[str, Any] = {"hypothesis": "noise_test", "experiment": exp}
    tables: Dict[str, list] = {}
    for ts in TSETS:
        def avg(arm, s, metric="ssi"):
            q = C[(C.arm == arm) & (C.seed == s) & (C.tset == ts) & (C.layer == SITE) & (C.epoch >= 5) & (C.epoch <= 30)]
            return float(q[metric].mean()) if len(q) else None
        prim = paired(_diff(avg, "between", "within", seeds), "SSI census mean over epochs 5-30, between - within")
        prim["verdict"] = verdict_dir(prim, -1)
        parts = {"primary_ssi_census_mean_5_30": prim,
                 "ssi_census_epoch5": paired(_diff(lambda a, s: _census_val(C, a, s, 5, "ssi", ts), "between", "within", seeds)),
                 "ssi_census_epoch30": paired(_diff(lambda a, s: _census_val(C, a, s, 30, "ssi", ts), "between", "within", seeds)),
                 "ssi_census_epoch30_between_minus_clean": paired(_diff(lambda a, s: _census_val(C, a, s, 30, "ssi", ts),
                                                                        "between", "clean", seeds)),
                 "excess_AbE_epochs_0_10": paired(_diff(lambda a, s: _events_sum(E, a, s, ts, "ab_h", (0, 10)),
                                                        "between", "within", seeds)),
                 "excess_DiE_epochs_0_10": paired(_diff(lambda a, s: _events_sum(E, a, s, ts, "di_h", (0, 10)),
                                                        "between", "within", seeds)),
                 "excess_DiE_all": paired(_diff(lambda a, s: _events_sum(E, a, s, ts, "di_h", (0, 1e9)),
                                                "between", "within", seeds)),
                 "csi_census_mean_5_30": paired(_diff(lambda a, s: avg(a, s, "csi"), "between", "within", seeds)),
                 "csi_census_epoch5": paired(_diff(lambda a, s: _census_val(C, a, s, 5, "csi", ts), "between", "within", seeds)),
                 "csi_census_epoch30": paired(_diff(lambda a, s: _census_val(C, a, s, 30, "csi", ts), "between", "within", seeds))}
        res[ts] = parts
        arms = ["clean", "between", "within"]
        tables.setdefault("census_by_epoch", []).extend(_per_epoch_table(C, arms, "ssi", ts) + _per_epoch_table(C, arms, "csi", ts))
        tables.setdefault("paired_by_epoch", []).extend(_paired_by_epoch(C, "between", "within", "ssi", ts, seeds)
                                                        + _paired_by_epoch(C, "between", "within", "csi", ts, seeds))
    res["primary"] = res["common"]["primary_ssi_census_mean_5_30"]
    res["verdict"] = res["primary"]["verdict"]
    return res, tables


def test_suppression(exp, spec, C, E, T) -> Dict[str, Any]:
    seeds = SEEDS
    W = spec.get("window_end", 10)
    delta = T["delta"]["ssi"]
    res: Dict[str, Any] = {"hypothesis": "suppression_test", "experiment": exp, "delta_ssi": delta, "window_epochs": [1, W]}
    tables: Dict[str, list] = {}
    final_ep = float(C.epoch.max())
    for ts in TSETS:
        prim = paired(_diff(lambda a, s: _census_val(C, a, s, final_ep, "ssi", ts), "coarse", "random", seeds),
                      f"SSI census @epoch {final_ep:g}, coarse - random")
        if "ci95" in prim:
            lo, hi = prim["ci95"]
            prim["verdict"] = ("RECOVERED" if (lo >= -delta and hi <= delta) else
                               "NOT RECOVERED" if hi < -delta else "INCONCLUSIVE")
            n, m, se = prim["n"], prim["mean"], prim["se"]
            if se > 0:
                p_lo = float(stats.t.sf((m + delta) / se, n - 1))   # H0: diff <= -delta
                p_hi = float(stats.t.cdf((m - delta) / se, n - 1))  # H0: diff >= +delta
                prim["p_tost"] = max(p_lo, p_hi)
                prim["p_below_minus_delta"] = float(stats.t.cdf((m + delta) / se, n - 1))
        eps_post = sorted(e for e in C.epoch.unique() if e > W)
        first_after = eps_post[0] if eps_post else None
        parts = {"primary_ssi_census_final": prim,
                 "ssi_census_first_after_window": paired(_diff(lambda a, s: _census_val(C, a, s, first_after, "ssi", ts),
                                                               "coarse", "random", seeds), f"@epoch {first_after}"),
                 "excess_DiE_window": paired(_diff(lambda a, s: _events_sum(E, a, s, ts, "di_h", (0, W)),
                                                   "coarse", "random", seeds))}
        rec = {}
        for s in seeds:
            got = None
            for e in eps_post:
                v = _diff(lambda a, s_: _census_val(C, a, s_, e, "ssi", ts), "coarse", "random", [s])[s]
                if v is not None and abs(v) <= delta:
                    got = e - W
                    break
            rec[str(s)] = got
        mean_rec = None
        for e in eps_post:
            r = paired(_diff(lambda a, s: _census_val(C, a, s, e, "ssi", ts), "coarse", "random", seeds))
            if r.get("n") and abs(r["mean"]) <= delta:
                mean_rec = e - W
                break
        parts["epochs_after_window_until_within_delta"] = {"per_seed": rec, "seed_mean_difference": mean_rec}
        during = [e for e in sorted(C.epoch.unique()) if 1 <= e <= W]
        parts["during_window_descriptive"] = {
            str(e): paired(_diff(lambda a, s: _census_val(C, a, s, e, "ssi", ts), "coarse", "random", seeds))
            for e in during}
        res[ts] = parts
        arms = ["coarse", "random"]
        tables.setdefault("census_by_epoch", []).extend(_per_epoch_table(C, arms, "ssi", ts) + _per_epoch_table(C, arms, "csi", ts))
        tables.setdefault("paired_by_epoch", []).extend(_paired_by_epoch(C, "coarse", "random", "ssi", ts, seeds))
    f = spec["out"] / f"matched_prewindow_{exp}.csv"
    if f.exists():
        Mx = pd.read_csv(f)
        res["matched_prewindow_fraction"] = Mx.to_dict("records")
        res["matched_prewindow_mean_common"] = Mx[Mx.tset == "common"].groupby("arm")["frac_matched"].mean().to_dict()
    res["primary"] = res["common"]["primary_ssi_census_final"]
    res["verdict"] = res["primary"].get("verdict")
    return res, tables


def cmd_test(exp, spec):
    T = load_thresholds(exp, spec)
    C = pd.read_csv(spec["out"] / f"census_{exp}.csv")
    E = pd.read_csv(spec["out"] / f"events_{exp}.csv")
    fn = {"curriculum_test": test_curriculum, "noise_test": test_noise, "suppression_test": test_suppression}[spec["test"]]
    res, tables = fn(exp, spec, C, E, T)
    integ = json.load(open(spec["out"] / f"integrity_{exp}_summary.json"))
    res["integrity"] = {k: v for k, v in integ.items() if k != "fail_rows"}
    res["thresholds_common"] = {s: v["thresholds"] for s, v in T["common"].items()}
    res["delta"] = T["delta"]
    res["notes"] = ("p_two_sided: paired t-test on per-seed differences (uncorrected; Holm correction across the "
                    "curriculum, static-noise and suppression tests is applied when they are combined). "
                    "CI: t-based 95%. tsets: common = reference-lane alive-only null p95 (primary); run = own-lane "
                    "alive-only null p95 (sensitivity); stored = pipeline-stored adaptive thresholds (sensitivity).")
    tag = spec["test"]
    json.dump(res, open(spec["out"] / f"{tag}_results.json", "w"), indent=1, default=float)
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(spec["out"] / f"{tag}_{name}.csv", index=False)
    # flat summary of every paired outcome
    flat = []

    def walk(o, path):
        if isinstance(o, dict) and "per_seed" in o and "n" in o:
            ci = o.get("ci95", [None, None])
            flat.append({"outcome": "/".join(path), "n": o["n"], "mean": o.get("mean"), "ci_lo": ci[0], "ci_hi": ci[1],
                         "p_two_sided": o.get("p_two_sided"), "n_pos": o.get("n_pos"), "n_neg": o.get("n_neg"),
                         "verdict": o.get("verdict"), **{f"s{k}": v for k, v in o["per_seed"].items()}})
        elif isinstance(o, dict):
            for k, v in o.items():
                if k not in ("primary", "integrity"):
                    walk(v, path + [str(k)])
    walk(res, [])
    pd.DataFrame(flat).to_csv(spec["out"] / f"{tag}_all_outcomes.csv", index=False)
    p = res["primary"]
    print(f"{tag} primary: n={p.get('n')} mean={p.get('mean')} CI={p.get('ci95')} p={p.get('p_two_sided')} "
          f"per-seed={p.get('per_seed')} -> {res['verdict']}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["integrity", "thresholds", "census", "events", "test", "all"])
    ap.add_argument("exp", choices=["EC", "EN", "SUP", "TESTC", "TESTN", "TESTH", "PT", "RN", "SW"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    spec = exp_spec(a.exp)
    spec["out"].mkdir(parents=True, exist_ok=True)
    cmds = ["integrity", "thresholds", "census", "events", "test"] if a.cmd == "all" else [a.cmd]
    for c in cmds:
        t0 = time.time()
        print(f"=== {c} {a.exp} ===", flush=True)
        if c == "integrity":
            cmd_integrity(a.exp, spec, a.workers)
        elif c == "thresholds":
            cmd_thresholds(a.exp, spec, a.workers, a.force)
        elif c == "census":
            cmd_census(a.exp, spec)
        elif c == "events":
            cmd_events(a.exp, spec, a.workers)
        else:
            cmd_test(a.exp, spec)
        print(f"=== {c} {a.exp} done ({time.time() - t0:.0f}s) ===", flush=True)


if __name__ == "__main__":
    main()
