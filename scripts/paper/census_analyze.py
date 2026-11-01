#!/usr/bin/env python3
"""Census robustness: analysis step (reads census_compute.py outputs).

Thresholds per run:
  stored        : the run's stored adaptive thresholds (pipeline, 1000 perms, all non-zero features)
  repro_all     : our replica of the pipeline pool (100 perms, all non-zero features)  [validation]
  alive_pooled  : identical permutation scheme, pool restricted to alive features (>=1% samples),
                  pooled over all checkpoints and layers of the run; same floors (SSI .1, CSI .15,
                  SAI .5) and CSI cap .7
  alive_local   : as alive_pooled but pooled over the layers of one checkpoint
  feature_level : SSI / CSI selective iff above the feature's own permutation p95; SAI uses the
                  alive_pooled threshold (the pipeline's SAI "null" is permutation-invariant, so
                  no feature-level SAI null exists)
Census (fractions of alive features, census layers pooled; ResNet: layer4 dropped, avgpool kept):
  Tg = SAI > t_sai; Ab = SSI > t_ssi & not Tg; Di = CSI > t_csi.
Variants: dedup_stored (stored thr, census layers), a_alive_pooled, b_alive_local, b2_local_ssicsi_pooled_sai,
  fl_feature_level, fl2_feature_and_pooled, c common sweep 3x4x5, d_mean threshold-free means (unweighted and
  activation-weighted; '_w' suffix for weighted curves). Validation: stored thresholds over all layers, counted as in
  noise_null_analyze and compared with its census_per_checkpoint.csv.
Timing: centre of mass (COM) over normalised checkpoint position (as in noise_null_analyze) and argmax; for
  the threshold-free means COM is taken of (curve - min(curve)).

Writes analysis_outputs/paper/CENSUS_robust/tables/*.csv and summary.json.
Usage: census_analyze.py --scratch DIR
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis_outputs" / "paper" / "CENSUS_robust"
NOISE_NULL_REF = REPO / "analysis_outputs" / "paper" / "NOISE_null" / "tables" / "census_per_checkpoint.csv"
FLOOR = {"ssi": 0.1, "csi": 0.15, "sai": 0.5}
CSI_CAP = 0.7
SWEEP_SAI = [0.8, 0.9, 0.95]
SWEEP_SSI = [0.2, 0.3, 0.4, 0.5]
SWEEP_CSI = [0.2, 0.3, 0.4, 0.5, 0.6]
ORDER = {"core": 0, "tinyimagenet": 1, "indep_init": 2}
AB_DECLINE = 0.8  # Ab "rise-and-fall": interior argmax and last <= 0.8 * max


def arch_of(name: str) -> str:
    n = name.lower()
    return "ViT-Small" if "vit" in n else "CCT-7" if "cct" in n else "ResNet-18"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def p95(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=np.float64)
    return float(np.percentile(v, 95)) if v.size else float("nan")


def final_thr(raw: Dict[str, float]) -> Dict[str, float]:
    return {"ssi": max(raw["ssi"], FLOOR["ssi"]), "sai": max(raw["sai"], FLOOR["sai"]),
            "csi": max(min(raw["csi"], CSI_CAP), FLOOR["csi"])}


def com(vals) -> float:
    v = np.clip(np.asarray(vals, dtype=float), 0, None)
    if not np.isfinite(v).all() or v.sum() <= 0:
        return math.nan
    tau = np.linspace(0, 1, len(v))
    return float((tau * v).sum() / v.sum())


def timing(f: Dict[str, List[float]], baseline_sub: bool = False) -> Dict[str, Any]:
    r: Dict[str, Any] = {}
    K = len(f["tg"])
    degenerate = False
    for x in ("tg", "ab", "di"):
        v = np.asarray(f[x], dtype=float)
        c = com(v - v.min()) if baseline_sub else com(v)
        if not np.isfinite(c):
            degenerate = True
        am = int(np.argmax(v))
        r[f"com_{x}"], r[f"argmax_{x}"] = c, am / (K - 1)
        r[f"{x}_first"], r[f"{x}_last"], r[f"{x}_max"] = float(v[0]), float(v[-1]), float(v.max())
    ab = np.asarray(f["ab"], dtype=float)
    am = int(np.argmax(ab))
    if baseline_sub:
        rng = ab.max() - ab.min()
        decl = (ab.max() - ab[-1]) / rng if rng > 0 else 0.0
        r["ab_decline_from_peak"] = float(decl)
        r["ab_rise_fall"] = bool(0 < am < K - 1 and decl >= 1 - AB_DECLINE)
    else:
        r["ab_last_over_max"] = float(ab[-1] / ab.max()) if ab.max() > 0 else math.nan
        r["ab_rise_fall"] = bool(0 < am < K - 1 and ab.max() > 0 and ab[-1] <= AB_DECLINE * ab.max())
    r["degenerate"] = degenerate
    r["order_com"] = bool(not degenerate and r["com_tg"] < r["com_ab"] < r["com_di"])
    r["order_argmax"] = bool(r["argmax_tg"] <= r["argmax_ab"] <= r["argmax_di"])
    r["tg_peak_init"] = bool(np.argmax(f["tg"]) == 0)
    r["all3"] = bool(r["order_com"] and r["order_argmax"] and r["ab_rise_fall"])
    return r


def load_lane(meta_path: Path, scratch: Path) -> Dict[str, Any]:
    m = json.load(open(meta_path))
    z = np.load(scratch / f"{m['lane_id']}.npz")
    m["arch"] = arch_of(m["name"])
    layers = m["layers"]
    m["census_layers"] = [L for L in layers if not ("layer4" in layers and "avgpool" in layers and L == "layer4")]
    amin = m["alive_min"]
    m["cell"] = {}
    for lab in m["labels"]:
        for L in layers:
            k = f"{lab}|{L}"
            cnt = z[f"{k}|cnt"]
            alive = cnt >= amin
            nn = cnt.size
            keep = np.unpackbits(z[f"{k}|csi_keep"], axis=1, count=nn).astype(bool)
            ssi_null = z[f"{k}|ssi_null"].astype(np.float32)
            csi_null = z[f"{k}|csi_null"].astype(np.float32)
            m["cell"][k] = {
                "cnt": cnt, "alive": alive,
                "ssi": z[f"{k}|ssi"], "csi": z[f"{k}|csi"], "sai": z[f"{k}|sai"],
                "ssi_null": ssi_null, "csi_null": csi_null, "csi_keep": keep,
                "sai_snull": z[f"{k}|sai_snull"].astype(np.float32),
            }
    return m


def pools(m: Dict[str, Any], labels, layers, alive_only: bool) -> Dict[str, np.ndarray]:
    s, c, a, sn = [], [], [], []
    for lab in labels:
        for L in layers:
            e = m["cell"][f"{lab}|{L}"]
            sel = e["alive"] if alive_only else np.ones_like(e["alive"])
            sv = e["ssi_null"][:, sel]
            s.append(sv[sv > 0])
            cv, ck = e["csi_null"][:, sel], e["csi_keep"][:, sel]
            c.append(cv[ck])
            av = e["sai"][sel]
            a.append(av[av > 0])
            sn.append(e["sai_snull"][:, sel].ravel())
    return {"ssi": np.concatenate(s), "csi": np.concatenate(c), "sai": np.concatenate(a),
            "sai_snull": np.concatenate(sn)}


def stack_ckpt(m, lab) -> Dict[str, np.ndarray]:
    """Alive features of the census layers at one checkpoint (+ feature-level null p95)."""
    out: Dict[str, List[np.ndarray]] = {k: [] for k in ("ssi", "csi", "sai", "w", "ssi_fp95", "csi_fp95")}
    for L in m["census_layers"]:
        e = m["cell"][f"{lab}|{L}"]
        al = e["alive"]
        out["ssi"].append(e["ssi"][al])
        out["csi"].append(e["csi"][al])
        out["sai"].append(e["sai"][al])
        out["w"].append(e["cnt"][al].astype(np.float64))
        out["ssi_fp95"].append(np.percentile(e["ssi_null"][:, al], 95, axis=0))
        out["csi_fp95"].append(np.percentile(e["csi_null"][:, al], 95, axis=0))
    return {k: np.concatenate(v).astype(np.float64) for k, v in out.items()}


def census_fracs(S: Dict[str, np.ndarray], t_sai, t_ssi, t_csi) -> Dict[str, float]:
    """t_* scalars or per-feature arrays."""
    tg = S["sai"] > t_sai
    ab = (S["ssi"] > t_ssi) & ~tg
    di = S["csi"] > t_csi
    n, w = S["sai"].size, S["w"]
    return {"n_alive": n, "frac_tg": tg.mean(), "frac_ab": ab.mean(), "frac_di": di.mean(),
            "wfrac_tg": w[tg].sum() / w.sum(), "wfrac_ab": w[ab].sum() / w.sum(), "wfrac_di": w[di].sum() / w.sum()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    a = ap.parse_args()
    scratch = Path(a.scratch)
    tdir = OUT / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    lanes = [load_lane(p, scratch) for p in sorted((OUT / "per_lane").glob("*_meta.json"))]
    lanes.sort(key=lambda m: (ORDER[m["group"]], m["arch"], m["name"]))
    summary: Dict[str, Any] = {"n_runs": len(lanes)}

    # thresholds
    thr_rows, loc_rows = [], []
    for m in lanes:
        P_all = pools(m, m["labels"], m["layers"], False)
        P_al = pools(m, m["labels"], m["layers"], True)
        P_alc = pools(m, m["labels"], m["census_layers"], True)
        raw_all = {k: p95(P_all[k]) for k in ("ssi", "csi", "sai")}
        raw_al = {k: p95(P_al[k]) for k in ("ssi", "csi", "sai")}
        raw_alc = {k: p95(P_alc[k]) for k in ("ssi", "csi", "sai")}
        st, sa = m["stored_thresholds"], m["stored_ati"]
        m["thr_stored"] = {"sai": st["sai_task_general_thresh"], "ssi": st["ssi_abstraction_thresh"],
                           "csi": st["csi_differentiation_thresh"]}
        m["thr_alive"] = final_thr(raw_al)
        row = {"group": m["group"], "arch": m["arch"], "lane": m["name"],
               "stored_thr_sai": m["thr_stored"]["sai"], "stored_thr_ssi": m["thr_stored"]["ssi"],
               "stored_thr_csi": m["thr_stored"]["csi"],
               "stored_p95_sai": sa.get("sai_null_percentile"), "stored_p95_ssi": sa.get("ssi_null_percentile"),
               "stored_p95_csi": sa.get("csi_null_percentile"), "stored_n_perm": sa.get("n_permutations"),
               "repro_all_p95_sai": raw_all["sai"], "repro_all_p95_ssi": raw_all["ssi"],
               "repro_all_p95_csi": raw_all["csi"],
               "alive_p95_sai": raw_al["sai"], "alive_p95_ssi": raw_al["ssi"], "alive_p95_csi": raw_al["csi"],
               "alive_thr_sai": m["thr_alive"]["sai"], "alive_thr_ssi": m["thr_alive"]["ssi"],
               "alive_thr_csi": m["thr_alive"]["csi"],
               "alive_censuslayers_p95_sai": raw_alc["sai"], "alive_censuslayers_p95_ssi": raw_alc["ssi"],
               "alive_censuslayers_p95_csi": raw_alc["csi"],
               "diag_sai_samplenull_alive_p95": p95(P_al["sai_snull"]),
               "diag_sai_samplenull_alive_p05": float(np.percentile(P_al["sai_snull"], 5)),
               "share_null_csi_from_nonalive": 1 - P_al["csi"].size / P_all["csi"].size,
               "share_null_ssi_from_nonalive": 1 - P_al["ssi"].size / P_all["ssi"].size}
        thr_rows.append(row)
        m["thr_local"] = {}
        for lab in m["labels"]:
            Pl = pools(m, [lab], m["layers"], True)
            Pla = pools(m, [lab], m["layers"], False)
            rl = {k: p95(Pl[k]) for k in ("ssi", "csi", "sai")}
            m["thr_local"][lab] = final_thr(rl)
            loc_rows.append({"group": m["group"], "lane": m["name"], "checkpoint": lab,
                             **{f"alive_p95_{k}": rl[k] for k in rl},
                             **{f"alive_thr_{k}": m["thr_local"][lab][k] for k in rl},
                             **{f"all_p95_{k}": p95(Pla[k]) for k in ("ssi", "csi", "sai")}})
    write_csv(tdir / "thresholds_per_run.csv", thr_rows)
    write_csv(tdir / "thresholds_ckpt_local.csv", loc_rows)
    summary["thresholds"] = thr_rows

    # census per checkpoint under all variants
    cen_rows: List[Dict[str, Any]] = []
    curves: Dict[str, Dict[str, Dict[str, List[float]]]] = {}  # variant -> lane -> {tg,ab,di}
    sweep_rows: List[Dict[str, Any]] = []
    val_rows = []
    noise_null_ref = {}
    if NOISE_NULL_REF.exists():
        for r in csv.DictReader(open(NOISE_NULL_REF)):
            noise_null_ref[(r["lane"], r["checkpoint"])] = (int(r["n_alive"]), int(r["n_tg"]), int(r["n_ab"]), int(r["n_di"]))

    def add(var, m, lab, fr, extra=None):
        cen_rows.append({"variant": var, "group": m["group"], "arch": m["arch"], "lane": m["name"],
                         "checkpoint": lab, **fr, **(extra or {})})
        c = curves.setdefault(var, {}).setdefault(m["name"], {"tg": [], "ab": [], "di": []})
        c["tg"].append(fr["frac_tg"]); c["ab"].append(fr["frac_ab"]); c["di"].append(fr["frac_di"])
        cw = curves.setdefault(var + "_w", {}).setdefault(m["name"], {"tg": [], "ab": [], "di": []})
        cw["tg"].append(fr["wfrac_tg"]); cw["ab"].append(fr["wfrac_ab"]); cw["di"].append(fr["wfrac_di"])

    combos = list(itertools.product(SWEEP_SAI, SWEEP_SSI, SWEEP_CSI))
    for m in lanes:
        ts, ta = m["thr_stored"], m["thr_alive"]
        sw = {c: {"tg": [], "ab": [], "di": []} for c in combos}
        n_val_ok = 0
        for lab in m["labels"]:
            # validation: stored thresholds, all layers (as in noise_null_analyze)
            nt = na = nd = nal = 0
            for L in m["layers"]:
                e = m["cell"][f"{lab}|{L}"]
                al = e["alive"]
                tg = al & (e["sai"] > ts["sai"]); ab = al & (e["ssi"] > ts["ssi"]) & ~tg
                di = al & (e["csi"] > ts["csi"])
                nal += al.sum(); nt += tg.sum(); na += ab.sum(); nd += di.sum()
            ref = noise_null_ref.get((m["name"], lab))
            ok = ref == (int(nal), int(nt), int(na), int(nd))
            n_val_ok += ok
            S = stack_ckpt(m, lab)
            add("dedup_stored", m, lab, census_fracs(S, ts["sai"], ts["ssi"], ts["csi"]))
            add("a_alive_pooled", m, lab, census_fracs(S, ta["sai"], ta["ssi"], ta["csi"]))
            tl = m["thr_local"][lab]
            add("b_alive_local", m, lab, census_fracs(S, tl["sai"], tl["ssi"], tl["csi"]))
            # b2: checkpoint-local SSI/CSI nulls, SAI from the alive-pooled pool (the SAI "null" is the
            # empirical SAI distribution, so a checkpoint-local SAI p95 pins Tg at ~5% by construction)
            add("b2_local_ssicsi_pooled_sai", m, lab, census_fracs(S, ta["sai"], tl["ssi"], tl["csi"]))
            add("fl_feature_level", m, lab, census_fracs(S, ta["sai"], S["ssi_fp95"], S["csi_fp95"]))
            # fl2: significant against own null AND above the alive-pooled effect-size threshold
            add("fl2_feature_and_pooled", m, lab,
                census_fracs(S, ta["sai"], np.maximum(S["ssi_fp95"], ta["ssi"]), np.maximum(S["csi_fp95"], ta["csi"])))
            # threshold-free means
            w = S["w"] / S["w"].sum()
            tf = {"n_alive": S["sai"].size, "frac_tg": S["sai"].mean(), "frac_ab": S["ssi"].mean(),
                  "frac_di": S["csi"].mean(), "wfrac_tg": (w * S["sai"]).sum(), "wfrac_ab": (w * S["ssi"]).sum(),
                  "wfrac_di": (w * S["csi"]).sum()}
            add("d_mean", m, lab, tf)
            for c in combos:
                fr = census_fracs(S, *c)
                for x in ("tg", "ab", "di"):
                    sw[c][x].append(fr[f"frac_{x}"])
        val_rows.append({"lane": m["name"], "census_all_layers_exact_vs_noise_null": f"{n_val_ok}/{len(m['labels'])}",
                         **{f"sel_maxabs_{k}": v for k, v in m["validation_selectivity_maxabs"].items()}})
        for c in combos:
            t = timing(sw[c])
            sweep_rows.append({"group": m["group"], "arch": m["arch"], "lane": m["name"], "t_sai": c[0],
                               "t_ssi": c[1], "t_csi": c[2], **t,
                               **{f"frac_{x}_curve": ";".join(f"{v:.4f}" for v in sw[c][x]) for x in ("tg", "ab", "di")}})
    write_csv(tdir / "validation.csv", val_rows)
    write_csv(tdir / "census_per_checkpoint.csv", cen_rows)
    write_csv(tdir / "sweep_timing.csv", sweep_rows)
    summary["validation"] = val_rows

    # timing per variant
    tim_rows = []
    meta_by = {m["name"]: m for m in lanes}
    for var, byl in curves.items():
        for ln, f in byl.items():
            m = meta_by[ln]
            t = timing(f, baseline_sub=var.startswith("d_mean"))
            tim_rows.append({"variant": var, "group": m["group"], "arch": m["arch"], "lane": ln,
                             "n_checkpoints": len(f["tg"]), **t})
    write_csv(tdir / "census_timing.csv", tim_rows)

    # robustness counts
    cnt_rows = []
    for var in curves:
        rr = [r for r in tim_rows if r["variant"] == var]
        cnt_rows.append({"variant": var, "n_runs": len(rr),
                         "n_order_com": sum(r["order_com"] for r in rr),
                         "n_order_argmax": sum(r["order_argmax"] for r in rr),
                         "n_ab_rise_fall": sum(r["ab_rise_fall"] for r in rr),
                         "n_tg_peak_init": sum(r["tg_peak_init"] for r in rr),
                         "n_all3": sum(r["all3"] for r in rr),
                         "n_degenerate": sum(r["degenerate"] for r in rr),
                         "fail_order_com": ";".join(r["lane"] for r in rr if not r["order_com"])})
    # sweep summary
    sw_lane = []
    for m in lanes:
        rr = [r for r in sweep_rows if r["lane"] == m["name"]]
        sw_lane.append({"group": m["group"], "arch": m["arch"], "lane": m["name"], "n_combos": len(rr),
                        "frac_order_com": np.mean([r["order_com"] for r in rr]),
                        "frac_order_argmax": np.mean([r["order_argmax"] for r in rr]),
                        "frac_ab_rise_fall": np.mean([r["ab_rise_fall"] for r in rr]),
                        "frac_all3": np.mean([r["all3"] for r in rr]),
                        "n_degenerate": int(sum(r["degenerate"] for r in rr))})
    write_csv(tdir / "sweep_per_run.csv", sw_lane)
    combo_rows = []
    for c in combos:
        rr = [r for r in sweep_rows if (r["t_sai"], r["t_ssi"], r["t_csi"]) == c]
        combo_rows.append({"t_sai": c[0], "t_ssi": c[1], "t_csi": c[2],
                           "n_order_com": sum(r["order_com"] for r in rr),
                           "n_order_argmax": sum(r["order_argmax"] for r in rr),
                           "n_ab_rise_fall": sum(r["ab_rise_fall"] for r in rr),
                           "n_all3": sum(r["all3"] for r in rr), "n_degenerate": sum(r["degenerate"] for r in rr)})
    write_csv(tdir / "sweep_per_combo.csv", combo_rows)
    cnt_rows.append({"variant": "c_sweep(all 60 combos x 11 runs)", "n_runs": len(lanes),
                     "n_order_com": float(np.mean([r["n_order_com"] for r in combo_rows])),
                     "n_order_argmax": float(np.mean([r["n_order_argmax"] for r in combo_rows])),
                     "n_ab_rise_fall": float(np.mean([r["n_ab_rise_fall"] for r in combo_rows])),
                     "n_all3": float(np.mean([r["n_all3"] for r in combo_rows])),
                     "n_degenerate": float(np.mean([r["n_degenerate"] for r in combo_rows])),
                     "fail_order_com": f"mean over combos; combos with 11/11 COM order: "
                                       f"{sum(r['n_order_com'] == len(lanes) for r in combo_rows)}/{len(combos)}; "
                                       f">=10/11: {sum(r['n_order_com'] >= len(lanes) - 1 for r in combo_rows)}"})
    write_csv(tdir / "robustness_counts.csv", cnt_rows)
    summary["robustness_counts"] = cnt_rows
    summary["sweep_per_run"] = sw_lane
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    for r in cnt_rows:
        print({k: r[k] for k in r if k != "fail_order_com"}, r["fail_order_com"][:200])


if __name__ == "__main__":
    main()
