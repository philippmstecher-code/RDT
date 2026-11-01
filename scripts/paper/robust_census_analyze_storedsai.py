#!/usr/bin/env python3
"""Robustness of the core census to three measurement issues (variants a, b, c; keys 'variant_a', 'variant_b', 'variant_c' in the
outputs). Lineages (claim 2) use the run's stored SAI threshold.

Inputs (read-only): per-run state in cache_census/state (per real checkpoint x layer: dictionary size, alive features,
stored Phase-4 SSI / CSI / SAI, label-free SAI null), Phase-5 matching caches (TRANSFORMED/cache for the 9 core
runs; cache_census/match for Tiny ImageNet, independent-init and the clean lanes), stored sae_results.json.

Runs / thresholds (as census_strict): 11 census runs with the two census threshold sets
  dedup_stored      stored per-run thresholds (Fig. 4d census); lineages: Fig. 4a thresholds (t_sai = 0.9, stored
                 t_ssi / t_csi)
  a_alive_pooled alive-pooled null thresholds for census and lineages,
and the 5 clean lanes with the 'EN common' thresholds (t_sai = 0.9), reported alongside.

Baseline categories (Methods rule, Fig. 4a/4d; exclusive):
  Tg = SAI > t_sai;  Ab = SSI > t_ssi & CSI <= t_csi & ~Tg;  Di = CSI > t_csi & ~Tg   (CSI NaN -> not selective)
Variants (output keys in quotes):
  a 'variant_a'     t_sai = the run's label-free null p95: SAI of alive features under 100 row-shuffled class assignments,
              pooled over the run's real checkpoints x census layers (replaces the fixed / stored t_sai; census
              and lineages).
  b 'variant_b'     Ab inclusive: Ab = SSI > t_ssi & ~Tg, regardless of CSI (Di unchanged).
  c 'variant_c'     census (and lineages) restricted to healthy SAE layers: alive >= 10% of the dictionary at a majority
              (> 1/2) of the real checkpoints; dropped layers listed per run.
  abc 'variant_abc' all three; the variant-a null is pooled over the healthy census layers only.
Census: fractions of alive features, census layers pooled (ResNet: layer4 dropped, avgpool kept), real checkpoints
('terminal' / duplicate final dropped). Claims per run:
  1  COM ordering Tg < Ab < Di (census_analyze.timing 'order_com');
  2  persistence: mean lineage lifespan (persist_lib.records_real 'paper', dedup; Fig. 4a) of Tg and of Ab
     each longer than Di (ratio > 1); the Fig. 4a coarse/fine ratio (Tg | Ab vs Di) is reported alongside;
  3  rise and fall of the superclass census (census_analyze.timing 'ab_rise_fall': interior argmax and
     last <= 0.8 x max; 'interior peak only' reported alongside).
Criterion: a claim is met if it holds in >= 10 of 11 runs under every variant a, b, c, abc (each threshold set);
otherwise the failing variants are listed.

Writes analysis_outputs/paper/ROBUST_census_storedSAI/{results.json, tables/*.csv, fig_robust_census.{png,pdf}}.
Usage: python scripts/paper/robust_census_analyze_storedsai.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import robust_census_lib as RL  # noqa: E402

import json  # noqa: E402
import pickle  # noqa: E402
from typing import Any, Dict, List  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from census_analyze import timing  # noqa: E402
from persist_lib import sel_arrays  # noqa: E402

SDIR = RL.CACHE / "state"
MDIR = RL.CACHE / "match"
RL.OUT = RL.OUT.parent / "ROBUST_census_storedSAI"
TAB = RL.OUT / "tables"
FIG4_RULE = RL.REPO / "analysis_outputs" / "paper" / "PERSIST_robust" / "fig4_methods_rule.json"
VARIANTS = ("base", "variant_a", "variant_b", "variant_c", "variant_abc", "variant_a_p05", "variant_abc_p05")
ROBUST = ("variant_a", "variant_b", "variant_c", "variant_abc")          # variants entering the criterion
DESCRIPTIVE = ("variant_a_p05", "variant_abc_p05")           # descriptive only: t_sai = pooled null p05 (not in the criterion)
FLAGS = {"base": (False, False, False, None), "variant_a": (True, False, False, 95.0), "variant_b": (False, True, False, None),
         "variant_c": (False, False, True, None), "variant_abc": (True, True, True, 95.0),
         "variant_a_p05": (True, False, False, 5.0), "variant_abc_p05": (True, True, True, 5.0)}
CLAIMS = ("c1_order_com", "c2_persist", "c3_rise_fall")
COL = {"tg": "#8B5CF6", "ab": "#3B82F6", "di": "#F59E0B"}
MRK = {"tg": "o", "ab": "s", "di": "^"}


# loading
def load_state(r):
    z = np.load(SDIR / f"{r['run']}.npz")
    meta = json.load(open(SDIR / f"{r['run']}_meta.json"))
    assert meta["real"] == r["real"]
    return {k: z[k] for k in z.files}, meta


def load_match(r):
    p = RL.TRANSFORMED_CACHE / f"{r['lane_id']}.pkl"
    if p.exists():
        d = pickle.load(open(p, "rb"))
        assert d["labels"][:-1] == r["real"] and d["labels"][-1] == "terminal", (r["run"], d["labels"])
        return d, "TRANSFORMED/cache"
    d = pickle.load(open(MDIR / f"{r['run']}.pkl", "rb"))
    assert d["real"] == r["real"]
    return d, "cache_census/match"


def health(r, S) -> Dict[str, Dict[str, Any]]:
    out = {}
    K = len(r["real"])
    for L in r["layers"]:
        fr = [len(S[f"{lab}|{L}|fid"]) / int(S[f"{lab}|{L}|n_dict"]) for lab in r["real"]]
        n_ok = int(sum(f >= RL.HEALTHY_FRAC for f in fr))
        out[L] = {"healthy": n_ok > K / 2, "n_ckpt_ok": n_ok, "n_ckpt": K,
                  "n_dict": int(S[f"{r['real'][0]}|{L}|n_dict"]),
                  "alive_min": int(min(len(S[f"{lab}|{L}|fid"]) for lab in r["real"])),
                  "alive_max": int(max(len(S[f"{lab}|{L}|fid"]) for lab in r["real"])),
                  "alive_frac_median": float(np.median(fr))}
    return out


def null_pct(r, S, layers, pct=RL.SAI_NULL_PCT) -> float:
    v = [S[f"{lab}|{L}|sai_null"].ravel() for lab in r["real"] for L in layers if f"{lab}|{L}|sai_null" in S]
    if not v:  # no (healthy) census layer: variant c / abc census not evaluable for this run
        return float("nan")
    return float(np.percentile(np.concatenate(v).astype(np.float64), pct))


# categories
def cats(ssi, csi, sai, t, incl_ab: bool):
    tg = sai > t["sai"]
    cs = np.nan_to_num(csi, nan=0.0) > t["csi"]
    ss = ssi > t["ssi"]
    ab = ss & ~tg if incl_ab else ss & ~cs & ~tg
    return tg, ab, cs & ~tg


# lineage records (persist_lib.records_real, real labels explicit)
def records(d, selectivity, real, layers):
    """Copy of persist_lib.records_real(rule='paper') with the real checkpoints given explicitly (the clean lanes
    have a duplicate final instead of 'terminal'). Returns columns (layer, life, ssi, csi, sai)."""
    trans = [f"{a}->{b}" for a, b in zip(real[:-1], real[1:])]
    n_t = len(trans)
    rows = []
    for li, layer in enumerate(layers):
        per_t = d["layer_data"][layer]
        fwd = [{i: j for i, j, _, _ in per_t[t]["stable"]} for t in trans]
        for b in range(n_t):
            alive = np.where(per_t[trans[b]]["alive_a"])[0]
            ssi, csi, sai = sel_arrays(selectivity, real[b], layer, len(per_t[trans[b]]["alive_a"]))
            for fid in alive:
                cur, life = int(fid), 0
                for k in range(b, n_t):
                    if cur in fwd[k]:
                        life += 1
                        cur = fwd[k][cur]
                    else:
                        break
                rows.append((li, life, ssi[fid], csi[fid], sai[fid]))
    return np.array(rows, dtype=np.float64), layers


def persistence(R, rec_layers, keep_layers, t, incl_ab):
    keep = np.isin(R[:, 0], [i for i, L in enumerate(rec_layers) if L in keep_layers])
    R = R[keep]
    life = R[:, 1]
    tg, ab, di = cats(R[:, 2], R[:, 3], R[:, 4], t, incl_ab)
    m = lambda x: float(life[x].mean()) if x.any() else np.nan  # noqa: E731
    out = {"n_tg": int(tg.sum()), "n_ab": int(ab.sum()), "n_di": int(di.sum()),
           "life_tg": m(tg), "life_ab": m(ab), "life_di": m(di), "life_coarse": m(tg | ab)}
    for k in ("tg", "ab", "coarse"):
        out[f"ratio_{k}_di"] = out[f"life_{k}"] / out["life_di"] if out["life_di"] and np.isfinite(out["life_di"]) else np.nan
    out["c2_persist"] = bool(out["ratio_tg_di"] > 1 and out["ratio_ab_di"] > 1)
    return out


# per run
def analyse_run(r) -> Dict[str, Any]:
    S, meta = load_state(r)
    hl = health(r, S)
    healthy = [L for L in r["layers"] if hl[L]["healthy"]]
    cl_all = r["census_layers"]
    cl_h = [L for L in cl_all if L in healthy]
    rec_layers_all = RL.record_layers(r)
    rec_h = [L for L in rec_layers_all if L in healthy]
    p95_all, p95_h = null_pct(r, S, cl_all), null_pct(r, S, cl_h)
    pnull = {(95.0, False): p95_all, (95.0, True): p95_h,
             (5.0, False): null_pct(r, S, cl_all, 5.0), (5.0, True): null_pct(r, S, cl_h, 5.0)}
    per_layer_p95 = {L: null_pct(r, S, [L]) for L in cl_all}
    per_layer_p05 = {L: float(np.percentile(np.concatenate([S[f"{lab}|{L}|sai_null"].ravel() for lab in r["real"]
                                                            if f"{lab}|{L}|sai_null" in S]), 5))
                     for L in cl_all if any(f"{lab}|{L}|sai_null" in S for lab in r["real"])}
    lane = __import__("noise_null_lib").Lane(Path(r["dir"]), dataset_id=r["dataset"])
    d, msrc = load_match(r)
    R, rec_layers = records(d, lane.results["selectivity"], r["real"], rec_layers_all)
    out = {"run": r["run"], "group": r["group"], "arch": r["arch"], "health": hl, "healthy_layers": healthy,
           "dropped_layers": [L for L in r["layers"] if L not in healthy],
           "dropped_census_layers": [L for L in cl_all if L not in cl_h],
           "dropped_lineage_layers": [L for L in rec_layers_all if L not in rec_h],
           "sai_null_p95_census_layers": p95_all, "sai_null_p95_healthy_census_layers": p95_h,
           "sai_null_p05_census_layers": pnull[(5.0, False)], "sai_null_p05_healthy_census_layers": pnull[(5.0, True)],
           "sai_null_p95_per_layer": per_layer_p95, "sai_null_p05_per_layer": per_layer_p05,
           "match_source": msrc, "match_mismatches": len(d.get("mismatches", [])), "n_records": int(len(R)),
           "rows_census": [], "rows_timing": [], "rows_persist": [], "rows_change": []}
    for tset in r["tsets"]:
        Tc = dict(r["thr"][tset])
        Tp = dict(Tc)  # lineages use the run's stored SAI threshold as well
        base_lab = {}
        for v in VARIANTS:
            fa, fb, fc, pct = FLAGS[v]
            t_sai_null = pnull[(pct, fc)] if fa else None
            tc = {**Tc, "sai": t_sai_null} if fa else Tc
            tp = {**Tp, "sai": t_sai_null} if fa else Tp
            layers = cl_h if fc else cl_all
            curves = {"tg": [], "ab": [], "di": []}
            ch = {"n_base_alive": 0, "n_changed": 0, "n_dropped": 0, "tg_lost": 0, "tg_gained": 0, "ab_lost": 0,
                  "ab_gained": 0, "di_lost": 0, "di_gained": 0}
            for lab in r["real"]:
                n_al = n_tg = n_ab = n_di = 0
                for L in cl_all:
                    k = f"{lab}|{L}"
                    tg, ab, di = cats(S[f"{k}|ssi"], S[f"{k}|csi"], S[f"{k}|sai"], tc, fb)
                    if v == "base":
                        base_lab[k] = (tg, ab, di)
                    b = base_lab[k]
                    ch["n_base_alive"] += len(tg)
                    if L not in layers:
                        ch["n_dropped"] += len(tg)
                        continue
                    diff = (tg != b[0]) | (ab != b[1]) | (di != b[2])
                    ch["n_changed"] += int(diff.sum())
                    for nm, x, y in (("tg", tg, b[0]), ("ab", ab, b[1]), ("di", di, b[2])):
                        ch[f"{nm}_lost"] += int((y & ~x).sum())
                        ch[f"{nm}_gained"] += int((x & ~y).sum())
                    n_al += len(tg)
                    n_tg += int(tg.sum())
                    n_ab += int(ab.sum())
                    n_di += int(di.sum())
                f = lambda x: x / n_al if n_al else np.nan  # noqa: E731
                curves["tg"].append(f(n_tg))
                curves["ab"].append(f(n_ab))
                curves["di"].append(f(n_di))
                out["rows_census"].append({"run": r["run"], "group": r["group"], "arch": r["arch"], "tset": tset,
                                           "variant": v, "checkpoint": lab, "n_alive": n_al, "n_tg": n_tg,
                                           "n_ab": n_ab, "n_di": n_di, "frac_tg": f(n_tg), "frac_ab": f(n_ab),
                                           "frac_di": f(n_di)})
            tm = timing(curves)
            ab = np.asarray(curves["ab"])
            am = int(np.argmax(ab))
            tm["ab_interior_peak"] = bool(0 < am < len(ab) - 1)
            ps = persistence(R, rec_layers, rec_h if fc else rec_layers_all, tp, fb)
            out["rows_timing"].append({"run": r["run"], "group": r["group"], "arch": r["arch"], "tset": tset,
                                       "variant": v, "evaluable": bool(layers), "t_sai_census": tc["sai"], "t_sai_lineage": tp["sai"],
                                       "t_ssi": tc["ssi"], "t_csi": tc["csi"], "census_layers": ";".join(layers),
                                       "c1_order_com": tm["order_com"], "c2_persist": ps["c2_persist"],
                                       "c3_rise_fall": tm["ab_rise_fall"], **tm,
                                       "share_tg_first": curves["tg"][0], "share_ab_peak": float(ab.max()),
                                       "share_ab_last": curves["ab"][-1], "share_di_last": curves["di"][-1],
                                       "share_tg_last": curves["tg"][-1], "share_di_first": curves["di"][0],
                                       "ab_curve": ";".join(f"{x:.4f}" for x in curves["ab"]),
                                       "tg_curve": ";".join(f"{x:.4f}" for x in curves["tg"]),
                                       "di_curve": ";".join(f"{x:.4f}" for x in curves["di"])})
            out["rows_persist"].append({"run": r["run"], "group": r["group"], "arch": r["arch"], "tset": tset,
                                        "variant": v, "t_sai": tp["sai"], "t_ssi": tp["ssi"], "t_csi": tp["csi"],
                                        "lineage_layers": ";".join(rec_h if fc else rec_layers_all), **ps})
            n0 = ch["n_base_alive"]
            out["rows_change"].append({"run": r["run"], "group": r["group"], "arch": r["arch"], "tset": tset,
                                       "variant": v, **ch,
                                       "frac_changed_of_all": ch["n_changed"] / n0 if n0 else np.nan,
                                       "frac_dropped": ch["n_dropped"] / n0 if n0 else np.nan,
                                       "frac_changed_or_dropped": (ch["n_changed"] + ch["n_dropped"]) / n0 if n0 else np.nan,
                                       "frac_changed_of_kept": (ch["n_changed"] / (n0 - ch["n_dropped"])
                                                                if n0 - ch["n_dropped"] else np.nan)})
    print(f"{r['run']}: dropped {out['dropped_layers']}, null p95 {p95_all:.4f} (healthy {p95_h:.4f})", flush=True)
    return out


# summary
def validate(tim: pd.DataFrame, per: pd.DataFrame) -> Dict[str, Any]:
    """Baseline reproduces Fig. 4a (methods rule ratios) and the census_strict counts (real checkpoints)."""
    rule = json.load(open(FIG4_RULE))["fig4a"]["methods"]["per_run"]
    p = per[(per.variant == "base") & (per.tset == "dedup_stored") & (per.group == "core")]
    dif = {}
    for _, x in p.iterrows():
        seed = x.run.split("seed")[1]
        dif[x.run] = abs(round(x.ratio_coarse_di, 4) - rule[f"{x.arch}|{seed}"])
    strict = pd.read_csv(RL.REPO / "analysis_outputs" / "paper" / "DATA_order" / "census_check" / "counts_per_ckpt_layer.csv",
                     dtype={"checkpoint": str})
    strict = strict[strict.census_layer].groupby(["run", "tset", "checkpoint"])[["n_alive", "n_tg"]].sum().reset_index()
    return {"fig4a_ratio_absdiff_max": float(max(dif.values())), "fig4a_ratio_absdiff": dif, "_a9": strict}


def main() -> None:
    RL.assert_writable(RL.OUT)
    TAB.mkdir(parents=True, exist_ok=True)
    res = [analyse_run(r) for r in RL.runs()]
    cen = pd.DataFrame([x for o in res for x in o["rows_census"]])
    tim = pd.DataFrame([x for o in res for x in o["rows_timing"]])
    per = pd.DataFrame([x for o in res for x in o["rows_persist"]])
    chg = pd.DataFrame([x for o in res for x in o["rows_change"]])
    cen.to_csv(TAB / "census_per_ckpt.csv", index=False)
    tim.to_csv(TAB / "timing_per_run.csv", index=False)
    per.to_csv(TAB / "persistence_per_run.csv", index=False)
    chg.to_csv(TAB / "label_change_per_run.csv", index=False)
    lay = pd.DataFrame([{"run": o["run"], "group": o["group"], "layer": L, **h,
                         "sai_null_p95_layer": o["sai_null_p95_per_layer"].get(L, np.nan),
                         "sai_null_p05_layer": o["sai_null_p05_per_layer"].get(L, np.nan)}
                        for o in res for L, h in o["health"].items()])
    lay.to_csv(TAB / "layer_health.csv", index=False)

    val = validate(tim, per)
    strict = val.pop("_a9")
    base = cen[(cen.variant == "base") & (cen.group != "clean30")]
    mm = base.merge(strict, on=["run", "tset", "checkpoint"], suffixes=("", "_a9"))
    val["strict_census_rows_compared"] = int(len(mm))
    val["strict_n_alive_exact"] = bool((mm.n_alive == mm.n_alive_strict).all())
    val["strict_n_tg_exact"] = bool((mm.n_tg == mm.n_tg_strict).all())
    val["sai_formula_vs_stored_maxabs"] = max(json.load(open(SDIR / f"{r['run']}_meta.json"))["validation"]
                                              ["sai_recomputed_vs_stored_maxabs"] for r in RL.runs())
    val["match_mismatches_total"] = int(sum(o["match_mismatches"] for o in res))

    # merge persistence claim into timing table
    tim = tim.drop(columns="c2_persist").merge(per[["run", "tset", "variant", "c2_persist", "ratio_tg_di", "ratio_ab_di",
                                                    "ratio_coarse_di"]], on=["run", "tset", "variant"])
    tim.to_csv(TAB / "timing_per_run.csv", index=False)

    verdicts: Dict[str, Any] = {}
    counts_rows = []
    for tset in ("dedup_stored", "a_alive_pooled", "EN_common"):
        t = tim[tim.tset == tset]
        if t.empty:
            continue
        vt = {}
        for c in CLAIMS + ("c1_tg_lt_ab", "c1_ab_lt_di", "c2_tg_di", "c2_ab_di", "c2_coarse_fine", "c3_interior_peak"):
            vt[c] = {}
            for v in VARIANTS:
                x = t[t.variant == v]
                ok = {"c1_tg_lt_ab": x.com_tg < x.com_ab, "c1_ab_lt_di": x.com_ab < x.com_di,
                      "c2_tg_di": x.ratio_tg_di > 1, "c2_ab_di": x.ratio_ab_di > 1, "c2_coarse_fine": x.ratio_coarse_di > 1,
                      "c3_interior_peak": x.ab_interior_peak}.get(c, x[c] if c in x else None)
                ok = ok.astype(bool)
                vt[c][v] = {"n": int(ok.sum()), "of": int(len(x)), "fail": x[~ok].run.tolist()}
                counts_rows.append({"tset": tset, "claim": c, "variant": v, "n_hold": int(ok.sum()), "n_runs": int(len(x)),
                                    "fail": ";".join(x[~ok].run.tolist())})
        if tset != "EN_common":
            for c in CLAIMS:
                bad = [v for v in ROBUST if vt[c][v]["n"] < 10]
                vt[c]["stands"] = not bad
                vt[c]["failing_variants"] = bad
        verdicts[tset] = vt
    pd.DataFrame(counts_rows).to_csv(TAB / "claim_counts.csv", index=False)
    final = {}
    for c in CLAIMS:
        bad = {ts: verdicts[ts][c]["failing_variants"] for ts in ("dedup_stored", "a_alive_pooled")}
        stands = all(not b for b in bad.values())
        final[c] = {"verdict": "criterion met" if stands else "criterion not met",
                    "failing": {ts: b for ts, b in bad.items() if b},
                    "counts_dedup_stored": {v: f"{verdicts['dedup_stored'][c][v]['n']}/11" for v in VARIANTS},
                    "counts_a_alive_pooled": {v: f"{verdicts['a_alive_pooled'][c][v]['n']}/11" for v in VARIANTS},
                    "clean30_EN_common": {v: f"{verdicts['EN_common'][c][v]['n']}/5" for v in VARIANTS}}

    # headline numbers
    head: Dict[str, Any] = {}
    for tset in ("dedup_stored", "a_alive_pooled", "EN_common"):
        h = {}
        for v in VARIANTS:
            p = per[(per.tset == tset) & (per.variant == v)]
            core = p[p.group == "core"]
            x = tim[(tim.tset == tset) & (tim.variant == v)]
            c = chg[(chg.tset == tset) & (chg.variant == v)]
            h[v] = {
                "fig4a_coarse_fine_core9_range": [float(core.ratio_coarse_di.min()), float(core.ratio_coarse_di.max())] if len(core) else None,
                "coarse_fine_range_all": [float(p.ratio_coarse_di.min()), float(p.ratio_coarse_di.max())],
                "tg_di_range": [float(p.ratio_tg_di.min()), float(p.ratio_tg_di.max())],
                "ab_di_range": [float(p.ratio_ab_di.min()), float(p.ratio_ab_di.max())],
                "tg_di_median": float(p.ratio_tg_di.median()), "ab_di_median": float(p.ratio_ab_di.median()),
                "mean_share_tg_first": float(x.share_tg_first.mean()), "mean_share_tg_last": float(x.share_tg_last.mean()),
                "mean_share_ab_peak": float(x.share_ab_peak.mean()), "mean_share_ab_last": float(x.share_ab_last.mean()),
                "mean_share_di_last": float(x.share_di_last.mean()),
                "mean_com": {k: float(x[f"com_{k}"].mean()) for k in ("tg", "ab", "di")},
                "label_change_pooled": {
                    "frac_changed_of_all": float(c.n_changed.sum() / c.n_base_alive.sum()),
                    "frac_dropped": float(c.n_dropped.sum() / c.n_base_alive.sum()),
                    "frac_changed_or_dropped": float((c.n_changed.sum() + c.n_dropped.sum()) / c.n_base_alive.sum()),
                    "frac_changed_of_kept": float(c.n_changed.sum() / (c.n_base_alive.sum() - c.n_dropped.sum())),
                    "frac_changed_of_all_per_run_range": [float(c.frac_changed_of_all.min()), float(c.frac_changed_of_all.max())],
                    **{k: int(c[k].sum()) for k in ("tg_lost", "tg_gained", "ab_lost", "ab_gained", "di_lost", "di_gained",
                                                    "n_base_alive")}},
            }
        head[tset] = h

    summary = {
        "rule": "criterion: a claim holds in >= 10 of 11 census runs under every variant (a, b, c, abc; keys variant_a, variant_b, "
                "variant_c, variant_abc); applied to both threshold sets (dedup_stored, a_alive_pooled); clean lanes reported alongside",
        "claims": {"c1_order_com": "census COM ordering Tg < Ab < Di",
                   "c2_persist": "mean lineage lifespan Tg > Di and Ab > Di (Fig. 4a records; both ratios > 1)",
                   "c3_rise_fall": "superclass census rises and falls (interior argmax and last <= 0.8 x max)"},
        "final_verdicts": final,
        "verdicts": verdicts,
        "headline": head,
        "dropped_layers": {o["run"]: {"all": o["dropped_layers"], "census": o["dropped_census_layers"],
                                      "lineage": o["dropped_lineage_layers"],
                                      "alive_frac_median": {L: round(o["health"][L]["alive_frac_median"], 4)
                                                            for L in o["dropped_layers"]}} for o in res},
        "sai_null": {o["run"]: {"p95_census_layers": o["sai_null_p95_census_layers"],
                                    "p95_healthy_census_layers": o["sai_null_p95_healthy_census_layers"],
                                    "p05_census_layers": o["sai_null_p05_census_layers"],
                                    "p05_healthy_census_layers": o["sai_null_p05_healthy_census_layers"],
                                    "per_layer_p95": o["sai_null_p95_per_layer"],
                                    "per_layer_p05": o["sai_null_p05_per_layer"]} for o in res},
        "validation": val,
        "notes": [
            "Census on real checkpoints only ('terminal' = duplicate of the last real milestone dropped), as Fig. 4d and the "
            "SAI null definition; the census check counts (DATA_order/census_check) also contain 'terminal', which is not compared.",
            "Baseline categories = the Methods rule (exclusive; Fig. 4a/4d), so that variant b (inclusive Ab) is a change; "
            "variant b keeps Ab & ~Tg (task-general priority) and leaves Di = CSI > t_csi & ~Tg unchanged.",
            "Variant-a null pooled over the census layers (ResNet layer4 dropped: same input as avgpool); under variant abc pooled over "
            "the healthy census layers only. The same 100 row permutations are used for every matrix of a run.",
            "Lineage thresholds: dedup_stored uses the Fig. 4a set (t_sai = 0.9, stored t_ssi / t_csi); variant a replaces t_sai by "
            "the null p95 in census and lineages alike.",
            "Lineages of the Tiny ImageNet, independent-init and clean runs use Phase-5 matching caches in cache_census/match (pipeline "
            "matching between consecutive real checkpoints, validated against stored counts); Fig. 4a itself covers the "
            "9 core runs only.",
            "Claim 2 holds in a run iff both Tg/Di and Ab/Di lifespan ratios exceed 1; the Fig. 4a coarse/fine ratio is reported.",
            "A run with no healthy census layer (variants c / abc) is not evaluable and counted as not holding.",
            "Descriptive extra variants variant_a_p05 / variant_abc_p05 (t_sai = pooled null p05) are reported but are not part of the criterion.",
            "Variant a is near-degenerate: the pooled label-free SAI null p95 is 0.998 (ViT, CCT) and 0.9998 "
            "(ResNet), because shuffled SAI of dense features approaches 1; only 0-4 lineage records per run (of thousands) "
            "and ~0.1% of alive census features at the first checkpoint exceed it. Variant-a failures of claims 1 and 2 are therefore "
            "the task-general category emptying, not a reordering of existing task-general features; the superclass parts "
            "(COM Ab < Di, lifespan Ab > Di) hold in >= 10/11 runs under every variant.",
            "VitSmall seed 256 has a 1536-feature dictionary (768 in the other ViTs); under the 10%-of-dictionary rule all six "
            "of its layers are unhealthy (median alive 6.7-8.8%), so it is not evaluable under variants c / abc.",
        ],
    }
    with open(RL.OUT / "results.json", "w") as f:
        json.dump(summary, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({"final_verdicts": final, "validation": val}, indent=1, default=str))
    figure(tim, cen, per, summary)


# figure
def figure(tim, cen, per, summary) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 7.5, "axes.labelsize": 7})
    fig = plt.figure(figsize=(7.4, 5.8))
    gs = fig.add_gridspec(2, 3, hspace=0.6, wspace=0.45)
    vlab = list(VARIANTS)
    xs = np.arange(len(vlab))
    vtick = [v.replace("_p05", "\n(p05)") for v in vlab]

    def sep(ax):
        ax.axvline(4.5, color="#9CA3AF", lw=0.6, ls="-")
    # a: runs holding each claim (of 11), both threshold sets
    ax = fig.add_subplot(gs[0, 0])
    cl = [("c1_order_com", "COM order", COL["tg"], "o"), ("c2_persist", "Persistence", COL["ab"], "s"),
          ("c3_rise_fall", "Ab rise-fall", COL["di"], "^")]
    for j, (c, nm, col, mk) in enumerate(cl):
        for ts, ls, fill in (("dedup_stored", "-", True), ("a_alive_pooled", "--", False)):
            y = [summary["verdicts"][ts][c][v]["n"] for v in vlab]
            ax.plot(xs + (j - 1) * 0.2 + (0.06 if not fill else -0.06), y, ls="none", color=col, marker=mk, ms=4.5,
                    mfc=col if fill else "white", mew=1,
                    label=f"{nm} ({'stored' if ts == 'dedup_stored' else 'alive-pooled'} thr.)")
    ax.axhline(10, color="#6B7280", lw=0.7, ls=":")
    sep(ax)
    ax.set_xticks(xs, vtick, fontsize=5.5)
    ax.set_ylim(-0.5, 11.5)
    ax.set_ylabel("Census runs holding (of 11)")
    ax.set_title("a  Claims by variant", loc="left")
    ax.legend(fontsize=4.8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    # b, c: mean census curves, base vs variant abc (stored thresholds, 11 runs)
    for k, v in enumerate(("base", "variant_abc")):
        ax = fig.add_subplot(gs[0, 1 + k])
        x = tim[(tim.tset == "dedup_stored") & (tim.variant == v) & tim.evaluable]
        grid = np.linspace(0, 1, 11)
        for c in ("tg", "ab", "di"):
            Y = np.array([np.interp(grid, np.linspace(0, 1, len(s.split(";"))), [float(q) for q in s.split(";")])
                          for s in x[f"{c}_curve"]])
            m, sd = np.nanmean(Y, 0), np.nanstd(Y, 0)
            ax.plot(grid, m, color=COL[c], marker=MRK[c], ms=3, lw=1.2,
                    label={"tg": "Task-general (SAI)", "ab": "Superclass (SSI)", "di": "Class-selective (CSI)"}[c])
            ax.fill_between(grid, m - sd, m + sd, color=COL[c], alpha=0.15, lw=0)
            if np.isfinite(x[f"com_{c}"]).any():
                ax.axvline(np.nanmean(x[f"com_{c}"]), color=COL[c], lw=0.7, ls="--")
        ax.set_ylim(0, 0.8)
        ax.set_xlabel("Training (normalised checkpoint)")
        ax.set_ylabel("Share of alive census features")
        ax.set_title(f"{'b' if k == 0 else 'c'}  Census, {v} ({len(x)} runs)", loc="left")
        if k == 0:
            ax.legend(fontsize=5.5, frameon=False)
    # d, e: persistence ratios per run and variant (stored thresholds / Fig. 4a)
    p = per[per.tset == "dedup_stored"]
    runs = list(dict.fromkeys(p.run))
    for k, (col, nm) in enumerate((("ratio_tg_di", "tg"), ("ratio_ab_di", "ab"))):
        ax = fig.add_subplot(gs[1, k])
        for i, v in enumerate(vlab):
            y = p[p.variant == v].set_index("run").loc[runs, col].to_numpy()
            jit = np.linspace(-0.25, 0.25, len(y))
            ok = np.isfinite(y)
            ax.scatter((np.full(len(y), i) + jit)[ok], y[ok], s=9, marker=MRK[nm], color=COL[nm], alpha=0.85, lw=0)
            ax.scatter((np.full(len(y), i) + jit)[~ok], np.zeros((~ok).sum()), s=9, marker="x", color="#6B7280", lw=0.7)
            if ok.any():
                ax.hlines(np.nanmedian(y), i - 0.3, i + 0.3, color="#111827", lw=1)
        ax.axhline(1, color="#6B7280", lw=0.7, ls=":")
        sep(ax)
        ax.set_xticks(xs, vtick, fontsize=5.5)
        ax.set_ylabel(f"Lifespan ratio {'Tg' if nm == 'tg' else 'Ab'} / Di")
        ax.set_title(f"{'d' if k == 0 else 'e'}  {'Task-general' if nm == 'tg' else 'Superclass'} vs class-sel. lifespan",
                     loc="left")
    # f: fraction of features changing label
    ax = fig.add_subplot(gs[1, 2])
    h = summary["headline"]["dedup_stored"]
    ch = [h[v]["label_change_pooled"]["frac_changed_of_all"] for v in vlab[1:]]
    dr = [h[v]["label_change_pooled"]["frac_dropped"] for v in vlab[1:]]
    ax.bar(xs[1:], ch, color=COL["ab"], label="label changed")
    ax.bar(xs[1:], dr, bottom=ch, color="#D1D5DB", label="in dropped layer")
    sep(ax)
    ax.set_xticks(xs[1:], vtick[1:], fontsize=5.5)
    ax.set_ylabel("Fraction of alive census features")
    ax.set_title("f  Features relabelled (11 runs)", loc="left")
    ax.legend(fontsize=5.5, frameon=False)
    fig.text(0.01, -0.01, "Stored (Fig. 4) thresholds unless noted; dotted lines: rule (10/11) and ratio = 1; x at 0 in d/e: "
             "no lineage of that category (ratio undefined); variant_a(p05), variant_abc(p05): descriptive, not in the rule.", fontsize=5.5)
    for ext in ("png", "pdf"):
        fig.savefig(RL.OUT / f"fig_robust_census.{ext}", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
