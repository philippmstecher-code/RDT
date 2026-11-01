#!/usr/bin/env python3
"""Is the SSI (superclass-selective) census inflated by class-selective features?

Overlapping census (census_analyze.census_fracs; key 'overlapping' in the outputs), overlapping categories:
  Tg = SAI > t_sai;  Ab = SSI > t_ssi & ~Tg;  Di = CSI > t_csi   (Di may overlap Tg and Ab)
Exclusive census (this script; Fig. 4d, Extended Data Fig. 1d-f), same thresholds / alive rule / census layers:
  Tg    = SAI > t_sai
  Ab_ex = SSI > t_ssi & ~(CSI > t_csi) & ~Tg     (CSI NaN counts as not class-selective)
  Di_ex = CSI > t_csi & ~Tg                       (whatever SSI)
  both  = SSI > t_ssi & CSI > t_csi & ~Tg         (= overlapping-census Ab removed by exclusivity)
  tg_di = Tg & CSI > t_csi                        (= overlapping-census Di removed by exclusivity)
Threshold variants: dedup_stored (stored per-run thresholds; Fig. 4d) and a_alive_pooled
(alive-pooled null thresholds). Timing criteria are census_analyze.timing (COM order
Tg < Ab < Di = 'ordering reproduced'; argmax order; Ab rise-and-fall; all3), computed on the
same checkpoint set census_analyze used (incl. 'terminal') and, additionally, without 'terminal'.

Reads the census_compute.py scratch arrays (read-only). Writes
analysis_outputs/paper/CENSUS_exclusive/{census_exclusive_per_checkpoint.csv,
timing_exclusive.csv, fig4d_counts.csv, summary.json, fig4d_exclusive_compare.png}.
Usage: census_exclusive.py --scratch DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from census_analyze import ORDER, final_thr, load_lane, p95, pools, timing, write_csv  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "analysis_outputs" / "paper" / "CENSUS_robust"
OUT = REPO / "analysis_outputs" / "paper" / "CENSUS_exclusive"
FIG4_CACHE = REPO / "analysis_outputs" / "paper" / "figures" / "fig4_lifespan_data.json"
ARCHS = ("ResNet-18", "ViT-Small", "CCT-7")


def stack(m, lab) -> Dict[str, np.ndarray]:
    o: Dict[str, List[np.ndarray]] = {k: [] for k in ("ssi", "csi", "sai")}
    for L in m["census_layers"]:
        e = m["cell"][f"{lab}|{L}"]
        al = e["alive"]
        for k in o:
            o[k].append(e[k][al])
    return {k: np.concatenate(v).astype(np.float64) for k, v in o.items()}


def counts(S, t) -> Dict[str, int]:
    tg = S["sai"] > t["sai"]
    ss = S["ssi"] > t["ssi"]
    cs = S["csi"] > t["csi"]  # NaN -> False
    return {"n_alive": int(S["sai"].size), "n_tg": int(tg.sum()),
            "n_ab_overlap": int((ss & ~tg).sum()), "n_di_overlap": int(cs.sum()),
            "n_ab_ex": int((ss & ~cs & ~tg).sum()), "n_di_ex": int((cs & ~tg).sum()),
            "n_both": int((ss & cs & ~tg).sum()), "n_tg_di": int((tg & cs).sum()),
            "n_csi_nan": int(np.isnan(S["csi"]).sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    a = ap.parse_args()
    scratch = Path(a.scratch)
    OUT.mkdir(parents=True, exist_ok=True)
    lanes = []
    for p in sorted((SRC / "per_lane").glob("*_meta.json")):
        m = load_lane(p, scratch)
        st = m["stored_thresholds"]
        m["thr"] = {"dedup_stored": {"sai": st["sai_task_general_thresh"], "ssi": st["ssi_abstraction_thresh"],
                                  "csi": st["csi_differentiation_thresh"]}}
        P = pools(m, m["labels"], m["layers"], True)
        m["thr"]["a_alive_pooled"] = final_thr({k: p95(P[k]) for k in ("ssi", "csi", "sai")})
        rows = []
        for lab in m["labels"]:
            S = stack(m, lab)
            for var, t in m["thr"].items():
                rows.append({"variant": var, "checkpoint": lab, **counts(S, t)})
        m["rows"] = rows
        for e in m["cell"].values():  # free null arrays
            for k in ("ssi_null", "csi_null", "csi_keep", "sai_snull"):
                e.pop(k, None)
        lanes.append(m)
        print(m["name"], "done", flush=True)
    lanes.sort(key=lambda m: (ORDER[m["group"]], m["arch"], m["name"]))

    # validation against the overlapping census table (census_analyze.py)
    import csv
    pub = {(r["variant"], r["lane"], r["checkpoint"]): r for r in csv.DictReader(open(SRC / "tables" / "census_per_checkpoint.csv"))}
    maxdiff = 0.0

    per_ckpt, tim = [], []
    for m in lanes:
        for r in m["rows"]:
            n = r["n_alive"]
            ref = pub[(r["variant"], m["name"], r["checkpoint"])]
            for k, rk in (("n_tg", "frac_tg"), ("n_ab_overlap", "frac_ab"), ("n_di_overlap", "frac_di")):
                maxdiff = max(maxdiff, abs(r[k] / n - float(ref[rk])))
            per_ckpt.append({"variant": r["variant"], "group": m["group"], "arch": m["arch"], "lane": m["name"],
                             "checkpoint": r["checkpoint"], **r,
                             "frac_tg": r["n_tg"] / n, "frac_ab_overlap": r["n_ab_overlap"] / n, "frac_ab_ex": r["n_ab_ex"] / n,
                             "frac_di_overlap": r["n_di_overlap"] / n, "frac_di_ex": r["n_di_ex"] / n, "frac_both": r["n_both"] / n,
                             "share_ab_pub_also_csi": r["n_both"] / r["n_ab_overlap"] if r["n_ab_overlap"] else np.nan,
                             "share_di_pub_also_ssi": r["n_both"] / r["n_di_overlap"] if r["n_di_overlap"] else np.nan,
                             "share_di_pub_in_tg": r["n_tg_di"] / r["n_di_overlap"] if r["n_di_overlap"] else np.nan,
                             **{f"thr_{k}": v for k, v in m["thr"][r["variant"]].items()}})
        for var in m["thr"]:
            rr = [r for r in m["rows"] if r["variant"] == var]
            for ckset in ("with_terminal", "no_terminal"):
                rs = rr if ckset == "with_terminal" else [r for r in rr if r["checkpoint"] != "terminal"]
                n = np.array([r["n_alive"] for r in rs], float)
                tg = np.array([r["n_tg"] for r in rs]) / n
                for defn, ab_k, di_k in (("overlapping", "n_ab_overlap", "n_di_overlap"), ("exclusive", "n_ab_ex", "n_di_ex"),
                                         ("ab_ex_only", "n_ab_ex", "n_di_overlap")):
                    f = {"tg": list(tg), "ab": list(np.array([r[ab_k] for r in rs]) / n),
                         "di": list(np.array([r[di_k] for r in rs]) / n)}
                    t = timing(f)
                    tim.append({"variant": var, "checkpoints": ckset, "definition": defn, "group": m["group"],
                                "arch": m["arch"], "lane": m["name"], "n_ckpt": len(rs), **t,
                                "ab_curve": ";".join(f"{v:.4f}" for v in f["ab"]),
                                "di_curve": ";".join(f"{v:.4f}" for v in f["di"])})
    assert maxdiff < 1e-6, maxdiff
    write_csv(OUT / "census_exclusive_per_checkpoint.csv", per_ckpt)
    write_csv(OUT / "timing_exclusive.csv", tim)

    # ordering counts
    cnt = []
    for var in ("dedup_stored", "a_alive_pooled"):
        for ckset in ("with_terminal", "no_terminal"):
            for defn in ("overlapping", "exclusive", "ab_ex_only"):
                rr = [r for r in tim if (r["variant"], r["checkpoints"], r["definition"]) == (var, ckset, defn)]
                cnt.append({"variant": var, "checkpoints": ckset, "definition": defn, "n_runs": len(rr),
                            **{f"n_{k}": int(sum(r[k] for r in rr)) for k in
                               ("order_com", "order_argmax", "ab_rise_fall", "tg_peak_init", "all3")},
                            "fail_order_com": ";".join(r["lane"] for r in rr if not r["order_com"]),
                            "fail_all3": ";".join(r["lane"] for r in rr if not r["all3"])})
    write_csv(OUT / "ordering_counts.csv", cnt)

    # overlap summary: share of overlapping-census Ab that is also CSI>t, early / mid / late (real checkpoints)
    ov = []
    for var in ("dedup_stored", "a_alive_pooled"):
        for m in lanes:
            rs = [r for r in per_ckpt if r["variant"] == var and r["lane"] == m["name"] and r["checkpoint"] != "terminal"]
            K = len(rs)
            seg = {"early": rs[1:max(2, K // 3)], "mid": rs[K // 3:2 * K // 3], "late": rs[2 * K // 3:]}
            o = {"variant": var, "group": m["group"], "arch": m["arch"], "lane": m["name"]}
            for s, g in seg.items():
                ab = sum(r["n_ab_overlap"] for r in g)
                o[f"share_ab_also_csi_{s}"] = sum(r["n_both"] for r in g) / ab if ab else np.nan
            ab = sum(r["n_ab_overlap"] for r in rs)
            o["share_ab_also_csi_all"] = sum(r["n_both"] for r in rs) / ab if ab else np.nan
            o["share_ab_also_csi_last"] = rs[-1]["share_ab_pub_also_csi"]
            pk = int(np.argmax([r["frac_ab_overlap"] for r in rs]))
            o["ab_pub_peak_ckpt"], o["share_ab_also_csi_at_ab_peak"] = rs[pk]["checkpoint"], rs[pk]["share_ab_pub_also_csi"]
            o["ab_pub_max_frac"] = max(r["frac_ab_overlap"] for r in rs)
            o["ab_pub_last_over_max"] = rs[-1]["frac_ab_overlap"] / o["ab_pub_max_frac"] if o["ab_pub_max_frac"] else np.nan
            o["ab_ex_max_frac"] = max(r["frac_ab_ex"] for r in rs)
            o["ab_ex_peak_ckpt"] = rs[int(np.argmax([r["frac_ab_ex"] for r in rs]))]["checkpoint"]
            o["ab_ex_last_over_max"] = rs[-1]["frac_ab_ex"] / o["ab_ex_max_frac"] if o["ab_ex_max_frac"] else np.nan
            o["share_di_pub_in_tg_all"] = (sum(r["n_tg_di"] for r in rs) / sum(r["n_di_overlap"] for r in rs))
            o["di_pub_last_frac"], o["di_ex_last_frac"] = rs[-1]["frac_di_overlap"], rs[-1]["frac_di_ex"]
            ov.append(o)
    write_csv(OUT / "overlap_by_stage.csv", ov)

    # Fig. 4d: stored thresholds, 9 core runs, terminal excluded, mean over seeds
    fig4 = json.loads(FIG4_CACHE.read_text()) if FIG4_CACHE.exists() else None
    f4rows, f4val = [], []
    stacks: Dict[str, Dict[str, np.ndarray]] = {}
    for arch in ARCHS:
        ms = [m for m in lanes if m["group"] == "core" and m["arch"] == arch]
        arrs = {}
        for defn, keys in (("overlapping", ("n_tg", "n_ab_overlap", "n_di_overlap")), ("exclusive", ("n_tg", "n_ab_ex", "n_di_ex"))):
            per = [np.array([[r[k] for k in keys] for r in m["rows"] if r["variant"] == "dedup_stored" and r["checkpoint"] != "terminal"], float)
                   for m in ms]
            arrs[defn] = np.mean(per, axis=0)
        both = np.mean([np.array([r["n_both"] for r in m["rows"] if r["variant"] == "dedup_stored" and r["checkpoint"] != "terminal"], float)
                        for m in ms], axis=0)
        stacks[arch] = arrs
        for i in range(len(both)):
            f4rows.append({"arch": arch, "checkpoint": i, **{f"{d}_{c}": float(arrs[d][i, j]) for d in arrs for j, c in enumerate(("tg", "ab", "di"))},
                           "both": float(both[i]), "share_ab_also_csi": float(both[i] / arrs["overlapping"][i, 1]) if arrs["overlapping"][i, 1] else np.nan,
                           "overlapping_total": float(arrs["overlapping"][i].sum()), "exclusive_total": float(arrs["exclusive"][i].sum())})
        if fig4:
            for r in fig4["runs"]:
                if r["arch"] != arch:
                    continue
                m = [m for m in ms if m["name"].endswith(f"seed{r['seed']}")][0]
                mine = [(x["n_tg"], x["n_ab_overlap"], x["n_di_overlap"]) for x in m["rows"] if x["variant"] == "dedup_stored" and x["checkpoint"] != "terminal"]
                f4val.append({"arch": arch, "seed": r["seed"], "exact_match": [list(t) for t in mine] == r.get("census_overlap", r["census"])})
    write_csv(OUT / "fig4d_counts.csv", f4rows)

    # figure: Fig. 4d overlapping vs exclusive
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["#4C78A8", "#F58518", "#54A24B"]
    fig, axes = plt.subplots(3, 2, figsize=(7, 6), sharex=True)
    for i, arch in enumerate(ARCHS):
        ymax = max(stacks[arch][d].sum(1).max() for d in stacks[arch]) * 1.05
        for j, d in enumerate(("overlapping", "exclusive")):
            c = stacks[arch][d]
            xs = np.linspace(0, 10, len(c))
            ax = axes[i, j]
            ax.stackplot(xs, c[:, 0], c[:, 1], c[:, 2], colors=cols, alpha=0.8, labels=["SAI (Tg)", "SSI (Ab)", "CSI (Di)"])
            ax.set_ylim(0, ymax)
            ax.set_title(f"{arch} - {d}", fontsize=9)
    axes[0, 0].legend(fontsize=7, loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("Training checkpoint (0-10)")
    fig.tight_layout()
    fig.savefig(OUT / "fig4d_exclusive_compare.png", dpi=150)

    summary: Dict[str, Any] = {
        "definitions": {"overlapping": "Tg=SAI>t_sai; Ab=SSI>t_ssi & ~Tg; Di=CSI>t_csi (overlapping: Ab&Di and Tg&Di allowed)",
                        "exclusive": "Tg=SAI>t_sai; Ab=SSI>t_ssi & ~(CSI>t_csi) & ~Tg; Di=CSI>t_csi & ~Tg",
                        "ab_ex_only": "exclusive Ab, overlapping-census Di"},
        "validation_maxabs_frac_vs_CENSUS_robust": maxdiff,
        "validation_fig4d_counts_exact": f4val,
        "ordering_counts": cnt,
        "overlap_by_stage": ov,
        "thresholds": {m["name"]: m["thr"] for m in lanes},
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=lambda x: None if isinstance(x, float) and np.isnan(x) else float(x))
    for r in cnt:
        print({k: r[k] for k in r if not k.startswith("fail")}, r["fail_order_com"], "|", r["fail_all3"])


if __name__ == "__main__":
    main()
