#!/usr/bin/env python3
"""Per-feature follow-up on transformed features.

Needs the saved lane artefacts (sae_analysis/activations/{ckpt}/{layer}.npy and
sae_analysis/sae_results.json), which are not in the repo. Re-runs the pipeline's
own Phase 5 matching (packages/ml/src/saeanalysis.py) so feature pairs are
identical to the paper, then answers:

  1. Fate: what happens at transition t+1 to a feature that entered checkpoint
     t+1 through a transformed link, compared with stable-linked and born
     features? Does the "dying in progress" sequence
     (stable -> transformed -> died) occur more often than chance?
  2. Specificity: SSI / CSI / SAI of transformed features vs. stable and died
     ones (are transformed features unspecific?).
  3. Sensitivity: per-sample process counts (a vectorised, count-exact
     re-implementation of Phase 6) under three treatments of transformed
     pairs: excluded (= paper), turnover (A died + B born), continuity (treated
     as stable, with the same z-score rule). Reports the construction/refinement
     ratio and the Tg-E -> Ab-E -> Di-E peak ordering for each.

The baseline counts are checked against sae_results.json before anything else
is reported; a mismatch aborts that lane.

Usage (needs torch and scipy):
    python scripts/trace_transformed_features.py \\
        --data-root /data/rdt/core
    python scripts/trace_transformed_features.py \\
        --sae-dir <lane>/sae_analysis --label ResNet18-CIFAR100-seed42
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "ml" / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from saeanalysis import _apply_thresholds, _phase5_feature_matching, _safe_layer_filename  # noqa: E402
from compute_feature_survival import LANES  # noqa: E402

OUT_DIR = ROOT / "paper" / "data" / "transformed_trace"
OUT_MD = ROOT / "reports" / "transformed_trace.md"

ACTIVE_EPS = 1e-6
Z_DELTA_MIN = 0.5
PROCS = ("tg_h", "ab_h", "di_h", "as_h", "de_h")
TREATMENTS = ("excluded", "turnover", "continuity")


# loading

def discover(data_root):
    """Map the 9 standard lanes (compute_feature_survival.LANES) to sae_analysis dirs."""
    found = []
    for exp_id, lane_id, label in LANES:
        sae_dir = data_root / "experiments" / exp_id / "lanes" / lane_id / "sae_analysis"
        if (sae_dir / "sae_results.json").exists():
            found.append((label, sae_dir))
        else:
            print(f"  SKIP {label}: {sae_dir} not found")
    return found


def load_layer(sae_dir, labels, layer):
    act = {}
    for label in labels:
        p = sae_dir / "activations" / label / f"{_safe_layer_filename(layer)}.npy"
        if not p.exists():
            return None
        act[label] = {layer: np.load(p)}
    return act


# labelling

class Labeller:
    """Phase 6 labelling rules with the lane's adaptive thresholds."""

    def __init__(self, meta):
        t = meta["analysis_thresholds"]
        a = meta.get("adaptive_thresholds") or {}
        self.ssi_t = t["ssi_abstraction_thresh"]
        self.csi_t = t["csi_differentiation_thresh"]
        self.sai_t = t["sai_task_general_thresh"]
        self.null = {k: (a.get(f"{k}_null_mean", 0.0), max(a.get(f"{k}_null_std", 1.0), 1e-10))
                     for k in ("ssi", "csi", "sai")}

    def _z(self, k, v):
        m, s = self.null[k]
        return (v - m) / s

    def born(self, ssi, csi, sai):
        procs = ["as_h"]
        if sai > self.sai_t:
            procs.append("tg_h")
        elif ssi > self.ssi_t:
            procs.append("ab_h")
        if csi > self.csi_t:
            procs.append("di_h")
        return procs

    def stable(self, a, b):
        """a, b: (ssi, csi, sai) at checkpoint k and k+1."""
        dz = {k: self._z(k, b[i]) - self._z(k, a[i]) for i, k in enumerate(("ssi", "csi", "sai"))}
        procs = []
        if b[2] > self.sai_t and dz["sai"] > Z_DELTA_MIN:
            procs.append("tg_h")
        elif b[0] > self.ssi_t and dz["ssi"] > Z_DELTA_MIN:
            procs.append("ab_h")
        if b[1] > self.csi_t and dz["csi"] > Z_DELTA_MIN:
            procs.append("di_h")
        return procs


def _sel(selectivity, label, layer):
    s = selectivity.get(label, {}).get(layer, {})
    return (np.asarray(s.get("feature_ssi", [])), np.asarray(s.get("feature_csi", [])),
            np.asarray(s.get("feature_sai", [])))


def _at(arrs, i):
    return tuple(float(x[i]) if i < len(x) else 0.0 for x in arrs)


# analyses

def count_processes(m, H_a, H_b, sel_a, sel_b, lab, treatment):
    """Count-exact Phase 6: every (sample, feature) event, summed over samples."""
    act_a = np.abs(H_a) > ACTIVE_EPS
    act_b = np.abs(H_b) > ACTIVE_EPS
    alive_a, alive_b = m["alive_a"], m["alive_b"]
    n_a = act_a.sum(axis=0)
    n_b = act_b.sum(axis=0)

    stable = list(m["stable"])
    died = list(m["died"])
    born = list(m["born"])
    if treatment == "turnover":
        died += [i for i, _, _ in m["transformed"]]
        born += [j for _, j, _ in m["transformed"]]
    elif treatment == "continuity":
        stable += list(m["transformed"])

    c = Counter()
    for i, j, _ in stable:
        if not alive_a[i]:
            continue
        n = int((act_a[:, i] & act_b[:, j]).sum())
        if n == 0:
            continue
        c["stable"] += n
        for p in lab.stable(_at(sel_a, i), _at(sel_b, j)):
            c[p] += n
    for i in died:
        if alive_a[i]:
            c["de_h"] += int(n_a[i])
    for j in born:
        if alive_b[j] and n_b[j]:
            for p in lab.born(*_at(sel_b, j)):
                c[p] += int(n_b[j])
    return c


def fates(fm_layer, transitions):
    """How each B-side feature at transition t is classified as A-side at t+1."""
    origin_by_fate = {o: Counter() for o in ("stable", "transformed", "born")}
    seq = Counter()  # (link into t, link out of t) for A-side features at t
    for t0, t1 in zip(transitions[:-1], transitions[1:]):
        m0, m1 = fm_layer[t0], fm_layer[t1]
        nxt = {}
        for i, _, _ in m1["stable"]:
            nxt[i] = "stable"
        for i, _, _ in m1["transformed"]:
            nxt[i] = "transformed"
        for i in m1["died"]:
            nxt[i] = "died"
        origin = {}
        for _, j, _ in m0["stable"]:
            origin[j] = "stable"
        for _, j, _ in m0["transformed"]:
            origin[j] = "transformed"
        for j in m0["born"]:
            origin[j] = "born"
        for j, o in origin.items():
            f = nxt.get(j)
            if f is not None:
                origin_by_fate[o][f] += 1
                seq[(o, f)] += 1
    return origin_by_fate, seq


def selectivity_by_class(fm_layer, selectivity, label_a_of, layer):
    out = {k: {"ssi": [], "csi": [], "sai": []} for k in ("stable", "transformed", "died")}
    for t, m in fm_layer.items():
        ssi, csi, sai = _sel(selectivity, label_a_of[t], layer)
        groups = {"stable": [i for i, _, _ in m["stable"]],
                  "transformed": [i for i, _, _ in m["transformed"]],
                  "died": list(m["died"])}
        for k, idx in groups.items():
            for i in idx:
                if i < len(ssi):
                    out[k]["ssi"].append(float(ssi[i]))
                    out[k]["csi"].append(float(csi[i]))
                    out[k]["sai"].append(float(sai[i]))
    return out


def summarize_processes(per_t):
    rows = []
    for t, c in per_t.items():
        con = c["tg_h"] + c["ab_h"]
        tot = con + c["di_h"]
        rows.append({
            "transition": t, **{p: int(c[p]) for p in PROCS},
            "cr_ratio": con / c["di_h"] if c["di_h"] else None,
            "tg_frac": c["tg_h"] / tot if tot else None,
            "ab_frac": c["ab_h"] / tot if tot else None,
            "di_frac": c["di_h"] / tot if tot else None,
        })
    peak = {k: int(np.nanargmax([r[k] if r[k] is not None else np.nan for r in rows]))
            for k in ("tg_frac", "ab_frac", "di_frac")}
    return {
        "per_transition": rows,
        "cr_first": rows[0]["cr_ratio"], "cr_last": rows[-1]["cr_ratio"],
        "peak_transition": peak,
        "ordering_holds": peak["tg_frac"] <= peak["ab_frac"] <= peak["di_frac"],
    }


# driver

def analyze_lane(label, sae_dir):
    res = json.loads((sae_dir / "sae_results.json").read_text())
    meta = res["metadata"]
    labels, layers = meta["checkpoint_labels"], meta["layers"]
    selectivity = res["selectivity"]
    lab = Labeller(meta)
    _apply_thresholds(meta["analysis_thresholds"])

    transitions = [f"{a}->{b}" for a, b in zip(labels[:-1], labels[1:])]
    label_a_of = {t: t.split("->")[0] for t in transitions}
    counts = {tr: {t: Counter() for t in transitions} for tr in TREATMENTS}
    fate_tot = {o: Counter() for o in ("stable", "transformed", "born")}
    seq_tot = Counter()
    sel_tot = {k: {"ssi": [], "csi": [], "sai": []} for k in ("stable", "transformed", "died")}
    corr = {"transformed": [], "stable": []}
    mismatches = []

    for layer in layers:
        act = load_layer(sae_dir, labels, layer)
        if act is None:
            print(f"  {label}/{layer}: activations missing, skipped")
            continue
        fm = _phase5_feature_matching(act, {}, {}, labels, [layer], None)
        fm_layer = {t: fm[t][layer] for t in transitions if layer in fm.get(t, {})}

        for t, m in fm_layer.items():
            stored = res["feature_matching"].get(t, {}).get(layer, {})
            for k in ("n_stable", "n_born", "n_died", "n_transformed"):
                if stored and stored.get(k) != m[k]:
                    mismatches.append((t, layer, k, stored.get(k), m[k]))

        for t, m in fm_layer.items():
            a, b = t.split("->")
            H_a, H_b = act[a][layer], act[b][layer]
            sel_a, sel_b = _sel(selectivity, a, layer), _sel(selectivity, b, layer)
            for tr in TREATMENTS:
                counts[tr][t].update(count_processes(m, H_a, H_b, sel_a, sel_b, lab, tr))
            corr["transformed"] += [r for _, _, r in m["transformed"]]
            corr["stable"] += [r for _, _, r in m["stable"]]

        f, s = fates(fm_layer, [t for t in transitions if t in fm_layer])
        for o in f:
            fate_tot[o].update(f[o])
        seq_tot.update(s)
        sl = selectivity_by_class(fm_layer, selectivity, label_a_of, layer)
        for k in sl:
            for idx in sl[k]:
                sel_tot[k][idx] += sl[k][idx]
        del act, fm

    if mismatches:
        print(f"  {label}: {len(mismatches)} matching count mismatches vs sae_results.json, e.g. {mismatches[:3]}")
        return {"label": label, "error": "matching mismatch", "mismatches": mismatches[:50]}

    # Validate baseline process counts against stored process_intensity
    stored_pi = {r["transition"]: r for r in res.get("process_intensity", [])}
    pi_check = []
    for t in transitions:
        if t in stored_pi:
            pi_check.append({p: (stored_pi[t].get(p), int(counts["excluded"][t][p])) for p in PROCS})
    pi_ok = all(s == r for row in pi_check for s, r in row.values())

    def dist(c):
        n = sum(c.values())
        return {k: v / n for k, v in c.items()} if n else {}

    # Sequence stable -> transformed at t, then died at t+1. Compare
    # P(died | entered via transformed) with P(died | entered via stable).
    return {
        "label": label,
        "baseline_matches_stored_process_intensity": pi_ok,
        "process_intensity_check": pi_check,
        "fate_next_transition": {o: {"counts": dict(c), "fractions": dist(c)} for o, c in fate_tot.items()},
        "link_sequences": {f"{a}->{b}": v for (a, b), v in seq_tot.items()},
        "selectivity": {k: {idx: {"mean": float(np.mean(v)), "median": float(np.median(v)), "n": len(v)}
                            for idx, v in d.items() if v} for k, d in sel_tot.items()},
        "match_correlation": {k: {"mean": float(np.mean(v)), "n": len(v)} for k, v in corr.items() if v},
        "processes": {tr: summarize_processes(counts[tr]) for tr in TREATMENTS},
    }


def write_report(results):
    ok = [r for r in results if "error" not in r]
    lines = ["# Transformed features: per-feature trace", "",
             "Generated by `scripts/trace_transformed_features.py`.", ""]
    bad = [r["label"] for r in results if "error" in r]
    if bad:
        lines += [f"**Lanes aborted (matching did not reproduce stored counts):** {', '.join(bad)}", ""]
    unval = [r["label"] for r in ok if not r["baseline_matches_stored_process_intensity"]]
    if unval:
        lines += [f"**Warning:** baseline process counts differ from stored process_intensity for: "
                  f"{', '.join(unval)}. Inspect `process_intensity_check` before using section 3.", ""]

    lines += ["## 1. Fate at the next transition", "",
              "| Run | Entered via | → stable | → transformed | → died |", "|---|---|---|---|---|"]
    for r in ok:
        for o in ("stable", "transformed", "born"):
            f = r["fate_next_transition"][o]["fractions"]
            lines.append(f"| {r['label']} | {o} | {f.get('stable', 0):.1%} | {f.get('transformed', 0):.1%} "
                         f"| {f.get('died', 0):.1%} |")
    lines += ["", "## 2. Selectivity of A-side features by link class (mean)", "",
              "| Run | Class | SSI | CSI | SAI | n |", "|---|---|---|---|---|---|"]
    for r in ok:
        for k in ("stable", "transformed", "died"):
            s = r["selectivity"].get(k)
            if s:
                lines.append(f"| {r['label']} | {k} | {s['ssi']['mean']:.3f} | {s['csi']['mean']:.3f} "
                             f"| {s['sai']['mean']:.3f} | {s['ssi']['n']} |")
    lines += ["", "## 3. Process ordering under each treatment", "",
              "| Run | Treatment | C/R first | C/R last | Peak Tg / Ab / Di (transition) | Ordering holds |",
              "|---|---|---|---|---|---|"]
    fmt = lambda v: "∞" if v is None else f"{v:.2f}"
    for r in ok:
        for tr in TREATMENTS:
            p = r["processes"][tr]
            pk = p["peak_transition"]
            lines.append(f"| {r['label']} | {tr} | {fmt(p['cr_first'])} | {fmt(p['cr_last'])} "
                         f"| {pk['tg_frac']} / {pk['ab_frac']} / {pk['di_frac']} | {'yes' if p['ordering_holds'] else 'no'} |")
    OUT_MD.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", type=Path, help="Root containing experiments/<exp>/lanes/<lane>/sae_analysis")
    ap.add_argument("--sae-dir", type=Path, help="Single lane's sae_analysis directory")
    ap.add_argument("--label", default=None, help="Label for --sae-dir")
    args = ap.parse_args()

    if args.sae_dir:
        lanes = [(args.label or args.sae_dir.parent.name, args.sae_dir)]
    elif args.data_root:
        lanes = discover(args.data_root)
    else:
        ap.error("pass --data-root or --sae-dir")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for label, sae_dir in lanes:
        print(f"== {label}")
        r = analyze_lane(label, sae_dir)
        (OUT_DIR / f"{label}.json").write_text(json.dumps(r, indent=2))
        results.append(r)
    write_report(results)
    print(OUT_MD.read_text())


if __name__ == "__main__":
    main()
