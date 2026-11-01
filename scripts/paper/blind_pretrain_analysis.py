#!/usr/bin/env python3
"""Superclass-blind pre-training (blind_pretrain): does any pre-training of equal length give the curriculum's head
start, or is it the coarse structure that helps? Outcome per measure: 'coarse structure helps' if the
95% CIs of blind - true and blind - random both lie below 0, 'generic transfer' if both include 0, else 'mixed'.

Arms (ResNet-18, CIFAR-100, seeds 7, 42, 137, 256, 314, same initial weights per seed): sc_blind_K20 (20 epochs on the
five-way within-superclass-index task, then 30 fine-class epochs), sc_true_K20, sc_rand_K20 and scratch_E30
(curriculum).

Main      accuracy lead after one fine-class epoch: P(fine) at fine epoch 1 (stored per-image
          test predictions) mapped onto the sham reference of the same seed (scratch_proxy_sham rows of
          CURRICULUM_routing/factors_per_checkpoint.csv) by linear interpolation over its running maximum (epoch 0 at
          chance, 1%), minus the fine epoch. The function is checked against Source Data fig6_c_lead.csv for the
          existing arms before it is applied to the blind arm.
Further   (1) true-superclass selectivity at the switch: superclass-selective SAE features (rdt_pace_curriculum.census,
          scratch thresholds) and a 20-way true-superclass probe on pooled features (curriculum_probe.probe);
          (2) confined class features at fine epochs 1, 5, 10, 15 and epochs ahead of scratch in confined class
          features at fine epochs 1, 5 (as rdt_pace_curriculum.py; existing arms from RDT_pace/curriculum_long.csv);
          (3) final fine-class validation accuracy (fine epoch 30).
Contrasts blind - true and blind - random, paired over seeds, mean, t-based 95% CI, seeds below 0.
Writes long.csv, summary.json, REPORT.md in analysis_outputs/paper/BLIND_PRETRAIN/.
Usage: python scripts/paper/blind_pretrain_analysis.py
"""
import json
import os
import sys
from datetime import date
from pathlib import Path
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lineage"))
sys.path.insert(0, str(HERE))
import rdt_pace_curriculum as R  # noqa: E402
from curriculum_probe import SUPER_OF, load, probe  # noqa: E402

torch.set_num_threads(2)
REPO = HERE.parents[1]
RESULTS = REPO / "analysis_outputs/paper"
OUT = RESULTS / "BLIND_PRETRAIN"
E = Path(RDT_DATA + "/extended/experiments")
CUR = E / "curriculum/lanes"
BLIND = E / "blind_pretrain/lanes"
SEEDS = [7, 42, 137, 256, 314]
K = 20
LANE = {"blind": lambda s: BLIND / f"sc_blind_K20_seed{s}", "true": lambda s: CUR / f"sc_true_K20_seed{s}",
        "rand": lambda s: CUR / f"sc_rand_K20_seed{s}", "scratch": lambda s: CUR / f"scratch_E30_seed{s}"}
FACTORS = RESULTS / "CURRICULUM_routing/factors_per_checkpoint.csv"
FIG6C = RESULTS / "figures/final/source_data/fig6_c_lead.csv"
RDT_LONG = RESULTS / "RDT_pace/curriculum_long.csv"
PROXY = "scratch_proxy_sham"
LEAD_EPOCHS = (1, 5, 10, 15, 20)
RDT_EPOCHS = (0, 1, 5, 10, 15)


def tci(x):
    """Mean, t-based 95% CI over the finite per-seed values (as lineage_lib.tci; NaN = outside scratch's range)."""
    x = np.asarray(x, float)
    d = x[np.isfinite(x)]
    m = float(d.mean()) if len(d) else float("nan")
    h = float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    return {"mean": m, "ci95": [m - h, m + h], "n": int(len(d)), "n_below_0": int((d < 0).sum()),
            "n_above_0": int((d > 0).sum()), "per_seed": dict(zip(SEEDS, np.round(x, 4).tolist()))}


def snapshots(ld):
    s = json.load(open(ld / "dev_snapshots/metrics/devtrain_snapshots.json"))
    s = s.get("snapshots", s) if isinstance(s, dict) else s
    return [x for x in s if x.get("snapshot_type") in ("intermediate", "t0_baseline")]


def lane_checks(ld):
    cfg = json.load(open(ld / "_config/lane.json"))
    ph = cfg["curriculum"]["phases"]
    assert [p["label_mode"] for p in ph] == ["superclass", "fine"] and ph[0]["end_epoch"] == K, ph
    assert ph[1]["start_epoch"] == K + 1 and ph[1]["end_epoch"] == 50 and cfg["epochs"] == 50, cfg
    assert all(p["learning_rate"] == 0.1 for p in ph), ph
    assert (ld / ".lane_done").exists(), ld
    return cfg["curriculum"].get("coarse_labels")


def p_fine(ms, val):
    """P(fine) from stored per-image test predictions (curriculum_routing.checkpoint_rows), checked against the
    snapshot's validation accuracy."""
    sp = torch.load(ms / "sample_predictions.pt", map_location="cpu", weights_only=False)
    n = sum(len(sp[c]["predictions"]) for c in range(100))
    k = sum(int((sp[c]["predictions"].numpy() == c).sum()) for c in range(100))
    assert n == 10000 and abs(100 * k / n - val) < 0.015, (ms, k, n, val)
    return k / n


def lead(v, ref, fe, chance=0.01):
    """Epoch at which the reference reaches v (running maximum, linear interpolation) minus fe."""
    x = np.r_[0, ref.index.values]
    y = np.maximum.accumulate(np.r_[chance, ref.values])
    return float(np.interp(v, y, x)) - fe


def accuracy_leads():
    F = pd.read_csv(FACTORS)
    rows = []
    for s in SEEDS:
        ref = F[(F.lane == PROXY) & (F.seed == s)].set_index("fine_epoch").sort_index()["p_fine"]
        for arm in ("true", "rand"):
            g = F[(F.design == "K20") & (F.arm == arm) & (F.seed == s)].set_index("fine_epoch")["p_fine"]
            for fe in LEAD_EPOCHS:
                rows.append({"arm": arm, "seed": s, "fine_epoch": fe, "p_fine": float(g[fe]),
                             "lead_p_fine": lead(float(g[fe]), ref, fe)})
        ld = LANE["blind"](s)
        for x in snapshots(ld):
            fe = int(x["epoch"]) - K
            if fe in LEAD_EPOCHS:
                v = p_fine(ld / "dev_snapshots" / f"milestone_{x['milestone_index']}", float(x["val_accuracy"]))
                rows.append({"arm": "blind", "seed": s, "fine_epoch": fe, "p_fine": v, "lead_p_fine": lead(v, ref, fe)})
    A = pd.DataFrame(rows)
    # the lead function reproduces the stored lead Source Data (fig6_c_lead.csv) for the existing arms
    C = pd.read_csv(FIG6C)
    C["arm"] = C.arm.map({"sc_true": "true", "sc_rand": "rand"})
    m = A.merge(C[["arm", "seed", "fine_epoch", "lead_p_fine"]], on=["arm", "seed", "fine_epoch"], suffixes=("", "_fig6"))
    assert len(m) == 2 * len(SEEDS) * len(LEAD_EPOCHS), len(m)
    assert np.allclose(m.lead_p_fine, m.lead_p_fine_fig6, atol=1e-9), m
    return A


def rdt_rows():
    D = pd.read_csv(RDT_LONG)
    D = D[D.arm.isin(["true", "rand", "scratch"])]
    rows = []
    for s in SEEDS:
        t = R.thresholds(s)
        d = LANE["blind"](s)
        for mi, ep in sorted(R.snaps(d).items()):
            fe = ep - K
            if fe not in RDT_EPOCHS or not (d / "sae_analysis" / "activations" / str(mi)).exists():
                continue
            n_sc, n_cl, n_conf = R.census(d, mi, t)
            rows.append({"arm": "blind", "seed": s, "fine_epoch": fe, "milestone": mi, "n_sc": n_sc, "n_cl": n_cl,
                         "n_cl_confined": n_conf, "share_confined": n_conf / n_cl if n_cl else np.nan})
        print("rdt blind", s, [r for r in rows if r["seed"] == s], flush=True)
    return pd.concat([D, pd.DataFrame(rows)], ignore_index=True)


def confined_leads(D):
    """rdt_pace_curriculum.main: scratch trajectory of confined class features, running maximum, interpolation."""
    out = []
    for s in SEEDS:
        sc = D[(D.arm == "scratch") & (D.seed == s)].sort_values("fine_epoch")
        x, y = sc.fine_epoch.values.astype(float), np.maximum.accumulate(sc.n_cl_confined.values.astype(float))
        for arm in ("true", "rand", "blind"):
            for fe in (1, 5):
                v = float(D[(D.arm == arm) & (D.seed == s) & (D.fine_epoch == fe)].n_cl_confined.iloc[0])
                eq = float(np.interp(v, y, x)) if v <= y[-1] else np.nan
                out.append({"arm": arm, "seed": s, "fine_epoch": fe, "epochs_ahead_confined": eq - fe})
    return pd.DataFrame(out)


def switch_probe():
    rows = []
    for s in SEEDS:
        for arm in ("true", "rand", "blind"):
            ld = LANE[arm](s)
            x = [x for x in snapshots(ld) if int(x["epoch"]) == K]
            assert len(x) == 1, (arm, s)
            X, y = load(ld / "dev_snapshots" / f"milestone_{x[0]['milestone_index']}", "avgpool")
            rows.append({"arm": arm, "seed": s, "sc_probe_switch": 100 * float(probe(X.astype(np.float64), SUPER_OF[y])),
                         "coarse_val_acc_switch": float(x[0]["val_accuracy"])})
        print("probe", s, rows[-3:], flush=True)
    return pd.DataFrame(rows)


def final_acc():
    rows = []
    for s in SEEDS:
        for arm in ("true", "rand", "blind", "scratch"):
            m = json.load(open(LANE[arm](s) / "dev_snapshots/metrics/devtrain_metrics.json"))
            c = {int(e["epoch"]): float(e["val_accuracy"]) for e in m}
            rows.append({"arm": arm, "seed": s, "final_val_acc": c[50 if arm != "scratch" else 30],
                         "fine_epoch1_val_acc": c[K + 1] if arm != "scratch" else c[1]})
    return pd.DataFrame(rows)


def per_seed(T, arm, col, **sel):
    q = T[T.arm == arm]
    for k, v in sel.items():
        q = q[q[k] == v]
    return q.set_index("seed")[col].loc[SEEDS].values.astype(float)


def verdict(c_true, c_rand):
    below = [c["ci95"][1] < 0 for c in (c_true, c_rand)]
    incl = [c["ci95"][0] <= 0 <= c["ci95"][1] for c in (c_true, c_rand)]
    if all(below):
        return "coarse structure helps"
    if all(incl):
        return "generic transfer"
    return "mixed"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for s in SEEDS:
        assert lane_checks(LANE["blind"](s)) == "within_superclass_index", s
        assert (LANE["blind"](s) / ".sae_done").exists(), f"SAE not finished for seed {s}"
    A = accuracy_leads()
    D = rdt_rows()
    CL = confined_leads(D)
    P = switch_probe()
    FA = final_acc()
    long = pd.concat([A.assign(table="accuracy_lead"), D.assign(table="rdt_census"),
                      CL.assign(table="confined_lead"), P.assign(table="switch"), FA.assign(table="final")],
                     ignore_index=True)
    long.to_csv(OUT / "long.csv", index=False)

    res = {}
    measures = {
        "primary|accuracy lead, fine epoch 1": (A, "lead_p_fine", {"fine_epoch": 1}),
        "secondary|confined-feature lead, fine epoch 1": (CL, "epochs_ahead_confined", {"fine_epoch": 1}),
        "secondary|confined-feature lead, fine epoch 5": (CL, "epochs_ahead_confined", {"fine_epoch": 5}),
        "secondary|superclass-selective SAE features at the switch": (D, "n_sc", {"fine_epoch": 0}),
        "secondary|true-superclass probe at the switch (%)": (P, "sc_probe_switch", {}),
        "secondary|final fine-class accuracy (%)": (FA, "final_val_acc", {}),
    }
    for fe in (1, 5, 10, 15):
        measures[f"secondary|confined class features, fine epoch {fe}"] = (D, "n_cl_confined", {"fine_epoch": fe})
        measures[f"secondary|share confined, fine epoch {fe}"] = (D, "share_confined", {"fine_epoch": fe})
    for fe in (5, 10, 15, 20):
        measures[f"descriptive|accuracy lead, fine epoch {fe}"] = (A, "lead_p_fine", {"fine_epoch": fe})
    measures["descriptive|five-/twenty-way accuracy at the end of pre-training (%)"] = (P, "coarse_val_acc_switch", {})
    for name, (T, col, sel) in measures.items():
        lv = {arm: per_seed(T, arm, col, **sel) for arm in ("blind", "true", "rand") if (T.arm == arm).any()}
        r = {"levels": {a: tci(v) for a, v in lv.items()}}
        if "blind" in lv:
            r["blind - true"] = tci(lv["blind"] - lv["true"])
            r["blind - rand"] = tci(lv["blind"] - lv["rand"])
        res[name] = r
    prim = res["primary|accuracy lead, fine epoch 1"]
    conf = res["secondary|confined-feature lead, fine epoch 1"]
    v_prim = verdict(prim["blind - true"], prim["blind - rand"])
    v_conf = verdict(conf["blind - true"], conf["blind - rand"])
    summary = {"date": str(date.today()), "seeds": SEEDS, "verdict_primary": v_prim,
               "confined_lead_same_rule": v_conf, "measures": res}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=float)

    f = lambda c: f"{c['mean']:+.2f} (CI {c['ci95'][0]:+.2f} to {c['ci95'][1]:+.2f}; {c['n_below_0']}/{c['n']} below 0)"
    lv = lambda c: f"{c['mean']:.2f}"
    L = [f"# Superclass-blind pre-training: results ({date.today()})", "",
         "Classification: 'coarse structure helps' if the 95% CIs of blind - true and blind - random both lie below 0, "
         "'generic transfer' if both include 0, else 'mixed'. Script: "
         "`scripts/paper/blind_pretrain_analysis.py`. Seeds 7, 42, 137, 256, 314; paired over seeds, t-based 95% CI.",
         "", f"**Outcome (primary measure): {v_prim}.**", "",
         f"Confined-feature lead at fine epoch 1, same classification (reported beside it): {v_conf}.", "",
         "Blind arm's own accuracy lead over scratch after one fine-class epoch: "
         f"{f(prim['levels']['blind'])} (CI against 0).", "",
         "| measure | blind | true superclasses | random grouping | blind − true | blind − random |",
         "|---|---|---|---|---|---|"]
    for name, r in res.items():
        L.append(f"| {name} | {lv(r['levels']['blind'])} | {lv(r['levels']['true'])} | {lv(r['levels']['rand'])} | "
                 f"{f(r['blind - true'])} | {f(r['blind - rand'])} |")
    L += ["", "## Per-seed values", "", "| measure | arm | " + " | ".join(f"seed {s}" for s in SEEDS) + " |",
          "|---|---|" + "---|" * len(SEEDS)]
    for name, r in res.items():
        if name.startswith("descriptive|accuracy lead"):
            continue
        for a, c in list(r["levels"].items()) + [("blind − true", r["blind - true"]), ("blind − random", r["blind - rand"])]:
            L.append(f"| {name} | {a} | " + " | ".join(f"{c['per_seed'][s]:.3f}" for s in SEEDS) + " |")
    L += ["", "Checks: lane configs (20 + 30 epochs, lr 0.1, coarse_labels = within_superclass_index); per-image "
          "predictions reproduce each snapshot's validation accuracy; the lead function reproduces the stored lead Source Data "
          "(fig6_c_lead.csv) for the existing arms at fine epochs 1, 5, 10, 15, 20.",
          "Scratch, final accuracy (fine epoch 30): " + ", ".join(
              f"{v:.2f}" for v in per_seed(FA, "scratch", "final_val_acc"))]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
