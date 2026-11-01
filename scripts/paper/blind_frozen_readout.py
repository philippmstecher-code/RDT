#!/usr/bin/env python3
"""Superclass-blind pre-training (blind_pretrain, arm sc_blind_K20): frozen read-out for Extended Data Fig. 9d.

Same read-out as CURRICULUM_probe/probe.csv (curriculum_probe.py, imported): pooled layer (avgpool) of the stored checkpoint
activations (individual_multilayer_activations.pt, 100 test images per class), StandardScaler + RidgeClassifier
(alpha=10), 5-fold stratified CV (shuffle, random_state=0), at every stored milestone (epochs 0, 5, ..., 20 = the
switch, 21, 25, ..., 50). Columns as probe.csv: fine100 (100-way), within_sc (5-way inside each true superclass,
mean over 20), super20 (20-way true superclass), val_acc.

Modes
  --repro             recompute sc_rand_K20 seed 42 (curriculum) and compare with probe.csv; exit 1 on mismatch
  --seeds 7 42 ...    compute every ready, not yet computed blind milestone of these seeds; a milestone is ready when
                      it is listed in devtrain_snapshots.json and its activations file is older than 60 s
  --report            only rewrite frozen_readout.md from frozen_readout.csv
Rows accumulate in analysis_outputs/paper/BLIND_PRETRAIN/frozen_readout.csv (computed milestones are skipped);
frozen_readout.md is rewritten after every run (t-based 95% CIs over seeds; paired blind - true and blind - random).
Usage: python scripts/paper/blind_frozen_readout.py --repro | --seeds 7 42 137 256 314 | --report
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from curriculum_probe import SUPER_OF, load, probe  # noqa: E402

import torch  # noqa: E402

torch.set_num_threads(2)
REPO = HERE.parents[1]
RESULTS = REPO / "analysis_outputs" / "paper"
PROBE = RESULTS / "CURRICULUM_probe" / "probe.csv"
OUT = RESULTS / "BLIND_PRETRAIN"
CSV = OUT / "frozen_readout.csv"
MD = OUT / "frozen_readout.md"
E = Path(RDT_DATA + "/extended/experiments")
BLIND = E / "blind_pretrain" / "lanes"
CUR = E / "curriculum" / "lanes"
SEEDS = [7, 42, 137, 256, 314]
K = 20
COLS = ["arm", "seed", "epoch", "layer", "fine100", "within_sc", "super20", "val_acc"]
LAYER = "avgpool"


def snaps(lane):
    s = json.load(open(lane / "dev_snapshots" / "metrics" / "devtrain_snapshots.json"))
    return s.get("snapshots", s) if isinstance(s, dict) else s


def ready(lane, s):
    f = lane / "dev_snapshots" / f"milestone_{s['milestone_index']}" / "individual_multilayer_activations.pt"
    return f.exists() and time.time() - f.stat().st_mtime > 60


def row(lane, s):
    """curriculum_probe.main, one milestone, one layer (same load, probe and arm/seed parsing)."""
    arm, seed = lane.name.rsplit("_", 2)[0], int(lane.name.rsplit("seed", 1)[1])
    X, y = load(lane / "dev_snapshots" / f"milestone_{s['milestone_index']}", LAYER)
    ysc = SUPER_OF[y]
    within = np.mean([probe(X[ysc == g], y[ysc == g]) for g in range(20)])
    return dict(arm=arm, seed=seed, epoch=s["epoch"], layer=LAYER, fine100=probe(X, y), within_sc=within,
                super20=probe(X, ysc), val_acc=s["val_accuracy"])


def repro():
    lane = CUR / "sc_rand_K20_seed42"
    ref = pd.read_csv(PROBE)
    ref = ref[(ref.arm == "sc_rand") & (ref.seed == 42) & (ref.layer == LAYER)].drop_duplicates("epoch").set_index("epoch")
    seen, worst = set(), 0.0
    for s in snaps(lane):
        if s["epoch"] in seen:
            continue
        seen.add(s["epoch"])
        r = row(lane, s)
        d = {c: abs(r[c] - ref.loc[s["epoch"], c]) for c in ("fine100", "within_sc", "super20", "val_acc")}
        worst = max(worst, *d.values())
        print(f"repro sc_rand seed 42 epoch {s['epoch']:>2}: fine100 {r['fine100']:.4f} (stored "
              f"{ref.loc[s['epoch'], 'fine100']:.4f}), within_sc {r['within_sc']:.4f} (stored "
              f"{ref.loc[s['epoch'], 'within_sc']:.4f}), super20 {r['super20']:.4f} (stored "
              f"{ref.loc[s['epoch'], 'super20']:.4f}); max |diff| {max(d.values()):.2e}", flush=True)
    assert len(seen) == len(ref), (sorted(seen), ref.index.tolist())
    ok = worst < 5e-4
    print(f"REPRO {'PASS' if ok else 'FAIL'}: {len(seen)} milestones, max |diff| {worst:.2e} (tolerance 5e-4)", flush=True)
    return ok


def compute(seeds):
    old = pd.read_csv(CSV) if CSV.exists() else pd.DataFrame(columns=COLS)
    have = Counter(zip(old.seed.astype(int), old.epoch.astype(int)))   # rows per (seed, epoch); the final snapshot
    new = []                                                            # repeats epoch 50, as in probe.csv
    for sd in seeds:
        lane = BLIND / f"sc_blind_K20_seed{sd}"
        if not (lane / "dev_snapshots" / "metrics" / "devtrain_snapshots.json").exists():
            print(f"seed {sd}: no snapshots yet", flush=True)
            continue
        occ = Counter()
        for s in snaps(lane):
            occ[s["epoch"]] += 1
            if occ[s["epoch"]] <= have[(sd, s["epoch"])]:
                continue
            if not ready(lane, s):
                print(f"seed {sd} milestone {s['milestone_index']} (epoch {s['epoch']}): not ready", flush=True)
                continue
            r = row(lane, s)
            new.append(r)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    if new:
        OUT.mkdir(parents=True, exist_ok=True)
        d = pd.concat([old, pd.DataFrame(new)], ignore_index=True) if len(old) else pd.DataFrame(new)
        d = d.sort_values(["seed", "epoch"], kind="stable")[COLS].reset_index(drop=True)
        d.to_csv(CSV, index=False)
    print(f"{len(new)} new rows -> {CSV}", flush=True)


def tci(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return m, m - h, m + h


def fmt(x):
    m, lo, hi = tci(x)
    return f"{m:.1f} [{lo:.1f}, {hi:.1f}]" if len(x) > 1 else f"{m:.1f}"


def report():
    P = pd.read_csv(PROBE)
    P = P[(P.layer == LAYER) & P.seed.isin(SEEDS)].drop_duplicates(["arm", "seed", "epoch"])
    B = pd.read_csv(CSV).drop_duplicates(["arm", "seed", "epoch"])
    D = pd.concat([P[COLS], B[COLS]], ignore_index=True)
    D["fine_epoch"] = np.where(D.arm == "scratch", D.epoch, D.epoch - K)
    arms = [("scratch", "scratch (initialisation)"), ("sc_true", "true superclasses"), ("sc_rand", "random grouping"),
            ("sc_blind", "superclass-blind task")]
    bseeds = sorted(B.seed.unique())
    sw = D[D.fine_epoch == 0]
    L = [f"# Frozen read-out at the switch, superclass-blind pre-training (ED Fig. 9d measure)", "",
         f"Written {datetime.now():%Y-%m-%d %H:%M} by scripts/paper/blind_frozen_readout.py. "
         f"Blind-arm switch computed for {(B.epoch == K).sum()} of 5 seeds "
         f"({', '.join(map(str, sorted(B[B.epoch == K].seed)))}); fine-class checkpoints per seed: "
         + ", ".join(f"{s}: {((B.seed == s) & (B.epoch > K)).sum()}" for s in bseeds) + ".", "",
         "Read-out: 100-class linear read-out (fine100) of frozen pooled (avgpool) features of the 10,000 stored test "
         "images, StandardScaler + RidgeClassifier(alpha=10), 5-fold stratified CV (curriculum_probe.py, unchanged). "
         "Switch = end of the 20 pre-training epochs, before any fine-class epoch (scratch, initialisation). "
         "Scratch, true superclasses and random grouping from CURRICULUM_probe/probe.csv; superclass-blind task from "
         "BLIND_PRETRAIN/frozen_readout.csv. Mean [t-based 95% CI over seeds], %.", "",
         "## fine100 at the switch, per seed", "",
         "| arm | " + " | ".join(f"seed {s}" for s in SEEDS) + " | mean [95% CI] |",
         "|---|" + "---|" * (len(SEEDS) + 1)]
    for a, lab in arms:
        q = sw[sw.arm == a].set_index("seed").fine100 * 100
        L.append(f"| {lab} | " + " | ".join(f"{q[s]:.1f}" if s in q.index else "–" for s in SEEDS)
                 + f" | {fmt(q.values)} (n={len(q)}) |")
    L += ["", "## Paired contrasts at the switch (seeds with the blind arm), percentage points", ""]
    bl = sw[sw.arm == "sc_blind"].set_index("seed").fine100 * 100
    for a, lab in arms[:3]:
        q = sw[sw.arm == a].set_index("seed").fine100.loc[bl.index] * 100
        dd = bl - q
        L.append(f"- blind - {lab}: {fmt(dd.values)} (n={len(dd)}; per seed "
                 + ", ".join(f"{s}: {v:+.1f}" for s, v in dd.items()) + ")")
    L += ["", "## Other read-outs at the switch, mean [95% CI] (%)", "",
          "| arm | within_sc (5-way inside each superclass, chance 20) | super20 (20-way superclass, chance 5) |",
          "|---|---|---|"]
    for a, lab in arms:
        q = sw[sw.arm == a]
        L.append(f"| {lab} | {fmt(q.within_sc.values * 100)} | {fmt(q.super20.values * 100)} |")
    L += ["", "## fine100 over fine-class epochs (ED 9d checkpoints), mean [95% CI] (n seeds)", "",
          "| fine epoch | " + " | ".join(lab for _, lab in arms) + " |", "|---|" + "---|" * len(arms)]
    for fe in sorted(D[(D.fine_epoch >= 0) & (D.fine_epoch <= 30)].fine_epoch.unique()):
        cells = []
        for a, _ in arms:
            q = D[(D.arm == a) & (D.fine_epoch == fe)].fine100.values * 100
            cells.append(f"{fmt(q)} ({len(q)})" if len(q) else "–")
        L.append(f"| {fe} | " + " | ".join(cells) + " |")
    L += ["", f"Initialisation check: blind-arm epoch-0 fine100 equals the curriculum arms' epoch 0 for the same seed "
          f"(same initial weights): " + ", ".join(
              f"seed {s} {'yes' if np.isclose(B[(B.seed == s) & (B.epoch == 0)].fine100.iloc[0], P[(P.arm == 'sc_rand') & (P.seed == s) & (P.epoch == 0)].fine100.iloc[0], atol=5e-4) else 'NO'}"
              for s in bseeds if len(B[(B.seed == s) & (B.epoch == 0)])) + "."]
    MD.write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.repro:
        sys.exit(0 if repro() else 1)
    if a.seeds:
        compute(a.seeds)
    if CSV.exists():
        report()


if __name__ == "__main__":
    main()
