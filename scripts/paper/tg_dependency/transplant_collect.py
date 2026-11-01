"""Collect the task-general transplant runs (run_tg_transplant.py; tgtransplant at transplant epoch 1,
tgtransplant_e5 at epoch 5). Reads the run records only.

Arms: A = sham_e00 (all layers trainable, reference); in the other arms stem + layer1-2 are frozen: Afrz = initial weights
frozen, B = stem + layer1-2 transplanted from the same seed's sham run at the transplant epoch and frozen, C = B with
the task-general subspace of layer1 and layer2 projected out, Crand = B with a random subspace projected out.
Per lane: validation accuracy per epoch (devtrain_metrics.json), the projected subspace (rank, variance share).
Contrasts paired over seeds (7, 42, 137): C - Crand (task-general vs random removal at matched rank/variance),
C - B, Crand - B, B - Afrz, B - A.
Writes analysis_outputs/paper/TG_dependency/transplant_long.csv and transplant_summary.csv.
Usage: python scripts/paper/tg_dependency/transplant_collect.py
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

X = RDT_DATA + "/extended/experiments"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "analysis_outputs", "paper",
                   "TG_dependency")
SEEDS = [7, 42, 137]


def metrics(lane):
    f = os.path.join(lane, "dev_snapshots", "metrics", "devtrain_metrics.json")
    return {r["epoch"]: r["val_accuracy"] for r in json.load(open(f))} if os.path.exists(f) else {}


def spec(lane):
    f = os.path.join(lane, "lesion_spec.json")
    if not os.path.exists(f):
        return {}
    lay = json.load(open(f)).get("layers", {})
    return {f"{L}_{k}": v[k] for L, v in lay.items() for k in ("subspace_rank", "var_share_removed") if k in v}


rows = []
for s in SEEDS:
    for e, a in metrics(f"{X}/critical_sham/lanes/sham_e00_seed{s}").items():
        rows.append(dict(run="ref", arm="A", seed=s, epoch=e, val_acc=a))
for run in ("tgtransplant", "tgtransplant_e5"):
    for arm in ("Afrz", "B", "C", "Crand"):
        for s in SEEDS:
            lane = f"{X}/{run}/lanes/{arm}_seed{s}"
            if not os.path.isdir(lane):
                continue
            sp = spec(lane)
            for e, a in metrics(lane).items():
                rows.append(dict(run=run, arm=arm, seed=s, epoch=e, val_acc=a, **sp))
d = pd.DataFrame(rows)
os.makedirs(OUT, exist_ok=True)
d.to_csv(f"{OUT}/transplant_long.csv", index=False)


def paired(run, a, b, e, ref_run=None):
    x = d[(d.run == run) & (d.arm == a) & (d.epoch == e)].set_index("seed").val_acc
    y = d[(d.run == (ref_run or run)) & (d.arm == b) & (d.epoch == e)].set_index("seed").val_acc
    z = (x - y).dropna()
    if len(z) < 2:
        return None
    h = stats.t.ppf(.975, len(z) - 1) * z.std(ddof=1) / np.sqrt(len(z))
    return dict(run=run, contrast=f"{a} - {b}", epoch=e, mean=z.mean(), lo=z.mean() - h, hi=z.mean() + h,
                n=len(z), n_neg=int((z < 0).sum()), per_seed=";".join(f"{k}:{v:.2f}" for k, v in z.items()))


out = []
for run in ("tgtransplant", "tgtransplant_e5"):
    for e in sorted(d[d.run == run].epoch.unique()):
        for a, b in (("C", "Crand"), ("C", "B"), ("Crand", "B"), ("B", "Afrz")):
            r = paired(run, a, b, e)
            if r:
                out.append(r)
        r = paired(run, "B", "A", e, ref_run="ref")
        if r:
            out.append(r)
S = pd.DataFrame(out)
S.to_csv(f"{OUT}/transplant_summary.csv", index=False)
print(d.groupby(["run", "arm", "epoch"]).val_acc.mean().unstack().round(1).to_string())
print(d.drop_duplicates(["run", "arm", "seed"]).filter(regex="run|arm|seed|layer").to_string())
print(S.round(2).to_string())
