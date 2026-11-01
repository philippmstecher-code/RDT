#!/usr/bin/env python3
"""Did the networks learn less while the coarse structure was suppressed? A read-out that does not depend on the
network's own classifier.

Evaluating the suppression checkpoints without the projection with the network's own classifier is confounded: the
classifier never received input along the removed directions during suppression, so its weights there are untrained
(and shrunk by weight decay). Here, instead, a fresh linear read-out is fitted to the network's unprojected pooled
features: each checkpoint is run on the 10,000 CIFAR-100 test images without any projection (transfer_lib.run_images),
and a 100-way ridge classifier is fitted and evaluated by 5-fold cross-validation (as curriculum_probe.probe). The
cross-validated predictions are also decomposed into P(superclass right) and P(fine right | superclass right).
Arms (last block, epochs 1-10, seeds 7, 42, 137, 256, 314): sham, superclass code, random-grouping code, random
directions. Epochs 1, 3 (in the window) and 10 (end of the window).
Reported: arm minus sham per seed, fine read-out accuracy and its two factors; paired t CI over seeds.
Writes analysis_outputs/paper/RANDGROUP_control/fresh_readout.csv and fresh_readout_summary.json, and Source Data
figures/final/source_data/fresh_readout.csv.
Usage: python scripts/paper/fresh_readout.py
"""
import json
import os
import sys
from pathlib import Path
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(2)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import transfer_lib as T  # noqa: E402
from class_level import FINE2SC  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis_outputs/paper/RANDGROUP_control"
SD = REPO / "analysis_outputs/paper/figures/final/source_data"
RUNS_EXTENDED = Path(RDT_DATA + "/extended/experiments")
FEST = Path(RDT_DATA + "/core/experiments/scaffold_lesion/lanes")
SEEDS = [7, 42, 137, 256, 314]
EPOCHS = [1, 3, 10]
LANES = {
    "sham": lambda s: RUNS_EXTENDED / "critical_sham/lanes" / f"sham_e00_seed{s}",
    "superclass": lambda s: (FEST if s == 42 else RUNS_EXTENDED / "prevention/lanes")
    / f"lesion_scaffold_e00_seed{s}_rp65_Llayer4-avgpool",
    "random": lambda s: (FEST if s == 42 else RUNS_EXTENDED / "prevention/lanes")
    / f"lesion_random_variance_e00_seed{s}_rp65_Llayer4-avgpool",
    "randgroup": lambda s: RUNS_EXTENDED / "randgroup/lanes" / f"lesion_randgroup_e00_seed{s}_rp65_Llayer4-avgpool",
}


def tci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean()), float(x.mean() - h), float(x.mean() + h)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cf = T.cifar_test()
    y = np.asarray(cf["y"])
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    rows = []
    for arm, lf in LANES.items():
        for s in SEEDS:
            d = lf(s)
            snaps = json.load(open(d / "dev_snapshots/metrics/devtrain_snapshots.json"))
            by_ep = {round(x["weight_updates"] / 391, 3): x for x in snaps if x.get("snapshot_type") != "final"}
            for e in EPOCHS:
                x = by_ep[float(e)]
                net = T.load_net({"dir": str(d), "arch": "ResNet-18", "group": "extended"}, str(x["milestone_index"]))
                feat = T.run_images(net, "ResNet-18", cf["X"], to_tensor=T.cifar_batch)["none"]["feat"]
                clf = make_pipeline(StandardScaler(), RidgeClassifier(alpha=10.0))
                pred = cross_val_predict(clf, feat, y, cv=cv)
                sc_ok = FINE2SC[pred] == FINE2SC[y]
                rows.append({"arm": arm, "seed": s, "epoch": e, "fine_readout": 100 * float((pred == y).mean()),
                             "superclass_assignment": 100 * float(sc_ok.mean()),
                             "within_superclass": 100 * float((pred == y)[sc_ok].mean())})
                print(rows[-1], flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "fresh_readout.csv", index=False)
    D.to_csv(SD / "fresh_readout.csv", index=False)
    res = {}
    for win, eps in (("epochs_1_3", [1, 3]), ("epoch_10", [10])):
        M = D[D.epoch.isin(eps)].groupby(["arm", "seed"])[["fine_readout", "superclass_assignment",
                                                             "within_superclass"]].mean()
        for arm in ("superclass", "randgroup", "random"):
            for k in M.columns:
                dlt = (M.loc[arm][k] - M.loc["sham"][k]).loc[SEEDS].values
                res[f"{win}|{arm} - sham|{k}"] = {"mean_ci": tci(dlt), "n_negative": int((dlt < 0).sum())}
        for k in M.columns:
            dlt = (M.loc["superclass"][k] - M.loc["randgroup"][k]).loc[SEEDS].values
            res[f"{win}|superclass - randgroup|{k}"] = {"mean_ci": tci(dlt), "n_negative": int((dlt < 0).sum())}
    json.dump(res, open(OUT / "fresh_readout_summary.json", "w"), indent=1)
    for k, v in res.items():
        print(k, [round(z, 2) for z in v["mean_ci"]], "neg", v["n_negative"])


if __name__ == "__main__":
    main()
