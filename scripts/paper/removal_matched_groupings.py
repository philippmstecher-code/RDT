#!/usr/bin/env python3
"""Removal from trained networks (Extended Data Fig. 6h procedure) with similarity-matched groupings as the control.

Networks: critical_sham sham_e00_seed{S}, epochs 1, 2, 3, 5, 10, 11, 12, 13, 15, 20 (as ED 6h), S in the 5 seeds.
Subspaces in the last block (layer4 + pooled layer), estimated on the pooled activations of test images 0-19 of every
class exactly as pseudo_superclass.run_network (mean_shift.estimate_coarse; true superclasses without rank cap; groupings
truncated to the true rank per stage):
  true     the superclasses
  matched  20 groupings per seed whose mean within-group cosine similarity of class means equals the superclasses'
           (nesting_similarity_null.matched_partition, partitions p = 1..20 of that script, built on the class means of
           the clean network of the same seed at epoch 30)
Removal: centred x - QQ^T(x - mu) (the main removal of ED 6h). Outcomes on the 10,000 test images with the network's own head:
fine-class accuracy cost and superclass-assignment cost (assignment = predicted class in the true superclass).
Writes analysis_outputs/paper/PSEUDO_superclass/matched/{raw_s{S}_e{E}.csv, summary.json}.
Usage: python scripts/paper/removal_matched_groupings.py [run|analyze]
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mean_shift as B  # noqa: E402
from mean_shift import (CLS2SC, LAST, cifar_test, epoch_label, estimate_coarse, lastblock_logits,  # noqa: E402
                                layer4_raw, load_net, proj_apply, sham_dir, stored_predictions)
from nesting_similarity_null import FINAL, matched_partition, within_mean  # noqa: E402
from dependency_lib import LaneData, standard_lanes  # noqa: E402

torch.set_num_threads(2)
B.WINDOWS["w11_20"]["sham"] = "sham_e00"     # as ED 6h (the epoch-0 sham trains to epoch 30)
REPO = HERE.parents[1]
OUT = REPO / "analysis_outputs/paper/PSEUDO_superclass/matched"
N_MATCHED = 20
EPOCHS = {"w01_10": [1, 2, 3, 5, 10], "w11_20": [11, 12, 13, 15, 20]}
NO_CAP = 10 ** 6


def matched_partitions():
    out = {}
    for r in standard_lanes():
        X = LaneData(r["dir"]).raw(FINAL)["avgpool"].astype(np.float64)
        mu = X.reshape(100, 100, -1).mean(1)
        mu /= np.linalg.norm(mu, axis=1, keepdims=True)
        S = mu @ mu.T
        target = within_mean(S, CLS2SC)
        rand_mean = np.mean([within_mean(S, np.random.RandomState(p).permutation(np.repeat(np.arange(20), 5)))
                             for p in range(200)])
        tol = 0.01 * (target - rand_mean)
        out[int(r["seed"])] = [matched_partition(S, target, tol, np.random.RandomState(p))[0]
                               for p in range(1, N_MATCHED + 1)]
    return out


def outcomes(logits, y):
    pred = logits.argmax(1).numpy()
    return 100.0 * float((pred == y).mean()), 100.0 * float((CLS2SC[pred] == CLS2SC[y]).mean())


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = matched_partitions()
    json.dump({s: [p.tolist() for p in v] for s, v in parts.items()}, open(OUT / "partitions.json", "w"))
    X, y = cifar_test()
    idx = np.concatenate([np.flatnonzero(y == c)[:20] for c in range(100)])
    for win, eps in EPOCHS.items():
        for e in eps:
            for s in sorted(parts):
                fn = OUT / f"raw_s{s}_e{e}.csv"
                if fn.exists():
                    continue
                t0 = time.time()
                sl = sham_dir(win, s)
                lab, _ = epoch_label(sl, e)
                net = load_net(sl, lab)
                base, h4 = layer4_raw(net, X)
                assert float((base.argmax(1).numpy() == stored_predictions(sl, lab)).mean()) >= 0.999
                h4e = h4[idx]
                C = {"layer4": h4.shape[1], "avgpool": h4.shape[1]}
                p4 = h4e.mean(dim=(2, 3))
                mus = {"layer4": p4.mean(0), "avgpool": p4.mean(0)}

                def get_acts(Q):
                    z4 = proj_apply(h4e, Q["layer4"], None, False)
                    za = proj_apply(z4.mean(dim=(2, 3)), Q["avgpool"], None, False)
                    return {"layer4": z4.mean(dim=(2, 3)), "avgpool": za}

                A0, S0 = outcomes(base, y)
                Qt, _ = estimate_coarse(get_acts, LAST, {st: NO_CAP for st in LAST},
                                        torch.as_tensor(CLS2SC[y[idx]]), C)
                r_true = {st: int(Qt[st].shape[1]) for st in LAST}
                rows = []
                subs = [("true", -1, Qt)]
                for k, g in enumerate(parts[s]):
                    Qm, _ = estimate_coarse(get_acts, LAST, r_true, torch.as_tensor(g[y[idx]]), C)
                    subs.append(("matched", k, Qm))
                for kind, k, Q in subs:
                    A, Sa = outcomes(lastblock_logits(net, h4, Q, True, mus), y)
                    rows.append({"seed": s, "epoch": e, "kind": kind, "partition": k, "A_net": A0, "S_net": S0,
                                 "cost_fine": A0 - A, "cost_assignment": S0 - Sa,
                                 **{f"rank_{st}": int(Q[st].shape[1]) for st in LAST}})
                pd.DataFrame(rows).to_csv(fn, index=False)
                d = pd.DataFrame(rows)
                t, m = d[d.kind == "true"].iloc[0], d[d.kind == "matched"]
                print(f"s{s} e{e}: fine true {t.cost_fine:.2f} matched {m.cost_fine.mean():.2f} "
                      f"[{m.cost_fine.min():.2f}, {m.cost_fine.max():.2f}] | assign true {t.cost_assignment:.2f} "
                      f"matched {m.cost_assignment.mean():.2f} | r {r_true} | {time.time() - t0:.0f}s", flush=True)


def tci(d):
    d = np.asarray(d, float)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return [float(d.mean()), float(d.mean() - h), float(d.mean() + h)]


def analyze():
    D = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("raw_s*_e*.csv"))], ignore_index=True)
    res = {"per_epoch": []}
    for e, g in D.groupby("epoch"):
        rec = {"epoch": int(e)}
        for k in ("cost_fine", "cost_assignment"):
            per = []
            for s, gs in g.groupby("seed"):
                t = float(gs[gs.kind == "true"][k].iloc[0])
                m = gs[gs.kind == "matched"][k].to_numpy(float)
                per.append({"seed": int(s), "true": t, "matched_mean": float(m.mean()), "matched_max": float(m.max()),
                            "rank": int((m >= t).sum()) + 1, "margin_mean": t - float(m.mean()),
                            "margin_best": t - float(m.max())})
            rec[k] = {"margin_mean_ci": tci([p["margin_mean"] for p in per]),
                      "n_seeds_true_above_all": int(sum(p["rank"] == 1 for p in per)),
                      "n_seeds_true_above_mean": int(sum(p["margin_mean"] > 0 for p in per)),
                      "rank_median": float(np.median([p["rank"] for p in per])), "per_seed": per}
        res["per_epoch"].append(rec)
    json.dump(res, open(OUT / "summary.json", "w"), indent=1)
    for r in res["per_epoch"]:
        f, a = r["cost_fine"], r["cost_assignment"]
        print(f"epoch {r['epoch']:2d} | fine: true - matched mean {f['margin_mean_ci'][0]:+.2f} "
              f"[{f['margin_mean_ci'][1]:+.2f}, {f['margin_mean_ci'][2]:+.2f}], above all in {f['n_seeds_true_above_all']}/5, "
              f"rank median {f['rank_median']:.0f}/21 | assignment: {a['margin_mean_ci'][0]:+.2f} "
              f"[{a['margin_mean_ci'][1]:+.2f}, {a['margin_mean_ci'][2]:+.2f}], above all {a['n_seeds_true_above_all']}/5")


if __name__ == "__main__":
    {"run": run, "analyze": analyze}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
