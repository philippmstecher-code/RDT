#!/usr/bin/env python3
"""transient_stage: a transient superclass stage in raw activations, tested on the held-out networks.

The estimators are imported unchanged from the scripts used on the open networks:
  - point estimate share_sc = B_sc / B_cls, split-half with split seed 0:  dictfree_compute.Ctx + dictfree_compute.population
    (raw space; within-class-whitened space via dictfree_compute.whiten);
  - SD: 20-replicate image bootstrap, RandomState(12):                     dictfree_share_splits.share / .splits
    (the 20 random 50/50 splits, RandomState(11), are also recomputed, descriptively);
  - superclass map:                                                        dictfree_lib.superclass_of_class.
Differences from that code:
  1. lane list = the 7 held-out networks (core s137 x 3; clean ResNet-18 s256/s314; label-noise "standard"
     ResNet-18 s256/s314); dictfree_lib.load_acts reads the open networks only, so it is copied below without that
     check (_guard);
  2. only the pooled layer is processed;
  3. specificity uses 200 random 20x5 partitions instead of 100. They continue the same RNG stream
     (RandomState(42), rng.permutation(100) per partition), so the first 100 are those of dictfree_compute; the null
     share is computed as in dictfree_compute.population (split seed 0).
A network is transient if share_sc peaks at an interior checkpoint and both rise and decline exceed Z_THR bootstrap
SDs. Criterion for the stage (outcome label 'HOLDS'): transient in all 3 core networks and in >= 3 of the 4 30-epoch
networks, with a median core decline >= 0.05 (RULE_NOTES gives the SD used for the rise). Read-only on data volumes. Writes analysis_outputs/paper/DATA_order/transient_stage/.

Usage: transient_stage.py
"""
from __future__ import annotations

import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)

from dictfree_lib import LANES_JSON, REPO, superclass_of_class  # noqa: E402
from dictfree_compute import EPS, Ctx, half_dev, population, var_parts, whiten  # noqa: E402
from dictfree_share_splits import N_SPLIT, share, splits  # noqa: E402

OUT = REPO / "analysis_outputs" / "paper" / "DATA_order" / "transient_stage"
LN_EXP = Path(RDT_DATA + "/core/experiments/0c28799c-ec9d-4e45-ac30-302dff810a8d/lanes")
# label-noise "standard" lanes, seeds 256/314 (NOISE_scaffold/per_lane_metrics.csv, condition == 'standard')
LN_STANDARD = {256: "5a1e6fe5-6b82-4532-8766-0895df9d2799", 314: "1da28ac9-4274-4ecd-a2b4-96a6adffc13a"}
HELDOUT_KEYS = ["core_ResNet-18_s137", "core_ViT-Small_s137", "core_CCT-7_s137",
                "clean_ResNet-18_s256", "clean_ResNet-18_s314"]
CORE = ["core_ResNet-18_s137", "core_ViT-Small_s137", "core_CCT-7_s137"]
N_PART = 200
Z_THR = 3.0
RULE_NOTES = ("rise/SD: SD_rise = sqrt(SD_first^2 + SD_peak^2) from the per-checkpoint bootstrap SDs (conservative "
              "and parallel to the decline rule). rise/SD_peak is reported alongside; the result is flagged if it "
              "differs under that reading.")


def heldout_lanes():
    """Same structure as dictfree_lib.open_lanes, held-out lanes only."""
    d = json.load(open(LANES_JSON))
    by = {ln["key"]: ln for ln in d["lanes"]}
    out = []
    for k in HELDOUT_KEYS:
        ln = dict(by[k])
        layers = [L for L in ln["layers"] if L != "layer4"]
        out.append({"key": k, "group": ln["group"], "arch": ln["arch"], "seed": ln["seed"], "dir": ln["dir"],
                    "pooled": ln["pooled"], "layers": layers,
                    "ckpts": [{"label": c["label"], "epoch": float(c["epoch_field"]), "val_acc": c["val_acc"]}
                              for c in ln["ckpts"]]})
    for s, lid in LN_STANDARD.items():
        d_ = LN_EXP / lid
        snaps = json.load(open(d_ / "dev_snapshots" / "metrics" / "devtrain_snapshots.json"))
        ck, seen = [], set()
        for sn in sorted(snaps, key=lambda z: z["milestone_index"]):
            if sn["weight_updates"] in seen:
                continue  # final milestone duplicates its predecessor
            seen.add(sn["weight_updates"])
            ck.append({"label": str(sn["milestone_index"]), "epoch": float(sn["epoch"]),
                       "val_acc": sn.get("val_accuracy")})
        out.append({"key": f"lnstd_ResNet-18_s{s}", "group": "lnstd", "arch": "ResNet-18", "seed": s, "dir": str(d_),
                    "pooled": "avgpool", "layers": ["layer1", "layer2", "layer3", "avgpool"], "ckpts": ck})
    for ln in out:
        n = len(ln["ckpts"])
        for i, c in enumerate(ln["ckpts"]):
            c["idx"], c["tau"] = i, i / (n - 1)
    return out


def load_acts(ln, label):
    """dictfree_lib.load_acts unchanged, without the check that restricts it to the open networks."""
    p = Path(ln["dir"]) / "dev_snapshots" / f"milestone_{label}" / "individual_multilayer_activations.pt"
    a = torch.load(p, map_location="cpu", weights_only=False)
    out = {}
    for L, v in a.items():
        assert sorted(v.keys()) == list(range(100))
        X = torch.cat([v[c].float() for c in range(100)], 0).numpy().astype(np.float64)
        assert X.shape[0] == 10000
        out[L] = X
    return out


def partition_masks_200():
    """Same RNG structure as dictfree_compute.Ctx.perm_masks, extended to 200 partitions."""
    rng = np.random.RandomState(42)
    pm = []
    for _ in range(N_PART):
        perm = rng.permutation(100)
        pm.append(np.stack([np.isin(np.arange(100), perm[s * 5:(s + 1) * 5]) for s in range(20)]))
    return pm


def compute_lane(ln, pm200):
    f = OUT / "cache" / f"{ln['key']}.csv"
    if f.exists():
        print("skip", ln["key"], flush=True)
        return pd.read_csv(f)
    t0 = time.time()
    ctx = Ctx(superclass_of_class(ln["dir"]))
    for a, b in zip(ctx.perm_masks, pm200[:100]):
        assert np.array_equal(a, b)
    L = ln["pooled"]
    rows = []
    for c in ln["ckpts"]:
        X = load_acts(ln, c["label"])[L]
        Xw = whiten(X)
        r = {"lane": ln["key"], "group": ln["group"], "arch": ln["arch"], "seed": ln["seed"], "layer": L,
             "checkpoint": c["label"], "idx": c["idx"], "tau": c["tau"], "epoch": c["epoch"], "val_acc": c["val_acc"]}
        for tag, Z in (("raw", X), ("wht", Xw)):
            p = population(Z, ctx, tag)
            r.update({k: p[k] for k in (f"{tag}_share_sc", f"{tag}_share_sc_null_mean", f"{tag}_share_sc_null_sd",
                                        f"{tag}_share_sc_null_p95", f"{tag}_share_sc_z", f"{tag}_fsc", f"{tag}_fwsc")})
            # 200-partition specificity null (dictfree_compute.population null, 200 partitions)
            DA, DB = half_dev(Z, ctx)
            Bc = float(var_parts(DA, DB, ctx.masks)[0].sum())
            null = np.array([var_parts(DA, DB, m)[1].sum() for m in pm200]) / (Bc + EPS)
            r[f"{tag}_null200_mean"], r[f"{tag}_null200_sd"] = float(null.mean()), float(null.std())
            r[f"{tag}_null200_min"], r[f"{tag}_null200_max"] = float(null.min()), float(null.max())
        # dictfree_share_splits (pooled layer): 20 random splits (descriptive) + 20-replicate image bootstrap SD
        rng = np.random.RandomState(11)
        sp = [splits(rng) for _ in range(N_SPLIT)]
        for tag, Z in (("raw", X), ("wht", Xw)):
            v = np.array([share(Z, a, b, ctx.masks) for a, b in sp])
            r[f"{tag}_share_mean"], r[f"{tag}_share_sd"] = v[:, 0].mean(), v[:, 0].std(ddof=1)
            rb = np.random.RandomState(12)
            vb = np.array([share(Z, *splits(rb, boot=True), ctx.masks)[0] for _ in range(N_SPLIT)])
            r[f"{tag}_share_boot_sd"] = vb.std(ddof=1)
        rows.append(r)
        print(f"  {ln['key']} ck {c['label']} raw {r['raw_share_sc']:.4f} sd {r['raw_share_boot_sd']:.4f} "
              f"{time.time() - t0:.0f}s", flush=True)
        del X, Xw
    d = pd.DataFrame(rows)
    d.to_csv(f, index=False)
    return d


def transient(g, tag):
    g = g.sort_values("idx")
    v = g[f"{tag}_share_sc"].values
    sd = g[f"{tag}_share_boot_sd"].values
    n = len(v)
    am = int(v.argmax())
    rise, decline = v[am] - v[0], v[am] - v[-1]
    sd_rise = np.sqrt(sd[0] ** 2 + sd[am] ** 2)
    sd_dec = np.sqrt(sd[am] ** 2 + sd[-1] ** 2)
    interior = 0 < am < n - 1
    z_rise, z_rise_peak, z_dec = rise / sd_rise, rise / sd[am], decline / sd_dec
    return {"space": tag, "n_ck": n, "share_first": v[0], "share_peak": v[am], "share_last": v[-1],
            "peak_idx": am, "peak_checkpoint": g.checkpoint.values[am], "peak_epoch": g.epoch.values[am],
            "peak_val_acc": g.val_acc.values[am], "interior": bool(interior), "rise": rise, "decline": decline,
            "sd_first": sd[0], "sd_peak": sd[am], "sd_last": sd[-1], "z_rise": z_rise, "z_rise_sdpeak": z_rise_peak,
            "z_decline": z_dec,
            "transient": bool(interior and z_rise > Z_THR and z_dec > Z_THR),
            "transient_alt_sdpeak": bool(interior and z_rise_peak > Z_THR and z_dec > Z_THR),
            "split_mean_rise": g[f"{tag}_share_mean"].values[am] - g[f"{tag}_share_mean"].values[0],
            "split_mean_decline": g[f"{tag}_share_mean"].values[am] - g[f"{tag}_share_mean"].values[-1]}


def verdict(S, tag, col):
    s = S[S.space == tag].set_index("lane")
    core = s.loc[CORE]
    r30 = s.drop(index=CORE, errors="ignore")
    n_core, n_r30 = int(core[col].sum()), int(r30[col].sum())
    med = float(np.median(core.decline))
    ok = bool(n_core == 3 and n_r30 >= 3 and med >= 0.05)
    return {"n_core_transient": n_core, "n_core": len(core), "n_30ep_transient": n_r30, "n_30ep": len(r30),
            "median_core_decline": med, "verdict": "HOLDS" if ok else "DOES NOT HOLD"}


def main():
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    pm200 = partition_masks_200()
    lanes = heldout_lanes()
    D = pd.concat([compute_lane(ln, pm200) for ln in lanes], ignore_index=True)
    D.to_csv(OUT / "share_sc_per_checkpoint.csv", index=False)
    S = pd.DataFrame([{"lane": k, **transient(g, tag)} for k, g in D.groupby("lane") for tag in ("raw", "wht")])
    S.to_csv(OUT / "transient_summary.csv", index=False)
    primary = verdict(S, "raw", "transient")
    alt = verdict(S, "raw", "transient_alt_sdpeak")
    sw = S[(S.space == "wht") & S.lane.isin(CORE)]
    sec = {"n_core_transient_whitened": int(sw.transient.sum()), "n_core": 3,
           "median_core_decline_whitened": float(np.median(sw.decline)),
           "all_core_transient_whitened": bool(sw.transient.all())}
    spec = {"null200_mean_min": float(D.raw_null200_mean.min()), "null200_mean_max": float(D.raw_null200_mean.max()),
            "within_0.17_0.22_every_checkpoint": bool(D.raw_null200_mean.between(0.17, 0.22).all()),
            "per_lane": {k: [float(g.raw_null200_mean.min()), float(g.raw_null200_mean.max())]
                         for k, g in D.groupby("lane")},
            "whitened_null200_mean_range": [float(D.wht_null200_mean.min()), float(D.wht_null200_mean.max())]}
    res = {"test": "transient_stage",
           "verdict": primary["verdict"], "primary": primary,
           "verdict_under_alt_rise_sd_reading": alt["verdict"], "alt_reading": alt, "rule_notes": RULE_NOTES,
           "per_network_raw": S[S.space == "raw"].set_index("lane").to_dict(orient="index"),
           "secondary_whitened_core": sec,
           "per_network_whitened": S[S.space == "wht"].set_index("lane").to_dict(orient="index"),
           "specificity_200_partitions": spec,
           "lanes": {ln["key"]: ln["dir"] for ln in lanes}}
    json.dump(res, open(OUT / "results.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    pd.set_option("display.width", 250)
    print(S.round(4).to_string())
    print(json.dumps({k: res[k] for k in ("verdict", "primary", "verdict_under_alt_rise_sd_reading",
                                          "secondary_whitened_core", "specificity_200_partitions")}, indent=1,
                     default=float))


if __name__ == "__main__":
    main()
