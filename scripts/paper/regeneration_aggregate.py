"""Per-seed contrasts from milestone_metrics.jsonl (+ endpoint geometry) -> summary CSV/JSON + figure."""
import json

import numpy as np
import pandas as pd

from regeneration_lib import OUT, WINDOWS

CH_SC, CH_FINE = 1 / 20, 1 / 100


def main():
    d = pd.read_json(OUT / "milestone_metrics.jsonl", lines=True, dtype={"window": str})
    d["proj_cls_share"] = d.proj_trBcls / d.proj_trT
    d["raw_cls_share"] = d.raw_trBcls / d.raw_trT
    key = ["window", "seed", "epoch"]
    sham = d[d.arm == "sham"].set_index(key)
    rows = []
    for arm in ("coarse", "random"):
        a = d[d.arm == arm].set_index(key)
        j = a.join(sham, rsuffix="_sham", how="inner")
        for idx, r in j.iterrows():
            w, s, ep = idx
            e0 = WINDOWS[w]
            out = dict(window=w, seed=s, epoch=ep, arm=arm, rel_epoch=ep - e0, in_window=bool(r.in_window))
            for m in ("sc_share", "cls_share", "sc_over_cls", "PR"):
                out[f"proj_{m}"] = r[f"proj_{m}"]
                out[f"sham_{m}"] = r[f"proj_{m}_sham"]
                out[f"{m}_ratio_to_sham"] = r[f"proj_{m}"] / r[f"proj_{m}_sham"]
            for m, ch in (("sc_ridge", CH_SC), ("fine_ridge", CH_FINE), ("sc_knn", CH_SC),
                          ("sc_via_fine_ridge", CH_SC), ("fc_fine_acc", CH_FINE), ("fc_sc_acc", CH_SC),
                          ("sc_logreg", CH_SC)):
                if f"proj_{m}" in r and pd.notna(r.get(f"proj_{m}")) and pd.notna(r.get(f"proj_{m}_sham")):
                    out[f"proj_{m}"] = r[f"proj_{m}"]
                    out[f"sham_{m}"] = r[f"proj_{m}_sham"]
                    out[f"{m}_d_sham"] = r[f"proj_{m}"] - r[f"proj_{m}_sham"]
                    out[f"{m}_aboveChance_ratio"] = (r[f"proj_{m}"] - ch) / (r[f"proj_{m}_sham"] - ch)
            for m in ("raw_sc_ridge", "raw_fine_ridge", "raw_sc_share", "raw_cls_share", "raw_sc_over_cls",
                      "raw_frac_Bsc_inQ", "raw_frac_Bcls_inQ", "raw_frac_T_inQ", "chance_frac_inQ",
                      "rawQ_sc_ridge", "rawQ_fine_ridge", "rawQ_sc_knn", "removed_dims", "raw_fc_fine_acc"):
                if m in r and pd.notna(r.get(m)):
                    out[m] = r[m]
            if pd.notna(r.get("raw_sc_share")):
                out["raw_sc_share_ratio_to_sham"] = r.raw_sc_share / r.proj_sc_share_sham
                out["raw_sc_ridge_aboveChance_ratio"] = (r.raw_sc_ridge - CH_SC) / (r.proj_sc_ridge_sham - CH_SC)
            rows.append(out)
    c = pd.DataFrame(rows)
    c.to_csv(OUT / "contrasts_vs_sham_per_seed_epoch.csv", index=False)

    # window summaries (per seed: mean over the 10 window epochs, and at first/last window epoch)
    metrics = ["sc_share_ratio_to_sham", "cls_share_ratio_to_sham", "sc_over_cls_ratio_to_sham",
               "sc_ridge_aboveChance_ratio", "fine_ridge_aboveChance_ratio", "sc_knn_aboveChance_ratio",
               "sc_via_fine_ridge_aboveChance_ratio", "fc_fine_acc_aboveChance_ratio", "sc_logreg_aboveChance_ratio",
               "raw_sc_share_ratio_to_sham", "raw_sc_ridge_aboveChance_ratio", "raw_frac_Bsc_inQ",
               "raw_frac_Bcls_inQ", "raw_frac_T_inQ", "chance_frac_inQ", "rawQ_sc_ridge", "proj_sc_over_cls",
               "sham_sc_over_cls", "PR_ratio_to_sham", "sc_ridge_d_sham", "fine_ridge_d_sham", "fc_fine_acc_d_sham"]
    metrics = [m for m in metrics if m in c]
    summ = []
    for (w, arm, s), g in c[c.in_window].groupby(["window", "arm", "seed"]):
        for sel, gg in (("window_mean", g), ("first_epoch", g[g.rel_epoch == 1]), ("last_epoch", g[g.rel_epoch == 10])):
            summ.append(dict(window=w, arm=arm, seed=s, which=sel, **{m: gg[m].mean() for m in metrics}))
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "window_summary_per_seed.csv", index=False)
    agg = S.groupby(["window", "arm", "which"])[metrics].agg(["mean", "min", "max"])
    agg.to_csv(OUT / "window_summary_across_seeds.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(S.groupby(["window", "arm", "which"])[metrics].mean().round(3).T.to_string())

    # post-release / endpoint
    post = c[~c.in_window]
    E = post.groupby(["window", "arm", "epoch"])[[m for m in ("sc_ridge_d_sham", "fine_ridge_d_sham", "fc_fine_acc_d_sham",
                                                               "sc_share_ratio_to_sham", "PR_ratio_to_sham",
                                                               "sc_over_cls_ratio_to_sham") if m in post]].agg(["mean", "min", "max"])
    E.to_csv(OUT / "post_release_vs_sham.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(E.round(4).to_string())


if __name__ == "__main__":
    main()
