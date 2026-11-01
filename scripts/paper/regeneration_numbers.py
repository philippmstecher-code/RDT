"""Collect headline numbers (means and per-seed min/max) into key_numbers.json."""
import json

import pandas as pd

from regeneration_lib import OUT


def mm(s):
    return dict(mean=round(float(s.mean()), 4), min=round(float(s.min()), 4), max=round(float(s.max()), 4), n=int(s.size))


def main():
    K = {}
    S = pd.read_csv(OUT / "window_summary_per_seed.csv", dtype={"window": str})
    for (w, arm, which), g in S.groupby(["window", "arm", "which"]):
        K.setdefault("in_window_vs_sham", {}).setdefault(w, {}).setdefault(arm, {})[which] = {
            m: mm(g[m]) for m in ("sc_share_ratio_to_sham", "proj_sc_over_cls", "sc_ridge_aboveChance_ratio",
                                  "fine_ridge_aboveChance_ratio", "fc_fine_acc_aboveChance_ratio",
                                  "raw_sc_share_ratio_to_sham", "raw_frac_Bsc_inQ", "chance_frac_inQ", "PR_ratio_to_sham")}
    p = pd.read_csv(OUT / "pseudo_superclass_control.csv", dtype={"window": str})
    p["adv"] = p.sc_true - p.sc_pseudo_mean
    k = ["window", "seed", "epoch"]
    sh = p[p.arm == "sham"].set_index(k).adv
    for arm in ("coarse", "random"):
        g = p[p.arm == arm].set_index(k)
        g["r"] = g.adv / sh.reindex(g.index)
        g = g.reset_index()
        for (w, iw), gg in g.groupby(["window", "in_window"]):
            K.setdefault("sc_specific_decodability_ratio_to_sham", {}).setdefault(w, {}).setdefault(arm, {})[
                "in_window" if iw else "post_release"] = mm(gg.groupby("seed").r.mean())
    f = pd.read_csv(OUT / "sawtooth_fit_per_seed.csv", dtype={"window": str})
    for (w, arm), g in f.groupby(["window", "arm"]):
        K.setdefault("sawtooth", {}).setdefault(w, {})[arm] = {m: mm(g[m]) for m in ("jump", "floor", "tau_updates",
                                                                                     "frac_excess_recovered_in_65")}
    d = pd.read_csv(OUT / "subspace_dynamics.csv", dtype={"window": str})
    for (w, arm, sat), g in d.groupby(["window", "arm", "sat"]):
        gs = g.groupby("seed")
        K.setdefault("subspace", {}).setdefault(w, {}).setdefault(arm, {})["saturated" if sat else "pre_cap"] = {
            m: mm(gs[m].mean()) for m in ("regen_frac_of_sham_scshare_per_interval", "raw_frac_Bsc_new", "chance_frac_new",
                                          "reuse_excess", "leak_excess", "readd_frac") if g[m].notna().any()}
    e = pd.read_csv(OUT / "endpoint_geometry.csv", dtype={"window": str})
    for (w, arm, ep), g in e.groupby(["window", "arm", "epoch"]):
        if ep in (10, 11, 20, 21, 30):
            K.setdefault("geometry_vs_sham", {}).setdefault(w, {}).setdefault(arm, {})[int(ep)] = {
                m: mm(g[m]) for m in ("cka_to_sham", "scRSM_corr_sham", "clsRSM_corr_sham")}
    json.dump(K, open(OUT / "key_numbers.json", "w"), indent=1)
    print(json.dumps(K["subspace"], indent=0)[:3000])


if __name__ == "__main__":
    main()
