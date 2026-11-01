#!/usr/bin/env python3
"""Effects in interpretable units (analysis of existing runs; no training).

1 label noise, training time lost (fixed labels, p = 0.3; static_noise, seeds 7, 42, 137, 256, 314):
  probe   per seed, the epoch at which the within-superclass (or clean) run's superclass probe (20-way, pooled layer;
          checkpoints every 5 epochs) first reaches the between-superclass run's epoch-30 value, linear interpolation
          between checkpoints; 'epochs lost' = 30 - that epoch
  fine    per seed, the epoch at which fine-class validation accuracy (every epoch) first reaches 30, 40, 50%
          (noise_lib.cross), in the clean, within, between and similarity-matched between arms; delays between - within,
          matched - within, between - matched
2 blocking, share of early learning lost (GRADBLOCK/final_long.csv; epochs 1-3, the dose-matched window): per seed,
  (control - blocked) / (control - 1) on the mean accuracy over epochs 1-3, controls random directions and
  energy-matched; epoch equivalent: the epoch at which the control's accuracy (linear interpolation from 1% at epoch 0,
  the shared branch checkpoint at chance) reaches the blocked network's epoch-3 accuracy, 3 - that epoch = epochs lost
3 routing share under blocking: fresh ridge read-out (fresh_fine = fresh_assignment x fresh_within), per seed
  dlog = log(blocked / control) per epoch (exact: dlog fine = dlog assignment + dlog within), mean over epochs 1-3; share = dlog assignment / dlog fine
4 persistence: per core run (9 CIFAR-100 runs), lifespan records as Fig. 4a (persist_lib.records_real 'paper',
  stored thresholds, 'methods' rule): median lifespan in transitions and share alive at the last checkpoint (censored),
  coarse (task-general or superclass-selective) vs class-selective
5 forest table: every intervention as % of the control's gain over chance, per seed: fine-class accuracy over epochs
  1-3 (interventions in training, the dose-matched window) or 1-10 (label noise, acting throughout); superclass probe
  at epoch 5 (first checkpoint) for label noise, gain over 5% (20-way chance)
Writes analysis_outputs/paper/EFFECT_units/{units.json, forest.csv, synthesis.csv, label_noise_delays.csv,
persistence.csv}.
Usage: python scripts/paper/effect_units.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "figures"))
import noise_lib as L  # noqa: E402

RESULTS = HERE.parents[1] / "analysis_outputs/paper"
OUT = RESULTS / "EFFECT_units"
RUNS_EXTENDED = Path(RDT_DATA + "/extended/experiments")
SEEDS = [7, 42, 137, 256, 314]
GBL = RESULTS / "GRADJOINT/long.csv"   # GRADBLOCK/final_long.csv + jointly matched, redrawn-random and layer3-pooled arms
ARMS = {"clean": "standard_resnet18", "within": "within_sc_resnet18_throughout",
        "between": "between_sc_resnet18_throughout", "matched": "between_sc_matched_resnet18_throughout"}


def tci(v):
    v = np.asarray(v, float)
    m = v.mean()
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return {"mean": float(m), "ci": [float(m - h), float(m + h)], "per_seed": [round(float(x), 4) for x in v],
            "n_below_zero": int((v < 0).sum()), "n_above_zero": int((v > 0).sum())}


def first_reach(x, y, t):
    """Linear interpolation: first x at which y reaches t (nan if never)."""
    y = np.asarray(y, float)
    if y[0] >= t:
        return float(x[0])
    for i in range(1, len(y)):
        if y[i] >= t:
            return float(x[i - 1] + (t - y[i - 1]) / (y[i] - y[i - 1]) * (x[i] - x[i - 1]))
    return np.nan


def label_noise():
    P = L.static_probe()
    ep = np.array(P["between_sc"].columns, float)
    lost = {"within": [], "clean": []}
    for s in SEEDS:
        target = P["between_sc"].loc[s, 30]
        for ref, key in (("within", "within_sc"), ("clean", "clean")):
            lost[ref].append(30 - first_reach(ep, P[key].loc[s].values, target))
    rows = []
    for s in SEEDS:
        c = {a: L._val_curve(RUNS_EXTENDED / f"static_noise_s{s}/lanes/{ln}") for a, ln in ARMS.items()}
        for t in (30, 40, 50):
            e = {a: first_reach(np.arange(31), c[a], t) for a in ARMS}
            rows.append({"seed": s, "threshold": t, **{f"epoch_{a}": v for a, v in e.items()}})
    D = pd.DataFrame(rows)
    D["between_minus_within"] = D.epoch_between - D.epoch_within
    D["matched_minus_within"] = D.epoch_matched - D.epoch_within
    D["between_minus_matched"] = D.epoch_between - D.epoch_matched
    D["between_minus_clean"] = D.epoch_between - D.epoch_clean
    D.to_csv(OUT / "label_noise_delays.csv", index=False)
    out = {"probe_epochs_lost_vs_within": tci(lost["within"]), "probe_epochs_lost_vs_clean": tci(lost["clean"]),
           "probe_checkpoints": [int(x) for x in ep]}
    for t in (30, 40, 50):
        g = D[D.threshold == t]
        out[f"fine_{t}pct"] = {k: tci(g[k].values) for k in ("between_minus_within", "matched_minus_within",
                                                              "between_minus_matched", "between_minus_clean")}
        out[f"fine_{t}pct"]["share_similarity_of_mean_delay"] = float(g.between_minus_matched.mean() /
                                                                       g.between_minus_within.mean())
    return out


def blocking():
    D = pd.read_csv(GBL)
    A = D.pivot_table(index=["arm", "seed"], columns="epoch", values="val_accuracy")
    out = {}
    for blk in ("gradsc", "gradrg"):
        for ctl in ("gradrand", "gradenergy", "gradjoint"):
            frac, eq = [], []
            for s in SEEDS:
                b, c = A.loc[(blk, s), [1, 2, 3]].values, A.loc[(ctl, s), [1, 2, 3]].values
                frac.append(100 * (c.mean() - b.mean()) / (c.mean() - 1.0))
                e = first_reach(np.arange(0, 4), np.r_[1.0, c], b[-1])
                eq.append(3 - e)
            out[f"{blk}_vs_{ctl}"] = {"pct_of_control_gain_lost_epochs_1_3": tci(frac),
                                      "epochs_behind_at_epoch_3": tci(eq)}
    # routing share (fresh read-out, log factorisation)
    M = D[D.epoch.isin([1, 2, 3])].set_index(["arm", "seed", "epoch"])[["fresh_fine", "fresh_assignment", "fresh_within"]]
    for ctl in ("gradjoint", "gradenergy", "gradrand"):
        sh, dl_f, dl_a, dl_w = [], [], [], []
        for s in SEEDS:
            b, c = M.loc[("gradsc", s)], M.loc[(ctl, s)]
            lf, la, lw = (float(np.log(b[k] / c[k]).mean()) for k in ("fresh_fine", "fresh_assignment", "fresh_within"))
            assert abs(lf - la - lw) < 1e-3, (s, lf, la, lw)
            sh.append(100 * la / lf)
            dl_f.append(lf), dl_a.append(la), dl_w.append(lw)
        out[f"routing_share_vs_{ctl}"] = {"pct_of_dlog_fine_in_assignment": tci(sh), "dlog_fine": tci(dl_f),
                                          "dlog_assignment": tci(dl_a), "dlog_within": tci(dl_w)}
    return out


def persistence():
    from persist_analyze import C as RC, cohort_masks
    from persist_lib import CORE_LANES, load_lane, records_real
    rows = []
    for lane, arch, seed in CORE_LANES:
        d, rr = load_lane(lane)
        at = rr["metadata"]["analysis_thresholds"]
        thr = {"sai": at["sai_task_general_thresh"], "ssi": at["ssi_abstraction_thresh"],
               "csi": at["csi_differentiation_thresh"]}
        R = records_real(d, rr, "paper", dedup=True)
        m = cohort_masks(R, thr, "methods")
        life, cens = R[:, RC["life"]], R[:, RC["cens"]]
        r = {"arch": arch, "seed": seed}
        for k in ("coarse", "fine"):
            r[f"mean_life_{k}"] = float(life[m[k]].mean())
            r[f"median_life_{k}"] = float(np.median(life[m[k]]))
            r[f"pct_alive_at_end_{k}"] = float(100 * cens[m[k]].mean())
            r[f"n_{k}"] = int(m[k].sum())
        r["mean_ratio"] = r["mean_life_coarse"] / r["mean_life_fine"]
        rows.append(r)
        print(arch, seed, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
    P = pd.DataFrame(rows)
    P.to_csv(OUT / "persistence.csv", index=False)
    return {"median_life_coarse_range": [P.median_life_coarse.min(), P.median_life_coarse.max()],
            "median_life_fine_range": [P.median_life_fine.min(), P.median_life_fine.max()],
            "runs_median_coarse_gt_fine": int((P.median_life_coarse > P.median_life_fine).sum()),
            "pct_alive_end_coarse_range": [P.pct_alive_at_end_coarse.min(), P.pct_alive_at_end_coarse.max()],
            "pct_alive_end_fine_range": [P.pct_alive_at_end_fine.min(), P.pct_alive_at_end_fine.max()],
            "runs_alive_end_coarse_gt_fine": int((P.pct_alive_at_end_coarse > P.pct_alive_at_end_fine).sum()),
            "mean_ratio_range": [P.mean_ratio.min(), P.mean_ratio.max()]}


def forest(write=True):
    """% of the control's early gain over chance lost, per seed (negative = loss)."""
    rows = []

    def add(name, unit, ctl, blk, group):
        v = 100 * (np.asarray(blk) - np.asarray(ctl)) / (np.asarray(ctl) - (5.0 if unit == "probe" else 1.0))
        rows.append({"row": name, "unit": unit, "group": group, **tci(v)})

    A = L.static_accuracy()
    P = L.static_probe()
    hn = pd.read_csv(RESULTS / "NOISE_matched/noise_matched_per_seed.csv").set_index(["arm", "seed"])
    i13, i110 = [1, 2, 3], list(range(1, 11))
    add("label noise, between - within", "fine", A["within_sc"][:, i110].mean(1), A["between_sc"][:, i110].mean(1),
        "noise")
    mt = np.array([L._val_curve(RUNS_EXTENDED / f"static_noise_s{s}/lanes/{ARMS['matched']}")[i110].mean() for s in SEEDS])
    add("label noise, similarity-matched - within", "fine", A["within_sc"][:, i110].mean(1), mt, "noise")
    add("label noise, between - within (superclass probe, epoch 5)", "probe", P["within_sc"][5].values,
        P["between_sc"][5].values, "noise")
    add("label noise, matched - within (superclass probe, epoch 5)", "probe",
        [hn.loc[("within", s), "probe_5"] for s in SEEDS], [hn.loc[("matched", s), "probe_5"] for s in SEEDS], "noise")
    D = pd.read_csv(GBL)
    M = D[D.epoch.isin(i13)].groupby(["arm", "seed"]).val_accuracy.mean()
    g = lambda a: np.array([M[(a, s)] for s in SEEDS])  # noqa: E731
    add("blocking, superclass - random directions", "fine", g("gradrand"), g("gradsc"), "blocking")
    add("blocking, superclass - energy-matched", "fine", g("gradenergy"), g("gradsc"), "blocking")
    add("blocking, random grouping - energy-matched", "fine", g("gradenergy"), g("gradrg"), "blocking")
    add("blocking, stage-matched pair (layer2 to pooled)", "fine", g("gradenergy_l2"), g("gradsc_l2"), "blocking")
    S = pd.read_csv(RESULTS / "CRITICAL_period/val_curves.csv")
    S = S[(S.window == "1-10") & (S.t0 == False) & S.epoch.isin(i13)].groupby(["arm", "seed"]).val_accuracy.mean()  # noqa: E712
    sp = lambda a: np.array([S[(a, s)] for s in SEEDS])  # noqa: E731
    add("suppression (use), superclass - random directions", "fine", sp("random"), sp("coarse"), "suppression")
    add("suppression (use), superclass - sham", "fine", sp("sham"), sp("coarse"), "suppression")
    F = pd.DataFrame(rows)
    if write:
        F.to_csv(OUT / "forest.csv", index=False)
    return F


SYNTH = [  # (label, family, targeted)
    ("Label noise, between vs within", "label noise", True),
    ("Label noise, similarity-matched vs within", "label noise", False),
    ("Blocking vs random directions", "blocking", True),
    ("Blocking vs jointly matched", "blocking", True),
    ("Random-grouping block vs jointly matched", "blocking", True),
    ("Upper-stage block, fully matched", "blocking", False),
    ("Random-direction block vs sham", "blocking", False),
    ("Jointly matched block vs sham", "blocking", False),
    ("Suppression vs random directions", "suppression", True),
    ("Suppression vs sham", "suppression", True),
]


def synthesis():
    """Per manipulation and seed: superclass learning lost and fine-grained learning lost, each as % of the
    control's gain over chance (superclass chance 5%, fine-class chance 1%).
      label noise  superclass probe at epoch 5 (first checkpoint) and fine-class accuracy over epochs 1-10, vs
                   within-superclass noise at the same rate (the rows of forest())
      blocking     superclass assignment of the fresh ridge read-out and the network's fine-class accuracy, epochs 1-3
                   (GRADBLOCK/final_long.csv)
      suppression  the network's own predictions: superclass assignment and fine-class accuracy, epochs 1-3
                   (ORACLE_within/oracle_long.csv)"""
    def pct(ctl, blk, chance):
        return 100 * (np.asarray(blk, float) - np.asarray(ctl, float)) / (np.asarray(ctl, float) - chance)
    hn = pd.read_csv(RESULTS / "NOISE_matched/noise_matched_per_seed.csv").set_index(["arm", "seed"])
    g = lambda a, c: [hn.loc[(a, s), c] for s in SEEDS]  # noqa: E731
    D = pd.read_csv(GBL)
    M = D[D.epoch.isin([1, 2, 3])].groupby(["arm", "seed"])[["val_accuracy", "fresh_assignment"]].mean()
    G = lambda a, c: [M.loc[(a, s), c] for s in SEEDS]  # noqa: E731
    O = pd.read_csv(RESULTS / "ORACLE_within/oracle_long.csv")
    O = O[O.epoch.isin([1, 2, 3])].groupby(["arm", "seed"])[["net_assignment", "net_fine"]].mean()
    Q = lambda a, c: [O.loc[(a, s), c] for s in SEEDS]  # noqa: E731
    F = forest(write=False).set_index("row")
    fs = lambda k: np.array(F.loc[k, "per_seed"], float)  # noqa: E731  (% change, as the forest rows)
    pairs = {
        "Label noise, between vs within": (fs("label noise, between - within (superclass probe, epoch 5)"),
                                           fs("label noise, between - within")),
        "Label noise, similarity-matched vs within": (fs("label noise, matched - within (superclass probe, epoch 5)"),
                                                      fs("label noise, similarity-matched - within"))}
    for lab, a, b in (("Blocking vs random directions", "gradsc", "gradrand"),
                      ("Blocking vs jointly matched", "gradsc", "gradjoint"),
                      ("Random-grouping block vs jointly matched", "gradrg", "gradjoint"),
                      ("Upper-stage block, fully matched", "gradsc_l3", "gradjoint_l3"),
                      ("Random-direction block vs sham", "gradrand", "sham"),
                      ("Jointly matched block vs sham", "gradjoint", "sham")):
        pairs[lab] = (pct(G(b, "fresh_assignment"), G(a, "fresh_assignment"), 5), pct(G(b, "val_accuracy"), G(a, "val_accuracy"), 1))
    for lab, a, b in (("Suppression vs random directions", "supp_sc", "supp_rand"), ("Suppression vs sham", "supp_sc", "sham")):
        pairs[lab] = (pct(Q(b, "net_assignment"), Q(a, "net_assignment"), 5), pct(Q(b, "net_fine"), Q(a, "net_fine"), 1))
    rows = []
    for lab, fam, tgt in SYNTH:
        sc, fi = pairs[lab]
        a, b = tci(sc), tci(fi)
        rows.append({"manipulation": lab, "family": fam, "targeted": tgt, "superclass_lost_pct": -a["mean"],
                     "superclass_ci": [-a["ci"][1], -a["ci"][0]], "fine_lost_pct": -b["mean"],
                     "fine_ci": [-b["ci"][1], -b["ci"][0]],
                     **{f"superclass_seed_{s}": -x for s, x in zip(SEEDS, sc)},
                     **{f"fine_seed_{s}": -x for s, x in zip(SEEDS, fi)}})
    S = pd.DataFrame(rows)
    r = float(np.corrcoef(S.superclass_lost_pct, S.fine_lost_pct)[0, 1])
    rho, p = stats.spearmanr(S.superclass_lost_pct, S.fine_lost_pct)
    slope, icpt = np.polyfit(S.superclass_lost_pct, S.fine_lost_pct, 1)
    return S, {"pearson_r": r, "spearman_rho": float(rho), "spearman_p": float(p), "slope": float(slope),
               "intercept": float(icpt), "n": len(S)}


def _lag(t, v, tc, vc):
    """Epochs behind: t minus the (interpolated) epoch at which the control curve (tc, vc) first reached v."""
    vc = np.maximum.accumulate(np.asarray(vc, float))
    if v <= vc[0]:
        return t - tc[0]
    if v >= vc[-1]:
        return np.nan
    return t - float(np.interp(v, vc, tc))


def blocking_lags(pairs=(("gradsc", "gradjoint"), ("gradsc", "gradrand"))):
    """Per seed and epoch: epochs a blocked arm is behind a control in superclass assignment (fresh read-out; chance
    5% at epoch 0) and in fine-class accuracy (chance 1% at epoch 0); epochs with a fresh read-out only."""
    D = pd.read_csv(GBL).dropna(subset=["fresh_assignment"])
    rows = []
    for blk, ctl in pairs:
        for s in SEEDS:
            c = D[(D.arm == ctl) & (D.seed == s)].sort_values("epoch")
            b = D[(D.arm == blk) & (D.seed == s)].set_index("epoch")
            for e in sorted(b.index):
                r = {"blocked": blk, "control": ctl, "seed": s, "epoch": int(e)}
                for col, k, ch in (("fresh_assignment", "superclass", 5.0), ("val_accuracy", "fine", 1.0)):
                    r[f"lag_{k}"] = _lag(e, b.loc[e, col], np.r_[0, c.epoch.values], np.r_[ch, c[col].values])
                rows.append(r)
    return pd.DataFrame(rows)


def lag_proportion(G, blk="gradsc", ctl="gradjoint", epochs=range(1, 11)):
    """How far fine-grained learning falls behind per epoch superclass learning falls behind: per seed, the slope
    through the origin of the fine-class lag on the superclass lag over the block (epochs 1-10), then t-CI over seeds;
    also the lags at the end of the block (epoch 10) and their ratio."""
    g = G[(G.blocked == blk) & (G.control == ctl)]
    sl, e10s, e10f = [], [], []
    for s in SEEDS:
        x = g[(g.seed == s) & g.epoch.isin(list(epochs))].dropna(subset=["lag_superclass", "lag_fine"])
        sl.append(float((x.lag_superclass * x.lag_fine).sum() / (x.lag_superclass ** 2).sum()))
        e = g[(g.seed == s) & (g.epoch == max(epochs))].iloc[0]
        e10s.append(float(e.lag_superclass)), e10f.append(float(e.lag_fine))
    return {"slope_fine_on_superclass": tci(sl), "lag_superclass_end": tci(e10s), "lag_fine_end": tci(e10f),
            "ratio_end": tci(np.array(e10f) / np.array(e10s))}


def milestones(t_sc=45.0, t_fine=30.0):
    """Per blocking network (every arm and seed of GRADBLOCK/final_long.csv): epoch at which superclass assignment
    reaches t_sc and fine-class accuracy reaches t_fine (linear interpolation from chance at epoch 0)."""
    D = pd.read_csv(GBL)
    rows = []
    for (a, s), g in D.groupby(["arm", "seed"]):
        g = g.sort_values("epoch")
        e = np.r_[0, g.epoch.values]
        rows.append({"arm": a, "seed": int(s),
                     "epoch_superclass": first_reach(e, np.r_[5.0, g.fresh_assignment.values], t_sc),
                     "epoch_fine": first_reach(e, np.r_[1.0, g.val_accuracy.values], t_fine)})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"label_noise": label_noise(), "blocking": blocking()}
    F = forest()
    res["forest"] = F.to_dict("records")
    S, st = synthesis()
    S.to_csv(OUT / "synthesis.csv", index=False)
    res["synthesis"] = {"rows": S.to_dict("records"), **st}
    res["persistence"] = persistence()
    json.dump(res, open(OUT / "units.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "forest"}, indent=1, default=float)[:6000])
    print(F[["row", "unit", "mean", "ci", "n_below_zero"]].to_string())


if __name__ == "__main__":
    main()
