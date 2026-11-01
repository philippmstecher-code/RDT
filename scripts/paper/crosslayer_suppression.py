#!/usr/bin/env python3
"""Cross-layer suppression of the superclass code (xlayer): behaviour and snapshot-removal analyses.

Subcommands
  behaviour      fine validation accuracy of the cross-layer coarse arm minus sham_e00 (mean of epochs 1-3; epoch 30);
                 its difference to the last-block suppression effect (prevention, epochs 1-3); manipulation check
                 (logged superclass probe at layer2/layer3, epochs 2-10, coarse < random in >= 4/5 seeds); directions
                 added per re-estimation and logged probes after release; random - sham descriptives.
  snapshot       snapshot accuracy A_snap(e), e in 1, 2, 3, 5, 10: sham_e00 at epoch e with all five stages projected
                 at the ranks of the arm's accumulated projection (last `reproject` report with epoch <= e).
                 Coarse arm: superclass directions estimated iteratively on the projected activations
                 (mean_shift.run_one; images 0-19 per class, evr_floor 0.02, q_max 19, numerical-zero guard),
                 achieved ranks reported. Random arm: random orthonormal directions of the random arm's ranks.
                 Uncentred removal (as the training hooks) is the main quantity; centred removal is descriptive.
  snapshot_prev  the same for prevention (last block): random arm at its own ranks; the coarse snapshot is read
                 from MEAN_shift/lastblock_raw.csv (same procedure).
  analyze        indices D_window, D_release, D_30 and outcome criteria; cross-layer mean-shift summary.
CPU only (2 threads), read-only on all data. Writes analysis_outputs/paper/XLAYER/.

Usage: crosslayer_suppression.py behaviour | snapshot [seeds] [arms] | snapshot_prev | analyze
"""
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis_outputs" / "paper" / "XLAYER"
OUT.mkdir(parents=True, exist_ok=True)
EXP = Path(RDT_DATA + "/extended/experiments")
SEEDS = [7, 42, 137, 256, 314]
ALL = ["layer1", "layer2", "layer3", "layer4", "avgpool"]
LAST = ["layer4", "avgpool"]
SNAP_EPOCHS = [1, 2, 3, 5, 10]


# lanes
def xl_dir(arm, seed):
    cond = {"coarse": "scaffold", "random": "random_variance"}[arm]
    return EXP / "xlayer" / "lanes" / f"lesion_{cond}_e00_seed{seed}_rp65_Llayer1-layer2-layer3-layer4-avgpool"


def prev_dir(arm, seed):
    cond = {"coarse": "scaffold", "random": "random_variance"}[arm]
    root = "prevention_s42_copy" if seed == 42 else "prevention"
    return EXP / root / "lanes" / f"lesion_{cond}_e00_seed{seed}_rp65_Llayer4-avgpool"


def sham_dir(seed, e0=0):
    return EXP / "critical_sham" / "lanes" / f"sham_e{e0:02d}_seed{seed}"


def val_acc(lane):
    m = json.load(open(lane / "dev_snapshots" / "metrics" / "devtrain_metrics.json"))
    d = {int(r["epoch"]): float(r["val_accuracy"]) for r in m}
    return d


def lesion_rows(lane):
    return [json.loads(l) for l in open(lane / "dev_snapshots" / "metrics" / "lesion_log.jsonl")]


def ranks_at(lane, e, stages):
    rp = [r for r in lesion_rows(lane) if r["event"] == "reproject" and int(r["epoch"]) <= e]
    last = rp[-1]
    return {s: int(last["report"][s]["total_rank"]) for s in stages}, int(last["weight_updates"]), int(last["epoch"])


def probe_log(lane):
    p = lane / "dev_snapshots" / "metrics" / "lesion_probe_log.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


# statistics (as critical_period.paired / verdict_dir)
def paired(vals, name=""):
    seeds = sorted(s for s, v in vals.items() if v is not None and np.isfinite(v))
    d = np.array([vals[s] for s in seeds], dtype=np.float64)
    n = len(d)
    out = {"name": name, "n": n, "per_seed": {str(s): float(vals[s]) for s in seeds}}
    if n == 0:
        return out
    m = float(d.mean())
    out.update({"mean": m, "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()), "n_zero": int((d == 0).sum())})
    if n >= 2:
        sd = float(d.std(ddof=1))
        se = sd / math.sqrt(n)
        tq = float(stats.t.ppf(0.975, n - 1))
        out.update({"sd": sd, "se": se, "ci95": [m - tq * se, m + tq * se], "df": n - 1})
        if se > 0:
            out.update({"t": m / se, "p_two_sided": float(2 * stats.t.sf(abs(m / se), n - 1))})
    return out


def verdict_dir(r, sign, min_count=4):
    if r.get("n", 0) < 2 or "ci95" not in r:
        return "NOT SUPPORTED (insufficient seeds)"
    lo, hi = r["ci95"]
    ok_mean = r["mean"] * sign > 0
    ok_ci = lo > 0 if sign > 0 else hi < 0
    ok_cnt = (r["n_pos"] if sign > 0 else r["n_neg"]) >= min_count
    r["criteria"] = {"mean_direction": ok_mean, "ci_excludes_0": ok_ci, f"count_ge_{min_count}": ok_cnt}
    r["verdict"] = "SUPPORTED" if (ok_mean and ok_ci and ok_cnt) else "NOT SUPPORTED"
    return r["verdict"]


def jdump(obj, path):
    json.dump(obj, open(path, "w"), indent=1, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))


# behaviour
def behaviour():
    acc = {}
    for s in SEEDS:
        acc[("xl_coarse", s)] = val_acc(xl_dir("coarse", s))
        acc[("xl_random", s)] = val_acc(xl_dir("random", s))
        acc[("sham", s)] = val_acc(sham_dir(s))
        acc[("lb_coarse", s)] = val_acc(prev_dir("coarse", s))
        acc[("lb_random", s)] = val_acc(prev_dir("random", s))
    rows = []
    for (arm, s), d in acc.items():
        for e, v in d.items():
            rows.append({"arm": arm, "seed": s, "epoch": e, "val_acc": v})
    curves = pd.DataFrame(rows)
    curves.to_csv(OUT / "val_acc_curves.csv", index=False)

    def m(arm, s, eps):
        return float(np.mean([acc[(arm, s)][e] for e in eps]))

    E13 = [1, 2, 3]
    R = {}
    R["primary_xl_coarse_minus_sham_e1_3"] = paired({s: m("xl_coarse", s, E13) - m("sham", s, E13) for s in SEEDS},
                                                    "fine val acc, mean epochs 1-3, xlayer coarse - sham_e00")
    verdict_dir(R["primary_xl_coarse_minus_sham_e1_3"], -1)
    R["sec1_lasting_cost_e30"] = paired({s: acc[("xl_coarse", s)][30] - acc[("sham", s)][30] for s in SEEDS},
                                        "fine val acc epoch 30, xlayer coarse - sham_e00")
    verdict_dir(R["sec1_lasting_cost_e30"], -1)
    R["sec2_stronger_than_last_block_e1_3"] = paired(
        {s: (m("xl_coarse", s, E13) - m("sham", s, E13)) - (m("lb_coarse", s, E13) - m("sham", s, E13)) for s in SEEDS},
        "(xlayer coarse - sham) - (prevention coarse - sham), mean epochs 1-3")
    verdict_dir(R["sec2_stronger_than_last_block_e1_3"], -1)
    # descriptive behaviour contrasts
    desc = {}
    desc["lb_coarse_minus_sham_e1_3"] = paired({s: m("lb_coarse", s, E13) - m("sham", s, E13) for s in SEEDS})
    desc["xl_random_minus_sham_e1_3"] = paired({s: m("xl_random", s, E13) - m("sham", s, E13) for s in SEEDS})
    desc["xl_coarse_minus_random_e1_3"] = paired({s: m("xl_coarse", s, E13) - m("xl_random", s, E13) for s in SEEDS})
    desc["xl_random_minus_sham_e30"] = paired({s: acc[("xl_random", s)][30] - acc[("sham", s)][30] for s in SEEDS})
    desc["xl_coarse_minus_random_e30"] = paired({s: acc[("xl_coarse", s)][30] - acc[("xl_random", s)][30] for s in SEEDS})
    desc["lb_coarse_minus_sham_e30"] = paired({s: acc[("lb_coarse", s)][30] - acc[("sham", s)][30] for s in SEEDS})
    desc["xl_coarse_minus_sham_window_e1_10"] = paired({s: m("xl_coarse", s, range(1, 11)) - m("sham", s, range(1, 11))
                                                        for s in SEEDS})
    desc["xl_coarse_minus_sham_e11_13"] = paired({s: m("xl_coarse", s, [11, 12, 13]) - m("sham", s, [11, 12, 13])
                                                  for s in SEEDS})
    for k in desc:
        verdict_dir(desc[k], -1 if "minus_sham" in k or "minus_random" in k else -1)
    per_epoch = {}
    for e in range(1, 31):
        per_epoch[e] = {a: float(np.mean([acc[(a, s)][e] for s in SEEDS]))
                        for a in ("xl_coarse", "xl_random", "sham", "lb_coarse", "lb_random")}
    R["descriptive"] = desc
    R["mean_curves"] = per_epoch

    # manipulation check (logged superclass probe, coarse < random, layer2 & layer3, epochs 2-10)
    pr = []
    for arm in ("coarse", "random"):
        for s in SEEDS:
            for r in probe_log(xl_dir(arm, s)):
                for st, v in r["results"].items():
                    pr.append({"arm": arm, "seed": s, "epoch": int(r["epoch"]), "phase": r["phase"], "stage": st,
                               "probe": float(v["superclass_probe_acc"])})
    P = pd.DataFrame(pr)
    P.to_csv(OUT / "probe_log_long.csv", index=False)
    W = P[(P.phase == "window") & P.epoch.between(2, 10)]
    sec3 = {}
    for lbl, stg in (("layer2_layer3_mean", ["layer2", "layer3"]), ("layer2", ["layer2"]), ("layer3", ["layer3"]),
                     ("layer1", ["layer1"]), ("layer4", ["layer4"]), ("avgpool", ["avgpool"])):
        g = W[W.stage.isin(stg)].groupby(["arm", "seed"]).probe.mean().unstack(0)
        diff = (g["coarse"] - g["random"]).to_dict()
        r = paired(diff, f"logged superclass probe {lbl}, mean epochs 2-10, xlayer coarse - random")
        r["coarse_mean"], r["random_mean"] = float(g["coarse"].mean()), float(g["random"].mean())
        r["n_coarse_lt_random"] = int((g["coarse"] < g["random"]).sum())
        r["count_rule_ge4"] = r["n_coarse_lt_random"] >= 4
        sec3[lbl] = r
    sec3["rule"] = ("criterion (label PASS): coarse < random in >= 4/5 seeds (applied to the mean over layer2 and "
                    "layer3; also reported per layer)")
    sec3["verdict"] = "PASS" if sec3["layer2_layer3_mean"]["count_rule_ge4"] else "FAIL"
    sec3["verdict_both_layers_separately"] = "PASS" if (sec3["layer2"]["count_rule_ge4"] and
                                                         sec3["layer3"]["count_rule_ge4"]) else "FAIL"
    R["sec3_manipulation_check"] = sec3

    # descriptive: n_added per re-estimation per layer; logged probes after release (11-15)
    na = []
    for arm in ("coarse", "random"):
        for s in SEEDS:
            for r in lesion_rows(xl_dir(arm, s)):
                if r["event"] != "reproject":
                    continue
                for st, v in r["report"].items():
                    na.append({"arm": arm, "seed": s, "epoch": int(r["epoch"]), "wu": int(r["weight_updates"]),
                               "stage": st, "n_added": int(v["n_added"]), "total_rank": int(v["total_rank"]),
                               "max_rank": int(v["max_rank"])})
    NA = pd.DataFrame(na)
    NA.to_csv(OUT / "reestimation_log_long.csv", index=False)
    R["sec4_n_added_per_reestimation"] = (NA.groupby(["arm", "stage", "seed"]).n_added.mean()
                                          .groupby(["arm", "stage"]).agg(["mean", "min", "max"]).round(3)
                                          .reset_index().to_dict(orient="records"))
    rel = P[P.phase != "window"]
    R["sec4_logged_probe_after_release"] = (rel.groupby(["phase", "epoch", "stage", "arm"]).probe.mean().round(4)
                                            .reset_index().to_dict(orient="records"))
    jdump(R, OUT / "behaviour_results.json")
    for k in ("primary_xl_coarse_minus_sham_e1_3", "sec1_lasting_cost_e30", "sec2_stronger_than_last_block_e1_3"):
        r = R[k]
        print(k, r["verdict"], "mean %.3f CI [%.3f, %.3f] n_neg %d" % (r["mean"], *r["ci95"], r["n_neg"]),
              {s: round(v, 2) for s, v in r["per_seed"].items()})
    for k, r in desc.items():
        print("  desc", k, "mean %.3f CI [%.3f, %.3f] n_neg %d" % (r["mean"], *r["ci95"], r["n_neg"]),
              {s: round(v, 2) for s, v in r["per_seed"].items()})
    print("sec3", sec3["verdict"], sec3["verdict_both_layers_separately"],
          {k: (round(v["coarse_mean"], 4), round(v["random_mean"], 4), v["n_coarse_lt_random"],
               {s: round(x, 4) for s, x in v["per_seed"].items()}) for k, v in sec3.items() if isinstance(v, dict)})
    return R


# snapshot
def _ms():
    sys.path.insert(0, str(REPO / "scripts" / "paper"))
    import mean_shift as ms  # no side effects at import
    return ms


def load_snapshot_rows():
    rows = []
    for f in sorted(OUT.glob("snapshot_rows*.jsonl")):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    return rows


def append_row(path, row):
    with open(path, "a") as fh:
        fh.write(json.dumps({k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}) + "\n")


def snapshot(seeds, arms):
    import torch
    ms = _ms()
    X, y = ms.cifar_test()
    idx = np.concatenate([np.flatnonzero(y == c)[:20] for c in range(100)])
    raw_path = OUT / f"snapshot_rows_s{'-'.join(map(str, seeds))}_{'-'.join(arms)}.jsonl"
    done = set((r["arm"], r["seed"], r["epoch"]) for r in load_snapshot_rows())
    # seed-7 coarse rows already computed by mean_shift (same code, same random seeds): reused
    prior = REPO / "analysis_outputs" / "paper" / "MEAN_shift" / "xlayer_raw.csv"
    for s in seeds:
        sl = sham_dir(s)
        for arm in arms:
            lane = xl_dir(arm, s)
            for e in SNAP_EPOCHS:
                if (arm, s, e) in done:
                    continue
                t0 = time.time()
                lab, va_sham = ms.epoch_label(sl, e)
                target, wu, rep_ep = ranks_at(lane, e, ALL)
                row = {"arm": arm, "seed": s, "epoch": e, "sham_label": lab, "rank_report_epoch": rep_ep,
                       "rank_report_wu": wu, "val_sham_logged": va_sham,
                       **{f"target_{k}": v for k, v in target.items()}}
                if arm == "coarse" and s == 7 and prior.exists():
                    p = pd.read_csv(prior)
                    p = p[(p.seed == 7) & (p.epoch == e)]
                    if len(p) == 1:
                        p = p.iloc[0]
                        assert all(int(p[f"target_{k}"]) == target[k] for k in ALL)
                        row.update({c: p[c] for c in p.index if c.startswith(("achieved_", "bsc_resid_", "A_"))
                                    or c == "passes"})
                        row["source"] = "MEAN_shift/xlayer_raw.csv"
                        append_row(raw_path, row)
                        print("reused", arm, s, e, flush=True)
                        continue
                net = ms.load_net(sl, lab)
                if arm == "coarse":
                    r = ms.run_one(net, X, y, idx, target, ALL, rseed=s * 1000 + e + 500, lastblock=False)
                    agree = float((r["base_logits"].argmax(1).numpy() == ms.stored_predictions(sl, lab)).mean())
                    row.update({"pred_agreement": agree, "passes": r["passes"], "A_sham": r["A_sham"],
                                **{f"achieved_{k}": v for k, v in r["achieved"].items()},
                                **{f"bsc_resid_{k}": v for k, v in r["bsc_resid"].items()},
                                **{k: v for k, v in r.items() if k.startswith("A_") and k != "A_sham"}})
                    json.dump(r["trace"], open(OUT / f"trace_snapshot_coarse_s{s}_e{e}.json", "w"))
                else:
                    base, _ = ms.forward(net, X)
                    _, A0 = ms.forward(net, X, capture=ALL, capture_idx=idx)
                    agree = float((base.argmax(1).numpy() == ms.stored_predictions(sl, lab)).mean())
                    C = {k: A0[k].shape[1] for k in ALL}
                    mus = {k: A0[k].mean(0) for k in ALL}
                    Qr = {k: ms.random_basis(C[k], target[k], (s * 1000 + e + 700) * 100 + j)
                          for j, k in enumerate(ALL)}
                    row.update({"pred_agreement": agree, "A_sham": ms.acc(base, y),
                                "A_randarm_unc": ms.acc(ms.forward(net, X, spec=Qr)[0], y),
                                "A_randarm_cen": ms.acc(ms.forward(net, X, spec=Qr, centred=True, mus=mus)[0], y)})
                row["source"] = "computed"
                if row.get("pred_agreement", 1) < 0.999:
                    print("ABORT normalisation check", arm, s, e, row["pred_agreement"])
                    return
                row["sec"] = time.time() - t0
                append_row(raw_path, row)
                print(arm, s, e, "target", target, {k: (round(v, 2) if isinstance(v, float) else v)
                                                   for k, v in row.items() if k.startswith(("A_", "achieved_"))},
                      "%.0fs" % row["sec"], flush=True)
                del net
                torch.set_num_threads(2)


def snapshot_prev(seeds):
    """prevention random arm: random orthonormal directions at its own last-block ranks (fast path)."""
    ms = _ms()
    X, y = ms.cifar_test()
    idx = np.concatenate([np.flatnonzero(y == c)[:20] for c in range(100)])
    rows = []
    for s in seeds:
        sl = sham_dir(s)
        for e in SNAP_EPOCHS:
            lab, _ = ms.epoch_label(sl, e)
            target, wu, rep_ep = ranks_at(prev_dir("random", s), e, LAST)
            net = ms.load_net(sl, lab)
            base, h4 = ms.layer4_raw(net, X)
            p4 = h4[idx].mean(dim=(2, 3))
            mus = {"layer4": p4.mean(0), "avgpool": p4.mean(0)}
            Qr = {k: ms.random_basis(512, target[k], (s * 1000 + e + 900) * 100 + j) for j, k in enumerate(LAST)}
            rows.append({"seed": s, "epoch": e, **{f"target_{k}": v for k, v in target.items()},
                         "A_sham": ms.acc(base, y),
                         "A_randarm_unc": ms.acc(ms.lastblock_logits(net, h4, Qr, False, mus), y),
                         "A_randarm_cen": ms.acc(ms.lastblock_logits(net, h4, Qr, True, mus), y)})
            print("prev random", rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "snapshot_prev_random.csv", index=False)


# snapshot indices
def analyze():
    snap = pd.DataFrame(load_snapshot_rows())
    assert not snap.duplicated(["arm", "seed", "epoch"]).any()
    R = {"note": "A in percent fine-class accuracy; D = A_arm - reference"}
    accs = {(a, s): val_acc(xl_dir(a, s)) for a in ("coarse", "random") for s in SEEDS}
    sham = {s: val_acc(sham_dir(s)) for s in SEEDS}
    rows = []
    for _, r in snap.iterrows():
        a, s, e = r.arm, int(r.seed), int(r.epoch)
        A_snap_unc = r.A_coarse_unc if a == "coarse" else r.A_randarm_unc
        A_snap_cen = r.A_coarse_cen if a == "coarse" else r.A_randarm_cen
        rows.append({"arm": a, "seed": s, "epoch": e, "A_arm": accs[(a, s)][e], "A_sham_logged": sham[s][e],
                     "A_sham_recomputed": r.A_sham, "A_snap_unc": A_snap_unc, "A_snap_cen": A_snap_cen,
                     "D_window_unc": accs[(a, s)][e] - A_snap_unc, "D_window_cen": accs[(a, s)][e] - A_snap_cen,
                     "snapshot_cost_unc": r.A_sham - A_snap_unc,
                     **{c: r[c] for c in snap.columns if c.startswith(("target_", "achieved_"))
                        and pd.notna(r[c])}})
    D = pd.DataFrame(rows).sort_values(["arm", "seed", "epoch"])
    D.to_csv(OUT / "xlayer_per_seed_epoch.csv", index=False)

    def dwin(a, eps, col="D_window_unc"):
        g = D[(D.arm == a) & D.epoch.isin(eps)].groupby("seed")[col].mean()
        return {int(k): float(v) for k, v in g.items()}

    def drel(a):
        return {s: float(np.mean([accs[(a, s)][e] - sham[s][e] for e in (11, 12, 13)])) for s in SEEDS}

    R["primary_D_release_coarse"] = paired(drel("coarse"), "D_release coarse: mean epochs 11-13 of A_arm - A_sham")
    verdict_dir(R["primary_D_release_coarse"], -1)
    R["sec_D_window_mean_e1_3_coarse"] = paired(dwin("coarse", [1, 2, 3]), "mean e=1,2,3 of D_window (uncentred), coarse")
    verdict_dir(R["sec_D_window_mean_e1_3_coarse"], -1)
    dc, dr = drel("coarse"), drel("random")
    R["sec_D_release_coarse_minus_random"] = paired({s: dc[s] - dr[s] for s in SEEDS}, "D_release coarse - random")
    verdict_dir(R["sec_D_release_coarse_minus_random"], -1)
    desc = {"D_release_random": paired(dr),
            "D_30_coarse": paired({s: accs[("coarse", s)][30] - sham[s][30] for s in SEEDS}),
            "D_30_random": paired({s: accs[("random", s)][30] - sham[s][30] for s in SEEDS}),
            "D_window_mean_e1_3_random": paired(dwin("random", [1, 2, 3])),
            "D_window_mean_e1_3_coarse_centred": paired(dwin("coarse", [1, 2, 3], "D_window_cen")),
            "D_window_mean_e1_3_coarse_minus_random": paired({s: dwin("coarse", [1, 2, 3])[s] - dwin("random", [1, 2, 3])[s]
                                                             for s in SEEDS})}
    for e in SNAP_EPOCHS:
        for a in ("coarse", "random"):
            desc[f"D_window_e{e}_{a}"] = paired(dwin(a, [e]))
            desc[f"snapshot_cost_e{e}_{a}"] = paired(dwin(a, [e], "snapshot_cost_unc"))
            desc[f"A_snap_e{e}_{a}"] = paired(dwin(a, [e], "A_snap_unc"))
            desc[f"A_arm_e{e}_{a}"] = paired(dwin(a, [e], "A_arm"))
    for k in desc:
        verdict_dir(desc[k], -1)
    R["descriptive"] = desc
    # achieved vs target ranks (coarse arm), and the early-stop note
    c = D[D.arm == "coarse"]
    rk = {}
    for st in ALL:
        rk[st] = {int(e): {"target_mean": float(g[f"target_{st}"].mean()), "achieved_mean": float(g[f"achieved_{st}"].mean()),
                           "achieved_min": int(g[f"achieved_{st}"].min()), "achieved_max": int(g[f"achieved_{st}"].max()),
                           "n_short": int((g[f"achieved_{st}"] < g[f"target_{st}"]).sum())}
                  for e, g in c.groupby("epoch")}
    R["coarse_ranks_target_vs_achieved"] = rk

    # prevention (descriptive, last block)
    lb_path = REPO / "analysis_outputs" / "paper" / "MEAN_shift" / "lastblock_raw.csv"
    if lb_path.exists() and (OUT / "snapshot_prev_random.csv").exists():
        lb = pd.read_csv(lb_path)
        lb = lb[lb.window == "w01_10"]
        pr = pd.read_csv(OUT / "snapshot_prev_random.csv")
        pacc = {(a, s): val_acc(prev_dir(a, s)) for a in ("coarse", "random") for s in SEEDS}
        prow = []
        for s in SEEDS:
            for e in SNAP_EPOCHS:
                cr = lb[(lb.seed == s) & (lb.epoch == e)].iloc[0]
                rr = pr[(pr.seed == s) & (pr.epoch == e)].iloc[0]
                prow.append({"seed": s, "epoch": e,
                             "D_window_coarse": pacc[("coarse", s)][e] - cr.A_coarse_unc,
                             "D_window_random": pacc[("random", s)][e] - rr.A_randarm_unc,
                             "A_snap_coarse": cr.A_coarse_unc, "A_snap_random": rr.A_randarm_unc,
                             "achieved_layer4": cr.achieved_layer4, "target_layer4": cr.target_layer4})
        PR = pd.DataFrame(prow)
        PR.to_csv(OUT / "xlayer_prevention_per_seed_epoch.csv", index=False)
        pd_ = {}
        for a in ("coarse", "random"):
            pd_[f"D_window_mean_e1_3_{a}"] = paired(PR[PR.epoch <= 3].groupby("seed")[f"D_window_{a}"].mean().to_dict())
            pd_[f"D_release_{a}"] = paired({s: float(np.mean([pacc[(a, s)][e] - sham[s][e] for e in (11, 12, 13)]))
                                           for s in SEEDS})
            pd_[f"D_30_{a}"] = paired({s: pacc[(a, s)][30] - sham[s][30] for s in SEEDS})
            for e in SNAP_EPOCHS:
                pd_[f"D_window_e{e}_{a}"] = paired(PR[PR.epoch == e].set_index("seed")[f"D_window_{a}"].to_dict())
        pd_["D_release_coarse_minus_random"] = paired({s: pd_["D_release_coarse"]["per_seed"][str(s)]
                                                       - pd_["D_release_random"]["per_seed"][str(s)] for s in SEEDS})
        for k in pd_:
            verdict_dir(pd_[k], -1)
        R["prevention_descriptive"] = pd_

    # mean-shift diagnostic, cross-layer part (all stages), coarse-arm ranks from the same snapshot runs
    cc = snap[snap.arm == "coarse"].copy()
    for k in ("coarse", "random", "random_ach"):
        for v in ("unc", "cen"):
            cc[f"cost_{k}_{v}"] = cc.A_sham - cc[f"A_{k}_{v}"]
    cc["F_coarse"] = (cc.cost_coarse_unc - cc.cost_coarse_cen) / cc.cost_coarse_unc
    cc["info_cen"] = cc.cost_coarse_cen - cc.cost_random_cen
    cc["info_cen_ach"] = cc.cost_coarse_cen - cc.cost_random_ach_cen
    cc.to_csv(OUT / "meanshift_xlayer_derived.csv", index=False)
    e13 = cc[cc.epoch <= 3].groupby("seed")
    mb = {"F_coarse_mean_e1_3": paired(e13.F_coarse.mean().to_dict()),
          "F_ratio_of_means_e1_3": paired(((e13.cost_coarse_unc.mean() - e13.cost_coarse_cen.mean())
                                           / e13.cost_coarse_unc.mean()).to_dict()),
          "info_cen_e1_3": paired(e13.info_cen.mean().to_dict()),
          "info_cen_ach_e1_3": paired(e13.info_cen_ach.mean().to_dict()),
          "cost_coarse_unc_e1_3": paired(e13.cost_coarse_unc.mean().to_dict()),
          "cost_coarse_cen_e1_3": paired(e13.cost_coarse_cen.mean().to_dict()),
          "cost_random_unc_e1_3": paired(e13.cost_random_unc.mean().to_dict()),
          "cost_random_cen_e1_3": paired(e13.cost_random_cen.mean().to_dict()),
          "cost_random_ach_cen_e1_3": paired(e13.cost_random_ach_cen.mean().to_dict()),
          "A_sham_e1_3": paired(e13.A_sham.mean().to_dict())}
    for k in mb:
        verdict_dir(mb[k], 1)
    mb["per_epoch_means"] = cc.groupby("epoch")[["A_sham", "A_coarse_unc", "A_coarse_cen", "A_random_unc",
                                                 "A_random_cen", "A_random_ach_unc", "A_random_ach_cen",
                                                 "F_coarse", "info_cen", "info_cen_ach"]].mean().round(3).to_dict(orient="index")
    R["meanshift_xlayer"] = mb
    jdump(R, OUT / "xlayer_results.json")
    for k in ("primary_D_release_coarse", "sec_D_window_mean_e1_3_coarse", "sec_D_release_coarse_minus_random"):
        r = R[k]
        print(k, r["verdict"], "mean %.3f CI [%.3f, %.3f] n_neg %d" % (r["mean"], *r["ci95"], r["n_neg"]),
              {s: round(v, 2) for s, v in r["per_seed"].items()})
    for k, r in desc.items():
        if "_e" not in k or k.startswith("D_window_mean"):
            print("  desc", k, "mean %.3f CI [%.3f, %.3f]" % (r["mean"], *r["ci95"]),
                  {s: round(v, 2) for s, v in r["per_seed"].items()})
    print(D.groupby(["arm", "epoch"])[["A_arm", "A_sham_logged", "A_snap_unc", "A_snap_cen", "D_window_unc"]].mean().round(2))
    print(json.dumps(rk, indent=0)[:1500])
    for k, r in R.get("prevention_descriptive", {}).items():
        if "_e1" in k or "_e2" in k or "_e3" in k or "_e5" in k or "_e10" in k:
            continue
        print("  prev", k, "mean %.3f CI [%.3f, %.3f]" % (r["mean"], *r["ci95"]), {s: round(v, 2) for s, v in r["per_seed"].items()})
    for k, r in mb.items():
        if k != "per_epoch_means":
            print("  mean-shift", k, "mean %.3f CI [%.3f, %.3f] n_pos %d" % (r["mean"], *r["ci95"], r["n_pos"]),
                  {s: round(v, 3) for s, v in r["per_seed"].items()})
    print(pd.DataFrame(mb["per_epoch_means"]).T)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "behaviour"
    if cmd == "behaviour":
        behaviour()
    elif cmd == "snapshot":
        seeds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else SEEDS
        arms = sys.argv[3].split(",") if len(sys.argv) > 3 else ["coarse", "random"]
        snapshot(seeds, arms)
    elif cmd == "snapshot_prev":
        snapshot_prev(SEEDS)
    elif cmd == "analyze":
        analyze()
