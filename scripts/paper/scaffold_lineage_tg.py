#!/usr/bin/env python3
"""Lineage check: are task-general features the building blocks of later ones?

Lanes: the five clean static-noise lanes (static_noise_s{S}/standard_resnet18), all real checkpoints (epochs 0-30).
Layers: layer1, layer2, layer3, layer4, avgpool (layer4 and avgpool = two SAEs on identical activations).

Roles (scaffold_profile_lib.roles): alive = active on >= 1 % of samples; tg = SAI > 0.9;
cs = CSI > t_csi (NaN -> no; may overlap tg); ss = SSI > t_ssi & ~cs & ~tg (exclusive rule); other = alive, none of these.
Exclusive label for breakdowns / permutations: tg > cs > ss > other. Thresholds: tk 'layer' (per-layer alive-only null p95,
primary) and tk 'pooled' (pooled p95, sensitivity); SAI 0.9 fixed.

Matching: pipeline Phase 5 (noise_null_lib.match = saeanalysis._phase5_feature_matching, Hungarian on activation correlations).
Predecessor of feature j at t+1 = the feature i at t with (i, j) stable (r >= 0.5) or transformed (0.2 <= r < 0.5);
features alive at t+1 without one are 'born'.

Per transition x layer x target category T at t+1 (ss, cs, and every exclusive label):
  observed  = share of T features with a predecessor whose predecessor is task-general (tg at t)
  chance    = share of task-general features among all matched predecessors at t (alive, r >= 0.2)
  enrichment = observed / chance; also counts, born share, full predecessor breakdown.
Chains (>= 2 transitions): tg(t) -> ss(t+1) -> cs(t+2) (plus tg -> ss -> ss, tg -> tg -> ss), and the 2-step
tg(t) -> ss(t+1) / tg(t) -> cs(t+1) counts, against 100 permutations of the role tuple among alive features within each
checkpoint (independently per checkpoint; counts, overlaps and the matching are preserved).
Same-data null: the same analysis between the layer4 and avgpool SAEs at the same checkpoint (both directions):
descent there is pure matching / categorisation noise.

Usage: scaffold_lineage_tg.py run ; analyze
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scaffold_profile_lib as E  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from noise_null_lib import match  # noqa: E402

OUT = E.OUT / "lineage_tg"
CD = OUT / "cache"
TKS = ("layer", "pooled")
N_PERM = 100
LAB = {1: "tg", 2: "cs", 3: "ss", 4: "other"}
C_SAI, C_SSI, C_CSI = "#8B5CF6", "#3B82F6", "#F59E0B"
G1, G2, G3 = "#374151", "#6B7280", "#9CA3AF"


def role_arrays(z, nl, lab, L, tk):
    R = E.roles(z[f"{lab}|{L}|cnt"], z[f"{lab}|{L}|feature_ssi"].astype(np.float64),
                z[f"{lab}|{L}|feature_csi"].astype(np.float64), z[f"{lab}|{L}|feature_sai"].astype(np.float64),
                nl["alive_min"], E.thresholds(nl, L, tk))
    ex = np.zeros(len(R["alive"]), int)
    ex[R["alive"]] = 4
    ex[R["ss"]] = 3
    ex[R["cs"] & ~R["tg"]] = 2
    ex[R["tg"]] = 1
    return {"alive": R["alive"], "tg": R["tg"], "ss": R["ss"], "cs": R["cs"], "ex": ex}


def pred_of(md, n_b):
    p = np.full(n_b, -1, int)
    for i, j, _ in md["stable"]:
        p[j] = i
    for i, j, _ in md["transformed"]:
        p[j] = i
    return p


def descent(A, B, pred, alive_b):
    """Rows of observed / chance per target category at b."""
    has = (pred >= 0) & alive_b
    preds = pred[has]
    chance = float(A["tg"][preds].mean()) if len(preds) else np.nan
    out = []
    targets = {"ss": B["ss"], "cs": B["cs"], **{f"ex_{v}": B["ex"] == k for k, v in LAB.items()}}
    for tn, T in targets.items():
        T = T & alive_b
        w = T & has
        n_tg = int(A["tg"][pred[w]].sum()) if w.any() else 0
        r = {"target": tn, "n_target": int(T.sum()), "n_with_pred": int(w.sum()), "n_born": int((T & ~has).sum()),
             "n_from_tg": n_tg, "obs_share_from_tg": n_tg / w.sum() if w.any() else np.nan, "chance_tg_share": chance,
             "n_matched_pred": int(len(preds)), "n_tg_matched_pred": int(A["tg"][preds].sum()) if len(preds) else 0}
        r["enrichment"] = r["obs_share_from_tg"] / chance if chance and np.isfinite(r["obs_share_from_tg"]) else np.nan
        r["share_born"] = r["n_born"] / r["n_target"] if r["n_target"] else np.nan
        for k, v in LAB.items():
            r[f"pred_{v}"] = int((A["ex"][pred[w]] == k).sum()) if w.any() else 0
        out.append(r)
    return out


def lane_run(ln):
    key, d = ln["key"], Path(ln["dir"])
    E.guard(d)
    t0 = time.time()
    nl, z = E.null_json(key), E.sel_npz(key)
    ci = E.ckpts(d)
    labs = [c["label"] for c in ci]
    ep = {c["label"]: c["epoch"] for c in ci}
    H = {(lab, L): E.load_codes(d, lab, L) for lab in labs for L in E.LAYERS}
    preds = {}
    for a, b in zip(labs[:-1], labs[1:]):
        for L in E.LAYERS:
            md = match(H[(a, L)], H[(b, L)])
            preds[(a, b, L)] = (pred_of(md, H[(b, L)].shape[1]), np.asarray(md["alive_b"]))
    nulls = {}
    for lab in labs:
        for x, y in (("layer4", "avgpool"), ("avgpool", "layer4")):
            md = match(H[(lab, x)], H[(lab, y)])
            nulls[(lab, x, y)] = (pred_of(md, H[(lab, y)].shape[1]), np.asarray(md["alive_b"]))
    rows, nrows, crows = [], [], []
    for tk in TKS:
        Rr = {(lab, L): role_arrays(z, nl, lab, L, tk) for lab in labs for L in E.LAYERS}
        for a, b in zip(labs[:-1], labs[1:]):
            for L in E.LAYERS:
                p, al = preds[(a, b, L)]
                for r in descent(Rr[(a, L)], Rr[(b, L)], p, al):
                    rows.append({"key": key, "seed": ln["seed"], "tk": tk, "layer": L, "epoch_a": ep[a], "epoch_b": ep[b], **r})
        for lab in labs:
            for x, y in (("layer4", "avgpool"), ("avgpool", "layer4")):
                p, al = nulls[(lab, x, y)]
                for r in descent(Rr[(lab, x)], Rr[(lab, y)], p, al):
                    nrows.append({"key": key, "seed": ln["seed"], "tk": tk, "direction": f"{x}->{y}", "epoch": ep[lab], **r})
        # permutation tests: 2-step counts per transition and 3-checkpoint chains
        rng = np.random.RandomState(1000 + ln["seed"])
        for L in E.LAYERS:
            perms = []
            for _ in range(N_PERM):
                pr = {}
                for lab in labs:
                    al = np.flatnonzero(Rr[(lab, L)]["alive"])
                    idx = np.arange(len(Rr[(lab, L)]["alive"]))
                    idx[al] = al[rng.permutation(len(al))]
                    pr[lab] = idx
                perms.append(pr)

            def counts(perm):
                g = (lambda lab, k: Rr[(lab, L)][k] if perm is None else Rr[(lab, L)][k][perm[lab]])
                res = {}
                for a, b in zip(labs[:-1], labs[1:]):
                    p, al = preds[(a, b, L)]
                    has = (p >= 0) & al
                    for tn in ("ss", "cs"):
                        m = has & g(b, tn)
                        res[("2step", f"tg->{tn}", ep[a], ep[b])] = int(g(a, "tg")[p[m]].sum())
                for a, b, c in zip(labs[:-2], labs[1:-1], labs[2:]):
                    p1, al1 = preds[(a, b, L)]
                    p2, al2 = preds[(b, c, L)]
                    for nm, (ra, rb, rc) in {"tg->ss->cs": ("tg", "ss", "cs"), "tg->ss->ss": ("tg", "ss", "ss"),
                                            "tg->tg->ss": ("tg", "tg", "ss")}.items():
                        k = np.flatnonzero(al2 & (p2 >= 0) & g(c, rc))
                        j = p2[k]
                        j = j[g(b, rb)[j]]
                        i = p1[j]
                        i = i[i >= 0]
                        res[("chain", nm, ep[a], ep[c])] = int(g(a, ra)[i].sum())
                return res
            obs = counts(None)
            pc = [counts(pr) for pr in perms]
            for kk, v in obs.items():
                pv = np.array([q[kk] for q in pc], float)
                crows.append({"key": key, "seed": ln["seed"], "tk": tk, "layer": L, "kind": kk[0], "pattern": kk[1],
                              "epoch_start": kk[2], "epoch_end": kk[3], "observed": v, "perm_mean": pv.mean(),
                              "perm_sd": pv.std(ddof=1), "ratio_obs_perm": v / pv.mean() if pv.mean() > 0 else np.nan,
                              "p_perm_ge": float((1 + (pv >= v).sum()) / (1 + N_PERM))})
    CD.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(CD / f"{key}_descent.csv", index=False)
    pd.DataFrame(nrows).to_csv(CD / f"{key}_samedata.csv", index=False)
    pd.DataFrame(crows).to_csv(CD / f"{key}_perm.csv", index=False)
    return f"{key}: {len(rows)} descent, {len(nrows)} same-data, {len(crows)} perm rows ({time.time() - t0:.0f}s)"


# analysis
def pooled_enrich(g):
    """Count-pooled enrichment over seeds: sum(n_from_tg)/sum(n_with_pred) / (sum(n_tg_matched_pred)/sum(n_matched_pred))."""
    o = g.n_from_tg.sum() / g.n_with_pred.sum() if g.n_with_pred.sum() else np.nan
    c = g.n_tg_matched_pred.sum() / g.n_matched_pred.sum() if g.n_matched_pred.sum() else np.nan
    return pd.Series({"n_target": g.n_target.sum(), "n_with_pred": g.n_with_pred.sum(), "n_born": g.n_born.sum(),
                      "n_from_tg": g.n_from_tg.sum(), "obs_share": o, "chance": c, "enrichment_pooled": o / c if c else np.nan,
                      "enrichment_seed_mean": g.enrichment.mean(), "enrichment_seed_sd": g.enrichment.std(ddof=1),
                      "n_seeds_gt1": int((g.enrichment > 1).sum()), "n_seeds_finite": int(g.enrichment.notna().sum()),
                      **{f"pred_{v}": g[f"pred_{v}"].sum() for v in LAB.values()}})


def analyze():
    D = pd.concat([pd.read_csv(f) for f in sorted(CD.glob("*_descent.csv"))], ignore_index=True)
    N = pd.concat([pd.read_csv(f) for f in sorted(CD.glob("*_samedata.csv"))], ignore_index=True)
    P = pd.concat([pd.read_csv(f) for f in sorted(CD.glob("*_perm.csv"))], ignore_index=True)
    D.to_csv(OUT / "descent_per_seed.csv", index=False)
    N.to_csv(OUT / "samedata_per_seed.csv", index=False)
    P.to_csv(OUT / "permutation_per_seed.csv", index=False)
    td = D.groupby(["tk", "layer", "target", "epoch_a", "epoch_b"]).apply(pooled_enrich).reset_index()
    td.to_csv(OUT / "enrichment_by_transition.csv", index=False)
    D["phase"] = np.where(D.epoch_a == 0, "0->5", np.where(D.epoch_b <= 15, "5->15", "15->30"))
    tp = D.groupby(["tk", "layer", "target", "phase"]).apply(pooled_enrich).reset_index()
    tp.to_csv(OUT / "enrichment_by_phase.csv", index=False)
    N["phase"] = np.where(N.epoch == 0, "ep0", np.where(N.epoch <= 15, "5-15", "20-30"))
    tn = N.groupby(["tk", "direction", "target", "phase"]).apply(pooled_enrich).reset_index()
    tn.to_csv(OUT / "samedata_by_phase.csv", index=False)
    tn2 = N.groupby(["tk", "direction", "target", "epoch"]).apply(pooled_enrich).reset_index()
    tn2.to_csv(OUT / "samedata_by_epoch.csv", index=False)
    pp = P.groupby(["tk", "layer", "kind", "pattern", "epoch_start", "epoch_end"]).agg(
        observed=("observed", "sum"), perm_mean=("perm_mean", "sum"), n_seeds_p05=("p_perm_ge", lambda x: int((x <= 0.05).sum())),
        n_seeds_obs_gt_perm=("ratio_obs_perm", lambda x: int((x > 1).sum())), median_p=("p_perm_ge", "median")).reset_index()
    pp["ratio_pooled"] = pp.observed / pp.perm_mean
    pp.to_csv(OUT / "permutation_summary.csv", index=False)
    figure(td, tn2)
    return td, tp, tn, pp


def figure(td, tn2):
    layers = ["layer1", "layer2", "layer3", "layer4", "avgpool"]
    fig, axs = plt.subplots(2, 5, figsize=(14, 5.4), sharey="row")
    for i, (tgt, c, mk, nm) in enumerate((("ss", C_SSI, "o", "superclass-specific at t+1"),
                                          ("cs", C_CSI, "s", "class-selective at t+1"))):
        for j, L in enumerate(layers):
            ax = axs[i, j]
            for tk, ls in (("layer", "-"), ("pooled", "--")):
                g = td[(td.tk == tk) & (td.layer == L) & (td.target == tgt)].sort_values("epoch_b")
                ax.plot(g.epoch_b, g.enrichment_pooled, ls, color=c, marker=mk, ms=4 if tk == "layer" else 0,
                        lw=1.6 if tk == "layer" else 1.0, label=f"across time ({'per-layer' if tk == 'layer' else 'pooled'} thr.)")
            if L in ("layer4", "avgpool"):
                dirn = "layer4->avgpool" if L == "avgpool" else "avgpool->layer4"
                g = tn2[(tn2.tk == "layer") & (tn2.direction == dirn) & (tn2.target == tgt) & (tn2.epoch >= 5)].sort_values("epoch")
                ax.plot(g.epoch, g.enrichment_pooled, "-.", color=G3, lw=1.2, label="same-data null (identical activations)")
            ax.axhline(1, color=G2, lw=0.8)
            ax.axvspan(-1, 5, color="#F3F4F6", zorder=-5, lw=0)
            ax.set_xticks([5, 10, 20, 30])
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(L if L != "avgpool" else "avgpool (= layer4 data)", fontsize=8.5, color=G1)
            if j == 0:
                ax.set_ylabel(f"enrichment of task-general\npredecessors: {nm}", fontsize=7.5)
            if i == 1:
                ax.set_xlabel("epoch of t+1")
            ax.tick_params(labelsize=7)
    axs[0, 4].legend(fontsize=6.5, loc="upper right")
    fig.suptitle("Do superclass / class-selective features descend from task-general ones? observed / chance "
                 "(1 = category-blind lineage); 5 clean seeds count-pooled; shaded = first transition (0->5)",
                 x=0.01, ha="left", fontsize=9, color=G1)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig_lineage_tg.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "run":
        for ln in E.lanes():
            if ln["cond"] != "clean":
                continue
            if (CD / f"{ln['key']}_perm.csv").exists():
                print(ln["key"], "cached")
                continue
            print(lane_run(ln), flush=True)
    else:
        analyze()


if __name__ == "__main__":
    main()
