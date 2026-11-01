#!/usr/bin/env python3
"""Ancestry analysis: are class-selective features carved out of superclass-selective features of their own superclass?

Lineage = Phase-5 stable matches between consecutive epochs (r >= 0.5; recomputed, validated in extract.py).
Event = a class-selective feature at epoch e (2..15) whose stable predecessor at e-1 was not class-selective
(differentiation of an existing feature; births have no predecessor and are counted separately).
  direct ancestry   predecessor at e-1 was superclass-selective with best superclass = superclass of the new feature's
                    class (k = argmax class-mean |code| at e)
  chain ancestry    any feature on the backward chain (e-1, e-2, ..., 1) was superclass-selective for that superclass
Null (could arise by construction: a feature turning k-selective already prefers k, and a k-preferring feature's best
superclass is k's): identical events, predecessors and thresholds, with every superclass label replaced by a random
partition of the 100 classes into 20 groups of five (100 partitions); ancestry = predecessor group-selective for the
random group containing k (selectivity recomputed under that partition). Enrichment = observed / null mean.
Forward rate: among stable pairs whose feature at e-1 was not class-selective, the share that become class-selective
at e for a class of the feature's best superclass, for superclass-selective vs other features at e-1.
Common thresholds (one set per seed, the sham's). Census layers pooled; avgpool reported separately.
Windows: all (2-15), block (2-10), release (11-15).
Reads the cache written by extract.py; writes analyses/ancestry_long.csv and ancestry_summary.json.
Usage: python scripts/paper/lineage/ancestry.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lineage_lib import LAYERS, OUT, SEEDS, TRUE_G, Cache, common_thresholds, fmt, masks, random_partitions, sel_from_M, sib_enrich, tci  # noqa: E402

ARMS = ["sham", "gradjoint", "gradsc", "gradrg"]
NP = 100
PARTS = random_partitions(NP)
WIN = {"all": range(2, 16), "block": range(2, 11), "release": range(11, 16)}


def lane_rows(arm, s):
    c, t = Cache(arm, s), common_thresholds(s)
    rows = []
    for L in LAYERS:
        z = c.z(L)
        M, cnt = z["M"], z["cnt"]
        S = [sel_from_M(M[e]) for e in range(16)]
        K = [masks(S[e], cnt[e] >= 100, t) for e in range(16)]
        back = {e: dict(zip(*c.stable(L, e)[:2])) for e in range(1, 16)}
        # events
        ev = []                                    # (e, ib, ia, k)
        for e in range(2, 16):
            ia, ib, _ = c.stable(L, e)
            keep = K[e]["cl"][ib] & ~K[e - 1]["cl"][ia] & (cnt[e - 1][ia] >= 100)
            for a, b in zip(ia[keep], ib[keep]):
                ev.append((e, int(b), int(a), int(S[e]["best_class"][b])))
        born = {e: c.born(L, e) for e in range(2, 16)}
        n_born_cl = {e: int(K[e]["cl"][born[e]].sum()) for e in range(2, 16)}
        # chains
        chains = []
        for e, b, a, k in ev:
            ch, ee, x = [(e - 1, a)], e - 1, a
            while ee >= 2 and x in back[ee]:
                x = back[ee][x]; ee -= 1
                ch.append((ee, x))
            chains.append(ch)
        # true labels
        def anc_true(ch, k):
            return [bool(K[ee]["sc"][x] and S[ee]["best_g"][x] == TRUE_G[k]) for ee, x in ch]
        need = {}
        for ch in chains:
            for ee, x in ch:
                need.setdefault(ee, set()).add(x)
        need = {ee: np.array(sorted(v)) for ee, v in need.items()}
        # null labels: for each partition, group-selective state of needed columns
        null_direct = np.zeros((NP, len(ev)), bool)
        null_chain = np.zeros((NP, len(ev)), bool)
        for p, g in enumerate(PARTS):
            lab = {}
            for ee, cols in need.items():
                sg = sel_from_M(M[ee][:, cols], g)
                mg = masks(sg, cnt[ee][cols] >= 100, t)
                lab[ee] = dict(zip(cols.tolist(), np.where(mg["sc"], sg["best_g"], -1).tolist()))
            for i, ((e, b, a, k), ch) in enumerate(zip(ev, chains)):
                hits = [lab[ee][x] == g[k] for ee, x in ch]
                null_direct[p, i] = hits[0]
                null_chain[p, i] = any(hits)
        for i, ((e, b, a, k), ch) in enumerate(zip(ev, chains)):
            h = anc_true(ch, k)
            rows.append({"arm": arm, "seed": s, "layer": L, "epoch": e, "kind": "event", "direct": h[0],
                         "chain": any(h), "chain_len": len(ch), "pred_sc_any": bool(K[e - 1]["sc"][a]),
                         "pred_tg": bool(K[e - 1]["tg"][a]),
                         "pred_sib": float(sib_enrich(M[e - 1], np.array([a]), np.array([k]))[0]),
                         "pred_sib_null": float(np.mean([sib_enrich(M[e - 1], np.array([a]), np.array([k]), g)[0]
                                                         for g in PARTS[:20]])),
                         "null_direct": float(null_direct[:, i].mean()), "null_chain": float(null_chain[:, i].mean()),
                         "_nd": null_direct[:, i], "_nc": null_chain[:, i]})
        for e in range(2, 16):
            rows.append({"arm": arm, "seed": s, "layer": L, "epoch": e, "kind": "born_cl", "n": n_born_cl[e]})
            # forward rates
            ia, ib, _ = c.stable(L, e)
            ok = (cnt[e - 1][ia] >= 100) & ~K[e - 1]["cl"][ia]
            ia, ib = ia[ok], ib[ok]
            own = K[e]["cl"][ib] & (TRUE_G[S[e]["best_class"][ib]] == S[e - 1]["best_g"][ia])
            sc = K[e - 1]["sc"][ia]
            rows.append({"arm": arm, "seed": s, "layer": L, "epoch": e, "kind": "forward",
                         "n_sc": int(sc.sum()), "k_sc": int((own & sc).sum()),
                         "n_other": int((~sc).sum()), "k_other": int((own & ~sc).sum())})
            # per pair: does coarse (sibling) structure at e-1 predict differentiation at e?
            sb = sib_enrich(M[e - 1], ia, S[e - 1]["best_class"][ia])
            for x, cs_, y in zip(sb, S[e - 1]["csi"][ia], own):
                rows.append({"arm": arm, "seed": s, "layer": L, "epoch": e, "kind": "pair",
                             "sib_prev": float(x), "csi_prev": float(cs_), "diff": bool(y)})
    return rows


def strat_gap(P):
    """log2 sibling enrichment at e-1 of features that differentiate at e minus those that do not, within CSI
    quintiles at e-1 (weighted by differentiators per bin)."""
    P = P.copy()
    P["diff"] = P["diff"].astype(bool)
    P["ls"] = np.log2(P.sib_prev.clip(lower=1e-3))
    P["bin"] = pd.qcut(P.csi_prev.rank(method="first"), 5, labels=False)
    g, w = [], []
    for _, B in P.groupby("bin"):
        d, n = B[B["diff"]], B[~B["diff"]]
        if len(d) and len(n):
            g.append(d.ls.mean() - n.ls.mean()); w.append(len(d))
    return float(np.average(g, weights=w)) if w else np.nan


def summarise(D):
    res = {}
    lane_stats = []
    for (arm, s), G in D.groupby(["arm", "seed"]):
        for w, eps in WIN.items():
            for lay in ("pooled", "avgpool"):
                g = G[G.epoch.isin(list(eps))]
                if lay == "avgpool":
                    g = g[g.layer == "avgpool"]
                ev = g[g.kind == "event"]
                fw = g[g.kind == "forward"]

                nb = g[g.kind == "born_cl"].n.sum()
                o = {"arm": arm, "seed": s, "window": w, "layers": lay, "n_events": len(ev), "n_born_cl": int(nb)}
                pr = g[g.kind == "pair"]
                o["diff_sib_gap_csi_strat"] = strat_gap(pr) if pr["diff"].astype(bool).sum() else np.nan
                o["diff_sib_gap_raw"] = float(np.log2(pr[pr["diff"].astype(bool)].sib_prev.mean() /
                                                      pr[~pr["diff"].astype(bool)].sib_prev.mean()))
                if len(ev):
                    for k in ("direct", "chain"):
                        obs = ev[k].mean()
                        nulls = np.stack(ev[f"_n{k[0]}"].values).mean(0)   # per partition rate
                        o[f"{k}_obs"] = float(obs); o[f"{k}_null"] = float(nulls.mean())
                        o[f"{k}_enrich"] = float(obs / nulls.mean()) if nulls.mean() > 0 else np.nan
                        o[f"{k}_p"] = float((1 + (nulls >= obs).sum()) / (1 + len(nulls)))
                    o["pred_sc_any"] = float(ev.pred_sc_any.mean())
                    o["log2_pred_sib"] = float(np.log2(ev.pred_sib.mean() / ev.pred_sib_null.mean()))
                    o["share_from_existing"] = len(ev) / (len(ev) + nb)
                if len(fw):
                    o["fwd_sc"] = fw.k_sc.sum() / max(fw.n_sc.sum(), 1)
                    o["fwd_other"] = fw.k_other.sum() / max(fw.n_other.sum(), 1)
                    o["fwd_ratio"] = o["fwd_sc"] / o["fwd_other"] if o["fwd_other"] > 0 else np.nan
                lane_stats.append(o)
    LS = pd.DataFrame(lane_stats)
    for k in ("direct", "chain"):
        if f"{k}_obs" in LS:
            LS[f"{k}_minus_null"] = LS[f"{k}_obs"] - LS[f"{k}_null"]
    res["lanes"] = LS.round(5).to_dict("records")
    for w in WIN:
        for lay in ("pooled", "avgpool"):
            X = LS[(LS.window == w) & (LS.layers == lay)].set_index(["arm", "seed"])
            for arm in ARMS:
                if arm not in X.index.get_level_values(0):
                    continue
                A = X.loc[arm].reindex(SEEDS)
                key = f"{w}|{lay}|{arm}"
                res[key] = {m: tci(A[m].values) for m in ("direct_obs", "direct_null", "chain_obs", "chain_null",
                                                          "direct_minus_null", "chain_minus_null", "log2_pred_sib",
                                                          "fwd_sc", "fwd_other", "share_from_existing", "n_events",
                                                          "diff_sib_gap_csi_strat", "diff_sib_gap_raw")
                            if m in A}
                if "direct_enrich" in A:
                    res[key]["log2_direct_enrich"] = tci(np.log2(A.direct_enrich.values.astype(float)))
                    res[key]["log2_chain_enrich"] = tci(np.log2(A.chain_enrich.values.astype(float)))
                    res[key]["log2_fwd_ratio"] = tci(np.log2(A.fwd_ratio.values.astype(float)))
                    res[key]["perm_p_direct"] = A.direct_p.round(4).tolist()
                    res[key]["perm_p_chain"] = A.chain_p.round(4).tolist()
            for x, y in (("gradsc", "gradjoint"), ("gradsc", "sham"), ("gradjoint", "sham"), ("gradrg", "sham")):
                if x in X.index.get_level_values(0) and y in X.index.get_level_values(0):
                    a, b = X.loc[x].reindex(SEEDS), X.loc[y].reindex(SEEDS)
                    res[f"{w}|{lay}|{x} - {y}"] = {m: tci((a[m] - b[m]).values) for m in
                                                   ("direct_obs", "chain_obs", "direct_minus_null", "chain_minus_null",
                                                    "share_from_existing", "fwd_sc", "log2_pred_sib") if m in a}
    return res


def main():
    import multiprocessing as mp
    jobs = [(a, s) for a in ARMS for s in SEEDS if (Cache(a, s).dir / "DONE.json").exists()]
    with mp.get_context("spawn").Pool(2) as pool:
        R = pool.starmap(lane_rows, jobs)
    D = pd.DataFrame([r for rr in R for r in rr])
    D[D.kind != "pair"].drop(columns=["_nd", "_nc"]).to_csv(OUT / "ancestry_long.csv", index=False)
    res = summarise(D)
    json.dump(res, open(OUT / "ancestry_summary.json", "w"), indent=1, default=float)
    for k, v in res.items():
        if k == "lanes":
            continue
        print(k)
        for m, r in v.items():
            if isinstance(r, dict) and "ci95" in r:
                print(f"   {m:24s} {fmt(r)}")
            else:
                print(f"   {m:24s} {r}")


if __name__ == "__main__":
    main()
