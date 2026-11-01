#!/usr/bin/env python3
"""Do coarse features split into finer ones over training ("splitting" / progressive
differentiation), or are class-selective features built as new features ("built within")?

Extends scaffold_lineage_tg.py (same lanes, checkpoints, roles, thresholds and matching) from task-general
predecessors to every predecessor category.

Lanes: five clean static-noise lanes (static_noise_s{S}/standard_resnet18), real checkpoints epochs 0..30
(duplicate final milestone dropped by scaffold_profile_lib.ckpts / real_ckpts). Layers: layer1-4, avgpool.
Roles: scaffold_profile_lib.roles (exclusive rule), exclusive label tg > cs > ss > other (as lineage_tg).
Thresholds: tk 'layer' (per-layer null p95, primary) and 'pooled' (sensitivity).
Matching: pipeline Phase 5 (Hungarian on activation correlations, noise_null_lib.match). Predecessor = Hungarian partner with
r >= 0.5 (stable) or 0.2 <= r < 0.5 (transformed); alive at t+1 without one = born; alive at t without one = no successor.
Split-aware variant ("best parent"): each alive feature at t+1 takes the alive feature at t it correlates with most
(many-to-one allowed, r >= 0.2), so two children of one parent are both counted.

Per seed x tk x layer x transition we store the source x target contingency (exclusive labels) for stable and
transformed pairs, births per target, deaths per source, the best-parent contingency, and N_PERM permuted contingencies
(source labels shuffled among the matched predecessors of that transition x layer; for best parent, among the distinct
parents). Same-data control: layer4 <-> avgpool SAEs at the same checkpoint (identical activations), both directions.

Usage: feature_origins.py run ; analyze
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scaffold_profile_lib as E  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from noise_null_lib import match  # noqa: E402
from noise_scaffold_compute import FINE_TO_COARSE  # noqa: E402
from scaffold_lineage_tg import role_arrays  # noqa: E402

OUT = E.REPO / "analysis_outputs" / "paper" / "ORIGINS"
CD = OUT / "cache"
TKS = ("layer", "pooled")
N_PERM = 1000
CATS = ["tg", "cs", "ss", "other"]  # exclusive codes 1..4 -> index 0..3
RNG_CONV = np.random.RandomState(7)
PAIRS = [("ss", "cs"), ("tg", "cs"), ("tg", "ss"), ("cs", "cs"), ("ss", "ss"), ("other", "cs"), ("ss", "tg")]


def phase_of(ea, eb):
    return "0->5" if ea == 0 else ("5->15" if eb <= 15 else "15->30")


def link(md, C, n_a, n_b):
    """Hungarian predecessor (+ type) and best-parent arrays for features at b."""
    pred = np.full(n_b, -1, int)
    typ = np.zeros(n_b, int)  # 1 stable, 2 transformed
    for i, j, _ in md["stable"]:
        pred[j], typ[j] = i, 1
    for i, j, _ in md["transformed"]:
        pred[j], typ[j] = i, 2
    al_a, al_b = np.asarray(md["alive_a"]), np.asarray(md["alive_b"])
    bp = np.full(n_b, -1, int)
    if al_a.any():
        Cs = np.where(al_a[:, None], np.asarray(C), -np.inf)
        arg = Cs.argmax(0)
        ok = (Cs[arg, np.arange(n_b)] >= 0.2) & al_b
        bp[ok] = arg[ok]
    return {"pred": pred, "typ": typ, "bp": bp, "alive_a": al_a, "alive_b": al_b}


def prefs(h):
    """Preferred class and preferred superclass (argmax of mean code; samples class-major, CIFAR-100 coarse map)."""
    mc = h.reshape(E.N_CLS, E.N_PER, -1).mean(1)
    ms = np.stack([mc[FINE_TO_COARSE == k].mean(0) for k in range(20)])
    return mc.argmax(0), ms.argmax(0)


def nest_tables(src, tgt, sc_par, cl_ch, sc_ch):
    """Per source x target cell: pairs whose child's preferred class lies in the parent's preferred superclass
    ('nested'), its expectation if parent superclasses and child classes were paired at random within the cell, and
    pairs whose preferred superclasses (argmax of superclass means) agree."""
    nest = np.zeros((4, 4))
    nest_exp = np.zeros((4, 4))
    same = np.zeros((4, 4))
    co = FINE_TO_COARSE[cl_ch]
    for a in range(4):
        for b in range(4):
            m = (src == a) & (tgt == b)
            n = m.sum()
            if not n:
                continue
            nest[a, b] = (co[m] == sc_par[m]).sum()
            same[a, b] = (sc_ch[m] == sc_par[m]).sum()
            fa = np.bincount(sc_par[m], minlength=20)
            fb = np.bincount(co[m], minlength=20)
            nest_exp[a, b] = (fa * fb).sum() / n
    return nest, nest_exp, same


def tables(lk, exa, exb, rng, pa, pb):
    """Contingencies (4x4, rows = source label at a, cols = target label at b) and permutation nulls."""
    pred, typ, bp, al_a, al_b = lk["pred"], lk["typ"], lk["bp"], lk["alive_a"], lk["alive_b"]
    has = (pred >= 0) & al_b
    j = np.flatnonzero(has)
    src, tgt = exa[pred[j]] - 1, exb[j] - 1
    assert (src >= 0).all() and (tgt >= 0).all()
    out = {}
    out["nest"], out["nest_exp"], out["same_sc"] = nest_tables(src, tgt, pa[1][pred[j]], pb[0][j], pb[1][j])
    for nm, m in (("stable", typ[j] == 1), ("transformed", typ[j] == 2)):
        out[nm] = np.bincount(src[m] * 4 + tgt[m], minlength=16).reshape(4, 4)
    born_mask = al_b & ~has
    out["born"] = np.bincount(exb[born_mask] - 1, minlength=4)
    out["alive_b"] = np.bincount(exb[al_b] - 1, minlength=4)
    out["alive_a"] = np.bincount(exa[al_a] - 1, minlength=4)
    succ = np.zeros(len(exa), bool)
    succ[pred[j]] = True
    out["nosucc"] = np.bincount(exa[al_a & ~succ] - 1, minlength=4)
    # permutation: shuffle source labels among the matched predecessors
    P = np.zeros((N_PERM, 4, 4), np.int32)
    for k in range(N_PERM):
        s = src[rng.permutation(len(src))]
        P[k] = np.bincount(s * 4 + tgt, minlength=16).reshape(4, 4)
    out["perm"] = P
    # best parent (many-to-one)
    jb = np.flatnonzero(bp >= 0)
    par = bp[jb]
    out["bp"] = np.bincount((exa[par] - 1) * 4 + (exb[jb] - 1), minlength=16).reshape(4, 4)
    out["bp_nest"], out["bp_nest_exp"], _ = nest_tables(exa[par] - 1, exb[jb] - 1, pa[1][par], pb[0][jb], pb[1][jb])
    out["bp_none"] = np.bincount(exb[al_b & (bp < 0)] - 1, minlength=4)
    upar, inv = np.unique(par, return_inverse=True)
    lab_u = exa[upar] - 1
    tb = exb[jb] - 1
    Pb = np.zeros((N_PERM, 4, 4), np.int32)
    for k in range(N_PERM):
        s = lab_u[rng.permutation(len(lab_u))][inv]
        Pb[k] = np.bincount(s * 4 + tb, minlength=16).reshape(4, 4)
    out["bp_perm"] = Pb
    # splits: parents at a that are best parent of >= 2 children at b, by parent label, with >= 1 cs child
    cnt = np.bincount(inv, minlength=len(upar))
    cs_child = np.bincount(inv, weights=(tb == 1), minlength=len(upar))
    out["multi_parent"] = np.bincount(lab_u[cnt >= 2], minlength=4)
    out["multi_parent_cs"] = np.bincount(lab_u[(cnt >= 2) & (cs_child >= 1)], minlength=4)
    out["n_parent"] = np.bincount(lab_u, minlength=4)
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
    PF = {k: prefs(h) for k, h in H.items()}
    links = {}
    for a, b in zip(labs[:-1], labs[1:]):
        for L in E.LAYERS:
            md = match(H[(a, L)], H[(b, L)])
            links[("t", a, b, L)] = link(md, md["correlation_matrix"], H[(a, L)].shape[1], H[(b, L)].shape[1])
    for lab in labs:
        for x, y in (("layer4", "avgpool"), ("avgpool", "layer4")):
            md = match(H[(lab, x)], H[(lab, y)])
            links[("s", lab, x, y)] = link(md, md["correlation_matrix"], H[(lab, x)].shape[1], H[(lab, y)].shape[1])
    rng = np.random.RandomState(2000 + ln["seed"])
    units, arrs = [], {}
    for tk in TKS:
        R = {(lab, L): role_arrays(z, nl, lab, L, tk) for lab in labs for L in E.LAYERS}
        for kk, lk in links.items():
            if kk[0] == "t":
                _, a, b, L = kk
                A, B = R[(a, L)], R[(b, L)]
                pa, pb = PF[(a, L)], PF[(b, L)]
                meta = {"kind": "time", "layer": L, "epoch_a": ep[a], "epoch_b": ep[b], "direction": ""}
            else:
                _, lab, x, y = kk
                A, B = R[(lab, x)], R[(lab, y)]
                pa, pb = PF[(lab, x)], PF[(lab, y)]
                meta = {"kind": "samedata", "layer": f"{x}->{y}", "epoch_a": ep[lab], "epoch_b": ep[lab],
                        "direction": f"{x}->{y}"}
            T = tables(lk, A["ex"], B["ex"], rng, pa, pb)
            n_tgcs = int((B["tg"] & B["cs"]).sum())
            uid = len(units)
            units.append({"uid": uid, "key": key, "seed": ln["seed"], "tk": tk, **meta, "n_cs_overlap_tg_b": n_tgcs,
                          "n_cs_incl_b": int(B["cs"].sum())})
            for nm, v in T.items():
                arrs[f"{uid}|{nm}"] = v
    CD.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(units).to_csv(CD / f"{key}_units.csv", index=False)
    np.savez_compressed(CD / f"{key}_tables.npz", **arrs)
    return f"{key}: {len(units)} units ({time.time() - t0:.0f}s)"


# analysis
def load_all():
    U, A = [], {}
    for f in sorted(CD.glob("*_units.csv")):
        u = pd.read_csv(f)
        z = np.load(f.with_name(f.name.replace("_units.csv", "_tables.npz")))
        key = u.key.iloc[0]
        for uid in u.uid:
            A[(key, uid)] = {k.split("|", 1)[1]: z[k] for k in z.files if k.split("|", 1)[0] == str(uid)}
        U.append(u)
    U = pd.concat(U, ignore_index=True)
    U["phase"] = [phase_of(a, b) if k == "time" else ("ep0" if a == 0 else "ep5-30")
                  for k, a, b in zip(U.kind, U.epoch_a, U.epoch_b)]
    return U, A


def summarize(rows, A, prefix=""):
    """Aggregate a set of units (count-pooled) into one row per (source, target) pair of interest."""
    tab = sum(A[r]["stable"] + A[r]["transformed"] for r in rows)
    st = sum(A[r]["stable"] for r in rows)
    tr = sum(A[r]["transformed"] for r in rows)
    P = sum(A[r]["perm"].astype(np.int64) for r in rows)
    bp = sum(A[r]["bp"] for r in rows)
    Pb = sum(A[r]["bp_perm"].astype(np.int64) for r in rows)
    born = sum(A[r]["born"] for r in rows)
    alive_b = sum(A[r]["alive_b"] for r in rows)
    nos = sum(A[r]["nosucc"] for r in rows)
    alive_a = sum(A[r]["alive_a"] for r in rows)
    nest = sum(A[r]["nest"] for r in rows)
    nest_exp = sum(A[r]["nest_exp"] for r in rows)
    same_sc = sum(A[r]["same_sc"] for r in rows)
    bpn = sum(A[r]["bp_nest"] for r in rows)
    bpne = sum(A[r]["bp_nest_exp"] for r in rows)
    out = {}
    n = tab.sum()
    nb = bp.sum()
    for s, t in PAIRS:
        i, j = CATS.index(s), CATS.index(t)
        n_st = tab[i, j]
        col, row = tab[:, j].sum(), tab[i, :].sum()
        bshare = n_st / col if col else np.nan          # backward: share of t-targets whose predecessor is s
        bchance = row / n if n else np.nan              # share of s among matched predecessors
        fshare = n_st / row if row else np.nan          # forward: share of s-sources whose successor is t
        fchance = col / n if n else np.nan              # share of t among matched successors
        pv = P[:, i, j]
        enr = bshare / bchance if bchance else np.nan
        bpcol, bprow = bp[:, j].sum(), bp[i, :].sum()
        bp_obs = bp[i, j] / bpcol if bpcol else np.nan
        bp_ch = bprow / nb if nb else np.nan
        pvb = Pb[:, i, j]
        p_ = f"{s}->{t}|"
        out.update({
            p_ + "n_pairs": int(n_st), p_ + "n_target_with_pred": int(col), p_ + "n_source_with_succ": int(row),
            p_ + "n_matched": int(n), p_ + "obs_back": bshare, p_ + "chance_back": bchance,
            p_ + "obs_fwd": fshare, p_ + "chance_fwd": fchance, p_ + "enrichment": enr,
            p_ + "perm_mean": pv.mean(), p_ + "ratio_obs_perm": n_st / pv.mean() if pv.mean() else np.nan,
            p_ + "p_ge": (1 + (pv >= n_st).sum()) / (1 + len(pv)), p_ + "p_le": (1 + (pv <= n_st).sum()) / (1 + len(pv)),
            p_ + "n_stable": int(st[i, j]), p_ + "n_transformed": int(tr[i, j]),
            p_ + "fwd_alive_source": int(alive_a[i]), p_ + "fwd_share_of_alive_source": n_st / alive_a[i] if alive_a[i] else np.nan,
            p_ + "fwd_nosucc_share": nos[i] / alive_a[i] if alive_a[i] else np.nan,
            p_ + "bp_obs": bp_obs, p_ + "bp_chance": bp_ch, p_ + "bp_enrichment": bp_obs / bp_ch if bp_ch else np.nan,
            p_ + "bp_ratio_obs_perm": bp[i, j] / pvb.mean() if pvb.mean() else np.nan,
            p_ + "bp_p_ge": (1 + (pvb >= bp[i, j]).sum()) / (1 + len(pvb)),
            p_ + "nested_share": nest[i, j] / n_st if n_st else np.nan,
            p_ + "nested_share_exp": nest_exp[i, j] / n_st if n_st else np.nan,
            p_ + "same_sc_share": same_sc[i, j] / n_st if n_st else np.nan,
            p_ + "bp_nested_share": bpn[i, j] / bp[i, j] if bp[i, j] else np.nan,
            p_ + "bp_nested_share_exp": bpne[i, j] / bp[i, j] if bp[i, j] else np.nan,
        })
    # conversion: cs targets whose predecessor was not class-selective; which non-cs category supplied them?
    # stratified expectation per unit = (# non-cs -> cs) x share of s among non-cs matched predecessors.
    ci = CATS.index("cs")
    for s in ("tg", "ss", "other"):
        i = CATS.index(s)
        o_, e_, sims = 0, 0.0, np.zeros(2000)
        for r in rows:
            t = A[r]["stable"] + A[r]["transformed"]
            nonc = [k for k in range(4) if k != ci]
            draws = int(t[nonc, ci].sum())
            pool = int(t[nonc, :].sum())
            good = int(t[i, :].sum())
            o_ += int(t[i, ci])
            if draws and pool:
                e_ += draws * good / pool
                sims += RNG_CONV.hypergeometric(good, pool - good, draws, size=len(sims))
        out[f"conv|{s}|obs"] = o_
        out[f"conv|{s}|exp"] = e_
        out[f"conv|{s}|ratio"] = o_ / e_ if e_ else np.nan
        out[f"conv|{s}|p_ge"] = (1 + (sims >= o_).sum()) / (1 + len(sims))
        out[f"conv|{s}|p_le"] = (1 + (sims <= o_).sum()) / (1 + len(sims))
        # forward, conditional on the source changing category
        row, diag = tab[i, :].sum(), tab[i, i]
        out[f"conv|{s}|fwd_cs_given_change"] = tab[i, ci] / (row - diag) if row - diag else np.nan
        out[f"conv|{s}|keep_share"] = diag / row if row else np.nan
    # origins of every target category (Hungarian) and birth shares
    for j, t in enumerate(CATS):
        nt = alive_b[j]
        out[f"origin|{t}|n_alive"] = int(nt)
        out[f"origin|{t}|born"] = born[j] / nt if nt else np.nan
        for i, s in enumerate(CATS):
            out[f"origin|{t}|from_{s}"] = tab[i, j] / nt if nt else np.nan
    mp = sum(A[r]["multi_parent"] for r in rows)
    mpc = sum(A[r]["multi_parent_cs"] for r in rows)
    npar = sum(A[r]["n_parent"] for r in rows)
    for i, s in enumerate(CATS):
        out[f"split|{s}|n_parent"] = int(npar[i])
        out[f"split|{s}|share_multi"] = mp[i] / npar[i] if npar[i] else np.nan
        out[f"split|{s}|share_multi_with_cs"] = mpc[i] / npar[i] if npar[i] else np.nan
    return out


def long(df, idcols):
    """wide 'pair|metric' -> long rows with columns pair + metrics."""
    rec = []
    for _, r in df.iterrows():
        for s, t in PAIRS:
            p_ = f"{s}->{t}|"
            rec.append({**{c: r[c] for c in idcols}, "pair": f"{s}->{t}",
                        **{k[len(p_):]: r[k] for k in df.columns if k.startswith(p_)}})
    return pd.DataFrame(rec)


def origins(df, idcols):
    rec = []
    for _, r in df.iterrows():
        for t in CATS:
            p_ = f"origin|{t}|"
            rec.append({**{c: r[c] for c in idcols}, "target": t,
                        **{k[len(p_):]: r[k] for k in df.columns if k.startswith(p_)}})
    return pd.DataFrame(rec)


def conv(df, idcols):
    rec = []
    for _, r in df.iterrows():
        for s in ("tg", "ss", "other"):
            p_ = f"conv|{s}|"
            rec.append({**{c: r[c] for c in idcols}, "source": s,
                        **{k[len(p_):]: r[k] for k in df.columns if k.startswith(p_)}})
    return pd.DataFrame(rec)


def group(U, A, by, mask):
    res = []
    for k, g in U[mask].groupby(by):
        k = k if isinstance(k, tuple) else (k,)
        res.append({**dict(zip(by, k)), "n_units": len(g),
                    **summarize([(kk, uid) for kk, uid in zip(g.key, g.uid)], A)})
    return pd.DataFrame(res)


def analyze():
    U, A = load_all()
    tm = U.kind == "time"
    U["layerset"] = np.where(U.layer == "avgpool", "avgpool", "l1-l4")
    PH = ["0->5", "5->15", "15->30"]
    outs = {}
    # per seed x phase x layer, + pooled over layers, + pooled over phases
    by_layer = group(U, A, ["tk", "seed", "layer", "phase"], tm)
    U2 = U.copy()
    U2["layer"], U2["phase_all"] = "all", "all"
    pooled_layers = group(U2, A, ["tk", "seed", "layer", "phase"], tm)
    U3 = U.copy()
    U3["phase"] = "all"
    all_phase_layer = group(U3, A, ["tk", "seed", "layer", "phase"], tm)
    U4 = U3.copy()
    U4["layer"] = "all"
    all_all = group(U4, A, ["tk", "seed", "layer", "phase"], tm)
    seedw = pd.concat([by_layer, pooled_layers, all_phase_layer, all_all], ignore_index=True)
    # seeds pooled
    U5 = U.copy()
    U5["seed"] = 0
    ps = [group(U5, A, ["tk", "seed", "layer", "phase"], tm)]
    U6 = U5.copy(); U6["layer"] = "all"
    ps.append(group(U6, A, ["tk", "seed", "layer", "phase"], tm))
    U7 = U5.copy(); U7["phase"] = "all"
    ps.append(group(U7, A, ["tk", "seed", "layer", "phase"], tm))
    U8 = U7.copy(); U8["layer"] = "all"
    ps.append(group(U8, A, ["tk", "seed", "layer", "phase"], tm))
    U9 = U5.copy(); U9["layer"] = "all_no_avgpool"
    ps.append(group(U9, A, ["tk", "seed", "layer", "phase"], tm & (U.layer != "avgpool")))
    U10 = U9.copy(); U10["phase"] = "all"
    ps.append(group(U10, A, ["tk", "seed", "layer", "phase"], tm & (U.layer != "avgpool")))
    poolw = pd.concat(ps, ignore_index=True)
    allw = pd.concat([seedw, poolw], ignore_index=True)
    allw["seed"] = allw.seed.replace({0: "pooled"})
    ids = ["tk", "seed", "layer", "phase", "n_units"]
    L = long(allw, ids)
    L.to_csv(OUT / "enrichment_pairs.csv", index=False)
    O = origins(allw, ids)
    O.to_csv(OUT / "origins.csv", index=False)
    S = allw[ids + [c for c in allw.columns if c.startswith("split|")]]
    S.to_csv(OUT / "split_multiparent.csv", index=False)
    CV = conv(allw, ids)
    CV.to_csv(OUT / "conversion_to_cs.csv", index=False)
    # per transition (seeds pooled, layers pooled)
    U11 = U5.copy(); U11["layer"] = "all"
    tr = group(U11, A, ["tk", "layer", "epoch_a", "epoch_b"], tm)
    long(tr, ["tk", "layer", "epoch_a", "epoch_b", "n_units"]).to_csv(OUT / "enrichment_by_transition.csv", index=False)
    tr2 = group(U5, A, ["tk", "layer", "epoch_a", "epoch_b"], tm)
    origins(pd.concat([tr2, tr], ignore_index=True), ["tk", "layer", "epoch_a", "epoch_b", "n_units"]).to_csv(
        OUT / "origins_by_transition.csv", index=False)
    # same-data control
    sd = U[~tm].copy()
    sd1 = group(sd.assign(seed=0), A, ["tk", "seed", "direction", "phase"], ~tm)
    sd2 = group(sd, A, ["tk", "seed", "direction", "phase"], ~tm)
    sd3 = group(sd.assign(seed=0, direction="both"), A, ["tk", "seed", "direction", "phase"], ~tm)
    sdw = pd.concat([sd2, sd1, sd3], ignore_index=True)
    sdw["seed"] = sdw.seed.replace({0: "pooled"})
    sids = ["tk", "seed", "direction", "phase", "n_units"]
    SDL = long(sdw, sids)
    SDL.to_csv(OUT / "samedata_pairs.csv", index=False)
    origins(sdw, sids).to_csv(OUT / "samedata_origins.csv", index=False)
    sdw[sids + [c for c in sdw.columns if c.startswith("split|")]].to_csv(OUT / "samedata_split_multiparent.csv", index=False)
    SDC = conv(sdw, sids)
    SDC.to_csv(OUT / "samedata_conversion_to_cs.csv", index=False)
    # comparable time-direction units for same-data comparison: layer4 & avgpool, epochs >= 5 transitions
    U12 = U5.copy(); U12["layer"] = "l4+avgpool"; U12["phase"] = "5->30"
    t45 = group(U12, A, ["tk", "seed", "layer", "phase"], tm & U.layer.isin(["layer4", "avgpool"]) & (U.epoch_a >= 5))
    T45 = long(t45, ids)
    T45.to_csv(OUT / "time_l4avgpool_5to30_pairs.csv", index=False)
    T45C = conv(t45, ids)
    T45C.to_csv(OUT / "time_l4avgpool_5to30_conversion.csv", index=False)
    # overlap check
    ov = U[tm].groupby("tk")[["n_cs_overlap_tg_b", "n_cs_incl_b"]].sum()
    # summary json (primary tk = layer)
    def pick(df, **kw):
        m = np.ones(len(df), bool)
        for k, v in kw.items():
            m &= (df[k].astype(str) == str(v)).values
        return df[m]
    summ = {"n_perm": N_PERM, "cs_tg_overlap_alive_b": ov.to_dict(), "primary_tk": "layer"}
    for tk in TKS:
        d = {}
        for pair in [f"{s}->{t}" for s, t in PAIRS]:
            q = pick(L, tk=tk, pair=pair)
            e = {}
            for ph in PH + ["all"]:
                ps_ = pick(q, seed="pooled", layer="all", phase=ph).iloc[0]
                per = pick(q, layer="all", phase=ph)
                per = per[per.seed.astype(str) != "pooled"]
                e[ph] = {"enrichment_strat": ps_.ratio_obs_perm, "enrichment_crude_pooled": ps_.enrichment, "obs_back": ps_.obs_back, "chance_back": ps_.chance_back,
                         "obs_fwd": ps_.obs_fwd, "chance_fwd": ps_.chance_fwd, "p_ge_pooled": ps_.p_ge,
                         "p_le_pooled": ps_.p_le, "bp_enrichment_pooled": ps_.bp_enrichment, "bp_p_ge": ps_.bp_p_ge,
                         "n_pairs": int(ps_.n_pairs), "n_stable": int(ps_.n_stable), "n_transformed": int(ps_.n_transformed),
                         "nested_share": ps_.nested_share, "nested_share_exp": ps_.nested_share_exp,
                         "enrichment_strat_per_seed": dict(zip(per.seed.astype(str), per.ratio_obs_perm.round(3))),
                         "n_seeds_gt1": int((per.ratio_obs_perm > 1).sum()),
                         "n_seeds_p05_ge": int((per.p_ge <= 0.05).sum()), "n_seeds_p05_le": int((per.p_le <= 0.05).sum())}
            d[pair] = e
        o = pick(O, tk=tk, seed="pooled", layer="all", phase="all")
        d["origins_all"] = {r.target: {k: r[k] for k in ["n_alive", "born", "from_tg", "from_cs", "from_ss", "from_other"]}
                            for _, r in o.iterrows()}
        sdq = pick(long(sdw, sids), tk=tk, seed="pooled", direction="both")
        d["samedata"] = {f"{r.pair}|{r.phase}": {"enrichment": r.enrichment, "obs_fwd": r.obs_fwd, "n_pairs": int(r.n_pairs)}
                         for _, r in sdq.iterrows()}
        so = pick(origins(sdw, sids), tk=tk, seed="pooled", direction="both", phase="ep5-30")
        d["samedata_origins_ep5_30"] = {r.target: {k: r[k] for k in ["n_alive", "born", "from_tg", "from_cs", "from_ss", "from_other"]}
                                        for _, r in so.iterrows()}
        d["samedata_strat"] = {r.pair: {"enrichment_strat": r.ratio_obs_perm, "obs_fwd": r.obs_fwd,
                                        "nested_share": r.nested_share, "nested_share_exp": r.nested_share_exp}
                               for _, r in sdq[sdq.phase == "ep5-30"].iterrows()}
        tq = pick(T45, tk=tk)
        d["time_l4avgpool_5to30"] = {r.pair: {"enrichment_strat": r.ratio_obs_perm, "obs_fwd": r.obs_fwd,
                                              "nested_share": r.nested_share, "nested_share_exp": r.nested_share_exp}
                                     for _, r in tq.iterrows()}
        cq = pick(CV, tk=tk, seed="pooled")
        d["conversion"] = {f"{r.source}|{r.layer}|{r.phase}": {"obs": int(r.obs), "exp": r.exp, "ratio": r.ratio,
                                                               "p_ge": r.p_ge, "p_le": r.p_le,
                                                               "fwd_cs_given_change": r.fwd_cs_given_change}
                           for _, r in cq.iterrows() if r.layer in ("all", "layer4", "avgpool", "all_no_avgpool")}
        d["conversion_samedata"] = {r.source: {"obs": int(r.obs), "exp": r.exp, "ratio": r.ratio,
                                               "fwd_cs_given_change": r.fwd_cs_given_change}
                                    for _, r in pick(SDC, tk=tk, seed="pooled", direction="both", phase="ep5-30").iterrows()}
        summ[tk] = d
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1, default=float)
    print(json.dumps(summ["layer"]["ss->cs"], indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "run":
        for ln in E.lanes():
            if ln["cond"] != "clean":
                continue
            if (CD / f"{ln['key']}_tables.npz").exists():
                print(ln["key"], "cached")
                continue
            print(lane_run(ln), flush=True)
    else:
        analyze()


if __name__ == "__main__":
    main()
