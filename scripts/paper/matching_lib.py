"""Matching diagnostics library.

Phase 5 (activation correlation + Hungarian matching) is called directly from ``packages/ml/src/saeanalysis.py``,
so the baseline is the pipeline's own code path. Phase 6 (process labels) is re-implemented in vectorised form at
the feature level and sample-weighted (which reproduces ``process_intensity``). Alternative matching rules
(mutual best, greedy, confirmed) reuse the pipeline's correlation matrix. Read-only on the data drive.
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

REPO = Path(__file__).resolve().parents[2]
ML_SRC = REPO / "packages" / "ml" / "src"
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))

import saeanalysis as sa  # noqa: E402

DATA = Path(RDT_DATA + "/core")
EXP = DATA / "experiments"
CORE_EXP = EXP / "abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb"
LESION_EXP = EXP / "scaffold_lesion"
OUT = REPO / "analysis_outputs" / "paper" / "MATCHING"

STABLE_R = 0.5
DEATH_R = 0.2
SPLIT_R = 0.35
EPS_ACT = 1e-6

CORE_LANES: List[Tuple[str, str, int]] = [
    ("7db28a9a-f436-4d8a-a6b4-37e69fbf54eb", "ResNet-18", 42),
    ("a0bcbcba-94a9-4009-bcb6-d725b47588ee", "ResNet-18", 137),
    ("935c13ae-a026-4eae-a2b2-0306879b2e8c", "ResNet-18", 256),
    ("10d31151-68a2-4491-836d-3ead8e00a7ad", "ViT-Small", 42),
    ("909d7f68-55e1-4cd9-8c7b-68d5793798d3", "ViT-Small", 137),
    ("abad8975-de55-4e7e-b95d-05067f0eed90", "ViT-Small", 256),
    ("4bebc329-c26a-4eb4-b18c-7586e513f49e", "CCT-7", 42),
    ("edf8275e-c99f-4436-a13d-4ed50f072a66", "CCT-7", 137),
    ("445dd4d4-770a-4f33-b82b-c0bc7baed9a7", "CCT-7", 256),
]


def arch_from_layers(layers: List[str]) -> str:
    if "avgpool" in layers and "layer1" in layers:
        return "ResNet-18"
    if any(l.startswith("encoder.layers") for l in layers):
        return "ViT-Small"
    if "norm" in layers:
        return "CCT-7"
    return "unknown"


def _as_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        return ast.literal_eval(x)
    return {}


class Lane:
    def __init__(self, lane_dir: Path):
        self.dir = Path(lane_dir)
        self.sae_dir = self.dir / "sae_analysis"
        with open(self.sae_dir / "sae_results.json") as f:
            self.results = json.load(f)
        meta = self.results["metadata"]
        self.layers: List[str] = meta["layers"]
        self.labels: List[str] = meta["checkpoint_labels"]
        self.ati = _as_dict(meta.get("adaptive_thresholds"))
        self.thr = _as_dict(meta.get("analysis_thresholds"))
        assert abs(self.thr.get("match_stable_thresh", 0.5) - STABLE_R) < 1e-12
        assert abs(self.thr.get("match_death_thresh", 0.2) - DEATH_R) < 1e-12
        self.arch = arch_from_layers(self.layers)

    def act_path(self, label: str, layer: str) -> Path:
        return self.sae_dir / "activations" / label / f"{sa._safe_layer_filename(layer)}.npy"

    def load(self, label: str, layer: str) -> np.ndarray:
        return np.load(self.act_path(label, layer))

    def sel(self, label: str, layer: str) -> Dict[str, np.ndarray]:
        e = self.results["selectivity"][label][layer]
        return {k: np.asarray(e[k], dtype=np.float64) for k in ("feature_ssi", "feature_csi", "feature_sai")}

    def stored_counts(self, pair: str, layer: str) -> Optional[Dict[str, int]]:
        return self.results.get("feature_matching", {}).get(pair, {}).get(layer)


# Pipeline matching (exact code path)

def pipeline_match(H_a: np.ndarray, H_b: np.ndarray) -> Dict[str, Any]:
    sa._apply_thresholds(None)  # 0.5 / 0.2 as in the stored runs (asserted per lane)
    with contextlib.redirect_stdout(io.StringIO()):  # silence _progress prints
        res = sa._phase5_feature_matching({"A": {"L": H_a}, "B": {"L": H_b}}, {}, {}, ["A", "B"], ["L"], None)
    return res["A->B"]["L"]


# Matching rules on a common representation
#   A matching outcome = dict(stable={i:(j,r)}, transformed={i:(j,r)}, died=set(i),
#                             born=set(j), ambiguous_a=set(i), ambiguous_b=set(j))

def outcome_from_pipeline(md: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "stable": {int(i): (int(j), float(r)) for i, j, r in md["stable"]},
        "transformed": {int(i): (int(j), float(r)) for i, j, r in md["transformed"]},
        "died": set(int(i) for i in md["died"]),
        "born": set(int(j) for j in md["born"]),
        "ambiguous_a": set(), "ambiguous_b": set(),
    }


def _classify_pairs(pairs: List[Tuple[int, int, float]], ia: np.ndarray, ib: np.ndarray) -> Dict[str, Any]:
    """Pipeline thresholds applied to an arbitrary one-to-one pair list (indices are global)."""
    stable, transformed, died = {}, {}, set()
    claimed_a = set()
    for i, j, r in pairs:
        claimed_a.add(i)
        if r >= STABLE_R:
            stable[i] = (j, r)
        elif r >= DEATH_R:
            transformed[i] = (j, r)
        else:
            died.add(i)
    for i in ia:
        if int(i) not in claimed_a:
            died.add(int(i))
    claimed_b = {v[0] for v in stable.values()} | {v[0] for v in transformed.values()}
    born = set(int(j) for j in ib if int(j) not in claimed_b)
    return {"stable": stable, "transformed": transformed, "died": died, "born": born,
            "ambiguous_a": set(), "ambiguous_b": set()}


def mutual_best_outcome(Cs: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> Dict[str, Any]:
    if Cs.size == 0:
        return _classify_pairs([], ia, ib)
    rbest = Cs.argmax(axis=1)
    cbest = Cs.argmax(axis=0)
    pairs = [(int(ia[r]), int(ib[c]), float(Cs[r, c])) for r, c in enumerate(rbest) if cbest[c] == r]
    return _classify_pairs(pairs, ia, ib)


def greedy_outcome(Cs: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> Dict[str, Any]:
    pairs = []
    if Cs.size:
        rr, cc = np.where(Cs >= DEATH_R)  # pairs below 0.2 are equivalent to no assignment
        order = np.argsort(-Cs[rr, cc], kind="stable")
        used_a, used_b = set(), set()
        for k in order:
            r, c = int(rr[k]), int(cc[k])
            if r in used_a or c in used_b:
                continue
            used_a.add(r); used_b.add(c)
            pairs.append((int(ia[r]), int(ib[c]), float(Cs[r, c])))
    return _classify_pairs(pairs, ia, ib)


def confirmed_outcome(hung: Dict[str, Any], Cs: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> Dict[str, Any]:
    """Confirmed-death semantics: stable = Hungarian r>=0.5;
    death/birth only if no candidate partner reaches r>=0.2; everything else ambiguous."""
    stable = dict(hung["stable"])
    fwd = Cs.max(axis=1) if Cs.size else np.full(len(ia), -1.0)
    bwd = Cs.max(axis=0) if Cs.size else np.full(len(ib), -1.0)
    sb = {v[0] for v in stable.values()}
    died, amb_a, born, amb_b = set(), set(), set(), set()
    for r, i in enumerate(ia):
        i = int(i)
        if i in stable:
            continue
        (died if fwd[r] < DEATH_R else amb_a).add(i)
    for c, j in enumerate(ib):
        j = int(j)
        if j in sb:
            continue
        (born if bwd[c] < DEATH_R else amb_b).add(j)
    return {"stable": stable, "transformed": {}, "died": died, "born": born,
            "ambiguous_a": amb_a, "ambiguous_b": amb_b}


def counts(o: Dict[str, Any]) -> Dict[str, int]:
    return {"n_stable": len(o["stable"]), "n_born": len(o["born"]), "n_died": len(o["died"]),
            "n_transformed": len(o["transformed"]) + len(o["ambiguous_a"]),
            "n_ambiguous_b": len(o["ambiguous_b"])}


# Diagnostics of the Hungarian outcome

def _top2(M: np.ndarray, axis: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """best value, argbest, second-best value along axis."""
    if M.shape[axis] == 0:
        n = M.shape[1 - axis]
        return np.full(n, np.nan), np.zeros(n, int), np.full(n, np.nan)
    arg = M.argmax(axis=axis)
    best = M.max(axis=axis)
    if M.shape[axis] < 2:
        return best, arg, np.full(best.shape, np.nan)
    part = -np.partition(-M, 1, axis=axis)
    second = part.take(1, axis=axis)
    return best, arg, second


def diagnostics(md: Dict[str, Any], hung: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    C = md["correlation_matrix"]
    ia = np.where(md["alive_a"])[0]
    ib = np.where(md["alive_b"])[0]
    Cs = C[np.ix_(ia, ib)].astype(np.float64)
    pos_a = {int(i): r for r, i in enumerate(ia)}
    pos_b = {int(j): c for c, j in enumerate(ib)}
    out: Dict[str, float] = {"n_alive_a": len(ia), "n_alive_b": len(ib)}
    arrs: Dict[str, np.ndarray] = {}
    if Cs.size == 0:
        return out, arrs
    rbest, rarg, rsec = _top2(Cs, 1)       # per A row
    cbest, carg, csec = _top2(Cs, 0)       # per B column
    cabs = np.abs(Cs).max(axis=0)
    rabs = np.abs(Cs).max(axis=1)

    # partner of each A under Hungarian (stable or transformed), r of that partner
    partner = {}
    for i, (j, r) in list(hung["stable"].items()) + list(hung["transformed"].items()):
        partner[i] = (j, r)
    # Hungarian also assigns pairs with r<0.2 (counted as died); recover them from C for completeness
    # (not needed for the scenario test, which concerns A features that have a successor).

    # (b) contested births
    born = sorted(hung["born"])
    bc = np.array([pos_b[j] for j in born], dtype=int)
    bb = cbest[bc] if len(bc) else np.array([])
    out["n_born"] = len(born)
    out["born_bwd_ge05"] = int((bb >= STABLE_R).sum())
    out["born_bwd_ge02"] = int((bb >= DEATH_R).sum())
    out["born_bwdabs_ge05"] = int((cabs[bc] >= STABLE_R).sum()) if len(bc) else 0
    out["born_bwdabs_ge02"] = int((cabs[bc] >= DEATH_R).sum()) if len(bc) else 0
    # scenario: newborn j is a better match (r>=0.5) for some A feature i than i's own successor
    scen = 0; scen_partner_status = {"stable": 0, "transformed": 0, "died": 0}
    scen_gap = []
    for c in bc:
        # all A features that correlate >=0.5 with this newborn
        rows = np.where(Cs[:, c] >= STABLE_R)[0]
        hit = False
        for r in rows:
            i = int(ia[r])
            rij = Cs[r, c]
            if i in partner:
                if rij > partner[i][1]:
                    hit = True
                    scen_gap.append(rij - partner[i][1])
                    scen_partner_status["stable" if i in hung["stable"] else "transformed"] += 1
                    break
            else:
                hit = True
                scen_gap.append(rij)
                scen_partner_status["died"] += 1
                break
        scen += int(hit)
    out["born_scenario"] = scen
    for k, v in scen_partner_status.items():
        out[f"born_scenario_partner_{k}"] = v
    # (b) contested deaths
    died = sorted(hung["died"])
    dr = np.array([pos_a[i] for i in died], dtype=int)
    dfw = rbest[dr] if len(dr) else np.array([])
    out["n_died"] = len(died)
    out["died_fwd_ge05"] = int((dfw >= STABLE_R).sum())
    out["died_fwd_ge02"] = int((dfw >= DEATH_R).sum())
    out["died_fwdabs_ge05"] = int((rabs[dr] >= STABLE_R).sum()) if len(dr) else 0
    # transformed: best partner >= 0.5 elsewhere?
    tr = sorted(hung["transformed"])
    trr = np.array([pos_a[i] for i in tr], dtype=int)
    out["n_transformed"] = len(tr)
    out["transformed_fwd_ge05"] = int((rbest[trr] >= STABLE_R).sum()) if len(trr) else 0

    # (c) stable pairs: mutual best and margins
    st = sorted(hung["stable"].items())
    out["n_stable"] = len(st)
    if st:
        sr = np.array([pos_a[i] for i, _ in st], dtype=int)
        sc = np.array([pos_b[j] for _, (j, _) in st], dtype=int)
        rv = Cs[sr, sc]
        row_is_best = rarg[sr] == sc
        col_is_best = carg[sc] == sr
        mutual = row_is_best & col_is_best
        out["stable_mutual_best"] = int(mutual.sum())
        out["stable_row_best"] = int(row_is_best.sum())
        out["stable_col_best"] = int(col_is_best.sum())
        # margin of the assigned pair against the best alternative partner (each direction)
        alt_row = np.empty(len(sr)); alt_col = np.empty(len(sr))
        for k in range(len(sr)):
            row = Cs[sr[k]].copy(); row[sc[k]] = -np.inf
            col = Cs[:, sc[k]].copy(); col[sr[k]] = -np.inf
            alt_row[k] = row.max() if row.size > 1 else np.nan
            alt_col[k] = col.max() if col.size > 1 else np.nan
        m_row = rv - alt_row
        m_col = rv - alt_col
        m_min = np.fmin(m_row, m_col)
        arrs["stable_r"] = rv
        arrs["stable_margin_row"] = m_row
        arrs["stable_margin_col"] = m_col
        arrs["stable_margin_min"] = m_min
        out["stable_margin_min_median"] = float(np.nanmedian(m_min))
        out["stable_margin_min_lt0"] = int((m_min < 0).sum())
        out["stable_margin_min_lt005"] = int((m_min < 0.05).sum())
        out["stable_margin_min_lt01"] = int((m_min < 0.1).sum())
        out["stable_alt_ge05"] = int((np.fmax(alt_row, alt_col) >= STABLE_R).sum())
    # (c') top-1 minus top-2 for every alive feature (both directions)
    arrs["rowmargin_all"] = rbest - rsec
    arrs["colmargin_all"] = cbest - csec

    # (d) splits / merges
    n_b_partners = (Cs >= SPLIT_R).sum(axis=1)
    n_a_partners = (Cs >= SPLIT_R).sum(axis=0)
    out["splits_a"] = int((n_b_partners >= 2).sum())
    out["merges_b"] = int((n_a_partners >= 2).sum())
    # splits/merges that touch an event (born B or died A)
    born_c = np.zeros(len(ib), bool); born_c[bc] = True
    died_r = np.zeros(len(ia), bool); died_r[dr] = True
    # split A whose >=0.35 partners include a born B
    if len(bc):
        hit = (Cs[:, born_c] >= SPLIT_R).any(axis=1)
        out["splits_a_with_born_partner"] = int(((n_b_partners >= 2) & hit).sum())
    else:
        out["splits_a_with_born_partner"] = 0
    out["merges_b_born"] = int(((n_a_partners >= 2) & born_c).sum())
    out["splits_a_died"] = int(((n_b_partners >= 2) & died_r).sum())
    return out, arrs


# Phase-6 labels (feature level + sample weighted), vectorised

class Labeler:
    def __init__(self, lane: Lane):
        t = lane.thr
        self.ssi_t = t["ssi_abstraction_thresh"]
        self.sai_t = t["sai_task_general_thresh"]
        self.csi_t = t["csi_differentiation_thresh"]
        a = lane.ati
        eps = 1e-10
        if a:
            self.ssi_m, self.ssi_s = a.get("ssi_null_mean", 0.0), max(a.get("ssi_null_std", 1.0), eps)
            self.csi_m, self.csi_s = a.get("csi_null_mean", 0.0), max(a.get("csi_null_std", 1.0), eps)
            self.sai_m, self.sai_s = a.get("sai_null_mean", 0.0), max(a.get("sai_null_std", 1.0), eps)
        else:
            self.ssi_m = self.csi_m = self.sai_m = 0.0
            self.ssi_s = self.csi_s = self.sai_s = 1.0
        self.zmin = 0.5

    def labels(self, o: Dict[str, Any], sel_a, sel_b) -> Dict[int, Tuple[str, bool, bool, bool]]:
        """B-feature -> (status, tg, ab, di); tg/ab exclusive as in the pipeline's Phase 6."""
        lab: Dict[int, Tuple[str, bool, bool, bool]] = {}
        for i, (j, _) in o["stable"].items():
            ssi_a, ssi_b = sel_a["feature_ssi"][i], sel_b["feature_ssi"][j]
            csi_a, csi_b = sel_a["feature_csi"][i], sel_b["feature_csi"][j]
            sai_a, sai_b = sel_a["feature_sai"][i], sel_b["feature_sai"][j]
            # same floating-point order as the pipeline: z_b - z_a with z = (v - m) / s
            dz_sai = (sai_b - self.sai_m) / self.sai_s - (sai_a - self.sai_m) / self.sai_s
            dz_ssi = (ssi_b - self.ssi_m) / self.ssi_s - (ssi_a - self.ssi_m) / self.ssi_s
            dz_csi = (csi_b - self.csi_m) / self.csi_s - (csi_a - self.csi_m) / self.csi_s
            tg = sai_b > self.sai_t and dz_sai > self.zmin
            ab = (not tg) and ssi_b > self.ssi_t and dz_ssi > self.zmin
            di = csi_b > self.csi_t and dz_csi > self.zmin
            lab[j] = ("stable", bool(tg), bool(ab), bool(di))
        for j in o["born"]:
            tg = sel_b["feature_sai"][j] > self.sai_t
            ab = (not tg) and sel_b["feature_ssi"][j] > self.ssi_t
            di = sel_b["feature_csi"][j] > self.csi_t
            lab[j] = ("born", bool(tg), bool(ab), bool(di))
        return lab

    def weighted(self, o, lab, act_a: np.ndarray, act_b: np.ndarray) -> Dict[str, int]:
        """Sample-weighted process counts == Phase-6 per-sample totals."""
        out = {"tg_h": 0, "ab_h": 0, "di_h": 0, "as_h": 0, "de_h": 0, "stable": 0}
        if o["stable"]:
            ia = np.fromiter(o["stable"].keys(), int)
            ib = np.array([o["stable"][i][0] for i in ia], int)
            co = (act_a[:, ia] & act_b[:, ib]).sum(axis=0)
            L = np.array([lab[j][1:] for j in ib], bool)
            out["stable"] += int(co.sum())
            out["tg_h"] += int(co[L[:, 0]].sum()); out["ab_h"] += int(co[L[:, 1]].sum()); out["di_h"] += int(co[L[:, 2]].sum())
        if o["born"]:
            jb = np.array(sorted(o["born"]), int)
            cb = act_b[:, jb].sum(axis=0)
            L = np.array([lab[j][1:] for j in jb], bool)
            out["as_h"] += int(cb.sum())
            out["tg_h"] += int(cb[L[:, 0]].sum()); out["ab_h"] += int(cb[L[:, 1]].sum()); out["di_h"] += int(cb[L[:, 2]].sum())
        if o["died"]:
            di = np.array(sorted(o["died"]), int)
            out["de_h"] += int(act_a[:, di].sum())
        return out


def feature_counts(lab: Dict[int, Tuple[str, bool, bool, bool]]) -> Dict[str, int]:
    out = {"f_tg": 0, "f_ab": 0, "f_di": 0, "f_tg_born": 0, "f_ab_born": 0, "f_di_born": 0}
    for st, tg, ab, di in lab.values():
        out["f_tg"] += tg; out["f_ab"] += ab; out["f_di"] += di
        if st == "born":
            out["f_tg_born"] += tg; out["f_ab_born"] += ab; out["f_di_born"] += di
    return out


def label_change(lab_ref, lab_alt, alive_b_idx) -> Dict[str, int]:
    """Compare Tg/Ab/Di label sets per alive B feature between two rules."""
    n_any = 0; n_diff = 0; n_diff_proc = 0; n_alive = len(alive_b_idx)
    per = {"tg": 0, "ab": 0, "di": 0}
    for j in alive_b_idx:
        a = lab_ref.get(int(j)); b = lab_alt.get(int(j))
        fa = a[1:] if a else (False, False, False)
        fb = b[1:] if b else (False, False, False)
        if any(fa) or any(fb):
            n_any += 1
            if fa != fb:
                n_diff += 1
        for k, nm in enumerate(("tg", "ab", "di")):
            if fa[k] != fb[k]:
                per[nm] += 1
    return {"n_alive_b": n_alive, "n_labelled_either": n_any, "n_label_changed": n_diff,
            "chg_tg": per["tg"], "chg_ab": per["ab"], "chg_di": per["di"]}


# Chains (checkpoint density)

def chain_compare(steps: List[Dict[str, Any]], direct: Dict[str, Any], alive_a_idx: np.ndarray,
                  prev: Optional[Dict[str, Any]] = None, direct_C: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Compare a chain of Hungarian outcomes k->k+1->...->k+m with the direct k->k+m outcome.

    steps[s] are outcome dicts; alive_a_idx are alive features at k.
    prev: outcome k-1 -> k (for 'consensus-matched' definition)."""
    n = len(alive_a_idx)
    chain_st = {}
    first_break = {}   # i -> (step index, status at break)
    for i in alive_a_idx:
        i = int(i); cur = i; ok = True
        for s, o in enumerate(steps):
            if cur in o["stable"]:
                cur = o["stable"][cur][0]
            else:
                st = "transformed" if cur in o["transformed"] else ("died" if cur in o["died"] else "dead_or_absent")
                first_break[i] = (s, st)
                ok = False
                break
        if ok:
            chain_st[i] = cur
    dir_st = {i: v[0] for i, v in direct["stable"].items()}
    A = set(int(i) for i in alive_a_idx)
    cs = set(chain_st) & A; ds = set(dir_st) & A
    both = cs & ds
    same = sum(1 for i in both if chain_st[i] == dir_st[i])
    neither = len(A - cs - ds)
    agree = (len(both) + neither) / max(n, 1)
    # Cohen's kappa
    p1 = len(cs) / max(n, 1); p2 = len(ds) / max(n, 1)
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    kappa = (agree - pe) / (1 - pe) if (1 - pe) > 1e-12 else np.nan
    direct_only = ds - cs
    brk = {"died": 0, "transformed": 0, "dead_or_absent": 0}
    brk_died_step = []
    for i in direct_only:
        s, st = first_break[i]
        brk[st] += 1
    # flicker: died at the first step (k->k+1) yet directly stable k->k+m
    died1 = set(steps[0]["died"]) & A
    flick = died1 & ds
    # any intermediate death but direct stable (gap)
    gap = {i for i in direct_only if first_break[i][1] == "died"}
    res = {"n_alive": n, "n_chain_stable": len(cs), "n_direct_stable": len(ds), "n_both": len(both),
           "n_both_same_partner": same, "n_chain_only": len(cs - ds), "n_direct_only": len(direct_only),
           "n_neither": neither, "agreement": agree, "kappa": kappa,
           "direct_only_break_died": brk["died"], "direct_only_break_transformed": brk["transformed"],
           "n_died_step1": len(died1), "n_flicker_step1": len(flick), "n_gap_any_death": len(gap)}
    if prev is not None:
        consensus = set(v[0] for v in prev["stable"].values()) & A
        res["n_consensus"] = len(consensus)
        res["n_died_step1_consensus"] = len(died1 & consensus)
        res["n_flicker_step1_consensus"] = len(flick & consensus)
    return res
