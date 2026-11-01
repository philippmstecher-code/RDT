"""Statistics of the Granger-timing figures (granger_panels.py, extended/ed2_panels.py): panel Granger tests with the reverse direction
and ΔR², the threshold sweep, the cumulative-dose incremental test (window 'all_post'), per-superclass slopes and
centre-of-mass timing. ``drop_terminal`` removes each lane's stored ``k_to_terminal`` transition (last checkpoint
vs itself).

Input: the per-superclass per-transition series rebuilt by superclass_series.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f as f_dist, spearmanr

from turnover_lib import STORE

CRITICAL_END = 2


def load_series(drop_terminal: bool) -> pd.DataFrame:
    c = pd.read_csv(STORE / "superclass_transition_series_cifar100.csv")
    c["dataset"] = "cifar100"
    t = pd.read_csv(STORE / "superclass_transition_series_tiny.csv")
    t["dataset"] = "tiny_imagenet"
    df = pd.concat([c, t], ignore_index=True)
    if drop_terminal:
        df = df[~df["is_terminal"]].reset_index(drop=True)
    return df


def build_panel(df: pd.DataFrame, max_t: int | None = None) -> pd.DataFrame:
    rows = []
    for (ds, lane, sc), g in df.groupby(["dataset", "lane", "superclass"]):
        g = g.sort_values("transition")
        if max_t is not None:
            g = g[g["transition"] <= max_t]
        t = g["transition"].values
        ab, di, tg = g["ab_h"].values, g["di_h"].values, g["tg_h"].values
        for i in range(len(t) - 1):
            rows.append(dict(dataset=ds, lane=lane, superclass=sc, t=t[i],
                             ab_h_t=ab[i], di_h_t=di[i], tg_h_t=tg[i],
                             ab_h_t1=ab[i + 1], di_h_t1=di[i + 1], tg_h_t1=tg[i + 1]))
    return pd.DataFrame(rows)


def fe_of(pw: pd.DataFrame, cols=("superclass", "lane", "dataset")) -> pd.DataFrame:
    parts = [pd.get_dummies(pw[c], prefix=c[:2], drop_first=True).astype(float).reset_index(drop=True)
             for c in cols]
    return pd.concat(parts, axis=1)


def granger(pw: pd.DataFrame, pred: str, out: str, own: str, fe: pd.DataFrame | None = None) -> dict:
    fe = fe_of(pw) if fe is None else fe
    y = pw[out].astype(float).reset_index(drop=True)
    Xr = sm.add_constant(pd.concat([pw[[own]].astype(float).reset_index(drop=True), fe], axis=1))
    Xf = sm.add_constant(pd.concat([pw[[own, pred]].astype(float).reset_index(drop=True), fe], axis=1))
    mr, mf = sm.OLS(y, Xr).fit(), sm.OLS(y, Xf).fit()
    F = (mr.ssr - mf.ssr) / (mf.ssr / mf.df_resid)
    return {"n_obs": len(pw), "F": F, "p": float(f_dist.sf(F, 1, mf.df_resid)), "beta": mf.params[pred],
            "delta_r2": mf.rsquared - mr.rsquared, "df_resid": mf.df_resid}


def panel_a(df: pd.DataFrame) -> pd.DataFrame:
    c = df[df["dataset"] == "cifar100"]
    out = []
    for proc, lab in [("tg_h", "Tg-E"), ("ab_h", "Ab-E"), ("di_h", "Di-E")]:
        agg = c.groupby("transition")[proc].mean()
        n = c.groupby("transition")["lane"].nunique()
        pct = agg / agg.sum() * 100
        for t, v in pct.items():
            out.append({"process": lab, "transition": int(t), "pct_of_total_events": v,
                        "mean_events_per_superclass": agg[t], "n_runs": int(n[t])})
    return pd.DataFrame(out)


def sweep(df: pd.DataFrame, max_ts) -> pd.DataFrame:
    rows = []
    for mt in max_ts:
        pw = build_panel(df, mt)
        if len(pw) < 50:
            continue
        fe = fe_of(pw)
        for lab, (pred, out, own) in {"Tg->Ab": ("tg_h_t", "ab_h_t1", "ab_h_t"),
                                      "Ab->Di": ("ab_h_t", "di_h_t1", "di_h_t"),
                                      "Tg->Di": ("tg_h_t", "di_h_t1", "di_h_t")}.items():
            g = granger(pw, pred, out, own, fe)
            rows.append({"max_transition_included": mt, "test": lab, "F": g["F"], "p": g["p"],
                         "beta": g["beta"], "n_obs": g["n_obs"]})
    return pd.DataFrame(rows)


def dose(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    for (ds, lane, sc), g in df.groupby(["dataset", "lane", "superclass"]):
        g = g.sort_values("transition")
        crit, post = g[g["transition"] <= CRITICAL_END], g[g["transition"] > CRITICAL_END]
        if len(crit) == 0 or len(post) == 0:
            continue
        rows.append(dict(dataset=ds, lane=lane, superclass=sc, tg_dose=crit["tg_h"].sum(),
                         ab_dose=crit["ab_h"].sum(), post_ab=post["ab_h"].sum(), post_di=post["di_h"].sum()))
    d = pd.DataFrame(rows)
    rho, p = spearmanr(d["tg_dose"], d["post_di"])
    # incremental: Tg dose beyond Ab dose -> later Di-E (dataset + lane FE)
    fe = fe_of(d, ("dataset", "lane"))
    y = d["post_di"].astype(float).reset_index(drop=True)
    Xr = sm.add_constant(pd.concat([d[["ab_dose"]].astype(float).reset_index(drop=True), fe], axis=1))
    Xf = sm.add_constant(pd.concat([d[["ab_dose", "tg_dose"]].astype(float).reset_index(drop=True), fe], axis=1))
    mr, mf = sm.OLS(y, Xr).fit(), sm.OLS(y, Xf).fit()
    Finc = (mr.ssr - mf.ssr) / (mf.ssr / mf.df_resid)
    return d, {"rho": rho, "p": p, "N": len(d), "F_inc": Finc, "p_inc": float(f_dist.sf(Finc, 1, mf.df_resid)),
               "delta_r2_inc": mf.rsquared - mr.rsquared}


def per_superclass(panel: pd.DataFrame) -> pd.DataFrame:
    cp = panel[panel["dataset"] == "cifar100"]
    rows = []
    for sc in sorted(cp["superclass"].unique()):
        sp = cp[cp["superclass"] == sc].reset_index(drop=True)
        if len(sp) < 10:
            continue
        ln = pd.get_dummies(sp["lane"], prefix="ln", drop_first=True).astype(float)
        y = sp["di_h_t1"].astype(float)
        X = sm.add_constant(pd.concat([sp[["di_h_t", "ab_h_t"]].astype(float), ln], axis=1))
        m = sm.OLS(y, X).fit()
        b, se, p = m.params["ab_h_t"], m.bse["ab_h_t"], float(m.pvalues["ab_h_t"])
        rows.append({"superclass": sc, "n_obs": len(sp), "beta": b, "ci_low": b - 1.96 * se,
                     "ci_high": b + 1.96 * se, "p": p, "p_bonferroni": min(p * 20, 1.0),
                     "significant_bonferroni": min(p * 20, 1.0) < 0.05})
    return pd.DataFrame(rows)


def centre_of_mass(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    for (ds, lane, sc), g in df.groupby(["dataset", "lane", "superclass"]):
        g = g.sort_values("transition")
        t = g["transition"].values.astype(float)
        a, d = g["ab_h"].values.astype(float), g["di_h"].values.astype(float)
        if a.sum() > 0 and d.sum() > 0:
            rows.append(dict(dataset=ds, lane=lane, superclass=sc, ab_com=np.average(t, weights=a),
                             di_com=np.average(t, weights=d)))
    c = pd.DataFrame(rows)
    rho, p = spearmanr(c["ab_com"], c["di_com"])
    return c, {"rho": rho, "p": p, "N": len(c), "n_above_identity": int((c["di_com"] > c["ab_com"]).sum())}


def all_stats(drop_terminal: bool) -> dict:
    df = load_series(drop_terminal)
    pw = build_panel(df)
    fwd = granger(pw, "ab_h_t", "di_h_t1", "di_h_t")
    rev = granger(pw, "di_h_t", "ab_h_t1", "ab_h_t")
    tmax = int(df["transition"].max())
    sw = sweep(df, range(1, tmax + 1))
    dd, dstat = dose(df)
    sc = per_superclass(pw)
    com, cstat = centre_of_mass(df)
    return {"df": df, "panel": pw, "a": panel_a(df), "sweep": sw, "fwd": fwd, "rev": rev,
            "dose_df": dd, "dose": dstat, "per_sc": sc, "com_df": com, "com": cstat}
