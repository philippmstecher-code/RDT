#!/usr/bin/env python3
"""
Tg-E Critical Period Analysis: Does Tg-E During the Critical Period
Scaffold Subsequent Ab-E and Di-E?

Key insight: Tg-E is concentrated in the earliest transitions (critical period).
A full-window Granger test is dominated by late transitions where Tg ≈ 0.
Instead we test:

  1. Critical-period Granger (transitions 0–4 only): Tg(t) → Ab(t+1), Tg(t) → Di(t+1)
  2. Cumulative dose → post-critical outcome: total Tg in t=0–K predicts Ab/Di at t=K+1+
  3. Cross-sectional dose-response: superclasses with more Tg during critical period
     → earlier Ab/Di onset and higher Ab/Di magnitude

Uses CIFAR-100 (9 lanes × 20 SC) + Tiny ImageNet (1 lane × 28 SC).
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, f as f_dist, mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "figures")


def load_data():
    """Load CIFAR-100 + Tiny ImageNet panel data."""
    cifar = pd.read_json(os.path.join(OUTPUT_DIR, "superclass_transition_series.json"))
    cifar["dataset"] = "cifar100"

    tiny_path = os.path.join(OUTPUT_DIR, "superclass_transition_series_tiny_imagenet.json")
    if not os.path.exists(tiny_path):
        print("ERROR: Run granger_causality_onset.py first to extract Tiny ImageNet data.")
        sys.exit(1)
    tiny = pd.read_json(tiny_path)
    tiny["dataset"] = "tiny_imagenet"

    return pd.concat([cifar, tiny], ignore_index=True)


def identify_critical_period(df):
    """Identify the critical period per dataset as transitions where Tg-E > 10% of its max."""
    print("\n── Critical Period Identification ──")
    for ds in df["dataset"].unique():
        ds_df = df[df["dataset"] == ds]
        tg_by_t = ds_df.groupby("transition")["tg_h"].mean()
        tg_max = tg_by_t.max()
        threshold = 0.10 * tg_max
        active = tg_by_t[tg_by_t > threshold].index.tolist()
        print(f"  {ds}: Tg-E active transitions = {active} "
              f"(threshold = {threshold:.0f}, max = {tg_max:.0f})")
        # Cumulative share
        cumsum = tg_by_t.cumsum() / tg_by_t.sum() * 100
        for pct in [50, 75, 90]:
            t_at_pct = cumsum[cumsum >= pct].index[0]
            print(f"    {pct}% of Tg-E accumulated by transition {t_at_pct}")


# ══════════════════════════════════════════════════════════════════════════
# Analysis 1: Critical-Period Restricted Granger
# ══════════════════════════════════════════════════════════════════════════

def restricted_window_granger(df, max_transition):
    """Run Granger tests restricted to transitions 0–max_transition."""
    import statsmodels.api as sm

    print(f"\n{'=' * 60}")
    print(f"ANALYSIS 1: Critical-Period Granger (transitions 0–{max_transition})")
    print(f"{'=' * 60}")

    # Build panel restricted to early transitions
    rows = []
    for (ds, lane, sc), group in df.groupby(["dataset", "lane", "superclass"]):
        group = group.sort_values("transition")
        group = group[group["transition"] <= max_transition]
        transitions = group["transition"].values
        ab_h = group["ab_h"].values
        di_h = group["di_h"].values
        tg_h = group["tg_h"].values

        for i in range(len(transitions) - 1):
            rows.append({
                "dataset": ds, "lane": lane, "superclass": sc,
                "t": transitions[i],
                "ab_h_t": ab_h[i], "di_h_t": di_h[i], "tg_h_t": tg_h[i],
                "ab_h_t1": ab_h[i + 1], "di_h_t1": di_h[i + 1], "tg_h_t1": tg_h[i + 1],
            })

    panel = pd.DataFrame(rows)
    print(f"  Panel: {len(panel)} observations (transitions 0–{max_transition})")

    sc_dummies = pd.get_dummies(panel["superclass"], prefix="sc", drop_first=True).astype(float)
    lane_dummies = pd.get_dummies(panel["lane"], prefix="lane", drop_first=True).astype(float)
    ds_dummies = pd.get_dummies(panel["dataset"], prefix="ds", drop_first=True).astype(float)
    fe = pd.concat([sc_dummies.reset_index(drop=True),
                     lane_dummies.reset_index(drop=True),
                     ds_dummies.reset_index(drop=True)], axis=1)

    results = {}
    for pred, out_lag, own, label in [
        ("tg_h_t", "ab_h_t1", "ab_h_t", "Tg-E(t) → Ab-E(t+1)"),
        ("tg_h_t", "di_h_t1", "di_h_t", "Tg-E(t) → Di-E(t+1)"),
        ("ab_h_t", "di_h_t1", "di_h_t", "Ab-E(t) → Di-E(t+1) [reference]"),
        # Reverses
        ("ab_h_t", "tg_h_t1", "tg_h_t", "Reverse: Ab-E(t) → Tg-E(t+1)"),
        ("di_h_t", "tg_h_t1", "tg_h_t", "Reverse: Di-E(t) → Tg-E(t+1)"),
    ]:
        y = panel[out_lag].astype(float).reset_index(drop=True)

        X_r = pd.concat([panel[[own]].astype(float).reset_index(drop=True), fe], axis=1)
        X_r = sm.add_constant(X_r)
        X_f = pd.concat([panel[[own, pred]].astype(float).reset_index(drop=True), fe], axis=1)
        X_f = sm.add_constant(X_f)

        model_r = sm.OLS(y, X_r).fit()
        model_f = sm.OLS(y, X_f).fit()

        f_stat = ((model_r.ssr - model_f.ssr) / 1) / (model_f.ssr / model_f.df_resid)
        p_value = 1 - f_dist.cdf(f_stat, 1, model_f.df_resid)
        coef = model_f.params[pred]
        tstat = model_f.tvalues[pred]
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

        print(f"\n  {label}:")
        print(f"    R² restricted: {model_r.rsquared:.4f} → full: {model_f.rsquared:.4f} "
              f"(ΔR² = {model_f.rsquared - model_r.rsquared:.4f})")
        print(f"    β = {coef:.4f}, t = {tstat:.2f}, F = {f_stat:.2f}, p = {p_value:.2e} {sig}")

        results[label] = {
            "restricted_r2": round(model_r.rsquared, 4),
            "full_r2": round(model_f.rsquared, 4),
            "delta_r2": round(model_f.rsquared - model_r.rsquared, 4),
            "coef": round(coef, 4),
            "tstat": round(tstat, 4),
            "granger_f": round(f_stat, 4),
            "granger_p": float(f"{p_value:.2e}"),
            "sig": sig,
        }

    return results, panel


# ══════════════════════════════════════════════════════════════════════════
# Analysis 2: Cumulative Tg Dose → Post-Critical Outcome
# ══════════════════════════════════════════════════════════════════════════

def cumulative_dose_analysis(df, critical_end=2):
    """Test whether cumulative Tg-E during t=0..critical_end predicts
    Ab-E and Di-E magnitude in subsequent transitions."""
    import statsmodels.api as sm

    print(f"\n{'=' * 60}")
    print(f"ANALYSIS 2: Cumulative Tg Dose (t ≤ {critical_end}) → Post-Critical Outcome")
    print(f"{'=' * 60}")

    rows = []
    for (ds, lane, sc), group in df.groupby(["dataset", "lane", "superclass"]):
        group = group.sort_values("transition")

        # Cumulative Tg during critical period
        critical = group[group["transition"] <= critical_end]
        tg_dose = critical["tg_h"].sum()
        ab_dose = critical["ab_h"].sum()

        # Post-critical outcomes at several windows
        for window_start, window_end, window_name in [
            (critical_end + 1, critical_end + 2, "immediate"),
            (critical_end + 1, critical_end + 4, "medium"),
            (critical_end + 1, 999, "all_post"),
        ]:
            post = group[(group["transition"] >= window_start) &
                         (group["transition"] <= window_end)]
            if len(post) == 0:
                continue

            rows.append({
                "dataset": ds, "lane": lane, "superclass": sc,
                "tg_dose": tg_dose, "ab_dose": ab_dose,
                "window": window_name,
                "post_ab_h": post["ab_h"].sum(),
                "post_di_h": post["di_h"].sum(),
                "post_tg_h": post["tg_h"].sum(),
                "n_transitions_post": len(post),
            })

    dose_df = pd.DataFrame(rows)

    results = {}
    for window in ["immediate", "medium", "all_post"]:
        w_df = dose_df[dose_df["window"] == window].copy()
        if len(w_df) < 10:
            continue

        print(f"\n  Window: {window} (N = {len(w_df)})")

        for predictor, pred_label in [("tg_dose", "Tg dose"), ("ab_dose", "Ab dose (control)")]:
            for outcome, out_label in [("post_ab_h", "post Ab-E"), ("post_di_h", "post Di-E")]:
                valid = w_df[[predictor, outcome]].dropna()
                if len(valid) < 10:
                    continue
                rho, p = spearmanr(valid[predictor], valid[outcome])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"    {pred_label} → {out_label}: ρ = {rho:.3f}, p = {p:.4f} {sig}")
                results[f"{window}_{predictor}_{outcome}"] = {
                    "rho": round(rho, 4), "p": round(p, 6), "n": len(valid)}

        # Regression: does Tg dose predict post-critical Di-E BEYOND Ab dose?
        w_df_clean = w_df[["tg_dose", "ab_dose", "post_di_h", "dataset", "lane"]].dropna()
        if len(w_df_clean) > 20:
            ds_dummies = pd.get_dummies(w_df_clean["dataset"], prefix="ds", drop_first=True).astype(float)
            lane_dummies = pd.get_dummies(w_df_clean["lane"], prefix="lane", drop_first=True).astype(float)
            fe = pd.concat([ds_dummies.reset_index(drop=True),
                             lane_dummies.reset_index(drop=True)], axis=1)

            y = w_df_clean["post_di_h"].astype(float).reset_index(drop=True)

            # Restricted: Ab dose + FE
            X_r = pd.concat([w_df_clean[["ab_dose"]].astype(float).reset_index(drop=True), fe], axis=1)
            X_r = sm.add_constant(X_r)
            # Full: Ab dose + Tg dose + FE
            X_f = pd.concat([w_df_clean[["ab_dose", "tg_dose"]].astype(float).reset_index(drop=True), fe], axis=1)
            X_f = sm.add_constant(X_f)

            m_r = sm.OLS(y, X_r).fit()
            m_f = sm.OLS(y, X_f).fit()

            if m_f.df_resid > 0:
                f_inc = ((m_r.ssr - m_f.ssr) / 1) / (m_f.ssr / m_f.df_resid)
                p_inc = 1 - f_dist.cdf(f_inc, 1, m_f.df_resid)
                tg_coef = m_f.params.get("tg_dose", 0)
                print(f"    Incremental test: Tg dose beyond Ab dose → post Di-E:")
                print(f"      ΔR² = {m_f.rsquared - m_r.rsquared:.4f}, "
                      f"F = {f_inc:.2f}, p = {p_inc:.4f}, β(Tg) = {tg_coef:.4f}")
                results[f"{window}_incremental_tg_beyond_ab"] = {
                    "delta_r2": round(m_f.rsquared - m_r.rsquared, 4),
                    "f": round(f_inc, 4), "p": round(p_inc, 6),
                    "tg_coef": round(tg_coef, 4)}

    return results, dose_df


# ══════════════════════════════════════════════════════════════════════════
# Analysis 3: Cross-Sectional Dose-Response
# ══════════════════════════════════════════════════════════════════════════

def dose_response_onset(df, dose_df):
    """Test whether Tg dose during critical period predicts Ab/Di onset timing."""
    print(f"\n{'=' * 60}")
    print(f"ANALYSIS 3: Tg Dose → Onset Timing (Cross-Sectional)")
    print(f"{'=' * 60}")

    # Compute onset metrics per (dataset, lane, superclass)
    onset_rows = []
    for (ds, lane, sc), group in df.groupby(["dataset", "lane", "superclass"]):
        group = group.sort_values("transition")
        transitions = group["transition"].values.astype(float)

        for process in ["ab_h", "di_h", "tg_h"]:
            counts = group[process].values.astype(float)
            total = counts.sum()
            if total == 0:
                continue
            peak_t = transitions[np.argmax(counts)]
            com = np.average(transitions, weights=counts)
            # Onset: first transition where count > 10% of max
            threshold = 0.10 * counts.max()
            onset_idx = np.where(counts > threshold)[0]
            onset_t = transitions[onset_idx[0]] if len(onset_idx) > 0 else np.nan

            onset_rows.append({
                "dataset": ds, "lane": lane, "superclass": sc,
                "process": process,
                "peak_t": peak_t, "onset_t": onset_t, "center_of_mass": com,
            })

    onset_df = pd.DataFrame(onset_rows)

    # Merge Tg dose with Ab/Di onset
    tg_dose = dose_df[dose_df["window"] == "all_post"][["dataset", "lane", "superclass", "tg_dose"]]
    ab_onset = onset_df[onset_df["process"] == "ab_h"][["dataset", "lane", "superclass",
                                                          "peak_t", "onset_t", "center_of_mass"]]
    di_onset = onset_df[onset_df["process"] == "di_h"][["dataset", "lane", "superclass",
                                                          "peak_t", "onset_t", "center_of_mass"]]

    results = {}

    for onset_name, onset_data in [("Ab-E", ab_onset), ("Di-E", di_onset)]:
        merged = tg_dose.merge(onset_data, on=["dataset", "lane", "superclass"])
        if len(merged) < 10:
            continue

        print(f"\n  Tg dose → {onset_name} timing (N = {len(merged)}):")
        for metric in ["peak_t", "onset_t", "center_of_mass"]:
            valid = merged[["tg_dose", metric]].dropna()
            if len(valid) < 10:
                continue
            rho, p = spearmanr(valid["tg_dose"], valid[metric])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    Tg dose vs {metric}: ρ = {rho:.3f}, p = {p:.4f} {sig}")
            results[f"tg_dose_vs_{onset_name}_{metric}"] = {
                "rho": round(rho, 4), "p": round(p, 6), "n": len(valid)}

    # Also: median split — high vs low Tg dose superclasses
    all_post = dose_df[dose_df["window"] == "all_post"].copy()
    if len(all_post) > 20:
        median_tg = all_post["tg_dose"].median()
        all_post["tg_group"] = np.where(all_post["tg_dose"] > median_tg, "high_tg", "low_tg")

        print(f"\n  Median split (Tg dose median = {median_tg:.0f}):")
        for outcome in ["post_ab_h", "post_di_h"]:
            high = all_post[all_post["tg_group"] == "high_tg"][outcome]
            low = all_post[all_post["tg_group"] == "low_tg"][outcome]
            u_stat, p = mannwhitneyu(high, low, alternative="two-sided")
            print(f"    {outcome}: high_tg mean = {high.mean():.0f}, low_tg mean = {low.mean():.0f}, "
                  f"U = {u_stat:.0f}, p = {p:.4f}")
            results[f"median_split_{outcome}"] = {
                "high_mean": round(high.mean(), 2), "low_mean": round(low.mean(), 2),
                "U": round(u_stat, 2), "p": round(p, 6)}

    return results, onset_df


# ══════════════════════════════════════════════════════════════════════════
# Analysis 4: Sweep across critical period cutoffs
# ══════════════════════════════════════════════════════════════════════════

def sweep_critical_cutoff(df):
    """Sweep the critical period boundary and show how Granger F changes."""
    import statsmodels.api as sm

    print(f"\n{'=' * 60}")
    print(f"ANALYSIS 4: Critical Period Cutoff Sweep")
    print(f"{'=' * 60}")

    results = []
    for max_t in range(1, 9):
        # Build restricted panel
        rows = []
        for (ds, lane, sc), group in df.groupby(["dataset", "lane", "superclass"]):
            group = group.sort_values("transition")
            group = group[group["transition"] <= max_t]
            transitions = group["transition"].values
            ab_h, di_h, tg_h = group["ab_h"].values, group["di_h"].values, group["tg_h"].values

            for i in range(len(transitions) - 1):
                rows.append({
                    "dataset": ds, "lane": lane, "superclass": sc,
                    "t": transitions[i],
                    "ab_h_t": ab_h[i], "di_h_t": di_h[i], "tg_h_t": tg_h[i],
                    "ab_h_t1": ab_h[i + 1], "di_h_t1": di_h[i + 1],
                })

        panel = pd.DataFrame(rows)
        if len(panel) < 50:
            continue

        sc_dummies = pd.get_dummies(panel["superclass"], prefix="sc", drop_first=True).astype(float)
        lane_dummies = pd.get_dummies(panel["lane"], prefix="lane", drop_first=True).astype(float)
        ds_dummies = pd.get_dummies(panel["dataset"], prefix="ds", drop_first=True).astype(float)
        fe = pd.concat([sc_dummies.reset_index(drop=True),
                         lane_dummies.reset_index(drop=True),
                         ds_dummies.reset_index(drop=True)], axis=1)

        y = panel["ab_h_t1"].astype(float).reset_index(drop=True)
        X_r = pd.concat([panel[["ab_h_t"]].astype(float).reset_index(drop=True), fe], axis=1)
        X_r = sm.add_constant(X_r)
        X_f = pd.concat([panel[["ab_h_t", "tg_h_t"]].astype(float).reset_index(drop=True), fe], axis=1)
        X_f = sm.add_constant(X_f)

        m_r = sm.OLS(y, X_r).fit()
        m_f = sm.OLS(y, X_f).fit()
        f_tg_ab = ((m_r.ssr - m_f.ssr) / 1) / (m_f.ssr / m_f.df_resid)
        p_tg_ab = 1 - f_dist.cdf(f_tg_ab, 1, m_f.df_resid)
        coef_tg_ab = m_f.params.get("tg_h_t", 0)

        # Also Tg → Di
        y_di = panel["di_h_t1"].astype(float).reset_index(drop=True)
        X_r_di = pd.concat([panel[["di_h_t"]].astype(float).reset_index(drop=True), fe], axis=1)
        X_r_di = sm.add_constant(X_r_di)
        X_f_di = pd.concat([panel[["di_h_t", "tg_h_t"]].astype(float).reset_index(drop=True), fe], axis=1)
        X_f_di = sm.add_constant(X_f_di)

        m_r_di = sm.OLS(y_di, X_r_di).fit()
        m_f_di = sm.OLS(y_di, X_f_di).fit()
        f_tg_di = ((m_r_di.ssr - m_f_di.ssr) / 1) / (m_f_di.ssr / m_f_di.df_resid)
        p_tg_di = 1 - f_dist.cdf(f_tg_di, 1, m_f_di.df_resid)
        coef_tg_di = m_f_di.params.get("tg_h_t", 0)

        # Mean Tg-E in this window
        mean_tg = panel["tg_h_t"].mean()

        results.append({
            "max_transition": max_t,
            "n_obs": len(panel),
            "mean_tg": round(mean_tg, 1),
            "tg_ab_F": round(f_tg_ab, 2),
            "tg_ab_p": float(f"{p_tg_ab:.2e}"),
            "tg_ab_coef": round(coef_tg_ab, 4),
            "tg_di_F": round(f_tg_di, 2),
            "tg_di_p": float(f"{p_tg_di:.2e}"),
            "tg_di_coef": round(coef_tg_di, 4),
        })

        sig_ab = "***" if p_tg_ab < 0.001 else "**" if p_tg_ab < 0.01 else "*" if p_tg_ab < 0.05 else "ns"
        sig_di = "***" if p_tg_di < 0.001 else "**" if p_tg_di < 0.01 else "*" if p_tg_di < 0.05 else "ns"
        print(f"  t ≤ {max_t}: N={len(panel):>5}, mean_Tg={mean_tg:>7.1f} | "
              f"Tg→Ab: F={f_tg_ab:>7.2f} β={coef_tg_ab:>7.4f} {sig_ab:<3} | "
              f"Tg→Di: F={f_tg_di:>7.2f} β={coef_tg_di:>7.4f} {sig_di:<3}")

    return results


# ══════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════

def plot_results(df, granger_results, dose_results, dose_df, sweep_results):
    """Create comprehensive figure."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # ── Panel A: Tg-E temporal profile ────────────────────────────────────
    ax = axes[0, 0]
    for ds, color, ls in [("cifar100", "#3B82F6", "-"), ("tiny_imagenet", "#EF4444", "--")]:
        ds_df = df[df["dataset"] == ds]
        for proc, proc_color, proc_label in [("tg_h", "#10B981", "Tg"), ("ab_h", "#3B82F6", "Ab"), ("di_h", "#F59E0B", "Di")]:
            agg = ds_df.groupby("transition")[proc].mean()
            # Normalize to percentage of total
            pct = agg / agg.sum() * 100
            marker = "o" if ds == "cifar100" else "x"
            alpha = 1.0 if ds == "cifar100" else 0.6
            label = f"{proc_label} ({ds[:6]})" if proc == "tg_h" else (f"{proc_label}" if ds == "cifar100" else None)
            ax.plot(pct.index, pct.values, color=proc_color, linestyle=ls,
                    marker=marker, markersize=4, alpha=alpha, label=label)
    ax.axvspan(0, 4, alpha=0.1, color="green", label="Critical period")
    ax.set_xlabel("Transition", fontsize=9)
    ax.set_ylabel("% of total events", fontsize=9)
    ax.set_title("Temporal profiles by process", fontsize=10)
    ax.legend(fontsize=6, ncol=2)

    # ── Panel B: Critical period Granger F sweep ─────────────────────────
    ax = axes[0, 1]
    if sweep_results:
        max_ts = [r["max_transition"] for r in sweep_results]
        f_ab = [r["tg_ab_F"] for r in sweep_results]
        f_di = [r["tg_di_F"] for r in sweep_results]
        ax.plot(max_ts, f_ab, "o-", color="#3B82F6", label="Tg→Ab F", linewidth=2)
        ax.plot(max_ts, f_di, "s-", color="#F59E0B", label="Tg→Di F", linewidth=2)
        ax.axhline(3.84, color="gray", linestyle="--", alpha=0.5, label="F crit (p=0.05)")
        ax.set_xlabel("Max transition included", fontsize=9)
        ax.set_ylabel("Granger F", fontsize=9)
        ax.set_title("Granger F vs window cutoff", fontsize=10)
        ax.legend(fontsize=8)

    # ── Panel C: Tg dose vs Ab-E center-of-mass ──────────────────────────
    ax = axes[0, 2]
    all_post = dose_df[dose_df["window"] == "all_post"]
    # Compute Ab center of mass per unit
    onset_data = []
    for (ds, lane, sc), group in df.groupby(["dataset", "lane", "superclass"]):
        group = group.sort_values("transition")
        ab_counts = group["ab_h"].values.astype(float)
        transitions = group["transition"].values.astype(float)
        if ab_counts.sum() > 0:
            com = np.average(transitions, weights=ab_counts)
            onset_data.append({"dataset": ds, "lane": lane, "superclass": sc, "ab_com": com})
    onset_com = pd.DataFrame(onset_data)
    merged = all_post.merge(onset_com, on=["dataset", "lane", "superclass"])
    if len(merged) > 10:
        is_c = merged["dataset"] == "cifar100"
        ax.scatter(np.log1p(merged["tg_dose"][is_c]), merged["ab_com"][is_c],
                   c="#3B82F6", s=20, alpha=0.5, label="CIFAR-100")
        ax.scatter(np.log1p(merged["tg_dose"][~is_c]), merged["ab_com"][~is_c],
                   c="#EF4444", s=30, marker="x", alpha=0.8, label="TinyIN")
        rho, p = spearmanr(merged["tg_dose"], merged["ab_com"])
        ax.set_xlabel("log(Tg dose, critical period)", fontsize=9)
        ax.set_ylabel("Ab-E center of mass", fontsize=9)
        ax.set_title(f"Tg dose → Ab-E timing\nρ={rho:.3f}, p={p:.4f}", fontsize=10)
        ax.legend(fontsize=8)

    # ── Panel D: Tg dose vs post-critical Di-E ───────────────────────────
    ax = axes[1, 0]
    if len(all_post) > 10:
        is_c = all_post["dataset"] == "cifar100"
        ax.scatter(np.log1p(all_post["tg_dose"][is_c]), np.log1p(all_post["post_di_h"][is_c]),
                   c="#3B82F6", s=20, alpha=0.5, label="CIFAR-100")
        ax.scatter(np.log1p(all_post["tg_dose"][~is_c]), np.log1p(all_post["post_di_h"][~is_c]),
                   c="#EF4444", s=30, marker="x", alpha=0.8, label="TinyIN")
        rho, p = spearmanr(all_post["tg_dose"], all_post["post_di_h"])
        ax.set_xlabel("log(Tg dose, critical period)", fontsize=9)
        ax.set_ylabel("log(post-critical Di-E)", fontsize=9)
        ax.set_title(f"Tg dose → post-critical Di-E\nρ={rho:.3f}, p={p:.4f}", fontsize=10)
        ax.legend(fontsize=8)

    # ── Panel E: Tg dose vs post-critical Ab-E ───────────────────────────
    ax = axes[1, 1]
    if len(all_post) > 10:
        is_c = all_post["dataset"] == "cifar100"
        ax.scatter(np.log1p(all_post["tg_dose"][is_c]), np.log1p(all_post["post_ab_h"][is_c]),
                   c="#3B82F6", s=20, alpha=0.5, label="CIFAR-100")
        ax.scatter(np.log1p(all_post["tg_dose"][~is_c]), np.log1p(all_post["post_ab_h"][~is_c]),
                   c="#EF4444", s=30, marker="x", alpha=0.8, label="TinyIN")
        rho, p = spearmanr(all_post["tg_dose"], all_post["post_ab_h"])
        ax.set_xlabel("log(Tg dose, critical period)", fontsize=9)
        ax.set_ylabel("log(post-critical Ab-E)", fontsize=9)
        ax.set_title(f"Tg dose → post-critical Ab-E\nρ={rho:.3f}, p={p:.4f}", fontsize=10)
        ax.legend(fontsize=8)

    # ── Panel F: Granger coefficients by window ──────────────────────────
    ax = axes[1, 2]
    if sweep_results:
        max_ts = [r["max_transition"] for r in sweep_results]
        coef_ab = [r["tg_ab_coef"] for r in sweep_results]
        coef_di = [r["tg_di_coef"] for r in sweep_results]
        ax.plot(max_ts, coef_ab, "o-", color="#3B82F6", label="Tg→Ab β", linewidth=2)
        ax.plot(max_ts, coef_di, "s-", color="#F59E0B", label="Tg→Di β", linewidth=2)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Max transition included", fontsize=9)
        ax.set_ylabel("Tg coefficient (β)", fontsize=9)
        ax.set_title("Tg coefficient vs window cutoff", fontsize=10)
        ax.legend(fontsize=8)

    plt.suptitle("Tg-E Critical Period Analysis\n"
                 "Does Tg-E during the critical period scaffold Ab-E and Di-E?",
                 fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(FIG_DIR, exist_ok=True)
    fig_path = os.path.join(FIG_DIR, "granger_causality_tg_critical.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved: {fig_path}")
    plt.close()


def main():
    print("=" * 60)
    print("Tg-E Critical Period Analysis")
    print("=" * 60)

    df = load_data()
    print(f"Combined: {len(df)} rows, {df['dataset'].nunique()} datasets, "
          f"{df['lane'].nunique()} lanes, {df['superclass'].nunique()} superclasses")

    # ── Temporal profiles ─────────────────────────────────────────────────
    identify_critical_period(df)

    # ── Analysis 1: Critical-period restricted Granger ────────────────────
    granger_results, panel_cp = restricted_window_granger(df, max_transition=4)

    # ── Analysis 2: Cumulative dose ───────────────────────────────────────
    dose_results, dose_df = cumulative_dose_analysis(df, critical_end=2)

    # ── Analysis 3: Dose-response onset ───────────────────────────────────
    onset_results, onset_df = dose_response_onset(df, dose_df)

    # ── Analysis 4: Sweep ─────────────────────────────────────────────────
    sweep_results = sweep_critical_cutoff(df)

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_results(df, granger_results, dose_results, dose_df, sweep_results)

    # ── Save ──────────────────────────────────────────────────────────────
    all_results = {
        "granger_critical_period": granger_results,
        "cumulative_dose": dose_results,
        "onset_dose_response": onset_results,
        "cutoff_sweep": sweep_results,
    }
    results_path = os.path.join(OUTPUT_DIR, "granger_causality_tg_critical_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\n  1. Critical-period Granger (t ≤ 4):")
    for label, res in granger_results.items():
        print(f"     {label}: F={res['granger_f']:.1f}, β={res['coef']:.4f}, "
              f"ΔR²={res['delta_r2']:.4f} {res['sig']}")

    print("\n  2. Cumulative dose correlations (critical period t ≤ 2):")
    for k, v in dose_results.items():
        if "rho" in v:
            print(f"     {k}: ρ={v['rho']:.3f}, p={v['p']:.4f}")

    print("\n  3. Dose-response onset:")
    for k, v in onset_results.items():
        if "rho" in v:
            print(f"     {k}: ρ={v['rho']:.3f}, p={v['p']:.4f}")

    print("\n  4. Cutoff sweep (Tg→Ab F):")
    for r in sweep_results:
        print(f"     t≤{r['max_transition']}: F={r['tg_ab_F']:.1f} (β={r['tg_ab_coef']:.4f})")


if __name__ == "__main__":
    main()
