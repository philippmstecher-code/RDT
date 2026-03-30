#!/usr/bin/env python3
"""
Granger Causality: Does Tg-H Temporally Precede Ab-H and Di-H?

Tests whether task-general feature events (Tg-H) at transition t predict
Ab-H and Di-H at transition t+1, using the same panel data as Experiment 3.

Reuses cached superclass_transition_series.json from granger_causality_superclass.py.

Directions tested:
  1. Tg-H(t) → Ab-H(t+1)  (does Tg scaffold Ab?)
  2. Tg-H(t) → Di-H(t+1)  (does Tg scaffold Di?)
  3. Reverse: Ab-H(t) → Tg-H(t+1), Di-H(t) → Tg-H(t+1)
  4. Per-superclass tests for each forward direction
  5. Cross-correlation functions
  6. Permutation tests
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, f as f_dist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "figures")

SUPERCLASS_NAMES = [
    'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
    'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
    'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores',
    'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals',
    'trees', 'vehicles_1', 'vehicles_2'
]


def build_panel(df):
    """Build lagged panel from per-superclass per-transition data."""
    rows = []
    for (lane, sc_idx), group in df.groupby(["lane", "superclass_idx"]):
        group = group.sort_values("transition")
        transitions = group["transition"].values
        ab_h = group["ab_h"].values
        di_h = group["di_h"].values
        tg_h = group["tg_h"].values

        for i in range(len(transitions) - 1):
            rows.append({
                "lane": lane,
                "superclass_idx": sc_idx,
                "superclass": SUPERCLASS_NAMES[sc_idx],
                "t": transitions[i],
                "ab_h_t": ab_h[i],
                "di_h_t": di_h[i],
                "tg_h_t": tg_h[i],
                "ab_h_t1": ab_h[i + 1],
                "di_h_t1": di_h[i + 1],
                "tg_h_t1": tg_h[i + 1],
            })

    return pd.DataFrame(rows)


def run_granger_pair(panel, predictor_col, outcome_col, outcome_lag_col,
                     own_history_col, sc_dummies, lane_dummies, label):
    """Run a full Granger causality test for one predictor→outcome direction.

    Args:
        panel: lagged panel DataFrame
        predictor_col: column name of the predictor at time t (e.g. 'tg_h_t')
        outcome_col: column name of the outcome's own history at t (e.g. 'ab_h_t')
        outcome_lag_col: column name of the outcome at t+1 (e.g. 'ab_h_t1')
        own_history_col: same as outcome_col (the outcome's lag-1 for the restricted model)
        sc_dummies: superclass fixed-effect dummies
        lane_dummies: lane fixed-effect dummies
        label: human-readable label for printing
    """
    import statsmodels.api as sm

    print(f"\n{'=' * 60}")
    print(f"GRANGER TEST: {label}")
    print(f"{'=' * 60}")

    y = panel[outcome_lag_col].astype(float).reset_index(drop=True)

    # Restricted model: outcome(t+1) ~ outcome(t) + FE
    X_r = pd.concat([panel[[own_history_col]].astype(float).reset_index(drop=True),
                      sc_dummies.reset_index(drop=True),
                      lane_dummies.reset_index(drop=True)], axis=1)
    X_r = sm.add_constant(X_r)
    model_r = sm.OLS(y, X_r).fit()

    # Full model: outcome(t+1) ~ outcome(t) + predictor(t) + FE
    X_f = pd.concat([panel[[own_history_col, predictor_col]].astype(float).reset_index(drop=True),
                      sc_dummies.reset_index(drop=True),
                      lane_dummies.reset_index(drop=True)], axis=1)
    X_f = sm.add_constant(X_f)
    model_f = sm.OLS(y, X_f).fit()

    # Granger F-test
    f_stat = ((model_r.ssr - model_f.ssr) / 1) / (model_f.ssr / model_f.df_resid)
    p_value = 1 - f_dist.cdf(f_stat, 1, model_f.df_resid)

    coef = model_f.params[predictor_col]
    tstat = model_f.tvalues[predictor_col]
    pval_coef = float(model_f.pvalues[predictor_col])

    print(f"  Restricted R² ({own_history_col} only): {model_r.rsquared:.4f}")
    print(f"  Full R² ({own_history_col} + {predictor_col}): {model_f.rsquared:.4f}")
    print(f"  ΔR²: {model_f.rsquared - model_r.rsquared:.4f}")
    print(f"\n  {predictor_col} coefficient: {coef:.4f}")
    print(f"  {predictor_col} t-statistic: {tstat:.4f}")
    print(f"  {predictor_col} p-value:     {pval_coef:.6f}")
    print(f"\n  Granger F-test: F = {f_stat:.4f}, p = {p_value:.6f}")
    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    print(f"  Significance: {sig}")

    return {
        "restricted_r2": round(model_r.rsquared, 4),
        "full_r2": round(model_f.rsquared, 4),
        "delta_r2": round(model_f.rsquared - model_r.rsquared, 4),
        "coef": round(coef, 4),
        "tstat": round(tstat, 4),
        "pvalue": round(pval_coef, 6),
        "granger_f": round(f_stat, 4),
        "granger_p": round(p_value, 6),
        "model_r": model_r,
        "model_f": model_f,
        "f_stat": f_stat,
    }


def run_per_superclass(panel, predictor_col, outcome_col, outcome_lag_col,
                       own_history_col, label):
    """Per-superclass Granger tests with Bonferroni correction."""
    import statsmodels.api as sm

    print(f"\n{'-' * 40}")
    print(f"Per-superclass: {label}")
    print(f"  {'Superclass':<35} {'F':>8} {'p':>10} {'p_adj':>10} {'Sig':>5} {'coef':>8}")
    print("  " + "-" * 80)

    results = {}
    for sc_idx in range(20):
        sc_panel = panel[panel["superclass_idx"] == sc_idx].copy().reset_index(drop=True)
        if len(sc_panel) < 10:
            continue

        sc_lane_dummies = pd.get_dummies(sc_panel["lane"], prefix="lane", drop_first=True).astype(float)

        y_sc = sc_panel[outcome_lag_col].astype(float)
        X_r = pd.concat([sc_panel[[own_history_col]].astype(float),
                          sc_lane_dummies], axis=1)
        X_r = sm.add_constant(X_r)
        X_f = pd.concat([sc_panel[[own_history_col, predictor_col]].astype(float),
                          sc_lane_dummies], axis=1)
        X_f = sm.add_constant(X_f)

        try:
            m_r = sm.OLS(y_sc.reset_index(drop=True), X_r).fit()
            m_f = sm.OLS(y_sc.reset_index(drop=True), X_f).fit()
            f_sc = ((m_r.ssr - m_f.ssr) / 1) / (m_f.ssr / m_f.df_resid)
            p_sc = 1 - f_dist.cdf(f_sc, 1, m_f.df_resid)
            p_adj = min(p_sc * 20, 1.0)
            coef = m_f.params.get(predictor_col, 0)
            sig_sc = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else ""
            results[SUPERCLASS_NAMES[sc_idx]] = {
                "F": round(f_sc, 4),
                "p": round(p_sc, 6),
                "p_adj": round(p_adj, 6),
                "coef": round(coef, 4),
                "sig": sig_sc,
            }
            print(f"  {SUPERCLASS_NAMES[sc_idx]:<35} {f_sc:>8.3f} {p_sc:>10.6f} {p_adj:>10.6f} {sig_sc:>5} {coef:>8.4f}")
        except Exception as e:
            print(f"  {SUPERCLASS_NAMES[sc_idx]:<35} ERROR: {e}")

    n_sig = sum(1 for v in results.values() if v["sig"])
    n_positive = sum(1 for v in results.values() if v["coef"] > 0)
    print(f"\n  Significant after Bonferroni: {n_sig}/20")
    print(f"  Positive coefficient: {n_positive}/20")

    return results, n_sig, n_positive


def run_cross_correlation(panel, col_a, col_b, label):
    """Cross-correlation of col_a vs col_b at different lags."""
    print(f"\n{'-' * 40}")
    print(f"Cross-correlation: {label}")

    lag_results = {}
    for lag in [-3, -2, -1, 0, 1, 2, 3]:
        corrs = []
        for (lane, sc_idx), group in panel.groupby(["lane", "superclass_idx"]):
            group = group.sort_values("t")
            a = group[col_a].values
            b = group[col_b].values
            n = len(a)
            if abs(lag) >= n:
                continue
            if lag > 0:
                r, _ = pearsonr(a[:n - lag], b[lag:])
            elif lag < 0:
                r, _ = pearsonr(a[-lag:], b[:n + lag])
            else:
                r, _ = pearsonr(a, b)
            if not np.isnan(r):
                corrs.append(r)
        corrs = [c for c in corrs if not np.isnan(c)]
        mean_r = np.mean(corrs)
        se_r = np.std(corrs) / np.sqrt(len(corrs))
        lag_results[lag] = (mean_r, se_r, len(corrs))
        print(f"  Lag {lag:>2}: r = {mean_r:.4f} ± {se_r:.4f} (n={len(corrs)})")

    return lag_results


def run_permutation_test(panel, predictor_col, outcome_lag_col, own_history_col,
                         sc_dummies, lane_dummies, observed_f, label, n_perms=1000):
    """Permutation test: shuffle predictor within (lane, superclass) panels."""
    import statsmodels.api as sm

    print(f"\n{'-' * 40}")
    print(f"Permutation test ({n_perms} perms): {label}")

    y = panel[outcome_lag_col].astype(float).reset_index(drop=True)

    # Restricted model (no predictor) - needed for F computation
    X_r = pd.concat([panel[[own_history_col]].astype(float).reset_index(drop=True),
                      sc_dummies.reset_index(drop=True),
                      lane_dummies.reset_index(drop=True)], axis=1)
    X_r = sm.add_constant(X_r)
    model_r = sm.OLS(y, X_r).fit()

    perm_f_stats = []
    for _ in range(n_perms):
        panel_perm = panel.copy()
        for (lane, sc_idx), group in panel_perm.groupby(["lane", "superclass_idx"]):
            idx = group.index
            panel_perm.loc[idx, predictor_col] = np.random.permutation(
                panel_perm.loc[idx, predictor_col].values)

        X_perm = pd.concat([panel_perm[[own_history_col, predictor_col]].astype(float).reset_index(drop=True),
                             sc_dummies.reset_index(drop=True),
                             lane_dummies.reset_index(drop=True)], axis=1)
        X_perm = sm.add_constant(X_perm)
        model_perm = sm.OLS(y, X_perm).fit()

        f_perm = ((model_r.ssr - model_perm.ssr) / 1) / (model_perm.ssr / model_perm.df_resid)
        perm_f_stats.append(f_perm)

    perm_p = np.mean(np.array(perm_f_stats) >= observed_f)
    print(f"  Observed F: {observed_f:.4f}")
    print(f"  Permutation p-value: {perm_p:.4f}")
    print(f"  95th percentile null F: {np.percentile(perm_f_stats, 95):.4f}")

    return round(perm_p, 4)


def plot_results(panel, all_results):
    """Create 2×3 figure: top row = Tg→Ab, bottom row = Tg→Di."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for row_idx, (direction_key, direction_label, outcome_t1, outcome_t) in enumerate([
        ("tg_to_ab", "Tg-H → Ab-H", "ab_h_t1", "ab_h_t"),
        ("tg_to_di", "Tg-H → Di-H", "di_h_t1", "di_h_t"),
    ]):
        res = all_results[direction_key]

        # Panel A: Scatter
        ax = axes[row_idx, 0]
        x = np.log1p(panel["tg_h_t"])
        y = np.log1p(panel[outcome_t1])
        ax.scatter(x, y, alpha=0.15, s=8, c="#F59E0B" if row_idx == 0 else "#10B981")
        z = np.polyfit(x, y, 1)
        p_fn = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p_fn(x_line), "r-", linewidth=2)
        ax.set_xlabel("log(Tg-H count at t)", fontsize=10)
        outcome_name = "Ab-H" if row_idx == 0 else "Di-H"
        ax.set_ylabel(f"log({outcome_name} count at t+1)", fontsize=10)
        f_val = res["pooled"]["granger_f"]
        p_val = res["pooled"]["granger_p"]
        ax.set_title(f"{direction_label}\nGranger F = {f_val:.1f}, p = {p_val:.2e}", fontsize=11)

        # Panel B: Per-superclass coefficients
        ax = axes[row_idx, 1]
        sc_results = res["per_superclass"]
        sc_names = sorted(sc_results.keys())
        coefs = [sc_results[sc]["coef"] for sc in sc_names]
        colors = ["#10B981" if sc_results[sc]["sig"] else "#9CA3AF" for sc in sc_names]
        y_pos = range(len(sc_names))
        ax.barh(y_pos, coefs, color=colors, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([sc.replace("_", " ") for sc in sc_names], fontsize=7)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel(f"Tg-H(t) coefficient", fontsize=10)
        n_sig = res["n_significant"]
        ax.set_title(f"Per-superclass ({n_sig}/20 sig.)", fontsize=11)
        ax.invert_yaxis()

        # Panel C: Cross-correlation
        ax = axes[row_idx, 2]
        lag_data = res["cross_correlation"]
        lags = sorted(lag_data.keys(), key=lambda x: int(x))
        means = [lag_data[l][0] for l in lags]
        sems = [lag_data[l][1] for l in lags]
        int_lags = [int(l) for l in lags]
        ax.bar(int_lags, means, yerr=sems,
               color="#F59E0B" if row_idx == 0 else "#10B981",
               alpha=0.7, capsize=3, width=0.6)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Lag (Tg-H leads when lag > 0)", fontsize=10)
        ax.set_ylabel("Mean Pearson r", fontsize=10)
        ax.set_title(f"Cross-correlation: Tg-H vs {outcome_name}", fontsize=11)
        ax.set_xticks(int_lags)

    plt.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_path = os.path.join(FIG_DIR, "granger_causality_tg.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved: {fig_path}")
    plt.close()


def main():
    print("=" * 60)
    print("Granger Causality: Tg-H → Ab-H and Tg-H → Di-H")
    print("=" * 60)

    # ── Load cached data ──────────────────────────────────────────────────
    cached_path = os.path.join(OUTPUT_DIR, "superclass_transition_series.json")
    if not os.path.exists(cached_path):
        print(f"ERROR: Cached data not found at {cached_path}")
        print("Run granger_causality_superclass.py first to extract transition data.")
        sys.exit(1)

    print("\nLoading cached per-superclass per-transition data...")
    df = pd.read_json(cached_path)
    print(f"Loaded {len(df)} rows")

    # ── Build panel ───────────────────────────────────────────────────────
    panel = build_panel(df)
    print(f"\nPanel dataset: {len(panel)} observations")
    print(f"  Lanes: {panel['lane'].nunique()}")
    print(f"  Superclasses: {panel['superclass_idx'].nunique()}")
    print(f"  Transitions: {panel['t'].nunique()}")

    # Fixed-effect dummies
    sc_dummies = pd.get_dummies(panel["superclass_idx"], prefix="sc", drop_first=True).astype(float)
    lane_dummies = pd.get_dummies(panel["lane"], prefix="lane", drop_first=True).astype(float)

    all_results = {}

    # ══════════════════════════════════════════════════════════════════════
    # Direction 1: Tg-H(t) → Ab-H(t+1)
    # ══════════════════════════════════════════════════════════════════════
    res_tg_ab = run_granger_pair(
        panel, predictor_col="tg_h_t", outcome_col="ab_h_t",
        outcome_lag_col="ab_h_t1", own_history_col="ab_h_t",
        sc_dummies=sc_dummies, lane_dummies=lane_dummies,
        label="Tg-H(t) → Ab-H(t+1)")

    # Reverse: Ab-H(t) → Tg-H(t+1)
    res_ab_tg = run_granger_pair(
        panel, predictor_col="ab_h_t", outcome_col="tg_h_t",
        outcome_lag_col="tg_h_t1", own_history_col="tg_h_t",
        sc_dummies=sc_dummies, lane_dummies=lane_dummies,
        label="Reverse: Ab-H(t) → Tg-H(t+1)")

    # Per-superclass
    sc_tg_ab, n_sig_tg_ab, n_pos_tg_ab = run_per_superclass(
        panel, "tg_h_t", "ab_h_t", "ab_h_t1", "ab_h_t",
        "Tg-H(t) → Ab-H(t+1)")

    # Cross-correlation
    cc_tg_ab = run_cross_correlation(panel, "tg_h_t", "ab_h_t", "Tg-H vs Ab-H")

    # Permutation test
    perm_p_tg_ab = run_permutation_test(
        panel, "tg_h_t", "ab_h_t1", "ab_h_t",
        sc_dummies, lane_dummies, res_tg_ab["f_stat"],
        "Tg-H(t) → Ab-H(t+1)")

    all_results["tg_to_ab"] = {
        "pooled": {k: v for k, v in res_tg_ab.items() if k not in ("model_r", "model_f", "f_stat")},
        "reverse": {k: v for k, v in res_ab_tg.items() if k not in ("model_r", "model_f", "f_stat")},
        "per_superclass": sc_tg_ab,
        "n_significant": n_sig_tg_ab,
        "n_positive_coef": n_pos_tg_ab,
        "cross_correlation": {str(k): list(v) for k, v in cc_tg_ab.items()},
        "permutation_p": perm_p_tg_ab,
    }

    # ══════════════════════════════════════════════════════════════════════
    # Direction 2: Tg-H(t) → Di-H(t+1)
    # ══════════════════════════════════════════════════════════════════════
    res_tg_di = run_granger_pair(
        panel, predictor_col="tg_h_t", outcome_col="di_h_t",
        outcome_lag_col="di_h_t1", own_history_col="di_h_t",
        sc_dummies=sc_dummies, lane_dummies=lane_dummies,
        label="Tg-H(t) → Di-H(t+1)")

    # Reverse: Di-H(t) → Tg-H(t+1)
    res_di_tg = run_granger_pair(
        panel, predictor_col="di_h_t", outcome_col="tg_h_t",
        outcome_lag_col="tg_h_t1", own_history_col="tg_h_t",
        sc_dummies=sc_dummies, lane_dummies=lane_dummies,
        label="Reverse: Di-H(t) → Tg-H(t+1)")

    # Per-superclass
    sc_tg_di, n_sig_tg_di, n_pos_tg_di = run_per_superclass(
        panel, "tg_h_t", "di_h_t", "di_h_t1", "di_h_t",
        "Tg-H(t) → Di-H(t+1)")

    # Cross-correlation
    cc_tg_di = run_cross_correlation(panel, "tg_h_t", "di_h_t", "Tg-H vs Di-H")

    # Permutation test
    perm_p_tg_di = run_permutation_test(
        panel, "tg_h_t", "di_h_t1", "di_h_t",
        sc_dummies, lane_dummies, res_tg_di["f_stat"],
        "Tg-H(t) → Di-H(t+1)")

    all_results["tg_to_di"] = {
        "pooled": {k: v for k, v in res_tg_di.items() if k not in ("model_r", "model_f", "f_stat")},
        "reverse": {k: v for k, v in res_di_tg.items() if k not in ("model_r", "model_f", "f_stat")},
        "per_superclass": sc_tg_di,
        "n_significant": n_sig_tg_di,
        "n_positive_coef": n_pos_tg_di,
        "cross_correlation": {str(k): list(v) for k, v in cc_tg_di.items()},
        "permutation_p": perm_p_tg_di,
    }

    # ══════════════════════════════════════════════════════════════════════
    # Bonus: Tg-H(t) → Tg-H(t+1) autocorrelation baseline
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("BONUS: Tg-H autocorrelation (baseline)")
    print(f"{'=' * 60}")
    tg_auto = panel.groupby(["lane", "superclass_idx"]).apply(
        lambda g: pearsonr(g.sort_values("t")["tg_h_t"].values[:-1] if len(g) > 1 else [0],
                           g.sort_values("t")["tg_h_t"].values[1:] if len(g) > 1 else [0])[0]
        if len(g) > 2 else np.nan
    )
    tg_auto = tg_auto.dropna()
    print(f"  Mean Tg-H autocorrelation (lag 1): {tg_auto.mean():.4f} ± {tg_auto.std() / np.sqrt(len(tg_auto)):.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_results(panel, all_results)

    # ── Save results ──────────────────────────────────────────────────────
    results_path = os.path.join(OUTPUT_DIR, "granger_causality_tg_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for direction, key in [("Tg-H → Ab-H", "tg_to_ab"), ("Tg-H → Di-H", "tg_to_di")]:
        res = all_results[key]
        pooled = res["pooled"]
        rev = res["reverse"]
        print(f"\n  {direction}:")
        print(f"    Forward:  F = {pooled['granger_f']:.2f}, p = {pooled['granger_p']:.6f}, "
              f"β = {pooled['coef']:.4f}, ΔR² = {pooled['delta_r2']:.4f}")
        print(f"    Reverse:  F = {rev['granger_f']:.2f}, p = {rev['granger_p']:.6f}, "
              f"β = {rev['coef']:.4f}")
        ratio = pooled["granger_f"] / rev["granger_f"] if rev["granger_f"] > 0 else float("inf")
        print(f"    Asymmetry ratio: {ratio:.1f}×")
        print(f"    Per-superclass: {res['n_significant']}/20 sig, {res['n_positive_coef']}/20 positive")
        print(f"    Permutation p: {res['permutation_p']}")

        if pooled["granger_p"] < 0.05:
            print(f"    ✓ Significant Granger causality")
        else:
            print(f"    ✗ No significant Granger causality")


if __name__ == "__main__":
    main()
