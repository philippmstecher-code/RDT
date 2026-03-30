#!/usr/bin/env python3
"""
Experiment 3: Granger Causality — Ab-H at t Predicts Di-H at t+1 Per Superclass

Step 1: Extract per-superclass per-transition process counts from raw transition files.
Step 2: Test whether Ab-H(t) Granger-causes Di-H(t+1) per superclass.

Uses raw transition files from external storage:
  $RDT_DATA_ROOT/experiments/{exp}/lanes/{lane}/
    sae_analysis/transitions/{t_to_t+1}/{layer}.json
"""

import json
import os
import sqlite3
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────
EXT_BASE = os.environ.get("RDT_DATA_ROOT", "/Volumes/ExternalDrive/EmpiricalSignatures_data")
DB_PATH = os.path.join(EXT_BASE, "rcx.db")  # experiment metadata database
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "figures")

# ── CIFAR-100 superclass mapping ──────────────────────────────────────────
COARSE_TO_FINE = {
    0: [4, 30, 55, 72, 95],     # aquatic_mammals
    1: [1, 32, 67, 73, 91],     # fish
    2: [54, 62, 70, 82, 92],    # flowers
    3: [9, 10, 16, 28, 61],     # food_containers
    4: [0, 51, 53, 57, 83],     # fruit_and_vegetables
    5: [22, 39, 40, 86, 87],    # household_electrical_devices
    6: [5, 20, 25, 84, 94],     # household_furniture
    7: [6, 7, 14, 18, 24],      # insects
    8: [3, 42, 43, 88, 97],     # large_carnivores
    9: [12, 17, 37, 68, 76],    # large_man-made_outdoor_things
    10: [23, 33, 49, 60, 71],   # large_natural_outdoor_scenes
    11: [15, 19, 21, 31, 38],   # large_omnivores_and_herbivores
    12: [34, 63, 64, 66, 75],   # medium_mammals
    13: [26, 45, 77, 79, 99],   # non-insect_invertebrates
    14: [2, 11, 35, 46, 98],    # people
    15: [27, 29, 44, 78, 93],   # reptiles
    16: [36, 50, 65, 74, 80],   # small_mammals
    17: [47, 52, 56, 59, 96],   # trees
    18: [8, 13, 48, 58, 90],    # vehicles_1
    19: [41, 69, 81, 85, 89],   # vehicles_2
}

SUPERCLASS_NAMES = [
    'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
    'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
    'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores',
    'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals',
    'trees', 'vehicles_1', 'vehicles_2'
]

# Build fine_to_coarse mapping
FINE_TO_COARSE = {}
for coarse_idx, fine_indices in COARSE_TO_FINE.items():
    for fine_idx in fine_indices:
        FINE_TO_COARSE[fine_idx] = coarse_idx


def extract_superclass_transition_data(lane_id, exp_id, lane_name):
    """Extract per-superclass per-transition process counts from raw files."""
    trans_dir = os.path.join(EXT_BASE, "experiments", exp_id, "lanes", lane_id,
                             "sae_analysis", "transitions")

    if not os.path.exists(trans_dir):
        print(f"  SKIP {lane_name}: no transitions dir")
        return None

    transition_names = sorted(os.listdir(trans_dir))
    # Sort numerically: 0_to_1, 1_to_2, ..., 10_to_terminal
    def sort_key(name):
        parts = name.split("_to_")
        return int(parts[0])
    transition_names = sorted(transition_names, key=sort_key)

    results = []
    for trans_name in transition_names:
        trans_path = os.path.join(trans_dir, trans_name)
        if not os.path.isdir(trans_path):
            continue

        # Get transition index
        parts = trans_name.split("_to_")
        trans_idx = int(parts[0])

        # Aggregate across all layers
        sc_counts = defaultdict(lambda: {"ab_h": 0, "di_h": 0, "tg_h": 0,
                                          "as_h": 0, "de_h": 0, "total": 0})

        layer_files = [f for f in os.listdir(trans_path) if f.endswith(".json")]
        for layer_file in layer_files:
            with open(os.path.join(trans_path, layer_file)) as f:
                samples = json.load(f)

            for sample in samples:
                class_idx = sample["class_idx"]
                if class_idx not in FINE_TO_COARSE:
                    continue
                sc_idx = FINE_TO_COARSE[class_idx]
                pc = sample["process_counts"]
                sc_counts[sc_idx]["ab_h"] += pc.get("ab_h", 0)
                sc_counts[sc_idx]["di_h"] += pc.get("di_h", 0)
                sc_counts[sc_idx]["tg_h"] += pc.get("tg_h", 0)
                sc_counts[sc_idx]["as_h"] += pc.get("as_h", 0)
                sc_counts[sc_idx]["de_h"] += pc.get("de_h", 0)
                sc_counts[sc_idx]["total"] += sum(pc.get(k, 0) for k in
                                                   ["ab_h", "di_h", "tg_h", "as_h", "de_h"])

        for sc_idx in range(20):
            counts = sc_counts[sc_idx]
            results.append({
                "lane": lane_name,
                "transition": trans_idx,
                "superclass_idx": sc_idx,
                "superclass": SUPERCLASS_NAMES[sc_idx],
                "ab_h": counts["ab_h"],
                "di_h": counts["di_h"],
                "tg_h": counts["tg_h"],
                "as_h": counts["as_h"],
                "de_h": counts["de_h"],
                "total": counts["total"],
            })

    return results


def run_granger_analysis(df):
    """Run Granger causality tests on the panel data."""
    print("\n" + "=" * 60)
    print("GRANGER CAUSALITY ANALYSIS: Ab-H(t) → Di-H(t+1)")
    print("=" * 60)

    # Build lagged panel: for each (lane, superclass), align Ab-H(t) with Di-H(t+1)
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
                "di_h_t1": di_h[i + 1],
                "ab_h_t1": ab_h[i + 1],
            })

    panel = pd.DataFrame(rows)
    print(f"\nPanel dataset: {len(panel)} observations")
    print(f"  Lanes: {panel['lane'].nunique()}")
    print(f"  Superclasses: {panel['superclass_idx'].nunique()}")
    print(f"  Transitions: {panel['t'].nunique()}")

    # ── 1. Pooled OLS with fixed effects ──────────────────────────────────
    import statsmodels.api as sm
    from statsmodels.stats.anova import anova_lm

    print("\n" + "-" * 40)
    print("1. Panel regression: Di-H(t+1) ~ Di-H(t) + Ab-H(t) + FE")

    # Create dummies for fixed effects (ensure float dtype)
    sc_dummies = pd.get_dummies(panel["superclass_idx"], prefix="sc", drop_first=True).astype(float)
    lane_dummies = pd.get_dummies(panel["lane"], prefix="lane", drop_first=True).astype(float)

    # Restricted model: Di-H(t+1) ~ Di-H(t) + FE
    X_restricted = pd.concat([panel[["di_h_t"]].astype(float).reset_index(drop=True),
                               sc_dummies.reset_index(drop=True),
                               lane_dummies.reset_index(drop=True)], axis=1)
    X_restricted = sm.add_constant(X_restricted)
    y = panel["di_h_t1"].astype(float).reset_index(drop=True)

    model_restricted = sm.OLS(y, X_restricted).fit()

    # Full model: Di-H(t+1) ~ Di-H(t) + Ab-H(t) + FE
    X_full = pd.concat([panel[["di_h_t", "ab_h_t"]].astype(float).reset_index(drop=True),
                         sc_dummies.reset_index(drop=True),
                         lane_dummies.reset_index(drop=True)], axis=1)
    X_full = sm.add_constant(X_full)

    model_full = sm.OLS(y, X_full).fit()

    # Granger F-test: does adding Ab-H(t) significantly improve the model?
    f_stat = ((model_restricted.ssr - model_full.ssr) / 1) / (model_full.ssr / model_full.df_resid)
    from scipy.stats import f as f_dist
    p_value = 1 - f_dist.cdf(f_stat, 1, model_full.df_resid)

    print(f"\n  Restricted R² (Di-H(t) only): {model_restricted.rsquared:.4f}")
    print(f"  Full R² (Di-H(t) + Ab-H(t)):  {model_full.rsquared:.4f}")
    print(f"  ΔR²: {model_full.rsquared - model_restricted.rsquared:.4f}")
    print(f"\n  Ab-H(t) coefficient: {model_full.params['ab_h_t']:.4f}")
    print(f"  Ab-H(t) t-statistic: {model_full.tvalues['ab_h_t']:.4f}")
    print(f"  Ab-H(t) p-value:     {model_full.pvalues['ab_h_t']:.6f}")
    print(f"\n  Granger F-test: F = {f_stat:.4f}, p = {p_value:.6f}")
    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    print(f"  Significance: {sig}")

    # ── 2. Reverse direction: Di-H(t) → Ab-H(t+1) ──────────────────────
    print("\n" + "-" * 40)
    print("2. Reverse test: Ab-H(t+1) ~ Ab-H(t) + Di-H(t) + FE")

    y_rev = panel["ab_h_t1"].astype(float).reset_index(drop=True)
    X_rev_restricted = pd.concat([panel[["ab_h_t"]].astype(float).reset_index(drop=True),
                                   sc_dummies.reset_index(drop=True),
                                   lane_dummies.reset_index(drop=True)], axis=1)
    X_rev_restricted = sm.add_constant(X_rev_restricted)
    model_rev_restricted = sm.OLS(y_rev, X_rev_restricted).fit()

    X_rev_full = pd.concat([panel[["ab_h_t", "di_h_t"]].astype(float).reset_index(drop=True),
                             sc_dummies.reset_index(drop=True),
                             lane_dummies.reset_index(drop=True)], axis=1)
    X_rev_full = sm.add_constant(X_rev_full)
    model_rev_full = sm.OLS(y_rev, X_rev_full).fit()

    f_stat_rev = ((model_rev_restricted.ssr - model_rev_full.ssr) / 1) / \
                 (model_rev_full.ssr / model_rev_full.df_resid)
    p_value_rev = 1 - f_dist.cdf(f_stat_rev, 1, model_rev_full.df_resid)

    print(f"  Restricted R²: {model_rev_restricted.rsquared:.4f}")
    print(f"  Full R²:       {model_rev_full.rsquared:.4f}")
    print(f"  Di-H(t) coefficient: {model_rev_full.params['di_h_t']:.4f}")
    print(f"  Granger F-test: F = {f_stat_rev:.4f}, p = {p_value_rev:.6f}")
    sig_rev = "***" if p_value_rev < 0.001 else "**" if p_value_rev < 0.01 else "*" if p_value_rev < 0.05 else "ns"
    print(f"  Significance: {sig_rev}")

    # ── 3. Per-superclass Granger tests ──────────────────────────────────
    print("\n" + "-" * 40)
    print("3. Per-superclass Granger tests (Bonferroni-corrected)")
    print(f"  {'Superclass':<35} {'F':>8} {'p':>10} {'p_adj':>10} {'Sig':>5} {'coef':>8}")
    print("  " + "-" * 80)

    per_sc_results = {}
    for sc_idx in range(20):
        sc_panel = panel[panel["superclass_idx"] == sc_idx].copy().reset_index(drop=True)
        if len(sc_panel) < 10:
            continue

        # Per-superclass: only lane FE
        sc_lane_dummies = pd.get_dummies(sc_panel["lane"], prefix="lane", drop_first=True).astype(float)

        y_sc = sc_panel["di_h_t1"].astype(float)
        X_sc_r = pd.concat([sc_panel[["di_h_t"]].astype(float),
                             sc_lane_dummies], axis=1)
        X_sc_r = sm.add_constant(X_sc_r)
        X_sc_f = pd.concat([sc_panel[["di_h_t", "ab_h_t"]].astype(float),
                             sc_lane_dummies], axis=1)
        X_sc_f = sm.add_constant(X_sc_f)

        try:
            m_r = sm.OLS(y_sc.reset_index(drop=True), X_sc_r).fit()
            m_f = sm.OLS(y_sc.reset_index(drop=True), X_sc_f).fit()
            f_sc = ((m_r.ssr - m_f.ssr) / 1) / (m_f.ssr / m_f.df_resid)
            p_sc = 1 - f_dist.cdf(f_sc, 1, m_f.df_resid)
            p_adj = min(p_sc * 20, 1.0)  # Bonferroni
            coef = m_f.params.get("ab_h_t", 0)
            sig_sc = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else ""
            per_sc_results[SUPERCLASS_NAMES[sc_idx]] = {
                "F": round(f_sc, 4),
                "p": round(p_sc, 6),
                "p_adj": round(p_adj, 6),
                "coef": round(coef, 4),
                "sig": sig_sc,
            }
            print(f"  {SUPERCLASS_NAMES[sc_idx]:<35} {f_sc:>8.3f} {p_sc:>10.6f} {p_adj:>10.6f} {sig_sc:>5} {coef:>8.4f}")
        except Exception as e:
            print(f"  {SUPERCLASS_NAMES[sc_idx]:<35} ERROR: {e}")

    n_sig = sum(1 for v in per_sc_results.values() if v["sig"])
    n_positive = sum(1 for v in per_sc_results.values() if v["coef"] > 0)
    print(f"\n  Significant after Bonferroni: {n_sig}/20")
    print(f"  Positive Ab-H coefficient: {n_positive}/20")

    # ── 4. Cross-correlation analysis ────────────────────────────────────
    print("\n" + "-" * 40)
    print("4. Cross-correlation: Ab-H vs Di-H at different lags")

    # Compute mean Ab-H and Di-H per transition (averaging across lanes and superclasses)
    for lag in [-2, -1, 0, 1, 2]:
        corrs = []
        for (lane, sc_idx), group in panel.groupby(["lane", "superclass_idx"]):
            group = group.sort_values("t")
            ab = group["ab_h_t"].values
            di = group["di_h_t"].values
            if lag > 0 and len(ab) > lag:
                r, _ = pearsonr(ab[:-lag], di[lag:])
                corrs.append(r)
            elif lag < 0 and len(ab) > abs(lag):
                r, _ = pearsonr(ab[abs(lag):], di[:lag])
                corrs.append(r)
            elif lag == 0:
                r, _ = pearsonr(ab, di)
                corrs.append(r)

        corrs = [c for c in corrs if not np.isnan(c)]
        mean_r = np.mean(corrs)
        se_r = np.std(corrs) / np.sqrt(len(corrs))
        print(f"  Lag {lag:>2}: r = {mean_r:.4f} ± {se_r:.4f} (n={len(corrs)})")

    # ── 5. Permutation test ──────────────────────────────────────────────
    print("\n" + "-" * 40)
    print("5. Permutation test (shuffle temporal order, 1000 permutations)")

    observed_f = f_stat
    n_perms = 1000
    perm_f_stats = []

    for perm_i in range(n_perms):
        panel_perm = panel.copy()
        # Shuffle Ab-H(t) within each (lane, superclass) to destroy temporal structure
        for (lane, sc_idx), group in panel_perm.groupby(["lane", "superclass_idx"]):
            idx = group.index
            panel_perm.loc[idx, "ab_h_t"] = np.random.permutation(
                panel_perm.loc[idx, "ab_h_t"].values)

        # Refit
        X_perm = pd.concat([panel_perm[["di_h_t", "ab_h_t"]].astype(float).reset_index(drop=True),
                             sc_dummies.reset_index(drop=True),
                             lane_dummies.reset_index(drop=True)], axis=1)
        X_perm = sm.add_constant(X_perm)
        model_perm = sm.OLS(y, X_perm).fit()

        f_perm = ((model_restricted.ssr - model_perm.ssr) / 1) / \
                 (model_perm.ssr / model_perm.df_resid)
        perm_f_stats.append(f_perm)

    perm_p = np.mean(np.array(perm_f_stats) >= observed_f)
    print(f"  Observed F: {observed_f:.4f}")
    print(f"  Permutation p-value: {perm_p:.4f}")
    print(f"  95th percentile null F: {np.percentile(perm_f_stats, 95):.4f}")

    return {
        "pooled": {
            "restricted_r2": round(model_restricted.rsquared, 4),
            "full_r2": round(model_full.rsquared, 4),
            "delta_r2": round(model_full.rsquared - model_restricted.rsquared, 4),
            "ab_h_coef": round(model_full.params["ab_h_t"], 4),
            "ab_h_tstat": round(model_full.tvalues["ab_h_t"], 4),
            "ab_h_pvalue": round(float(model_full.pvalues["ab_h_t"]), 6),
            "granger_f": round(f_stat, 4),
            "granger_p": round(p_value, 6),
        },
        "reverse": {
            "granger_f": round(f_stat_rev, 4),
            "granger_p": round(p_value_rev, 6),
        },
        "per_superclass": per_sc_results,
        "n_significant": n_sig,
        "n_positive_coef": n_positive,
        "permutation_p": round(perm_p, 4),
    }


def plot_granger_results(df, granger_results):
    """Create visualization of Granger causality results."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Scatter of Ab-H(t) vs Di-H(t+1) aggregated
    ax = axes[0]
    rows = []
    for (lane, sc_idx), group in df.groupby(["lane", "superclass_idx"]):
        group = group.sort_values("transition")
        ab_h = group["ab_h"].values
        di_h = group["di_h"].values
        for i in range(len(ab_h) - 1):
            rows.append({"ab_h_t": ab_h[i], "di_h_t1": di_h[i + 1],
                         "superclass": SUPERCLASS_NAMES[sc_idx]})
    scatter_df = pd.DataFrame(rows)

    # Log-transform for visibility
    x = np.log1p(scatter_df["ab_h_t"])
    y = np.log1p(scatter_df["di_h_t1"])
    ax.scatter(x, y, alpha=0.15, s=8, c="#3B82F6")
    # Regression line
    z = np.polyfit(x, y, 1)
    p_fn = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p_fn(x_line), "r-", linewidth=2)
    rho, pval = spearmanr(scatter_df["ab_h_t"], scatter_df["di_h_t1"])
    ax.set_xlabel("log(Ab-H count at t)", fontsize=10)
    ax.set_ylabel("log(Di-H count at t+1)", fontsize=10)
    ax.set_title(f"Ab-H(t) vs Di-H(t+1)\nSpearman ρ = {rho:.3f}, p = {pval:.2e}", fontsize=11)

    # Panel B: Per-superclass coefficients
    ax = axes[1]
    sc_results = granger_results["per_superclass"]
    sc_names = sorted(sc_results.keys())
    coefs = [sc_results[sc]["coef"] for sc in sc_names]
    colors = ["#10B981" if sc_results[sc]["sig"] else "#9CA3AF" for sc in sc_names]
    y_pos = range(len(sc_names))
    ax.barh(y_pos, coefs, color=colors, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([sc.replace("_", " ") for sc in sc_names], fontsize=7)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Ab-H(t) coefficient in Granger model", fontsize=10)
    ax.set_title(f"Per-superclass Granger coefficients\n"
                 f"(green = significant after Bonferroni)", fontsize=11)
    ax.invert_yaxis()

    # Panel C: Cross-correlation
    ax = axes[2]
    # Recompute cross-correlations
    lag_results = {}
    for lag in [-3, -2, -1, 0, 1, 2, 3]:
        corrs = []
        for (lane, sc_idx), group in df.groupby(["lane", "superclass_idx"]):
            group = group.sort_values("transition")
            ab = group["ab_h"].values
            di = group["di_h"].values
            n = len(ab)
            if abs(lag) >= n:
                continue
            if lag > 0:
                r, _ = pearsonr(ab[:n-lag], di[lag:])
            elif lag < 0:
                r, _ = pearsonr(ab[-lag:], di[:n+lag])
            else:
                r, _ = pearsonr(ab, di)
            if not np.isnan(r):
                corrs.append(r)
        lag_results[lag] = (np.mean(corrs), np.std(corrs) / np.sqrt(len(corrs)))

    lags = sorted(lag_results.keys())
    means = [lag_results[l][0] for l in lags]
    sems = [lag_results[l][1] for l in lags]
    ax.bar(lags, means, yerr=sems, color="#3B82F6", alpha=0.7, capsize=3, width=0.6)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Lag (Ab-H leads Di-H when lag > 0)", fontsize=10)
    ax.set_ylabel("Mean Pearson r", fontsize=10)
    ax.set_title("Cross-correlation function\nAb-H vs Di-H", fontsize=11)
    ax.set_xticks(lags)
    ax.set_xticklabels([str(l) for l in lags])

    plt.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_path = os.path.join(FIG_DIR, "granger_causality_superclass.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved: {fig_path}")
    plt.close()


def main():
    print("=" * 60)
    print("Experiment 3: Granger Causality — Ab-H(t) → Di-H(t+1)")
    print("=" * 60)

    # ── Step 1: Extract per-superclass per-transition data ────────────────
    cached_path = os.path.join(OUTPUT_DIR, "superclass_transition_series.json")

    if os.path.exists(cached_path) and "--force" not in sys.argv:
        print("\nStep 1: Loading cached per-superclass per-transition data...")
        df = pd.read_json(cached_path)
        print(f"Loaded {len(df)} rows from cache")
    else:
        print("\nStep 1: Extracting per-superclass per-transition data from raw files...")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT l.id, l.name, e.id, e.name
            FROM lanes l JOIN experiments e ON l.experiment_id = e.id
            WHERE e.name = 'Multi-Arch/-Seed'
            ORDER BY l.name
        """)

        all_rows = []
        lanes_loaded = []
        for lane_id, lane_name, exp_id, exp_name in c.fetchall():
            print(f"  Processing {lane_name}...")
            rows = extract_superclass_transition_data(lane_id, exp_id, lane_name)
            if rows:
                all_rows.extend(rows)
                lanes_loaded.append(lane_name)

        conn.close()

        df = pd.DataFrame(all_rows)
        print(f"\nExtracted {len(df)} rows ({len(lanes_loaded)} lanes × "
              f"{df['superclass_idx'].nunique()} superclasses × "
              f"~{len(df) // len(lanes_loaded) // 20} transitions)")

        # Save extracted data
        df.to_json(cached_path, orient="records", indent=2)
        print(f"Saved: {cached_path}")

    # ── Step 2: Granger causality analysis ────────────────────────────────
    granger_results = run_granger_analysis(df)

    # ── Step 3: Plot ──────────────────────────────────────────────────────
    plot_granger_results(df, granger_results)

    # ── Save results ──────────────────────────────────────────────────────
    results_path = os.path.join(OUTPUT_DIR, "granger_causality_results.json")
    with open(results_path, "w") as f:
        json.dump(granger_results, f, indent=2)
    print(f"Results saved: {results_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    pooled = granger_results["pooled"]
    rev = granger_results["reverse"]
    print(f"  Forward (Ab-H → Di-H): F = {pooled['granger_f']:.2f}, p = {pooled['granger_p']:.6f}")
    print(f"  Reverse (Di-H → Ab-H): F = {rev['granger_f']:.2f}, p = {rev['granger_p']:.6f}")
    print(f"  Ab-H coefficient: {pooled['ab_h_coef']:.4f} (t = {pooled['ab_h_tstat']:.2f})")
    print(f"  ΔR² from adding Ab-H: {pooled['delta_r2']:.4f}")
    print(f"  Per-superclass significant: {granger_results['n_significant']}/20")
    print(f"  Per-superclass positive coef: {granger_results['n_positive_coef']}/20")
    print(f"  Permutation p-value: {granger_results['permutation_p']:.4f}")

    if pooled["granger_p"] < 0.05:
        print("\n  ✓ Ab-H(t) Granger-causes Di-H(t+1) — abstraction scaffolds differentiation")
    else:
        print("\n  ✗ No significant Granger causality detected")


if __name__ == "__main__":
    main()
