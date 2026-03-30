#!/usr/bin/env python3
"""
Experiment 1: Early Process Signature Predicts Terminal Accuracy

Uses process counts from the first 1-2 transitions (first ~10% of training)
to predict terminal validation accuracy across all available lanes.

Data sources:
- External drive: sae_results.json (process_intensity) + devtrain_metrics.json (accuracy)
- Experiment metadata database for lane/experiment metadata
"""

import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────
EXT_BASE = os.environ.get("RDT_DATA_ROOT", "/Volumes/ExternalDrive/EmpiricalSignatures_data")
DB_PATH = os.path.join(EXT_BASE, "rcx.db")  # experiment metadata database
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "figures")

# Experiments to include (skip E2E tests and iNaturalist)
SKIP_EXPERIMENTS = {
    "E2E Pipeline Test 2026-03-20T11:23",
    "E2E Pipeline Test 2026-03-20T11:24",
    "iNaturalist Dataset",
}

# Architecture mapping from lane name
def get_architecture(lane_name):
    ln = lane_name.lower()
    if "resnet" in ln:
        return "ResNet-18"
    elif "vit" in ln or "vitsmall" in ln:
        return "ViT-Small"
    elif "cct" in ln:
        return "CCT-7"
    return "Unknown"

# Condition mapping from lane name
def get_condition(lane_name, exp_name):
    if "noise" in exp_name.lower():
        if "standard" in lane_name:
            return "noise_standard"
        elif "between_sc_p03" in lane_name:
            return "noise_between_sc_p03"
        elif "between_sc_p01" in lane_name:
            return "noise_between_sc_p01"
        elif "within_sc_p03" in lane_name:
            return "noise_within_sc_p03"
        elif "within_sc_p01" in lane_name:
            return "noise_within_sc_p01"
        elif "random_p03" in lane_name:
            return "noise_random_p03"
        elif "random_p01" in lane_name:
            return "noise_random_p01"
    elif "curriculum" in exp_name.lower():
        if "standard" in lane_name:
            return "curriculum_standard"
        else:
            return "curriculum_switch"
    elif "16xSAE" in lane_name:
        return "16x_SAE"
    elif "8x" in lane_name:
        return "8x_SAE"
    elif "200ep" in exp_name or "Long-Term" in exp_name:
        return "200_epoch"
    elif "IndependentInit" in exp_name:
        return "independent_init"
    else:
        return "standard"


def load_all_lanes():
    """Load process_intensity + terminal accuracy for all lanes from external drive."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT l.id, l.name, e.id, e.name
        FROM lanes l JOIN experiments e ON l.experiment_id = e.id
        ORDER BY e.name, l.name
    """)

    rows = []
    for lane_id, lane_name, exp_id, exp_name in c.fetchall():
        if exp_name in SKIP_EXPERIMENTS:
            continue

        lane_dir = os.path.join(EXT_BASE, "experiments", exp_id, "lanes", lane_id)
        sae_path = os.path.join(lane_dir, "sae_analysis", "sae_results.json")
        metrics_path = os.path.join(lane_dir, "dev_snapshots", "metrics", "devtrain_metrics.json")

        if not os.path.exists(sae_path) or not os.path.exists(metrics_path):
            print(f"  SKIP {lane_name}: missing files")
            continue

        # Load terminal accuracy
        with open(metrics_path) as f:
            metrics = json.load(f)
        terminal = metrics[-1]
        val_acc = terminal["val_accuracy"]
        train_acc = terminal["train_accuracy"]

        # Load process intensity
        with open(sae_path) as f:
            sae = json.load(f)
        process_intensity = sae.get("process_intensity", [])
        if not process_intensity:
            print(f"  SKIP {lane_name}: no process_intensity")
            continue

        # Extract first transition (0->1)
        t0 = process_intensity[0]

        arch = get_architecture(lane_name)
        condition = get_condition(lane_name, exp_name)

        row = {
            "lane_id": lane_id,
            "lane_name": lane_name,
            "experiment": exp_name,
            "architecture": arch,
            "condition": condition,
            "val_accuracy": val_acc,
            "train_accuracy": train_acc,
            "overfit_gap": train_acc - val_acc,
            # Transition 0->1 features
            "t0_ab_h": t0.get("ab_h", 0),
            "t0_di_h": t0.get("di_h", 0),
            "t0_tg_h": t0.get("tg_h", 0),
            "t0_as_h": t0.get("as_h", 0),
            "t0_de_h": t0.get("de_h", 0),
            "t0_total": t0.get("total", 1),
            "t0_churn": t0.get("churn", 0),
        }

        # Derived features (use fractions and log-transforms to avoid overflow)
        total = max(row["t0_total"], 1)
        row["t0_ab_frac"] = row["t0_ab_h"] / total
        row["t0_di_frac"] = row["t0_di_h"] / total
        row["t0_tg_frac"] = row["t0_tg_h"] / total
        row["t0_as_frac"] = row["t0_as_h"] / total
        row["t0_de_frac"] = row["t0_de_h"] / total
        row["t0_ab_di_ratio"] = row["t0_ab_h"] / max(row["t0_di_h"], 1)
        row["t0_log_ab"] = np.log1p(row["t0_ab_h"])
        row["t0_log_di"] = np.log1p(row["t0_di_h"])
        row["t0_log_tg"] = np.log1p(row["t0_tg_h"])
        row["t0_log_total"] = np.log1p(total)

        # Also extract transition 1->2 if available
        if len(process_intensity) > 1:
            t1 = process_intensity[1]
            t1_total = max(t1.get("total", 1), 1)
            row["t1_ab_frac"] = t1.get("ab_h", 0) / t1_total
            row["t1_di_frac"] = t1.get("di_h", 0) / t1_total
            row["t1_tg_frac"] = t1.get("tg_h", 0) / t1_total
            row["t1_churn"] = t1.get("churn", 0)
            row["t1_ab_di_ratio"] = t1.get("ab_h", 0) / max(t1.get("di_h", 1), 1)
        else:
            row["t1_ab_frac"] = 0
            row["t1_di_frac"] = 0
            row["t1_tg_frac"] = 0
            row["t1_churn"] = 0
            row["t1_ab_di_ratio"] = 0

        rows.append(row)

    conn.close()
    return pd.DataFrame(rows)


def run_prediction(df, feature_cols, label, include_arch=False):
    """Run LOO-CV ridge regression and return results."""
    X = df[feature_cols].values.copy()

    if include_arch:
        # One-hot encode architecture
        arch_dummies = pd.get_dummies(df["architecture"], prefix="arch")
        X = np.hstack([X, arch_dummies.values])

    y = df["val_accuracy"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    loo = LeaveOneOut()
    y_pred = np.zeros_like(y)

    for train_idx, test_idx in loo.split(X_scaled):
        model = RidgeCV(alphas=np.logspace(-3, 3, 20))
        model.fit(X_scaled[train_idx], y[train_idx])
        y_pred[test_idx] = model.predict(X_scaled[test_idx])

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    # Fit final model for feature importance
    final_model = RidgeCV(alphas=np.logspace(-3, 3, 20))
    final_model.fit(X_scaled, y)

    col_names = feature_cols.copy()
    if include_arch:
        col_names += list(arch_dummies.columns)

    importance = pd.Series(
        np.abs(final_model.coef_) * X_scaled.std(axis=0),
        index=col_names
    ).sort_values(ascending=False)

    return {
        "label": label,
        "r2": r2,
        "mae": mae,
        "y_true": y,
        "y_pred": y_pred,
        "importance": importance,
        "alpha": final_model.alpha_,
    }


def plot_results(df, results_process, results_null, results_combined):
    """Create scatter plot of predicted vs actual accuracy."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # Color mapping
    condition_colors = {
        "standard": "#2196F3",
        "noise_standard": "#4CAF50",
        "noise_between_sc_p01": "#FF9800",
        "noise_between_sc_p03": "#F44336",
        "noise_within_sc_p01": "#9C27B0",
        "noise_within_sc_p03": "#E91E63",
        "noise_random_p01": "#795548",
        "noise_random_p03": "#607D8B",
        "curriculum_standard": "#00BCD4",
        "curriculum_switch": "#CDDC39",
        "200_epoch": "#3F51B5",
        "8x_SAE": "#FF5722",
        "16x_SAE": "#009688",
        "independent_init": "#FFC107",
    }

    arch_markers = {
        "ResNet-18": "o",
        "ViT-Small": "^",
        "CCT-7": "s",
    }

    for ax, result in zip(axes, [results_null, results_process, results_combined]):
        y_true = result["y_true"]
        y_pred = result["y_pred"]

        for _, row in df.iterrows():
            idx = df.index.get_loc(row.name)
            color = condition_colors.get(row["condition"], "#999999")
            marker = arch_markers.get(row["architecture"], "o")
            ax.scatter(y_true[idx], y_pred[idx], c=color, marker=marker,
                      s=40, alpha=0.7, edgecolors="white", linewidth=0.5)

        # Perfect prediction line
        lo, hi = min(y_true.min(), y_pred.min()) - 1, max(y_true.max(), y_pred.max()) + 1
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Actual Terminal Val Accuracy (%)", fontsize=10)
        ax.set_ylabel("Predicted Terminal Val Accuracy (%)", fontsize=10)
        ax.set_title(f"{result['label']}\nR² = {result['r2']:.3f}, MAE = {result['mae']:.2f}pp",
                     fontsize=11)
        ax.set_aspect("equal")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for cond, color in sorted(condition_colors.items()):
        if cond in df["condition"].values:
            legend_elements.append(Line2D([0], [0], marker="o", color="w",
                                         markerfacecolor=color, markersize=6,
                                         label=cond.replace("_", " ")))
    for arch, marker in arch_markers.items():
        if arch in df["architecture"].values:
            legend_elements.append(Line2D([0], [0], marker=marker, color="w",
                                         markerfacecolor="gray", markersize=7,
                                         label=arch))

    fig.legend(handles=legend_elements, loc="lower center", ncol=5,
              fontsize=8, bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_path = os.path.join(FIG_DIR, "predict_terminal_accuracy.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved: {fig_path}")
    plt.close()


def main():
    print("=" * 60)
    print("Experiment 1: Early Process Signature Predicts Terminal Accuracy")
    print("=" * 60)

    # Load data
    print("\nLoading all lanes from external drive...")
    df = load_all_lanes()
    print(f"Loaded {len(df)} lanes")
    print(f"  Architectures: {df['architecture'].value_counts().to_dict()}")
    print(f"  Conditions: {df['condition'].value_counts().to_dict()}")
    print(f"  Val accuracy range: {df['val_accuracy'].min():.2f} – {df['val_accuracy'].max():.2f}")

    # Feature sets (using fractions and log-transforms to avoid numerical issues)
    process_features = [
        "t0_ab_frac", "t0_di_frac", "t0_tg_frac", "t0_as_frac", "t0_de_frac",
        "t0_churn", "t0_ab_di_ratio",
        "t0_log_ab", "t0_log_di", "t0_log_tg", "t0_log_total",
        "t1_ab_frac", "t1_di_frac", "t1_tg_frac", "t1_churn", "t1_ab_di_ratio",
    ]

    # Compact feature set for single-architecture analysis (fewer features, less overfitting)
    compact_features = [
        "t0_ab_frac", "t0_di_frac", "t0_tg_frac", "t0_de_frac",
        "t0_churn", "t0_ab_di_ratio",
        "t1_ab_frac", "t1_di_frac",
    ]

    # ── Model 1: Architecture-only baseline (null model) ──
    print("\n" + "-" * 40)
    print("Model 1: Architecture-only (null baseline)")
    arch_dummies = pd.get_dummies(df["architecture"], prefix="arch")
    null_features = list(arch_dummies.columns)
    df_with_arch = pd.concat([df, arch_dummies], axis=1)
    results_null = run_prediction(df_with_arch, null_features, "Architecture Only (Null)")
    print(f"  LOO-CV R² = {results_null['r2']:.3f}")
    print(f"  MAE = {results_null['mae']:.2f} pp")

    # ── Model 2: Process signature only (no architecture) ──
    print("\n" + "-" * 40)
    print("Model 2: Process signature only (transitions 0→1, 1→2)")
    results_process = run_prediction(df, process_features, "Process Signature Only")
    print(f"  LOO-CV R² = {results_process['r2']:.3f}")
    print(f"  MAE = {results_process['mae']:.2f} pp")
    print(f"  Top features:")
    for feat, imp in results_process["importance"].head(5).items():
        print(f"    {feat}: {imp:.3f}")

    # ── Model 3: Process + Architecture ──
    print("\n" + "-" * 40)
    print("Model 3: Process signature + Architecture")
    results_combined = run_prediction(df, process_features, "Process + Architecture",
                                      include_arch=True)
    print(f"  LOO-CV R² = {results_combined['r2']:.3f}")
    print(f"  MAE = {results_combined['mae']:.2f} pp")
    print(f"  Top features:")
    for feat, imp in results_combined["importance"].head(5).items():
        print(f"    {feat}: {imp:.3f}")

    # ── ResNet-18 only (same architecture, different conditions) ──
    print("\n" + "-" * 40)
    df_resnet = df[df["architecture"] == "ResNet-18"].reset_index(drop=True)
    print(f"Model 4: Process signature (compact), ResNet-18 only (N={len(df_resnet)})")
    results_resnet = run_prediction(df_resnet, compact_features,
                                     f"ResNet-18 Only (N={len(df_resnet)})")
    print(f"  LOO-CV R² = {results_resnet['r2']:.3f}")
    print(f"  MAE = {results_resnet['mae']:.2f} pp")
    print(f"  Val accuracy range: {df_resnet['val_accuracy'].min():.2f} – {df_resnet['val_accuracy'].max():.2f}")
    print(f"  Top features:")
    for feat, imp in results_resnet["importance"].head(5).items():
        print(f"    {feat}: {imp:.3f}")

    # ── Also predict overfit gap ──
    print("\n" + "=" * 40)
    print("BONUS: Predict Overfit Gap (train - val accuracy)")
    df_resnet_gap = df_resnet.copy()
    df_resnet_gap["val_accuracy"] = df_resnet_gap["overfit_gap"]
    results_gap = run_prediction(df_resnet_gap, compact_features,
                                  "Overfit Gap (ResNet-18)")
    print(f"  LOO-CV R² = {results_gap['r2']:.3f}")
    print(f"  MAE = {results_gap['mae']:.2f} pp")
    print(f"  Overfit gap range: {df_resnet['overfit_gap'].min():.1f} – {df_resnet['overfit_gap'].max():.1f} pp")

    # ── Spearman correlations (all lanes) ──
    from scipy.stats import spearmanr
    print("\n" + "=" * 40)
    print("SPEARMAN CORRELATIONS with terminal val accuracy (all lanes)")
    print(f"{'Feature':<25} {'ρ':>8} {'p-value':>10}")
    print("-" * 45)
    spearman_results = {}
    for feat in process_features:
        rho, pval = spearmanr(df[feat], df["val_accuracy"])
        spearman_results[feat] = {"rho": round(rho, 3), "p": round(pval, 4)}
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {feat:<23} {rho:>8.3f} {pval:>10.4f} {sig}")

    print("\nSPEARMAN CORRELATIONS with terminal val accuracy (ResNet-18 only)")
    print(f"{'Feature':<25} {'ρ':>8} {'p-value':>10}")
    print("-" * 45)
    for feat in compact_features:
        rho, pval = spearmanr(df_resnet[feat], df_resnet["val_accuracy"])
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {feat:<23} {rho:>8.3f} {pval:>10.4f} {sig}")

    print("\nSPEARMAN CORRELATIONS with overfit gap (ResNet-18 only)")
    print(f"{'Feature':<25} {'ρ':>8} {'p-value':>10}")
    print("-" * 45)
    for feat in compact_features:
        rho, pval = spearmanr(df_resnet[feat], df_resnet["overfit_gap"])
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {feat:<23} {rho:>8.3f} {pval:>10.4f} {sig}")

    # ── Plot ──
    plot_results(df, results_process, results_null, results_combined)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<40} {'R²':>8} {'MAE':>8}")
    print("-" * 60)
    for r in [results_null, results_process, results_combined, results_resnet, results_gap]:
        label = r["label"]
        if "Overfit" in label:
            label = "Overfit Gap prediction (ResNet-18)"
        print(f"{label:<40} {r['r2']:>8.3f} {r['mae']:>7.2f}pp")

    # Save results
    output = {
        "n_lanes": len(df),
        "models": {}
    }
    for r in [results_null, results_process, results_combined, results_resnet, results_gap]:
        output["models"][r["label"]] = {
            "r2": round(r["r2"], 4),
            "mae": round(r["mae"], 4),
            "alpha": round(r["alpha"], 4),
            "top_features": {k: round(v, 4) for k, v in r["importance"].head(5).items()},
        }

    out_path = os.path.join(OUTPUT_DIR, "predict_terminal_accuracy_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Save per-lane scatter data for plotting without external drive
    scatter_data = []
    for _, row in df.iterrows():
        scatter_data.append({
            "architecture": row["architecture"],
            "condition": row["condition"],
            "val_accuracy": round(float(row["val_accuracy"]), 4),
            "train_accuracy": round(float(row["train_accuracy"]), 4),
            "overfit_gap": round(float(row["overfit_gap"]), 4),
            "t0_ab_di_ratio": round(float(row["t0_ab_di_ratio"]), 4),
            "t0_ab_frac": round(float(row["t0_ab_frac"]), 6),
            "t0_di_frac": round(float(row["t0_di_frac"]), 6),
        })
    scatter_path = os.path.join(OUTPUT_DIR, "predict_scatter_cache.json")
    with open(scatter_path, "w") as f:
        json.dump(scatter_data, f, indent=2)
    print(f"Scatter cache saved: {scatter_path} ({len(scatter_data)} lanes)")


if __name__ == "__main__":
    main()
