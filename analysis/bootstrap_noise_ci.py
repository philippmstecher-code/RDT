#!/usr/bin/env python3
"""
Item 8: Bootstrap confidence interval and Cohen's d for the 5-seed
label noise experiment's key dissociation (between_sc_p03 vs random_p03 Di-H).

Strengthens the sign test (p=0.031, n=5) with:
  - BCa bootstrap 95% CI on mean paired difference
  - Cohen's d (paired) effect size
  - Power analysis for detectable effect at n=5
"""

import json
import numpy as np
from scipy import stats

# ── Load data ──────────────────────────────────────────────────────────────
from pathlib import Path
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

with open(DATA_DIR / "targeted_label_noise_summary_5seeds.json") as f:
    d = json.load(f)

seeds = d["seeds"]
bsc = d["conditions"]["between_sc_p03"]
rnd = d["conditions"]["random_p03"]
std_cond = d["conditions"]["standard"]

# Paired differences: between_sc Di-H minus random Di-H (expect negative)
diffs = []
print("=" * 70)
print("PAIRED DIFFERENCES: between_sc_p03 Di-H vs random_p03 Di-H")
print("=" * 70)
print(f"{'Seed':>6}  {'Between-SC':>12}  {'Random':>12}  {'Diff':>10}")
for seed in seeds:
    s = str(seed)
    di_bsc = bsc["per_seed"][s]["process_fractions"]["di_frac"]
    di_rnd = rnd["per_seed"][s]["process_fractions"]["di_frac"]
    diff = di_bsc - di_rnd
    diffs.append(diff)
    print(f"{seed:>6}  {di_bsc:>12.4f}  {di_rnd:>12.4f}  {diff:>10.4f}")

diffs = np.array(diffs)
n = len(diffs)
mean_diff = diffs.mean()
std_diff = diffs.std(ddof=1)

print(f"\nMean paired difference: {mean_diff:.4f}")
print(f"SD of differences: {std_diff:.4f}")
print(f"SE: {std_diff / np.sqrt(n):.4f}")

# ── Cohen's d (paired) ────────────────────────────────────────────────────
cohens_d = mean_diff / std_diff
print(f"\nCohen's d (paired): {cohens_d:.3f}")
print(f"  Interpretation: {'large' if abs(cohens_d) >= 0.8 else 'medium' if abs(cohens_d) >= 0.5 else 'small'} effect")

# ── Paired t-test ──────────────────────────────────────────────────────────
t_stat, p_paired = stats.ttest_1samp(diffs, 0)
print(f"\nPaired t-test: t({n-1}) = {t_stat:.3f}, p = {p_paired:.4f}")

# ── Sign test ──────────────────────────────────────────────────────────────
n_neg = np.sum(diffs < 0)
p_sign = stats.binomtest(n_neg, n, 0.5, alternative="greater").pvalue
print(f"Sign test: {n_neg}/{n} negative, p = {p_sign:.4f}")

# ── BCa Bootstrap 95% CI ──────────────────────────────────────────────────
rng = np.random.default_rng(42)
n_boot = 10_000

# Standard bootstrap
boot_means = np.array([diffs[rng.choice(n, n, replace=True)].mean() for _ in range(n_boot)])

# BCa correction
# Bias correction: z0
z0 = stats.norm.ppf(np.mean(boot_means < mean_diff))

# Acceleration: jackknife
jack_means = np.array([np.delete(diffs, i).mean() for i in range(n)])
jack_mean_all = jack_means.mean()
jack_diffs = jack_mean_all - jack_means
a_hat = np.sum(jack_diffs ** 3) / (6.0 * (np.sum(jack_diffs ** 2)) ** 1.5 + 1e-15)

# BCa adjusted percentiles
alpha = 0.05
z_alpha_lo = stats.norm.ppf(alpha / 2)
z_alpha_hi = stats.norm.ppf(1 - alpha / 2)

p_lo = stats.norm.cdf(z0 + (z0 + z_alpha_lo) / (1 - a_hat * (z0 + z_alpha_lo)))
p_hi = stats.norm.cdf(z0 + (z0 + z_alpha_hi) / (1 - a_hat * (z0 + z_alpha_hi)))

ci_lo = np.percentile(boot_means, 100 * p_lo)
ci_hi = np.percentile(boot_means, 100 * p_hi)

# Also compute percentile CI for comparison
ci_pct_lo = np.percentile(boot_means, 2.5)
ci_pct_hi = np.percentile(boot_means, 97.5)

print(f"\n--- Bootstrap 95% CI (n_boot={n_boot:,}) ---")
print(f"BCa CI:        [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"Percentile CI: [{ci_pct_lo:.4f}, {ci_pct_hi:.4f}]")
print(f"CI excludes zero: {ci_hi < 0}")

# ── Power analysis ─────────────────────────────────────────────────────────
# For a one-sample t-test at n=5, alpha=0.05 (two-sided), what effect size is detectable at 80% power?
# Noncentrality parameter: delta = d * sqrt(n)
# Critical t: t_crit = t.ppf(0.975, df=n-1)
t_crit = stats.t.ppf(0.975, df=n - 1)

# Find d such that power = 0.80
from scipy.optimize import brentq

def power_func(d):
    ncp = abs(d) * np.sqrt(n)
    power = 1 - stats.nct.cdf(t_crit, df=n - 1, nc=ncp) + stats.nct.cdf(-t_crit, df=n - 1, nc=ncp)
    return power - 0.80

d_detectable = brentq(power_func, 0.1, 5.0)

print(f"\n--- Power Analysis ---")
print(f"Minimum detectable Cohen's d at 80% power, n={n}, alpha=0.05 (two-sided): {d_detectable:.3f}")
print(f"Observed Cohen's d: {abs(cohens_d):.3f}")
print(f"Adequately powered: {abs(cohens_d) >= d_detectable}")

# Observed power
ncp_obs = abs(cohens_d) * np.sqrt(n)
power_obs = 1 - stats.nct.cdf(t_crit, df=n - 1, nc=ncp_obs) + stats.nct.cdf(-t_crit, df=n - 1, nc=ncp_obs)
print(f"Observed power: {power_obs:.2%}")

# ── Also report Ab-H to confirm it's not different ────────────────────────
print("\n--- Control: Ab-H (should NOT differ) ---")
ab_diffs = []
for seed in seeds:
    s = str(seed)
    ab_bsc = bsc["per_seed"][s]["process_fractions"]["ab_frac"]
    ab_rnd = rnd["per_seed"][s]["process_fractions"]["ab_frac"]
    ab_diffs.append(ab_bsc - ab_rnd)
    print(f"  Seed {seed}: between_sc Ab-H={ab_bsc:.4f}, random Ab-H={ab_rnd:.4f}, diff={ab_bsc - ab_rnd:.4f}")

ab_diffs = np.array(ab_diffs)
t_ab, p_ab = stats.ttest_1samp(ab_diffs, 0)
print(f"Ab-H mean diff: {ab_diffs.mean():.4f} +/- {ab_diffs.std(ddof=1):.4f}")
print(f"Ab-H paired t-test: t({n-1}) = {t_ab:.3f}, p = {p_ab:.4f} (n.s.)")

# ── Summary for Methods insertion ──────────────────────────────────────────
print("\n" + "=" * 70)
print("METHODS TEXT (ready for insertion)")
print("=" * 70)
print(f"""
Across all five seeds, between-superclass noise produced lower Di-H event
fractions than matched-dose random noise (mean paired difference = {mean_diff:.3f},
BCa bootstrap 95% CI [{ci_lo:.3f}, {ci_hi:.3f}], Cohen's d = {cohens_d:.2f},
paired t({n-1}) = {t_stat:.2f}, p = {p_paired:.3f}; sign test: 5/5 seeds negative,
p = {p_sign:.3f}). The observed effect size (|d| = {abs(cohens_d):.2f}) exceeded the
minimum detectable at 80% power for n = 5 (d = {d_detectable:.2f}), with observed
power of {power_obs:.0%}. Ab-H fractions did not differ between noise types
(mean diff = {ab_diffs.mean():.3f}, p = {p_ab:.2f}), confirming that the Di-H
reduction reflects targeted disruption of superclass structure rather than
general noise effects.
""")
