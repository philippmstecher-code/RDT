"""
Statistical testing utilities for developmental hypothesis testing.

Provides permutation tests, multiple comparison corrections,
paired t-tests with effect sizes, bootstrap confidence intervals,
and Fisher's exact test — all returning a common StatTestResult dataclass.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class StatTestResult:
    """Standardised container for a single statistical test outcome."""
    p_value: float
    effect_size: float
    ci_low: float
    ci_high: float
    significant: bool
    test_name: str


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 10_000,
    correlation: str = "spearman",
    seed: int = 42,
) -> StatTestResult:
    """
    Non-parametric permutation test for association between x and y.

    Shuffles y n_permutations times and computes the chosen correlation
    each time to build a null distribution, then derives a two-sided
    p-value and bootstrap CI on the observed correlation.

    Args:
        x, y: 1-D arrays of equal length.
        n_permutations: Number of permutations for the null distribution.
        correlation: 'spearman' or 'pearson'.
        seed: RNG seed for reproducibility.

    Returns:
        StatTestResult with observed correlation as effect_size.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert len(x) == len(y), "x and y must be the same length"

    if correlation == "spearman":
        corr_fn = lambda a, b: float(stats.spearmanr(a, b).statistic)
    elif correlation == "pearson":
        corr_fn = lambda a, b: float(stats.pearsonr(a, b).statistic)
    else:
        raise ValueError(f"Unknown correlation type: {correlation}")

    observed = corr_fn(x, y)

    rng = np.random.default_rng(seed)
    null_dist = np.empty(n_permutations)
    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        null_dist[i] = corr_fn(x, y_perm)

    # Two-sided p-value
    p_value = float(np.mean(np.abs(null_dist) >= np.abs(observed)))
    p_value = max(p_value, 1.0 / (n_permutations + 1))  # floor

    # Bootstrap CI on observed correlation
    ci_low, ci_high = bootstrap_ci(
        np.column_stack([x, y]),
        stat_fn=lambda data: corr_fn(data[:, 0], data[:, 1]),
        n_bootstrap=min(n_permutations, 5000),
        seed=seed,
    )

    return StatTestResult(
        p_value=p_value,
        effect_size=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        significant=p_value < 0.05,
        test_name=f"permutation_{correlation}",
    )


# ---------------------------------------------------------------------------
# Bonferroni correction
# ---------------------------------------------------------------------------

def bonferroni_correct(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[bool]:
    """
    Bonferroni correction for multiple comparisons.

    Returns a list of booleans indicating which tests remain significant
    after correcting for family-wise error rate.
    """
    n = len(p_values)
    if n == 0:
        return []
    threshold = alpha / n
    return [p <= threshold for p in p_values]


# ---------------------------------------------------------------------------
# Paired t-test with Cohen's d
# ---------------------------------------------------------------------------

def paired_ttest(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str = "two-sided",
) -> StatTestResult:
    """
    Paired t-test with Cohen's d (paired) effect size and bootstrap CI.

    Args:
        x, y: Paired 1-D arrays.
        alternative: 'two-sided', 'less', or 'greater'.

    Returns:
        StatTestResult.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diffs = x - y
    n = len(diffs)

    # Cohen's d for paired samples
    sd = float(diffs.std(ddof=1)) if n > 1 else 0.0
    cohens_d = float(diffs.mean() / sd) if sd > 1e-10 else 0.0

    try:
        result = stats.ttest_rel(x, y, alternative=alternative)
        p_value = float(result.pvalue)
    except Exception:
        p_value = 1.0

    ci_low, ci_high = bootstrap_ci(
        diffs,
        stat_fn=lambda d: float(np.mean(d)),
        n_bootstrap=5000,
    )

    return StatTestResult(
        p_value=p_value,
        effect_size=cohens_d,
        ci_low=ci_low,
        ci_high=ci_high,
        significant=p_value < 0.05,
        test_name="paired_ttest",
    )


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data: np.ndarray,
    stat_fn: Optional[Callable] = None,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Percentile bootstrap confidence interval.

    Args:
        data: 1-D or 2-D array. If 1-D, stat_fn defaults to np.mean.
        stat_fn: Function mapping a resampled array to a scalar statistic.
        n_bootstrap: Number of bootstrap resamples.
        alpha: Significance level (default 0.05 → 95% CI).
        seed: RNG seed.

    Returns:
        (ci_low, ci_high)
    """
    data = np.asarray(data)
    if stat_fn is None:
        stat_fn = lambda d: float(np.mean(d))

    rng = np.random.default_rng(seed)
    n = len(data)
    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_stats[i] = stat_fn(data[idx])

    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return lo, hi


# ---------------------------------------------------------------------------
# Fisher's exact test
# ---------------------------------------------------------------------------

def fisher_exact_test(
    table_2x2: np.ndarray,
    alternative: str = "two-sided",
) -> StatTestResult:
    """
    Fisher's exact test on a 2x2 contingency table.

    Args:
        table_2x2: 2x2 array-like [[a,b],[c,d]].
        alternative: 'two-sided', 'less', or 'greater'.

    Returns:
        StatTestResult with odds_ratio as effect_size.
    """
    table = np.asarray(table_2x2, dtype=int)
    try:
        result = stats.fisher_exact(table, alternative=alternative)
        odds_ratio = float(result.statistic) if hasattr(result, "statistic") else float(result[0])
        p_value = float(result.pvalue) if hasattr(result, "pvalue") else float(result[1])
    except Exception:
        odds_ratio, p_value = 1.0, 1.0

    return StatTestResult(
        p_value=p_value,
        effect_size=odds_ratio,
        ci_low=0.0,
        ci_high=0.0,
        significant=p_value < 0.05,
        test_name="fisher_exact",
    )
