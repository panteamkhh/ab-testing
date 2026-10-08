"""Frequentist, non-parametric and bootstrap inference."""
from __future__ import annotations

import numpy as np
from scipy import stats


def cohen_h(p1: float, p2: float) -> float:
    """Effect size for the difference between two proportions."""
    return float(2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1)))


def proportion_test(
    success_control: int,
    n_control: int,
    success_treatment: int,
    n_treatment: int,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Two-proportion z-test with Wald CI, chi-square and Fisher confirmations."""
    p_c = success_control / n_control
    p_t = success_treatment / n_treatment
    diff = p_t - p_c
    p_pool = (success_control + success_treatment) / (n_control + n_treatment)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_control + 1 / n_treatment))
    z = diff / se_pool if se_pool else 0.0
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

    se_unpooled = np.sqrt(p_c * (1 - p_c) / n_control + p_t * (1 - p_t) / n_treatment)
    zcrit = stats.norm.ppf(1 - alpha / 2)
    ci_low = diff - zcrit * se_unpooled
    ci_high = diff + zcrit * se_unpooled

    table = [[success_control, n_control - success_control],
             [success_treatment, n_treatment - success_treatment]]
    chi2, p_chi, _, _ = stats.chi2_contingency(table, correction=False)
    _, p_fisher = stats.fisher_exact(table)

    return {
        "metric": "proportion",
        "n_control": n_control,
        "n_treatment": n_treatment,
        "p_control": p_c,
        "p_treatment": p_t,
        "absolute_diff": diff,
        "relative_lift_pct": 100.0 * diff / p_c if p_c else float("nan"),
        "z_stat": float(z),
        "p_value": p_value,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "chi2": float(chi2),
        "p_chi2": float(p_chi),
        "p_fisher": float(p_fisher),
        "cohens_h": cohen_h(p_c, p_t),
        "significant": bool(p_value < alpha),
    }


def bootstrap_proportions(
    success_control: int,
    n_control: int,
    success_treatment: int,
    n_treatment: int,
    iterations: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, object]:
    """Bootstrap (binomial) sampling distribution for the difference in rates."""
    rng = np.random.default_rng(seed)
    p_c = success_control / n_control
    p_t = success_treatment / n_treatment
    c_draws = rng.binomial(n_control, p_c, iterations) / n_control
    t_draws = rng.binomial(n_treatment, p_t, iterations) / n_treatment
    diff = t_draws - c_draws
    ci = np.quantile(diff, [alpha / 2, 1 - alpha / 2])
    # bootstrap p-value under the sharp null (pool the groups)
    p_pool = (success_control + success_treatment) / (n_control + n_treatment)
    null_c = rng.binomial(n_control, p_pool, iterations) / n_control
    null_t = rng.binomial(n_treatment, p_pool, iterations) / n_treatment
    obs = abs(p_t - p_c)
    p_boot = float(np.mean(np.abs(null_t - null_c) >= obs))
    return {
        "diff_distribution": diff,
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "bootstrap_p_value": p_boot,
        "mean_diff": float(diff.mean()),
    }


def mann_whitney(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Mann-Whitney U test (non-parametric) + Cliff's delta effect size."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    u_stat, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = len(a), len(b)
    cliffs = 2 * u_stat / (n1 * n2) - 1
    return {
        "u_stat": float(u_stat),
        "p_value": float(p_value),
        "cliffs_delta": float(cliffs),
        "median_control": float(np.median(a)),
        "median_treatment": float(np.median(b)),
        "mean_control": float(a.mean()),
        "mean_treatment": float(b.mean()),
    }


def bootstrap_means(
    a: np.ndarray,
    b: np.ndarray,
    iterations: int = 3000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, object]:
    """Bootstrap CI for the difference in means (heavy-tailed continuous metric)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diffs = np.empty(iterations)
    for i in range(iterations):
        diffs[i] = rng.choice(b, size=len(b), replace=True).mean() - rng.choice(
            a, size=len(a), replace=True
        ).mean()
    ci = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {
        "diff_distribution": diffs,
        "mean_diff": float(b.mean() - a.mean()),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
    }
