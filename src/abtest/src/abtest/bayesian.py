"""Bayesian A/B analysis via a Beta-Binomial model."""
from __future__ import annotations

import numpy as np


def beta_binomial(
    control_success: int,
    control_n: int,
    treatment_success: int,
    treatment_n: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    samples: int = 200000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, object]:
    """Posterior for each rate and derived decision metrics."""
    a_c, b_c = prior_alpha + control_success, prior_beta + control_n - control_success
    a_t, b_t = prior_alpha + treatment_success, prior_beta + treatment_n - treatment_success

    rng = np.random.default_rng(seed)
    post_c = rng.beta(a_c, b_c, samples)
    post_t = rng.beta(a_t, b_t, samples)

    diff = post_t - post_c
    rel = diff / post_c
    prob_better = float(np.mean(post_t > post_c))
    ci_diff = np.quantile(diff, [alpha / 2, 1 - alpha / 2])
    ci_rel = np.quantile(rel, [alpha / 2, 1 - alpha / 2])

    # Expected loss (a.k.a. risk) of choosing each option.
    loss_ship_treatment = float(np.mean(np.maximum(post_c - post_t, 0)))
    loss_ship_control = float(np.mean(np.maximum(post_t - post_c, 0)))

    return {
        "posterior_control_alpha": a_c,
        "posterior_control_beta": b_c,
        "posterior_treatment_alpha": a_t,
        "posterior_treatment_beta": b_t,
        "posterior_control_mean": a_c / (a_c + b_c),
        "posterior_treatment_mean": a_t / (a_t + b_t),
        "prob_treatment_better": prob_better,
        "expected_abs_uplift": float(diff.mean()),
        "expected_rel_uplift_pct": float(rel.mean() * 100),
        "ci_abs_low": float(ci_diff[0]),
        "ci_abs_high": float(ci_diff[1]),
        "ci_rel_low_pct": float(ci_rel[0] * 100),
        "ci_rel_high_pct": float(ci_rel[1] * 100),
        "expected_loss_ship_treatment": loss_ship_treatment,
        "expected_loss_ship_control": loss_ship_control,
        "posterior_control": post_c,
        "posterior_treatment": post_t,
    }
