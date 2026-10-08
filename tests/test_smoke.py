"""Unit tests for the statistical toolkit."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abtest import bayesian, design, sanity, stats_tests  # noqa: E402


def test_proportion_test_detects_clear_effect():
    res = stats_tests.proportion_test(100, 1000, 160, 1000, alpha=0.05)
    assert res["significant"] is True
    assert res["absolute_diff"] > 0
    assert res["ci_low"] > 0


def test_proportion_test_no_effect():
    res = stats_tests.proportion_test(100, 1000, 102, 1000, alpha=0.05)
    assert res["significant"] is False
    assert res["ci_low"] < 0 < res["ci_high"]


def test_bootstrap_ci_contains_effect():
    boot = stats_tests.bootstrap_proportions(100, 1000, 160, 1000, iterations=5000, seed=1)
    assert boot["ci_low"] < boot["mean_diff"] < boot["ci_high"]


def test_bayesian_probability():
    res = bayesian.beta_binomial(100, 1000, 160, 1000, samples=50000, seed=1)
    assert res["prob_treatment_better"] > 0.95
    assert res["expected_rel_uplift_pct"] > 0


def test_srm_detects_imbalance():
    balanced = sanity.sample_ratio_mismatch({"a": 5000, "b": 5000})
    assert balanced["srm_detected"] is False
    imbalanced = sanity.sample_ratio_mismatch({"a": 5500, "b": 4500})
    assert imbalanced["srm_detected"] is True


def test_manning_whitney_runs():
    rng = np.random.default_rng(0)
    a = rng.normal(10, 2, 500)
    b = rng.normal(10.5, 2, 500)
    res = stats_tests.mann_whitney(a, b)
    assert -1 <= res["cliffs_delta"] <= 1


def test_sample_size_positive():
    res = design.sample_size_per_group(0.20, 0.05, 0.05, 0.8)
    assert res["n_per_group"] > 0
    assert res["target_p"] > res["baseline_p"]
