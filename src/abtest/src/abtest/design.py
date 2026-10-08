"""Experiment design: hypotheses, power, sample size and MDE."""
from __future__ import annotations

from typing import Any

from scipy.optimize import brentq
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from .config import Config


def hypotheses(cfg: Config) -> dict[str, str]:
    control = cfg.get("experiment", "control")
    treatment = cfg.get("experiment", "treatment")
    metric = cfg.get("experiment", "primary_metric")
    return {
        "primary_metric": metric,
        "control": control,
        "treatment": treatment,
        "H0": f"The {metric} rate is identical for {control} and {treatment}.",
        "H1": f"The {metric} rate differs between {control} and {treatment} (two-sided).",
        "test": "Two-proportion z-test (primary) with bootstrap & Bayesian confirmation",
        "alpha": f"{cfg.get('experiment', 'alpha', default=0.05):.2f}",
        "power": f"{cfg.get('experiment', 'power', default=0.80):.2f}",
    }


def sample_size_per_group(
    baseline_p: float, mde_relative: float, alpha: float, power: float, ratio: float = 1.0
) -> dict[str, float]:
    """Required sample size per group to detect a relative MDE."""
    p2 = min(baseline_p * (1.0 + mde_relative), 1.0)
    effect = proportion_effectsize(p2, baseline_p)
    analysis = NormalIndPower()
    n = analysis.solve_power(
        effect_size=effect, alpha=alpha, power=power, ratio=ratio, alternative="two-sided"
    )
    return {
        "baseline_p": baseline_p,
        "target_p": p2,
        "relative_mde": mde_relative,
        "cohens_h": effect,
        "n_per_group": float(n),
        "n_total": float(n * (1 + ratio)),
    }


def achieved_power(p1: float, p2: float, n1: int, n2: int, alpha: float) -> float:
    """Post-hoc power given the observed rates and sample sizes."""
    effect = abs(proportion_effectsize(p2, p1))
    ratio = n2 / n1 if n1 else 1.0
    return float(
        NormalIndPower().power(
            effect_size=effect, nobs1=n1, alpha=alpha, ratio=ratio, alternative="two-sided"
        )
    )


def detectable_mde(
    baseline_p: float, n1: int, n2: int, alpha: float, power: float
) -> dict[str, float]:
    """Smallest relative effect this experiment could reliably detect."""
    ratio = n2 / n1 if n1 else 1.0
    required_h = NormalIndPower().solve_power(
        effect_size=None, nobs1=n1, alpha=alpha, power=power, ratio=ratio,
        alternative="two-sided",
    )

    def gap(p2: float) -> float:
        return proportion_effectsize(p2, baseline_p) - required_h

    try:
        p2 = brentq(gap, min(baseline_p + 1e-6, 0.999), 0.9999)
    except ValueError:
        p2 = min(baseline_p * 1.5, 1.0)
    return {
        "absolute_mde": abs(p2 - baseline_p),
        "relative_mde": abs(p2 - baseline_p) / baseline_p if baseline_p else float("nan"),
        "target_p": p2,
    }


def power_curve(
    baseline_p: float, n1: int, n2: int, alpha: float, effects: list[float] | None = None
) -> list[dict[str, Any]]:
    """Power for a range of relative effects (for a plot)."""
    effects = effects or [i / 100 for i in range(1, 21)]
    rows = []
    for rel in effects:
        p2 = min(baseline_p * (1 + rel), 1.0)
        rows.append(
            {
                "relative_effect": rel,
                "p2": p2,
                "power": achieved_power(baseline_p, p2, n1, n2, alpha),
            }
        )
    return rows
