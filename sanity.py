"""Randomisation health checks: SRM, balance and outliers."""
from __future__ import annotations

import pandas as pd
from scipy import stats


def sample_ratio_mismatch(
    counts: dict[str, int], expected: float = 0.5, alpha: float = 0.001
) -> dict[str, float]:
    """Chi-square goodness-of-fit test against the expected traffic split.

    The default threshold is deliberately conservative (α = 0.001), matching
    industry practice: SRM checks are repeated, so a tight threshold avoids
    false alarms from tiny, harmless imbalances.
    """
    labels = sorted(counts)
    observed = [counts[k] for k in labels]
    total = sum(observed)
    exp = [total * expected, total * (1 - expected)]
    chi2, p = stats.chisquare(observed, f_exp=exp)
    return {
        "total": total,
        **{f"n_{k}": counts[k] for k in labels},
        "observed_ratio": observed[1] / total if total else 0.0,
        "expected_ratio": expected,
        "chi2": float(chi2),
        "p_value": float(p),
        "alpha": alpha,
        "srm_detected": bool(p < alpha),
    }


def balance_check(df: pd.DataFrame, group_col: str, metrics: list[str]) -> pd.DataFrame:
    """Compare baseline metric means between groups."""
    rows = []
    for metric in metrics:
        means = df.groupby(group_col)[metric].mean()
        overall = df[metric].mean()
        row = {"metric": metric, "overall_mean": overall}
        for group, value in means.items():
            row[f"mean_{group}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def outliers(df: pd.DataFrame, metric: str, threshold: float) -> pd.DataFrame:
    mask = df[metric] > threshold
    return df.loc[mask, ["userid", "version", metric]]


def duration_warning(n_days: int | None = None) -> str:
    return (
        "The dataset is a fixed historical snapshot; we cannot run a live sample-size "
        "check over time, so power is reported post-hoc."
    )
