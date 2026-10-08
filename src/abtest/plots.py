"""Figure generation for the A/B test."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path)
    plt.close(fig)
    return path


def group_sizes(counts: dict[str, int], out_dir: Path, expected: float = 0.5) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = list(counts)
    total = sum(counts.values())
    sns.barplot(x=labels, y=[counts[k] for k in labels], ax=ax, color="#4C72B0")
    for i, k in enumerate(labels):
        ax.text(i, counts[k], f"{counts[k]/total:.1%}", ha="center", va="bottom")
    ax.axhline(total * expected, color="red", linestyle="--", label=f"Expected ({expected:.0%})")
    ax.set_title("Traffic split — sample ratio")
    ax.set_ylabel("Users")
    ax.legend()
    return _save(fig, out_dir, "01_group_sizes.png")


def retention_rates(results: list[dict], out_dir: Path) -> Path:
    rows = []
    for r in results:
        rows.append({"metric": r["label"], "group": "control", "rate": r["p_control"],
                     "err": r["ci_err_control"]})
        rows.append({"metric": r["label"], "group": "treatment", "rate": r["p_treatment"],
                     "err": r["ci_err_treatment"]})
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(data=data, x="metric", y="rate", hue="group", ax=ax)
    for patch, (_, row) in zip(ax.patches, data.iterrows()):
        ax.errorbar(patch.get_x() + patch.get_width() / 2, row["rate"],
                    yerr=row["err"], fmt="none", color="black", capsize=4)
    ax.set_title("Retention by variant (95% CI)")
    ax.set_ylabel("Retention rate")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    return _save(fig, out_dir, "02_retention_rates.png")


def bootstrap_distribution(diff: np.ndarray, ci: tuple[float, float],
                           observed: float, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(diff, bins=60, ax=ax, color="#55A868", stat="density")
    ax.axvline(observed, color="black", linestyle="-", label=f"Observed ({observed:+.4f})")
    ax.axvline(ci[0], color="red", linestyle="--", label="95% CI")
    ax.axvline(ci[1], color="red", linestyle="--")
    ax.axvline(0, color="grey", linestyle=":")
    ax.set_title("Bootstrap distribution of the retention difference")
    ax.set_xlabel("Difference in retention (treatment − control)")
    ax.legend(fontsize=11)
    return _save(fig, out_dir, "03_bootstrap_diff.png")


def bayesian_posteriors(bayes: dict, out_dir: Path) -> Path:
    x = np.linspace(0.0, 0.35, 500)
    c_pdf = stats.beta.pdf(x, bayes["posterior_control_alpha"], bayes["posterior_control_beta"])
    t_pdf = stats.beta.pdf(x, bayes["posterior_treatment_alpha"], bayes["posterior_treatment_beta"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, c_pdf, color="#4C72B0", label="Control (posterior)")
    ax.plot(x, t_pdf, color="#DD8452", label="Treatment (posterior)")
    ax.fill_between(x, c_pdf, color="#4C72B0", alpha=0.15)
    ax.fill_between(x, t_pdf, color="#DD8452", alpha=0.15)
    ax.set_title(f"Posterior retention rates — P(treatment > control) = {bayes['prob_treatment_better']:.1%}")
    ax.set_xlabel("7-day retention rate")
    ax.set_ylabel("Density")
    ax.legend()
    return _save(fig, out_dir, "04_bayesian_posteriors.png")


def gamerounds(control: pd.Series, treatment: pd.Series, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cap = 500
    axes[0].hist(control.clip(upper=cap), bins=50, alpha=0.6, label="control", color="#4C72B0")
    axes[0].hist(treatment.clip(upper=cap), bins=50, alpha=0.6, label="treatment", color="#DD8452")
    axes[0].set_title("Game rounds (capped at 500)")
    axes[0].set_xlabel("Rounds")
    axes[0].legend()
    sns.boxplot(data=pd.DataFrame({"control": control, "treatment": treatment}),
                ax=axes[1], showfliers=False)
    axes[1].set_title("Game rounds (box, outliers hidden)")
    return _save(fig, out_dir, "05_gamerounds.png")


def power_curve(curve: pd.DataFrame, baseline_p: float, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(curve["relative_effect"] * 100, curve["power"], marker="o", color="#8172B3")
    ax.axhline(0.8, color="red", linestyle="--", label="80% power")
    ax.set_title(f"Power vs. relative effect (baseline {baseline_p:.1%})")
    ax.set_xlabel("Relative effect (%)")
    ax.set_ylabel("Power")
    ax.legend()
    return _save(fig, out_dir, "06_power_curve.png")
