"""Orchestration of the A/B testing pipeline."""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from scipy import stats

from . import bayesian, data, design, plots, sanity, stats_tests
from .config import Config
from .report import build_report

log = logging.getLogger(__name__)


def _decide(primary: dict, bayes: dict, srm: dict) -> dict[str, str]:
    if srm["srm_detected"]:
        return {
            "verdict": "Inconclusive — randomisation broken",
            "rationale": "A sample-ratio mismatch was detected, so results cannot be trusted. "
                         "Re-run the experiment.",
        }
    if primary["significant"] and primary["absolute_diff"] > 0:
        if bayes["prob_treatment_better"] >= 0.95:
            return {
                "verdict": "Ship gate_40",
                "rationale": "The primary metric improves significantly and the Bayesian "
                             "probability of superiority is ≥ 95%.",
            }
        return {
            "verdict": "Promising — extend the test",
            "rationale": "The effect is statistically significant but Bayesian evidence is "
                         "not yet decisive.",
        }
    if primary["significant"] and primary["absolute_diff"] < 0:
        return {
            "verdict": "Do not ship gate_40",
            "rationale": "The primary metric is significantly worse in the treatment group.",
        }
    return {
        "verdict": "No significant difference — keep control",
        "rationale": "The primary metric difference is not statistically significant; "
                     "keep gate_30 (the simpler / status-quo option).",
    }


def _plot_metric(control: pd.DataFrame, treatment: pd.DataFrame, label: str,
                 col: str, alpha: float) -> dict:
    res = stats_tests.proportion_test(
        int(control[col].sum()), len(control), int(treatment[col].sum()), len(treatment), alpha
    )
    zcrit = stats.norm.ppf(1 - alpha / 2)
    err_c = zcrit * math.sqrt(res["p_control"] * (1 - res["p_control"]) / len(control))
    err_t = zcrit * math.sqrt(res["p_treatment"] * (1 - res["p_treatment"]) / len(treatment))
    return {
        "label": label,
        "p_control": res["p_control"],
        "p_treatment": res["p_treatment"],
        "ci_err_control": err_c,
        "ci_err_treatment": err_t,
    }


def run(cfg: Config, force_download: bool = False) -> dict:
    cfg.ensure_dirs()
    figures = cfg.path("paths", "figures")
    alpha = float(cfg.get("experiment", "alpha", default=0.05))
    seed = cfg.seed
    primary_metric = cfg.get("experiment", "primary_metric")

    # 1. Data ---------------------------------------------------------------
    data.download(cfg, force=force_download)
    df = data.load(cfg)
    control, treatment = data.split(df, cfg)
    log.info("Loaded %s users (%s control / %s treatment)",
             f"{len(df):,}", f"{len(control):,}", f"{len(treatment):,}")

    # 2. Design -------------------------------------------------------------
    baseline_p = float(control[primary_metric].mean())
    sample_size = design.sample_size_per_group(
        baseline_p, float(cfg.get("experiment", "mde_relative", default=0.02)),
        alpha, float(cfg.get("experiment", "power", default=0.80)),
    )
    detectable = design.detectable_mde(baseline_p, len(control), len(treatment), alpha,
                                       float(cfg.get("experiment", "power", default=0.80)))
    power_curve = pd.DataFrame(design.power_curve(baseline_p, len(control), len(treatment), alpha))
    design_res = {"baseline_p": baseline_p, "sample_size": sample_size,
                  "detectable_mde": detectable}

    # 3. Sanity -------------------------------------------------------------
    counts = df["version"].value_counts().to_dict()
    srm = sanity.sample_ratio_mismatch(
        counts, alpha=float(cfg.get("experiment", "srm_alpha", default=0.001))
    )
    metrics = [primary_metric] + list(cfg.get("experiment", "secondary_metrics"))
    balance = sanity.balance_check(df, "version", metrics)
    outliers = sanity.outliers(df, "sum_gamerounds",
                               float(cfg.get("experiment", "outlier_threshold", default=49000)))

    # 4. Primary metric -----------------------------------------------------
    c_succ, c_n = int(control[primary_metric].sum()), len(control)
    t_succ, t_n = int(treatment[primary_metric].sum()), len(treatment)
    primary = stats_tests.proportion_test(c_succ, c_n, t_succ, t_n, alpha)
    boot = stats_tests.bootstrap_proportions(
        c_succ, c_n, t_succ, t_n,
        iterations=int(cfg.get("experiment", "bootstrap_iterations", default=10000)),
        alpha=alpha, seed=seed,
    )
    bayes = bayesian.beta_binomial(
        c_succ, c_n, t_succ, t_n,
        prior_alpha=float(cfg.get("experiment", "prior_alpha", default=1.0)),
        prior_beta=float(cfg.get("experiment", "prior_beta", default=1.0)),
        samples=int(cfg.get("experiment", "bayes_samples", default=200000)),
        alpha=alpha, seed=seed,
    )

    # 5. Secondary metrics --------------------------------------------------
    r1 = stats_tests.proportion_test(
        int(control["retention_1"].sum()), len(control),
        int(treatment["retention_1"].sum()), len(treatment), alpha,
    )
    mw = stats_tests.mann_whitney(
        control["sum_gamerounds"].to_numpy(), treatment["sum_gamerounds"].to_numpy()
    )
    gb = stats_tests.bootstrap_means(
        control["sum_gamerounds"].to_numpy(), treatment["sum_gamerounds"].to_numpy(),
        iterations=min(int(cfg.get("experiment", "bootstrap_iterations", default=10000)), 3000),
        alpha=alpha, seed=seed,
    )
    secondary = {"retention_1": r1, "gamerounds": mw, "gamerounds_bootstrap": gb}

    # 6. Power & decision ---------------------------------------------------
    achieved = design.achieved_power(primary["p_control"], primary["p_treatment"], c_n, t_n, alpha)
    power = {"achieved_power": achieved}
    decision = _decide(primary, bayes, srm)

    # 7. Figures ------------------------------------------------------------
    plots.group_sizes(counts, figures)
    plot_results = [
        _plot_metric(control, treatment, "1-day retention", "retention_1", alpha),
        _plot_metric(control, treatment, "7-day retention", "retention_7", alpha),
    ]
    plots.retention_rates(plot_results, figures)
    plots.bootstrap_distribution(boot["diff_distribution"],
                                 (boot["ci_low"], boot["ci_high"]),
                                 primary["absolute_diff"], figures)
    plots.bayesian_posteriors(bayes, figures)
    plots.gamerounds(control["sum_gamerounds"], treatment["sum_gamerounds"], figures)
    plots.power_curve(power_curve, baseline_p, figures)

    # 8. Tables & report ----------------------------------------------------
    tables = cfg.path("paths", "tables")
    pd.DataFrame([primary]).to_csv(tables / "primary_metric.csv", index=False)
    pd.DataFrame([{k: v for k, v in bayes.items()
                   if not isinstance(v, np.ndarray)}]).to_csv(tables / "bayesian.csv", index=False)
    pd.DataFrame([srm]).to_csv(tables / "srm.csv", index=False)
    balance.to_csv(tables / "balance.csv", index=False)
    power_curve.to_csv(tables / "power_curve.csv", index=False)

    results = {
        "hypotheses": design.hypotheses(cfg),
        "design": design_res,
        "srm": srm,
        "balance": balance,
        "outliers": outliers,
        "primary": primary,
        "bootstrap": {k: v for k, v in boot.items() if k != "diff_distribution"},
        "bayesian": bayes,
        "secondary": secondary,
        "power": power,
        "decision": decision,
    }
    report_path = build_report(results, cfg)
    log.info("Report -> %s", report_path)
    _print_summary(results)
    return results


def _print_summary(results: dict) -> None:
    p = results["primary"]
    b = results["bayesian"]
    print("\n================ A/B TEST — Cookie Cats ================")
    print(f"Control  ({p['n_control']:,} users): retention_7 = {p['p_control']:.3%}")
    print(f"Treatment({p['n_treatment']:,} users): retention_7 = {p['p_treatment']:.3%}")
    print(f"Lift             : {p['relative_lift_pct']:+.2f}%  (diff {p['absolute_diff']:+.4%})")
    print(f"95% CI (diff)    : [{p['ci_low']:+.4%}, {p['ci_high']:+.4%}]")
    print(f"p-value          : {p['p_value']:.4f}  (significant: {p['significant']})")
    print(f"P(treatment>ctrl): {b['prob_treatment_better']:.2%}")
    print(f"SRM detected     : {results['srm']['srm_detected']}")
    print(f"DECISION         : {results['decision']['verdict']}")
    print("========================================================\n")
