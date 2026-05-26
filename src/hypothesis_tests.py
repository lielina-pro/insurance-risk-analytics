"""
src/hypothesis_tests.py
-----------------------
Reusable statistical testing utilities for ACIS hypothesis testing (Task 3).

Functions
---------
chi_squared_test        : Chi-squared test of independence for categorical KPIs.
t_test_two_sample       : Welch two-sample t-test for continuous KPIs.
anova_test              : One-way ANOVA for continuous KPIs with >2 groups.
summarise_hypothesis_results : Pretty-print a summary DataFrame of all test outcomes.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from scipy import stats
from typing import Sequence


ALPHA = 0.05  # significance level


# ──────────────────────────────────────────────────────────────────────────────
# Chi-squared test
# ──────────────────────────────────────────────────────────────────────────────

def chi_squared_test(
    df: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    alpha: float = ALPHA,
    verbose: bool = True,
) -> dict:
    """
    Chi-squared test of independence between a grouping variable and a
    binary/categorical outcome.

    Parameters
    ----------
    df          : DataFrame containing both columns.
    group_col   : Column that defines the groups (e.g. 'Province', 'Gender').
    outcome_col : Binary/categorical outcome column (e.g. 'Claimed').
    alpha       : Significance threshold. Default 0.05.
    verbose     : Print results to stdout.

    Returns
    -------
    dict with keys: chi2, p_value, dof, decision, contingency_table
    """
    ct = pd.crosstab(df[group_col], df[outcome_col])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    decision = "Reject H₀" if p < alpha else "Fail to Reject H₀"

    if verbose:
        print(f"\n=== Chi-squared: {group_col} × {outcome_col} ===")
        print(ct)
        print(f"\nχ² = {chi2:.4f}  |  df = {dof}  |  p = {p:.4e}")
        print(f"→  {decision}  (α = {alpha})")

    return {
        "test": "chi-squared",
        "group_col": group_col,
        "outcome_col": outcome_col,
        "chi2": chi2,
        "p_value": p,
        "dof": dof,
        "decision": decision,
        "contingency_table": ct,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Two-sample t-test
# ──────────────────────────────────────────────────────────────────────────────

def t_test_two_sample(
    group_a: pd.Series | np.ndarray,
    group_b: pd.Series | np.ndarray,
    label_a: str = "Group A",
    label_b: str = "Group B",
    kpi_name: str = "KPI",
    equal_var: bool = False,
    alpha: float = ALPHA,
    verbose: bool = True,
) -> dict:
    """
    Welch two-sample t-test (equal_var=False by default) for a continuous KPI.

    Parameters
    ----------
    group_a, group_b : Arrays/Series of KPI values for each group.
    label_a, label_b : Human-readable group labels for printing.
    kpi_name         : Name of the KPI being tested.
    equal_var        : Assume equal variances (Student's t) if True.
    alpha            : Significance threshold.
    verbose          : Print results.

    Returns
    -------
    dict with keys: t_stat, p_value, mean_a, mean_b, decision
    """
    a = pd.Series(group_a).dropna().values
    b = pd.Series(group_b).dropna().values

    t_stat, p = stats.ttest_ind(a, b, equal_var=equal_var)
    decision = "Reject H₀" if p < alpha else "Fail to Reject H₀"
    test_name = "Student t-test" if equal_var else "Welch t-test"

    if verbose:
        print(f"\n=== {test_name}: {label_a} vs {label_b} on {kpi_name} ===")
        print(f"n({label_a}) = {len(a):,}  |  mean = {a.mean():,.4f}")
        print(f"n({label_b}) = {len(b):,}  |  mean = {b.mean():,.4f}")
        print(f"t = {t_stat:.4f}  |  p = {p:.4e}")
        print(f"→  {decision}  (α = {alpha})")

    return {
        "test": test_name,
        "kpi": kpi_name,
        "label_a": label_a,
        "label_b": label_b,
        "t_stat": t_stat,
        "p_value": p,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "n_a": len(a),
        "n_b": len(b),
        "decision": decision,
    }


# ──────────────────────────────────────────────────────────────────────────────
# One-way ANOVA
# ──────────────────────────────────────────────────────────────────────────────

def anova_test(
    df: pd.DataFrame,
    group_col: str,
    kpi_col: str,
    alpha: float = ALPHA,
    verbose: bool = True,
) -> dict:
    """
    One-way ANOVA for a continuous KPI across multiple groups.

    Parameters
    ----------
    df        : DataFrame.
    group_col : Column defining the groups (e.g. 'Province').
    kpi_col   : Continuous KPI column (e.g. 'ClaimAmount').
    alpha     : Significance threshold.
    verbose   : Print results.

    Returns
    -------
    dict with keys: f_stat, p_value, group_means, decision
    """
    groups = [
        grp[kpi_col].dropna().values
        for _, grp in df.groupby(group_col)
        if len(grp[kpi_col].dropna()) > 1
    ]
    f_stat, p = stats.f_oneway(*groups)
    decision = "Reject H₀" if p < alpha else "Fail to Reject H₀"

    group_means = df.groupby(group_col)[kpi_col].mean().round(4)

    if verbose:
        print(f"\n=== One-way ANOVA: {group_col} × {kpi_col} ===")
        print(f"F = {f_stat:.4f}  |  p = {p:.4e}")
        print(f"→  {decision}  (α = {alpha})")
        print("\nGroup means:")
        print(group_means.to_string())

    return {
        "test": "one-way ANOVA",
        "group_col": group_col,
        "kpi_col": kpi_col,
        "f_stat": f_stat,
        "p_value": p,
        "group_means": group_means,
        "decision": decision,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────────────

def summarise_hypothesis_results(results: Sequence[dict]) -> pd.DataFrame:
    """
    Build a summary DataFrame from a list of result dicts returned by
    chi_squared_test, t_test_two_sample, or anova_test.

    Returns
    -------
    pd.DataFrame with columns: Hypothesis, Test, Statistic, p-value, Decision
    """
    rows = []
    for r in results:
        test = r.get("test", "")
        if "chi" in test.lower():
            stat = r.get("chi2")
            stat_label = f"χ² = {stat:.4f}"
            hypothesis = f"{r.get('group_col')} × {r.get('outcome_col')}"
        elif "anova" in test.lower():
            stat = r.get("f_stat")
            stat_label = f"F = {stat:.4f}"
            hypothesis = f"{r.get('group_col')} × {r.get('kpi_col')}"
        else:  # t-test
            stat = r.get("t_stat")
            stat_label = f"t = {stat:.4f}"
            hypothesis = f"{r.get('label_a')} vs {r.get('label_b')} on {r.get('kpi')}"

        rows.append({
            "Hypothesis": hypothesis,
            "Test": test,
            "Statistic": stat_label,
            "p-value": f"{r.get('p_value'):.4e}",
            "Decision": r.get("decision"),
        })

    return pd.DataFrame(rows)
