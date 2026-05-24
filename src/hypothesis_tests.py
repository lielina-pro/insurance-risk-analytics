"""
hypothesis_tests.py
-------------------
Statistical test wrappers for Task 3 — A/B Hypothesis Testing.
Populated fully in the task-3 branch.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict


def chi_squared_test(df: pd.DataFrame, group_col: str,
                     kpi_col: str) -> Dict:
    """Chi-squared test for independence between a grouping column and a binary KPI."""
    contingency = pd.crosstab(df[group_col], df[kpi_col])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    return {"test": "chi-squared", "chi2": chi2, "p_value": p,
            "dof": dof, "reject_h0": p < 0.05}


def t_test_two_groups(group_a: pd.Series, group_b: pd.Series,
                      equal_var: bool = False) -> Dict:
    """Welch t-test comparing means of two independent groups."""
    t_stat, p = stats.ttest_ind(group_a.dropna(), group_b.dropna(),
                                equal_var=equal_var)
    return {"test": "t-test (Welch)", "t_stat": t_stat, "p_value": p,
            "reject_h0": p < 0.05}


def z_test_proportions(n_a: int, x_a: int, n_b: int, x_b: int) -> Dict:
    """Two-proportion z-test (e.g., claim frequency between two groups)."""
    from statsmodels.stats.proportion import proportions_ztest
    count = np.array([x_a, x_b])
    nobs = np.array([n_a, n_b])
    z_stat, p = proportions_ztest(count, nobs)
    return {"test": "z-test (proportions)", "z_stat": z_stat,
            "p_value": p, "reject_h0": p < 0.05}
