"""
src/modeling.py
---------------
Reusable modeling utilities for ACIS Task 4 — Statistical Modeling & Risk-Based Pricing.

Functions
---------
prepare_features        : Feature engineering + encoding + train/test split.
evaluate_regression     : RMSE and R² for regression models.
evaluate_classification : Accuracy, Precision, Recall, F1 for classifiers.
plot_feature_importance : Bar chart of top-N feature importances.
plot_shap_summary       : SHAP beeswarm summary plot for the best model.
build_pricing_table     : Combines P(claim) × Severity + loading into a premium estimate.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering & preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_features(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Engineer features, encode categoricals, and split into train/test sets.

    Parameters
    ----------
    df           : Raw DataFrame.
    target_col   : Name of the target column.
    drop_cols    : Additional columns to drop before modeling.
    test_size    : Fraction reserved for testing (default 0.2).
    random_state : Reproducibility seed.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names
    """
    data = df.copy()

    # ── Date features ────────────────────────────────────────────────────────
    if "TransactionDate" in data.columns:
        data["TransactionDate"] = pd.to_datetime(data["TransactionDate"], errors="coerce")
        data["TxYear"]  = data["TransactionDate"].dt.year
        data["TxMonth"] = data["TransactionDate"].dt.month
        data.drop(columns=["TransactionDate"], inplace=True)

    # ── Drop identifier and target-leaking columns ───────────────────────────
    always_drop = ["CustomerID", "ClaimAmount", "TotalClaims", "Claimed"]
    to_drop = list(set(always_drop + (drop_cols or [])) - {target_col})
    data.drop(columns=[c for c in to_drop if c in data.columns], inplace=True)

    # ── Encode categoricals ───────────────────────────────────────────────────
    cat_cols = data.select_dtypes(include=["object", "string", "bool"]).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        data[col] = le.fit_transform(data[col].astype(str))

    # ── Split ─────────────────────────────────────────────────────────────────
    X = data.drop(columns=[target_col])
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Train: {X_train.shape[0]:,} rows  |  Test: {X_test.shape[0]:,} rows")
    print(f"Target: '{target_col}'  |  Features: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test, list(X.columns)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_regression(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    verbose: bool = True,
) -> dict:
    """Compute RMSE and R² for a regression model."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    if verbose:
        print(f"\n{'='*45}")
        print(f"  {model_name}")
        print(f"{'='*45}")
        print(f"  RMSE : {rmse:,.2f}")
        print(f"  R²   : {r2:.4f}")

    return {"model": model_name, "RMSE": rmse, "R2": r2}


def evaluate_classification(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    verbose: bool = True,
) -> dict:
    """Compute Accuracy, Precision, Recall, F1 for a classifier."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    if verbose:
        print(f"\n{'='*45}")
        print(f"  {model_name}")
        print(f"{'='*45}")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1        : {f1:.4f}")
        print(classification_report(y_true, y_pred, zero_division=0))

    return {
        "model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(
    model,
    feature_names: list[str],
    model_name: str = "Model",
    top_n: int = 10,
    save_path: str | None = None,
) -> None:
    """
    Bar chart of top-N feature importances.
    Works with sklearn (feature_importances_) and XGBoost models.
    """
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))
    top.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontsize=13)
    ax.set_xlabel("Importance")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def plot_shap_summary(
    model,
    X_test: pd.DataFrame,
    model_name: str = "Model",
    save_path: str | None = None,
    max_display: int = 10,
) -> None:
    """
    SHAP beeswarm summary plot. Requires the `shap` package.
    Falls back gracefully with a message if shap is not installed.
    """
    try:
        import shap
        explainer = shap.Explainer(model, X_test)
        shap_values = explainer(X_test)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values, X_test,
            max_display=max_display,
            show=False,
            plot_title=f"SHAP Summary — {model_name}",
        )
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Saved → {save_path}")
        plt.show()

    except ImportError:
        print(
            "⚠  'shap' is not installed. Run:  pip install shap\n"
            "   Then re-run this cell to generate the SHAP plot."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Risk-based pricing framework
# ─────────────────────────────────────────────────────────────────────────────

def build_pricing_table(
    df_test: pd.DataFrame,
    prob_claim: np.ndarray,
    pred_severity: np.ndarray,
    expense_loading: float = 0.10,
    profit_margin: float = 0.05,
) -> pd.DataFrame:
    """
    Compute model-implied premium using:
        Premium = P(claim) × Predicted Severity × (1 + expense + profit)

    Parameters
    ----------
    df_test         : Test-set rows (used for index alignment).
    prob_claim      : Predicted probability of a claim (from classifier).
    pred_severity   : Predicted claim amount (from severity model).
    expense_loading : Expense ratio added on top (default 10%).
    profit_margin   : Profit loading (default 5%).

    Returns
    -------
    DataFrame with columns: ProbClaim, PredSeverity, ModelPremium
    """
    loading = 1 + expense_loading + profit_margin

    pricing = pd.DataFrame({
        "ProbClaim"    : prob_claim,
        "PredSeverity" : pred_severity,
        "ModelPremium" : prob_claim * pred_severity * loading,
    }, index=df_test.index)

    print(f"\nPricing summary (n={len(pricing):,}):")
    print(pricing[["ProbClaim", "PredSeverity", "ModelPremium"]].describe().round(2))

    return pricing
