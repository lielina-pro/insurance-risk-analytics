"""
eda_utils.py
------------
Reusable EDA helper functions called by notebooks/01_eda.ipynb.
All plotting functions return the matplotlib Figure so callers can save or display.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from typing import List, Optional

# ── Global style ──────────────────────────────────────────────────────────────
PALETTE = "Set2"
ACCENT = "#2E75B6"
WARN = "#C00000"
FIG_DPI = 120

sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.05)
plt.rcParams.update({
    "figure.dpi": FIG_DPI,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


# ── 1. Descriptive statistics ─────────────────────────────────────────────────
def summarise_numerics(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Return a formatted descriptive-statistics table for numeric columns."""
    available = [c for c in cols if c in df.columns]
    summary = df[available].describe(percentiles=[0.25, 0.5, 0.75, 0.95]).T
    summary["skewness"] = df[available].skew()
    summary["kurtosis"] = df[available].kurt()
    return summary.round(4)


def summarise_categoricals(df: pd.DataFrame, cols: List[str], top_n: int = 10) -> dict:
    """Return a dict of value-count tables for categorical columns."""
    result = {}
    for col in cols:
        if col in df.columns:
            result[col] = df[col].value_counts(dropna=False).head(top_n)
    return result


# ── 2. Univariate ─────────────────────────────────────────────────────────────
def plot_numeric_distributions(df: pd.DataFrame, cols: List[str],
                                log_scale: bool = True) -> plt.Figure:
    """Grid of histograms with KDE for numeric columns."""
    available = [c for c in cols if c in df.columns]
    ncols = 3
    nrows = int(np.ceil(len(available) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(available):
        data = df[col].dropna()
        if log_scale and data.min() >= 0 and data.max() > 0:
            data = np.log1p(data)
            xlabel = f"log(1 + {col})"
        else:
            xlabel = col
        axes[i].hist(data, bins=50, color=ACCENT, alpha=0.75, edgecolor="white")
        axes[i].set_title(col)
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel("Count")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribution of Numerical Features (log-scale where applicable)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_categorical_bars(df: pd.DataFrame, cols: List[str], top_n: int = 12) -> plt.Figure:
    """Grid of horizontal bar charts for categorical columns."""
    available = [c for c in cols if c in df.columns]
    ncols = 2
    nrows = int(np.ceil(len(available) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(available):
        counts = df[col].value_counts(dropna=False).head(top_n)
        colors = sns.color_palette(PALETTE, len(counts))
        axes[i].barh(counts.index.astype(str)[::-1], counts.values[::-1],
                     color=colors[::-1])
        axes[i].set_title(col)
        axes[i].set_xlabel("Count")
        for spine in ["top", "right"]:
            axes[i].spines[spine].set_visible(False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Categorical Feature Distributions (top values)", fontsize=14,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ── 3. Missing values ─────────────────────────────────────────────────────────
def plot_missing_values(df: pd.DataFrame, threshold: float = 0.0) -> plt.Figure:
    """Horizontal bar chart of missing-value percentages."""
    pct = (df.isnull().mean() * 100).sort_values(ascending=False)
    pct = pct[pct > threshold]
    if pct.empty:
        print("No missing values found.")
        return None

    fig, ax = plt.subplots(figsize=(9, max(4, len(pct) * 0.35)))
    colors = [WARN if v > 50 else ACCENT for v in pct.values]
    ax.barh(pct.index[::-1], pct.values[::-1], color=colors[::-1])
    ax.axvline(50, color=WARN, linestyle="--", linewidth=1, label="50% threshold")
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing Value Profile", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return fig


# ── 4. Bivariate / Multivariate ───────────────────────────────────────────────
def plot_premium_vs_claims(df: pd.DataFrame,
                            hue_col: Optional[str] = "Province") -> plt.Figure:
    """Scatter plot of TotalPremium vs TotalClaims coloured by a categorical."""
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = df[(df["TotalPremium"] > 0) & (df["TotalClaims"] >= 0)].copy()
    plot_df["log_premium"] = np.log1p(plot_df["TotalPremium"])
    plot_df["log_claims"] = np.log1p(plot_df["TotalClaims"])

    if hue_col and hue_col in df.columns:
        categories = plot_df[hue_col].astype(str).unique()[:12]
        palette = sns.color_palette(PALETTE, len(categories))
        color_map = dict(zip(categories, palette))
        for cat in categories:
            sub = plot_df[plot_df[hue_col].astype(str) == cat]
            ax.scatter(sub["log_premium"], sub["log_claims"],
                       label=cat, color=color_map[cat], alpha=0.4, s=15)
        ax.legend(title=hue_col, bbox_to_anchor=(1.02, 1), loc="upper left",
                  fontsize=8, title_fontsize=9)
    else:
        ax.scatter(plot_df["log_premium"], plot_df["log_claims"],
                   color=ACCENT, alpha=0.3, s=10)

    ax.set_xlabel("log(1 + TotalPremium)")
    ax.set_ylabel("log(1 + TotalClaims)")
    ax.set_title(f"TotalPremium vs TotalClaims by {hue_col}", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_correlation_matrix(df: pd.DataFrame, cols: List[str]) -> plt.Figure:
    """Annotated heatmap of Pearson correlations."""
    available = [c for c in cols if c in df.columns]
    corr = df[available].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Pearson Correlation Matrix — Numerical Features", fontweight="bold")
    fig.tight_layout()
    return fig


# ── 5. Geographic trends ──────────────────────────────────────────────────────
def plot_province_metrics(df: pd.DataFrame) -> plt.Figure:
    """3-panel chart: mean premium, loss ratio, and claim frequency by Province."""
    if "Province" not in df.columns:
        raise ValueError("'Province' column not found.")

    grp = df.groupby("Province", observed=True).agg(
        MeanPremium=("TotalPremium", "mean"),
        LossRatio=("LossRatio", "mean"),
        ClaimFrequency=("HasClaim", "mean"),
    ).sort_values("LossRatio", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1 — Mean Premium
    axes[0].barh(grp.index, grp["MeanPremium"], color=ACCENT)
    axes[0].set_title("Mean Premium (ZAR)")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"R{x:,.0f}"))

    # Panel 2 — Loss Ratio
    colors = [WARN if v > 1 else ACCENT for v in grp["LossRatio"]]
    axes[1].barh(grp.index, grp["LossRatio"], color=colors)
    axes[1].axvline(1.0, color=WARN, linestyle="--", linewidth=1.2, label="Break-even")
    axes[1].set_title("Mean Loss Ratio")
    axes[1].legend(fontsize=8)

    # Panel 3 — Claim Frequency
    axes[2].barh(grp.index, grp["ClaimFrequency"] * 100, color=sns.color_palette(PALETTE)[2])
    axes[2].set_title("Claim Frequency (%)")
    axes[2].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    for ax in axes:
        ax.set_xlabel("")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle("Risk & Profitability Metrics by Province", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    return fig


# ── 6. Outlier detection ──────────────────────────────────────────────────────
def plot_boxplots(df: pd.DataFrame, cols: List[str], log_scale: bool = True) -> plt.Figure:
    """Box plots for outlier detection on key numerical features."""
    available = [c for c in cols if c in df.columns]
    fig, axes = plt.subplots(1, len(available), figsize=(len(available) * 3, 5))
    if len(available) == 1:
        axes = [axes]

    for ax, col in zip(axes, available):
        data = df[col].dropna()
        if log_scale and data.min() >= 0:
            data = np.log1p(data)
            label = f"log(1+{col})"
        else:
            label = col
        ax.boxplot(data, vert=True, patch_artist=True,
                   boxprops=dict(facecolor=ACCENT, alpha=0.6),
                   medianprops=dict(color=WARN, linewidth=2))
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle("Outlier Detection — Key Financial Features (log scale)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ── 7. Creative insight plots ─────────────────────────────────────────────────
def plot_loss_ratio_heatmap(df: pd.DataFrame,
                             row_col: str = "Province",
                             col_col: str = "VehicleType") -> plt.Figure:
    """Pivot heatmap of mean loss ratio across two categorical dimensions."""
    if row_col not in df.columns or col_col not in df.columns:
        raise ValueError(f"Columns '{row_col}' or '{col_col}' not in dataframe.")

    pivot = df.pivot_table(values="LossRatio", index=row_col,
                           columns=col_col, aggfunc="mean", observed=True)
    # Keep top vehicle types for readability
    top_cols = df[col_col].value_counts().head(8).index
    pivot = pivot[pivot.columns.intersection(top_cols)]

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn_r",
                center=1.0, linewidths=0.4, ax=ax,
                cbar_kws={"label": "Mean Loss Ratio"})
    ax.set_title(f"Mean Loss Ratio — {row_col} × {col_col}", fontweight="bold", fontsize=13)
    ax.set_xlabel(col_col)
    ax.set_ylabel(row_col)
    fig.tight_layout()
    return fig


def plot_top_vehicle_makes(df: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """Bar chart of mean claim amount for top vehicle makes."""
    if "Make" not in df.columns:
        raise ValueError("'Make' column not found.")

    claims_df = df[df["TotalClaims"] > 0].copy()
    top_makes = claims_df.groupby("Make", observed=True)["TotalClaims"].agg(
        ["mean", "count"]
    ).rename(columns={"mean": "MeanClaim", "count": "Count"})
    top_makes = top_makes[top_makes["Count"] >= 30].nlargest(top_n, "MeanClaim")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_makes.index[::-1], top_makes["MeanClaim"][::-1],
                   color=sns.color_palette("flare", len(top_makes)))
    ax.set_xlabel("Mean Claim Amount (ZAR)")
    ax.set_title(f"Top {top_n} Vehicle Makes by Mean Claim Amount\n(policies with ≥30 claims)",
                 fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R{x:,.0f}"))
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def plot_temporal_trends(df: pd.DataFrame) -> plt.Figure:
    """Monthly trend of total premiums, total claims, and mean loss ratio."""
    if "TransactionMonth" not in df.columns:
        raise ValueError("'TransactionMonth' column not found.")

    monthly = df.groupby(pd.Grouper(key="TransactionMonth", freq="ME")).agg(
        TotalPremium=("TotalPremium", "sum"),
        TotalClaims=("TotalClaims", "sum"),
        PolicyCount=("PolicyID", "count"),
    )
    monthly["LossRatio"] = monthly["TotalClaims"] / monthly["TotalPremium"]
    monthly = monthly.dropna()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top panel — Premium & Claims
    axes[0].fill_between(monthly.index, monthly["TotalPremium"] / 1e6,
                         alpha=0.4, color=ACCENT, label="Total Premium (M ZAR)")
    axes[0].fill_between(monthly.index, monthly["TotalClaims"] / 1e6,
                         alpha=0.5, color=WARN, label="Total Claims (M ZAR)")
    axes[0].set_ylabel("Amount (M ZAR)")
    axes[0].set_title("Monthly Premium & Claims Volume", fontweight="bold")
    axes[0].legend()
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R{x:.1f}M"))

    # Bottom panel — Loss Ratio
    axes[1].plot(monthly.index, monthly["LossRatio"], color=WARN,
                 linewidth=2, marker="o", markersize=4)
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Break-even (1.0)")
    axes[1].set_ylabel("Loss Ratio")
    axes[1].set_title("Monthly Portfolio Loss Ratio", fontweight="bold")
    axes[1].legend()
    axes[1].set_xlabel("Month")

    fig.suptitle("Temporal Trends — Feb 2014 to Aug 2015", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_gender_risk(df: pd.DataFrame) -> plt.Figure:
    """Side-by-side comparison of loss ratio and claim frequency by gender."""
    if "Gender" not in df.columns:
        raise ValueError("'Gender' column not found.")

    valid_genders = ["Male", "Female", "M", "F", "MALE", "FEMALE"]
    grp = df[df["Gender"].astype(str).str.strip().isin(valid_genders)].copy()
    grp["Gender"] = grp["Gender"].astype(str).str.capitalize().str[:6]

    agg = grp.groupby("Gender", observed=True).agg(
        MeanLossRatio=("LossRatio", "mean"),
        ClaimFrequency=("HasClaim", "mean"),
        MeanClaimAmount=("TotalClaims", lambda x: x[x > 0].mean()),
        Count=("PolicyID", "count"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metrics = [("MeanLossRatio", "Mean Loss Ratio"),
               ("ClaimFrequency", "Claim Frequency"),
               ("MeanClaimAmount", "Mean Claim Amount (ZAR)")]

    colors = [ACCENT, WARN]
    for ax, (metric, label) in zip(axes, metrics):
        ax.bar(agg["Gender"], agg[metric], color=colors[:len(agg)])
        ax.set_title(label)
        ax.set_xlabel("Gender")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle("Risk Profile Comparison by Gender", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig
