"""
data_loader.py
--------------
Handles all data ingestion and preprocessing for the ACIS insurance dataset.
Run directly to produce the cleaned dataset: python src/data_loader.py
"""

import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Column type mappings ───────────────────────────────────────────────────────
CATEGORICAL_COLS = [
    "IsVATRegistered", "Citizenship", "LegalType", "Title", "Language",
    "Bank", "AccountType", "MaritalStatus", "Gender",
    "Country", "Province", "MainCrestaZone", "SubCrestaZone",
    "ItemType", "VehicleType", "Make", "Model", "Bodytype",
    "AlarmImmobiliser", "TrackingDevice", "NewVehicle", "WrittenOff",
    "Rebuilt", "Converted", "CrossBorder",
    "CoverCategory", "CoverType", "CoverGroup", "Section",
    "Product", "StatutoryClass", "StatutoryRiskType", "TermFrequency",
]

NUMERICAL_COLS = [
    "TotalPremium", "TotalClaims", "SumInsured", "CalculatedPremiumPerTerm",
    "CustomValueEstimate", "CapitalOutstanding", "Kilowatts",
    "Cubiccapacity", "Cylinders", "NumberOfDoors", "NumberOfVehiclesInFleet",
    "ExcessSelected", "RegistrationYear",
]

DATE_COLS = ["TransactionMonth", "VehicleIntroDate"]


def load_raw(filepath: str = "data/insurance_data.csv") -> pd.DataFrame:
    """Load the raw insurance CSV into a DataFrame."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. "
            "Download it from the course Google Drive link and place it at data/insurance_data.csv"
        )
    logger.info(f"Loading raw data from {filepath} ...")

    # The file uses '|' as delimiter based on common ACIS dataset formats;
    # fall back to comma if that fails.
    try:
        df = pd.read_csv(filepath, sep="|", low_memory=False)
        if df.shape[1] < 5:
            raise ValueError("Too few columns — trying comma separator.")
    except Exception:
        df = pd.read_csv(filepath, low_memory=False)

    logger.info(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns.")
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to their correct types."""
    df = df.copy()

    # Dates
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Categoricals
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Numerics — coerce bad strings to NaN
    for col in NUMERICAL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("Column types cast successfully.")
    return df


def assess_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary DataFrame of missing values per column."""
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    summary = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": pct
    })
    return summary[summary["missing_count"] > 0].sort_values("missing_pct", ascending=False)


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a tiered missing-value strategy:
      - Columns with > 50% missing: drop entirely.
      - Numerical columns with ≤ 50% missing: fill with median.
      - Categorical columns with ≤ 50% missing: fill with mode.
    """
    df = df.copy()
    missing_summary = assess_missing(df)

    # Drop high-missing columns
    high_missing = missing_summary[missing_summary["missing_pct"] > 50].index.tolist()
    if high_missing:
        logger.info(f"Dropping {len(high_missing)} columns with >50% missing: {high_missing}")
        df.drop(columns=high_missing, inplace=True)

    # Impute remaining
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])

    logger.info("Missing values handled.")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features used across all tasks."""
    df = df.copy()

    # Core risk metrics
    df["LossRatio"] = np.where(
        df["TotalPremium"] > 0,
        df["TotalClaims"] / df["TotalPremium"],
        np.nan
    )
    df["Margin"] = df["TotalPremium"] - df["TotalClaims"]
    df["HasClaim"] = (df["TotalClaims"] > 0).astype(int)

    # Vehicle age at transaction time
    if "TransactionMonth" in df.columns and "RegistrationYear" in df.columns:
        df["VehicleAge"] = df["TransactionMonth"].dt.year - df["RegistrationYear"]
        df["VehicleAge"] = df["VehicleAge"].clip(lower=0)

    logger.info("Feature engineering complete (LossRatio, Margin, HasClaim, VehicleAge).")
    return df


def load_clean(raw_path: str = "data/insurance_data.csv") -> pd.DataFrame:
    """Full pipeline: load → cast → handle missing → engineer features."""
    df = load_raw(raw_path)
    df = cast_types(df)
    df = handle_missing(df)
    df = engineer_features(df)
    return df


# ── Run as script (DVC stage) ─────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_clean("data/insurance_data.csv")
    out_path = "data/insurance_data_cleaned.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Cleaned dataset saved to {out_path}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")
