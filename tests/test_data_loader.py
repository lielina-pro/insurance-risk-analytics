"""
test_data_loader.py — Unit tests for src/data_loader.py
Uses a minimal DataFrame matching the real insurance_data.csv schema.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import cast_types
import assess_missing
import handle_missing
import engineer_features


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "CustomerID": ["AC-001", "AC-002", "AC-003", "AC-004"],
        "Age": [30, 45, np.nan, 60],
        "Gender": ["Male", "Female", "Male", "Female"],
        "Province": ["Addis Ababa", "Oromia", "Tigray", "Addis Ababa"],
        "VehicleType": ["Sedan", "SUV", "Sedan", "Truck"],
        "AnnualIncome": [50000.0, 80000.0, 60000.0, np.nan],
        "RiskScore": [40, 70, 55, 80],
        "AnnualPremium": [2000.0, 3000.0, 1500.0, 2500.0],
        "Deductible": [500.0, 500.0, 250.0, 500.0],
        "NCD": [20, 0, 10, 0],
        "PastClaims": [1, 3, 0, 2],
        "Claimed": ["False", "True", "False", "True"],
        "ClaimAmount": [0.0, 8000.0, 0.0, 12000.0],
        "TotalPremium": [2000.0, 3000.0, 1500.0, 2500.0],
        "TotalClaims": [0.0, 8000.0, 0.0, 12000.0],
        "CoverType": ["Comprehensive", "Third Party", "Comprehensive", "Comprehensive"],
        "AutoMake": ["Toyota", "Lifan", "Suzuki", "Toyota"],
        "VehicleModel": ["Corolla", "620", "Grand Vitara", "Hilux"],
        "CustomValueEstimate": [30000.0, 25000.0, 40000.0, 55000.0],
        "ZipCode": [10001, 20001, 30001, 10002],
        "TransactionDate": ["2024-01-15", "2024-03-20", "2024-06-10", "2025-01-05"],
    })


def test_cast_types_dates(sample_df):
    df = cast_types(sample_df)
    assert pd.api.types.is_datetime64_any_dtype(df["TransactionDate"])


def test_cast_types_categoricals(sample_df):
    df = cast_types(sample_df)
    assert str(df["Province"].dtype) == "category"
    assert str(df["Gender"].dtype) == "category"


def test_cast_types_numerics(sample_df):
    df = cast_types(sample_df)
    assert pd.api.types.is_float_dtype(df["TotalPremium"])
    assert pd.api.types.is_float_dtype(df["TotalClaims"])


def test_assess_missing_detects_nan(sample_df):
    summary = assess_missing(sample_df)
    assert "Age" in summary.index or "AnnualIncome" in summary.index


def test_assess_missing_no_false_positives(sample_df):
    summary = assess_missing(sample_df)
    assert "CustomerID" not in summary.index


def test_handle_missing_fills_numeric(sample_df):
    df = cast_types(sample_df)
    df_clean = handle_missing(df)
    assert df_clean["Age"].isnull().sum() == 0
    assert df_clean["AnnualIncome"].isnull().sum() == 0


def test_engineer_features_loss_ratio(sample_df):
    df = cast_types(sample_df)
    df = handle_missing(df)
    df = engineer_features(df)
    assert "LossRatio" in df.columns


def test_engineer_features_margin(sample_df):
    df = cast_types(sample_df)
    df = handle_missing(df)
    df = engineer_features(df)
    assert "Margin" in df.columns
    assert df.loc[1, "Margin"] == pytest.approx(-5000.0)


def test_engineer_features_has_claim(sample_df):
    df = cast_types(sample_df)
    df = handle_missing(df)
    df = engineer_features(df)
    assert df.loc[0, "HasClaim"] == 0
    assert df.loc[1, "HasClaim"] == 1


def test_engineer_features_transaction_year(sample_df):
    df = cast_types(sample_df)
    df = handle_missing(df)
    df = engineer_features(df)
    assert "TransactionYear" in df.columns
    assert df.loc[0, "TransactionYear"] == 2024
