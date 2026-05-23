"""
test_data_loader.py
-------------------
Unit tests for src/data_loader.py — runs in CI without the real dataset.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import cast_types, assess_missing, handle_missing, engineer_features


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_df():
    """Minimal DataFrame mimicking the ACIS schema."""
    return pd.DataFrame({
        "PolicyID": ["P001", "P002", "P003", "P004"],
        "TransactionMonth": ["2014-03-01", "2014-04-01", "2014-05-01", "2014-06-01"],
        "TotalPremium": [1000.0, 1500.0, np.nan, 800.0],
        "TotalClaims": [0.0, 500.0, 200.0, 0.0],
        "Province": ["Gauteng", "Western Cape", "KwaZulu-Natal", "Gauteng"],
        "Gender": ["Male", "Female", "Male", "Female"],
        "VehicleType": ["Passenger", "Passenger", "Light Commercial", "Passenger"],
        "Make": ["Toyota", "BMW", "Toyota", "Ford"],
        "RegistrationYear": [2010, 2015, 2008, 2012],
    })


# ── cast_types ────────────────────────────────────────────────────────────────
def test_cast_types_dates(sample_df):
    df = cast_types(sample_df)
    assert pd.api.types.is_datetime64_any_dtype(df["TransactionMonth"])


def test_cast_types_categoricals(sample_df):
    df = cast_types(sample_df)
    assert str(df["Province"].dtype) == "category"
    assert str(df["Gender"].dtype) == "category"


def test_cast_types_numerics(sample_df):
    df = cast_types(sample_df)
    assert pd.api.types.is_float_dtype(df["TotalPremium"])
    assert pd.api.types.is_float_dtype(df["TotalClaims"])


# ── assess_missing ────────────────────────────────────────────────────────────
def test_assess_missing_detects_nan(sample_df):
    summary = assess_missing(sample_df)
    assert "TotalPremium" in summary.index


def test_assess_missing_no_false_positives(sample_df):
    summary = assess_missing(sample_df)
    assert "PolicyID" not in summary.index


# ── handle_missing ────────────────────────────────────────────────────────────
def test_handle_missing_fills_numeric(sample_df):
    df = cast_types(sample_df)
    df_clean = handle_missing(df)
    assert df_clean["TotalPremium"].isnull().sum() == 0


# ── engineer_features ─────────────────────────────────────────────────────────
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
    # Row 1: premium=1500, claims=500 → margin=1000
    assert df.loc[1, "Margin"] == pytest.approx(1000.0)


def test_engineer_features_has_claim(sample_df):
    df = cast_types(sample_df)
    df = handle_missing(df)
    df = engineer_features(df)
    assert "HasClaim" in df.columns
    assert df.loc[0, "HasClaim"] == 0   # claims = 0
    assert df.loc[1, "HasClaim"] == 1   # claims = 500


def test_engineer_features_vehicle_age(sample_df):
    df = cast_types(sample_df)
    df = handle_missing(df)
    df = engineer_features(df)
    assert "VehicleAge" in df.columns
    # TransactionMonth 2014, RegistrationYear 2010 → age = 4
    assert df.loc[0, "VehicleAge"] == 4
