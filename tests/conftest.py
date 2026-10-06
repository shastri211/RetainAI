"""Shared fixtures: small synthetic raw tables that satisfy the dataset contract."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from retainai.data import schema
from retainai.data.load import load_raw

REAL_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "Telco-Customer-Churn.csv"

_BASE_ROW = {
    "customerID": "0001-AAAAA",
    "gender": "Female",
    "SeniorCitizen": "0",
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": "12",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": "50.00",
    "TotalCharges": "600.00",
    "Churn": "No",
}

_NO_INTERNET = dict(
    InternetService="No",
    OnlineSecurity="No internet service",
    OnlineBackup="No internet service",
    DeviceProtection="No internet service",
    TechSupport="No internet service",
    StreamingTV="No internet service",
    StreamingMovies="No internet service",
)


def make_row(**overrides: str) -> dict[str, str]:
    row = dict(_BASE_ROW)
    row.update(overrides)
    return row


def make_raw(rows: list[dict[str, str]]) -> pd.DataFrame:
    """Build a raw-style (all text) table, assigning unique IDs unless one was given."""
    built = []
    for i, row in enumerate(rows):
        row = dict(row)
        row.setdefault("customerID", f"{i:04d}-AAAAA")
        built.append(row)
    df = pd.DataFrame(built, columns=list(schema.RAW_COLUMNS))
    return df.astype(str)


@pytest.fixture
def small_raw() -> pd.DataFrame:
    """Eight valid rows with varied values, including both churn classes."""
    return make_raw(
        [
            make_row(customerID="0001-AAAAA", Churn="No"),
            make_row(customerID="0002-AAAAB", Churn="Yes", tenure="3", TotalCharges="150.00"),
            make_row(customerID="0003-AAAAC", Churn="No", tenure="30", TotalCharges="1500.00", Contract="One year"),
            make_row(
                customerID="0004-AAAAD", Churn="No", tenure="40", MonthlyCharges="80.00",
                TotalCharges="3200.00", Contract="Two year", InternetService="Fiber optic",
            ),
            make_row(
                customerID="0005-AAAAE", Churn="Yes", tenure="5", MonthlyCharges="20.00",
                TotalCharges="100.00", PhoneService="No", MultipleLines="No phone service", **_NO_INTERNET,
            ),
            make_row(customerID="0006-AAAAF", Churn="No", gender="Male", SeniorCitizen="1", tenure="24", TotalCharges="1200.00"),
            make_row(customerID="0007-AAAAG", Churn="No", tenure="0", TotalCharges=" "),
            make_row(customerID="0008-AAAAH", Churn="Yes", tenure="8", TotalCharges="400.00", PaymentMethod="Mailed check"),
        ]
    )


@pytest.fixture(scope="session")
def real_raw() -> pd.DataFrame:
    return load_raw(REAL_CSV)


@pytest.fixture(scope="session")
def real_csv_path() -> Path:
    return REAL_CSV
