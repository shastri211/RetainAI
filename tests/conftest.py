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


def synthetic_raw(n: int = 400, seed: int = 0, n_duplicates: int = 6, shuffle_labels: bool = False) -> pd.DataFrame:
    """A valid raw-style table with a learnable churn signal (month-to-month, short tenure, high price)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        tenure = int(rng.integers(0, 73))
        monthly = round(float(rng.uniform(20, 110)), 2)
        contract = str(rng.choice(["Month-to-month", "One year", "Two year"], p=[0.55, 0.25, 0.20]))
        internet = str(rng.choice(["DSL", "Fiber optic", "No"], p=[0.35, 0.45, 0.20]))
        phone = str(rng.choice(["Yes", "No"], p=[0.9, 0.1]))
        addon = "No internet service" if internet == "No" else None
        row = make_row(
            customerID=f"{i:04d}-ABCDE",
            gender=str(rng.choice(["Female", "Male"])),
            SeniorCitizen=str(int(rng.random() < 0.16)),
            Partner=str(rng.choice(["Yes", "No"])),
            Dependents=str(rng.choice(["Yes", "No"])),
            tenure=str(tenure),
            PhoneService=phone,
            MultipleLines="No phone service" if phone == "No" else str(rng.choice(["Yes", "No"])),
            InternetService=internet,
            Contract=contract,
            PaperlessBilling=str(rng.choice(["Yes", "No"])),
            PaymentMethod=str(rng.choice(sorted(schema.RAW_CATEGORICAL_DOMAINS["PaymentMethod"]))),
            MonthlyCharges=f"{monthly:.2f}",
            TotalCharges=" " if tenure == 0 else f"{tenure * monthly * float(rng.uniform(0.9, 1.1)):.2f}",
        )
        for col in schema.INTERNET_ADDON_COLUMNS:
            row[col] = addon or str(rng.choice(["Yes", "No"]))
        logit = -1.2 + 1.8 * (contract == "Month-to-month") - 0.04 * tenure + 0.015 * (monthly - 65)
        row["Churn"] = "Yes" if rng.random() < 1 / (1 + np.exp(-logit)) else "No"
        rows.append(row)
    # identical-profile rows with new IDs (like the real file's tenure-1 collisions)
    for j in range(n_duplicates):
        clone = dict(rows[j])
        clone["customerID"] = f"{n + j:04d}-ABCDE"
        rows.append(clone)
    df = pd.DataFrame(rows, columns=list(schema.RAW_COLUMNS)).astype(str)
    if shuffle_labels:
        df["Churn"] = rng.permutation(df["Churn"].to_numpy())
    return df


@pytest.fixture(scope="session")
def synthetic_cleaned() -> pd.DataFrame:
    from retainai.data.clean import clean_telco

    return clean_telco(synthetic_raw())


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run tests marked slow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow: use --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
