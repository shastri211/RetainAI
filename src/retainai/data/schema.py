"""Dataset contract for the IBM Telco Customer Churn sample (PUBLIC, fictional company).

Single source of truth for column names, roles and allowed values. Validation runs on the
*raw* column names; the cleaned table uses the snake_case names in ``RAW_TO_CLEAN``.

Provenance label for this dataset: PUBLIC (vendor sample data for a fictional company).
It is not REAL data and contains no treatment, offer, cost, date or outcome-after-intervention fields.
"""

from __future__ import annotations

PROVENANCE_LABEL = "PUBLIC"

# --- raw layout -------------------------------------------------------------------------

RAW_COLUMNS: tuple[str, ...] = (
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
)

RAW_ID = "customerID"
RAW_TARGET = "Churn"

RAW_TO_CLEAN: dict[str, str] = {
    "customerID": "customer_id",
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure_months",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn",
}

# --- column roles (raw names) -----------------------------------------------------------

YES_NO = frozenset({"Yes", "No"})

# Categorical columns and their allowed raw values.
RAW_CATEGORICAL_DOMAINS: dict[str, frozenset[str]] = {
    "gender": frozenset({"Female", "Male"}),
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": frozenset({"Yes", "No", "No phone service"}),
    "InternetService": frozenset({"DSL", "Fiber optic", "No"}),
    "OnlineSecurity": frozenset({"Yes", "No", "No internet service"}),
    "OnlineBackup": frozenset({"Yes", "No", "No internet service"}),
    "DeviceProtection": frozenset({"Yes", "No", "No internet service"}),
    "TechSupport": frozenset({"Yes", "No", "No internet service"}),
    "StreamingTV": frozenset({"Yes", "No", "No internet service"}),
    "StreamingMovies": frozenset({"Yes", "No", "No internet service"}),
    "Contract": frozenset({"Month-to-month", "One year", "Two year"}),
    "PaperlessBilling": YES_NO,
    "PaymentMethod": frozenset(
        {
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        }
    ),
}

RAW_BINARY_INT_COLUMNS = ("SeniorCitizen",)  # allowed values "0" / "1"
RAW_INTEGER_COLUMNS = ("tenure",)  # whole months, >= 0
RAW_DECIMAL_COLUMNS = ("MonthlyCharges", "TotalCharges")

# Add-on services that carry the structural value "No internet service".
INTERNET_ADDON_COLUMNS = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)

# Contract term in months (for the informational contract/tenure check only).
CONTRACT_TERM_MONTHS = {"Month-to-month": 1, "One year": 12, "Two year": 24}

# --- target -----------------------------------------------------------------------------

TARGET = "churn"  # cleaned name
POSITIVE_LABEL = "Yes"  # raw value encoded as 1
NEGATIVE_LABEL = "No"  # raw value encoded as 0
TARGET_ENCODING = {NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1}

# --- thresholds used by validation ------------------------------------------------------

ID_PATTERN = r"^\d{4}-[A-Z]{5}$"
TENURE_MAX_PLAUSIBLE_MONTHS = 120  # warn above; observed max is 72
TOTAL_CHARGES_REL_DEV_WARN = 1.0  # warn if |TotalCharges / (tenure*MonthlyCharges) - 1| exceeds this
TOTAL_CHARGES_REL_DEV_TIGHT = 0.02  # reported share within this band (informational)

# --- cleaned layout ---------------------------------------------------------------------

CLEAN_ID = "customer_id"
CLEAN_FEATURE_SOURCE_COLUMNS: tuple[str, ...] = tuple(
    RAW_TO_CLEAN[c] for c in RAW_COLUMNS if c not in (RAW_ID, RAW_TARGET)
)
# Audit-only columns added by cleaning. They are NOT model features.
AUDIT_COLUMNS: tuple[str, ...] = (
    "total_charges_was_blank",
    "duplicate_group_id",
    "duplicate_group_size",
    "contract_term_exceeds_tenure",
)
CLEAN_COLUMNS: tuple[str, ...] = (CLEAN_ID, *CLEAN_FEATURE_SOURCE_COLUMNS, TARGET, *AUDIT_COLUMNS)
