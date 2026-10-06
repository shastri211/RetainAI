"""Project-wide paths and constants."""

from __future__ import annotations

from pathlib import Path

# Repository root (assumes the package is installed in editable mode from ``src/``).
ROOT = Path(__file__).resolve().parents[2]

RAW_CSV = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

CLEANED_CSV_NAME = "telco_cleaned.csv"
FEATURES_CSV_NAME = "telco_features.csv"
VALIDATION_REPORT_NAME = "data_validation.json"
BASELINE_REPORT_NAME = "baseline_metrics.json"
MODEL_DIR = MODELS_DIR / "churn_baseline"

SEED = 42

# Risk bands on the calibrated churn score. Chosen from the meaning of the scale, not tuned on outcomes:
# 0.25 is about the dataset base rate (0.265); 0.50 is "more likely than not". They are communication
# bands, not decision thresholds (threshold analysis lives in reports/baseline_metrics.json).
RISK_BAND_CUTOFFS = {"MEDIUM": 0.25, "HIGH": 0.50}

# ASSUMPTION (not estimated from data): number of future months of revenue treated as "at stake" in the CLV proxy.
# Changing it rescales every CLV proxy proportionally. There is no margin, cost-to-serve or discount rate in the data.
ASSUMED_VALUE_MONTHS = 12
