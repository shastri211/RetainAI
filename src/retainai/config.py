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

SEED = 42
