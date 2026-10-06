"""Risk bands on the calibrated churn score."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np
import pandas as pd

from retainai.config import RISK_BAND_CUTOFFS
from retainai.models import evaluate as ev

RiskBand = Literal["LOW", "MEDIUM", "HIGH"]
BAND_ORDER: tuple[RiskBand, ...] = ("LOW", "MEDIUM", "HIGH")


def validate_cutoffs(cutoffs: Mapping[str, float]) -> None:
    medium, high = cutoffs["MEDIUM"], cutoffs["HIGH"]
    if not (0.0 < medium < high < 1.0):
        raise ValueError(f"cutoffs must satisfy 0 < MEDIUM < HIGH < 1, got {dict(cutoffs)}")


def assign_band(score: float, cutoffs: Mapping[str, float] = RISK_BAND_CUTOFFS) -> RiskBand:
    """LOW below MEDIUM; MEDIUM from MEDIUM (inclusive) up to HIGH; HIGH from HIGH (inclusive)."""
    validate_cutoffs(cutoffs)
    if not 0.0 <= score <= 1.0 or np.isnan(score):
        raise ValueError(f"score must be in [0, 1], got {score}")
    if score >= cutoffs["HIGH"]:
        return "HIGH"
    if score >= cutoffs["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


def assign_bands(scores: np.ndarray, cutoffs: Mapping[str, float] = RISK_BAND_CUTOFFS) -> list[RiskBand]:
    return [assign_band(float(s), cutoffs) for s in scores]


def risk_band_report(
    oof: pd.DataFrame,
    holdout: pd.DataFrame,
    cutoffs: Mapping[str, float] = RISK_BAND_CUTOFFS,
) -> dict[str, object]:
    """Observed churn rate per band on dev OOF predictions and on the holdout set."""
    validate_cutoffs(cutoffs)
    return {
        "cutoffs": dict(cutoffs),
        "basis": (
            "Cutoffs chosen from the meaning of a calibrated probability (0.25 ~ dataset base rate, 0.50 = more likely "
            "than not), not tuned on outcomes. Bands are communication labels, not decision thresholds."
        ),
        "dev_oof": ev.band_table(oof["y"].to_numpy(), assign_bands(oof["p_oof"].to_numpy(), cutoffs), BAND_ORDER),
        "holdout": ev.band_table(holdout["y"].to_numpy(), assign_bands(holdout["p"].to_numpy(), cutoffs), BAND_ORDER),
    }
