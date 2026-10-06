"""Deterministic customer risk output.

``risk_score`` is the trained model's predicted churn probability for the dataset's churn
definition (static snapshot, undefined horizon). It is never produced or altered by an LLM.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.pipeline import Pipeline

from retainai.config import RISK_BAND_CUTOFFS
from retainai.data import schema
from retainai.features.build import derive_features
from retainai.scoring.bands import RiskBand, assign_band

SCORE_DECIMALS = 6


class RiskRecord(BaseModel):
    """One customer's risk output. ``top_reasons`` stays empty until a model-grounded explainer fills it."""

    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(min_length=1)
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_band: RiskBand
    model_version: str = Field(min_length=1)
    prediction_timestamp: datetime
    top_reasons: list[dict] = Field(default_factory=list)
    in_sample: bool = Field(
        default=False,
        description="True when the scored customers were part of the model's training data (scores are then optimistic).",
    )


def predict_scores(model: Pipeline, cleaned: pd.DataFrame) -> pd.Series:
    """Predicted churn probability per row, using exactly the input columns the model was fitted on."""
    features = derive_features(cleaned)
    columns = list(model.named_steps["prep"].feature_names_in_)
    proba = model.predict_proba(features[columns])[:, 1]
    return pd.Series(proba, index=cleaned.index, name="risk_score").round(SCORE_DECIMALS)


def score_cleaned(
    cleaned: pd.DataFrame,
    model: Pipeline,
    metadata: dict,
    *,
    now: datetime | None = None,
    in_sample: bool = False,
    cutoffs: Mapping[str, float] = RISK_BAND_CUTOFFS,
) -> list[RiskRecord]:
    """Score every row of a cleaned table. Row order is preserved."""
    stamp = now or datetime.now(UTC)
    scores = predict_scores(model, cleaned)
    version = metadata["model_version"]
    return [
        RiskRecord(
            customer_id=str(customer_id),
            risk_score=float(score),
            risk_band=assign_band(float(score), cutoffs),
            model_version=version,
            prediction_timestamp=stamp,
            in_sample=in_sample,
        )
        for customer_id, score in zip(cleaned[schema.CLEAN_ID], scores, strict=True)
    ]


def records_to_frame(records: list[RiskRecord]) -> pd.DataFrame:
    """Flat table of records; ``top_reasons`` is serialised as a JSON string."""
    import json

    rows = []
    for record in records:
        row = record.model_dump(mode="json")
        row["top_reasons"] = json.dumps(row["top_reasons"], sort_keys=True)
        rows.append(row)
    return pd.DataFrame(rows)
