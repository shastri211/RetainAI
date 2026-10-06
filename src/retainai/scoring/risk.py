"""Deterministic customer risk output.

``risk_score`` is the trained model's predicted churn probability for the dataset's churn
definition (static snapshot, undefined horizon). It is never produced or altered by an LLM.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.pipeline import Pipeline

from retainai.config import RISK_BAND_CUTOFFS
from retainai.data import schema
from retainai.explain.linear import LinearExplainer, top_reasons
from retainai.features.build import derive_features
from retainai.scoring.bands import RiskBand, assign_band

SCORE_DECIMALS = 6


class Reason(BaseModel):
    """One model-grounded reason. It is a MODEL_SIGNAL and can never carry a causal claim."""

    model_config = ConfigDict(extra="forbid")

    feature: str
    value: str | int | float
    contribution_log_odds: float
    direction: Literal["increases_risk", "decreases_risk"]
    evidence_type: Literal["MODEL_SIGNAL"] = "MODEL_SIGNAL"
    causal_claim: Literal[False] = False
    statement: str


class RiskRecord(BaseModel):
    """One customer's risk output. ``top_reasons`` is empty unless a model-grounded explainer filled it."""

    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(min_length=1)
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_band: RiskBand
    model_version: str = Field(min_length=1)
    prediction_timestamp: datetime
    top_reasons: list[Reason] = Field(default_factory=list)
    in_sample: bool = Field(
        default=False,
        description="True when the scored customers were part of the model's training data (scores are then optimistic).",
    )


def _model_inputs(model: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    return features[list(model.named_steps["prep"].feature_names_in_)]


def predict_scores(model: Pipeline, cleaned: pd.DataFrame) -> pd.Series:
    """Predicted churn probability per row, using exactly the input columns the model was fitted on."""
    features = derive_features(cleaned)
    proba = model.predict_proba(_model_inputs(model, features))[:, 1]
    return pd.Series(proba, index=cleaned.index, name="risk_score").round(SCORE_DECIMALS)


def score_cleaned(
    cleaned: pd.DataFrame,
    model: Pipeline,
    metadata: dict,
    *,
    now: datetime | None = None,
    in_sample: bool = False,
    cutoffs: Mapping[str, float] = RISK_BAND_CUTOFFS,
    explainer: LinearExplainer | None = None,
    top_k: int = 3,
) -> list[RiskRecord]:
    """Score every row of a cleaned table. Row order is preserved.

    With an ``explainer``, each record gets up to ``top_k`` risk-increasing reasons computed from
    the model's own coefficients; without one, ``top_reasons`` is left empty (never invented).
    """
    stamp = now or datetime.now(UTC)
    features = derive_features(cleaned)
    inputs = _model_inputs(model, features)
    scores = pd.Series(model.predict_proba(inputs)[:, 1], index=cleaned.index).round(SCORE_DECIMALS)
    version = metadata["model_version"]
    reasons = (
        top_reasons(explainer, inputs, top_k=top_k)
        if explainer is not None
        else [[] for _ in range(len(cleaned))]
    )
    return [
        RiskRecord(
            customer_id=str(customer_id),
            risk_score=float(score),
            risk_band=assign_band(float(score), cutoffs),
            model_version=version,
            prediction_timestamp=stamp,
            top_reasons=customer_reasons,
            in_sample=in_sample,
        )
        for customer_id, score, customer_reasons in zip(
            cleaned[schema.CLEAN_ID], scores, reasons, strict=True
        )
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
