"""Exact, deterministic explanations for the shipped logistic-regression pipeline.

For a logistic regression on transformed features ``z``:  ``logit(x) = b + sum_j w_j * z_j``.
Relative to a reference profile ``z_ref`` (the mean of the transformed *training* features),

    logit(x) - logit(ref) = sum_j w_j * (z_j - z_ref_j)

Summing the terms of all encoded columns that come from one original feature gives that
feature's contribution in log-odds. Contributions add up *exactly* to the model's own
log-odds minus the reference log-odds, so every explanation is tied to what the model computed.

What these explanations ARE and ARE NOT
* MODEL SIGNAL: how the fitted model moves its estimate for this customer relative to the
  average training profile. This is what is reported.
* BUSINESS INTERPRETATION: what a pattern might mean commercially. It is a human judgement and is
  never generated here.
* CAUSAL EXPLANATION: not available. The dataset is observational with no interventions, so a
  contribution is a pattern in the data and never a proven cause or an effect of changing the feature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from retainai.features.build import FeatureSpec, original_feature_of

EVIDENCE_TYPE = "MODEL_SIGNAL"
CAVEAT = "Model signal (a pattern in this dataset), not a proven cause."
_DECIMALS = 6


def reference_means(model: Pipeline, X_train: pd.DataFrame) -> dict[str, float]:
    """Mean of each transformed feature over the training rows (stored with the model)."""
    transformed = model.named_steps["prep"].transform(X_train[list(model.named_steps["prep"].feature_names_in_)])
    names = model.named_steps["prep"].get_feature_names_out()
    return {str(n): float(v) for n, v in zip(names, transformed.mean(axis=0), strict=True)}


@dataclass(frozen=True)
class LinearExplainer:
    pipeline: Pipeline
    reference: np.ndarray  # mean transformed training row
    groups: tuple[str, ...]  # original feature for each transformed column
    features: tuple[str, ...]  # distinct original features, in model input order

    @classmethod
    def from_pipeline(cls, pipeline: Pipeline, reference: dict[str, float]) -> LinearExplainer:
        if not isinstance(pipeline.named_steps.get("clf"), LogisticRegression):
            raise NotImplementedError(
                "exact contributions are only implemented for LogisticRegression; "
                f"got {type(pipeline.named_steps.get('clf')).__name__}"
            )
        prep = pipeline.named_steps["prep"]
        inputs = tuple(prep.feature_names_in_)
        names = [str(n) for n in prep.get_feature_names_out()]
        missing = [n for n in names if n not in reference]
        if missing:
            raise ValueError(f"reference profile is missing transformed columns: {missing[:3]}")
        spec = FeatureSpec(
            tuple(prep.transformers_[0][2]), tuple(prep.transformers_[1][2]), tuple(prep.transformers_[2][2])
        )
        groups = tuple(original_feature_of(n, spec) for n in names)
        return cls(pipeline, np.array([reference[n] for n in names]), groups, inputs)

    @classmethod
    def from_artifacts(cls, pipeline: Pipeline, metadata: dict) -> LinearExplainer:
        """Rebuild the explainer from a saved model and its metadata (which stores the reference profile)."""
        if "explanation_reference" not in metadata:
            raise KeyError("metadata has no 'explanation_reference'; retrain with `retainai train`")
        return cls.from_pipeline(pipeline, metadata["explanation_reference"])

    @property
    def _coef(self) -> np.ndarray:
        return self.pipeline.named_steps["clf"].coef_[0]

    @property
    def _intercept(self) -> float:
        return float(self.pipeline.named_steps["clf"].intercept_[0])

    @property
    def reference_logit(self) -> float:
        """Log-odds the model assigns to the average training profile."""
        return self._intercept + float(self._coef @ self.reference)

    def _membership(self) -> np.ndarray:
        matrix = np.zeros((len(self.groups), len(self.features)))
        index = {f: i for i, f in enumerate(self.features)}
        for j, group in enumerate(self.groups):
            matrix[j, index[group]] = 1.0
        return matrix

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Log-odds contribution of each original feature, one row per customer."""
        transformed = self.pipeline.named_steps["prep"].transform(X[list(self.features)])
        per_column = self._coef * (transformed - self.reference)
        return pd.DataFrame(per_column @ self._membership(), columns=list(self.features), index=X.index)

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.decision_function(X[list(self.features)])

    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Mean absolute contribution per feature over ``X``, largest first."""
        contrib = self.contributions(X)
        table = pd.DataFrame(
            {
                "mean_abs_contribution": contrib.abs().mean(),
                "mean_signed_contribution": contrib.mean(),
            }
        )
        return table.sort_values(["mean_abs_contribution"], ascending=False, kind="stable")


def _display_value(feature: str, raw_value: object, binary_features: set[str]) -> str | float | int:
    if feature in binary_features:
        return "Yes" if int(raw_value) == 1 else "No"
    if isinstance(raw_value, (np.integer, int)):
        return int(raw_value)
    if isinstance(raw_value, (np.floating, float)):
        return round(float(raw_value), 2)
    return str(raw_value)


def _label(feature: str) -> str:
    return feature.replace("_", " ").capitalize()


def top_reasons(
    explainer: LinearExplainer,
    features: pd.DataFrame,
    *,
    top_k: int = 3,
    direction: str = "increases_risk",
) -> list[list[dict]]:
    """Per customer, up to ``top_k`` features that move the estimate in ``direction``, largest first.

    Only features with a non-zero contribution in that direction are returned, so a customer
    with nothing pushing risk up gets an empty list rather than filler.
    """
    if direction not in ("increases_risk", "decreases_risk"):
        raise ValueError(direction)
    sign = 1.0 if direction == "increases_risk" else -1.0
    contrib = explainer.contributions(features)
    binary = set(explainer.pipeline.named_steps["prep"].transformers_[1][2])
    out: list[list[dict]] = []
    for pos in range(len(features)):
        row = contrib.iloc[pos]
        ranked = sorted(
            ((f, float(row[f])) for f in explainer.features if sign * row[f] > 1e-12),
            key=lambda item: (-sign * item[1], item[0]),
        )[:top_k]
        reasons = []
        for feature, value in ranked:
            shown = _display_value(feature, features.iloc[pos][feature], binary)
            verb = "raises" if sign > 0 else "lowers"
            reasons.append(
                {
                    "feature": feature,
                    "value": shown,
                    "contribution_log_odds": round(value, _DECIMALS),
                    "direction": direction,
                    "evidence_type": EVIDENCE_TYPE,
                    "causal_claim": False,
                    "statement": (
                        f"{_label(feature)} = {shown} {verb} the model's risk estimate "
                        f"(log-odds {value:+.2f}) relative to the average customer profile. {CAVEAT}"
                    ),
                }
            )
        out.append(reasons)
    return out
