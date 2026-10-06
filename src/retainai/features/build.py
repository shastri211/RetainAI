"""Deterministic, stateless feature engineering plus the fit-time preprocessor.

Two stages, kept apart so training and inference share exactly the same code path:

1. ``derive_features``  - stateless row-wise transform (no statistics learned from data), so a
   row gives the same features alone or in a batch.
2. ``make_preprocessor`` - scikit-learn ``ColumnTransformer`` (scaling, one-hot). It *learns*
   statistics, so it is only ever fitted on training data and then stored inside the model
   pipeline.

Leakage rules:
* The identifier, the target and the audit columns are never model inputs.
* Protected attributes (``gender``, ``senior_citizen``) are excluded from the default model
  and retained only for the subgroup audit (see ``docs/PHASE_1_PIPELINE.md``).
* No feature is derived from the target.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from retainai.data import schema

FEATURE_PIPELINE_VERSION = "1"

ADDON_COLUMNS = tuple(schema.RAW_TO_CLEAN[c] for c in schema.INTERNET_ADDON_COLUMNS)

# Protected / sensitive attributes: audit-only in the default model.
PROTECTED_ATTRIBUTES = ("gender", "senior_citizen")
# Family-status fields can proxy for age or marital status; kept as features but audited.
SENSITIVE_ADJACENT = ("partner", "dependents")

_YES_NO_BINARY = (
    "partner",
    "dependents",
    "phone_service",
    "paperless_billing",
    "multiple_lines",
    *ADDON_COLUMNS,
)


@dataclass(frozen=True)
class FeatureSpec:
    """Which derived columns the model may use."""

    numeric: tuple[str, ...]
    binary: tuple[str, ...]
    categorical: tuple[str, ...]

    @property
    def all(self) -> tuple[str, ...]:
        return (*self.numeric, *self.binary, *self.categorical)

    def without(self, *columns: str) -> FeatureSpec:
        drop = set(columns)
        return FeatureSpec(
            tuple(c for c in self.numeric if c not in drop),
            tuple(c for c in self.binary if c not in drop),
            tuple(c for c in self.categorical if c not in drop),
        )

    def with_protected(self) -> FeatureSpec:
        """Ablation only: add the protected attributes."""
        return FeatureSpec(
            self.numeric,
            (*self.binary, "senior_citizen"),
            (*self.categorical, "gender"),
        )

    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "version": FEATURE_PIPELINE_VERSION,
                "numeric": self.numeric,
                "binary": self.binary,
                "categorical": self.categorical,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


DEFAULT_SPEC = FeatureSpec(
    numeric=("tenure_months", "monthly_charges", "total_charges", "num_addon_services"),
    binary=_YES_NO_BINARY,
    categorical=("internet_service", "contract", "payment_method"),
)

# Columns the derivation needs from the cleaned table.
_REQUIRED_CLEAN = (
    schema.CLEAN_ID,
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "internet_service",
    "contract",
    "payment_method",
    "gender",
    "senior_citizen",
    "duplicate_group_id",
    *_YES_NO_BINARY,
)


def derive_features(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Stateless derivation from the cleaned table. Keeps the id, target (if any) and group id.

    Yes/No fields become 0/1. ``"No phone service"`` and ``"No internet service"`` collapse to 0
    because that structural information is carried by ``phone_service`` / ``internet_service``.
    ``num_addon_services`` counts the six internet add-ons that are "Yes".
    """
    missing = [c for c in _REQUIRED_CLEAN if c not in cleaned.columns]
    if missing:
        raise KeyError(f"cleaned table is missing columns: {missing}")

    out = pd.DataFrame({schema.CLEAN_ID: cleaned[schema.CLEAN_ID].to_numpy()}, index=cleaned.index)
    out["tenure_months"] = cleaned["tenure_months"].astype("int64")
    out["monthly_charges"] = cleaned["monthly_charges"].astype("float64")
    out["total_charges"] = cleaned["total_charges"].astype("float64")

    for col in _YES_NO_BINARY:
        out[col] = (cleaned[col] == "Yes").astype("int8")
    out["num_addon_services"] = out[list(ADDON_COLUMNS)].sum(axis=1).astype("int64")

    for col in ("internet_service", "contract", "payment_method", "gender"):
        out[col] = cleaned[col].astype(object)
    out["senior_citizen"] = cleaned["senior_citizen"].astype("int8")

    if schema.TARGET in cleaned.columns:
        out[schema.TARGET] = cleaned[schema.TARGET].astype("int64")
    out["duplicate_group_id"] = cleaned["duplicate_group_id"].astype("int64")
    return out.reset_index(drop=True)


def model_ready_frame(features: pd.DataFrame, spec: FeatureSpec = DEFAULT_SPEC) -> pd.DataFrame:
    """The model-ready table: id, the spec's input columns, target (if any) and split-group id."""
    columns = [schema.CLEAN_ID, *spec.all]
    if schema.TARGET in features.columns:
        columns.append(schema.TARGET)
    columns.append("duplicate_group_id")
    return features.loc[:, columns].copy()


def make_preprocessor(spec: FeatureSpec = DEFAULT_SPEC) -> ColumnTransformer:
    """Unfitted preprocessor: scale numerics, pass binaries through, one-hot categoricals.

    Unseen categories at inference time encode to all zeros rather than raising.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(spec.numeric)),
            ("bin", "passthrough", list(spec.binary)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(spec.categorical),
            ),
        ],
        verbose_feature_names_out=True,
    )


def original_feature_of(transformed_name: str, spec: FeatureSpec) -> str:
    """Map a transformed column name (e.g. ``cat__contract_Two year``) to its source feature."""
    _, _, rest = transformed_name.partition("__")
    if transformed_name.startswith("cat__"):
        # longest match first so e.g. "internet_service" never collides with a shorter prefix
        for col in sorted(spec.categorical, key=len, reverse=True):
            if rest == col or rest.startswith(col + "_"):
                return col
        raise ValueError(f"cannot map {transformed_name!r} to a categorical feature")
    return rest
