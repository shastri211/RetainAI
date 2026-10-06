"""Deterministic cleaning: validated raw table -> cleaned table.

Treatment of each known issue (see ``validate.py`` for detection):

KI-1  Blank TotalCharges (tenure == 0)   -> set to 0.0 and flag ``total_charges_was_blank``.
                                            Rationale: a customer with zero months of tenure has
                                            nothing billed yet, so 0.0 is the arithmetic value.
                                            Rows are kept so that brand-new customers can be scored.
KI-2  Duplicate non-ID rows              -> rows are KEPT (distinct customerIDs; all sit at
                                            tenure == 1). Each gets ``duplicate_group_id`` so that
                                            cross-validation can keep a group on one side of a split.
KI-3  Fixed-term contract, short tenure  -> nothing changed (plausible); flagged for audit only via
                                            ``contract_term_exceeds_tenure``.
KI-4  TotalCharges != tenure*Monthly     -> nothing changed. TotalCharges is kept as reported.

The audit columns are never model features. Row order of the raw file is preserved.
"""

from __future__ import annotations

import pandas as pd

from retainai.data import schema
from retainai.data.validate import ValidationReport, validate_raw


class DataContractError(ValueError):
    """Raised when the raw data violates the dataset contract."""

    def __init__(self, report: ValidationReport):
        self.report = report
        summary = "; ".join(f"{c.check_id} (count={c.count})" for c in report.errors)
        super().__init__(f"Dataset contract violated: {summary}")


def clean_telco(df_raw: pd.DataFrame, report: ValidationReport | None = None) -> pd.DataFrame:
    """Return the cleaned table. Raises ``DataContractError`` if validation has errors."""
    report = report if report is not None else validate_raw(df_raw)
    if not report.passed:
        raise DataContractError(report)

    df = df_raw.astype("string").fillna("")
    out = pd.DataFrame(index=df.index)

    # Duplicate groups are defined on the raw non-ID columns, numbered by first appearance.
    non_id = [c for c in schema.RAW_COLUMNS if c != schema.RAW_ID]
    group_id = df.groupby(non_id, sort=False).ngroup()
    group_size = group_id.map(group_id.value_counts())

    tenure = pd.to_numeric(df["tenure"]).astype("int64")
    blank_total = df["TotalCharges"].str.strip() == ""
    total = pd.to_numeric(df["TotalCharges"].where(~blank_total, "0")).astype("float64")

    for raw_col in schema.RAW_COLUMNS:
        clean_col = schema.RAW_TO_CLEAN[raw_col]
        if raw_col == schema.RAW_ID:
            out[clean_col] = df[raw_col].astype(object)
        elif raw_col == "tenure":
            out[clean_col] = tenure
        elif raw_col == "MonthlyCharges":
            out[clean_col] = pd.to_numeric(df[raw_col]).astype("float64")
        elif raw_col == "TotalCharges":
            out[clean_col] = total
        elif raw_col == "SeniorCitizen":
            out[clean_col] = pd.to_numeric(df[raw_col]).astype("int64")
        elif raw_col == schema.RAW_TARGET:
            out[clean_col] = df[raw_col].map(schema.TARGET_ENCODING).astype("int64")
        else:
            out[clean_col] = df[raw_col].astype(object)

    term = df["Contract"].map(schema.CONTRACT_TERM_MONTHS).astype("int64")
    out["total_charges_was_blank"] = blank_total.to_numpy()
    out["duplicate_group_id"] = group_id.astype("int64").to_numpy()
    out["duplicate_group_size"] = group_size.astype("int64").to_numpy()
    out["contract_term_exceeds_tenure"] = ((term > 1) & (tenure < term)).to_numpy()

    return out.loc[:, list(schema.CLEAN_COLUMNS)].reset_index(drop=True)
