"""Validation of the raw Telco CSV against the dataset contract.

Each check returns a ``CheckResult``. ``triggered=True`` means the condition was observed.
Severity decides the consequence:

* ERROR   - violates the contract; the pipeline must stop.
* WARNING - a known or notable issue that is *handled by a documented rule* downstream.
* INFO    - descriptive, nothing is changed.

Known issues from the Phase 0 audit are reported under stable IDs (KI-1 ... KI-4). They are
detected here and treated explicitly in ``clean.py``; nothing is dropped or altered silently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from retainai.data import schema


class Severity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class CheckResult:
    check_id: str
    severity: Severity
    triggered: bool
    count: int
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    n_rows: int
    n_columns: int
    checks: list[CheckResult]
    dataset_sha256: str | None = None

    @property
    def errors(self) -> list[CheckResult]:
        return [c for c in self.checks if c.triggered and c.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.triggered and c.severity is Severity.WARNING]

    @property
    def passed(self) -> bool:
        return not self.errors

    def get(self, check_id: str) -> CheckResult:
        for check in self.checks:
            if check.check_id == check_id:
                return check
        raise KeyError(check_id)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checks"] = [
            {**asdict(c), "severity": c.severity.value} for c in self.checks
        ]
        payload["passed"] = self.passed
        payload["n_errors"] = len(self.errors)
        payload["n_warnings"] = len(self.warnings)
        payload["provenance_label"] = schema.PROVENANCE_LABEL
        return payload


def _sample_ids(df: pd.DataFrame, mask: pd.Series, limit: int = 5) -> list[str]:
    if schema.RAW_ID not in df.columns:
        return []
    return [str(v) for v in df.loc[mask, schema.RAW_ID].head(limit)]


def _blank(series: pd.Series) -> pd.Series:
    return series.str.strip() == ""


def _to_float(series: pd.Series) -> pd.Series:
    """Parse text to float64; anything unparseable becomes NaN (never raises)."""
    parsed = pd.to_numeric(series, errors="coerce")
    return pd.Series(parsed.to_numpy(dtype="float64", na_value=np.nan), index=series.index)


def _result(
    check_id: str,
    severity: Severity,
    count: int,
    message: str,
    details: dict[str, Any] | None = None,
) -> CheckResult:
    return CheckResult(check_id, severity, count > 0, int(count), message, details or {})


def validate_raw(df: pd.DataFrame, require_target: bool = True) -> ValidationReport:
    """Validate a raw Telco table (all fields as text, as produced by ``load_raw``).

    With ``require_target=False`` the ``Churn`` column may be absent (scoring new customers);
    if it is present it is still validated.
    """
    checks: list[CheckResult] = []
    add = checks.append

    missing = [c for c in schema.RAW_COLUMNS if c not in df.columns and (require_target or c != schema.RAW_TARGET)]
    extra = [c for c in df.columns if c not in schema.RAW_COLUMNS]
    add(_result("required_columns", Severity.ERROR, len(missing), "Required columns are present.", {"missing": missing}))
    add(_result("unexpected_columns", Severity.ERROR, len(extra), "No columns outside the contract.", {"unexpected": extra}))
    add(_result("non_empty", Severity.ERROR, int(len(df) == 0), "Table has at least one row."))
    if missing or df.empty:
        return ValidationReport(len(df), df.shape[1], checks)

    df = df.astype("string").fillna("")
    present = [c for c in schema.RAW_COLUMNS if c in df.columns]
    has_target = schema.RAW_TARGET in df.columns

    # --- identifiers ---------------------------------------------------------------------
    ids = df[schema.RAW_ID]
    add(_result("customer_id_missing", Severity.ERROR, int(_blank(ids).sum()), "Every row has a customerID."))
    dup_ids = ids.duplicated(keep=False) & ~_blank(ids)
    add(
        _result(
            "customer_id_unique",
            Severity.ERROR,
            int(dup_ids.sum()),
            "customerID values are unique.",
            {"sample_ids": _sample_ids(df, dup_ids)},
        )
    )
    bad_pattern = ~ids.str.match(schema.ID_PATTERN) & ~_blank(ids)
    add(
        _result(
            "customer_id_format",
            Severity.WARNING,
            int(bad_pattern.sum()),
            f"customerID matches {schema.ID_PATTERN} (informational; format is not relied on).",
            {"sample_ids": _sample_ids(df, bad_pattern)},
        )
    )

    # --- whitespace padding ----------------------------------------------------------------
    padded = {
        c: int((df[c] != df[c].str.strip()).sum())
        for c in present
        if c != "TotalCharges"  # blank TotalCharges is the known " " case (KI-1)
    }
    padded = {c: n for c, n in padded.items() if n}
    add(_result("string_padding", Severity.ERROR, sum(padded.values()), "No leading/trailing whitespace in text fields.", {"by_column": padded}))

    # --- missing values (blank cells) -----------------------------------------------------
    blank_by_col = {
        c: int(_blank(df[c]).sum()) for c in present if c != "TotalCharges"
    }
    blank_by_col = {c: n for c, n in blank_by_col.items() if n}
    add(_result("missing_values", Severity.ERROR, sum(blank_by_col.values()), "No blank cells outside TotalCharges.", {"by_column": blank_by_col}))

    # --- numeric parsing ------------------------------------------------------------------
    tenure_raw = df["tenure"].str.strip()
    tenure = _to_float(tenure_raw)
    tenure_bad = _blank(tenure_raw) | ~np.isfinite(tenure)
    tenure_nonint = np.isfinite(tenure) & (tenure % 1 != 0)
    add(_result("tenure_numeric", Severity.ERROR, int((tenure_bad | tenure_nonint).sum()), "tenure is a whole number.", {"sample_ids": _sample_ids(df, tenure_bad | tenure_nonint)}))
    tenure_neg = tenure < 0
    add(_result("tenure_non_negative", Severity.ERROR, int(tenure_neg.sum()), "tenure >= 0.", {"sample_ids": _sample_ids(df, tenure_neg)}))
    tenure_huge = tenure > schema.TENURE_MAX_PLAUSIBLE_MONTHS
    add(_result("tenure_plausible_range", Severity.WARNING, int(tenure_huge.sum()), f"tenure <= {schema.TENURE_MAX_PLAUSIBLE_MONTHS} months.", {"observed_max": None if tenure.dropna().empty else float(tenure.max())}))

    monthly_raw = df["MonthlyCharges"].str.strip()
    monthly = _to_float(monthly_raw)
    monthly_bad = _blank(monthly_raw) | ~np.isfinite(monthly)
    add(_result("monthly_charges_numeric", Severity.ERROR, int(monthly_bad.sum()), "MonthlyCharges is a finite number.", {"sample_ids": _sample_ids(df, monthly_bad)}))
    monthly_nonpos = monthly <= 0
    add(_result("monthly_charges_positive", Severity.ERROR, int(monthly_nonpos.sum()), "MonthlyCharges > 0.", {"sample_ids": _sample_ids(df, monthly_nonpos)}))

    total_raw = df["TotalCharges"].str.strip()
    total_blank = _blank(total_raw)
    total = _to_float(total_raw.where(~total_blank))
    total_bad = ~total_blank & ~np.isfinite(total)
    add(_result("total_charges_numeric", Severity.ERROR, int(total_bad.sum()), "Non-blank TotalCharges is a finite number.", {"sample_ids": _sample_ids(df, total_bad)}))
    total_neg = total < 0
    add(_result("total_charges_non_negative", Severity.ERROR, int(total_neg.sum()), "TotalCharges >= 0.", {"sample_ids": _sample_ids(df, total_neg)}))

    senior_bad = ~df["SeniorCitizen"].str.strip().isin(["0", "1"])
    add(_result("senior_citizen_binary", Severity.ERROR, int(senior_bad.sum()), "SeniorCitizen is 0 or 1.", {"sample_ids": _sample_ids(df, senior_bad)}))

    # --- KI-1: blank TotalCharges ----------------------------------------------------------
    blank_new = total_blank & (tenure == 0)
    blank_other = total_blank & ~(tenure == 0)
    add(
        _result(
            "total_charges_blank_with_tenure",
            Severity.ERROR,
            int(blank_other.sum()),
            "Blank TotalCharges is only acceptable when tenure == 0.",
            {"sample_ids": _sample_ids(df, blank_other)},
        )
    )
    add(
        _result(
            "known_issue_ki1_blank_total_charges",
            Severity.WARNING,
            int(blank_new.sum()),
            "KI-1: blank TotalCharges on tenure == 0 rows. Rule: treated as 0.0 (nothing billed yet) and flagged.",
            {"sample_ids": _sample_ids(df, blank_new), "churn_values": sorted(set(df.loc[blank_new, schema.RAW_TARGET])) if has_target else []},
        )
    )
    total_with_zero_tenure = total.notna() & (tenure == 0) & (total > 0)
    add(_result("total_charges_zero_tenure", Severity.ERROR, int(total_with_zero_tenure.sum()), "tenure == 0 implies no billed total.", {"sample_ids": _sample_ids(df, total_with_zero_tenure)}))
    total_zero_with_tenure = total.notna() & (tenure > 0) & (total == 0)
    add(_result("total_charges_positive_with_tenure", Severity.ERROR, int(total_zero_with_tenure.sum()), "tenure > 0 implies a positive billed total.", {"sample_ids": _sample_ids(df, total_zero_with_tenure)}))

    # --- categorical domains ---------------------------------------------------------------
    invalid_by_col: dict[str, list[str]] = {}
    for col, allowed in schema.RAW_CATEGORICAL_DOMAINS.items():
        bad = ~df[col].isin(allowed)
        if bad.any():
            invalid_by_col[col] = sorted(set(df.loc[bad, col]))[:10]
    add(
        _result(
            "categorical_domains",
            Severity.ERROR,
            len(invalid_by_col),
            "Categorical columns only contain contract values.",
            {"invalid_values_by_column": invalid_by_col},
        )
    )

    # --- target ----------------------------------------------------------------------------
    target = df[schema.RAW_TARGET] if has_target else pd.Series("", index=df.index, dtype="string")
    if has_target:
        target_bad = ~target.isin(schema.TARGET_ENCODING)
        add(_result("target_label_validity", Severity.ERROR, int(target_bad.sum()), "Churn is 'Yes' or 'No'.", {"invalid_values": sorted(set(target[target_bad]))[:10]}))
        present_labels = set(target[~target_bad])
        if require_target:
            add(_result("target_both_classes", Severity.ERROR, int(len(present_labels) < 2), "Both churn classes are present."))
        n_pos = int((target == schema.POSITIVE_LABEL).sum())
        add(
            CheckResult(
                "target_prevalence",
                Severity.INFO,
                False,
                n_pos,
                "Churn prevalence (informational).",
                {"positive": n_pos, "negative": int((target == schema.NEGATIVE_LABEL).sum()), "positive_rate": round(n_pos / len(df), 6)},
            )
        )

    # --- structural consistency --------------------------------------------------------------
    ml_inconsistent = (df["PhoneService"] == "No") != (df["MultipleLines"] == "No phone service")
    add(_result("structural_multiple_lines", Severity.ERROR, int(ml_inconsistent.sum()), "MultipleLines == 'No phone service' exactly when PhoneService == 'No'.", {"sample_ids": _sample_ids(df, ml_inconsistent)}))
    no_internet = df["InternetService"] == "No"
    addon_bad = pd.Series(False, index=df.index)
    for col in schema.INTERNET_ADDON_COLUMNS:
        addon_bad |= no_internet != (df[col] == "No internet service")
    add(_result("structural_internet_addons", Severity.ERROR, int(addon_bad.sum()), "Add-on columns == 'No internet service' exactly when InternetService == 'No'.", {"sample_ids": _sample_ids(df, addon_bad)}))

    # --- KI-4: TotalCharges vs tenure x MonthlyCharges ---------------------------------------
    expected = tenure * monthly
    usable = total.notna() & (tenure > 0) & monthly.notna() & (expected > 0)
    rel_dev = (total[usable] / expected[usable]) - 1.0
    if len(rel_dev):
        wide = rel_dev.abs() > schema.TOTAL_CHARGES_REL_DEV_WARN
        wide_mask = pd.Series(False, index=df.index)
        wide_mask.loc[rel_dev.index] = wide
        details = {
            "rows_compared": int(len(rel_dev)),
            "share_within_tight_band": round(float((rel_dev.abs() <= schema.TOTAL_CHARGES_REL_DEV_TIGHT).mean()), 4),
            "tight_band": schema.TOTAL_CHARGES_REL_DEV_TIGHT,
            "rel_dev_p05": round(float(rel_dev.quantile(0.05)), 4),
            "rel_dev_median": round(float(rel_dev.median()), 4),
            "rel_dev_p95": round(float(rel_dev.quantile(0.95)), 4),
            "rel_dev_min": round(float(rel_dev.min()), 4),
            "rel_dev_max": round(float(rel_dev.max()), 4),
        }
        add(
            CheckResult(
                "known_issue_ki4_total_vs_tenure_x_monthly",
                Severity.INFO,
                False,
                int(len(rel_dev)),
                "KI-4: TotalCharges is NOT assumed equal to tenure x MonthlyCharges; the deviation is reported, not corrected.",
                details,
            )
        )
        add(
            _result(
                "total_charges_gross_deviation",
                Severity.WARNING,
                int(wide.sum()),
                f"TotalCharges within +/-{schema.TOTAL_CHARGES_REL_DEV_WARN:.0%} of tenure x MonthlyCharges.",
                {"sample_ids": _sample_ids(df, wide_mask)},
            )
        )

    # --- KI-2: duplicate rows across all non-ID columns ----------------------------------------
    non_id = [c for c in present if c != schema.RAW_ID]
    dup_all = df.duplicated(subset=non_id, keep=False)
    extra_rows = int(df.duplicated(subset=non_id, keep="first").sum())
    add(
        _result(
            "known_issue_ki2_duplicate_rows",
            Severity.WARNING,
            extra_rows,
            "KI-2: rows identical on every non-ID column. Rule: kept (distinct customerIDs), grouped for split safety.",
            {
                "extra_rows": extra_rows,
                "groups": int(df[dup_all].groupby(non_id, sort=False).ngroups) if dup_all.any() else 0,
                "tenure_of_duplicated_rows": df.loc[dup_all, "tenure"].value_counts().to_dict(),
            },
        )
    )

    # --- KI-3: contract term vs tenure (informational) -------------------------------------------
    term = df["Contract"].map(schema.CONTRACT_TERM_MONTHS)
    within_first_term = (term > 1) & (tenure < term)
    by_contract = {
        c: {
            "rows": int((within_first_term & (df["Contract"] == c)).sum()),
            "churned": int((within_first_term & (df["Contract"] == c) & (target == schema.POSITIVE_LABEL)).sum()),
        }
        for c in ("One year", "Two year")
    }
    add(
        CheckResult(
            "known_issue_ki3_contract_within_first_term",
            Severity.INFO,
            False,
            int(within_first_term.sum()),
            "KI-3: fixed-term contracts with tenure shorter than the term. Plausible (still in first term); not an error. Time reference of Contract is undocumented.",
            by_contract,
        )
    )

    return ValidationReport(len(df), df.shape[1], checks)


def render_text(report: ValidationReport) -> str:
    """Compact human-readable summary."""
    lines = [
        f"Rows: {report.n_rows}  Columns: {report.n_columns}  Result: {'PASS' if report.passed else 'FAIL'}"
        f"  (errors={len(report.errors)}, warnings={len(report.warnings)})",
    ]
    if report.dataset_sha256:
        lines.append(f"Dataset sha256 (LF-normalised): {report.dataset_sha256}")
    for c in report.checks:
        if c.triggered or c.severity is Severity.INFO:
            lines.append(f"  [{c.severity.value:<7}] {c.check_id}: count={c.count} - {c.message}")
    return "\n".join(lines)
