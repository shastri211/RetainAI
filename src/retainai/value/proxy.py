"""Revenue exposure proxies built only from quantities available in the dataset.

Nothing here is a true customer lifetime value. The dataset has no margin, cost-to-serve,
discount rate, churn horizon or future billing, and a survival-based CLV was rejected because a
single snapshot cannot validate it (DECISIONS.md D-18). Each column states its basis:

* ``monthly_revenue``          OBSERVED  - MonthlyCharges as reported (currency units as in the file).
* ``revenue_to_date``          OBSERVED  - TotalCharges as reported (0.0 for the 11 never-billed rows, KI-1).
* ``revenue_at_risk_monthly``  DERIVED   - risk_score x monthly_revenue. Risk-weighted monthly recurring
                                           revenue. It implies NO time horizon because the churn horizon is undefined.
* ``clv_proxy``                ASSUMED   - monthly_revenue x ``assumed_months`` (config). Revenue, not margin.
* ``clv_at_risk_proxy``        DERIVED + ASSUMED - risk_score x clv_proxy.
"""

from __future__ import annotations

import pandas as pd

from retainai.config import ASSUMED_VALUE_MONTHS
from retainai.data import schema
from retainai.scoring.risk import RiskRecord

BASIS: dict[str, str] = {
    "monthly_revenue": "OBSERVED: MonthlyCharges as reported",
    "revenue_to_date": "OBSERVED: TotalCharges as reported (0.0 where never billed)",
    "revenue_at_risk_monthly": "DERIVED: risk_score x monthly_revenue; risk-weighted recurring revenue, no time horizon implied",
    "clv_proxy": "ASSUMED: monthly_revenue x assumed_months; revenue not margin; not a survival-based or true CLV",
    "clv_at_risk_proxy": "DERIVED + ASSUMED: risk_score x clv_proxy",
}

BAND_ORDER = ("LOW", "MEDIUM", "HIGH")


def value_table(
    cleaned: pd.DataFrame,
    records: list[RiskRecord],
    assumed_months: float = ASSUMED_VALUE_MONTHS,
) -> pd.DataFrame:
    """One row per customer, aligned with ``records`` (ids must match row for row)."""
    if assumed_months <= 0:
        raise ValueError("assumed_months must be positive")
    ids = cleaned[schema.CLEAN_ID].astype(str).tolist()
    if ids != [r.customer_id for r in records]:
        raise ValueError("risk records are not aligned with the cleaned table (customer ids differ)")

    risk = pd.Series([r.risk_score for r in records], index=cleaned.index, dtype="float64")
    monthly = cleaned["monthly_charges"].astype("float64")
    table = pd.DataFrame(
        {
            schema.CLEAN_ID: ids,
            "risk_score": risk.to_numpy(),
            "risk_band": [r.risk_band for r in records],
            "monthly_revenue": monthly.to_numpy(),
            "revenue_to_date": cleaned["total_charges"].astype("float64").to_numpy(),
            "revenue_at_risk_monthly": (risk * monthly).round(4).to_numpy(),
            "clv_proxy": (monthly * assumed_months).round(4).to_numpy(),
            "clv_at_risk_proxy": (risk * monthly * assumed_months).round(4).to_numpy(),
        }
    )
    return table


def band_summary(table: pd.DataFrame, cleaned: pd.DataFrame | None = None) -> dict[str, object]:
    """Totals per risk band. If the cleaned table carries the churn label, also report the observed
    monthly revenue of customers who actually churned, as a reasonableness check on the risk weighting."""
    total_at_risk = float(table["revenue_at_risk_monthly"].sum())
    rows = []
    for band in BAND_ORDER:
        part = table[table["risk_band"] == band]
        at_risk = float(part["revenue_at_risk_monthly"].sum())
        rows.append(
            {
                "band": band,
                "customers": int(len(part)),
                "monthly_revenue": round(float(part["monthly_revenue"].sum()), 2),
                "revenue_at_risk_monthly": round(at_risk, 2),
                "share_of_revenue_at_risk": round(at_risk / total_at_risk, 4) if total_at_risk else None,
            }
        )
    summary: dict[str, object] = {
        "bands": rows,
        "total_monthly_revenue": round(float(table["monthly_revenue"].sum()), 2),
        "total_revenue_at_risk_monthly": round(total_at_risk, 2),
        "basis": BASIS,
    }
    if cleaned is not None and schema.TARGET in cleaned.columns:
        churned = cleaned[cleaned[schema.TARGET] == 1]["monthly_charges"].sum()
        summary["observed_monthly_revenue_of_churned_customers"] = round(float(churned), 2)
        summary["observed_note"] = (
            "Label-based, for comparison only. Equals the risk-weighted total only if the model is well calibrated, "
            "and scoring the training file is in-sample."
        )
    return summary


def top_revenue_at_risk(table: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Customers with the largest risk-weighted monthly revenue (ties broken by id for determinism)."""
    return table.sort_values(["revenue_at_risk_monthly", schema.CLEAN_ID], ascending=[False, True], kind="stable").head(n)
