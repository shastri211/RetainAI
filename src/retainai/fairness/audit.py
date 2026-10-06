"""Subgroup performance audit.

This is an audit of how the churn-risk model *behaves* across groups. It is NOT a fairness
certificate and returns no pass/fail verdict:

* Differences in selection rate are expected whenever observed churn rates differ between groups;
  they are reported for context, not as proof of bias or of its absence.
* Error-rate gaps (TPR/FPR) and calibration gaps are descriptive. Whether a gap matters depends on how
  the score will be used, which is a policy decision that has not been made.
* Intervals are bootstrap percentile intervals and ignore model-fitting uncertainty; small groups are
  flagged instead of being given falsely precise numbers.
* Only attributes present in the data can be audited; absence of a gap here says nothing about
  attributes we cannot see, nor about intersections of groups.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from retainai.data import schema
from retainai.features.build import PROTECTED_ATTRIBUTES, SENSITIVE_ADJACENT, FeatureSpec

AUDIT_ATTRIBUTES = ("gender", "senior_citizen", "partner", "dependents")
MIN_GROUP_SIZE = 30
MIN_CLASS_COUNT = 5
DECIMALS = 6

CAVEATS = (
    "This audit describes model behaviour on PUBLIC sample data for a fictional company. It does not establish fairness or its absence.",
    "Selection-rate differences are expected when observed churn rates differ; they are context, not proof of bias.",
    "Error-rate and calibration gaps are descriptive; their importance depends on how scores will be used, which is undecided.",
    "Intervals are bootstrap percentile intervals over customers and exclude model-fitting uncertainty.",
    "Only the listed attributes can be audited; other attributes and intersections of groups are not covered.",
    "Excluding an attribute from the model does not remove proxies for it (e.g. tenure, partner, dependents may correlate with age).",
)


def _r(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), DECIMALS)


def _safe_ratio(num: float, den: float) -> float:
    return num / den if den > 0 else np.nan


def sample_metrics(y: np.ndarray, p: np.ndarray, thresholds: Mapping[str, float]) -> dict[str, float]:
    """Metrics for one sample of customers; undefined values are NaN."""
    y_bool = y.astype(bool)
    out: dict[str, float] = {
        "observed_churn_rate": float(y.mean()),
        "mean_predicted_risk": float(p.mean()),
        "calibration_gap": float(p.mean() - y.mean()),  # mean predicted minus observed
    }
    both = 0 < y.sum() < len(y)
    out["roc_auc"] = float(roc_auc_score(y, p)) if both else np.nan
    out["pr_auc"] = float(average_precision_score(y, p)) if both else np.nan
    for name, t in thresholds.items():
        flagged = p >= t
        tp = float((flagged & y_bool).sum())
        fp = float((flagged & ~y_bool).sum())
        out[f"selection_rate@{name}"] = float(flagged.mean())
        out[f"tpr@{name}"] = _safe_ratio(tp, float(y_bool.sum()))
        out[f"fpr@{name}"] = _safe_ratio(fp, float((~y_bool).sum()))
        out[f"precision@{name}"] = _safe_ratio(tp, tp + fp)
    return out


def _bootstrap(
    y: np.ndarray, p: np.ndarray, thresholds: Mapping[str, float], n_boot: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    n = len(y)
    samples: dict[str, list[float]] = {}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for key, value in sample_metrics(y[idx], p[idx], thresholds).items():
            samples.setdefault(key, []).append(value)
    return {k: np.asarray(v) for k, v in samples.items()}


def _interval(values: np.ndarray) -> list[float | None]:
    finite = values[np.isfinite(values)]
    if len(finite) < max(10, int(0.5 * len(values))):
        return [None, None]
    return [_r(np.percentile(finite, 2.5)), _r(np.percentile(finite, 97.5))]


def _enough_data(y: np.ndarray) -> bool:
    return len(y) >= MIN_GROUP_SIZE and y.sum() >= MIN_CLASS_COUNT and (len(y) - y.sum()) >= MIN_CLASS_COUNT


def audit_attribute(
    frame: pd.DataFrame,
    attribute: str,
    thresholds: Mapping[str, float],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    """Per-group metrics with bootstrap intervals, plus pairwise differences between groups."""
    rng = np.random.default_rng(seed)
    groups: dict[str, dict] = {}
    boots: dict[str, dict[str, np.ndarray]] = {}
    for value in sorted(frame[attribute].astype(str).unique()):
        part = frame[frame[attribute].astype(str) == value]
        y, p = part["y"].to_numpy(), part["p"].to_numpy(dtype=float)
        entry: dict[str, object] = {"n": int(len(part)), "n_churned": int(y.sum())}
        if not _enough_data(y):
            entry["status"] = f"insufficient_data (needs >= {MIN_GROUP_SIZE} customers and >= {MIN_CLASS_COUNT} of each class)"
            groups[value] = entry
            continue
        entry["status"] = "ok"
        point = sample_metrics(y, p, thresholds)
        samples = _bootstrap(y, p, thresholds, n_boot, rng)
        boots[value] = samples
        entry["metrics"] = {k: {"estimate": _r(point[k]), "ci95": _interval(samples[k])} for k in point}
        groups[value] = entry

    disparities: dict[str, dict] = {}
    names = [g for g in groups if g in boots]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            diff: dict[str, dict] = {}
            for key in boots[a]:
                est = groups[b]["metrics"][key]["estimate"]
                ref = groups[a]["metrics"][key]["estimate"]
                if est is None or ref is None:
                    continue
                delta = boots[b][key] - boots[a][key]
                ci = _interval(delta)
                diff[key] = {
                    "estimate": _r(est - ref),
                    "ci95": ci,
                    "ci_excludes_zero": bool(ci[0] is not None and (ci[0] > 0 or ci[1] < 0)),
                }
            disparities[f"{b}_minus_{a}"] = diff
    return {"groups": groups, "differences": disparities}


def run_fairness_audit(
    oof: pd.DataFrame,
    holdout: pd.DataFrame,
    cleaned: pd.DataFrame,
    spec: FeatureSpec,
    operating_thresholds: Mapping[str, float],
    *,
    n_boot: int = 500,
    seed: int = 42,
) -> dict[str, object]:
    """Audit dev out-of-fold predictions (primary, larger) and the final holdout (secondary)."""
    missing = [a for a in AUDIT_ATTRIBUTES if a not in cleaned.columns]
    if missing:
        raise KeyError(f"cleaned table is missing audit attributes: {missing}")
    thresholds = {
        "best_f1": float(operating_thresholds["best_f1_threshold"]),
        "high_band": float(operating_thresholds.get("high_band", 0.5)),
    }
    attrs = cleaned.set_index(schema.CLEAN_ID)[list(AUDIT_ATTRIBUTES)]

    def prepare(pred: pd.DataFrame, score_col: str) -> pd.DataFrame:
        joined = pred.rename(columns={score_col: "p"}).join(attrs, on=schema.CLEAN_ID, how="left")
        if joined[list(AUDIT_ATTRIBUTES)].isna().any().any():
            raise ValueError("some scored customers have no audit attributes (customer ids do not match)")
        return joined

    sets = {
        "dev_oof": prepare(oof, "p_oof"),
        "holdout": prepare(holdout, "p"),
    }
    used = set(spec.all)
    role = {}
    for attr in AUDIT_ATTRIBUTES:
        if attr in PROTECTED_ATTRIBUTES:
            role[attr] = "protected: audit only" if attr not in used else "protected: USED in model (ablation)"
        elif attr in SENSITIVE_ADJACENT:
            role[attr] = "sensitive-adjacent: used in model and audited"
        else:
            role[attr] = "audited"

    result: dict[str, object] = {
        "method": f"bootstrap percentile 95% intervals over customers, {n_boot} resamples, seed {seed}",
        "thresholds": thresholds,
        "caveats": list(CAVEATS),
        "attribute_roles": {
            attr: {"used_in_model": attr in used, "role": role[attr]} for attr in AUDIT_ATTRIBUTES
        },
        "evaluation_sets": {},
    }
    for set_name, frame in sets.items():
        result["evaluation_sets"][set_name] = {
            "n_customers": int(len(frame)),
            "attributes": {
                attr: audit_attribute(frame, attr, thresholds, n_boot=n_boot, seed=seed + k)
                for k, attr in enumerate(AUDIT_ATTRIBUTES)
            },
        }
    return result
