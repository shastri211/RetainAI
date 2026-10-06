"""Evaluation metrics, calibration and threshold analysis.

All functions are deterministic and return plain Python values (JSON-ready). Undefined ratios
(e.g. precision with no flagged customers) are ``None``, never silently 0.

Threshold selection note: there is no intervention cost, offer or capacity data in this dataset.
The cost-ratio analysis below therefore uses *assumed* relative costs purely as a sensitivity
analysis; it is not an estimate of any real business cost.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

DECIMALS = 6


def _r(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), DECIMALS)


def _ratio(num: float, den: float) -> float | None:
    return None if den == 0 else _r(num / den)


def ranking_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | None]:
    """Threshold-free metrics."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    return {
        "roc_auc": _r(roc_auc_score(y, p)),
        "pr_auc": _r(average_precision_score(y, p)),
        "brier": _r(brier_score_loss(y, p)),
        "log_loss": _r(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])),
    }


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, object]:
    """Confusion matrix and precision/recall/F1 when flagging ``p >= threshold``."""
    y = np.asarray(y).astype(bool)
    flagged = np.asarray(p, dtype=float) >= threshold
    tp = int((flagged & y).sum())
    fp = int((flagged & ~y).sum())
    fn = int((~flagged & y).sum())
    tn = int((~flagged & ~y).sum())
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = _r(2 * precision * recall / (precision + recall))
    return {
        "threshold": _r(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": _ratio(fp, fp + tn),
        "selection_rate": _r(flagged.mean()),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def threshold_sweep(y: np.ndarray, p: np.ndarray, thresholds: Sequence[float]) -> list[dict[str, object]]:
    return [threshold_metrics(y, p, t) for t in thresholds]


def best_f1_threshold(y: np.ndarray, p: np.ndarray, grid: np.ndarray | None = None) -> float:
    """Threshold on a fixed 0.01 grid that maximises F1 (ties -> lowest threshold)."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else grid
    best_t, best_f1 = float(grid[0]), -1.0
    for t in grid:
        f1 = threshold_metrics(y, p, float(t))["f1"]
        if f1 is not None and f1 > best_f1 + 1e-12:
            best_t, best_f1 = float(t), f1
    return round(best_t, 2)


def capacity_table(y: np.ndarray, p: np.ndarray, fractions: Sequence[float]) -> list[dict[str, object]]:
    """Precision/recall/lift when only the top ``fraction`` of customers by score can be contacted."""
    y = np.asarray(y).astype(bool)
    p = np.asarray(p, dtype=float)
    order = np.argsort(-p, kind="stable")
    base_rate = y.mean()
    rows = []
    for frac in fractions:
        k = max(1, int(round(frac * len(y))))
        top = order[:k]
        hits = int(y[top].sum())
        precision = hits / k
        rows.append(
            {
                "top_fraction": _r(frac),
                "customers_contacted": k,
                "precision": _r(precision),
                "recall": _ratio(hits, int(y.sum())),
                "lift_over_base_rate": _r(precision / base_rate) if base_rate > 0 else None,
            }
        )
    return rows


def cost_ratio_analysis(
    y: np.ndarray,
    p: np.ndarray,
    ratios: Sequence[float],
    grid: np.ndarray | None = None,
) -> list[dict[str, object]]:
    """ASSUMED-cost sensitivity: threshold minimising ``ratio * FN + 1 * FP``.

    ``ratio`` is how many times worse a missed churner is than an unnecessary contact. These
    ratios are assumptions, not data; the output shows how the threshold moves as they change.
    """
    grid = np.round(np.arange(0.02, 0.98, 0.01), 2) if grid is None else grid
    y_bool = np.asarray(y).astype(bool)
    rows = []
    for ratio in ratios:
        best_t, best_cost = float(grid[0]), np.inf
        for t in grid:
            flagged = np.asarray(p, dtype=float) >= t
            fn = int((~flagged & y_bool).sum())
            fp = int((flagged & ~y_bool).sum())
            cost = ratio * fn + fp
            if cost < best_cost - 1e-12:
                best_t, best_cost = float(t), cost
        metrics = threshold_metrics(y, p, best_t)
        rows.append(
            {
                "assumed_cost_ratio_fn_to_fp": ratio,
                "basis": "ASSUMED (sensitivity analysis, not an estimated cost)",
                "threshold": round(best_t, 2),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "selection_rate": metrics["selection_rate"],
            }
        )
    return rows


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict[str, object]:
    """Quantile-binned reliability table and expected calibration error (ECE)."""
    y = np.asarray(y).astype(float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    bins = np.array_split(order, n_bins)
    rows = []
    ece = 0.0
    for idx in bins:
        if len(idx) == 0:
            continue
        mean_pred = float(p[idx].mean())
        observed = float(y[idx].mean())
        ece += len(idx) / len(y) * abs(mean_pred - observed)
        rows.append({"n": int(len(idx)), "mean_predicted": _r(mean_pred), "observed_rate": _r(observed)})
    return {
        "bins": rows,
        "expected_calibration_error": _r(ece),
        "mean_predicted": _r(p.mean()),
        "observed_rate": _r(y.mean()),
    }


def band_table(
    y: np.ndarray,
    bands: Sequence[str],
    order: Sequence[str] = ("LOW", "MEDIUM", "HIGH"),
) -> list[dict[str, object]]:
    """Observed churn rate within each risk band."""
    y = np.asarray(y).astype(float)
    bands = np.asarray(list(bands))
    rows = []
    for name in order:
        mask = bands == name
        n = int(mask.sum())
        rows.append(
            {
                "band": name,
                "customers": n,
                "share_of_customers": _r(n / len(y)) if len(y) else None,
                "observed_churn_rate": _r(y[mask].mean()) if n else None,
                "churners": int(y[mask].sum()),
            }
        )
    return rows
