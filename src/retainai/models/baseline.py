"""Baseline training and evaluation protocol.

Protocol (all seeded, deterministic):

1. Hold out ~20% of customers as a final test set (stratified on churn, grouped on
   ``duplicate_group_id`` so identical-profile rows never straddle the split). Used once, for reporting.
2. On the remaining development rows run repeated stratified, grouped K-fold CV. Every candidate
   and every ablation sees *identical* folds, so differences are paired.
3. Threshold-free metrics are summarised as mean and SD across folds. Thresholds, calibration and
   capacity tables come from the out-of-fold (OOF) predictions of the shipped model, averaged
   over repeats (each prediction made by a model that never saw that customer).
4. The shipped model is the interpretable logistic regression. Tree models are benchmarks; if one
   beats it by more than one fold-level SD in PR-AUC the report flags it for an owner decision.
5. A final model is refitted on all rows. Reported metrics describe the dev-fitted models, not this refit.

Caveats that apply to every number produced here: single static snapshot, no temporal split is
possible, churn horizon undefined, PUBLIC data for a fictional company.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from retainai.config import SEED
from retainai.data import schema
from retainai.features.build import (
    DEFAULT_SPEC,
    PROTECTED_ATTRIBUTES,
    FeatureSpec,
    derive_features,
)
from retainai.models import evaluate as ev
from retainai.models.candidates import (
    COMPLEXITY_ORDER,
    ESTIMATOR_FACTORIES,
    SHIPPED_MODEL,
    build_pipeline,
)

REPORT_VERSION = 1
FOLD_METRICS = ("roc_auc", "pr_auc", "brier", "log_loss")
CALIBRATION_BINS_DEV = 10
CALIBRATION_BINS_HOLDOUT = 5
SWEEP_THRESHOLDS = tuple(round(t, 2) for t in np.arange(0.10, 0.91, 0.10))
CAPACITY_FRACTIONS = (0.05, 0.10, 0.20, 0.30, 0.50)
ASSUMED_COST_RATIOS = (1, 2, 3, 5, 10)

LIMITATIONS = (
    "Single static snapshot of a fictional company's customers (PUBLIC sample data); no temporal validation is possible.",
    "The churn horizon is undefined, so scores are dataset-specific churn probabilities, not 'probability of leaving within N days'.",
    "Metrics come from a small dataset (7,043 rows); expect wide uncertainty. Fold SDs understate true uncertainty because folds share training data.",
    "Model signal is associational. Nothing here estimates the effect of any intervention.",
    "Cost-ratio thresholds use ASSUMED relative costs; no real intervention cost exists in the data.",
)


@dataclass
class BaselineRun:
    report: dict
    final_model: Pipeline
    shipped_model: str
    spec: FeatureSpec
    frame: pd.DataFrame
    dev_index: np.ndarray
    test_index: np.ndarray
    oof: pd.DataFrame
    holdout_pred: pd.DataFrame
    operating_thresholds: dict[str, float]


def make_folds(
    y: np.ndarray, groups: np.ndarray, seed: int, n_splits: int, n_repeats: int
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """Repeated stratified grouped K-fold; returns (repeat, fold, train_idx, val_idx)."""
    folds = []
    dummy = np.zeros(len(y))
    for repeat in range(n_repeats):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
        for fold, (train_idx, val_idx) in enumerate(splitter.split(dummy, y, groups)):
            folds.append((repeat, fold, train_idx, val_idx))
    return folds


def _cv_predict(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[int, int, np.ndarray, np.ndarray]],
    n_repeats: int,
) -> tuple[list[dict[str, float | None]], np.ndarray]:
    oof = np.full((n_repeats, len(y)), np.nan)
    fold_metrics = []
    for repeat, _fold, train_idx, val_idx in folds:
        model = clone(pipeline).fit(X.iloc[train_idx], y[train_idx])
        proba = model.predict_proba(X.iloc[val_idx])[:, 1]
        oof[repeat, val_idx] = proba
        fold_metrics.append(ev.ranking_metrics(y[val_idx], proba))
    if np.isnan(oof).any():  # every row must be predicted exactly once per repeat
        raise RuntimeError("OOF predictions incomplete")
    return fold_metrics, oof


def _summarise(fold_metrics: list[dict[str, float | None]]) -> dict[str, dict[str, float | None]]:
    summary = {}
    for metric in FOLD_METRICS:
        values = [m[metric] for m in fold_metrics if m[metric] is not None]
        sd = statistics.stdev(values) if len(values) > 1 else None
        summary[metric] = {
            "mean": ev._r(statistics.fmean(values)),
            "sd_across_folds": ev._r(sd),
            "n_folds": len(values),
        }
    return summary


def _paired_difference(a: list[dict], b: list[dict], metric: str) -> dict[str, float | None]:
    diffs = [x[metric] - y[metric] for x, y in zip(a, b, strict=True)]
    return {
        "mean_difference": ev._r(statistics.fmean(diffs)),
        "sd_of_differences": ev._r(statistics.stdev(diffs) if len(diffs) > 1 else None),
    }


def run_baseline(
    cleaned: pd.DataFrame,
    *,
    dataset_sha256: str,
    seed: int = SEED,
    n_splits: int = 5,
    n_repeats: int = 3,
    spec: FeatureSpec = DEFAULT_SPEC,
    candidate_names: tuple[str, ...] = tuple(ESTIMATOR_FACTORIES),
    run_ablations: bool = True,
    shipped_model: str = SHIPPED_MODEL,
) -> BaselineRun:
    if shipped_model not in candidate_names:
        raise ValueError("the shipped model must be among the candidates")

    features = derive_features(cleaned)
    frame = features.loc[:, [schema.CLEAN_ID, *spec.all, schema.TARGET, "duplicate_group_id"]]
    X_all = features  # superset of columns, used so ablations can add/remove inputs
    y_all = features[schema.TARGET].to_numpy()
    groups_all = features["duplicate_group_id"].to_numpy()

    # 1. final holdout --------------------------------------------------------------------
    holdout_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 1000)
    dev_idx, test_idx = next(iter(holdout_splitter.split(np.zeros(len(y_all)), y_all, groups_all)))
    dev = X_all.iloc[dev_idx].reset_index(drop=True)
    test = X_all.iloc[test_idx].reset_index(drop=True)
    y_dev, y_test = y_all[dev_idx], y_all[test_idx]
    X_dev, X_test = dev[list(spec.all)], test[list(spec.all)]

    # 2. repeated grouped CV on the dev rows ---------------------------------------------------
    folds = make_folds(y_dev, dev["duplicate_group_id"].to_numpy(), seed, n_splits, n_repeats)
    candidates: dict[str, dict] = {}
    oof_by_model: dict[str, np.ndarray] = {}
    fold_metrics_by_model: dict[str, list[dict]] = {}
    for name in candidate_names:
        pipe = build_pipeline(name, spec)
        fold_metrics, oof = _cv_predict(pipe, X_dev, y_dev, folds, n_repeats)
        oof_mean = oof.mean(axis=0)
        oof_by_model[name] = oof_mean
        fold_metrics_by_model[name] = fold_metrics
        best_t = ev.best_f1_threshold(y_dev, oof_mean)
        candidates[name] = {
            "hyperparameters": {k: v for k, v in sorted(ESTIMATOR_FACTORIES[name]().get_params().items())},
            "cv": _summarise(fold_metrics),
            "oof_mean_prediction": {
                "ranking": ev.ranking_metrics(y_dev, oof_mean),
                "at_threshold_0.5": ev.threshold_metrics(y_dev, oof_mean, 0.5),
                "at_best_f1_threshold": ev.threshold_metrics(y_dev, oof_mean, best_t),
            },
        }

    # 3. selection verdict ----------------------------------------------------------------
    benchmarks = [n for n in candidate_names if n != "prior_baseline"]
    pr_mean = {n: candidates[n]["cv"]["pr_auc"]["mean"] for n in benchmarks}
    best_name = max(benchmarks, key=lambda n: (pr_mean[n], -COMPLEXITY_ORDER.index(n)))
    shipped_pr = candidates[shipped_model]["cv"]["pr_auc"]
    best_sd = candidates[best_name]["cv"]["pr_auc"]["sd_across_folds"] or 0.0
    gap = pr_mean[best_name] - shipped_pr["mean"]
    exceeds = best_name != shipped_model and gap > best_sd
    model_selection = {
        "shipped_model": shipped_model,
        "best_by_cv_pr_auc": best_name,
        "pr_auc_gap_best_minus_shipped": ev._r(gap),
        "best_model_pr_auc_sd_across_folds": ev._r(best_sd),
        "benchmark_exceeds_shipped_by_more_than_1sd": bool(exceeds),
        "rule": (
            "Ship the interpretable model unless a benchmark beats it by more than one fold-level SD "
            "in CV PR-AUC; in that case the report flags it for an owner decision. Nothing is switched silently."
        ),
    }

    # 4. shipped-model analysis on dev OOF ----------------------------------------------------
    oof_shipped = oof_by_model[shipped_model]
    best_f1_t = ev.best_f1_threshold(y_dev, oof_shipped)
    operating = {"best_f1_threshold": best_f1_t, "default_threshold": 0.5}

    # 5. holdout ---------------------------------------------------------------------------
    holdout: dict[str, dict] = {}
    shipped_test_proba = None
    for name in candidate_names:
        model = build_pipeline(name, spec).fit(X_dev, y_dev)
        proba = model.predict_proba(X_test)[:, 1]
        entry = {
            "ranking": ev.ranking_metrics(y_test, proba),
            "at_threshold_0.5": ev.threshold_metrics(y_test, proba, 0.5),
            "note": "reported for transparency; not used for selection",
        }
        if name == shipped_model:
            shipped_test_proba = proba
            entry["at_dev_best_f1_threshold"] = ev.threshold_metrics(y_test, proba, best_f1_t)
            entry["calibration"] = ev.calibration_table(y_test, proba, CALIBRATION_BINS_HOLDOUT)
            entry["capacity"] = ev.capacity_table(y_test, proba, CAPACITY_FRACTIONS)
            entry["note"] = "primary final-test result for the shipped model; thresholds were fixed using dev OOF only"
        holdout[name] = entry

    shipped_analysis = {
        "model": shipped_model,
        "calibration_dev_oof": ev.calibration_table(y_dev, oof_shipped, CALIBRATION_BINS_DEV),
        "threshold_sweep_dev_oof": ev.threshold_sweep(y_dev, oof_shipped, SWEEP_THRESHOLDS),
        "best_f1_threshold_dev_oof": best_f1_t,
        "capacity_dev_oof": ev.capacity_table(y_dev, oof_shipped, CAPACITY_FRACTIONS),
        "cost_ratio_sensitivity_dev_oof": ev.cost_ratio_analysis(y_dev, oof_shipped, ASSUMED_COST_RATIOS),
        "holdout": holdout[shipped_model],
    }

    # 6. ablations on the shipped model (identical folds) -------------------------------------------
    ablations: dict[str, dict] = {}
    if run_ablations:
        variants = {
            "default": spec,
            "with_protected_attributes": spec.with_protected(),
            "without_contract": spec.without("contract"),
            "without_total_charges": spec.without("total_charges"),
            "without_contract_and_total_charges": spec.without("contract", "total_charges"),
        }
        reference = fold_metrics_by_model[shipped_model]
        for label, variant in variants.items():
            if label == "default":
                metrics = reference
            else:
                metrics, _ = _cv_predict(build_pipeline(shipped_model, variant), dev[list(variant.all)], y_dev, folds, n_repeats)
            ablations[label] = {
                "n_features": len(variant.all),
                "cv": _summarise(metrics),
                "paired_difference_vs_default": {
                    m: _paired_difference(metrics, reference, m) for m in ("roc_auc", "pr_auc")
                },
            }

    # 7. final refit on all rows -------------------------------------------------------------
    final_model = build_pipeline(shipped_model, spec).fit(features[list(spec.all)], y_all)

    n_pos = int(y_all.sum())
    report = {
        "report_version": REPORT_VERSION,
        "provenance_label": schema.PROVENANCE_LABEL,
        "dataset": {
            "sha256_lf_normalised": dataset_sha256,
            "n_rows": int(len(y_all)),
            "n_positive": n_pos,
            "prevalence": ev._r(n_pos / len(y_all)),
            "target": f"{schema.TARGET} ({schema.POSITIVE_LABEL}=1); horizon undefined",
        },
        "feature_pipeline": {
            "version": spec.fingerprint(),
            "numeric": list(spec.numeric),
            "binary": list(spec.binary),
            "categorical": list(spec.categorical),
            "protected_attributes_excluded_from_model": list(PROTECTED_ATTRIBUTES),
        },
        "protocol": {
            "seed": seed,
            "holdout": {
                "method": "StratifiedGroupKFold(5), first fold as test; groups=duplicate_group_id",
                "n_dev": int(len(dev_idx)),
                "n_test": int(len(test_idx)),
                "test_prevalence": ev._r(y_test.mean()),
            },
            "cv": {
                "method": "repeated StratifiedGroupKFold on dev rows",
                "n_splits": n_splits,
                "n_repeats": n_repeats,
                "folds_total": len(folds),
            },
            "final_model": "refitted on all rows; reported metrics describe dev-fitted models",
        },
        "candidates": {
            name: {**candidates[name], "holdout": holdout[name]} for name in candidate_names
        },
        "model_selection": model_selection,
        "shipped_model_analysis": shipped_analysis,
        "ablations": ablations,
        "limitations": list(LIMITATIONS),
    }

    oof_frame = pd.DataFrame(
        {
            schema.CLEAN_ID: dev[schema.CLEAN_ID].to_numpy(),
            "y": y_dev,
            "p_oof": oof_shipped,
        }
    )
    holdout_frame = pd.DataFrame(
        {
            schema.CLEAN_ID: test[schema.CLEAN_ID].to_numpy(),
            "y": y_test,
            "p": shipped_test_proba,
        }
    )
    return BaselineRun(
        report=report,
        final_model=final_model,
        shipped_model=shipped_model,
        spec=spec,
        frame=frame,
        dev_index=np.asarray(dev_idx),
        test_index=np.asarray(test_idx),
        oof=oof_frame,
        holdout_pred=holdout_frame,
        operating_thresholds=operating,
    )
