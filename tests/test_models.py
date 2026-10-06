"""Baseline models: metrics, protocol integrity, determinism, leakage guards, registry."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pytest
from conftest import synthetic_raw

from retainai.data import schema
from retainai.data.clean import clean_telco
from retainai.features.build import DEFAULT_SPEC, PROTECTED_ATTRIBUTES
from retainai.models import evaluate as ev
from retainai.models.baseline import make_folds, run_baseline
from retainai.models.candidates import ESTIMATOR_FACTORIES, SHIPPED_MODEL, build_pipeline
from retainai.models.registry import (
    build_metadata,
    load_artifacts,
    model_version,
    save_artifacts,
)

FAST = dict(n_splits=3, n_repeats=1, candidate_names=("prior_baseline", "logistic_regression"), run_ablations=False)


@pytest.fixture(scope="module")
def run(synthetic_cleaned):
    return run_baseline(synthetic_cleaned, dataset_sha256="abc123", **FAST)


# --- metrics -----------------------------------------------------------------------------------


def test_threshold_metrics_confusion_matrix_and_ratios():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.4, 0.6, 0.1, 0.8, 0.2])
    m = ev.threshold_metrics(y, p, 0.5)
    assert m["confusion_matrix"] == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}
    assert m["precision"] == pytest.approx(2 / 3, abs=1e-6)
    assert m["recall"] == pytest.approx(2 / 3, abs=1e-6)
    assert m["f1"] == pytest.approx(2 / 3, abs=1e-6)
    assert m["selection_rate"] == pytest.approx(0.5)


def test_undefined_precision_is_none_not_zero():
    m = ev.threshold_metrics(np.array([1, 0]), np.array([0.2, 0.1]), 0.9)
    assert m["precision"] is None and m["f1"] is None and m["recall"] == 0.0


def test_best_f1_threshold_picks_the_separating_value():
    y = np.array([0] * 50 + [1] * 50)
    p = np.array([0.1] * 50 + [0.7] * 50)
    assert 0.1 < ev.best_f1_threshold(y, p) <= 0.7


def test_capacity_table_top_fraction_precision():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    p = np.linspace(1, 0, 10)
    row = ev.capacity_table(y, p, [0.3])[0]
    assert row["customers_contacted"] == 3 and row["precision"] == 1.0 and row["recall"] == 1.0
    assert row["lift_over_base_rate"] == pytest.approx(1 / 0.3, abs=1e-3)


def test_calibration_table_counts_and_ece():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=1000)
    y = (rng.uniform(size=1000) < p).astype(int)  # perfectly calibrated by construction
    cal = ev.calibration_table(y, p, 10)
    assert sum(b["n"] for b in cal["bins"]) == 1000
    assert cal["expected_calibration_error"] < 0.08


def test_cost_ratio_sensitivity_lowers_threshold_as_misses_get_costlier():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=2000)
    y = (rng.uniform(size=2000) < p).astype(int)
    rows = ev.cost_ratio_analysis(y, p, [1, 3, 10])
    thresholds = [r["threshold"] for r in rows]
    assert thresholds == sorted(thresholds, reverse=True)
    assert all("ASSUMED" in r["basis"] for r in rows)


# --- candidates and folds -----------------------------------------------------------------------------


def test_every_candidate_builds_fits_and_predicts_probabilities(synthetic_cleaned):
    from retainai.features.build import derive_features

    feats = derive_features(synthetic_cleaned)
    X, y = feats[list(DEFAULT_SPEC.all)], feats[schema.TARGET]
    for name in ESTIMATOR_FACTORIES:
        proba = build_pipeline(name).fit(X, y).predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)


def test_unknown_candidate_name_is_rejected():
    with pytest.raises(KeyError):
        build_pipeline("deep_net")


def test_folds_cover_each_row_once_per_repeat_and_keep_groups_together(synthetic_cleaned):
    y = synthetic_cleaned[schema.TARGET].to_numpy()
    groups = synthetic_cleaned["duplicate_group_id"].to_numpy()
    assert (synthetic_cleaned["duplicate_group_size"] > 1).any()  # fixture contains duplicates
    folds = make_folds(y, groups, seed=7, n_splits=4, n_repeats=2)
    assert len(folds) == 8
    for repeat in range(2):
        val = np.concatenate([v for r, _, _, v in folds if r == repeat])
        assert sorted(val) == list(range(len(y)))  # each row validated exactly once
    for _, _, train_idx, val_idx in folds:
        assert not set(groups[train_idx]) & set(groups[val_idx])  # no group straddles the split


def test_folds_are_deterministic_and_seed_dependent(synthetic_cleaned):
    y = synthetic_cleaned[schema.TARGET].to_numpy()
    g = synthetic_cleaned["duplicate_group_id"].to_numpy()
    a = make_folds(y, g, 1, 3, 1)
    b = make_folds(y, g, 1, 3, 1)
    c = make_folds(y, g, 2, 3, 1)
    assert all(np.array_equal(x[3], z[3]) for x, z in zip(a, b, strict=True))
    assert not all(np.array_equal(x[3], z[3]) for x, z in zip(a, c, strict=True))


# --- end-to-end baseline run -----------------------------------------------------------------------------


def test_run_baseline_produces_expected_report_structure(run):
    r = run.report
    assert {"dataset", "feature_pipeline", "protocol", "candidates", "model_selection", "shipped_model_analysis", "limitations"} <= set(r)
    assert r["provenance_label"] == "PUBLIC"
    assert run.shipped_model == SHIPPED_MODEL == r["model_selection"]["shipped_model"]
    for name in ("prior_baseline", "logistic_regression"):
        assert {"cv", "holdout", "hyperparameters"} <= set(r["candidates"][name])
    assert "ASSUMED" in r["shipped_model_analysis"]["cost_ratio_sensitivity_dev_oof"][0]["basis"]


def test_model_beats_the_prior_baseline_on_a_learnable_signal(run):
    lr = run.report["candidates"]["logistic_regression"]["cv"]["roc_auc"]["mean"]
    prior = run.report["candidates"]["prior_baseline"]["cv"]["roc_auc"]["mean"]
    assert prior == pytest.approx(0.5, abs=1e-6)
    assert lr > 0.65


def test_holdout_is_disjoint_and_group_safe(run, synthetic_cleaned):
    assert not set(run.dev_index) & set(run.test_index)
    assert len(run.dev_index) + len(run.test_index) == len(synthetic_cleaned)
    groups = synthetic_cleaned["duplicate_group_id"].to_numpy()
    assert not set(groups[run.dev_index]) & set(groups[run.test_index])


def test_model_inputs_exclude_identifier_target_protected_and_audit_columns(run):
    names = set(run.final_model.named_steps["prep"].feature_names_in_)
    assert names == set(DEFAULT_SPEC.all)
    forbidden = {schema.CLEAN_ID, schema.TARGET, *PROTECTED_ATTRIBUTES, *schema.AUDIT_COLUMNS}
    assert not names & forbidden


def test_baseline_run_is_deterministic(synthetic_cleaned, run):
    again = run_baseline(synthetic_cleaned, dataset_sha256="abc123", **FAST)
    assert json.dumps(again.report, sort_keys=True, default=str) == json.dumps(run.report, sort_keys=True, default=str)
    assert np.array_equal(again.oof["p_oof"].to_numpy(), run.oof["p_oof"].to_numpy())


def test_shuffled_labels_give_chance_level_performance_so_no_leakage():
    """If any feature leaked the label, shuffling labels would not destroy performance."""
    cleaned = clean_telco(synthetic_raw(n=600, seed=3, shuffle_labels=True))
    shuffled = run_baseline(cleaned, dataset_sha256="x", **FAST)
    auc = shuffled.report["candidates"]["logistic_regression"]["cv"]["roc_auc"]["mean"]
    assert 0.40 < auc < 0.60


def test_shipped_model_must_be_a_candidate(synthetic_cleaned):
    with pytest.raises(ValueError):
        run_baseline(synthetic_cleaned, dataset_sha256="x", shipped_model="random_forest", **FAST)


# --- registry ---------------------------------------------------------------------------------------------


def test_model_version_is_deterministic_and_input_sensitive():
    kw = dict(model_name="logistic_regression", dataset_sha256="a", feature_fingerprint="f", hyperparameters={"C": 1.0}, seed=42)
    assert model_version(**kw) == model_version(**kw)
    assert model_version(**kw).startswith("churn-lr-")
    assert model_version(**{**kw, "dataset_sha256": "b"}) != model_version(**kw)
    assert model_version(**{**kw, "hyperparameters": {"C": 2.0}}) != model_version(**kw)


def test_metadata_records_required_provenance(run):
    stamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    meta = build_metadata(run, dataset_sha256="abc123", seed=42, trained_at=stamp)
    assert meta["training_timestamp_utc"] == "2026-01-02T03:04:05+00:00"
    assert meta["dataset"]["sha256_lf_normalised"] == "abc123"
    assert meta["dataset"]["provenance_label"] == "PUBLIC"
    assert meta["model_type"] == "logistic_regression" and meta["hyperparameters"]["C"] == 1.0
    assert meta["feature_pipeline"]["version"] == DEFAULT_SPEC.fingerprint()
    assert {"cv_dev", "holdout", "operating_thresholds"} <= set(meta["evaluation"])
    assert {"python", "scikit_learn", "pandas", "numpy"} <= set(meta["libraries"])
    assert meta["model_version"].startswith("churn-lr-")


def test_artifact_round_trip_gives_identical_predictions(run, tmp_path):
    meta = build_metadata(run, dataset_sha256="abc123", seed=42)
    save_artifacts(run.final_model, meta, tmp_path / "m")
    model, loaded_meta = load_artifacts(tmp_path / "m")
    X = run.frame[list(DEFAULT_SPEC.all)]
    assert np.array_equal(model.predict_proba(X), run.final_model.predict_proba(X))
    assert loaded_meta["model_version"] == meta["model_version"]


def test_loading_a_missing_model_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="retainai train"):
        load_artifacts(tmp_path / "nothing")


def test_holdout_fraction_is_about_twenty_percent(run):
    frac = len(run.test_index) / (len(run.dev_index) + len(run.test_index))
    assert 0.15 < frac < 0.25
