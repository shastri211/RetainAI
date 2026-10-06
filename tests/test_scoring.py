"""Risk scoring: record schema, band assignment, determinism, CLI."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_raw
from pydantic import ValidationError

from retainai.cli import main
from retainai.data import schema
from retainai.data.clean import clean_telco
from retainai.features.build import derive_features
from retainai.io import write_csv
from retainai.models.baseline import run_baseline
from retainai.models.registry import build_metadata, save_artifacts
from retainai.scoring.bands import assign_band, assign_bands, risk_band_report
from retainai.scoring.risk import RiskRecord, predict_scores, records_to_frame, score_cleaned

FAST = dict(n_splits=3, n_repeats=1, candidate_names=("prior_baseline", "logistic_regression"), run_ablations=False)
STAMP = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


@pytest.fixture(scope="module")
def trained(synthetic_cleaned):
    run = run_baseline(synthetic_cleaned, dataset_sha256="abc123", **FAST)
    meta = build_metadata(run, dataset_sha256="abc123", seed=42, trained_at=STAMP)
    return run, meta


# --- bands -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "band"),
    [(0.0, "LOW"), (0.2499, "LOW"), (0.25, "MEDIUM"), (0.4999, "MEDIUM"), (0.5, "HIGH"), (1.0, "HIGH")],
)
def test_band_boundaries_are_lower_inclusive(score, band):
    assert assign_band(score) == band


@pytest.mark.parametrize("score", [-0.01, 1.01, float("nan")])
def test_out_of_range_scores_are_rejected(score):
    with pytest.raises(ValueError):
        assign_band(score)


@pytest.mark.parametrize("cutoffs", [{"MEDIUM": 0.6, "HIGH": 0.5}, {"MEDIUM": 0.0, "HIGH": 0.5}, {"MEDIUM": 0.3, "HIGH": 1.0}])
def test_invalid_cutoffs_are_rejected(cutoffs):
    with pytest.raises(ValueError):
        assign_band(0.4, cutoffs)


def test_custom_cutoffs_are_honoured():
    assert assign_band(0.4, {"MEDIUM": 0.1, "HIGH": 0.3}) == "HIGH"
    assert assign_bands(np.array([0.1, 0.3, 0.9]), {"MEDIUM": 0.2, "HIGH": 0.8}) == ["LOW", "MEDIUM", "HIGH"]


def test_band_report_counts_are_complete(trained):
    run, _ = trained
    rep = risk_band_report(run.oof, run.holdout_pred)
    assert sum(b["customers"] for b in rep["dev_oof"]) == len(run.oof)
    assert sum(b["customers"] for b in rep["holdout"]) == len(run.holdout_pred)
    rates = [b["observed_churn_rate"] for b in rep["dev_oof"] if b["observed_churn_rate"] is not None]
    assert rates == sorted(rates)  # higher band, higher observed churn on a learnable signal


# --- record schema ---------------------------------------------------------------------------------


def _record(**overrides):
    base = dict(customer_id="0001-ABCDE", risk_score=0.73, risk_band="HIGH", model_version="v", prediction_timestamp=STAMP)
    return RiskRecord(**{**base, **overrides})


def test_record_matches_the_documented_shape_and_round_trips():
    record = _record()
    dumped = record.model_dump(mode="json")
    assert set(dumped) == {"customer_id", "risk_score", "risk_band", "model_version", "prediction_timestamp", "top_reasons", "in_sample"}
    assert dumped["top_reasons"] == []  # never fabricated
    assert RiskRecord.model_validate_json(record.model_dump_json()) == record


@pytest.mark.parametrize(
    "bad",
    [dict(risk_score=1.2), dict(risk_score=-0.1), dict(risk_band="CRITICAL"), dict(customer_id=""), dict(model_version=""), dict(extra_field=1)],
)
def test_invalid_records_are_rejected(bad):
    with pytest.raises(ValidationError):
        _record(**bad)


# --- scoring -----------------------------------------------------------------------------------------


def test_scores_come_from_the_trained_model(trained, synthetic_cleaned):
    run, meta = trained
    records = score_cleaned(synthetic_cleaned, run.final_model, meta, now=STAMP)
    X = derive_features(synthetic_cleaned)[list(run.spec.all)]
    expected = run.final_model.predict_proba(X)[:, 1].round(6)
    assert [r.risk_score for r in records] == pytest.approx(list(expected), abs=1e-9)
    assert [r.customer_id for r in records] == synthetic_cleaned[schema.CLEAN_ID].tolist()
    assert all(r.model_version == meta["model_version"] and r.prediction_timestamp == STAMP for r in records)
    assert all(r.risk_band == assign_band(r.risk_score) for r in records)
    assert all(r.top_reasons == [] for r in records)


def test_scoring_is_deterministic_and_batch_independent(trained, synthetic_cleaned):
    run, meta = trained
    a = score_cleaned(synthetic_cleaned, run.final_model, meta, now=STAMP)
    b = score_cleaned(synthetic_cleaned.copy(), run.final_model, meta, now=STAMP)
    assert a == b
    single = score_cleaned(synthetic_cleaned.iloc[[5]], run.final_model, meta, now=STAMP)[0]
    assert single.risk_score == pytest.approx(a[5].risk_score, abs=1e-9)


def test_unlabeled_customers_get_the_same_scores(trained):
    run, meta = trained
    raw = synthetic_raw()
    labeled = predict_scores(run.final_model, clean_telco(raw))
    unlabeled = predict_scores(run.final_model, clean_telco(raw.drop(columns=["Churn"]), require_target=False))
    pd.testing.assert_series_equal(labeled, unlabeled)


def test_flat_table_serialises_reasons_as_json(trained, synthetic_cleaned):
    run, meta = trained
    frame = records_to_frame(score_cleaned(synthetic_cleaned.iloc[:3], run.final_model, meta, now=STAMP))
    assert list(frame["top_reasons"]) == ["[]"] * 3
    assert {"customer_id", "risk_score", "risk_band", "model_version", "prediction_timestamp"} <= set(frame.columns)


# --- CLI ---------------------------------------------------------------------------------------------


def test_score_cli_writes_csv_and_flags_in_sample_data(trained, tmp_path, capsys):
    run, meta = trained
    model_dir = tmp_path / "model"
    save_artifacts(run.final_model, meta, model_dir)

    new_customers = synthetic_raw(n=60, seed=99, n_duplicates=0).drop(columns=["Churn"])
    csv_in = tmp_path / "new.csv"
    write_csv(new_customers, csv_in)
    out_csv = tmp_path / "scores.csv"
    assert main(["score", "--input", str(csv_in), "--model-dir", str(model_dir), "--output", str(out_csv)]) == 0
    scores = pd.read_csv(out_csv)
    assert len(scores) == 60 and scores["risk_score"].between(0, 1).all()
    assert set(scores["risk_band"]) <= {"LOW", "MEDIUM", "HIGH"}
    assert not scores["in_sample"].any()
    assert "WARNING" not in capsys.readouterr().out

    out_json = tmp_path / "scores.json"
    assert main(["score", "--input", str(csv_in), "--model-dir", str(model_dir), "--output", str(out_json)]) == 0
    assert len(json.loads(out_json.read_text())) == 60


def test_score_cli_marks_training_file_as_in_sample(trained, tmp_path, capsys):
    run, meta = trained
    model_dir = tmp_path / "model"
    csv_in = tmp_path / "train.csv"
    write_csv(synthetic_raw(), csv_in)
    from retainai.data.load import dataset_sha256

    meta = {**meta, "dataset": {**meta["dataset"], "sha256_lf_normalised": dataset_sha256(csv_in)}}
    save_artifacts(run.final_model, meta, model_dir)
    out = tmp_path / "s.csv"
    assert main(["score", "--input", str(csv_in), "--model-dir", str(model_dir), "--output", str(out)]) == 0
    assert pd.read_csv(out)["in_sample"].all()
    assert "in-sample" in capsys.readouterr().out


def test_score_cli_rejects_contract_violations(trained, tmp_path):
    run, meta = trained
    model_dir = tmp_path / "model"
    save_artifacts(run.final_model, meta, model_dir)
    bad = synthetic_raw(n=20, seed=5, n_duplicates=0).drop(columns=["Churn"])
    bad.loc[0, "Contract"] = "Weekly"
    csv_in = tmp_path / "bad.csv"
    write_csv(bad, csv_in)
    assert main(["score", "--input", str(csv_in), "--model-dir", str(model_dir), "--output", str(tmp_path / "o.csv")]) == 1
    assert not (tmp_path / "o.csv").exists()
