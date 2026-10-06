"""Value / revenue-at-risk proxies: arithmetic, labelling, alignment, CLI."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd
import pytest
from conftest import synthetic_raw

from retainai.cli import main
from retainai.data import schema
from retainai.data.clean import clean_telco
from retainai.io import write_csv
from retainai.models.baseline import run_baseline
from retainai.models.registry import build_metadata, save_artifacts
from retainai.scoring.risk import RiskRecord, score_cleaned
from retainai.value.proxy import BASIS, band_summary, top_revenue_at_risk, value_table

FAST = dict(n_splits=3, n_repeats=1, candidate_names=("prior_baseline", "logistic_regression"), run_ablations=False)
STAMP = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
SMALL_RAW_IDS = [f"{i:04d}-AAAA{chr(64 + i)}" for i in range(1, 9)]  # ids of the small_raw fixture


def _records(scores, bands=None, ids=None):
    bands = bands or ["HIGH" if s >= 0.5 else "MEDIUM" if s >= 0.25 else "LOW" for s in scores]
    ids = ids or SMALL_RAW_IDS[: len(scores)]
    return [
        RiskRecord(customer_id=i, risk_score=s, risk_band=b, model_version="v", prediction_timestamp=STAMP)
        for i, s, b in zip(ids, scores, bands, strict=True)
    ]


@pytest.fixture
def cleaned(small_raw):
    return clean_telco(small_raw)  # 8 rows, includes a tenure-0 never-billed customer


def test_formulas_match_the_documented_definitions(cleaned):
    scores = [0.1, 0.9, 0.5, 0.0, 1.0, 0.25, 0.3, 0.6]
    table = value_table(cleaned, _records(scores), assumed_months=12)
    monthly = cleaned["monthly_charges"]
    assert table["monthly_revenue"].tolist() == monthly.tolist()
    assert table["revenue_to_date"].tolist() == cleaned["total_charges"].tolist()
    assert table["revenue_at_risk_monthly"].tolist() == pytest.approx([s * m for s, m in zip(scores, monthly, strict=True)], abs=1e-4)
    assert table["clv_proxy"].tolist() == pytest.approx((monthly * 12).tolist(), abs=1e-4)
    assert table["clv_at_risk_proxy"].tolist() == pytest.approx([s * m * 12 for s, m in zip(scores, monthly, strict=True)], abs=1e-4)


def test_assumed_months_only_rescales_the_clv_proxy(cleaned):
    records = _records([0.4] * len(cleaned))
    a = value_table(cleaned, records, assumed_months=6)
    b = value_table(cleaned, records, assumed_months=12)
    assert (b["clv_proxy"] == 2 * a["clv_proxy"]).all()
    pd.testing.assert_series_equal(a["revenue_at_risk_monthly"], b["revenue_at_risk_monthly"])  # no horizon in this one


@pytest.mark.parametrize("months", [0, -3])
def test_non_positive_assumed_months_are_rejected(cleaned, months):
    with pytest.raises(ValueError):
        value_table(cleaned, _records([0.4] * len(cleaned)), assumed_months=months)


def test_misaligned_records_are_rejected(cleaned):
    wrong = _records([0.4] * len(cleaned), ids=[f"X{i}" for i in range(len(cleaned))])
    with pytest.raises(ValueError, match="aligned"):
        value_table(cleaned, wrong)
    with pytest.raises(ValueError, match="aligned"):
        value_table(cleaned, _records([0.4] * (len(cleaned) - 1)))


def test_never_billed_customer_is_valued_from_monthly_charges_only(cleaned):
    table = value_table(cleaned, _records([0.5] * len(cleaned)))
    row = table[table["customer_id"] == "0007-AAAAG"].iloc[0]  # tenure 0, TotalCharges blank -> 0.0
    assert row["revenue_to_date"] == 0.0 and row["monthly_revenue"] == 50.0 and row["revenue_at_risk_monthly"] == 25.0


def test_every_value_column_declares_its_basis_and_none_claims_true_clv():
    assert {"monthly_revenue", "revenue_to_date", "revenue_at_risk_monthly", "clv_proxy", "clv_at_risk_proxy"} == set(BASIS)
    assert BASIS["clv_proxy"].startswith("ASSUMED") and "not a survival-based or true CLV" in BASIS["clv_proxy"]
    assert BASIS["monthly_revenue"].startswith("OBSERVED")
    assert "no time horizon" in BASIS["revenue_at_risk_monthly"]


def test_band_summary_totals_are_consistent(cleaned):
    table = value_table(cleaned, _records([0.1, 0.9, 0.5, 0.0, 1.0, 0.25, 0.3, 0.6]))
    summary = band_summary(table, cleaned)
    assert sum(b["customers"] for b in summary["bands"]) == len(table)
    assert sum(b["revenue_at_risk_monthly"] for b in summary["bands"]) == pytest.approx(summary["total_revenue_at_risk_monthly"], abs=0.02)
    assert sum(b["share_of_revenue_at_risk"] for b in summary["bands"]) == pytest.approx(1.0, abs=1e-3)
    churned = cleaned.loc[cleaned[schema.TARGET] == 1, "monthly_charges"].sum()
    assert summary["observed_monthly_revenue_of_churned_customers"] == round(float(churned), 2)
    assert "observed_monthly_revenue_of_churned_customers" not in band_summary(table, cleaned.drop(columns=[schema.TARGET]))


def test_top_revenue_at_risk_is_sorted_and_deterministic(cleaned):
    table = value_table(cleaned, _records([0.3, 0.9, 0.5, 0.2, 1.0, 0.25, 0.3, 0.6]))
    top = top_revenue_at_risk(table, 3)
    assert list(top["revenue_at_risk_monthly"]) == sorted(top["revenue_at_risk_monthly"], reverse=True)
    pd.testing.assert_frame_equal(top, top_revenue_at_risk(table.copy(), 3))


def test_value_cli_end_to_end(synthetic_cleaned, tmp_path, capsys):
    run = run_baseline(synthetic_cleaned, dataset_sha256="abc123", **FAST)
    meta = build_metadata(run, dataset_sha256="abc123", seed=42, trained_at=STAMP)
    save_artifacts(run.final_model, meta, tmp_path / "model")
    new = synthetic_raw(n=80, seed=7, n_duplicates=0)
    csv_in = tmp_path / "new.csv"
    write_csv(new, csv_in)
    out, summary_path = tmp_path / "v.csv", tmp_path / "s.json"
    argv = ["value", "--input", str(csv_in), "--model-dir", str(tmp_path / "model"), "--output", str(out), "--summary", str(summary_path)]
    assert main([*argv, "--assumed-months", "6"]) == 0
    table = pd.read_csv(out)
    assert len(table) == 80 and (table["clv_proxy"] == (table["monthly_revenue"] * 6).round(4)).all()
    summary = json.loads(summary_path.read_text())
    assert summary["assumed_months"] == 6 and summary["scoring_context"] == "out_of_sample"
    text = capsys.readouterr().out
    assert "proxies only" in text and "ASSUMED" in text


def test_value_cli_rejects_bad_input_and_scores_unlabeled_data(synthetic_cleaned, tmp_path):
    run = run_baseline(synthetic_cleaned, dataset_sha256="abc123", **FAST)
    meta = build_metadata(run, dataset_sha256="abc123", seed=42, trained_at=STAMP)
    save_artifacts(run.final_model, meta, tmp_path / "model")
    unlabeled = synthetic_raw(n=30, seed=8, n_duplicates=0).drop(columns=["Churn"])
    ok = tmp_path / "ok.csv"
    write_csv(unlabeled, ok)
    common = ["--model-dir", str(tmp_path / "model"), "--summary", str(tmp_path / "s.json")]
    assert main(["value", "--input", str(ok), "--output", str(tmp_path / "o.csv"), *common]) == 0
    assert "observed_monthly_revenue_of_churned_customers" not in json.loads((tmp_path / "s.json").read_text())
    unlabeled.loc[0, "MonthlyCharges"] = "-5"
    bad = tmp_path / "bad.csv"
    write_csv(unlabeled, bad)
    assert main(["value", "--input", str(bad), "--output", str(tmp_path / "o2.csv"), *common]) == 1
    assert not (tmp_path / "o2.csv").exists()


def test_scoring_and_value_agree_on_scores(synthetic_cleaned):
    run = run_baseline(synthetic_cleaned, dataset_sha256="abc123", **FAST)
    meta = build_metadata(run, dataset_sha256="abc123", seed=42, trained_at=STAMP)
    records = score_cleaned(synthetic_cleaned, run.final_model, meta, now=STAMP)
    table = value_table(synthetic_cleaned, records)
    assert table["risk_score"].tolist() == [r.risk_score for r in records]
