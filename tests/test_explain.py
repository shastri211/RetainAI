"""Model-grounded explanations: exactness, faithfulness to the model, and honesty about causality."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.base import clone

from retainai.data import schema
from retainai.explain.linear import LinearExplainer, reference_means, top_reasons
from retainai.features.build import PROTECTED_ATTRIBUTES, derive_features
from retainai.models.baseline import run_baseline
from retainai.models.candidates import build_pipeline
from retainai.models.registry import build_metadata
from retainai.scoring.risk import Reason, RiskRecord, score_cleaned

FAST = dict(n_splits=3, n_repeats=1, candidate_names=("prior_baseline", "logistic_regression"), run_ablations=False)
STAMP = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


@pytest.fixture(scope="module")
def setup(synthetic_cleaned):
    run = run_baseline(synthetic_cleaned, dataset_sha256="abc123", **FAST)
    meta = build_metadata(run, dataset_sha256="abc123", seed=42, trained_at=STAMP)
    inputs = derive_features(synthetic_cleaned)[list(run.spec.all)]
    explainer = LinearExplainer.from_artifacts(run.final_model, meta)
    return run, meta, inputs, explainer


def test_contributions_add_up_exactly_to_the_models_own_log_odds(setup):
    _, _, inputs, explainer = setup
    contrib = explainer.contributions(inputs)
    reconstructed = contrib.sum(axis=1).to_numpy() + explainer.reference_logit
    assert np.allclose(reconstructed, explainer.decision_function(inputs), atol=1e-9)


def test_every_model_input_gets_a_contribution_column(setup):
    run, _, inputs, explainer = setup
    assert set(explainer.contributions(inputs).columns) == set(run.spec.all)
    assert set(explainer.features) == set(run.spec.all)


def test_reasons_are_tied_to_model_weights_not_to_text(setup):
    """Flipping the contract coefficients must flip the contract contribution's sign."""
    run, meta, inputs, explainer = setup
    flipped = clone(run.final_model).fit(inputs, derive_features_target(inputs, run))
    flipped.named_steps["clf"].coef_ = -explainer.pipeline.named_steps["clf"].coef_.copy()
    flipped.named_steps["clf"].intercept_ = explainer.pipeline.named_steps["clf"].intercept_.copy()
    flipped.named_steps["prep"] = explainer.pipeline.named_steps["prep"]
    other = LinearExplainer.from_pipeline(flipped, meta["explanation_reference"])
    a = explainer.contributions(inputs)["contract"]
    b = other.contributions(inputs)["contract"]
    assert np.allclose(a, -b, atol=1e-9) and (a.abs() > 0).any()


def derive_features_target(inputs, run):
    return run.frame[schema.TARGET].to_numpy()


def test_month_to_month_short_tenure_customer_gets_contract_and_tenure_reasons(setup):
    run, _, inputs, explainer = setup
    risky = inputs[(inputs["tenure_months"] <= 3)].head(1).copy()
    risky["contract"] = "Month-to-month"
    reasons = top_reasons(explainer, risky, top_k=3)[0]
    assert reasons, "a customer with clear risk factors must get reasons"
    assert {r["feature"] for r in reasons} & {"contract", "tenure_months"}
    contributions = [r["contribution_log_odds"] for r in reasons]
    assert contributions == sorted(contributions, reverse=True) and all(c > 0 for c in contributions)
    assert len(reasons) <= 3


def test_reasons_are_labelled_model_signal_and_never_causal(setup):
    _, _, inputs, explainer = setup
    for reason in top_reasons(explainer, inputs.head(20), top_k=3)[0:20]:
        for item in reason:
            assert item["evidence_type"] == "MODEL_SIGNAL" and item["causal_claim"] is False
            assert "not a proven cause" in item["statement"]
            assert item["direction"] == "increases_risk"


def test_customers_with_nothing_pushing_risk_up_get_no_filler(setup):
    _, _, inputs, explainer = setup
    contrib = explainer.contributions(inputs)
    low = contrib.index[(contrib < 1e-12).all(axis=1)]
    reasons = top_reasons(explainer, inputs, top_k=3)
    for idx in low:
        assert reasons[inputs.index.get_loc(idx)] == []
    decreasing = top_reasons(explainer, inputs.head(5), top_k=2, direction="decreases_risk")
    assert all(item["contribution_log_odds"] < 0 for customer in decreasing for item in customer)


def test_invalid_direction_is_rejected(setup):
    _, _, inputs, explainer = setup
    with pytest.raises(ValueError):
        top_reasons(explainer, inputs.head(1), direction="causes_churn")


def test_values_are_displayed_in_original_terms(setup):
    _, _, inputs, explainer = setup
    shown = {item["feature"]: item["value"] for r in top_reasons(explainer, inputs, top_k=10) for item in r}
    assert isinstance(shown["contract"], str) and shown["contract"] in {"Month-to-month", "One year", "Two year"}
    binaries = [item["value"] for r in top_reasons(explainer, inputs, top_k=10) for item in r if item["feature"] == "paperless_billing"]
    assert set(binaries) <= {"Yes", "No"}


def test_protected_attributes_never_appear_as_reasons(setup):
    _, _, inputs, explainer = setup
    features = {item["feature"] for r in top_reasons(explainer, inputs, top_k=20) for item in r}
    assert not features & set(PROTECTED_ATTRIBUTES)
    assert not set(explainer.features) & set(PROTECTED_ATTRIBUTES)


def test_global_importance_is_sorted_and_deterministic(setup):
    _, _, inputs, explainer = setup
    table = explainer.global_importance(inputs)
    assert list(table["mean_abs_contribution"]) == sorted(table["mean_abs_contribution"], reverse=True)
    pd.testing.assert_frame_equal(table, explainer.global_importance(inputs.copy()))
    # the synthetic signal is month-to-month, tenure and price: contract/tenure should rank high
    assert {"contract", "tenure_months"} & set(table.index[:3])


def test_reference_round_trips_through_metadata(setup):
    run, meta, inputs, explainer = setup
    rebuilt = LinearExplainer.from_pipeline(run.final_model, reference_means(run.final_model, inputs))
    assert np.allclose(rebuilt.reference, explainer.reference)


def test_non_linear_models_are_refused_rather_than_approximated(synthetic_cleaned):
    feats = derive_features(synthetic_cleaned)
    run = run_baseline(synthetic_cleaned, dataset_sha256="x", **FAST)
    forest = build_pipeline("random_forest", run.spec).fit(feats[list(run.spec.all)], feats[schema.TARGET])
    with pytest.raises(NotImplementedError, match="LogisticRegression"):
        LinearExplainer.from_pipeline(forest, {})


def test_missing_reference_is_a_clear_error(setup):
    run, meta, _, _ = setup
    stripped = {k: v for k, v in meta.items() if k != "explanation_reference"}
    with pytest.raises(KeyError, match="retrain"):
        LinearExplainer.from_artifacts(run.final_model, stripped)


# --- integration with risk records ---------------------------------------------------------------------


def test_scoring_with_an_explainer_adds_reasons_without_changing_scores(setup, synthetic_cleaned):
    run, meta, _, explainer = setup
    plain = score_cleaned(synthetic_cleaned, run.final_model, meta, now=STAMP)
    explained = score_cleaned(synthetic_cleaned, run.final_model, meta, now=STAMP, explainer=explainer, top_k=3)
    assert [r.risk_score for r in plain] == [r.risk_score for r in explained]
    assert [r.risk_band for r in plain] == [r.risk_band for r in explained]
    assert any(r.top_reasons for r in explained) and all(len(r.top_reasons) <= 3 for r in explained)
    assert all(isinstance(reason, Reason) for r in explained for reason in r.top_reasons)


def test_a_reason_cannot_claim_causality():
    base = dict(feature="contract", value="Month-to-month", contribution_log_odds=0.5, direction="increases_risk", statement="s")
    Reason(**base)
    with pytest.raises(ValidationError):
        Reason(**base, causal_claim=True)
    with pytest.raises(ValidationError):
        Reason(**{**base, "evidence_type": "CAUSAL"})
    with pytest.raises(ValidationError):
        RiskRecord(customer_id="x", risk_score=0.5, risk_band="MEDIUM", model_version="v", prediction_timestamp=STAMP, top_reasons=[{"feature": "contract"}])
