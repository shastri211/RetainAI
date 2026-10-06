"""Subgroup audit: correct arithmetic, honest uncertainty, no fairness verdict."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_raw

from retainai.data import schema
from retainai.data.clean import clean_telco
from retainai.fairness.audit import (
    AUDIT_ATTRIBUTES,
    CAVEATS,
    MIN_GROUP_SIZE,
    audit_attribute,
    run_fairness_audit,
    sample_metrics,
)
from retainai.features.build import DEFAULT_SPEC
from retainai.models.baseline import run_baseline

THRESHOLDS = {"best_f1": 0.3, "high_band": 0.5}
FAST = dict(n_splits=3, n_repeats=1, candidate_names=("prior_baseline", "logistic_regression"), run_ablations=False)


def _frame(n=600, seed=0, bias_b=0.0):
    """Two groups, calibrated scores; group B's scores are shifted up by ``bias_b``."""
    rng = np.random.default_rng(seed)
    group = np.where(rng.random(n) < 0.5, "A", "B")
    p_true = rng.uniform(0.05, 0.9, n)
    y = (rng.random(n) < p_true).astype(int)
    p = np.clip(p_true + np.where(group == "B", bias_b, 0.0), 0, 1)
    return pd.DataFrame({"y": y, "p": p, "attr": group})


def test_sample_metrics_hand_checked():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.4, 0.6, 0.1, 0.8, 0.2])
    m = sample_metrics(y, p, {"t": 0.5})
    assert m["tpr@t"] == pytest.approx(2 / 3)
    assert m["fpr@t"] == pytest.approx(1 / 3)
    assert m["precision@t"] == pytest.approx(2 / 3)
    assert m["selection_rate@t"] == pytest.approx(0.5)
    assert m["calibration_gap"] == pytest.approx(p.mean() - y.mean())
    assert m["observed_churn_rate"] == pytest.approx(0.5)


def test_undefined_metrics_are_nan_not_zero():
    m = sample_metrics(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]), {"t": 0.9})
    assert np.isnan(m["roc_auc"]) and np.isnan(m["tpr@t"]) and np.isnan(m["precision@t"])


def test_group_counts_cover_every_row_and_estimates_sit_in_their_intervals():
    frame = _frame()
    result = audit_attribute(frame, "attr", THRESHOLDS, n_boot=100, seed=1)
    groups = result["groups"]
    assert sum(g["n"] for g in groups.values()) == len(frame)
    for g in groups.values():
        assert g["status"] == "ok"
        for key in ("roc_auc", "tpr@best_f1", "calibration_gap"):
            est, (lo, hi) = g["metrics"][key]["estimate"], g["metrics"][key]["ci95"]
            assert lo <= hi and lo - 0.05 <= est <= hi + 0.05


def test_audit_is_deterministic_for_a_seed():
    frame = _frame()
    a = audit_attribute(frame, "attr", THRESHOLDS, n_boot=100, seed=3)
    b = audit_attribute(frame.copy(), "attr", THRESHOLDS, n_boot=100, seed=3)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_planted_miscalibration_in_one_group_is_detected():
    frame = _frame(n=2000, bias_b=0.2)
    diff = audit_attribute(frame, "attr", THRESHOLDS, n_boot=100, seed=2)["differences"]["B_minus_A"]
    assert diff["calibration_gap"]["estimate"] > 0.1
    assert diff["calibration_gap"]["ci_excludes_zero"] is True


def test_no_planted_gap_gives_an_interval_that_includes_zero():
    frame = _frame(n=2000, bias_b=0.0)
    diff = audit_attribute(frame, "attr", THRESHOLDS, n_boot=100, seed=2)["differences"]["B_minus_A"]
    assert diff["calibration_gap"]["ci_excludes_zero"] is False


def test_small_groups_are_flagged_not_given_false_precision():
    frame = _frame(n=400)
    tiny = pd.concat([frame, pd.DataFrame({"y": [1, 0, 1], "p": [0.9, 0.1, 0.7], "attr": "C"})], ignore_index=True)
    result = audit_attribute(tiny, "attr", THRESHOLDS, n_boot=50, seed=1)
    assert result["groups"]["C"]["status"].startswith("insufficient_data") and "metrics" not in result["groups"]["C"]
    assert result["groups"]["C"]["n"] < MIN_GROUP_SIZE
    assert not any("C" in name for name in result["differences"])


# --- full audit on a trained model --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def audit_setup():
    cleaned = clean_telco(synthetic_raw(n=800, seed=11))
    run = run_baseline(cleaned, dataset_sha256="x", **FAST)
    audit = run_fairness_audit(
        run.oof, run.holdout_pred, cleaned, run.spec, run.operating_thresholds | {"high_band": 0.5}, n_boot=60, seed=5
    )
    return cleaned, run, audit


def test_audit_covers_all_attributes_on_both_evaluation_sets(audit_setup):
    _, run, audit = audit_setup
    assert set(audit["evaluation_sets"]) == {"dev_oof", "holdout"}
    assert audit["evaluation_sets"]["dev_oof"]["n_customers"] == len(run.oof)
    assert audit["evaluation_sets"]["holdout"]["n_customers"] == len(run.holdout_pred)
    for evaluation in audit["evaluation_sets"].values():
        assert set(evaluation["attributes"]) == set(AUDIT_ATTRIBUTES)


def test_attribute_roles_state_what_the_model_actually_uses(audit_setup):
    _, _, audit = audit_setup
    roles = audit["attribute_roles"]
    assert roles["gender"]["used_in_model"] is False and "audit only" in roles["gender"]["role"]
    assert roles["senior_citizen"]["used_in_model"] is False
    assert roles["partner"]["used_in_model"] is True and "sensitive-adjacent" in roles["partner"]["role"]


def test_attribute_roles_reflect_an_ablation_spec_that_includes_protected_attributes(audit_setup):
    cleaned, run, _ = audit_setup
    audit = run_fairness_audit(
        run.oof, run.holdout_pred, cleaned, DEFAULT_SPEC.with_protected(), run.operating_thresholds | {"high_band": 0.5}, n_boot=10, seed=1
    )
    assert audit["attribute_roles"]["gender"]["used_in_model"] is True
    assert "USED in model" in audit["attribute_roles"]["gender"]["role"]


def test_audit_makes_no_fairness_claim(audit_setup):
    _, _, audit = audit_setup
    text = json.dumps(audit).lower()
    assert "verdict" not in text and "passed" not in text and "is fair" not in text
    assert any("does not establish fairness" in c for c in audit["caveats"]) and tuple(audit["caveats"]) == CAVEATS
    assert any("proxies" in c for c in audit["caveats"])  # excluding an attribute is not removing its signal


def test_unmatched_customers_or_missing_attributes_are_errors(audit_setup):
    cleaned, run, _ = audit_setup
    with pytest.raises(KeyError, match="gender"):
        run_fairness_audit(run.oof, run.holdout_pred, cleaned.drop(columns=["gender"]), run.spec, run.operating_thresholds, n_boot=5)
    renamed = run.oof.assign(**{schema.CLEAN_ID: "UNKNOWN-ID"})
    with pytest.raises(ValueError, match="do not match"):
        run_fairness_audit(renamed, run.holdout_pred, cleaned, run.spec, run.operating_thresholds, n_boot=5)
