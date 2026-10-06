"""Feature engineering: correctness, leakage guards, and training/inference consistency."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import make_raw, make_row

from retainai.data import schema
from retainai.data.clean import clean_telco
from retainai.data.validate import validate_raw
from retainai.features.build import (
    ADDON_COLUMNS,
    DEFAULT_SPEC,
    PROTECTED_ATTRIBUTES,
    FeatureSpec,
    derive_features,
    make_preprocessor,
    model_ready_frame,
    original_feature_of,
)


@pytest.fixture
def features(small_raw):
    return derive_features(clean_telco(small_raw))


def test_yes_no_fields_become_binary_and_structural_values_collapse(features):
    assert set(np.unique(features[list(DEFAULT_SPEC.binary)].to_numpy())) <= {0, 1}
    no_internet = features[features["customer_id"] == "0005-AAAAE"].iloc[0]
    assert no_internet["internet_service"] == "No"
    assert no_internet["multiple_lines"] == 0  # "No phone service" -> 0
    assert all(no_internet[c] == 0 for c in ADDON_COLUMNS)  # "No internet service" -> 0
    assert no_internet["num_addon_services"] == 0


def test_num_addon_services_counts_yes_only(small_raw):
    small_raw.loc[0, ["OnlineSecurity", "OnlineBackup", "TechSupport"]] = "Yes"
    small_raw.loc[0, ["DeviceProtection", "StreamingTV", "StreamingMovies"]] = "No"
    out = derive_features(clean_telco(small_raw))
    assert out.loc[0, "num_addon_services"] == 3


def test_default_spec_excludes_identifier_target_protected_and_audit_columns():
    used = set(DEFAULT_SPEC.all)
    assert schema.CLEAN_ID not in used and schema.TARGET not in used
    assert not used & set(PROTECTED_ATTRIBUTES)
    assert not used & set(schema.AUDIT_COLUMNS)
    assert len(used) == len(DEFAULT_SPEC.all)  # no duplicates


def test_model_ready_frame_contains_only_inputs_id_target_and_group(features):
    frame = model_ready_frame(features)
    assert list(frame.columns) == [schema.CLEAN_ID, *DEFAULT_SPEC.all, schema.TARGET, "duplicate_group_id"]
    assert "gender" not in frame.columns and "senior_citizen" not in frame.columns


def test_ablation_helpers_change_the_spec_without_mutating_the_default():
    assert "contract" not in DEFAULT_SPEC.without("contract").all
    with_protected = DEFAULT_SPEC.with_protected()
    assert {"gender", "senior_citizen"} <= set(with_protected.all)
    assert not {"gender", "senior_citizen"} & set(DEFAULT_SPEC.all)
    assert DEFAULT_SPEC.fingerprint() != with_protected.fingerprint()


def test_derivation_is_deterministic(small_raw):
    cleaned = clean_telco(small_raw)
    pd.testing.assert_frame_equal(derive_features(cleaned), derive_features(cleaned.copy()))


def test_derivation_is_row_wise_so_batch_equals_single_row(small_raw):
    cleaned = clean_telco(small_raw)
    batch = derive_features(cleaned)
    for i in range(len(cleaned)):
        single = derive_features(cleaned.iloc[[i]])
        pd.testing.assert_series_equal(single.iloc[0], batch.iloc[i], check_names=False)


def test_missing_cleaned_column_is_reported(small_raw):
    cleaned = clean_telco(small_raw).drop(columns=["contract"])
    with pytest.raises(KeyError, match="contract"):
        derive_features(cleaned)


def test_inference_path_works_without_a_target_column(small_raw):
    unlabeled = small_raw.drop(columns=["Churn"])
    report = validate_raw(unlabeled, require_target=False)
    assert report.passed
    out = derive_features(clean_telco(unlabeled, report, require_target=False))
    assert schema.TARGET not in out.columns
    labeled = derive_features(clean_telco(small_raw))
    pd.testing.assert_frame_equal(out, labeled.drop(columns=[schema.TARGET]))
    assert "target_label_validity" not in {c.check_id for c in report.checks}


def test_target_is_still_required_by_default(small_raw):
    assert not validate_raw(small_raw.drop(columns=["Churn"])).passed


def test_preprocessor_fits_on_training_rows_only_and_shapes_match(features):
    spec = DEFAULT_SPEC
    pre = make_preprocessor(spec)
    train, other = features.iloc[:5], features.iloc[5:]
    pre.fit(train[list(spec.all)])
    # scaler statistics come from the training rows only
    scaler = pre.named_transformers_["num"]
    assert scaler.mean_[0] == pytest.approx(train["tenure_months"].mean())
    names = list(pre.get_feature_names_out())
    assert pre.transform(other[list(spec.all)]).shape == (len(other), len(names))
    assert len(names) == len(set(names))


def test_unseen_category_at_inference_encodes_to_zeros_not_an_error(features):
    spec = DEFAULT_SPEC
    pre = make_preprocessor(spec).fit(features[list(spec.all)])
    row = features.iloc[[0]].copy()
    row["payment_method"] = "Crypto"
    transformed = pre.transform(row[list(spec.all)])
    names = list(pre.get_feature_names_out())
    pay_cols = [i for i, n in enumerate(names) if n.startswith("cat__payment_method_")]
    assert transformed[0, pay_cols].sum() == 0


def test_original_feature_mapping_for_transformed_columns(features):
    spec = DEFAULT_SPEC
    pre = make_preprocessor(spec).fit(features[list(spec.all)])
    mapped = {original_feature_of(n, spec) for n in pre.get_feature_names_out()}
    assert mapped == set(spec.all)


def test_feature_values_do_not_depend_on_target_label(small_raw):
    """Flipping every label must leave all model inputs untouched (no target leakage)."""
    flipped = small_raw.copy()
    flipped["Churn"] = flipped["Churn"].map({"Yes": "No", "No": "Yes"})
    a = model_ready_frame(derive_features(clean_telco(small_raw)))
    b = model_ready_frame(derive_features(clean_telco(flipped)))
    inputs = [schema.CLEAN_ID, *DEFAULT_SPEC.all]
    pd.testing.assert_frame_equal(a[inputs], b[inputs])


def test_feature_spec_is_a_value_object():
    assert FeatureSpec((), (), ()).all == ()
    assert make_raw([make_row()]).shape[0] == 1  # fixture helper sanity
