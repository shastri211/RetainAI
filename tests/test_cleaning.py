"""Cleaning rules: explicit, deterministic, and non-destructive."""

from __future__ import annotations

import pandas as pd
import pytest
from conftest import make_raw

from retainai.data import schema
from retainai.data.clean import DataContractError, clean_telco


def test_clean_output_matches_contract_layout(small_raw):
    cleaned = clean_telco(small_raw)
    assert list(cleaned.columns) == list(schema.CLEAN_COLUMNS)
    assert len(cleaned) == len(small_raw)
    assert cleaned["tenure_months"].dtype == "int64"
    assert cleaned["monthly_charges"].dtype == "float64"
    assert cleaned["total_charges"].dtype == "float64"
    assert not cleaned["total_charges"].isna().any()


def test_row_order_and_ids_are_preserved(small_raw):
    cleaned = clean_telco(small_raw)
    assert cleaned["customer_id"].tolist() == small_raw["customerID"].tolist()


def test_target_encoding_yes_is_one(small_raw):
    cleaned = clean_telco(small_raw)
    expected = small_raw["Churn"].map({"Yes": 1, "No": 0}).tolist()
    assert cleaned["churn"].tolist() == expected
    assert set(cleaned["churn"]) == {0, 1}


def test_ki1_blank_total_charges_becomes_zero_and_is_flagged(small_raw):
    cleaned = clean_telco(small_raw)
    row = cleaned[cleaned["customer_id"] == "0007-AAAAG"].iloc[0]
    assert row["total_charges"] == 0.0 and bool(row["total_charges_was_blank"])
    assert cleaned["total_charges_was_blank"].sum() == 1
    assert len(cleaned) == len(small_raw)  # the row is kept, not dropped


def test_ki2_duplicates_are_kept_and_grouped(small_raw):
    clone = small_raw.iloc[[0]].copy()
    clone["customerID"] = "9999-ZZZZZ"
    table = make_raw(list(small_raw.to_dict("records")) + list(clone.to_dict("records")))
    cleaned = clean_telco(table)
    assert len(cleaned) == len(table)  # nothing dropped
    first, last = cleaned.iloc[0], cleaned.iloc[-1]
    assert first["duplicate_group_id"] == last["duplicate_group_id"]
    assert first["duplicate_group_size"] == last["duplicate_group_size"] == 2
    others = cleaned.iloc[1:-1]
    assert (others["duplicate_group_size"] == 1).all()
    assert others["duplicate_group_id"].is_unique


def test_ki3_contract_flag_marks_first_term_rows_only(small_raw):
    small_raw.loc[3, "tenure"] = "10"
    cleaned = clean_telco(small_raw)
    flagged = cleaned.loc[cleaned["contract_term_exceeds_tenure"], "customer_id"].tolist()
    assert flagged == ["0004-AAAAD"]  # two-year contract, 10 months in; month-to-month never flagged


def test_ki4_total_charges_is_left_as_reported(small_raw):
    small_raw.loc[0, "TotalCharges"] = "700.00"
    cleaned = clean_telco(small_raw)
    assert cleaned.loc[0, "total_charges"] == 700.0


def test_clean_refuses_data_that_violates_the_contract(small_raw):
    small_raw.loc[0, "Churn"] = "Maybe"
    with pytest.raises(DataContractError) as exc:
        clean_telco(small_raw)
    assert "target_label_validity" in str(exc.value)


def test_cleaning_is_deterministic(small_raw):
    pd.testing.assert_frame_equal(clean_telco(small_raw), clean_telco(small_raw.copy()))


def test_input_table_is_not_mutated(small_raw):
    before = small_raw.copy()
    clean_telco(small_raw)
    pd.testing.assert_frame_equal(small_raw, before)


def test_real_dataset_cleaning_facts(real_raw):
    cleaned = clean_telco(real_raw)
    assert len(cleaned) == 7043 and cleaned["churn"].sum() == 1869
    assert cleaned["total_charges_was_blank"].sum() == 11
    assert (cleaned.loc[cleaned["total_charges_was_blank"], "tenure_months"] == 0).all()
    assert (cleaned["duplicate_group_size"] > 1).sum() == 42
    assert cleaned["customer_id"].is_unique
    assert cleaned["contract_term_exceeds_tenure"].sum() == 244
