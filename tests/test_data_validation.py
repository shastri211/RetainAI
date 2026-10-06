"""Schema validation, known-issue handling and dataset fingerprinting."""

from __future__ import annotations

import pytest
from conftest import make_raw

from retainai.data import schema
from retainai.data.load import dataset_sha256
from retainai.data.validate import Severity, validate_raw


def _triggered_errors(report) -> set[str]:
    return {c.check_id for c in report.errors}


def test_contract_has_21_columns_and_a_known_target():
    assert len(schema.RAW_COLUMNS) == 21
    assert schema.RAW_ID in schema.RAW_COLUMNS and schema.RAW_TARGET in schema.RAW_COLUMNS
    assert set(schema.RAW_TO_CLEAN) == set(schema.RAW_COLUMNS)
    assert schema.TARGET_ENCODING == {"No": 0, "Yes": 1}


def test_valid_small_table_passes(small_raw):
    report = validate_raw(small_raw)
    assert report.passed, [c.check_id for c in report.errors]


def test_missing_required_column_is_an_error(small_raw):
    report = validate_raw(small_raw.drop(columns=["Contract"]))
    assert not report.passed
    assert _triggered_errors(report) == {"required_columns"}  # later checks are skipped


def test_unexpected_column_is_an_error(small_raw):
    small_raw["Extra"] = "x"
    assert "unexpected_columns" in _triggered_errors(validate_raw(small_raw))


def test_empty_table_is_an_error(small_raw):
    assert "non_empty" in _triggered_errors(validate_raw(small_raw.iloc[0:0]))


def test_duplicate_customer_id_is_an_error(small_raw):
    small_raw.loc[1, "customerID"] = small_raw.loc[0, "customerID"]
    assert "customer_id_unique" in _triggered_errors(validate_raw(small_raw))


def test_blank_cell_outside_total_charges_is_an_error(small_raw):
    small_raw.loc[2, "Contract"] = " "
    errors = _triggered_errors(validate_raw(small_raw))
    assert "missing_values" in errors and "categorical_domains" in errors


@pytest.mark.parametrize(
    ("column", "value", "check"),
    [
        ("tenure", "abc", "tenure_numeric"),
        ("tenure", "2.5", "tenure_numeric"),
        ("tenure", "-1", "tenure_non_negative"),
        ("MonthlyCharges", "free", "monthly_charges_numeric"),
        ("MonthlyCharges", "0", "monthly_charges_positive"),
        ("TotalCharges", "n/a", "total_charges_numeric"),
        ("TotalCharges", "-5", "total_charges_non_negative"),
        ("SeniorCitizen", "2", "senior_citizen_binary"),
    ],
)
def test_invalid_numeric_values_are_errors(small_raw, column, value, check):
    small_raw.loc[0, column] = value
    assert check in _triggered_errors(validate_raw(small_raw))


def test_invalid_categorical_value_is_an_error(small_raw):
    small_raw.loc[0, "PaymentMethod"] = "Cash"
    check = validate_raw(small_raw).get("categorical_domains")
    assert check.triggered and check.details["invalid_values_by_column"] == {"PaymentMethod": ["Cash"]}


@pytest.mark.parametrize("bad", ["yes", "1", "Maybe"])
def test_invalid_target_label_is_an_error(small_raw, bad):
    small_raw.loc[0, "Churn"] = bad
    assert "target_label_validity" in _triggered_errors(validate_raw(small_raw))


def test_single_class_target_is_an_error(small_raw):
    small_raw["Churn"] = "No"
    assert "target_both_classes" in _triggered_errors(validate_raw(small_raw))


def test_structural_inconsistency_is_an_error(small_raw):
    small_raw.loc[0, "PhoneService"] = "No"  # but MultipleLines stays "No"
    assert "structural_multiple_lines" in _triggered_errors(validate_raw(small_raw))
    small_raw.loc[0, "PhoneService"] = "Yes"
    small_raw.loc[0, "InternetService"] = "No"  # but add-ons are not "No internet service"
    assert "structural_internet_addons" in _triggered_errors(validate_raw(small_raw))


def test_ki1_blank_total_charges_with_zero_tenure_is_a_warning_not_an_error(small_raw):
    report = validate_raw(small_raw)
    ki1 = report.get("known_issue_ki1_blank_total_charges")
    assert ki1.triggered and ki1.severity is Severity.WARNING and ki1.count == 1
    assert report.passed


def test_blank_total_charges_with_positive_tenure_is_an_error(small_raw):
    small_raw.loc[2, "TotalCharges"] = " "  # tenure 30
    assert "total_charges_blank_with_tenure" in _triggered_errors(validate_raw(small_raw))


def test_total_charges_inconsistent_with_tenure_zero_is_an_error(small_raw):
    small_raw.loc[6, "TotalCharges"] = "50.00"  # tenure 0 but billed
    assert "total_charges_zero_tenure" in _triggered_errors(validate_raw(small_raw))


def test_ki2_duplicate_rows_are_detected_but_do_not_fail(small_raw):
    clone = small_raw.iloc[[0]].copy()
    clone["customerID"] = "9999-ZZZZZ"
    table = make_raw(list(small_raw.to_dict("records")) + list(clone.to_dict("records")))
    report = validate_raw(table)
    ki2 = report.get("known_issue_ki2_duplicate_rows")
    assert ki2.triggered and ki2.severity is Severity.WARNING
    assert ki2.details["extra_rows"] == 1 and ki2.details["groups"] == 1
    assert report.passed


def test_ki3_is_informational_and_counts_fixed_terms_in_first_term(small_raw):
    ki3 = validate_raw(small_raw).get("known_issue_ki3_contract_within_first_term")
    assert ki3.severity is Severity.INFO and not ki3.triggered
    assert ki3.details["One year"]["rows"] == 0 and ki3.details["Two year"]["rows"] == 0
    small_raw.loc[3, "tenure"] = "10"  # two-year contract, 10 months in
    report = validate_raw(small_raw)
    assert report.get("known_issue_ki3_contract_within_first_term").details["Two year"]["rows"] == 1
    assert report.passed  # plausible, not an error


def test_ki4_total_charges_not_forced_to_equal_tenure_times_monthly(small_raw):
    small_raw.loc[0, "TotalCharges"] = "700.00"  # +16.7% vs 12 * 50, still plausible
    report = validate_raw(small_raw)
    assert report.passed and not report.get("total_charges_gross_deviation").triggered
    assert report.get("known_issue_ki4_total_vs_tenure_x_monthly").severity is Severity.INFO


def test_gross_total_charges_deviation_warns(small_raw):
    small_raw.loc[0, "TotalCharges"] = "6000.00"  # 10x
    check = validate_raw(small_raw).get("total_charges_gross_deviation")
    assert check.triggered and check.severity is Severity.WARNING


# --- the real dataset -------------------------------------------------------------------------


def test_real_dataset_passes_and_matches_phase0_known_issues(real_raw):
    report = validate_raw(real_raw)
    assert report.passed and report.n_rows == 7043 and report.n_columns == 21
    assert report.get("known_issue_ki1_blank_total_charges").count == 11
    ki2 = report.get("known_issue_ki2_duplicate_rows")
    assert ki2.details["extra_rows"] == 22 and ki2.details["groups"] == 20
    assert set(ki2.details["tenure_of_duplicated_rows"]) == {"1"}
    ki3 = report.get("known_issue_ki3_contract_within_first_term")
    assert ki3.details["Two year"] == {"rows": 142, "churned": 0}
    assert ki3.details["One year"] == {"rows": 102, "churned": 8}
    prevalence = report.get("target_prevalence").details
    assert (prevalence["positive"], prevalence["negative"]) == (1869, 5174)


def test_dataset_hash_is_independent_of_line_endings(tmp_path):
    lf = tmp_path / "a.csv"
    crlf = tmp_path / "b.csv"
    lf.write_bytes(b"a,b\n1,2\n")
    crlf.write_bytes(b"a,b\r\n1,2\r\n")
    assert dataset_sha256(lf) == dataset_sha256(crlf)
    crlf.write_bytes(b"a,b\r\n1,3\r\n")
    assert dataset_sha256(lf) != dataset_sha256(crlf)
