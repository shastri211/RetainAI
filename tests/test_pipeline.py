"""End-to-end pipeline, reproducibility and evidence-boundary guards."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from conftest import REAL_CSV, synthetic_raw

from retainai.cli import main
from retainai.data import schema
from retainai.data.clean import clean_telco
from retainai.data.load import dataset_sha256, load_raw
from retainai.data.validate import validate_raw
from retainai.features.build import PROTECTED_ATTRIBUTES, derive_features, model_ready_frame
from retainai.io import write_csv
from retainai.models.baseline import run_baseline
from retainai.models.registry import load_artifacts

# Fingerprint of the committed raw dataset (line-ending normalised). A change here means the raw file was edited.
RAW_DATASET_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def pipeline_dir(tmp_path_factory):
    """Run prepare -> train -> score -> value once on a small synthetic file."""
    d = tmp_path_factory.mktemp("pipeline")
    write_csv(synthetic_raw(n=500, seed=21), d / "raw.csv")
    assert main(["prepare", "--input", str(d / "raw.csv"), "--out-dir", str(d / "processed")]) == 0
    assert (
        main(
            [
                "train", "--input", str(d / "raw.csv"), "--model-dir", str(d / "model"),
                "--report", str(d / "baseline.json"), "--fairness-report", str(d / "fair.json"),
                "--n-repeats", "1", "--bootstrap", "20",
            ]
        )
        == 0
    )
    assert main(["score", "--input", str(d / "raw.csv"), "--model-dir", str(d / "model"), "--output", str(d / "scores.csv")]) == 0
    assert (
        main(
            [
                "value", "--input", str(d / "raw.csv"), "--model-dir", str(d / "model"),
                "--output", str(d / "value.csv"), "--summary", str(d / "value.json"),
            ]
        )
        == 0
    )
    return d


# --- raw data integrity ----------------------------------------------------------------------------------


def test_raw_dataset_is_unchanged():
    assert dataset_sha256(REAL_CSV) == RAW_DATASET_SHA256


def test_pipeline_never_writes_to_the_raw_directory(pipeline_dir):
    assert sorted(p.name for p in (ROOT / "data" / "raw").iterdir()) == ["Telco-Customer-Churn.csv"]


def test_committed_validation_report_matches_a_fresh_validation():
    fresh = validate_raw(load_raw(REAL_CSV))
    fresh.dataset_sha256 = dataset_sha256(REAL_CSV)
    committed = json.loads((ROOT / "reports" / "data_validation.json").read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(fresh.to_dict(), sort_keys=True, default=str))


# --- end to end ------------------------------------------------------------------------------------------------


def test_every_pipeline_stage_produced_its_outputs(pipeline_dir):
    for name in ("processed/telco_cleaned.csv", "processed/telco_features.csv", "baseline.json", "fair.json", "scores.csv", "value.csv", "value.json", "model/model.joblib", "model/metadata.json"):
        assert (pipeline_dir / name).exists(), name
    assert len(pd.read_csv(pipeline_dir / "scores.csv")) == len(pd.read_csv(pipeline_dir / "raw.csv"))


def test_model_ready_table_has_no_identifier_target_leakage_or_protected_columns(pipeline_dir):
    columns = list(pd.read_csv(pipeline_dir / "processed" / "telco_features.csv").columns)
    assert not set(columns) & set(PROTECTED_ATTRIBUTES)
    assert not set(columns) & set(schema.AUDIT_COLUMNS) - {"duplicate_group_id"}
    assert columns[0] == schema.CLEAN_ID and columns[-2] == schema.TARGET
    forbidden = ("offer", "discount", "treat", "campaign", "intervention", "cost")
    assert not [c for c in columns if any(word in c.lower() for word in forbidden)]


def test_saved_model_inputs_exclude_id_target_protected_and_audit_columns(pipeline_dir):
    model, _ = load_artifacts(pipeline_dir / "model")
    inputs = set(model.named_steps["prep"].feature_names_in_)
    assert not inputs & {schema.CLEAN_ID, schema.TARGET, "duplicate_group_id", *PROTECTED_ATTRIBUTES, *schema.AUDIT_COLUMNS}


def test_scoring_unlabeled_customers_works_end_to_end(pipeline_dir):
    new = synthetic_raw(n=40, seed=77, n_duplicates=0).drop(columns=["Churn"])
    write_csv(new, pipeline_dir / "new.csv")
    out = pipeline_dir / "new_scores.csv"
    assert main(["score", "--input", str(pipeline_dir / "new.csv"), "--model-dir", str(pipeline_dir / "model"), "--output", str(out)]) == 0
    scores = pd.read_csv(out)
    assert len(scores) == 40 and not scores["in_sample"].any()
    reasons = [json.loads(r) for r in scores["top_reasons"]]
    assert any(reasons)


# --- reproducibility ------------------------------------------------------------------------------------------------


def test_prepare_is_byte_reproducible(pipeline_dir, tmp_path):
    again = tmp_path / "processed"
    assert main(["prepare", "--input", str(pipeline_dir / "raw.csv"), "--out-dir", str(again)]) == 0
    for name in ("telco_cleaned.csv", "telco_features.csv"):
        assert _sha(again / name) == _sha(pipeline_dir / "processed" / name)


def test_training_is_reproducible_report_scores_and_version(pipeline_dir, tmp_path):
    again = tmp_path / "run2"
    assert (
        main(
            [
                "train", "--input", str(pipeline_dir / "raw.csv"), "--model-dir", str(again / "model"),
                "--report", str(again / "baseline.json"), "--fairness-report", str(again / "fair.json"),
                "--n-repeats", "1", "--bootstrap", "20",
            ]
        )
        == 0
    )
    assert _sha(again / "baseline.json") == _sha(pipeline_dir / "baseline.json")
    assert _sha(again / "fair.json") == _sha(pipeline_dir / "fair.json")
    meta1 = json.loads((pipeline_dir / "model" / "metadata.json").read_text())
    meta2 = json.loads((again / "model" / "metadata.json").read_text())
    assert meta1["model_version"] == meta2["model_version"]
    differing = {k for k in meta1 if meta1[k] != meta2[k]}
    assert differing <= {"training_timestamp_utc"}  # only the timestamp may differ
    m1, _ = load_artifacts(pipeline_dir / "model")
    m2, _ = load_artifacts(again / "model")
    X = derive_features(clean_telco(load_raw(pipeline_dir / "raw.csv")))
    cols = list(m1.named_steps["prep"].feature_names_in_)
    assert (m1.predict_proba(X[cols]) == m2.predict_proba(X[cols])).all()


def test_metadata_records_the_required_provenance(pipeline_dir):
    meta = json.loads((pipeline_dir / "model" / "metadata.json").read_text())
    assert meta["dataset"]["sha256_lf_normalised"] == dataset_sha256(pipeline_dir / "raw.csv")
    assert meta["feature_pipeline"]["version"] and meta["model_type"] == "logistic_regression"
    assert meta["hyperparameters"] and meta["training_timestamp_utc"] and meta["model_version"]
    assert meta["evaluation"]["cv_dev"]["roc_auc"]["mean"] > 0.5


# --- no causal claims, no invented effects ------------------------------------------------------------------------------


def test_outputs_carry_the_evidence_boundary_not_causal_claims(pipeline_dir):
    report = json.loads((pipeline_dir / "baseline.json").read_text())
    assert report["global_explanation"]["causal_explanation_available"] is False
    assert any("Nothing here estimates the effect of any intervention" in s for s in report["limitations"])
    assert report["provenance_label"] == "PUBLIC"
    scores = pd.read_csv(pipeline_dir / "scores.csv")
    for cell in scores["top_reasons"]:
        for reason in json.loads(cell):
            assert reason["evidence_type"] == "MODEL_SIGNAL" and reason["causal_claim"] is False
            assert "not a proven cause" in reason["statement"]
    summary = json.loads((pipeline_dir / "value.json").read_text())
    assert summary["basis"]["clv_proxy"].startswith("ASSUMED") and "not a survival-based or true CLV" in summary["basis"]["clv_proxy"]
    assert "scoring_context" in summary


DEFERRED_PACKAGES = {
    "openai", "anthropic", "langchain", "langgraph", "llama_index", "qdrant_client", "chromadb", "faiss",
    "causalml", "econml", "sklift", "lifelines", "shap", "torch", "tensorflow", "keras", "redis", "psycopg2",
    "sqlalchemy", "fastapi", "flask", "celery", "mlflow",
}


def test_phase_1_imports_no_deferred_technology():
    """LLM, RAG, vector DB, uplift/causal, deep learning, DB, web and tracking packages are deferred to later phases."""
    for path in (ROOT / "src" / "retainai").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not imported & DEFERRED_PACKAGES, (path.name, imported & DEFERRED_PACKAGES)
    pinned = {line.split("==")[0].lower().replace("-", "_") for line in (ROOT / "requirements.txt").read_text().splitlines() if "==" in line}
    assert not pinned & DEFERRED_PACKAGES


# --- leakage on the real data -----------------------------------------------------------------------------------------------


def test_shuffled_labels_on_real_data_give_chance_performance():
    cleaned = clean_telco(load_raw(REAL_CSV))
    shuffled = cleaned.copy()
    shuffled[schema.TARGET] = shuffled[schema.TARGET].sample(frac=1, random_state=0).to_numpy()
    run = run_baseline(
        shuffled, dataset_sha256="shuffled", n_splits=5, n_repeats=1,
        candidate_names=("prior_baseline", "logistic_regression"), run_ablations=False,
    )
    auc = run.report["candidates"]["logistic_regression"]["cv"]["roc_auc"]["mean"]
    assert 0.46 < auc < 0.54, auc  # any leakage of the label through features would push this well above 0.5


def test_no_single_real_input_is_a_near_copy_of_the_label():
    frame = model_ready_frame(derive_features(clean_telco(load_raw(REAL_CSV))))
    label = frame[schema.TARGET].astype(float)
    inputs = [c for c in frame.columns if c not in (schema.CLEAN_ID, schema.TARGET, "duplicate_group_id")]
    for col in inputs:
        encoded = pd.get_dummies(frame[col]).astype(float) if frame[col].dtype == object else frame[[col]].astype(float)
        strongest = max(abs(encoded[c].corr(label)) for c in encoded.columns)
        assert strongest < 0.5, (col, strongest)


# --- slow: full real retrain reproduces the committed reports ---------------------------------------------------------------------


@pytest.mark.slow
def test_real_training_reproduces_committed_reports(tmp_path):
    out = tmp_path
    assert (
        main(["train", "--model-dir", str(out / "m"), "--report", str(out / "b.json"), "--fairness-report", str(out / "f.json")]) == 0
    )
    assert _sha(out / "b.json") == _sha(ROOT / "reports" / "baseline_metrics.json")
    assert _sha(out / "f.json") == _sha(ROOT / "reports" / "fairness_audit.json")
