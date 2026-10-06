"""Model versioning, metadata and artefact persistence (local files; no tracking platform).

The model binary (``model.joblib``) is git-ignored. Metadata records everything needed to
reproduce or audit a model: dataset hash, feature pipeline version, model type and
hyperparameters, seed, training timestamp, library versions and evaluation metrics.

Security note: ``joblib`` files are pickles. Only load artefacts that this project produced locally.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from retainai import __version__
from retainai.data import schema
from retainai.io import write_json
from retainai.models.baseline import BaselineRun
from retainai.models.candidates import ESTIMATOR_FACTORIES

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"

_SHORT_NAMES = {
    "logistic_regression": "lr",
    "random_forest": "rf",
    "hist_gradient_boosting": "hgb",
    "prior_baseline": "prior",
}


def model_version(
    *,
    model_name: str,
    dataset_sha256: str,
    feature_fingerprint: str,
    hyperparameters: dict,
    seed: int,
) -> str:
    """Content-derived version: identical inputs always give the same string."""
    payload = json.dumps(
        {
            "model": model_name,
            "dataset": dataset_sha256,
            "features": feature_fingerprint,
            "params": hyperparameters,
            "seed": seed,
            "sklearn": sklearn.__version__,
            "scope": "final refit on all rows",
        },
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:10]
    return f"churn-{_SHORT_NAMES.get(model_name, model_name)}-{digest}"


def build_metadata(
    run: BaselineRun,
    *,
    dataset_sha256: str,
    seed: int,
    trained_at: datetime | None = None,
    extra: dict | None = None,
) -> dict:
    name = run.shipped_model
    params = dict(sorted(ESTIMATOR_FACTORIES[name]().get_params().items()))
    version = model_version(
        model_name=name,
        dataset_sha256=dataset_sha256,
        feature_fingerprint=run.spec.fingerprint(),
        hyperparameters=params,
        seed=seed,
    )
    shipped = run.report["candidates"][name]
    metadata = {
        "model_version": version,
        "model_type": name,
        "hyperparameters": params,
        "training_timestamp_utc": (trained_at or datetime.now(UTC)).isoformat(timespec="seconds"),
        "trained_on": "all rows (final refit); evaluation metrics describe dev-fitted models",
        "seed": seed,
        "dataset": {
            "sha256_lf_normalised": dataset_sha256,
            "provenance_label": schema.PROVENANCE_LABEL,
            "n_rows": run.report["dataset"]["n_rows"],
            "target": run.report["dataset"]["target"],
        },
        "feature_pipeline": run.report["feature_pipeline"],
        "evaluation": {
            "cv_dev": shipped["cv"],
            "holdout": shipped["holdout"],
            "operating_thresholds": run.operating_thresholds,
        },
        "libraries": {
            "python": platform.python_version(),
            "retainai": __version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "limitations": run.report["limitations"],
    }
    if extra:
        metadata.update(extra)
    return metadata


def save_artifacts(model: Pipeline, metadata: dict, model_dir: str | Path) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / MODEL_FILE)
    write_json(metadata, model_dir / METADATA_FILE)
    return model_dir


def load_artifacts(model_dir: str | Path) -> tuple[Pipeline, dict]:
    model_dir = Path(model_dir)
    model_path, meta_path = model_dir / MODEL_FILE, model_dir / METADATA_FILE
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"no trained model in {model_dir}; run `retainai train` first")
    return joblib.load(model_path), json.loads(meta_path.read_text(encoding="utf-8"))
