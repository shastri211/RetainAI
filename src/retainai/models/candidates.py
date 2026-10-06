"""Candidate baseline models. Fixed, modest configurations: no hyperparameter search in Phase 1.

``SHIPPED_MODEL`` is the interpretable model whose contributions can be decomposed exactly
(see ``retainai.explain``). Tree models are benchmarks: if one beats it by more than fold-to-fold
noise the report says so and the trade-off is handed to the owner rather than decided silently.
"""

from __future__ import annotations

from collections.abc import Callable

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from retainai.config import SEED
from retainai.features.build import DEFAULT_SPEC, FeatureSpec, make_preprocessor

SHIPPED_MODEL = "logistic_regression"

# Simplest first. Used only to describe benchmark comparisons.
COMPLEXITY_ORDER = ("prior_baseline", "logistic_regression", "hist_gradient_boosting", "random_forest")

ESTIMATOR_FACTORIES: dict[str, Callable[[], BaseEstimator]] = {
    "prior_baseline": lambda: DummyClassifier(strategy="prior"),
    "logistic_regression": lambda: LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, n_jobs=1, random_state=SEED
    ),
    "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=SEED
    ),
}


def build_pipeline(name: str, spec: FeatureSpec = DEFAULT_SPEC) -> Pipeline:
    """Fresh, unfitted preprocessing + estimator pipeline."""
    if name not in ESTIMATOR_FACTORIES:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(ESTIMATOR_FACTORIES)}")
    return Pipeline([("prep", make_preprocessor(spec)), ("clf", ESTIMATOR_FACTORIES[name]())])
