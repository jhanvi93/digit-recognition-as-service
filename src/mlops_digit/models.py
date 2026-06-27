"""Model factory.

Each training config names a ``model.type`` and a dict of ``model.params``.
Every model is wrapped in a Pipeline with a StandardScaler so that the
serving layer only ever needs to deal with raw 64-feature vectors.
"""
from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

_MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "svm": SVC,
    "mlp": MLPClassifier,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "knn": KNeighborsClassifier,
}


def available_models() -> list[str]:
    return sorted(_MODEL_REGISTRY)


def build_model(model_type: str, params: dict[str, Any] | None = None) -> Pipeline:
    """Return a scaler + estimator pipeline for the requested model type."""
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type {model_type!r}. "
            f"Available: {', '.join(available_models())}"
        )
    params = dict(params or {})
    estimator = _MODEL_REGISTRY[model_type](**params)
    return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
