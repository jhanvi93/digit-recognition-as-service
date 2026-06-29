"""FastAPI serving layer for the digit-recognition model.

Endpoints:
    GET  /livez    - liveness probe (process is up)
    GET  /readyz   - readiness probe (model is loaded)
    GET  /metrics  - Prometheus exposition format
    POST /predict  - classify one or more 64-feature digit vectors

The served model is resolved in this order: the ``MODEL_PATH`` environment
variable, then the auto-promoted ``models/champion.joblib`` (written by the
selector), then the legacy ``models/balanced.joblib`` manual default.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field, field_validator

from mlops_digit.data import N_FEATURES
from mlops_digit.monitor import DEFAULT_THRESHOLD, check_drift

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
CHAMPION_MODEL = MODELS_DIR / "champion.joblib"
CHAMPION_META = MODELS_DIR / "champion.json"
LEGACY_MODEL_PATH = MODELS_DIR / "balanced.joblib"

PREDICTIONS = Counter(
    "digit_predictions_total", "Total number of predicted digit samples."
)
PREDICT_LATENCY = Histogram(
    "digit_predict_latency_seconds",
    "Latency of /predict requests in seconds.",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
MODEL_ACCURACY = Gauge(
    "digit_model_accuracy",
    "Most recently monitored accuracy of the served model (via /monitor).",
)
DRIFT_ALERTS = Counter(
    "digit_drift_alerts_total",
    "Number of performance-drop alerts raised by the monitor.",
)


class PredictRequest(BaseModel):
    """A batch of digit images, each flattened to 64 grayscale features (0-16)."""

    instances: List[List[float]] = Field(
        ...,
        description="List of equal-length feature vectors (64 for digits, 784 for MNIST).",
        examples=[[[0.0] * N_FEATURES]],
    )

    @field_validator("instances")
    @classmethod
    def _check_rectangular(cls, v: List[List[float]]) -> List[List[float]]:
        # Dataset-agnostic: only enforce a non-empty, rectangular batch here.
        # The exact feature count is validated against the loaded model.
        if not v:
            raise ValueError("instances must contain at least one sample")
        width = len(v[0])
        if width == 0:
            raise ValueError("feature vectors must be non-empty")
        for i, row in enumerate(v):
            if len(row) != width:
                raise ValueError(
                    f"instance {i} has {len(row)} features, expected {width}"
                )
        return v


class Prediction(BaseModel):
    label: int
    probabilities: List[float] | None = None


class PredictResponse(BaseModel):
    predictions: List[Prediction]
    model_name: str
    inference_ms: float


class MonitorRequest(BaseModel):
    """A labelled batch used to check the served model for performance drops."""

    instances: List[List[float]]
    labels: List[int]


class MonitorResponse(BaseModel):
    status: str
    current_accuracy: float
    baseline_accuracy: float
    drop: float
    threshold: float
    n_samples: int
    errors: int
    message: str


def _model_path() -> Path:
    """Resolve which model file to serve (env > champion > legacy)."""
    override = os.environ.get("MODEL_PATH")
    if override:
        return Path(override)
    if CHAMPION_MODEL.exists():
        return CHAMPION_MODEL
    return LEGACY_MODEL_PATH


def _baseline_accuracy() -> float | None:
    if CHAMPION_META.exists():
        with open(CHAMPION_META, "r", encoding="utf-8") as fh:
            return float(json.load(fh).get("baseline_accuracy", 0.0))
    return None


def _validate_features(model, X: np.ndarray) -> None:
    """Reject a batch whose feature count does not match the loaded model."""
    expected = getattr(model, "n_features_in_", None)
    if expected is not None and X.shape[1] != expected:
        raise HTTPException(
            status_code=422,
            detail=f"expected {expected} features per instance, got {X.shape[1]}",
        )


def create_app() -> FastAPI:
    state: dict = {
        "model": None,
        "model_name": _model_path().stem,
        "baseline_accuracy": None,
    }

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        path = _model_path()
        if path.exists():
            state["model"] = joblib.load(path)
            state["model_name"] = path.stem
            state["baseline_accuracy"] = _baseline_accuracy()
        yield

    app = FastAPI(title="Digit Recognition Service", version="1.0.0", lifespan=lifespan)

    @app.get("/livez")
    def livez() -> dict:
        return {"status": "alive"}

    @app.get("/readyz")
    def readyz() -> dict:
        if state["model"] is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return {"status": "ready", "model": state["model_name"]}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        model = state["model"]
        if model is None:
            raise HTTPException(status_code=503, detail="model not loaded")

        X = np.asarray(req.instances, dtype=np.float64)
        _validate_features(model, X)
        start = time.perf_counter()
        with PREDICT_LATENCY.time():
            labels = model.predict(X)
            proba = (
                model.predict_proba(X) if hasattr(model, "predict_proba") else None
            )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        PREDICTIONS.inc(len(labels))

        predictions = [
            Prediction(
                label=int(labels[i]),
                probabilities=(proba[i].tolist() if proba is not None else None),
            )
            for i in range(len(labels))
        ]
        return PredictResponse(
            predictions=predictions,
            model_name=state["model_name"],
            inference_ms=elapsed_ms,
        )

    @app.post("/monitor", response_model=MonitorResponse)
    def monitor(req: MonitorRequest) -> MonitorResponse:
        model = state["model"]
        if model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        if len(req.instances) != len(req.labels):
            raise HTTPException(
                status_code=422,
                detail=f"instances ({len(req.instances)}) and labels "
                f"({len(req.labels)}) must have equal length",
            )

        X = np.asarray(req.instances, dtype=np.float64)
        y = np.asarray(req.labels, dtype=np.int64)
        _validate_features(model, X)

        preds = model.predict(X)
        current_acc = float(np.mean(preds == y))
        errors = int(np.sum(preds != y))
        MODEL_ACCURACY.set(current_acc)

        baseline = state["baseline_accuracy"]
        threshold = float(os.environ.get("DRIFT_THRESHOLD", DEFAULT_THRESHOLD))
        if baseline is None:
            # No champion baseline recorded; report accuracy without alerting.
            return MonitorResponse(
                status="ok",
                current_accuracy=round(current_acc, 4),
                baseline_accuracy=0.0,
                drop=0.0,
                threshold=threshold,
                n_samples=len(y),
                errors=errors,
                message="no baseline recorded; reported accuracy only.",
            )

        result = check_drift(current_acc, baseline, len(y), errors, threshold)
        if result.status == "alert":
            DRIFT_ALERTS.inc()
        return MonitorResponse(**asdict(result))

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "mlops_digit.serve:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
