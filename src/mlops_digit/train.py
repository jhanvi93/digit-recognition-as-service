"""Config-driven training with MLflow experiment tracking.

Usage:
    python -m mlops_digit.train --config configs/balanced.yaml

Each run trains one model, evaluates it on train/val/test, logs parameters,
metrics and the serialized pipeline to MLflow, and saves a joblib artifact
under ``models/`` for the serving layer.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import yaml
from sklearn.metrics import accuracy_score, f1_score, log_loss

from mlops_digit.data import make_splits
from mlops_digit.models import build_model

EXPERIMENT_NAME = "digit-recognition"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _safe_log_loss(pipeline, X, y) -> float | None:
    """Compute log loss when the estimator exposes probabilities."""
    if not hasattr(pipeline, "predict_proba"):
        return None
    try:
        proba = pipeline.predict_proba(X)
        return float(log_loss(y, proba, labels=list(range(proba.shape[1]))))
    except Exception:  # noqa: BLE001 - some estimators lack calibrated proba
        return None


def evaluate(pipeline, X, y) -> dict[str, float]:
    preds = pipeline.predict(X)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "f1_macro": float(f1_score(y, preds, average="macro")),
    }


def fitting_verdict(train_acc: float, val_acc: float) -> str:
    """Heuristic under/overfitting label from train and val accuracy."""
    gap = train_acc - val_acc
    if val_acc < 0.80 and train_acc < 0.85:
        return "underfit"
    if gap > 0.08:
        return "overfit"
    return "good_fit"


def train_from_config(
    config_path: str | Path,
    tracking_uri: str | None = None,
    data_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    run_name = config.get("run_name", Path(config_path).stem)
    model_cfg = config.get("model", {})
    data_cfg = dict(config.get("data", {}))
    # Runtime overrides let callers (e.g. the experiment harness) swap the
    # dataset or subsample size without editing the YAML configs.
    if data_overrides:
        data_cfg.update({k: v for k, v in data_overrides.items() if v is not None})

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri((PROJECT_ROOT / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    splits = make_splits(
        test_size=data_cfg.get("test_size", 0.2),
        val_size=data_cfg.get("val_size", 0.2),
        random_state=data_cfg.get("random_state", 42),
        dataset=data_cfg.get("dataset", "digits"),
        subsample=data_cfg.get("subsample"),
    )

    pipeline = build_model(model_cfg.get("type"), model_cfg.get("params"))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"run_name": run_name, "notes": config.get("notes", "")})
        mlflow.log_param("model_type", model_cfg.get("type"))
        for key, value in (model_cfg.get("params") or {}).items():
            mlflow.log_param(f"param.{key}", value)
        for key, value in splits.summary().items():
            mlflow.log_param(f"data.{key}", value)

        start = time.perf_counter()
        pipeline.fit(splits.X_train, splits.y_train)
        fit_seconds = time.perf_counter() - start

        train_m = evaluate(pipeline, splits.X_train, splits.y_train)
        val_m = evaluate(pipeline, splits.X_val, splits.y_val)
        test_m = evaluate(pipeline, splits.X_test, splits.y_test)

        train_acc = train_m["accuracy"]
        val_acc = val_m["accuracy"]
        gap = train_acc - val_acc
        verdict = fitting_verdict(train_acc, val_acc)

        metrics = {
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "test_accuracy": test_m["accuracy"],
            "train_f1_macro": train_m["f1_macro"],
            "val_f1_macro": val_m["f1_macro"],
            "test_f1_macro": test_m["f1_macro"],
            "train_val_gap": gap,
            "fit_seconds": fit_seconds,
        }
        val_loss = _safe_log_loss(pipeline, splits.X_val, splits.y_val)
        if val_loss is not None:
            metrics["val_log_loss"] = val_loss

        mlflow.log_metrics(metrics)
        mlflow.set_tag("fitting_verdict", verdict)

        model_path = MODELS_DIR / f"{run_name}.joblib"
        joblib.dump(pipeline, model_path)
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            input_example=splits.X_val[:2],
        )
        mlflow.log_artifact(str(model_path))

        summary = {
            "run_name": run_name,
            "run_id": run.info.run_id,
            "model_type": model_cfg.get("type"),
            "params": model_cfg.get("params"),
            "dataset": data_cfg.get("dataset", "digits"),
            "subsample": data_cfg.get("subsample"),
            "metrics": metrics,
            "fitting_verdict": verdict,
        }
        metrics_path = METRICS_DIR / f"{run_name}.json"
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        mlflow.log_artifact(str(metrics_path))

    print(
        f"[{run_name}] train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
        f"test_acc={test_m['accuracy']:.4f} gap={gap:+.4f} -> {verdict}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a digit-recognition model.")
    parser.add_argument("--config", required=True, help="Path to a YAML run config.")
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Optional MLflow tracking URI (defaults to local ./mlruns).",
    )
    args = parser.parse_args()
    train_from_config(args.config, tracking_uri=args.tracking_uri)


if __name__ == "__main__":
    main()
