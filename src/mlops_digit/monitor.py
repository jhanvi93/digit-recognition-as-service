"""Performance-drop / drift monitoring for the promoted champion.

The second half of the original contribution. Once a champion is serving, this
module watches its accuracy on freshly labelled batches and raises an **alert**
when accuracy falls more than ``threshold`` below the champion's recorded
baseline. That alert is the signal an operator (or an automated rollback) acts
on - the thing manual, eyeball-only monitoring tends to miss until users
complain.

It provides:
  * ``accuracy`` / ``check_drift`` - the pure detection logic (unit-testable),
  * ``corrupt`` - a reproducible input-degradation helper used to *induce* a
    drop so we can prove the monitor catches it,
  * a CLI that scores the champion on clean (and optionally corrupted) data,
    writes a JSON report, and exits non-zero on alert (CI / Alertmanager hook).

Usage:
    python -m mlops_digit.monitor
    python -m mlops_digit.monitor --corrupt 0.6 --threshold 0.05
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score

from mlops_digit.data import make_splits

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
CHAMPION_MODEL = MODELS_DIR / "champion.joblib"
CHAMPION_META = MODELS_DIR / "champion.json"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_THRESHOLD = 0.05


@dataclass
class DriftResult:
    status: str  # "ok" | "alert"
    current_accuracy: float
    baseline_accuracy: float
    drop: float
    threshold: float
    n_samples: int
    errors: int  # misclassifications in this batch (errors caught early)
    message: str


def check_drift(
    current_accuracy: float,
    baseline_accuracy: float,
    n_samples: int,
    errors: int,
    threshold: float = DEFAULT_THRESHOLD,
) -> DriftResult:
    """Compare live accuracy to the baseline and decide whether to alert."""
    drop = baseline_accuracy - current_accuracy
    alert = drop > threshold
    status = "alert" if alert else "ok"
    if alert:
        message = (
            f"PERFORMANCE DROP: accuracy fell {drop:.3f} below baseline "
            f"({current_accuracy:.3f} < {baseline_accuracy:.3f} - {threshold:g})."
        )
    else:
        message = (
            f"healthy: accuracy {current_accuracy:.3f} within {threshold:g} "
            f"of baseline {baseline_accuracy:.3f}."
        )
    return DriftResult(
        status=status,
        current_accuracy=round(current_accuracy, 4),
        baseline_accuracy=round(baseline_accuracy, 4),
        drop=round(drop, 4),
        threshold=threshold,
        n_samples=n_samples,
        errors=errors,
        message=message,
    )


def corrupt(
    X: np.ndarray,
    severity: float,
    random_state: int = 42,
) -> np.ndarray:
    """Return a degraded copy of ``X`` to simulate input drift.

    ``severity`` in [0, 1] scales additive Gaussian noise relative to each
    feature's spread, emulating sensor noise / distribution shift at serving
    time. Deterministic for a given ``random_state`` so experiments reproduce.
    """
    if severity <= 0:
        return X.copy()
    rng = np.random.default_rng(random_state)
    scale = float(np.std(X)) * severity
    noisy = X + rng.normal(0.0, scale, size=X.shape)
    return np.clip(noisy, X.min(), X.max())


def monitor_champion(
    X: np.ndarray,
    y: np.ndarray,
    model=None,
    baseline_accuracy: float | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> DriftResult:
    """Score the (champion) model on a labelled batch and run the drift check."""
    if model is None:
        if not CHAMPION_MODEL.exists():
            raise FileNotFoundError(
                f"No champion at {CHAMPION_MODEL}. Run `python -m mlops_digit.selector`."
            )
        model = joblib.load(CHAMPION_MODEL)
    if baseline_accuracy is None:
        baseline_accuracy = _load_baseline()

    preds = model.predict(X)
    acc = float(accuracy_score(y, preds))
    errors = int(np.sum(preds != y))
    return check_drift(acc, baseline_accuracy, len(y), errors, threshold)


def _load_baseline() -> float:
    if CHAMPION_META.exists():
        with open(CHAMPION_META, "r", encoding="utf-8") as fh:
            return float(json.load(fh).get("baseline_accuracy", 0.0))
    return 0.0


def _load_dataset_for_champion() -> str:
    if CHAMPION_META.exists():
        with open(CHAMPION_META, "r", encoding="utf-8") as fh:
            return json.load(fh).get("dataset", "digits")
    return "digits"


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor the champion for performance drops.")
    parser.add_argument(
        "--corrupt",
        type=float,
        default=0.0,
        help="Input-degradation severity in [0,1] to simulate drift (default 0).",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    dataset = _load_dataset_for_champion()
    splits = make_splits(dataset=dataset)
    X, y = splits.X_test, splits.y_test
    if args.corrupt > 0:
        X = corrupt(X, args.corrupt, random_state=args.random_state)

    result = monitor_champion(X, y, threshold=args.threshold)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS_DIR / "monitor_result.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"corrupt_severity": args.corrupt, **asdict(result)}, fh, indent=2)

    tag = "ALERT" if result.status == "alert" else "OK"
    print(f"[{tag}] {result.message}")
    print(
        f"samples={result.n_samples} errors={result.errors} "
        f"current={result.current_accuracy:.4f} baseline={result.baseline_accuracy:.4f} "
        f"drop={result.drop:+.4f} threshold={result.threshold:g}"
    )
    print(f"Report written to: {out}")
    raise SystemExit(1 if result.status == "alert" else 0)


if __name__ == "__main__":
    main()
