"""Dataset loading and reproducible, stratified splitting.

Two datasets are supported through a single config-driven switch:

* ``digits`` - the scikit-learn 8x8 handwritten-digits set (1,797 samples,
  64 features). Bundled with scikit-learn, so it runs fully offline on CPU.
* ``mnist``  - the classic MNIST set (70,000 samples, 28x28 = 784 features)
  fetched once via ``fetch_openml`` and cached under ``data/``.

The dataset and an optional stratified ``subsample`` are selected per run so
the same pipeline can scale from the tiny demo set to full MNIST without code
changes. ``N_FEATURES`` / ``IMAGE_SHAPE`` / ``N_CLASSES`` remain exported as the
*digits* defaults for backwards compatibility with the serving layer and tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml, load_digits
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_HOME = PROJECT_ROOT / "data"

# Static spec for each supported dataset.
DATASET_SPECS: dict[str, dict] = {
    "digits": {"n_features": 64, "image_shape": (8, 8), "n_classes": 10},
    "mnist": {"n_features": 784, "image_shape": (28, 28), "n_classes": 10},
}
DEFAULT_DATASET = "digits"

# Backwards-compatible module constants (digits defaults).
N_FEATURES = DATASET_SPECS[DEFAULT_DATASET]["n_features"]
IMAGE_SHAPE = DATASET_SPECS[DEFAULT_DATASET]["image_shape"]
N_CLASSES = DATASET_SPECS[DEFAULT_DATASET]["n_classes"]


def dataset_spec(dataset: str = DEFAULT_DATASET) -> dict:
    """Return the ``{n_features, image_shape, n_classes}`` spec for a dataset."""
    if dataset not in DATASET_SPECS:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Available: {', '.join(DATASET_SPECS)}"
        )
    return DATASET_SPECS[dataset]


@dataclass
class DataSplits:
    """Container for stratified train / validation / test splits."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    dataset: str = DEFAULT_DATASET

    def summary(self) -> dict:
        return {
            "dataset": self.dataset,
            "n_train": int(len(self.y_train)),
            "n_val": int(len(self.y_val)),
            "n_test": int(len(self.y_test)),
            "n_features": int(self.X_train.shape[1]),
            "n_classes": int(len(np.unique(self.y_train))),
        }


def load_raw(dataset: str = DEFAULT_DATASET) -> tuple[np.ndarray, np.ndarray]:
    """Return the full ``(X, y)`` feature matrix and labels for ``dataset``.

    ``mnist`` is downloaded once via OpenML and cached under ``data/``; every
    later call reads from the local cache, keeping subsequent runs offline.
    """
    if dataset not in DATASET_SPECS:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Available: {', '.join(DATASET_SPECS)}"
        )

    if dataset == "digits":
        digits = load_digits()
        X = digits.data.astype(np.float64)
        y = digits.target.astype(np.int64)
        return X, y

    DATA_HOME.mkdir(parents=True, exist_ok=True)
    bunch = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        data_home=str(DATA_HOME),
        parser="auto",
    )
    X = np.asarray(bunch.data, dtype=np.float64)
    y = np.asarray(bunch.target).astype(np.int64)
    return X, y


def make_splits(
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    dataset: str = DEFAULT_DATASET,
    subsample: int | None = None,
) -> DataSplits:
    """Create stratified train/val/test splits for the chosen dataset.

    ``val_size`` is expressed as a fraction of the *remaining* data after the
    test split has been carved out, so the three sets are disjoint.
    ``subsample`` (optional) draws a stratified subset of the full dataset
    *before* splitting - useful to keep kernel methods tractable on MNIST.
    """
    X, y = load_raw(dataset)

    if subsample is not None and 0 < subsample < len(y):
        X, _, y, _ = train_test_split(
            X, y, train_size=subsample, stratify=y, random_state=random_state
        )

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size,
        stratify=y_temp,
        random_state=random_state,
    )
    return DataSplits(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        dataset=dataset,
    )
