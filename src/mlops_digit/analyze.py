"""Under/overfitting analysis across the three contrasted runs.

Trains (or re-uses) the underfit / balanced / overfit configs, then produces:
  * a grouped bar chart of train vs validation vs test accuracy,
  * a train/validation-gap chart,
  * a learning curve for the balanced model,
and writes a machine-readable comparison table.

Usage:
    python -m mlops_digit.analyze
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / CI-safe backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.model_selection import learning_curve  # noqa: E402

from mlops_digit.data import make_splits  # noqa: E402
from mlops_digit.models import build_model  # noqa: E402
from mlops_digit.train import load_config, train_from_config  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RUN_ORDER = ["underfit", "balanced", "overfit"]


def run_all() -> list[dict]:
    summaries = []
    for name in RUN_ORDER:
        summaries.append(train_from_config(CONFIG_DIR / f"{name}.yaml"))
    return summaries


def plot_accuracy_comparison(summaries: list[dict], out_path: Path) -> None:
    names = [s["run_name"] for s in summaries]
    train = [s["metrics"]["train_accuracy"] for s in summaries]
    val = [s["metrics"]["val_accuracy"] for s in summaries]
    test = [s["metrics"]["test_accuracy"] for s in summaries]

    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, train, width, label="Train", color="#4C72B0")
    ax.bar(x, val, width, label="Validation", color="#DD8452")
    ax.bar(x + width, test, width, label="Test", color="#55A868")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Train / Validation / Test accuracy by run")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    for i, v in enumerate(train):
        ax.text(i - width, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(val):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(test):
        ax.text(i + width, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_gap(summaries: list[dict], out_path: Path) -> None:
    names = [s["run_name"] for s in summaries]
    gap = [s["metrics"]["train_val_gap"] for s in summaries]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#C44E52" if g > 0.08 else "#55A868" for g in gap]
    ax.bar(names, gap, color=colors)
    ax.axhline(0.08, ls="--", color="gray", label="overfit threshold (0.08)")
    ax.set_ylabel("Train - Validation accuracy gap")
    ax.set_title("Generalisation gap by run")
    ax.legend()
    for i, v in enumerate(gap):
        ax.text(i, v + 0.005, f"{v:+.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_learning_curve(out_path: Path) -> None:
    cfg = load_config(CONFIG_DIR / "balanced.yaml")
    splits = make_splits(
        test_size=cfg["data"]["test_size"],
        val_size=cfg["data"]["val_size"],
        random_state=cfg["data"]["random_state"],
    )
    X = np.vstack([splits.X_train, splits.X_val])
    y = np.concatenate([splits.y_train, splits.y_val])
    model = build_model(cfg["model"]["type"], cfg["model"]["params"])

    sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        train_sizes=np.linspace(0.1, 1.0, 6),
        cv=5,
        scoring="accuracy",
        random_state=42,
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, train_mean, "o-", color="#4C72B0", label="Train accuracy")
    ax.plot(sizes, val_mean, "o-", color="#DD8452", label="CV accuracy")
    ax.fill_between(
        sizes,
        train_scores.min(axis=1),
        train_scores.max(axis=1),
        alpha=0.1,
        color="#4C72B0",
    )
    ax.fill_between(
        sizes, val_scores.min(axis=1), val_scores.max(axis=1), alpha=0.1, color="#DD8452"
    )
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning curve - balanced model (RBF SVM)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    summaries = run_all()
    plot_accuracy_comparison(summaries, FIGURES_DIR / "accuracy_comparison.png")
    plot_gap(summaries, FIGURES_DIR / "generalisation_gap.png")
    plot_learning_curve(FIGURES_DIR / "learning_curve.png")

    comparison = {
        s["run_name"]: {
            "model_type": s["model_type"],
            **s["metrics"],
            "fitting_verdict": s["fitting_verdict"],
        }
        for s in summaries
    }
    with open(ARTIFACTS_DIR / "comparison.json", "w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2)

    header = f"{'run':<10}{'model':<16}{'train':>8}{'val':>8}{'test':>8}{'gap':>9}  verdict"
    print(header)
    print("-" * len(header))
    for name, m in comparison.items():
        print(
            f"{name:<10}{m['model_type']:<16}"
            f"{m['train_accuracy']:>8.3f}{m['val_accuracy']:>8.3f}"
            f"{m['test_accuracy']:>8.3f}{m['train_val_gap']:>+9.3f}  {m['fitting_verdict']}"
        )
    print(f"\nFigures written to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
