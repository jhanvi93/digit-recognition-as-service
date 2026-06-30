"""Proof-of-value experiment: pipeline WITHOUT vs WITH the new module.

This harness answers the dissertation question -

    "Can an automated selection-and-monitoring module choose better-generalising
     models and catch performance degradation faster than manual selection?"

- by running the pipeline two ways on the same data and the same candidate
models, then emitting a side-by-side proof table the examiner can read at a
glance:

  * WITHOUT module - a human hard-picks the ``balanced`` run; there is no
    automated drift monitor, so degraded inputs reach users unflagged.
  * WITH module    - ``selector`` auto-promotes a champion by a
    generalisation-aware rule, and ``monitor`` raises an alert the moment
    accuracy drops below baseline.

Outputs (all regenerated from real runs, never hand-typed):
  * ``artifacts/experiment_results.json``
  * ``reports/proof_table.md``
  * ``reports/figures/proof_table.png``
  * ``reports/figures/drift_sweep.png``

Usage:
    python -m mlops_digit.experiment
    python -m mlops_digit.experiment --dataset mnist --subsample 12000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mlops_digit.data import make_splits  # noqa: E402
from mlops_digit.monitor import check_drift, corrupt  # noqa: E402
from mlops_digit.selector import (  # noqa: E402
    DEFAULT_MAX_GAP,
    MANUAL_CHOICE,
    load_summaries,
    promote,
    select_champion,
)
from mlops_digit.train import train_from_config  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

CANDIDATES = ["underfit", "balanced", "overfit", "logreg", "rf", "knn", "mlp"]
SEVERITIES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
DRIFT_THRESHOLD = 0.05


def train_candidates(dataset: str, subsample: int | None) -> None:
    overrides = {"dataset": dataset, "subsample": subsample}
    for name in CANDIDATES:
        train_from_config(CONFIG_DIR / f"{name}.yaml", data_overrides=overrides)


def drift_sweep(model, X, y, baseline: float) -> list[dict]:
    """Score the model across increasing corruption severities."""
    sweep = []
    for sev in SEVERITIES:
        Xc = corrupt(X, sev) if sev > 0 else X
        preds = model.predict(Xc)
        acc = float(np.mean(preds == y))
        errors = int(np.sum(preds != y))
        result = check_drift(acc, baseline, len(y), errors, DRIFT_THRESHOLD)
        sweep.append(
            {
                "severity": sev,
                "accuracy": round(acc, 4),
                "errors": errors,
                "drop": result.drop,
                "status": result.status,
            }
        )
    return sweep


def first_alert(sweep: list[dict]) -> dict | None:
    for row in sweep:
        if row["severity"] > 0 and row["status"] == "alert":
            return row
    return None


def plot_drift_sweep(sweep: list[dict], baseline: float, out_path: Path) -> None:
    sev = [r["severity"] for r in sweep]
    acc = [r["accuracy"] for r in sweep]
    alert = first_alert(sweep)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sev, acc, "o-", color="#4C72B0", label="Champion accuracy")
    ax.axhline(baseline, ls="--", color="#55A868", label=f"baseline ({baseline:.3f})")
    ax.axhline(
        baseline - DRIFT_THRESHOLD,
        ls=":",
        color="#C44E52",
        label=f"alert threshold ({baseline - DRIFT_THRESHOLD:.3f})",
    )
    if alert is not None:
        ax.axvline(alert["severity"], color="#C44E52", alpha=0.4)
        ax.annotate(
            "first alert",
            xy=(alert["severity"], alert["accuracy"]),
            xytext=(alert["severity"], baseline),
            arrowprops=dict(arrowstyle="->", color="#C44E52"),
            color="#C44E52",
            fontsize=9,
        )
    ax.set_xlabel("Input corruption severity")
    ax.set_ylabel("Accuracy")
    ax.set_title("Monitor: champion accuracy vs induced input drift")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_proof_table(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(rows) + 1.2))
    ax.axis("off")
    table = ax.table(
        cellText=[[r["metric"], r["without_module"], r["with_module"]] for r in rows],
        colLabels=["Metric", "WITHOUT module (manual)", "WITH module (automated)"],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for c in range(3):
        cell = table[0, c]
        cell.set_facecolor("#4C72B0")
        cell.set_text_props(color="white", fontweight="bold")
    ax.set_title("Proof of value: manual vs automated pipeline", pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_markdown(rows: list[dict], meta: dict, out_path: Path) -> None:
    lines = [
        "# Proof of value: manual vs automated pipeline",
        "",
        f"Dataset: **{meta['dataset']}** "
        f"(train/val/test = {meta['n_train']}/{meta['n_val']}/{meta['n_test']}, "
        f"{meta['n_features']} features)",
        "",
        "| Metric | WITHOUT module (manual) | WITH module (automated) |",
        "|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['metric']} | {r['without_module']} | {r['with_module']} |")
    lines += [
        "",
        f"Selection rule: `{meta['selection_rule']}`",
        f"Drift threshold: {DRIFT_THRESHOLD} (alert if accuracy falls more than "
        f"this below baseline).",
        "",
        "_Figures: `reports/figures/proof_table.png`, "
        "`reports/figures/drift_sweep.png`._",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run(dataset: str = "digits", subsample: int | None = None, retrain: bool = True) -> dict:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if retrain:
        train_candidates(dataset, subsample)

    candidates = load_summaries()
    by_name = {c.run_name: c for c in candidates}
    if MANUAL_CHOICE not in by_name:
        raise RuntimeError(f"Manual baseline '{MANUAL_CHOICE}' not found among runs.")

    # --- Treatment: automated selection + promotion ---
    decision = select_champion(candidates, max_gap=DEFAULT_MAX_GAP)
    champ_meta = promote(decision, candidates)

    manual = by_name[MANUAL_CHOICE]
    auto = by_name[decision.champion]

    # --- Drift experiment on the served (champion) model ---
    splits = make_splits(dataset=dataset, subsample=subsample)
    model = joblib.load(MODELS_DIR / "champion.joblib")
    baseline = float(champ_meta["baseline_accuracy"])
    sweep = drift_sweep(model, splits.X_test, splits.y_test, baseline)
    alert = first_alert(sweep)

    manual_acc = manual.metrics["test_accuracy"]
    auto_acc = auto.metrics["test_accuracy"]
    acc_delta = auto_acc - manual_acc

    detect_sev = f"{alert['severity']:.1f}" if alert else "n/a"
    errors_caught = alert["errors"] if alert else 0

    rows = [
        {
            "metric": "Model served",
            "without_module": f"{manual.run_name} ({manual.model_type}) - hand-picked",
            "with_module": f"{auto.run_name} ({auto.model_type}) - auto-selected",
        },
        {
            "metric": "Test accuracy",
            "without_module": f"{manual_acc:.4f}",
            "with_module": f"{auto_acc:.4f}  ({acc_delta:+.4f})",
        },
        {
            "metric": "Train-val gap (generalisation)",
            "without_module": f"{manual.metrics['train_val_gap']:+.4f}",
            "with_module": f"{auto.metrics['train_val_gap']:+.4f}",
        },
        {
            "metric": "Selection effort",
            "without_module": "manual review of all runs",
            "with_module": f"{decision.selection_seconds * 1000:.2f} ms (automatic)",
        },
        {
            "metric": "Drift detection",
            "without_module": "none (manual / reactive)",
            "with_module": f"alerts at severity {detect_sev}",
        },
        {
            "metric": "Errors caught before users",
            "without_module": "0 (no monitor)",
            "with_module": f"{errors_caught} flagged at first alert",
        },
    ]

    meta = {
        "dataset": dataset,
        "selection_rule": decision.rule,
        **{k: splits.summary()[k] for k in ("n_train", "n_val", "n_test", "n_features")},
    }

    results = {
        "dataset": dataset,
        "subsample": subsample,
        "question": (
            "Can an automated selection-and-monitoring module choose "
            "better-generalising models and catch performance degradation "
            "faster than manual selection?"
        ),
        "manual": {
            "run_name": manual.run_name,
            "model_type": manual.model_type,
            "metrics": manual.metrics,
        },
        "automated": {
            "run_name": auto.run_name,
            "model_type": auto.model_type,
            "metrics": auto.metrics,
            "selection_seconds": decision.selection_seconds,
            "selection_rule": decision.rule,
            "gap_constraint_relaxed": decision.gap_constraint_relaxed,
        },
        "accuracy_delta": acc_delta,
        "ranking": decision.ranking,
        "drift_threshold": DRIFT_THRESHOLD,
        "drift_sweep": sweep,
        "first_alert": alert,
        "proof_table": rows,
    }

    with open(ARTIFACTS_DIR / "experiment_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    plot_proof_table(rows, FIGURES_DIR / "proof_table.png")
    plot_drift_sweep(sweep, baseline, FIGURES_DIR / "drift_sweep.png")
    write_markdown(rows, meta, REPORTS_DIR / "proof_table.md")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the manual-vs-automated proof experiment.")
    parser.add_argument("--dataset", default="digits", choices=["digits", "mnist"])
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument(
        "--no-retrain",
        action="store_true",
        help="Reuse existing run metrics instead of retraining candidates.",
    )
    args = parser.parse_args()

    results = run(
        dataset=args.dataset,
        subsample=args.subsample,
        retrain=not args.no_retrain,
    )

    print("\n=== PROOF TABLE: manual vs automated ===")
    header = f"{'Metric':<34}{'WITHOUT module':<34}{'WITH module'}"
    print(header)
    print("-" * len(header))
    for r in results["proof_table"]:
        print(f"{r['metric']:<34}{r['without_module']:<34}{r['with_module']}")
    print(
        f"\nResults: artifacts/experiment_results.json | "
        f"reports/proof_table.md | reports/figures/*.png"
    )


if __name__ == "__main__":
    main()
