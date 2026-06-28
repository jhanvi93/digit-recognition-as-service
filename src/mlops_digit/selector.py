"""Automatic model selection - the "champion picker".

This is the project's original contribution. Instead of a human eyeballing the
runs and hard-coding ``models/balanced.joblib`` into the service, the selector
reads every training run's metrics, applies a transparent, generalisation-aware
rule, and *promotes* a single champion that the serving layer then loads.

Selection rule (deliberately simple and auditable):
    among candidates whose train/val gap <= ``max_gap`` (i.e. not over-fit),
    pick the highest validation accuracy, breaking ties on test accuracy.
If no candidate satisfies the gap constraint we fall back to the best
validation accuracy overall and flag the decision as ``gap_constraint_relaxed``.

The manual baseline that this automation replaces is the historically
hand-picked ``balanced`` run - exposed here as ``MANUAL_CHOICE`` so the
experiment harness can compare "manual vs automated" head to head.

Usage:
    python -m mlops_digit.selector
    python -m mlops_digit.selector --max-gap 0.05 --metric val_accuracy
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"
MODELS_DIR = PROJECT_ROOT / "models"
CHAMPION_MODEL = MODELS_DIR / "champion.joblib"
CHAMPION_META = MODELS_DIR / "champion.json"

# The run a human would historically have picked by hand.
MANUAL_CHOICE = "balanced"
DEFAULT_MAX_GAP = 0.08
DEFAULT_METRIC = "val_accuracy"


@dataclass
class Candidate:
    run_name: str
    model_type: str
    dataset: str
    metrics: dict[str, float]
    fitting_verdict: str
    model_path: Path

    @property
    def gap(self) -> float:
        return float(self.metrics.get("train_val_gap", 0.0))

    def score(self, metric: str) -> float:
        return float(self.metrics.get(metric, 0.0))


@dataclass
class Decision:
    champion: str
    metric: str
    max_gap: float
    rule: str
    gap_constraint_relaxed: bool
    ranking: list[dict[str, Any]] = field(default_factory=list)
    selection_seconds: float = 0.0


def load_summaries(metrics_dir: Path = METRICS_DIR) -> list[Candidate]:
    """Load every per-run metrics summary written by ``train.py``."""
    candidates: list[Candidate] = []
    for path in sorted(metrics_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        run_name = data["run_name"]
        candidates.append(
            Candidate(
                run_name=run_name,
                model_type=data.get("model_type", "unknown"),
                dataset=data.get("dataset", "digits"),
                metrics=data.get("metrics", {}),
                fitting_verdict=data.get("fitting_verdict", "unknown"),
                model_path=MODELS_DIR / f"{run_name}.joblib",
            )
        )
    return candidates


def select_champion(
    candidates: list[Candidate],
    max_gap: float = DEFAULT_MAX_GAP,
    metric: str = DEFAULT_METRIC,
) -> Decision:
    """Apply the generalisation-aware selection rule to rank candidates."""
    if not candidates:
        raise ValueError("No candidate runs found. Train some models first.")

    start = time.perf_counter()
    eligible = [c for c in candidates if c.gap <= max_gap]
    relaxed = not eligible
    pool = eligible or candidates

    ordered = sorted(
        pool,
        key=lambda c: (c.score(metric), c.score("test_accuracy")),
        reverse=True,
    )
    winner = ordered[0]
    elapsed = time.perf_counter() - start

    ranking = [
        {
            "run_name": c.run_name,
            "model_type": c.model_type,
            metric: round(c.score(metric), 4),
            "test_accuracy": round(c.score("test_accuracy"), 4),
            "train_val_gap": round(c.gap, 4),
            "fitting_verdict": c.fitting_verdict,
            "eligible": c.gap <= max_gap,
        }
        for c in sorted(candidates, key=lambda c: c.score(metric), reverse=True)
    ]

    rule = (
        f"max({metric}) subject to train_val_gap <= {max_gap:g}; "
        f"tie-break on test_accuracy"
    )
    return Decision(
        champion=winner.run_name,
        metric=metric,
        max_gap=max_gap,
        rule=rule,
        gap_constraint_relaxed=relaxed,
        ranking=ranking,
        selection_seconds=elapsed,
    )


def promote(decision: Decision, candidates: list[Candidate]) -> dict[str, Any]:
    """Copy the chosen model to ``champion.joblib`` and write its metadata."""
    by_name = {c.run_name: c for c in candidates}
    winner = by_name[decision.champion]
    if not winner.model_path.exists():
        raise FileNotFoundError(
            f"Champion model artifact missing: {winner.model_path}. "
            "Re-run training for that config."
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(winner.model_path, CHAMPION_MODEL)

    meta = {
        "champion": winner.run_name,
        "model_type": winner.model_type,
        "dataset": winner.dataset,
        # Baseline = clean test accuracy; the monitor alerts on drops from this.
        "baseline_accuracy": float(winner.metrics.get("test_accuracy", 0.0)),
        "metrics": winner.metrics,
        "fitting_verdict": winner.fitting_verdict,
        "selection_rule": decision.rule,
        "selection_metric": decision.metric,
        "max_gap": decision.max_gap,
        "gap_constraint_relaxed": decision.gap_constraint_relaxed,
        "selection_seconds": decision.selection_seconds,
        "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_model": str(winner.model_path),
    }
    with open(CHAMPION_META, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def _print_decision(decision: Decision, meta: dict[str, Any]) -> None:
    header = (
        f"{'run':<14}{'model':<20}{decision.metric:>10}"
        f"{'test':>8}{'gap':>9}{'elig':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in decision.ranking:
        marker = "*" if row["run_name"] == decision.champion else " "
        print(
            f"{marker}{row['run_name']:<13}{row['model_type']:<20}"
            f"{row[decision.metric]:>10.3f}{row['test_accuracy']:>8.3f}"
            f"{row['train_val_gap']:>+9.3f}{('Y' if row['eligible'] else 'n'):>6}"
        )
    relaxed = " (gap constraint relaxed)" if decision.gap_constraint_relaxed else ""
    print(
        f"\nChampion: {decision.champion} "
        f"[{meta['model_type']}] baseline_acc={meta['baseline_accuracy']:.4f}"
        f"{relaxed}"
    )
    print(f"Rule: {decision.rule}")
    print(f"Selected in {decision.selection_seconds * 1000:.2f} ms -> {CHAMPION_MODEL}")
    if decision.champion != MANUAL_CHOICE:
        print(
            f"NOTE: automated choice ({decision.champion}) differs from the "
            f"manual baseline ({MANUAL_CHOICE})."
        )


def run(max_gap: float = DEFAULT_MAX_GAP, metric: str = DEFAULT_METRIC) -> dict[str, Any]:
    candidates = load_summaries()
    decision = select_champion(candidates, max_gap=max_gap, metric=metric)
    meta = promote(decision, candidates)
    return {"decision": decision, "meta": meta}


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-select and promote a champion model.")
    parser.add_argument("--max-gap", type=float, default=DEFAULT_MAX_GAP)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    args = parser.parse_args()

    candidates = load_summaries()
    decision = select_champion(candidates, max_gap=args.max_gap, metric=args.metric)
    meta = promote(decision, candidates)
    _print_decision(decision, meta)


if __name__ == "__main__":
    main()
