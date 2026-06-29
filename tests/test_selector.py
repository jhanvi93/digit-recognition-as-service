from pathlib import Path

from mlops_digit.selector import Candidate, select_champion


def _cand(name, model, val, test, gap, verdict):
    return Candidate(
        run_name=name,
        model_type=model,
        dataset="digits",
        metrics={"val_accuracy": val, "test_accuracy": test, "train_val_gap": gap},
        fitting_verdict=verdict,
        model_path=Path(f"{name}.joblib"),
    )


def test_selects_best_eligible_within_gap():
    candidates = [
        _cand("underfit", "decision_tree", 0.31, 0.30, 0.01, "underfit"),
        _cand("balanced", "svm", 0.99, 0.983, 0.01, "good_fit"),
        _cand("overfit", "decision_tree", 0.83, 0.786, 0.17, "overfit"),
        _cand("rf", "random_forest", 0.992, 0.985, 0.02, "good_fit"),
    ]
    decision = select_champion(candidates, max_gap=0.08)
    # rf has the highest val accuracy among the low-gap candidates.
    assert decision.champion == "rf"
    assert not decision.gap_constraint_relaxed


def test_overfit_excluded_by_gap_constraint():
    candidates = [
        _cand("overfit", "decision_tree", 0.999, 0.80, 0.20, "overfit"),
        _cand("balanced", "svm", 0.99, 0.983, 0.01, "good_fit"),
    ]
    decision = select_champion(candidates, max_gap=0.08)
    # Despite higher val accuracy, the over-fit run is not eligible.
    assert decision.champion == "balanced"


def test_relaxes_constraint_when_none_eligible():
    candidates = [
        _cand("overfit_a", "decision_tree", 0.90, 0.70, 0.20, "overfit"),
        _cand("overfit_b", "decision_tree", 0.95, 0.72, 0.23, "overfit"),
    ]
    decision = select_champion(candidates, max_gap=0.08)
    assert decision.gap_constraint_relaxed
    assert decision.champion == "overfit_b"
