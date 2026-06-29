import numpy as np

from mlops_digit.data import make_splits
from mlops_digit.models import build_model
from mlops_digit.monitor import check_drift, corrupt, monitor_champion


def test_check_drift_ok_when_within_threshold():
    res = check_drift(0.97, 0.98, n_samples=100, errors=3, threshold=0.05)
    assert res.status == "ok"
    assert res.drop == 0.01


def test_check_drift_alerts_on_large_drop():
    res = check_drift(0.80, 0.98, n_samples=100, errors=20, threshold=0.05)
    assert res.status == "alert"
    assert res.drop > 0.05
    assert "PERFORMANCE DROP" in res.message


def test_corrupt_is_deterministic_and_shaped():
    splits = make_splits(random_state=0)
    X = splits.X_test
    a = corrupt(X, 0.5, random_state=1)
    b = corrupt(X, 0.5, random_state=1)
    assert a.shape == X.shape
    assert np.array_equal(a, b)
    # Zero severity is a no-op copy.
    assert np.array_equal(corrupt(X, 0.0), X)


def test_monitor_flags_degraded_inputs():
    splits = make_splits(random_state=0)
    model = build_model("decision_tree", {"max_depth": 8, "random_state": 0})
    model.fit(splits.X_train, splits.y_train)
    baseline = float(np.mean(model.predict(splits.X_test) == splits.y_test))

    clean = monitor_champion(
        splits.X_test, splits.y_test, model=model, baseline_accuracy=baseline
    )
    assert clean.status == "ok"

    Xc = corrupt(splits.X_test, 1.0, random_state=0)
    drifted = monitor_champion(
        Xc, splits.y_test, model=model, baseline_accuracy=baseline
    )
    assert drifted.status == "alert"
    assert drifted.errors > clean.errors
