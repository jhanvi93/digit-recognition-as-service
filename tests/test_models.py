import pytest

from mlops_digit.data import make_splits
from mlops_digit.models import available_models, build_model
from mlops_digit.train import fitting_verdict


def test_available_models_contains_core_types():
    models = available_models()
    for expected in ("svm", "decision_tree", "logistic_regression"):
        assert expected in models


def test_build_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model("not_a_model")


def test_pipeline_trains_and_predicts():
    splits = make_splits(random_state=0)
    model = build_model("decision_tree", {"max_depth": 3, "random_state": 0})
    model.fit(splits.X_train, splits.y_train)
    preds = model.predict(splits.X_val)
    assert len(preds) == len(splits.y_val)


@pytest.mark.parametrize(
    "train_acc,val_acc,expected",
    [
        (0.70, 0.68, "underfit"),
        (1.00, 0.85, "overfit"),
        (0.99, 0.98, "good_fit"),
    ],
)
def test_fitting_verdict(train_acc, val_acc, expected):
    assert fitting_verdict(train_acc, val_acc) == expected
