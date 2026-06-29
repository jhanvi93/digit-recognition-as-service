import joblib
import pytest
from fastapi.testclient import TestClient

from mlops_digit.data import N_FEATURES, make_splits
from mlops_digit.models import build_model


@pytest.fixture()
def client(tmp_path, monkeypatch):
    splits = make_splits(random_state=0)
    model = build_model("decision_tree", {"max_depth": 6, "random_state": 0})
    model.fit(splits.X_train, splits.y_train)
    model_path = tmp_path / "test_model.joblib"
    joblib.dump(model, model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    # Import after env var is set so the app picks up the right model path.
    from mlops_digit.serve import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def test_livez(client):
    r = client.get("/livez")
    assert r.status_code == 200
    assert r.json()["status"] == "alive"


def test_readyz(client):
    r = client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_metrics(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "digit_predictions_total" in r.text


def test_predict_returns_labels(client):
    splits = make_splits(random_state=0)
    payload = {"instances": splits.X_test[:3].tolist()}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 3
    for pred in body["predictions"]:
        assert 0 <= pred["label"] <= 9


def test_predict_rejects_bad_shape(client):
    r = client.post("/predict", json={"instances": [[0.0] * (N_FEATURES - 1)]})
    assert r.status_code == 422


def test_monitor_reports_accuracy(client):
    splits = make_splits(random_state=0)
    payload = {
        "instances": splits.X_test[:20].tolist(),
        "labels": splits.y_test[:20].tolist(),
    }
    r = client.post("/monitor", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["current_accuracy"] <= 1.0
    assert body["n_samples"] == 20
    assert "status" in body


def test_monitor_rejects_length_mismatch(client):
    splits = make_splits(random_state=0)
    payload = {
        "instances": splits.X_test[:5].tolist(),
        "labels": splits.y_test[:4].tolist(),
    }
    r = client.post("/monitor", json=payload)
    assert r.status_code == 422
