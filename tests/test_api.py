import json

import numpy as np


def txn(**over):
    d = {f"V{i}": 0.0 for i in range(1, 29)} | {"Amount": 10.0}
    return d | over


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "model_version": "test"}


def test_predict_shape_and_default_policy(client):
    body = client.post("/predict", json=txn()).json()
    assert set(body) == {"fraud_probability", "is_fraud", "threshold", "policy", "model_version"}
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["policy"] == "cost" and body["threshold"] == 0.5


def test_predict_recall_policy_uses_its_threshold(client):
    assert client.post("/predict?policy=recall", json=txn()).json()["threshold"] == 0.2


def test_missing_field_is_422(client):
    d = txn()
    del d["V3"]
    assert client.post("/predict", json=d).status_code == 422


def test_bad_policy_is_422(client):
    assert client.post("/predict?policy=yolo", json=txn()).status_code == 422


def test_negative_amount_is_422(client):
    assert client.post("/predict", json=txn(Amount=-5.0)).status_code == 422


def test_nan_is_422(client):
    raw = json.dumps(txn(V1=float("nan")))  # Python emits a bare NaN token
    r = client.post("/predict", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_batch_ok_and_empty(client):
    assert len(client.post("/predict/batch", json=[txn()] * 3).json()) == 3
    r = client.post("/predict/batch", json=[])
    assert r.status_code == 200 and r.json() == []


def test_batch_too_large_is_413(client):
    assert client.post("/predict/batch", json=[txn()] * 1001).status_code == 413


def test_drift_before_traffic_is_insufficient(client):
    assert client.get("/drift").json() == {"status": "insufficient_data", "n": 0}


def test_drift_detects_shifted_traffic(client):
    rng = np.random.default_rng(1)
    rows = [
        txn(**{f"V{i}": float(rng.normal()) for i in range(1, 29)}) | {"V1": 5.0}
        for _ in range(100)
    ]
    client.post("/predict/batch", json=rows)
    report = client.get("/drift?n=100").json()
    assert report["n"] == 100 and len(report["features"]) == 29
    assert report["features"]["V1"]["status"] == "drift" and report["status"] == "drift"


def test_curves(client):
    assert client.get("/curves").json()["pr"]["precision"] == [1.0]
