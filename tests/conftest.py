import json

import lightgbm as lgb
import numpy as np
import pytest
from fastapi.testclient import TestClient

from fraud.drift import build_reference
from fraud.features import FEATURES


@pytest.fixture
def client(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, len(FEATURES)))
    y = (X[:, 0] > 1.5).astype(int)
    booster = lgb.train(
        {"objective": "binary", "verbose": -1}, lgb.Dataset(X, y), num_boost_round=10
    )
    booster.save_model(str(tmp_path / "model.txt"))
    config = {
        "model_version": "test",
        "features": FEATURES,
        "thresholds": {"cost": 0.5, "recall": 0.2, "precision": 0.8},
        "expected_flag_rate": 0.07,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    ref = build_reference(X, FEATURES, booster.predict(X))
    (tmp_path / "reference.json").write_text(json.dumps(ref))
    curves = {"pr": {"precision": [1.0], "recall": [0.0]}, "sweep": []}
    (tmp_path / "curves.json").write_text(json.dumps(curves))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "logs" / "p.db"))

    from api.main import app

    with TestClient(app) as c:
        yield c
