import pandas as pd
import pytest

from ui import client
from ui.client import REQUIRED, prepare_rows


def frame(n=3, **over):
    df = pd.DataFrame({c: [1.0] * n for c in REQUIRED})
    for k, v in over.items():
        df[k] = v
    return df


def test_extra_columns_dropped():
    rows = prepare_rows(frame(Class=[0, 1, 0], Time=[1, 2, 3]))
    assert len(rows) == 3 and set(rows[0]) == set(REQUIRED)


def test_missing_columns_named():
    with pytest.raises(ValueError, match="V3"):
        prepare_rows(frame().drop(columns=["V3"]))


def test_non_numeric_and_negative_rejected():
    with pytest.raises(ValueError, match="2 rows"):
        prepare_rows(frame(V1=["x", 1.0, 1.0], Amount=[1.0, -1.0, 1.0]))


def test_predict_batch_chunks_by_1000(monkeypatch):
    sizes = []

    class Resp:
        def __init__(self, n):
            self.n = n

        def raise_for_status(self):
            pass

        def json(self):
            return [{}] * self.n

    def fake_post(url, params, json, timeout):
        sizes.append(len(json))
        return Resp(len(json))

    monkeypatch.setattr(client.requests, "post", fake_post)
    assert len(client.predict_batch([{}] * 2500, "cost")) == 2500
    assert sizes == [1000, 1000, 500]
