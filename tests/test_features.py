import numpy as np

from fraud.features import FEATURES, to_matrix


def test_to_matrix_order_and_log_amount():
    row = {f"V{i}": float(i) for i in range(1, 29)} | {"Amount": np.e - 1, "Time": 5.0}
    X = to_matrix([row])
    assert X.shape == (1, len(FEATURES)) == (1, 29)
    assert X[0, 0] == 1.0 and X[0, 27] == 28.0
    assert np.isclose(X[0, 28], 1.0)  # log1p(e - 1) == 1; Time is ignored
