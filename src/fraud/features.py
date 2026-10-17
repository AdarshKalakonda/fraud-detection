"""Single source of truth for turning raw transactions into model input."""

import numpy as np

V_COLS = [f"V{i}" for i in range(1, 29)]
FEATURES = V_COLS + ["log_amount"]


def to_matrix(rows: list[dict]) -> np.ndarray:
    """Raw rows (V1..V28, Amount; extra keys ignored) -> (n, 29) float matrix."""
    return np.array(
        [[r[c] for c in V_COLS] + [np.log1p(r["Amount"])] for r in rows], dtype=float
    ).reshape(len(rows), len(FEATURES))
