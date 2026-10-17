"""Population Stability Index between training reference and live inputs."""

import numpy as np

EPS = 1e-4
MIN_ROWS = 50


def bin_edges(x: np.ndarray, n_bins: int = 10) -> np.ndarray:
    return np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]))


def proportions(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts = np.bincount(np.searchsorted(edges, x, side="right"), minlength=len(edges) + 1)
    return counts / max(len(x), 1)


def psi(ref_props, live_props) -> float:
    r = np.clip(np.asarray(ref_props, dtype=float), EPS, None)
    live = np.clip(np.asarray(live_props, dtype=float), EPS, None)
    return float(np.sum((live - r) * np.log(live / r)))


def status(v: float) -> str:
    return "ok" if v < 0.1 else "watch" if v <= 0.25 else "drift"


def _ref(x: np.ndarray) -> dict:
    e = bin_edges(x)
    return {"edges": e.tolist(), "props": proportions(x, e).tolist()}


def build_reference(X: np.ndarray, names: list[str], scores: np.ndarray) -> dict:
    return {
        "features": {name: _ref(X[:, i]) for i, name in enumerate(names)},
        "score": _ref(scores),
    }


def _check(ref: dict, x: np.ndarray) -> dict:
    v = psi(ref["props"], proportions(x, np.array(ref["edges"])))
    return {"psi": round(v, 4), "status": status(v)}


def drift_report(reference, X, scores, flags, expected_flag_rate) -> dict:
    n = len(X)
    if n < MIN_ROWS:
        return {"status": "insufficient_data", "n": n}
    features = {
        name: _check(ref, X[:, i]) for i, (name, ref) in enumerate(reference["features"].items())
    }
    score = _check(reference["score"], np.asarray(scores))
    statuses = {f["status"] for f in features.values()} | {score["status"]}
    overall = "drift" if "drift" in statuses else "watch" if "watch" in statuses else "ok"
    return {
        "status": overall,
        "n": n,
        "features": features,
        "score": score,
        "flag_rate": {"live": float(np.mean(flags)), "expected": expected_flag_rate},
    }
