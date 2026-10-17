import numpy as np

from fraud.drift import MIN_ROWS, bin_edges, build_reference, drift_report, proportions, psi

rng = np.random.default_rng(0)


def test_identical_distribution_psi_near_zero():
    x = rng.normal(size=20000)
    e = bin_edges(x)
    assert psi(proportions(x, e), proportions(rng.normal(size=20000), e)) < 0.02


def test_shifted_distribution_is_drift():
    x = rng.normal(size=20000)
    e = bin_edges(x)
    assert psi(proportions(x, e), proportions(rng.normal(loc=1.0, size=20000), e)) > 0.25


def test_constant_feature_does_not_crash():
    x = np.full(1000, 3.0)
    p = proportions(x, bin_edges(x))
    assert np.isclose(p.sum(), 1.0) and psi(p, p) == 0.0


def test_empty_input_proportions_do_not_divide_by_zero():
    assert proportions(np.array([]), np.array([0.0])).sum() == 0.0


def test_report_insufficient_data():
    X = rng.normal(size=(1000, 2))
    ref = build_reference(X, ["a", "b"], rng.uniform(size=1000))
    n = MIN_ROWS - 1
    assert drift_report(ref, X[:n], np.zeros(n), np.zeros(n), 0.01) == {
        "status": "insufficient_data",
        "n": n,
    }


def test_report_flags_only_shifted_feature():
    X = rng.normal(size=(5000, 2))
    ref = build_reference(X, ["a", "b"], rng.uniform(size=5000))
    live = rng.normal(size=(500, 2))
    live[:, 1] += 2
    r = drift_report(ref, live, rng.uniform(size=500), np.zeros(500), 0.01)
    assert r["features"]["a"]["status"] == "ok"
    assert r["features"]["b"]["status"] == "drift"
    assert r["status"] == "drift" and r["n"] == 500
    assert r["flag_rate"] == {"live": 0.0, "expected": 0.01}
