import numpy as np

from fraud.threshold import pick_thresholds, sweep

# 4 legit, 2 fraud. Hand-computed: any t in (0.6, 0.7] gives fp=0, fn=0.
Y = np.array([0, 0, 0, 0, 1, 1])
S = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])


def test_cost_policy_picks_zero_cost_threshold_highest_on_tie():
    picked = pick_thresholds(Y, S, fn_cost=10, fp_cost=1)
    assert picked["thresholds"]["cost"] == 0.70


def test_recall_policy_is_highest_threshold_meeting_target():
    assert pick_thresholds(Y, S, fn_cost=10, fp_cost=1)["thresholds"]["recall"] == 0.70


def test_precision_policy_is_lowest_threshold_meeting_target():
    assert pick_thresholds(Y, S, fn_cost=10, fp_cost=1)["thresholds"]["precision"] == 0.61


def test_sweep_row_values():
    row = next(r for r in sweep(Y, S, 10, 1) if r["threshold"] == 0.5)
    assert (row["tp"], row["fp"], row["fn"], row["tn"]) == (2, 1, 0, 3)
    assert np.isclose(row["precision"], 2 / 3) and row["recall"] == 1.0 and row["cost"] == 1


def test_unreachable_precision_falls_back_with_warning():
    # Best achievable precision is 0.5 (at t <= 0.1), so 0.90 is unreachable.
    picked = pick_thresholds(np.array([0, 1]), np.array([0.9, 0.1]), 10, 1)
    assert picked["thresholds"]["precision"] == 0.01
    assert any("precision" in w for w in picked["warnings"])
