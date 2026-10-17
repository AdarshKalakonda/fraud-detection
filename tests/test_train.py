import pandas as pd

from fraud.train import passes_gate, time_split


def test_gate_requires_floor_and_beating_baseline():
    assert passes_gate(0.749, 0.690)
    assert not passes_gate(0.69, 0.60)  # below the 0.70 floor
    assert not passes_gate(0.72, 0.75)  # worse than baseline


def test_time_split_is_ordered_and_sized():
    df = pd.DataFrame({"Time": list(range(100))[::-1], "Class": [0] * 100})
    tr, va, te = time_split(df)
    assert (len(tr), len(va), len(te)) == (70, 15, 15)
    assert tr.Time.max() < va.Time.min() and va.Time.max() < te.Time.min()
