import pandas as pd

from fraud.train import passes_gate, sample_rows, time_split


def test_sample_rows_tags_source_and_keeps_test_frauds_test_only():
    n = 1000
    df = pd.DataFrame({f"V{i}": [0.0] * n for i in range(1, 29)})
    df["Amount"], df["Time"] = 1.0, range(n)
    df["Class"] = [1 if i % 50 == 0 else 0 for i in range(n)]
    tr, _, te = time_split(df)
    out = sample_rows(tr, te, n_train=100, n_test=50)
    assert set(out.source) == {"train", "test"}
    assert (out.source == "train").sum() == 100 and (out[out.source == "train"].Class == 0).all()
    assert (out[(out.source == "test")].Class == 1).sum() == (te.Class == 1).sum()


def test_gate_requires_floor_and_beating_baseline():
    assert passes_gate(0.749, 0.690)
    assert not passes_gate(0.69, 0.60)  # below the 0.70 floor
    assert not passes_gate(0.72, 0.75)  # worse than baseline


def test_time_split_is_ordered_and_sized():
    df = pd.DataFrame({"Time": list(range(100))[::-1], "Class": [0] * 100})
    tr, va, te = time_split(df)
    assert (len(tr), len(va), len(te)) == (70, 15, 15)
    assert tr.Time.max() < va.Time.min() and va.Time.max() < te.Time.min()
