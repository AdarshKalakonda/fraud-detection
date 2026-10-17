"""HTTP client and input cleaning for the Streamlit UI."""

import os

import pandas as pd
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")
REQUIRED = [f"V{i}" for i in range(1, 29)] + ["Amount"]
MAX_BATCH = 1000


def prepare_rows(df: pd.DataFrame) -> list[dict]:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {', '.join(missing)}")
    clean = df[REQUIRED].apply(pd.to_numeric, errors="coerce")
    bad = clean.isna().any(axis=1) | (clean["Amount"] < 0)
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} rows have empty, non-numeric or negative values "
            f"(first bad row index: {bad.idxmax()})"
        )
    return clean.to_dict("records")


def predict_batch(rows: list[dict], policy: str) -> list[dict]:
    out = []
    for i in range(0, len(rows), MAX_BATCH):
        r = requests.post(
            f"{API_URL}/predict/batch",
            params={"policy": policy},
            json=rows[i : i + MAX_BATCH],
            timeout=30,
        )
        r.raise_for_status()
        out += r.json()
    return out


def get(path: str, **params) -> dict:
    r = requests.get(f"{API_URL}{path}", params=params, timeout=10)
    r.raise_for_status()
    return r.json()
