# Credit Card Fraud Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A LightGBM fraud model with cost-based threshold tuning and PSI drift monitoring, served by FastAPI, fronted by Streamlit, shipped as two Docker images on Docker Hub, with code on GitHub and CI via GitHub Actions.

**Architecture:**
- Pure-numpy modules (`features`, `threshold`, `drift`) are shared by the training script and the API.
- Training writes small committed artifacts to `models/`. The API loads them at startup, logs every prediction to SQLite, and computes drift from those logs.
- Streamlit talks to the API over HTTP only.

**Tech Stack:** Python 3.11, LightGBM, scikit-learn, Optuna, FastAPI, Pydantic v2, SQLite (stdlib), Streamlit, Plotly, pytest, ruff, uv, Docker, docker compose, GitHub Actions.

**Spec:** `docs/specs/2026-10-17-fraud-detection-design.md`

## Global Constraints

- Repo root: `E:\Docker Project`. Shell: PowerShell. Use the venv interpreter explicitly: `.venv\Scripts\python`.
- Python 3.11 in the venv and in the images (`python:3.11-slim`). The system Python 3.14 stays untouched.
- Data: `data/creditcard.csv.zip`, read directly by pandas. Never committed. Ignore `data/`, `*.zip`, `.venv/`, `logs/`, `*.db`, `.env`.
- Model features: `V1`–`V28` plus `log1p(Amount)`, 29 in total. `Time` is never a feature.
- Split: sort by `Time`, then 70 / 15 / 15 for train / val / test. The test set is used only for final reporting.
- Primary metric: PR-AUC (average precision). Quality gate: test PR-AUC must be ≥ 0.70 and beat the baseline, else exit 1 and write no artifacts (changed from 0.75 by the user after the first full run).
- Costs: FN = €120, FP = €5. Policies: `cost` (default), `recall` (≥ 0.90), `precision` (≥ 0.90). Threshold grid: 0.001–0.999, step 0.001 (changed after the smoke run).
- PSI bands: < 0.1 is `ok`, 0.1–0.25 is `watch`, > 0.25 is `drift`. Below 50 logged rows, return `insufficient_data`.
- API: max batch 1000 (413 above that). Env vars `MODEL_DIR` (default `models`) and `DB_PATH` (default `logs/predictions.db`).
- UI: env var `API_URL` (default `http://localhost:8000`). The UI never reads model files.
- Model is saved in LightGBM native text format (`model.txt`). No pickle.
- Before every commit, run `.venv\Scripts\ruff format .` and `.venv\Scripts\ruff check --fix .`.
- Check with the user before each step. Every `git push`, `docker login` and `docker push` happens only after the user says go. Claude never handles credentials.
- Learning mode: before each task, explain in 3–5 lines what we're about to build and why.

## Review Focus

1. **NaN or Infinity in a JSON request** (Python's `json` accepts `NaN`). Expect a 422 rejection, not a silent NaN probability. Pinned in Task 5.
2. **Negative `Amount`.** `log1p` of a value below −1 is NaN, so the request must be rejected with a 422. Pinned in Task 5.
3. **Empty batch `[]` to `/predict/batch`.** Expect 200 with `[]`, not a LightGBM crash. Pinned in Task 5.
4. **`/drift` called before any traffic, or with fewer than 50 rows.** Expect `insufficient_data`, not a reshape or division error. Pinned in Tasks 3 and 5.
5. **Uploaded CSV with missing, extra, non-numeric or negative values, or more than 1000 rows.** Expect a clear UI error naming the problem, extra columns silently dropped, and large files sent in chunks of 1000. Pinned in Task 6.

---

### Task 1: Scaffold the repo, environment and feature module

**Files:**
- Create: `pyproject.toml`, `requirements-api.txt`, `requirements-ui.txt`, `requirements-dev.txt`, `.gitignore`, `.dockerignore`
- Create: `src/fraud/__init__.py`, `src/fraud/features.py`, `api/__init__.py`, `ui/__init__.py`
- Test: `tests/test_features.py`
- Move: `creditcard.csv.zip` → `data/creditcard.csv.zip`

**Interfaces:**
- Produces:
  - `fraud.features.V_COLS: list[str]` = `["V1", …, "V28"]`
  - `fraud.features.FEATURES: list[str]` = `V_COLS + ["log_amount"]`
  - `fraud.features.to_matrix(rows: list[dict]) -> np.ndarray` returns shape `(len(rows), 29)` as float64

- [ ] **Step 1: Check uv, create the venv, move the data**

```powershell
uv --version
# if missing: python -m pip install uv
uv venv --python 3.11 .venv
New-Item -ItemType Directory -Force data, src\fraud, api, ui, tests, models
Move-Item creditcard.csv.zip data\
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "fraud"
version = "0.1.0"
requires-python = ">=3.11"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
```

- [ ] **Step 3: Write the requirements files**

`requirements-api.txt`:
```
fastapi==0.115.12
uvicorn==0.34.0
lightgbm==4.6.0
numpy==2.2.4
scipy==1.15.2
```

`requirements-ui.txt`:
```
streamlit==1.44.1
pandas==2.2.3
plotly==6.0.1
requests==2.32.3
```

`requirements-dev.txt`:
```
-r requirements-api.txt
-r requirements-ui.txt
scikit-learn==1.6.1
optuna==4.2.1
pytest==8.3.5
httpx==0.28.1
ruff==0.11.4
-e .
```

Install: `uv pip install --python .venv -r requirements-dev.txt`

If a pin fails to resolve, bump only that package to the nearest version that does, and note it in the commit message.

- [ ] **Step 4: Write `.gitignore` and `.dockerignore`**

`.gitignore`:
```
data/
*.zip
.venv/
logs/
*.db
.env
__pycache__/
*.egg-info/
.pytest_cache/
.ruff_cache/
```

`.dockerignore`:
```
data/
*.zip
.venv/
logs/
*.db
.env
.git/
.github/
docs/
tests/
**/__pycache__/
*.egg-info/
.pytest_cache/
.ruff_cache/
```

- [ ] **Step 5: Write the failing test `tests/test_features.py`**

```python
import numpy as np

from fraud.features import FEATURES, to_matrix


def test_to_matrix_order_and_log_amount():
    row = {f"V{i}": float(i) for i in range(1, 29)} | {"Amount": np.e - 1, "Time": 5.0}
    X = to_matrix([row])
    assert X.shape == (1, len(FEATURES)) == (1, 29)
    assert X[0, 0] == 1.0 and X[0, 27] == 28.0
    assert np.isclose(X[0, 28], 1.0)  # log1p(e - 1) == 1; Time is ignored
```

- [ ] **Step 6: Run it to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fraud.features'`

- [ ] **Step 7: Write `src/fraud/features.py`, plus empty `src/fraud/__init__.py`, `api/__init__.py` and `ui/__init__.py`**

```python
"""Single source of truth for turning raw transactions into model input."""

import numpy as np

V_COLS = [f"V{i}" for i in range(1, 29)]
FEATURES = V_COLS + ["log_amount"]


def to_matrix(rows: list[dict]) -> np.ndarray:
    """Raw rows (V1..V28, Amount; extra keys ignored) -> (n, 29) float matrix."""
    return np.array(
        [[r[c] for c in V_COLS] + [np.log1p(r["Amount"])] for r in rows], dtype=float
    ).reshape(len(rows), len(FEATURES))
```

- [ ] **Step 8: Run the test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_features.py -v`
Expected: PASS

- [ ] **Step 9: git init and the first commit (the spec and plan go in too)**

```powershell
git init -b main
.venv\Scripts\ruff format . ; .venv\Scripts\ruff check --fix .
git add .
git status   # confirm data/ and .venv/ are NOT listed
git commit -m "scaffold: repo layout, pinned deps, feature transform"
```

- [ ] **Step 10: Connect to GitHub (the user does this)**

1. The user creates an empty public repo `fraud-detection` on github.com, with no README.
2. Then, after the user says go:

```powershell
git remote add origin https://github.com/<user>/fraud-detection.git
git push -u origin main
```

---

### Task 2: Threshold tuning module

**Files:**
- Create: `src/fraud/threshold.py`
- Test: `tests/test_threshold.py`

**Interfaces:**
- Produces:
  - `GRID: np.ndarray`: thresholds 0.01..0.99, step 0.01
  - `confusion_at(y, scores, t: float) -> tuple[int, int, int, int]`: `(tp, fp, fn, tn)`, where predicted fraud means `score >= t`
  - `sweep(y, scores, fn_cost: float, fp_cost: float) -> list[dict]`: one row per grid threshold, with keys `threshold, tp, fp, fn, tn, precision, recall, cost`. Precision is 0.0 when nothing is predicted.
  - `pick_thresholds(y, scores, fn_cost=120.0, fp_cost=5.0, target=0.90) -> dict`: returns `{"thresholds": {"cost": float, "recall": float, "precision": float}, "warnings": list[str]}`

- [ ] **Step 1: Write the failing tests `tests/test_threshold.py`**

```python
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_threshold.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fraud.threshold'`

- [ ] **Step 3: Write `src/fraud/threshold.py`**

```python
"""Pick decision thresholds from validation scores by cost, recall or precision."""

import numpy as np

GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


def confusion_at(y, scores, t: float) -> tuple[int, int, int, int]:
    pred = scores >= t
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return tp, fp, fn, tn


def sweep(y, scores, fn_cost: float, fp_cost: float) -> list[dict]:
    rows = []
    for t in GRID:
        tp, fp, fn, tn = confusion_at(y, scores, t)
        rows.append(
            {
                "threshold": float(t),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "cost": fn * fn_cost + fp * fp_cost,
            }
        )
    return rows


def pick_thresholds(y, scores, fn_cost=120.0, fp_cost=5.0, target=0.90) -> dict:
    rows = sweep(y, scores, fn_cost, fp_cost)
    warnings = []

    # Lowest cost; on ties prefer the higher threshold (fewer alerts).
    cost = min(rows, key=lambda r: (r["cost"], -r["threshold"]))

    ok = [r for r in rows if r["recall"] >= target]
    if ok:
        recall = max(ok, key=lambda r: r["threshold"])
    else:
        recall = max(rows, key=lambda r: (r["recall"], r["threshold"]))
        warnings.append(f"recall {target} unreachable; best is {recall['recall']:.3f}")

    ok = [r for r in rows if r["precision"] >= target]
    if ok:
        precision = min(ok, key=lambda r: r["threshold"])
    else:
        precision = max(rows, key=lambda r: (r["precision"], -r["threshold"]))
        warnings.append(f"precision {target} unreachable; best is {precision['precision']:.3f}")

    return {
        "thresholds": {
            "cost": cost["threshold"],
            "recall": recall["threshold"],
            "precision": precision["threshold"],
        },
        "warnings": warnings,
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_threshold.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit, and push after the user says go**

```powershell
.venv\Scripts\ruff format . ; .venv\Scripts\ruff check --fix .
git add src/fraud/threshold.py tests/test_threshold.py
git commit -m "threshold: cost/recall/precision policies with fallback warnings"
```

---

### Task 3: Drift module (PSI)

**Files:**
- Create: `src/fraud/drift.py`
- Test: `tests/test_drift.py`

**Interfaces:**
- Produces:
  - `MIN_ROWS = 50`
  - `bin_edges(x: np.ndarray, n_bins=10) -> np.ndarray`: unique inner decile edges
  - `proportions(x, edges) -> np.ndarray`: length `len(edges) + 1`, sums to 1 when x is non-empty
  - `psi(ref_props, live_props) -> float`
  - `status(v: float) -> str`: `"ok"`, `"watch"` or `"drift"`
  - `build_reference(X: np.ndarray, names: list[str], scores: np.ndarray) -> dict`: `{"features": {name: {"edges": [...], "props": [...]}}, "score": {"edges": [...], "props": [...]}}`
  - `drift_report(reference: dict, X: np.ndarray, scores: np.ndarray, flags: np.ndarray, expected_flag_rate: float) -> dict`: returns `{"status": "insufficient_data", "n": int}`, or `{"status", "n", "features": {name: {"psi", "status"}}, "score": {"psi", "status"}, "flag_rate": {"live", "expected"}}`

- [ ] **Step 1: Write the failing tests `tests/test_drift.py`**

```python
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_drift.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fraud.drift'`

- [ ] **Step 3: Write `src/fraud/drift.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_drift.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit, and push after the user says go**

```powershell
.venv\Scripts\ruff format . ; .venv\Scripts\ruff check --fix .
git add src/fraud/drift.py tests/test_drift.py
git commit -m "drift: PSI per feature and score with status bands"
```

---

### Task 4: Training script and artifacts

Note: this runs after threshold and drift (the spec listed training first) because `train.py` imports both.

**Files:**
- Create: `src/fraud/train.py`
- Test: `tests/test_train.py`
- Generates (committed): `models/model.txt`, `models/config.json`, `models/reference.json`, `models/curves.json`, `ui/samples.csv`

**Interfaces:**
- Consumes: `FEATURES`, `to_matrix` (Task 1); `confusion_at`, `sweep`, `pick_thresholds` (Task 2); `build_reference` (Task 3)
- Produces:
  - `time_split(df) -> (train, val, test)` DataFrames
  - `main(argv=None) -> int`
  - Artifacts:
    - `config.json` keys: `model_version, features, thresholds{cost,recall,precision}, warnings, costs{fn,fp}, target, expected_flag_rate, metrics{test_pr_auc, test_roc_auc, baseline_test_pr_auc, test{policy: {threshold,tp,fp,fn,tn,precision,recall,cost}}}`
    - `curves.json` keys: `pr{precision[], recall[]}, sweep[]` (rows from `sweep`)
    - `samples.csv` columns: `V1..V28, Amount, Class`

- [ ] **Step 1: Write the failing test `tests/test_train.py`**

```python
import pandas as pd

from fraud.train import time_split


def test_time_split_is_ordered_and_sized():
    df = pd.DataFrame({"Time": list(range(100))[::-1], "Class": [0] * 100})
    tr, va, te = time_split(df)
    assert (len(tr), len(va), len(te)) == (70, 15, 15)
    assert tr.Time.max() < va.Time.min() and va.Time.max() < te.Time.min()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fraud.train'`

- [ ] **Step 3: Write `src/fraud/train.py`**

```python
"""Train baseline + LightGBM, tune thresholds, write artifacts to models/.

Usage: python -m fraud.train [--data data/creditcard.csv.zip] [--trials 30]
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fraud.drift import build_reference
from fraud.features import FEATURES, V_COLS, to_matrix
from fraud.threshold import confusion_at, pick_thresholds, sweep

FN_COST, FP_COST, TARGET = 120.0, 5.0, 0.90
PR_AUC_FLOOR = 0.75


def time_split(df: pd.DataFrame):
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    a, b = int(len(df) * 0.70), int(len(df) * 0.85)
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def xy(df: pd.DataFrame):
    return to_matrix(df.to_dict("records")), df["Class"].to_numpy()


def tune_lgbm(Xtr, ytr, Xva, yva, trials: int) -> lgb.LGBMClassifier:
    max_spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

    def fit(params: dict) -> lgb.LGBMClassifier:
        model = lgb.LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.05,
            subsample_freq=1,
            metric="average_precision",
            random_state=42,
            verbose=-1,
            **params,
        )
        model.fit(
            Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        return model

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, max_spw, log=True),
        }
        model = fit(params)
        return average_precision_score(yva, model.predict_proba(Xva)[:, 1])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=trials)
    print(f"best val PR-AUC={study.best_value:.4f} params={study.best_params}")
    return fit(study.best_params)


def policy_report(y, scores, t: float) -> dict:
    tp, fp, fn, tn = confusion_at(y, scores, t)
    return {
        "threshold": t,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "cost": fn * FN_COST + fp * FP_COST,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/creditcard.csv.zip")
    p.add_argument("--out", default="models")
    p.add_argument("--samples", default="ui/samples.csv")
    p.add_argument("--trials", type=int, default=30)
    args = p.parse_args(argv)

    df = pd.read_csv(args.data)
    train, val, test = time_split(df)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = xy(train), xy(val), xy(test)
    print(f"rows train/val/test={len(ytr)}/{len(yva)}/{len(yte)} frauds={ytr.sum()}/{yva.sum()}/{yte.sum()}")

    base = make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)
    ).fit(Xtr, ytr)
    base_ap = float(average_precision_score(yte, base.predict_proba(Xte)[:, 1]))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    model = tune_lgbm(Xtr, ytr, Xva, yva, args.trials)
    s_val, s_te = model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1]
    ap = float(average_precision_score(yte, s_te))
    print(f"test PR-AUC baseline={base_ap:.4f} lightgbm={ap:.4f}")
    if ap < PR_AUC_FLOOR:
        print(f"FAIL: test PR-AUC {ap:.4f} < {PR_AUC_FLOOR}; artifacts not written", file=sys.stderr)
        return 1

    picked = pick_thresholds(yva, s_val, FN_COST, FP_COST, TARGET)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    model.booster_.save_model(str(out / "model.txt"))

    config = {
        "model_version": datetime.now(UTC).strftime("%Y%m%d-%H%M%S"),
        "features": FEATURES,
        "thresholds": picked["thresholds"],
        "warnings": picked["warnings"],
        "costs": {"fn": FN_COST, "fp": FP_COST},
        "target": TARGET,
        "expected_flag_rate": float(np.mean(s_val >= picked["thresholds"]["cost"])),
        "metrics": {
            "test_pr_auc": ap,
            "test_roc_auc": float(roc_auc_score(yte, s_te)),
            "baseline_test_pr_auc": base_ap,
            "test": {k: policy_report(yte, s_te, t) for k, t in picked["thresholds"].items()},
        },
    }
    prec, rec, _ = precision_recall_curve(yva, s_val)
    step = max(1, len(prec) // 300)
    curves = {
        "pr": {"precision": prec[::step].tolist(), "recall": rec[::step].tolist()},
        "sweep": sweep(yva, s_val, FN_COST, FP_COST),
    }
    # Feature reference from train; score reference from val (train scores are overfit).
    reference = build_reference(Xtr, FEATURES, s_val)

    for name, obj in [("config", config), ("curves", curves), ("reference", reference)]:
        (out / f"{name}.json").write_text(json.dumps(obj, indent=2))

    cols = V_COLS + ["Amount", "Class"]
    legit = test[test.Class == 0].sample(400, random_state=42)
    fraud = test[test.Class == 1]
    fraud = fraud.sample(min(100, len(fraud)), random_state=42)
    pd.concat([legit, fraud])[cols].round(6).to_csv(args.samples, index=False)

    print(json.dumps(config["metrics"]["test"], indent=2))
    for w in picked["warnings"]:
        print(f"WARNING: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the unit test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_train.py -v`
Expected: PASS

- [ ] **Step 5: Run a quick training smoke run (about 1–2 min)**

Run: `.venv\Scripts\python -m fraud.train --trials 3`

Expected:
- Row counts print, then baseline and LightGBM test PR-AUC.
- Exit code 0, and `models\` contains 4 files and `ui\samples.csv` exists.

If the PR-AUC gate fails, stop and discuss with the user. Do not lower the floor silently.

- [ ] **Step 6: Run the full training (about 5–15 min)**

Run: `.venv\Scripts\python -m fraud.train`

Expected: LightGBM test PR-AUC ≥ 0.75 and above the baseline. Show the user the per-policy test table and explain the trade-off.

- [ ] **Step 7: Commit, and push after the user says go**

```powershell
.venv\Scripts\python -m pytest -q
.venv\Scripts\ruff format . ; .venv\Scripts\ruff check --fix .
git add src/fraud/train.py tests/test_train.py models/ ui/samples.csv
git commit -m "train: time split, LR baseline vs tuned LightGBM, thresholds, drift reference"
```

---

### Task 5: FastAPI service

**Files:**
- Create: `api/main.py`
- Test: `tests/conftest.py`, `tests/test_api.py`

**Interfaces:**
- Consumes: `V_COLS`, `FEATURES`, `to_matrix` (Task 1); `drift_report` (Task 3); artifact files (Task 4 format)
- Produces (HTTP; the UI in Task 6 relies on these):
  - `GET /health` returns `{"status": "ok", "model_version": str}`
  - `POST /predict?policy=cost|recall|precision`, body is a Transaction. Returns `{"fraud_probability", "is_fraud", "threshold", "policy", "model_version"}`
  - `POST /predict/batch?policy=...`, body is `list[Transaction]` (≤ 1000). Returns a list of the above.
  - `GET /drift?n=500` returns the `drift_report` dict
  - `GET /curves` returns the contents of `curves.json`
  - The Transaction shape: `V1..V28: float` (required), `Amount: float >= 0` (required), `Time: float | None`. NaN and Infinity are rejected.

- [ ] **Step 1: Write the test fixture `tests/conftest.py`**

It builds a tiny model so the tests never need the real dataset, which matters in CI.

```python
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
```

- [ ] **Step 2: Write the failing tests `tests/test_api.py`**

```python
import json

import numpy as np


def txn(**over):
    d = {f"V{i}": 0.0 for i in range(1, 29)} | {"Amount": 10.0}
    return d | over


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "model_version": "test"}


def test_predict_shape_and_default_policy(client):
    body = client.post("/predict", json=txn()).json()
    assert set(body) == {"fraud_probability", "is_fraud", "threshold", "policy", "model_version"}
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["policy"] == "cost" and body["threshold"] == 0.5


def test_predict_recall_policy_uses_its_threshold(client):
    assert client.post("/predict?policy=recall", json=txn()).json()["threshold"] == 0.2


def test_missing_field_is_422(client):
    d = txn()
    del d["V3"]
    assert client.post("/predict", json=d).status_code == 422


def test_bad_policy_is_422(client):
    assert client.post("/predict?policy=yolo", json=txn()).status_code == 422


def test_negative_amount_is_422(client):
    assert client.post("/predict", json=txn(Amount=-5.0)).status_code == 422


def test_nan_is_422(client):
    raw = json.dumps(txn(V1=float("nan")))  # Python emits a bare NaN token
    r = client.post("/predict", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_batch_ok_and_empty(client):
    assert len(client.post("/predict/batch", json=[txn()] * 3).json()) == 3
    r = client.post("/predict/batch", json=[])
    assert r.status_code == 200 and r.json() == []


def test_batch_too_large_is_413(client):
    assert client.post("/predict/batch", json=[txn()] * 1001).status_code == 413


def test_drift_before_traffic_is_insufficient(client):
    assert client.get("/drift").json() == {"status": "insufficient_data", "n": 0}


def test_drift_detects_shifted_traffic(client):
    rng = np.random.default_rng(1)
    rows = [txn(**{f"V{i}": float(rng.normal()) for i in range(1, 29)}, V1=5.0) for _ in range(100)]
    client.post("/predict/batch", json=rows)
    report = client.get("/drift?n=100").json()
    assert report["n"] == 100 and len(report["features"]) == 29
    assert report["features"]["V1"]["status"] == "drift" and report["status"] == "drift"


def test_curves(client):
    assert client.get("/curves").json()["pr"]["precision"] == [1.0]
```

- [ ] **Step 3: Run them to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.main'`

- [ ] **Step 4: Write `api/main.py`**

```python
"""Fraud scoring API: predict, log every prediction, report drift."""

import json
import logging
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import lightgbm as lgb
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import ConfigDict, Field, create_model

from fraud.drift import drift_report
from fraud.features import FEATURES, V_COLS, to_matrix

log = logging.getLogger("fraud.api")
MAX_BATCH = 1000
Policy = Literal["cost", "recall", "precision"]

Transaction = create_model(
    "Transaction",
    __config__=ConfigDict(allow_inf_nan=False),
    **{c: (float, ...) for c in V_COLS},
    Amount=(float, Field(..., ge=0)),
    Time=(float | None, None),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = Path(os.getenv("MODEL_DIR", "models"))
    st = app.state
    st.booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
    st.config = json.loads((model_dir / "config.json").read_text())
    st.reference = json.loads((model_dir / "reference.json").read_text())
    st.curves = json.loads((model_dir / "curves.json").read_text())
    db_path = Path(os.getenv("DB_PATH", "logs/predictions.db"))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    st.db = sqlite3.connect(db_path)
    st.db.execute(
        "CREATE TABLE IF NOT EXISTS predictions "
        "(ts REAL, features TEXT, score REAL, is_fraud INTEGER, policy TEXT, model_version TEXT)"
    )
    yield
    st.db.close()


app = FastAPI(title="Credit Card Fraud API", lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


# ponytail: async endpoints run on the single event-loop thread, so one sqlite
# connection needs no lock; move to a thread pool + lock if predict gets slow.
def score(request: Request, rows: list, policy: str) -> list[dict]:
    if not rows:
        return []
    st = request.app.state
    X = to_matrix([r.model_dump() for r in rows])
    t, version = st.config["thresholds"][policy], st.config["model_version"]
    results = [
        {
            "fraud_probability": float(p),
            "is_fraud": bool(p >= t),
            "threshold": t,
            "policy": policy,
            "model_version": version,
        }
        for p in st.booster.predict(X)
    ]
    now = time.time()
    st.db.executemany(
        "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)",
        [
            (now, json.dumps(x.tolist()), r["fraud_probability"], int(r["is_fraud"]), policy, version)
            for x, r in zip(X, results)
        ],
    )
    st.db.commit()
    return results


@app.get("/health")
async def health(request: Request):
    return {"status": "ok", "model_version": request.app.state.config["model_version"]}


@app.post("/predict")
async def predict(request: Request, txn: Transaction, policy: Policy = "cost"):
    return score(request, [txn], policy)[0]


@app.post("/predict/batch")
async def predict_batch(request: Request, txns: list[Transaction], policy: Policy = "cost"):
    if len(txns) > MAX_BATCH:
        raise HTTPException(413, f"batch size {len(txns)} exceeds {MAX_BATCH}")
    return score(request, txns, policy)


@app.get("/drift")
async def drift(request: Request, n: int = Query(500, ge=1, le=10_000)):
    st = request.app.state
    rows = st.db.execute(
        "SELECT features, score, is_fraud FROM predictions ORDER BY rowid DESC LIMIT ?", (n,)
    ).fetchall()
    X = np.array([json.loads(f) for f, _, _ in rows], dtype=float).reshape(len(rows), len(FEATURES))
    scores = np.array([s for _, s, _ in rows], dtype=float)
    flags = np.array([f for _, _, f in rows], dtype=float)
    return drift_report(st.reference, X, scores, flags, st.config["expected_flag_rate"])


@app.get("/curves")
async def curves(request: Request):
    return request.app.state.curves
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv\Scripts\python -m pytest -v`
Expected: all PASS (12 API tests plus the earlier ones)

- [ ] **Step 6: Manual check against the real model**

1. Run: `.venv\Scripts\python -m uvicorn api.main:app --port 8000`
2. Open `http://localhost:8000/docs`, try `/predict` with a row from `ui/samples.csv`, and show the user Swagger.
3. Stop the server with Ctrl+C.

- [ ] **Step 7: Commit, and push after the user says go**

```powershell
.venv\Scripts\ruff format . ; .venv\Scripts\ruff check --fix .
git add api/main.py tests/conftest.py tests/test_api.py
git commit -m "api: predict/batch/drift/curves with validation and SQLite logging"
```

---

### Task 6: Streamlit UI

**Files:**
- Create: `ui/client.py`, `ui/app.py`
- Test: `tests/test_ui_client.py`

**Interfaces:**
- Consumes: the HTTP API from Task 5; `ui/samples.csv` from Task 4
- Produces:
  - `ui.client.REQUIRED: list[str]` (V1..V28 plus Amount)
  - `prepare_rows(df) -> list[dict]`: raises `ValueError` with a human-readable message
  - `predict_batch(rows, policy) -> list[dict]`: sends chunks of 1000
  - `get(path, **params) -> dict`

- [ ] **Step 1: Write the failing tests `tests/test_ui_client.py`**

```python
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_ui_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ui.client'`

- [ ] **Step 3: Write `ui/client.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_ui_client.py -v`
Expected: 4 PASS

- [ ] **Step 5: Write `ui/app.py`**

```python
"""Streamlit front end: predict, explore thresholds, watch drift."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from client import get, predict_batch, prepare_rows

st.set_page_config(page_title="Fraud Detection", layout="wide")
SAMPLES = pd.read_csv(Path(__file__).parent / "samples.csv")
LEGIT = SAMPLES[SAMPLES.Class == 0]
COLORS = {"ok": "#2e7d32", "watch": "#f9a825", "drift": "#c62828"}

try:
    health = get("/health")
except requests.RequestException:
    st.error("API unreachable. Is the api service running?")
    st.stop()

st.title("Credit card fraud detection")
st.caption(f"Model version {health['model_version']}")
policy = st.sidebar.radio(
    "Threshold policy",
    ["cost", "recall", "precision"],
    help="cost: cheapest on validation (FN €120, FP €5). recall/precision: hit ≥ 90%.",
)

predict_tab, threshold_tab, drift_tab = st.tabs(["Predict", "Threshold", "Drift"])

with predict_tab:
    c1, c2 = st.columns(2)
    pick = None
    if c1.button("Random legit transaction"):
        pick = LEGIT.sample(1)
    if c2.button("Random fraud transaction"):
        pick = SAMPLES[SAMPLES.Class == 1].sample(1)
    if pick is not None:
        res = predict_batch(prepare_rows(pick), policy)[0]
        actual = "fraud" if pick.Class.iloc[0] == 1 else "legit"
        st.metric("Fraud probability", f"{res['fraud_probability']:.2%}")
        verdict = "FRAUD" if res["is_fraud"] else "legit"
        show = st.error if res["is_fraud"] else st.success
        show(
            f"Decision: {verdict} (threshold {res['threshold']:.2f}, policy {res['policy']}). "
            f"Actual label: {actual}"
        )
        st.dataframe(pick.drop(columns="Class"))

    upload = st.file_uploader("Or score a CSV (columns V1–V28, Amount)", type="csv")
    if upload:
        df = pd.read_csv(upload)
        try:
            rows = prepare_rows(df)
        except ValueError as e:
            st.error(str(e))
        else:
            results = pd.DataFrame(predict_batch(rows, policy))
            st.write(f"{int(results.is_fraud.sum())} of {len(results)} flagged")
            st.dataframe(
                pd.concat(
                    [df.reset_index(drop=True), results[["fraud_probability", "is_fraud"]]], axis=1
                )
            )

with threshold_tab:
    curves = get("/curves")
    sweep = pd.DataFrame(curves["sweep"])
    t = st.slider("Threshold", 0.001, 0.999, 0.05, 0.001, format="%.3f")
    row = sweep.loc[(sweep.threshold - t).abs().idxmin()]
    m = st.columns(4)
    m[0].metric("Frauds caught", f"{int(row.tp)} / {int(row.tp + row.fn)}")
    m[1].metric("False alarms", int(row.fp))
    m[2].metric("Precision", f"{row.precision:.1%}")
    m[3].metric("Cost (validation)", f"€{row.cost:,.0f}")

    pr = go.Figure(go.Scatter(x=curves["pr"]["recall"], y=curves["pr"]["precision"], name="PR"))
    pr.add_trace(go.Scatter(x=[row.recall], y=[row.precision], mode="markers", name="current"))
    pr.update_layout(title="Precision vs recall (validation)", xaxis_title="Recall",
                     yaxis_title="Precision")
    cost = go.Figure(go.Scatter(x=sweep.threshold, y=sweep.cost, name="cost"))
    cost.add_vline(x=t, line_dash="dash")
    cost.update_layout(title="Total cost vs threshold (validation)", xaxis_title="Threshold",
                       yaxis_title="€")
    g1, g2 = st.columns(2)
    g1.plotly_chart(pr, use_container_width=True)
    g2.plotly_chart(cost, use_container_width=True)

with drift_tab:
    c1, c2 = st.columns(2)
    if c1.button("Send normal traffic (500 legit rows)"):
        predict_batch(prepare_rows(LEGIT.sample(500, replace=True)), policy)
    if c2.button("Send drifted traffic (V14 −3, Amount ×5)"):
        d = LEGIT.sample(500, replace=True).copy()
        d["V14"] -= 3
        d["Amount"] *= 5
        predict_batch(prepare_rows(d), policy)

    report = get("/drift", n=500)
    if report["status"] == "insufficient_data":
        st.info(f"Need at least 50 logged predictions; have {report['n']}. Send some traffic.")
    else:
        st.subheader(f"Overall: {report['status'].upper()} (last {report['n']} predictions)")
        feats = pd.DataFrame(report["features"]).T.reset_index(names="feature")
        feats["psi"] = feats.psi.astype(float)
        fig = go.Figure(
            go.Bar(x=feats.feature, y=feats.psi, marker_color=feats.status.map(COLORS))
        )
        fig.add_hline(y=0.1, line_dash="dot")
        fig.add_hline(y=0.25, line_dash="dash")
        fig.update_layout(title="PSI per feature", yaxis_title="PSI")
        st.plotly_chart(fig, use_container_width=True)
        m = st.columns(3)
        m[0].metric("Score PSI", f"{report['score']['psi']:.3f}", report["score"]["status"])
        m[1].metric("Live flag rate", f"{report['flag_rate']['live']:.2%}")
        m[2].metric("Expected flag rate", f"{report['flag_rate']['expected']:.2%}")
```

- [ ] **Step 6: Manual run (two terminals)**

1. Terminal 1: `.venv\Scripts\python -m uvicorn api.main:app --port 8000`
2. Terminal 2: `.venv\Scripts\streamlit run ui/app.py`
3. Check in the browser:
   - Both random buttons score a transaction.
   - The threshold slider moves the metrics and charts.
   - On the Drift tab, "normal" traffic shows mostly ok, and "drifted" traffic turns V14 and log_amount red.
   - Uploading `ui/samples.csv` scores 500 rows.
   - Uploading a CSV with a deleted column shows the error message.
4. Stop the API and reload the UI. It should show "API unreachable".

- [ ] **Step 7: Commit, and push after the user says go**

```powershell
.venv\Scripts\python -m pytest -q
.venv\Scripts\ruff format . ; .venv\Scripts\ruff check --fix .
git add ui/client.py ui/app.py tests/test_ui_client.py
git commit -m "ui: streamlit predict, threshold explorer, drift dashboard"
```

---

### Task 7: Docker images and compose

**Files:**
- Create: `Dockerfile.api`, `Dockerfile.ui`, `docker-compose.yml`, `.env` (gitignored)

**Interfaces:**
- Consumes: everything above
- Produces:
  - Images `${DOCKERHUB_USER}/fraud-api:${TAG}` and `${DOCKERHUB_USER}/fraud-ui:${TAG}`
  - Ports: 8000 for the API, 8501 for the UI

- [ ] **Step 1: Write `Dockerfile.api`**

```dockerfile
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src:/app
# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
RUN useradd --create-home app && mkdir -p /app/logs && chown app /app/logs
COPY src/ src/
COPY api/ api/
COPY models/ models/
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Write `Dockerfile.ui`**

```dockerfile
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-ui.txt .
RUN pip install --no-cache-dir -r requirements-ui.txt
RUN useradd --create-home app
COPY ui/ ui/
USER app
EXPOSE 8501
CMD ["streamlit", "run", "ui/app.py", "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
```

- [ ] **Step 3: Write `docker-compose.yml`, and `.env` with the user's Docker Hub username**

```yaml
services:
  api:
    build: { context: ., dockerfile: Dockerfile.api }
    image: ${DOCKERHUB_USER:-local}/fraud-api:${TAG:-latest}
    ports: ["8000:8000"]
    volumes: [logs:/app/logs]
    restart: unless-stopped
  ui:
    build: { context: ., dockerfile: Dockerfile.ui }
    image: ${DOCKERHUB_USER:-local}/fraud-ui:${TAG:-latest}
    ports: ["8501:8501"]
    environment: [API_URL=http://api:8000]
    depends_on:
      api: { condition: service_healthy }
    restart: unless-stopped
volumes:
  logs: {}
```

`.env`:
```
DOCKERHUB_USER=<user's docker hub username>
```

- [ ] **Step 4: Build and run**

Run: `docker compose up --build -d` then `docker compose ps`

Expected:
- `api` shows `healthy` and `ui` shows `running`.
- `docker images` lists both, around 300–450 MB each.

- [ ] **Step 5: Smoke test in the browser**

1. At `http://localhost:8501`, send drifted traffic and confirm the Drift tab turns red.
2. Run `docker compose restart api` and reload. The drift count must survive the restart, which proves the volume works.
3. Run `docker compose exec api whoami`. Expected output: `app`, not root.
4. Run `docker compose down` (no `-v`).

- [ ] **Step 6: Commit, and push after the user says go**

```powershell
git add Dockerfile.api Dockerfile.ui docker-compose.yml
git status   # confirm .env is NOT listed
git commit -m "docker: slim non-root images, compose with healthcheck and log volume"
```

---

### Task 8: Push the images to Docker Hub (manual)

**Files:** none

- [ ] **Step 1: The user logs in**

1. The user creates a Docker Hub access token (Account settings → Personal access tokens).
2. The user runs `docker login -u <user>` in their own terminal and pastes the token.
3. Claude never sees the token.

- [ ] **Step 2: Tag v1 and push (after the user says go)**

```powershell
$env:TAG = "v1"; docker compose build
docker compose push
$env:TAG = "latest"; docker compose build
docker compose push
```

- [ ] **Step 3: Verify that a pull-only run works**

```powershell
docker compose down
docker image rm <user>/fraud-api:v1 <user>/fraud-ui:v1
$env:TAG = "v1"; docker compose pull
docker compose up -d --no-build
```

Expected: the app works at `:8501`, using images pulled from Docker Hub.

Explain to the user:
- Tags vs digests.
- Why `:latest` alone is risky.
- Layer caching on push.

---

### Task 9: README

**Files:**
- Create: `README.md`, `docs/img/predict.png`, `docs/img/threshold.png`, `docs/img/drift.png`

- [ ] **Step 1: Capture screenshots of the 3 tabs from the running compose app into `docs/img/`.**

- [ ] **Step 2: Write `README.md`.** Fill every number from `models/config.json`; nothing is invented. Sections:

  1. **Title and one-line pitch**, plus Docker Hub links.
  2. **Architecture**, as a mermaid diagram:
     ```mermaid
     flowchart LR
       U[User] --> UI[Streamlit :8501]
       UI -->|HTTP| API[FastAPI :8000]
       API --> M[(LightGBM model.txt)]
       API --> DB[(SQLite prediction log)]
       DB --> D[PSI drift report]
     ```
  3. **Results table:**
     - Baseline vs LightGBM test PR-AUC.
     - Per-policy threshold, precision, recall, FP, FN and cost on test.
  4. **Screenshots.**
  5. **Run it:**
     - Download the dataset into `data/`, then `docker compose up`.
     - Or, without the data, run the published images: `TAG=v1 docker compose up --no-build`.
  6. **Design decisions:**
     - Time-based split
     - PR-AUC over accuracy and ROC
     - Class weights, not SMOTE
     - Cost-based threshold with €120/€5
     - PSI bands
     - Score reference from val
     - LightGBM text format, not pickle
  7. **Limitations / next steps:**
     - Labels arrive late, so no live performance metric yet.
     - AWS deployment is deferred.
     - No auth.
  8. **Local dev:** uv venv, pytest, ruff.

- [ ] **Step 3: Commit, and push after the user says go**

```powershell
git add README.md docs/img/
git commit -m "docs: README with results, architecture, design decisions"
```

---

### Task 10: GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: The user adds repo secrets.** GitHub repo → Settings → Secrets and variables → Actions:
  - `DOCKERHUB_USERNAME`
  - `DOCKERHUB_TOKEN`: a new Docker Hub token with read and write scope

- [ ] **Step 2: Write `.github/workflows/ci.yml`**

```yaml
name: ci
on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
          cache-dependency-path: requirements-*.txt
      - run: pip install -r requirements-dev.txt
      - run: ruff check . && ruff format --check .
      - run: pytest -q

  docker:
    needs: test
    if: github.event_name == 'push'
    runs-on: ubuntu-latest
    strategy:
      matrix:
        image: [api, ui]
    steps:
      - uses: actions/checkout@v4
      - run: echo "SHORT_SHA=${GITHUB_SHA::7}" >> "$GITHUB_ENV"
      - uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
      - uses: docker/build-push-action@v6
        with:
          context: .
          file: Dockerfile.${{ matrix.image }}
          push: true
          tags: |
            ${{ secrets.DOCKERHUB_USERNAME }}/fraud-${{ matrix.image }}:latest
            ${{ secrets.DOCKERHUB_USERNAME }}/fraud-${{ matrix.image }}:sha-${{ env.SHORT_SHA }}
```

- [ ] **Step 3: Commit and push (after the user says go), then watch the run**

```powershell
git add .github/workflows/ci.yml
git commit -m "ci: test, then build and push images to Docker Hub on main"
git push
```

Expected:
- The Actions tab shows `test` green, then `docker (api)` and `docker (ui)` green.
- Docker Hub shows new `sha-xxxxxxx` tags.
- Add a CI badge to the README in a follow-up commit.
