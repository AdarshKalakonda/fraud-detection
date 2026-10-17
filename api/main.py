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
from fastapi.exceptions import RequestValidationError
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


@app.exception_handler(RequestValidationError)
async def invalid(request: Request, exc: RequestValidationError):
    # Drop echoed input/ctx: NaN input can't be JSON-encoded, and we don't echo payloads.
    errors = [{k: e[k] for k in ("type", "loc", "msg") if k in e} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


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
            (
                now,
                json.dumps(x.tolist()),
                r["fraud_probability"],
                int(r["is_fraud"]),
                policy,
                version,
            )
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
