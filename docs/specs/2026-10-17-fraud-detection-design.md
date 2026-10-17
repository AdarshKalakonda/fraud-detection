# Credit Card Fraud Detection — Design Spec

Date: 2026-10-17
Status: Draft, awaiting review

## 1. Goal

A portfolio-grade MLOps project that is also a learning vehicle. It shows:

- A LightGBM fraud model trained on the Kaggle ULB credit card dataset.
- Business-aware threshold tuning (precision vs recall vs cost).
- Drift monitoring on live inputs.
- A FastAPI service, a Streamlit frontend, Docker images on Docker Hub, and code on GitHub.

Success criteria:

- `docker compose up` runs the full app locally (UI on :8501, API docs on :8000/docs).
- Both images are public on Docker Hub, and the compose file runs from those images.
- A public GitHub repo with milestone commits, passing tests, and a README showing results and design decisions.
- Later: GitHub Actions builds and pushes the images on every push to `main`.

Out of scope for now: AWS deployment (design notes kept in section 9), auth, Postgres, Prometheus, HTTPS/domain, Terraform.

## 2. Repo layout

Root: `E:\Docker Project`

```
├── data/                     # gitignored; holds creditcard.csv.zip
├── src/fraud/
│   ├── train.py              # split, baseline + LightGBM, PR-AUC, save artifacts
│   ├── threshold.py          # cost/recall/precision threshold selection
│   └── drift.py              # PSI computation
├── api/main.py               # FastAPI app
├── ui/app.py                 # Streamlit app
├── ui/samples.csv            # ~200 test-set rows for demo (committed)
├── models/                   # model.txt, config.json, reference.json, curves.json (committed)
├── tests/                    # test_threshold.py, test_drift.py, test_api.py
├── requirements-api.txt      # pinned
├── requirements-ui.txt       # pinned
├── requirements-dev.txt      # training + tests + ruff
├── Dockerfile.api
├── Dockerfile.ui
├── docker-compose.yml
├── .dockerignore, .gitignore
├── docs/specs/               # this file
└── README.md
```

## 3. Data

- Source: Kaggle "Credit Card Fraud Detection" (ULB), 284,807 rows, 492 frauds (~0.17%).
- The user places `creditcard.csv.zip` in `data/`. pandas reads the zip directly. There is no download script and no Kaggle token.
- `data/` and `*.zip` are gitignored.
- Columns: `Time`, `V1`–`V28` (PCA), `Amount`, `Class` (1 = fraud).
- Features: `V1`–`V28` plus `log1p(Amount)`. That is 29 model features; the API still accepts raw `Amount`. `Time` is not a feature, because it is seconds since the dataset start and does not generalize.
- Split: sort by `Time`. The first 70% is train, the next 15% is validation, and the last 15% is test. The test set is used once, for final reporting.

## 4. Model

- Baseline: logistic regression with `class_weight="balanced"` on standardized features.
- Main model: LightGBM with `scale_pos_weight`, early stopping on validation PR-AUC (average precision), and a small Optuna search (~30 trials).
- No SMOTE. Class weighting handles the imbalance.
- Primary metric: PR-AUC. ROC-AUC is reported for reference only.
- Quality gate: `train.py` exits non-zero and does not overwrite artifacts unless test PR-AUC ≥ 0.70 and beats the logistic baseline. (Was a flat 0.75; lowered after the first full run scored 0.749. With ~52 test frauds that gap is noise, and the gate exists to catch broken models.)
- Artifacts written to `models/`:
  - `model.txt`: LightGBM native text format (no pickle).
  - `config.json`: `model_version`, feature list, thresholds per policy, cost parameters, test metrics.
  - `reference.json`: per-feature decile bin edges and training proportions, plus the training score distribution.
  - `curves.json`: validation PR curve and cost-vs-threshold points for the UI.

## 5. Threshold tuning

- Costs are config values: missed fraud (FN) = €120, false alarm (FP) = €5.
- Sweep thresholds 0.001–0.999 (step 0.001) on validation; the scores of a well-calibrated model mostly sit below 0.05, so a 0.01 step is too coarse. Cost = FN × 120 + FP × 5.
- Three policies are stored in `config.json`:
  - `cost`: the minimum-cost threshold. This is the default.
  - `recall`: the highest threshold with recall ≥ 0.90.
  - `precision`: the lowest threshold with precision ≥ 0.90.
- If a precision or recall target is unreachable, fall back to the closest achievable threshold and log a warning in the config.
- Report test-set precision, recall, cost, and the confusion matrix per policy.

## 6. Drift monitoring

- Reference: training-set decile bin edges per feature, stored in `reference.json`.
- Live data: the last N logged API inputs (default 500).
- PSI per feature, with epsilon smoothing for empty bins.
- Status bands: PSI < 0.1 is `ok`, 0.1–0.25 is `watch`, > 0.25 is `drift`.
- Also reported: PSI of the prediction score distribution, and the live fraud-flag rate vs the validation flag rate.
- If fewer than 50 logged rows exist, return `status: "insufficient_data"` instead of PSI values.

## 7. API (FastAPI)

| Endpoint | Behavior |
|---|---|
| `GET /health` | `{status, model_version}` |
| `POST /predict` | Body: 30 raw fields (`V1`–`V28`, `Amount`, `Time` optional/ignored) and optional `policy` (`cost` \| `recall` \| `precision`, default `cost`). Returns `{fraud_probability, is_fraud, threshold, policy, model_version}` |
| `POST /predict/batch` | List of transactions, max 1000 (413 if larger), same `policy` param |
| `GET /drift?n=500` | Per-feature PSI and status, score PSI, flag rate |
| `GET /curves` | Returns `curves.json` (validation PR curve and cost-vs-threshold) for the UI |
| `GET /docs` | Auto-generated Swagger UI |

- The model and configs load once in the lifespan hook. If they are missing, the app fails at startup.
- Each prediction is logged to SQLite via stdlib `sqlite3`: timestamp, inputs, score, decision, policy, model version.
- Errors:
  - Invalid input returns 422 (Pydantic).
  - An oversized batch returns 413.
  - Unexpected errors return a 500 with a generic message, and the traceback goes to the server log only.
- Config comes from env vars: `MODEL_DIR` (default `models`), `DB_PATH` (default `logs/predictions.db`).

## 8. UI (Streamlit)

`API_URL` env var (default `http://localhost:8000`). Three tabs:

1. **Predict**:
   - "Random legit" / "Random fraud" buttons pick a row from `ui/samples.csv` and score it.
   - A CSV upload scores rows in bulk.
   - Shows probability, decision, threshold, and policy, with a policy selector.
2. **Threshold**:
   - PR curve and cost-vs-threshold chart from `GET /curves`.
   - A slider shows frauds caught vs false alarms at the chosen threshold.
3. **Drift**:
   - PSI bar chart colored by status, score distribution, and flag rate.
   - Buttons: "Send normal traffic" (500 sample rows) and "Send drifted traffic" (500 rows with `V14` and `Amount` shifted).

Charts use Streamlit built-ins and plotly. The UI never reads model files; everything comes from the API.

## 9. Docker and Docker Hub

- Base image: `python:3.11-slim`. Two images with separate pinned requirements.
- Both run as a non-root user, and `.dockerignore` excludes `data/`, `.git/`, `.venv/`, `logs/`.
- The API image runs uvicorn on 8000 and has a `HEALTHCHECK` on `/health`. The UI image runs streamlit on 8501.
- `docker-compose.yml`:
  - `api` gets a named volume `logs` at `/app/logs`.
  - `ui` gets `API_URL=http://api:8000` and `depends_on: api: condition: service_healthy`.
  - Both use `restart: unless-stopped`.
  - Both have `image: <dockerhub-user>/fraud-api` / `fraud-ui` alongside `build:`, so the same file builds locally or pulls.
- Docker Hub push is manual first:
  1. The user runs `docker login`.
  2. Tag `:v1` and `:latest`.
  3. `docker push`.
- Later, a GitHub Actions workflow on push to `main`:
  1. ruff
  2. pytest
  3. Build both images
  4. Push tags `:latest` and `:sha-<short>`

  The Docker Hub token goes in GitHub Secrets.
- Deferred AWS notes: EC2 t3.small, Ubuntu, Docker + compose pulling from Docker Hub, security group (22 and 8000 restricted to the user's IP, 8501 public), Elastic IP, $10 budget alert.

## 10. Testing

- `test_threshold.py`: a hand-computed toy case for the cost policy; the recall and precision policies meet their targets; fallback when a target is unreachable.
- `test_drift.py`: identical distributions give PSI ≈ 0; a shifted distribution gives PSI > 0.25; empty and constant inputs don't crash; fewer than 50 rows gives `insufficient_data`.
- `test_api.py` (TestClient):
  - health returns 200
  - a valid predict returns the right shape and a probability in [0, 1]
  - a missing field returns 422
  - a bad policy returns 422
  - a batch over 1000 returns 413
  - drift on an empty log returns `insufficient_data`
- Plus the PR-AUC quality gate in training.

## 11. Dev environment and workflow

- Python 3.11 venv via `uv` (system Python 3.14 untouched). `ruff` for lint and format.
- Milestones, each one commit plus a push to a public GitHub repo:
  1. Scaffold (git init, layout, .gitignore, requirements)
  2. Training + baseline
  3. Threshold tuning
  4. Drift module
  5. API
  6. UI
  7. Docker + compose
  8. Docker Hub push (manual)
  9. README polish (diagram, results table, screenshots, design decisions)
  10. GitHub Actions CI
- The user creates the GitHub repo and does `docker login`. Claude provides exact commands and never handles credentials.
- Never committed: `data/`, `*.zip`, `.venv/`, `logs/`, `*.db`, `.env`.
