# Intelligent Credit Card Fraud Detection System
### Using Ensemble Machine Learning and Explainable AI

An academic mini-project that trains and compares five ensemble ML models
(Random Forest, AdaBoost, XGBoost, LightGBM, CatBoost) on the Kaggle
European Credit Card Fraud dataset, selects the best model, and serves
explainable, authenticated, real-time fraud predictions through a FastAPI
backend and React dashboard — with SHAP explanations and downloadable PDF
reports.

---

## 1. Features

- Secure JWT authentication (register / login / protected routes)
- Data-leakage-safe ML pipeline: clean → split → scale → **SMOTE only on
  training folds** → hyperparameter search → evaluate on untouched test set
- 5 ensemble models trained and compared: Random Forest, AdaBoost, XGBoost,
  LightGBM, CatBoost
- Hyperparameter optimization via `RandomizedSearchCV`
- Evaluation with Precision, Recall, F1, ROC-AUC, PR-AUC, MCC, confusion
  matrix, ROC/PR curves
- Automatic best-model selection (by F1-score, PR-AUC as tie-break — **not**
  accuracy)
- Real-time transaction prediction with a 0–100 fraud risk score
- SHAP-based explainability for every individual prediction
- Interactive dashboard: dataset overview, model comparison, ROC/PR curves,
  confusion matrices, feature importance, prediction history
- Dynamic PDF report generation (ReportLab) with real prediction + SHAP data
- Clean React frontend ↔ FastAPI REST API ↔ service layer ↔ ML layer ↔
  PostgreSQL architecture

---

## 2. Architecture

```
React Frontend  →  FastAPI REST API  →  Service Layer  →  ML/Prediction Layer  →  DB / Model Storage
```

The frontend **never** talks to PostgreSQL directly — all data access goes
through FastAPI.

```
project-root/
├── frontend/          React app (Vite)
├── backend/            FastAPI app
│   └── app/
│       ├── api/        route handlers (auth, dashboard, predict, predictions, reports)
│       ├── auth/        JWT + password hashing
│       ├── database/    SQLAlchemy engine/session
│       ├── models/      SQLAlchemy ORM models (User, Prediction)
│       ├── schemas/     Pydantic request/response models
│       ├── ml/           preprocessing, training pipeline, predictor, SHAP explainer
│       ├── reports/      PDF generation (ReportLab)
│       └── services/     orchestration layer used by the API routes
├── data/                place creditcard.csv here
├── ml_models/            trained models + evaluation results (generated)
├── scripts/
│   ├── train_models.py           offline training pipeline entrypoint
│   └── generate_sample_data.py   synthetic dataset for local dev/testing
├── docker-compose.yml    optional local PostgreSQL
├── .env.example
└── README.md
```

---

## 3. Tech Stack

| Layer          | Technology |
|----------------|------------|
| Frontend       | React 18 (Vite), React Router, Recharts, Axios |
| Backend        | FastAPI, Pydantic |
| ML             | scikit-learn, imbalanced-learn, XGBoost, LightGBM, CatBoost, SHAP |
| Database       | PostgreSQL + SQLAlchemy |
| Auth           | JWT (python-jose) + bcrypt (passlib) |
| Reporting      | ReportLab |

---

## 4. Dataset Setup

1. Download the Kaggle **Credit Card Fraud Detection** dataset:
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Place the file at `data/creditcard.csv`.

No real dataset yet? Generate a small synthetic one (schema-compatible)
purely to test the pipeline end-to-end:

```bash
python scripts/generate_sample_data.py --rows 20000 --fraud-rate 0.0017
```

**Do not use synthetic data for your final academic results** — swap in the
real Kaggle file before generating report/viva numbers.

---

## 5. PostgreSQL Setup

Option A — Docker (recommended for local dev):

```bash
docker compose up -d
```

This starts PostgreSQL with:
- user: `fraud_user`
- password: `fraud_pass`
- database: `fraud_detection_db`
- port: `5432`

Option B — Existing local PostgreSQL install: create a matching database and
user yourself, then point `DATABASE_URL` at it.

Tables are auto-created on backend startup (`init_db()` in `app/main.py`)
using SQLAlchemy `create_all()` — no separate migration step is required for
this project's scope.

---

## 6. Environment Variables

Copy the example env file and edit it:

```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
```

Key backend variables (`backend/.env`):

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `JWT_SECRET_KEY` | Secret used to sign JWTs — set a long random value |
| `JWT_ALGORITHM` | Default `HS256` |
| `JWT_EXPIRE_MINUTES` | Token lifetime |
| `CORS_ORIGINS` | Comma-separated allowed frontend origins |
| `DATASET_PATH` | Path to `creditcard.csv` |
| `MODEL_DIR` | Where trained models/evaluation results are saved |
| `RISK_LOW_MAX` / `RISK_MEDIUM_MAX` | Risk score thresholds (0-100 scale) |

Frontend (`frontend/.env`):

| Variable | Purpose |
|---|---|
| `VITE_API_BASE_URL` | FastAPI backend URL, e.g. `http://localhost:8000` |

**Never commit real `.env` files.**

---

## 7. Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then edit values as needed
```

## 8. Train the Models

From the **backend** directory (with the venv active):

```bash
python ../scripts/train_models.py
```

This runs the full leakage-safe pipeline and writes to `ml_models/`:
- `model_<name>.joblib` for each of the 5 models
- `scaler.joblib` (fitted preprocessing)
- `evaluation_results.json` (real metrics — nothing hard-coded)
- `best_model.json` (selected best model metadata)

Console output ends with a summary block and the selected `BEST MODEL`.

## 9. Run the Backend

```bash
uvicorn app.main:app --reload
```

API docs: http://localhost:8000/docs
Health check: http://localhost:8000/health

## 10. Frontend Setup & Run

```bash
cd frontend
npm install
npm run dev
```

App: http://localhost:5173

---

## 11. API Overview

```
POST   /auth/register
POST   /auth/login
GET    /auth/me

GET    /dashboard/summary
GET    /models/performance
GET    /models/best

POST   /predict
POST   /explain/{prediction_id}

GET    /predictions
GET    /predictions/{id}

POST   /reports/{prediction_id}
GET    /reports/{prediction_id}/download
```

All routes except `/auth/register`, `/auth/login`, and `/health` require a
`Authorization: Bearer <token>` header.

---

## 12. ML Methodology

**Pipeline (leakage-safe):**

```
Raw Dataset → Cleaning → Train/Test Split → StandardScaler(Time, Amount)
  fit on train only → SMOTE (inside CV folds / applied once to full
  training data for the final fit) → RandomizedSearchCV per model
  → Evaluate on the ORIGINAL untouched test set → Select best model
  → Persist
```

**Why not apply SMOTE before splitting?** Doing so would leak information
from synthetic minority samples derived from test-set-adjacent data into
training, inflating evaluation metrics unrealistically. SMOTE is applied
only inside `imblearn.pipeline.Pipeline`, so during hyperparameter search
it re-runs independently on each CV training fold, and at final-fit time it
runs once on the full training split only. The test set is never resampled.

**Why 5 ensemble algorithms?** To compare bagging (Random Forest), boosting
(AdaBoost), and three modern gradient-boosting implementations (XGBoost,
LightGBM, CatBoost) under identical preprocessing, giving a fair,
reproducible comparison — a natural ML mini-project narrative for a viva.

**Why not optimize/select by accuracy?** With fraud at roughly 0.17% of
transactions, a model predicting "legitimate" for everything would score
~99.8% accuracy while being useless. F1-score (harmonic mean of precision
and recall) and PR-AUC properly weight minority-class performance, so they
are used for hyperparameter scoring and best-model selection instead.

**Fraud Risk Score.** Deterministically derived as
`risk_score = fraud_probability * 100`, then bucketed into LOW / MEDIUM /
HIGH using the configurable `RISK_LOW_MAX` / `RISK_MEDIUM_MAX` thresholds.
It is never randomly generated.

**SHAP Explainability.** `shap.TreeExplainer` is used against the trained
classifier for each of the five tree-based models, producing per-feature
contribution values for a single transaction. The top contributors pushing
toward fraud and toward legitimacy are surfaced in the UI and PDF report as
readable bar charts rather than raw arrays.

---

## 13. PDF Reports

Every PDF is generated dynamically per-request from ReportLab using the
actual stored `Prediction` record (prediction, probabilities, risk score,
model used, SHAP contributors) plus that model's persisted evaluation
metrics. No static templates or fabricated values are used.

---

## 14. Testing

```bash
cd backend
pytest
```

Covers:
- Registration / login / protected route enforcement / duplicate rejection
- Prediction input validation and auth requirement
- 503 handling when no model has been trained yet
- Full ML pipeline: all 5 models train, SMOTE is train-only, all metrics are
  generated, best model is selected by F1, models/preprocessing are
  persisted to disk

---

## 15. Troubleshooting

| Problem | Fix |
|---|---|
| `Dataset not found` on training | Place `creditcard.csv` in `data/`, or run `scripts/generate_sample_data.py` |
| `503 Service Unavailable` on dashboard/predict | Run `python scripts/train_models.py` first |
| Backend can't connect to PostgreSQL | Check `DATABASE_URL`, ensure the DB is running (`docker compose up -d`) |
| `401 Unauthorized` in the UI | Your JWT expired — log in again |
| CORS errors in the browser console | Ensure `CORS_ORIGINS` in `backend/.env` includes your frontend URL |
| SHAP explanation errors for a model | `app/ml/explainer.py` falls back to a model-agnostic `shap.Explainer` automatically |

---

## 16. Demonstrating the Project (Viva Guide)

1. Show `data-leakage prevention`: point to `app/ml/train.py` — split before
   scaler/SMOTE, SMOTE inside the imblearn Pipeline, evaluation on the raw
   test set.
2. Show the training console output / `ml_models/evaluation_results.json`
   proving all 5 models were actually trained with real metrics.
3. Open the **Dashboard** — dataset overview, model comparison chart, ROC
   curve, prediction history.
4. Open **Model Performance** — full metric table, ROC/PR curves for all 5
   models, confusion matrices, and the best-model justification text.
5. Go to **Predict Transaction**, submit a transaction (or use "Fill Random
   Sample"), and walk through: prediction verdict → fraud probability →
   risk score/level → SHAP bar chart explanation.
6. Click **Download PDF Report** and open the generated file to show it
   contains the same real values as the UI.
7. Explain the security model: bcrypt password hashing, JWT-protected
   routes, environment-variable secrets, no plaintext passwords ever
   returned by the API.

---

## 17. Important Notes

- This is an **academic prototype**, not hardened for production banking
  infrastructure.
- No fabricated business fields (card number, CVV, merchant, location) are
  used — the model only ever sees `Time`, `V1`–`V28`, `Amount`, matching the
  real dataset schema.
- All dashboard/report values are generated by the actual trained models —
  nothing is hard-coded.
