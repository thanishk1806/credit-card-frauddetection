# Credit Card Fraud Detection & Risk Analysis System
### Using Ensemble Machine Learning, Dynamic Feature Engineering, and Explainable AI (SHAP)

A production-grade, full-stack Credit Card Fraud Detection platform built with **FastAPI**, **React**, **PostgreSQL / SQLite**, **scikit-learn**, **XGBoost**, **LightGBM**, and **CatBoost**.

---

## 1. System Architecture Overview

The system follows a strict **Three-Layer Architecture** designed for enterprise banking workflows while remaining scientifically rigorous:

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: REALISTIC USER / TRANSACTION INPUT                │
│  - Transaction Amount (e.g. ₹5,000 / $149.99)               │
│  - Transaction Date & Time                                  │
│  - Transaction Channel / Type (Online, POS, ATM, Wallet)     │
│  - Merchant Category (Grocery, Electronics, Travel, etc.)   │
│  - Location (City, Country) & Device Type (Mobile, Desktop) │
│  - Card Present Flag & International Flag                   │
└──────────────────────────────┬──────────────────────────────┘
                               │  (REST API / Pydantic Validation)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2: DYNAMIC FEATURE ENGINEERING                       │
│  - transaction_hour: (Time / 3600) % 24                     │
│  - hour_sin / hour_cos: Cyclical time representation        │
│  - log_amount: log(1 + Amount) non-linear scaling           │
│  - time_since_prev: Backward-looking interval from prior tx │
│  - tx_velocity_5m: Backward-looking 5-min transaction count │
└──────────────────────────────┬──────────────────────────────┘
                               │  (StandardScaler fit on Train only)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3: ML MODEL VECTOR, INFERENCE & EXPLAINABILITY       │
│  - 36-Feature Input Vector (8 Time/Amount/Velocity + 28 PCA)│
│  - 5 Tuned Ensemble Models (RF, AdaBoost, XGB, LGBM, Cat)   │
│  - Fraud Probability & Composite 0–100 Risk Score           │
│  - SHAP TreeExplainer Factor Breakdown & Forensic PDF Report│
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Dataset & Features

### Source Dataset
The system utilizes the standard benchmark **European Credit Card Fraud Detection dataset** (`creditcard.csv`).
- Total Transactions: 20,000 recorded transactions across 48 hours (172,800 seconds).
- Legitimate Transactions: 19,966 (99.83%)
- Fraudulent Transactions: 34 (0.1700%) — highly imbalanced classification.

> **CRITICAL DATASET NOTICE & TECHNICAL HONESTY:**
> The European Credit Card Fraud dataset contains anonymized PCA-transformed features `V1`–`V28`, elapsed seconds `Time`, `Amount`, and `Class`. It **does not** contain merchant names, merchant categories, customer locations, device identifiers, or customer-profile information.
> The system **does not fabricate** fake mappings (such as assigning `V1` to merchant or `V2` to location). Instead, realistic transaction metadata is captured in Layer 1 for compliance, audit trails, and future model training, while derived features (amount, time, velocity) and genuine statistical signals drive the current ML model.

### Feature Breakdown
1. **Raw Numerical Features**: `Time`, `Amount`, `V1`..`V28` (30 features)
2. **Engineered Features**:
   - `transaction_hour`: Hour of day (0.00 – 23.99)
   - `hour_sin` & `hour_cos`: Continuous trigonometric encoding of cyclical 24-hour cycles
   - `log_amount`: `log(1 + Amount)` to normalize right-skewed monetary distributions
   - `time_since_prev`: Elapsed seconds from previous transaction
   - `tx_velocity_5m`: Rolling count of transactions in the preceding 300-second window
3. **Total Model Feature Vector**: **36 features**

---

## 3. Data Leakage Prevention & Training Methodology

To ensure production validity and prevent data leakage:
1. **Train/Test Split First**: Data is partitioned (80% train, 20% test) using **Stratified Splitting** to maintain identical minority-class ratios in both sets.
2. **Preprocessing Fit Strictly on Train**: The `StandardScaler` is fitted *only* on the training split and then used to transform both train and test splits.
3. **SMOTE Inside CV Folds**: Synthetic Minority Over-sampling Technique (SMOTE) is applied **only inside cross-validation training folds** via `imblearn.pipeline.Pipeline`. The test set is **never** resampled or exposed to synthetic data.
4. **Backward-Looking Historical Features**: Velocity and interval calculations strictly evaluate past transactions ($t \le t_{\text{current}}$) with no lookahead.
5. **Imbalance-Aware Metric Optimization**: Hyperparameters are optimized via `RandomizedSearchCV` scoring on **F1-Score** and **PR-AUC** (Precision-Recall Area Under Curve), never simple accuracy.

---

## 4. Machine Learning Ensemble Models

Five state-of-the-art ensemble architectures are trained and benchmarked:

| Model | Architecture Type | Key Strengths |
|---|---|---|
| **Random Forest** | Bagging Ensemble | Resilient to outliers, low variance |
| **AdaBoost** | Adaptive Boosting | Focuses sequentially on hard-to-classify samples |
| **XGBoost** | Gradient Boosting | Exact tree split optimization, regularization |
| **LightGBM** | Histogram Gradient Boosting | Leaf-wise tree growth, high throughput |
| **CatBoost** | Ordered Boosting | Symmetric trees, robust against overfitting |

---

## 5. Explainable AI (SHAP) & Risk Scoring

### Risk Score Calculation
The continuous composite risk score is derived directly from the model's calibrated fraud probability:
$$\text{Risk Score} = \text{round}(\text{fraud\_probability} \times 100, 1)$$

- **0 – 29**: 🟢 **LOW RISK** (Standard authorization approved)
- **30 – 69**: 🟡 **MEDIUM RISK** (Requires Step-Up 2FA / OTP verification)
- **70 – 100**: 🔴 **HIGH RISK** (Transaction flagged / blocked for manual review)

### SHAP Explainability
`shap.TreeExplainer` computes exact Shapley attributions for all 36 engineered features.
- High-risk pushes (e.g. extreme amounts, off-peak transaction hours, anomalous PCA security indices $V_{14}, V_{17}, V_4$) are surfaced with both visual impact bars and plain-English operational explanations.
- Legitimacy anchors (e.g. routine daytime hours, moderate amounts, standard card patterns) are displayed in green.

---

## 6. Project Structure

```
fraud-detection-system/
├── backend/
│   ├── app/
│   │   ├── api/              # FastAPI endpoints (auth, predict, dashboard, reports)
│   │   ├── auth/             # JWT authentication & password hashing
│   │   ├── database/         # SQLAlchemy session & SQLite / PostgreSQL setup
│   │   ├── ml/               # Preprocessing, training pipeline, predictor, SHAP
│   │   │   ├── explainer.py      # SHAP TreeExplainer & feature attribution
│   │   │   ├── model_registry.py # 5 ensemble model definitions & tuning spaces
│   │   │   ├── predictor.py      # Online prediction orchestrator
│   │   │   ├── preprocessing.py  # 36-feature engineering pipeline & scaler
│   │   │   ├── risk_score.py     # 0-100 score & risk-level classifier
│   │   │   └── train.py          # Leakage-safe SMOTE + RandomizedSearchCV training
│   │   ├── models/           # DB Models (User, Prediction)
│   │   ├── reports/          # Dynamic PDF generation (ReportLab)
│   │   ├── schemas/          # Pydantic schemas for realistic transactions
│   │   └── services/         # Business orchestration layer
│   └── tests/                # Pytest automated test suite
├── data/
│   └── creditcard.csv        # Dataset
├── frontend/
│   └── src/
│       ├── components/       # UI components (RiskBadge, ShapBar, ErrorBanner)
│       ├── pages/            # Predict, Reports, Dashboard, ModelComparison, Auth
│       └── services/         # Axios API client
├── ml_models/                # Saved joblib models, scaler, feature order, metrics
├── scripts/
│   ├── train_models.py       # Offline model training entrypoint
│   └── generate_sample_data.py
├── docker-compose.yml
└── README.md
```

---

## 7. How to Run the System

### Step 1: Backend Setup
```bash
cd backend
python -m venv .venv
# Activate venv:
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

### Step 2: Retrain the ML Models
```bash
# From the project root or backend directory:
python scripts/train_models.py
```
This will:
- Load `data/creditcard.csv`
- Engineer all 36 temporal, amount, and velocity features
- Run 3-fold stratified cross-validation with SMOTE across 5 ensemble models
- Persist `scaler.joblib`, `best_model.json`, `evaluation_results.json`, and the 5 model binaries in `ml_models/`

### Step 3: Start FastAPI Backend
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```
- Interactive Swagger API Docs: `http://localhost:8000/docs`

### Step 4: Start React Frontend
```bash
cd frontend
npm install
npm run dev
```
- Web Application: `http://localhost:5173`

---

## 8. Automated Test Suite
Run the comprehensive test suite verifying authentication, validation, leakage-safe pipeline, and prediction endpoints:
```bash
cd backend
pytest
```

---

## 9. API Reference

### Prediction Endpoint
- **URL**: `POST /predict`
- **Headers**: `Authorization: Bearer <token>`
- **Request Body (Realistic Transaction)**:
```json
{
  "amount": 149.99,
  "timestamp": "2026-10-03T10:30:00Z",
  "transaction_type": "online",
  "merchant_category": "electronics",
  "location": "Hyderabad, India",
  "device_type": "mobile",
  "card_present": false,
  "international_transaction": false
}
```

- **Response Body**:
```json
{
  "prediction_id": "c71a3991-88be-4d92-96a8-20d0f2eb1234",
  "prediction": "LEGITIMATE",
  "is_fraud": false,
  "prediction_label": "NOT FRAUD",
  "fraud_probability": 0.0012,
  "legitimate_probability": 0.9988,
  "risk_score": 0.1,
  "risk_level": "LOW",
  "model_used": "XGBoost",
  "top_factors": [
    {
      "feature": "Amount",
      "label": "Transaction Amount ($)",
      "impact": 0.28,
      "direction": "legitimate",
      "description": "Transaction amount of $149.99 supported transaction legitimacy."
    }
  ],
  "top_fraud_contributors": [],
  "top_legitimate_contributors": [...],
  "transaction_summary": {
    "amount": 149.99,
    "transaction_type": "online",
    "merchant_category": "electronics",
    "location": "Hyderabad, India",
    "device_type": "mobile",
    "card_present": false,
    "international_transaction": false
  },
  "created_at": "2026-10-03T10:30:00Z"
}
```

---

## 10. Current Limitations & Recommended Future Work

1. **Dataset Anonymity**: The European dataset anonymizes 28 features (`V1`–`V28`) via PCA. To train models with real-world merchant categories, geographical distance from cardholder home address, and device fingerprint hashes directly in the loss function, future work should integrate a richer real-world transaction dataset (e.g. IEEE-CIS Fraud Detection dataset or synthetic Spark/Kafka transaction event streams).
2. **Streaming Feature Store**: Future iterations can integrate Redis or Feast to compute real-time cardholder aggregate velocity metrics across 1h, 24h, and 7d rolling windows in production.
