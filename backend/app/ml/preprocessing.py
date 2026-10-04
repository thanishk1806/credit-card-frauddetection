"""
Preprocessing & Feature Engineering Pipeline.

Shared between offline training (scripts/train_models.py) and online prediction API.

Unified Feature Architecture:
1. Realistic banking features:
   - Amount, log_amount
   - Time, transaction_hour, hour_sin, hour_cos, is_night
   - time_since_prev (seconds), tx_velocity_5m (count)
   - is_online, is_card_present, is_international
   - is_high_risk_merchant, is_untrusted_device, risk_feature_sum
2. Kaggle PCA components:
   - V1 through V28 (representing card/account behavioral pattern signals)

Total feature vector: 15 domain/engineered features + 28 PCA features = 43 features.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "Class"

HIGH_RISK_MERCHANTS = {"electronics", "luxury_jewelry", "travel_airlines", "entertainment"}
ONLINE_CHANNELS = {"online", "wire_transfer", "mobile_app"}
UNTRUSTED_DEVICES = {"unknown_device"}

ENGINEERED_COLUMNS: List[str] = [
    "Time",
    "Amount",
    "log_amount",
    "transaction_hour",
    "hour_sin",
    "hour_cos",
    "time_since_prev",
    "tx_velocity_5m",
    "is_night",
    "is_online",
    "is_card_present",
    "is_international",
    "is_high_risk_merchant",
    "is_untrusted_device",
    "risk_feature_sum",
]

PCA_COLUMNS: List[str] = [f"V{i}" for i in range(1, 29)]

# Full 43-feature model input vector (15 domain/temporal + 28 PCA)
FEATURE_COLUMNS: List[str] = ENGINEERED_COLUMNS + PCA_COLUMNS

# Numerical columns that require standard scaling (binary flags and PCA features are bounded/normalized)
SCALED_COLUMNS: List[str] = [
    "Time",
    "Amount",
    "log_amount",
    "transaction_hour",
    "time_since_prev",
    "tx_velocity_5m",
    "risk_feature_sum",
]

SCALER_FILENAME = "scaler.joblib"
FEATURE_ORDER_FILENAME = "feature_order.json"
PREPROCESSING_META_FILENAME = "preprocessing_meta.json"


def engineer_dataframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive time, amount, behavioral risk indicators, and temporal velocity features.
    
    CRITICAL: All operations are strictly backward-looking or point-wise to prevent target leakage.
    Works seamlessly on both multi-row training DataFrames and single-row inference payloads.
    """
    df = df.copy()

    # 1. Base Time and Amount
    if "Time" not in df.columns:
        df["Time"] = 0.0
    if "Amount" not in df.columns:
        df["Amount"] = 0.0

    df["Time"] = df["Time"].astype(float)
    df["Amount"] = df["Amount"].astype(float)

    # Ensure chronological sort if multi-row training set
    if len(df) > 1 and "Time" in df.columns:
        df = df.sort_values("Time").reset_index(drop=True)

    # 2. Amount features
    df["log_amount"] = np.log1p(np.maximum(df["Amount"].values, 0.0))

    # 3. Temporal features
    df["transaction_hour"] = (df["Time"].values / 3600.0) % 24.0
    df["hour_sin"] = np.sin(2.0 * np.pi * df["transaction_hour"].values / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * df["transaction_hour"].values / 24.0)
    df["is_night"] = np.where(
        (df["transaction_hour"].values < 6.0) | (df["transaction_hour"].values >= 23.0),
        1.0,
        0.0,
    )

    # 4. Sequential / Temporal velocity (strictly backward looking)
    if "time_since_prev" not in df.columns:
        if len(df) > 1:
            time_diff = df["Time"].diff().fillna(1800.0)
            df["time_since_prev"] = np.maximum(time_diff.values, 0.0)
        else:
            df["time_since_prev"] = 1800.0
    else:
        df["time_since_prev"] = df["time_since_prev"].astype(float)

    if "tx_velocity_5m" not in df.columns:
        if len(df) > 1:
            times = df["Time"].values
            idx_300 = np.searchsorted(times, times - 300.0, side="left")
            df["tx_velocity_5m"] = (np.arange(len(df)) - idx_300 + 1).astype(float)
        else:
            df["tx_velocity_5m"] = 1.0
    else:
        df["tx_velocity_5m"] = df["tx_velocity_5m"].astype(float)

    # 5. Channel & Behavioral risk indicators
    tx_type = df["transaction_type"].astype(str).str.lower() if "transaction_type" in df.columns else pd.Series(["online"] * len(df))
    df["is_online"] = tx_type.isin(ONLINE_CHANNELS).astype(float)

    card_pres = df["card_present"] if "card_present" in df.columns else pd.Series([False] * len(df))
    # Convert bool or string representations safely
    if card_pres.dtype == object:
        df["is_card_present"] = card_pres.astype(str).str.lower().isin(["true", "1", "yes"]).astype(float)
    else:
        df["is_card_present"] = card_pres.astype(float)

    intl = df["international_transaction"] if "international_transaction" in df.columns else pd.Series([False] * len(df))
    if intl.dtype == object:
        df["is_international"] = intl.astype(str).str.lower().isin(["true", "1", "yes"]).astype(float)
    else:
        df["is_international"] = intl.astype(float)

    merch = df["merchant_category"].astype(str).str.lower() if "merchant_category" in df.columns else pd.Series(["retail_shopping"] * len(df))
    df["is_high_risk_merchant"] = merch.isin(HIGH_RISK_MERCHANTS).astype(float)

    dev = df["device_type"].astype(str).str.lower() if "device_type" in df.columns else pd.Series(["mobile"] * len(df))
    df["is_untrusted_device"] = dev.isin(UNTRUSTED_DEVICES).astype(float)

    # Composite sum of triggered risk flags
    is_high_amt = (df["Amount"].values > 500.0).astype(float)
    is_high_vel = (df["tx_velocity_5m"].values > 2.0).astype(float)
    df["risk_feature_sum"] = (
        df["is_night"].values
        + df["is_online"].values
        + (1.0 - df["is_card_present"].values)
        + df["is_international"].values
        + df["is_high_risk_merchant"].values
        + df["is_untrusted_device"].values
        + is_high_amt
        + is_high_vel
    )

    # 6. Ensure all PCA columns V1..V28 exist
    for i in range(1, 29):
        col = f"V{i}"
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].astype(float)

    return df[FEATURE_COLUMNS]


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load, clean, and enrich dataset for training."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Download 'creditcard.csv' and place it "
            "at the configured DATASET_PATH, or run scripts/generate_sample_data.py."
        )

    df = pd.read_csv(csv_path)

    # If raw Kaggle CSV without realistic banking columns, synthesize realistic columns conditioned on Class
    if "transaction_type" not in df.columns:
        rng = np.random.default_rng(42)
        n = len(df)
        is_fraud = df[TARGET_COLUMN].values == 1 if TARGET_COLUMN in df.columns else np.zeros(n, dtype=bool)

        df["transaction_type"] = np.where(
            is_fraud,
            rng.choice(["online", "wire_transfer", "atm"], size=n, p=[0.70, 0.20, 0.10]),
            rng.choice(["pos", "mobile_app", "online", "contactless"], size=n, p=[0.45, 0.30, 0.18, 0.07]),
        )
        df["merchant_category"] = np.where(
            is_fraud,
            rng.choice(["electronics", "luxury_jewelry", "travel_airlines", "grocery"], size=n, p=[0.45, 0.35, 0.15, 0.05]),
            rng.choice(["grocery", "food_dining", "retail_shopping", "fuel", "utilities"], size=n, p=[0.35, 0.25, 0.20, 0.12, 0.08]),
        )
        df["device_type"] = np.where(
            is_fraud,
            rng.choice(["unknown_device", "desktop", "mobile"], size=n, p=[0.55, 0.30, 0.15]),
            rng.choice(["mobile", "pos_terminal", "desktop"], size=n, p=[0.55, 0.35, 0.10]),
        )
        df["card_present"] = np.where(is_fraud, rng.choice([False, True], size=n, p=[0.95, 0.05]), rng.choice([True, False], size=n, p=[0.80, 0.20]))
        df["international_transaction"] = np.where(is_fraud, rng.choice([True, False], size=n, p=[0.60, 0.40]), rng.choice([False, True], size=n, p=[0.96, 0.04]))

    if df.isnull().values.any():
        before = len(df)
        df = df.dropna()
        print(f"[preprocessing] Dropped {before - len(df)} rows containing missing values.")

    duplicate_count = df.duplicated().sum()
    if duplicate_count:
        df = df.drop_duplicates()
        print(f"[preprocessing] Dropped {duplicate_count} duplicate rows.")

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df.reset_index(drop=True)


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract engineered feature matrix X and target y."""
    X_engineered = engineer_dataframe_features(df)
    y = df[TARGET_COLUMN].copy()
    return X_engineered, y


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on numerical features using ONLY the training split."""
    scaler = StandardScaler()
    scaler.fit(X_train[SCALED_COLUMNS])
    return scaler


def apply_scaler(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Transform features using pre-fitted scaler."""
    X = X.copy()
    X[SCALED_COLUMNS] = scaler.transform(X[SCALED_COLUMNS])
    return X[FEATURE_COLUMNS]


def save_preprocessing(scaler: StandardScaler, model_dir: Path) -> None:
    """Persist scaler and feature order metadata."""
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, model_dir / SCALER_FILENAME)
    with open(model_dir / FEATURE_ORDER_FILENAME, "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    meta = {
        "feature_columns": FEATURE_COLUMNS,
        "scaled_columns": SCALED_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
        "engineered_count": len(ENGINEERED_COLUMNS),
        "pca_count": len(PCA_COLUMNS),
    }
    with open(model_dir / PREPROCESSING_META_FILENAME, "w") as f:
        json.dump(meta, f, indent=2)


def load_preprocessing(model_dir: Path) -> StandardScaler:
    """Load pre-fitted scaler artifact."""
    scaler_path = model_dir / SCALER_FILENAME
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Preprocessing scaler not found at {scaler_path}. "
            "Run `python scripts/train_models.py` first."
        )
    return joblib.load(scaler_path)


def transaction_to_dataframe(transaction: dict) -> pd.DataFrame:
    """Convert user or API transaction payload into an engineered 1-row DataFrame.
    
    Supports:
    1. Realistic transaction fields: amount, timestamp, transaction_type, merchant_category,
       location, device_type, card_present, international_transaction, time_since_prev, tx_velocity_5m.
    2. Direct ML/PCA fields: V1..V28 (for tests and simulations).
    """
    # 1. Resolve Amount
    raw_amount = transaction.get("amount")
    if raw_amount is None:
        raw_amount = transaction.get("Amount", 0.0)
    amount_val = float(raw_amount)

    # 2. Resolve Time / Timestamp
    timestamp_val = transaction.get("timestamp")
    time_offset = transaction.get("Time")

    if timestamp_val is not None:
        try:
            if isinstance(timestamp_val, str):
                dt = datetime.fromisoformat(timestamp_val.replace("Z", "+00:00"))
            elif isinstance(timestamp_val, (int, float)):
                dt = datetime.fromtimestamp(timestamp_val)
            else:
                dt = datetime.utcnow()
            hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
            time_sec = dt.hour * 3600.0 + dt.minute * 60.0 + dt.second
        except Exception:
            hour = 12.0
            time_sec = 43200.0
    elif time_offset is not None:
        time_sec = float(time_offset)
        hour = (time_sec / 3600.0) % 24.0
    else:
        now = datetime.utcnow()
        hour = now.hour + now.minute / 60.0 + now.second / 3600.0
        time_sec = now.hour * 3600.0 + now.minute * 60.0 + now.second

    # 3. Behavioral and channel inputs
    tx_type = str(transaction.get("transaction_type") or "online").lower()
    merch = str(transaction.get("merchant_category") or "retail_shopping").lower()
    dev = str(transaction.get("device_type") or "mobile").lower()
    card_pres = bool(transaction.get("card_present", False))
    intl = bool(transaction.get("international_transaction", False))
    time_since_prev = float(transaction.get("time_since_prev", 1800.0) if transaction.get("time_since_prev") is not None else 1800.0)
    tx_velocity_5m = float(transaction.get("tx_velocity_5m", 0.0) if transaction.get("tx_velocity_5m") is not None else 0.0)

    row = {
        "Time": time_sec,
        "Amount": amount_val,
        "transaction_type": tx_type,
        "merchant_category": merch,
        "device_type": dev,
        "card_present": card_pres,
        "international_transaction": intl,
        "time_since_prev": time_since_prev,
        "tx_velocity_5m": tx_velocity_5m,
    }

    # 4. Fill V1..V28 (defaults to 0.0 neutral PCA mean if not in payload)
    for i in range(1, 29):
        col = f"V{i}"
        row[col] = float(transaction.get(col, 0.0))

    df_single = pd.DataFrame([row])
    return engineer_dataframe_features(df_single)
