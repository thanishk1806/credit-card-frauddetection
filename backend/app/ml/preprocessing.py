"""
Preprocessing utilities shared between the offline training script
(scripts/train_models.py) and the online prediction API.

Design decisions (documented for the viva):

1. `Time` and `Amount` have very different scales/characteristics than the
   V1..V28 PCA components (which are already roughly standardized by the
   original PCA transform). We apply a StandardScaler to ONLY `Time` and
   `Amount`, leaving V1..V28 untouched. This mirrors common practice on
   this dataset and avoids distorting features that are already
   well-scaled.
2. The exact same fitted scaler used at training time is persisted
   (joblib) and reloaded for every prediction request, so the online and
   offline pipelines are guaranteed to match.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS: List[str] = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"
SCALED_COLUMNS = ["Time", "Amount"]

SCALER_FILENAME = "scaler.joblib"
FEATURE_ORDER_FILENAME = "feature_order.json"


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load and validate the Kaggle creditcard.csv dataset."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Download 'creditcard.csv' from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place it "
            "at the configured DATASET_PATH, or run scripts/generate_sample_data.py "
            "to create a small synthetic dataset for local development/testing."
        )

    df = pd.read_csv(csv_path)

    missing_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    if df.isnull().values.any():
        # Document the decision: drop rows with any missing values rather than
        # silently imputing anonymized PCA features, since imputing PCA
        # components is not scientifically well justified for this dataset.
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
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on Time/Amount using ONLY the training split."""
    scaler = StandardScaler()
    scaler.fit(X_train[SCALED_COLUMNS])
    return scaler


def apply_scaler(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    X = X.copy()
    X[SCALED_COLUMNS] = scaler.transform(X[SCALED_COLUMNS])
    return X


def save_preprocessing(scaler: StandardScaler, model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, model_dir / SCALER_FILENAME)
    with open(model_dir / FEATURE_ORDER_FILENAME, "w") as f:
        json.dump(FEATURE_COLUMNS, f)


def load_preprocessing(model_dir: Path) -> StandardScaler:
    scaler_path = model_dir / SCALER_FILENAME
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Preprocessing scaler not found at {scaler_path}. "
            "Run `python scripts/train_models.py` first."
        )
    return joblib.load(scaler_path)


def transaction_to_dataframe(transaction: dict) -> pd.DataFrame:
    """Convert a single TransactionInput payload (dict) into a 1-row DataFrame
    with columns in the exact order used during training."""
    row = {col: float(transaction.get(col, 0.0)) for col in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)

