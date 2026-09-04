"""
Loads the best trained model + preprocessing objects and serves
predictions. Models are trained OFFLINE (scripts/train_models.py) and
simply loaded here - never retrained per request.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from app.config import get_settings
from app.ml.preprocessing import apply_scaler, load_preprocessing, transaction_to_dataframe
from app.ml.risk_score import probability_to_risk_score, risk_score_to_level
from app.ml.train import BEST_MODEL_META_FILENAME, MODEL_FILENAME_TEMPLATE


class ModelNotTrainedError(RuntimeError):
    pass


@lru_cache
def _get_model_dir() -> Path:
    return get_settings().model_dir_resolved


def _load_best_model_meta() -> dict:
    meta_path = _get_model_dir() / BEST_MODEL_META_FILENAME
    if not meta_path.exists():
        raise ModelNotTrainedError(
            "No trained model found. Run `python scripts/train_models.py` before making predictions."
        )
    with open(meta_path) as f:
        return json.load(f)


@lru_cache
def get_best_model():
    meta = _load_best_model_meta()
    model_path = _get_model_dir() / MODEL_FILENAME_TEMPLATE.format(name=meta["model_key"])
    if not model_path.exists():
        raise ModelNotTrainedError(f"Model file missing: {model_path}. Re-run training.")
    pipeline = joblib.load(model_path)
    return meta["model_name"], pipeline


@lru_cache
def get_scaler():
    return load_preprocessing(_get_model_dir())


def load_evaluation_results() -> dict:
    results_path = _get_model_dir() / "evaluation_results.json"
    if not results_path.exists():
        raise ModelNotTrainedError(
            "No evaluation results found. Run `python scripts/train_models.py` before viewing the dashboard."
        )
    with open(results_path) as f:
        return json.load(f)


def predict_transaction(transaction: dict) -> dict:
    """Run the full online prediction pipeline for a single transaction."""
    model_name, pipeline = get_best_model()
    scaler = get_scaler()

    X = transaction_to_dataframe(transaction)
    X_scaled = apply_scaler(X, scaler)

    proba = pipeline.predict_proba(X_scaled)[0]
    legit_proba, fraud_proba = float(proba[0]), float(proba[1])
    is_fraud = fraud_proba >= 0.5

    risk_score = probability_to_risk_score(fraud_proba)
    risk_level = risk_score_to_level(risk_score)

    return {
        "model_name": model_name,
        "is_fraud": is_fraud,
        "fraud_probability": round(fraud_proba, 6),
        "legitimate_probability": round(legit_proba, 6),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "scaled_input": X_scaled,
        "raw_input": X,
    }
