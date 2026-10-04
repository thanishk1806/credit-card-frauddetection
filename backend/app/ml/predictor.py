"""
Loads ALL trained models + preprocessing objects and serves
per-transaction predictions.  Models are trained OFFLINE
(scripts/train_models.py) and simply loaded here.

Key changes from original:
- ALL five models are loaded and run inference per transaction (not just
  the "best" model).
- Class ordering is explicitly verified via model.classes_.
- Comprehensive debug logging traces the full data flow.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from app.config import get_settings
from app.ml.preprocessing import apply_scaler, load_preprocessing, transaction_to_dataframe
from app.ml.risk_score import probability_to_risk_score, risk_score_to_level
from app.ml.train import BEST_MODEL_META_FILENAME, MODEL_FILENAME_TEMPLATE
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Canonical model keys in display order
MODEL_KEYS: List[str] = [
    "random_forest",
    "adaboost",
    "xgboost",
    "lightgbm",
    "catboost",
]

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "random_forest": "Random Forest",
    "adaboost": "AdaBoost",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}


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
    """Return (model_display_name, imblearn_pipeline) for the best model."""
    meta = _load_best_model_meta()
    model_path = _get_model_dir() / MODEL_FILENAME_TEMPLATE.format(name=meta["model_key"])
    if not model_path.exists():
        raise ModelNotTrainedError(f"Model file missing: {model_path}. Re-run training.")
    pipeline = joblib.load(model_path)
    return meta["model_name"], pipeline


@lru_cache
def get_all_models() -> Dict[str, object]:
    """Load all five trained model pipelines. Returns {model_key: pipeline}."""
    model_dir = _get_model_dir()
    loaded = {}
    for key in MODEL_KEYS:
        path = model_dir / MODEL_FILENAME_TEMPLATE.format(name=key)
        if path.exists():
            loaded[key] = joblib.load(path)
            logger.info("Loaded model: %s from %s", key, path)
        else:
            logger.warning("Model file not found: %s (skipping)", path)
    if not loaded:
        raise ModelNotTrainedError(
            "No trained models found. Run `python scripts/train_models.py` first."
        )
    return loaded


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


def _get_fraud_probability(pipeline, X_scaled: pd.DataFrame) -> float:
    """
    Get fraud probability using explicit class ordering check.
    Returns the probability corresponding to the fraud class (1).
    """
    clf = pipeline.named_steps["clf"]
    proba = pipeline.predict_proba(X_scaled)[0]

    classes = clf.classes_
    # Find index of class 1 (fraud)
    fraud_idx = int(np.where(classes == 1)[0][0])
    legit_idx = int(np.where(classes == 0)[0][0])

    fraud_prob = float(proba[fraud_idx])
    legit_prob = float(proba[legit_idx])

    logger.debug(
        "Model classes=%s, proba=%s, fraud_idx=%d => fraud_prob=%.6f, legit_prob=%.6f",
        classes, proba, fraud_idx, fraud_prob, legit_prob,
    )
    return fraud_prob


def predict_transaction(transaction: dict) -> dict:
    """
    Run the full online prediction pipeline for a single transaction
    through ALL models.

    Returns a dict containing:
    - Per-model predictions (model_predictions)
    - Best model result (final prediction)
    - Raw + scaled feature vectors for SHAP
    - Debug info for logging
    """
    best_model_meta = _load_best_model_meta()
    best_key = best_model_meta["model_key"]
    best_name = best_model_meta["model_name"]

    all_models = get_all_models()
    scaler = get_scaler()

    # Feature engineering
    X = transaction_to_dataframe(transaction)
    X_scaled = apply_scaler(X, scaler)

    # ── Debug logging: trace the complete data flow ──
    logger.info("=" * 60)
    logger.info("PREDICTION REQUEST - Full Pipeline Trace")
    logger.info("=" * 60)
    logger.info("RAW USER INPUT: %s", {
        k: v for k, v in transaction.items()
        if k not in ("extra_data",)
    })
    logger.info("ENGINEERED FEATURES (raw, pre-scaling):")
    for col in X.columns:
        logger.info("  %-20s = %.6f", col, float(X.iloc[0][col]))
    logger.info("SCALED FEATURES (model input):")
    for col in X_scaled.columns:
        logger.info("  %-20s = %.6f", col, float(X_scaled.iloc[0][col]))

    # ── Run inference through ALL models ──
    model_predictions = {}
    for key, pipeline in all_models.items():
        fraud_prob = _get_fraud_probability(pipeline, X_scaled)
        legit_prob = round(1.0 - fraud_prob, 6)
        is_fraud = fraud_prob >= 0.5

        model_predictions[key] = {
            "model_key": key,
            "model_name": MODEL_DISPLAY_NAMES.get(key, key),
            "prediction": "FRAUD" if is_fraud else "LEGITIMATE",
            "is_fraud": is_fraud,
            "fraud_probability": round(fraud_prob, 6),
            "legitimate_probability": legit_prob,
            "risk_score": probability_to_risk_score(fraud_prob),
            "risk_level": risk_score_to_level(probability_to_risk_score(fraud_prob)),
        }

    # ── Best model result ──
    if best_key in model_predictions:
        best_result = model_predictions[best_key]
    else:
        # Fallback: use first available model
        best_key = next(iter(model_predictions))
        best_result = model_predictions[best_key]
        best_name = best_result["model_name"]

    fraud_probability = best_result["fraud_probability"]
    risk_score = best_result["risk_score"]
    risk_level = best_result["risk_level"]
    is_fraud = best_result["is_fraud"]

    # ── Log all model predictions ──
    logger.info("-" * 40)
    logger.info("PER-MODEL PREDICTIONS:")
    for key, pred in model_predictions.items():
        marker = " ★ BEST" if key == best_key else ""
        logger.info(
            "  %-15s => %s  fraud=%.4f%%  legit=%.4f%%%s",
            pred["model_name"],
            pred["prediction"],
            pred["fraud_probability"] * 100,
            pred["legitimate_probability"] * 100,
            marker,
        )
    logger.info("-" * 40)
    logger.info(
        "FINAL RESULT: %s | fraud_prob=%.4f%% | risk=%s (%s) | model=%s",
        "FRAUD" if is_fraud else "LEGITIMATE",
        fraud_probability * 100,
        risk_score,
        risk_level,
        best_name,
    )
    logger.info("=" * 60)

    # Get best model pipeline for SHAP
    best_pipeline = all_models.get(best_key)

    return {
        "model_name": best_name,
        "model_key": best_key,
        "is_fraud": is_fraud,
        "fraud_probability": round(fraud_probability, 6),
        "legitimate_probability": round(1.0 - fraud_probability, 6),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "scaled_input": X_scaled,
        "raw_input": X,
        "model_predictions": model_predictions,
        "best_pipeline": best_pipeline,
    }
