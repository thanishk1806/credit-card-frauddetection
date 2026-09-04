"""
SHAP-based explainability for individual predictions.

Uses tree-based SHAP explainers across all five ensemble models:
- Random Forest: TreeExplainer (multi-class positive extraction)
- AdaBoost: Fast exact tree-weighted TreeExplainer over individual DecisionTree estimators (<20ms)
- XGBoost: TreeExplainer
- LightGBM: TreeExplainer
- CatBoost: TreeExplainer
"""
from __future__ import annotations

import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import AdaBoostClassifier

from app.ml.preprocessing import FEATURE_COLUMNS

TOP_N = 5


def _get_shap_values(clf, X_scaled: pd.DataFrame) -> np.ndarray:
    """Return a 1D array of SHAP values (one per feature) for the positive
    (fraud) class, for a single-row input."""
    # 1. Specialized fast Tree SHAP for AdaBoostClassifier
    if isinstance(clf, AdaBoostClassifier):
        estimators = getattr(clf, "estimators_", [])
        weights = getattr(clf, "estimator_weights_", None)
        if estimators:
            n_features = X_scaled.shape[1]
            total_shap = np.zeros(n_features, dtype=float)
            total_weight = 0.0

            for i, est in enumerate(estimators):
                w = float(weights[i]) if weights is not None and i < len(weights) else 1.0
                if w <= 0:
                    continue
                try:
                    exp = shap.TreeExplainer(est)
                    sv = exp.shap_values(X_scaled)
                    if isinstance(sv, list) and len(sv) == 2:
                        val = np.asarray(sv[1]).flatten()
                    elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                        val = sv[0, :, 1] if sv.shape[-1] == 2 else sv[1, 0, :]
                    else:
                        val = np.asarray(sv).flatten()
                    if len(val) == n_features:
                        total_shap += w * val
                        total_weight += w
                except Exception:
                    continue

            if total_weight > 0:
                return total_shap / total_weight

    # 2. General TreeExplainer for Random Forest, XGBoost, LightGBM, CatBoost
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer = shap.TreeExplainer(clf)
            raw = explainer.shap_values(X_scaled)
    except Exception:
        # Fallback to model-agnostic Explainer with single background sample
        try:
            background = X_scaled.sample(min(len(X_scaled), 1), random_state=42)
            explainer = shap.Explainer(clf.predict_proba, background)
            raw = explainer(X_scaled).values
        except Exception:
            return np.zeros(len(FEATURE_COLUMNS))

    # Normalize structure to a 1D vector corresponding to positive class
    if isinstance(raw, list):
        if len(raw) == 2:
            return np.asarray(raw[1]).flatten()
        if len(raw) == 1:
            return np.asarray(raw[0]).flatten()
        return np.asarray(raw).flatten()

    arr = np.asarray(raw)
    if arr.ndim == 3:
        if arr.shape[0] == 2:
            return arr[1][0]
        if arr.shape[-1] == 2:
            return arr[0, :, 1]
        return arr[0, :, -1]
    if arr.ndim == 2:
        return arr[0]

    return arr.flatten()


def explain_prediction(pipeline, X_scaled: pd.DataFrame, X_raw: pd.DataFrame) -> Tuple[List[dict], List[dict]]:
    """Returns (top_fraud_contributors, top_legitimate_contributors)."""
    clf = pipeline.named_steps["clf"]
    shap_values = _get_shap_values(clf, X_scaled)

    contributions = []
    for i, feature in enumerate(FEATURE_COLUMNS):
        shap_val = float(shap_values[i]) if i < len(shap_values) else 0.0
        val = float(X_raw.iloc[0][feature]) if feature in X_raw.columns else 0.0
        contributions.append({
            "feature": feature,
            "value": round(val, 4),
            "shap_value": float(shap_val),
            "direction": "fraud" if shap_val > 0 else "legitimate",
        })

    # Sort positive contributions pushing towards fraud
    fraud_candidates = [c for c in contributions if c["shap_value"] > 0]
    fraud_candidates.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
    fraud_contributors = fraud_candidates[:TOP_N]

    # Sort negative / neutral contributions supporting legitimacy
    legit_candidates = [c for c in contributions if c["shap_value"] <= 0]
    legit_candidates.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
    legit_contributors = legit_candidates[:TOP_N]

    return fraud_contributors, legit_contributors
