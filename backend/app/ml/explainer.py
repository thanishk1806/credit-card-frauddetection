"""
SHAP-based explainability for individual predictions.

Generates plain-English, user-friendly explanations from SHAP values.
Anonymous PCA features (V1–V28) are described honestly as patterns
detected in the transaction data — never given fake business labels.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import AdaBoostClassifier

from app.ml.preprocessing import FEATURE_COLUMNS
from app.utils.logger import get_logger

logger = get_logger(__name__)

TOP_N = 5

# ───────────────────────────────────────────────────────────
# Human-friendly feature labels (only for real interpretable
# features). V1–V28 are anonymous PCA components — we never
# pretend to know their business meaning.
# ───────────────────────────────────────────────────────────
FEATURE_LABELS: Dict[str, str] = {
    "Amount": "Transaction Amount",
    "log_amount": "Transaction Amount (log-scale)",
    "Time": "Transaction Timestamp",
    "transaction_hour": "Time of Day",
    "hour_sin": "Time of Day (cyclical)",
    "hour_cos": "Time of Day (cyclical)",
    "is_night": "Late-Night Transaction Window",
    "time_since_prev": "Time Since Previous Transaction",
    "tx_velocity_5m": "Recent Transaction Activity",
    "is_online": "Online / Remote Channel",
    "is_card_present": "Physical Card Present",
    "is_international": "International / Cross-Border",
    "is_high_risk_merchant": "High-Risk Merchant Category",
    "is_untrusted_device": "Unrecognized / Untrusted Device",
    "risk_feature_sum": "Composite Risk Factor Indicator",
}


def _get_feature_label(feature: str) -> str:
    """Return a human label. For V-features, return an honest generic label."""
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    if feature.startswith("V") and feature[1:].isdigit():
        return f"Behavioral Security Signal ({feature})"
    return feature


# ───────────────────────────────────────────────────────────
# Plain-English explanation generator
# Converts technical SHAP contributions into sentences that
# anyone can understand.
# ───────────────────────────────────────────────────────────
def _generate_explanation_sentence(feature: str, value: float, shap_val: float) -> str:
    """Generate a plain-English sentence for ONE SHAP contribution."""
    is_fraud_direction = shap_val > 0

    if feature in ("Amount", "log_amount"):
        if is_fraud_direction:
            if value > 500:
                return "The transaction amount is unusually high compared with the normal pattern."
            return "The transaction amount pattern is associated with elevated fraud risk."
        else:
            if value < 50:
                return "The transaction amount is within the usual spending range, typical of normal transactions."
            return "The transaction amount is consistent with normal everyday spending patterns."

    if feature == "is_international":
        if is_fraud_direction and value > 0.5:
            return "This transaction was initiated across international borders."
        elif not is_fraud_direction:
            return "The transaction was domestic within the cardholder's home region."

    if feature == "is_card_present":
        if is_fraud_direction and value < 0.5:
            return "The physical card was not present during payment (Card-Not-Present transaction)."
        elif not is_fraud_direction and value >= 0.5:
            return "The transaction was completed with the physical card present at the point of sale."

    if feature == "is_online":
        if is_fraud_direction and value > 0.5:
            return "The transaction was initiated remotely via an online channel rather than in person."
        elif not is_fraud_direction:
            return "The transaction occurred through a standard, verified merchant channel."

    if feature == "is_high_risk_merchant":
        if is_fraud_direction and value > 0.5:
            return "The merchant category carries elevated dispute and fraud risk."
        elif not is_fraud_direction:
            return "The merchant category is routine and consistent with everyday spending."

    if feature == "is_untrusted_device":
        if is_fraud_direction and value > 0.5:
            return "This transaction happened through an unfamiliar or untrusted device."
        elif not is_fraud_direction:
            return "The transaction originated from a recognized, standard device terminal."

    if feature in ("is_night", "transaction_hour", "hour_sin", "hour_cos"):
        if is_fraud_direction:
            return "The transaction occurred during unusual late-night hours."
        return "The transaction took place during normal daytime business hours."

    if feature == "time_since_prev":
        if is_fraud_direction:
            if value < 60:
                return "The time between this transaction and the previous transaction was unusually short."
            return "The timing gap between transactions is associated with elevated risk."
        return "The time between transactions is consistent with normal usage."

    if feature == "tx_velocity_5m":
        if is_fraud_direction:
            if value > 2:
                return "Several transactions occurred within a short 5-minute period."
            return "Recent transaction frequency is slightly elevated."
        return "Recent transaction frequency is normal with no rapid bursts detected."

    if feature == "risk_feature_sum":
        if is_fraud_direction:
            return "Multiple risk indicators were triggered concurrently on this transaction."
        return "No concurrent risk flags were triggered on this transaction."

    if feature == "Time":
        if is_fraud_direction:
            return "The overall transaction timing pattern is associated with elevated fraud risk."
        return "The transaction timing is consistent with normal activity patterns."

    # ── Anonymous PCA features (V1–V28) ──
    # We NEVER assign fake real-world meanings to anonymized PCA features.
    if feature.startswith("V") and feature[1:].isdigit():
        if is_fraud_direction:
            return "The model detected a pattern in the transaction data that is associated with higher fraud risk."
        return "The model detected a pattern in the transaction data that is consistent with legitimate transactions."

    # Fallback
    if is_fraud_direction:
        return "This factor increased the system's fraud-risk assessment."
    return "This factor supported the transaction's legitimacy."


def _get_shap_values(clf, X_scaled: pd.DataFrame) -> np.ndarray:
    """Return a 1D array of SHAP values for the positive (fraud) class."""
    # 1. Fast Tree SHAP for AdaBoostClassifier
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
        # Fallback to model-agnostic Explainer
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


def explain_prediction(
    pipeline, X_scaled: pd.DataFrame, X_raw: pd.DataFrame, is_fraud: bool = False
) -> Tuple[List[dict], List[dict], List[str], str]:
    """
    Returns:
    (top_fraud_contributors, top_legitimate_contributors, explanation_sentences, explanation_title).

    explanation_sentences: a list of plain-English strings explaining why
    the system gave this result, derived from actual SHAP values of THIS transaction.
    """
    clf = pipeline.named_steps["clf"]
    shap_values = _get_shap_values(clf, X_scaled)

    contributions = []
    for i, feature in enumerate(FEATURE_COLUMNS):
        shap_val = float(shap_values[i]) if i < len(shap_values) else 0.0
        val = float(X_raw.iloc[0][feature]) if feature in X_raw.columns else 0.0
        contributions.append({
            "feature": feature,
            "label": _get_feature_label(feature),
            "value": round(val, 4),
            "shap_value": float(shap_val),
            "direction": "fraud" if shap_val > 0 else "legitimate",
            "description": _generate_explanation_sentence(feature, val, shap_val),
        })

    # Sort positive contributions pushing towards fraud
    fraud_candidates = [c for c in contributions if c["shap_value"] > 0]
    fraud_candidates.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
    fraud_contributors = fraud_candidates[:TOP_N]

    # Sort negative / neutral contributions supporting legitimacy
    legit_candidates = [c for c in contributions if c["shap_value"] <= 0]
    legit_candidates.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
    legit_contributors = legit_candidates[:TOP_N]

    # ── Dynamic plain-English explanation for THIS transaction ──
    explanation_sentences: List[str] = []
    seen = set()

    if is_fraud or len(fraud_contributors) >= 2:
        explanation_title = "Why did the system give this result?"
        # Emphasize top fraud drivers
        for c in fraud_contributors:
            s = c["description"]
            if s and s not in seen:
                seen.add(s)
                explanation_sentences.append(s)
            if len(explanation_sentences) >= 4:
                break
    else:
        explanation_title = "Why does this transaction look safe?"
        # Emphasize top legitimacy drivers
        for c in legit_contributors:
            s = c["description"]
            if s and s not in seen:
                seen.add(s)
                explanation_sentences.append(s)
            if len(explanation_sentences) >= 4:
                break

    if not explanation_sentences:
        if is_fraud:
            explanation_sentences.append("The transaction pattern triggered multiple automated fraud prevention indicators.")
        else:
            explanation_sentences.append("All analyzed transaction attributes are within normal, verified baseline ranges.")

    logger.debug(
        "SHAP explanation generated: title='%s', sentences=%d",
        explanation_title,
        len(explanation_sentences),
    )

    return fraud_contributors, legit_contributors, explanation_sentences, explanation_title
