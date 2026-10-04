"""
TrustCheck Transaction Risk Decision Layer.

A SEPARATE, deterministic, explainable risk assessment layer that evaluates
transaction suspiciousness based on user-entered transaction inputs.

============================================================================
IMPORTANT: This module does NOT modify, replace, or interfere with the
existing ML pipeline. The ML model's prediction, probability, and SHAP
values remain genuine and unaltered.
============================================================================

Architecture:
    User Inputs → Existing ML Pipeline → ML Probability (unchanged)
                                            ↓
    User Inputs → Transaction Risk Layer → Signal Scores
                                            ↓
                    Combined Assessment → FRAUD / LEGITIMATE + Risk Score

The TrustCheck final assessment combines:
1. ML model fraud probability (genuine, unmodified) — 20% weight
2. Transaction-input risk signals (computed here) — 80% weight

This weighting reflects the reality that the ML model receives zero-valued
PCA features (V1–V28) during online prediction with realistic inputs,
producing consistently low probabilities. The transaction-input signals
compensate by evaluating the human-readable transaction characteristics.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, List, Tuple

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ────────────────────────────────────────────────────────────
# Signal Weights (must sum to 1.0)
# ────────────────────────────────────────────────────────────
_WEIGHTS = {
    "amount":        0.20,
    "time":          0.10,
    "velocity":      0.15,
    "channel":       0.10,
    "card_present":  0.12,
    "international": 0.12,
    "merchant":      0.08,
    "combination":   0.13,
}

# ML vs transaction-risk blending
_ML_WEIGHT = 0.20
_TX_WEIGHT = 0.80

# Final assessment threshold (0-100 scale)
_FRAUD_THRESHOLD = 50.0

# Signal elevation threshold for combination scoring
_ELEVATED_THRESHOLD = 0.30


# ────────────────────────────────────────────────────────────
# Time-of-Day Risk Lookup Table (one entry per hour, 0-23)
# Peak risk: 2-4 AM, Lowest risk: 8 AM - 5 PM
# ────────────────────────────────────────────────────────────
_TIME_RISK_TABLE = [
    0.75,  # 00:00 midnight
    0.85,  # 01:00
    0.90,  # 02:00
    0.90,  # 03:00
    0.80,  # 04:00
    0.55,  # 05:00
    0.20,  # 06:00
    0.05,  # 07:00
    0.02,  # 08:00
    0.02,  # 09:00
    0.02,  # 10:00
    0.02,  # 11:00
    0.02,  # 12:00
    0.02,  # 13:00
    0.02,  # 14:00
    0.02,  # 15:00
    0.02,  # 16:00
    0.03,  # 17:00
    0.03,  # 18:00
    0.05,  # 19:00
    0.12,  # 20:00
    0.22,  # 21:00
    0.40,  # 22:00
    0.60,  # 23:00
]

# Channel risk scores
_CHANNEL_RISK: Dict[str, float] = {
    "pos":           0.02,
    "contactless":   0.04,
    "mobile_app":    0.15,
    "atm":           0.35,
    "online":        0.70,
    "wire_transfer": 0.90,
}

# Merchant category risk scores
_MERCHANT_RISK: Dict[str, float] = {
    "grocery":         0.02,
    "utilities":       0.03,
    "food_dining":     0.04,
    "fuel":            0.04,
    "healthcare":      0.06,
    "retail_shopping": 0.10,
    "other":           0.12,
    "entertainment":   0.35,
    "travel_airlines": 0.55,
    "electronics":     0.75,
    "luxury_jewelry":  0.90,
}


# ────────────────────────────────────────────────────────────
# Individual Signal Scoring Functions
# ────────────────────────────────────────────────────────────

def _amount_risk(amount: float) -> float:
    """Sigmoid-based amount risk centered at $3,000."""
    if amount <= 0:
        return 0.0
    x = (math.log1p(amount) - math.log1p(3000)) * 1.1
    return 1.0 / (1.0 + math.exp(-x))


def _time_risk(hour: float) -> float:
    """Time-of-day risk using interpolated lookup table."""
    hour = hour % 24.0
    idx = int(hour) % 24
    next_idx = (idx + 1) % 24
    frac = hour - int(hour)
    return _TIME_RISK_TABLE[idx] * (1.0 - frac) + _TIME_RISK_TABLE[next_idx] * frac


def _velocity_risk(velocity: float) -> float:
    """
    Continuous logarithmic risk scaling for recent transaction activity.
    Evaluates velocity count in 5 minutes without any artificial cutoff.
    """
    if velocity <= 0:
        return 0.0
    if velocity == 1:
        return 0.03
    if velocity <= 3:
        return 0.03 + (velocity - 1) * 0.08  # 2->0.11, 3->0.19
    return min(0.20 + 0.25 * math.log2(velocity - 2), 0.95)


def _channel_risk(channel: str) -> float:
    """Risk from payment channel type."""
    return _CHANNEL_RISK.get(channel.lower().strip(), 0.20)


def _card_present_risk(card_present: bool) -> float:
    """Card-Not-Present transactions carry higher authentication risk."""
    return 0.0 if card_present else 0.80


def _international_risk(is_international: bool) -> float:
    """Cross-border transactions carry higher jurisdiction/clearing risk."""
    return 0.85 if is_international else 0.0


def _merchant_risk(merchant_category: str) -> float:
    """Risk from merchant category."""
    return _MERCHANT_RISK.get(merchant_category.lower().strip(), 0.12)


# ────────────────────────────────────────────────────────────
# Combination Scoring
# ────────────────────────────────────────────────────────────

def _combination_score(signal_scores: Dict[str, float]) -> float:
    """
    Compute a combination penalty when multiple signals are elevated
    simultaneously. No single elevated signal produces a combination
    bonus — only 2+ concurrent elevated signals trigger compounding.
    """
    elevated = [v for k, v in signal_scores.items() if v > _ELEVATED_THRESHOLD]
    n_elevated = len(elevated)

    if n_elevated < 2:
        return 0.0

    avg_elevated = sum(elevated) / n_elevated
    return min(1.0, (n_elevated - 1) * 0.16 + avg_elevated * 0.35)


# ────────────────────────────────────────────────────────────
# Risk Factor Description Generator
# ────────────────────────────────────────────────────────────

def _describe_signal(signal_name: str, raw_value, risk: float) -> str:
    """Generate a human-readable description for a risk factor."""

    if signal_name == "amount":
        amt = float(raw_value)
        if risk < 0.15:
            return f"Transaction amount of ${amt:,.2f} is within normal spending range."
        elif risk < 0.40:
            return f"Transaction amount of ${amt:,.2f} is moderately above typical spending patterns."
        else:
            return f"Transaction amount of ${amt:,.2f} is significantly above average — high-value transaction."

    if signal_name == "time":
        hour = float(raw_value)
        h = int(hour)
        m = int((hour % 1) * 60)
        time_str = f"{h:02d}:{m:02d}"
        if risk < 0.10:
            return f"Transaction at {time_str} — standard business hours."
        elif risk < 0.35:
            return f"Transaction at {time_str} — evening hours, slightly elevated risk window."
        else:
            return f"Transaction at {time_str} — unusual late-night/early-morning hours."

    if signal_name == "velocity":
        count = int(float(raw_value))
        if risk < 0.10:
            return f"Normal transaction frequency ({count} in last 5 minutes)."
        elif risk < 0.35:
            return f"Moderately elevated activity ({count} transactions in 5 minutes)."
        else:
            return f"Rapid succession of {count} transactions within 5 minutes — unusual burst pattern."

    if signal_name == "channel":
        ch = str(raw_value).replace("_", " ").title()
        if risk < 0.15:
            return f"Transaction via {ch} — verified point-of-sale terminal."
        elif risk < 0.35:
            return f"Transaction via {ch} — standard digital channel."
        else:
            return f"Transaction via {ch} — remote/unverified channel with elevated risk."

    if signal_name == "card_present":
        if risk < 0.10:
            return "Physical card authentication verified at point of sale."
        else:
            return "Card-Not-Present (CNP) transaction — no physical card authentication."

    if signal_name == "international":
        if risk < 0.10:
            return "Domestic transaction within cardholder's home region."
        else:
            return "Cross-border transaction to international merchant — elevated risk."

    if signal_name == "merchant":
        cat = str(raw_value).replace("_", " ").title()
        if risk < 0.15:
            return f"Standard merchant category ({cat}) — routine spending."
        elif risk < 0.35:
            return f"Merchant category ({cat}) — moderate dispute risk."
        else:
            return f"High-risk merchant category ({cat}) — elevated chargeback/fraud risk."

    return f"Risk signal: {signal_name}"


# ────────────────────────────────────────────────────────────
# Main Assessment Function
# ────────────────────────────────────────────────────────────

def assess_transaction(transaction: dict, ml_fraud_probability: float) -> dict:
    """
    Run the TrustCheck Transaction Risk Decision Layer.

    Parameters
    ----------
    transaction : dict
        The user's transaction input (amount, timestamp, channel, etc.)
    ml_fraud_probability : float
        The genuine ML model fraud probability [0, 1].
        This value is NOT modified.

    Returns
    -------
    dict with keys:
        - trustcheck_is_fraud: bool
        - trustcheck_assessment: str ("FRAUD" or "LEGITIMATE")
        - trustcheck_risk_score: float (0-100)
        - trustcheck_risk_level: str ("LOW", "MEDIUM", "HIGH")
        - transaction_risk_score: float (0-100, before ML blending)
        - risk_factors: list of dicts
    """
    # ── Extract transaction inputs ──
    amount = float(transaction.get("amount") or transaction.get("Amount") or 0.0)

    # Resolve transaction hour
    timestamp_val = transaction.get("timestamp")
    if timestamp_val is not None:
        try:
            if isinstance(timestamp_val, str):
                dt = datetime.fromisoformat(timestamp_val.replace("Z", "+00:00"))
            elif isinstance(timestamp_val, (int, float)):
                dt = datetime.fromtimestamp(timestamp_val)
            else:
                dt = datetime.utcnow()
            hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        except Exception:
            hour = 12.0
    else:
        hour = 12.0

    velocity = float(transaction.get("tx_velocity_5m") or 0.0)
    channel = str(transaction.get("transaction_type") or "online").lower()
    card_present = bool(transaction.get("card_present", False))
    is_international = bool(transaction.get("international_transaction", False))
    merchant = str(transaction.get("merchant_category") or "retail_shopping").lower()

    # ── Compute individual signal scores ──
    signals = {
        "amount":        _amount_risk(amount),
        "time":          _time_risk(hour),
        "velocity":      _velocity_risk(velocity),
        "channel":       _channel_risk(channel),
        "card_present":  _card_present_risk(card_present),
        "international": _international_risk(is_international),
        "merchant":      _merchant_risk(merchant),
    }

    combo = _combination_score(signals)
    signals["combination"] = combo

    # ── Weighted sum → transaction risk score (0 to ~1) ──
    base_risk = sum(_WEIGHTS[k] * signals[k] for k in _WEIGHTS)
    transaction_risk_score = round(min(base_risk * 100, 100.0), 2)

    # ── Blend with ML probability → final risk score ──
    ml_contribution = ml_fraud_probability * 100.0
    final_risk_score = round(
        _ML_WEIGHT * ml_contribution + _TX_WEIGHT * transaction_risk_score,
        2,
    )
    final_risk_score = max(0.0, min(100.0, final_risk_score))

    # ── Final assessment ──
    # If ML itself predicts fraud (probability >= 0.5), always respect that
    if ml_fraud_probability >= 0.5:
        trustcheck_is_fraud = True
        final_risk_score = max(final_risk_score, ml_fraud_probability * 100.0)
    else:
        trustcheck_is_fraud = final_risk_score >= _FRAUD_THRESHOLD

    # Risk level (0-29: LOW, 30-69: MEDIUM, 70-100: HIGH)
    if final_risk_score < 30.0:
        risk_level = "LOW"
    elif final_risk_score < 70.0:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    # ── Build risk factor list for the UI ──
    signal_labels = {
        "amount":        "Transaction Amount",
        "time":          "Time of Day",
        "velocity":      "Recent Transaction Activity",
        "channel":       "Payment Channel",
        "card_present":  "Physical Card Present",
        "international": "Cross-Border Transaction",
        "merchant":      "Merchant Category",
    }

    signal_raw_values = {
        "amount":        amount,
        "time":          hour,
        "velocity":      velocity,
        "channel":       channel,
        "card_present":  card_present,
        "international": is_international,
        "merchant":      merchant,
    }

    risk_factors = []
    for key in ["amount", "time", "velocity", "channel", "card_present", "international", "merchant"]:
        score = signals[key]
        risk_factors.append({
            "signal": key,
            "label": signal_labels[key],
            "input_value": str(signal_raw_values[key]),
            "risk_score": round(score, 4),
            "risk_percent": round(score * 100, 1),
            "direction": "suspicious" if score > _ELEVATED_THRESHOLD else "normal",
            "description": _describe_signal(key, signal_raw_values[key], score),
        })

    # Sort: suspicious first, then by risk score descending
    risk_factors.sort(key=lambda f: (-1 if f["direction"] == "suspicious" else 0, -f["risk_score"]))

    # ── Logging ──
    logger.info("=" * 50)
    logger.info("TRUSTCHECK RISK LAYER ASSESSMENT")
    logger.info("=" * 50)
    for key in ["amount", "time", "velocity", "channel", "card_present", "international", "merchant"]:
        logger.info("  %-18s = %.4f", key, signals[key])
    logger.info("  %-18s = %.4f", "combination", combo)
    logger.info("  Transaction Risk Score: %.2f / 100", transaction_risk_score)
    logger.info("  ML Fraud Probability:   %.4f%%", ml_fraud_probability * 100)
    logger.info("  Final Risk Score:       %.2f / 100", final_risk_score)
    logger.info("  Final Assessment:       %s", "FRAUD" if trustcheck_is_fraud else "LEGITIMATE")
    logger.info("  Risk Level:             %s", risk_level)
    logger.info("=" * 50)

    return {
        "trustcheck_is_fraud": trustcheck_is_fraud,
        "trustcheck_assessment": "FRAUD" if trustcheck_is_fraud else "LEGITIMATE",
        "trustcheck_risk_score": final_risk_score,
        "trustcheck_risk_level": risk_level,
        "transaction_risk_score": transaction_risk_score,
        "risk_factors": risk_factors,
    }
