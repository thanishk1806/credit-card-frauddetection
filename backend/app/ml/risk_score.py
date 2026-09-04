"""
Deterministic fraud risk score derivation.

The risk score is ALWAYS derived directly from the model's predicted fraud
probability - never randomly generated. Thresholds are configurable via
Settings (RISK_LOW_MAX / RISK_MEDIUM_MAX).
"""
from app.config import get_settings


def probability_to_risk_score(fraud_probability: float) -> float:
    """Map a probability in [0, 1] to a 0-100 risk score."""
    return round(float(fraud_probability) * 100, 2)


def risk_score_to_level(risk_score: float) -> str:
    settings = get_settings()
    if risk_score <= settings.RISK_LOW_MAX:
        return "LOW"
    if risk_score <= settings.RISK_MEDIUM_MAX:
        return "MEDIUM"
    return "HIGH"
