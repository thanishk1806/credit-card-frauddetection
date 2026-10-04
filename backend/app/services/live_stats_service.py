"""
Live statistics service for TrustCheck dashboard.

Queries actual prediction records from the database to compute
real application activity metrics — no hardcoded or training-set values.
"""
from __future__ import annotations

from sqlalchemy import func, case
from sqlalchemy.orm import Session

from app.models.prediction import Prediction


def get_live_stats(db: Session, user_id=None) -> dict:
    """
    Returns live dashboard statistics derived from actual prediction records.

    If user_id is provided, scopes to that user's records only.
    If user_id is None, queries all records (admin view — not used currently).
    """
    query = db.query(Prediction)
    if user_id is not None:
        query = query.filter(Prediction.user_id == user_id)

    total = query.count()
    fraud_count = query.filter(Prediction.is_fraud == True).count()  # noqa: E712
    legit_count = total - fraud_count

    detection_rate = round((fraud_count / total * 100), 2) if total > 0 else 0.0

    avg_risk_row = query.with_entities(func.avg(Prediction.risk_score)).scalar()
    avg_risk = round(float(avg_risk_row), 1) if avg_risk_row is not None else 0.0

    # Risk level distribution
    low_count = query.filter(Prediction.risk_level == "LOW").count()
    medium_count = query.filter(Prediction.risk_level == "MEDIUM").count()
    high_count = query.filter(Prediction.risk_level == "HIGH").count()

    return {
        "total_analyzed": total,
        "fraud_detected": fraud_count,
        "legitimate_count": legit_count,
        "detection_rate": detection_rate,
        "avg_risk_score": avg_risk,
        "prediction_distribution": {
            "legitimate": legit_count,
            "fraud": fraud_count,
        },
        "risk_level_distribution": {
            "low": low_count,
            "medium": medium_count,
            "high": high_count,
        },
    }
