"""
Orchestrates a full prediction request: model inference + SHAP explanation
+ persistence of prediction history.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.ml.explainer import explain_prediction
from app.ml.predictor import get_best_model, predict_transaction
from app.models.prediction import Prediction


def run_prediction(db: Session, user_id: uuid.UUID, transaction: dict) -> Prediction:
    result = predict_transaction(transaction)
    _, pipeline = get_best_model()

    fraud_contributors, legit_contributors = explain_prediction(
        pipeline, result["scaled_input"], result["raw_input"]
    )

    record = Prediction(
        id=uuid.uuid4(),
        user_id=user_id,
        input_features=transaction,
        model_name=result["model_name"],
        is_fraud=result["is_fraud"],
        fraud_probability=result["fraud_probability"],
        risk_score=result["risk_score"],
        risk_level=result["risk_level"],
        shap_explanation={
            "top_fraud_contributors": fraud_contributors,
            "top_legitimate_contributors": legit_contributors,
        },
        created_at=datetime.now(timezone.utc),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record
