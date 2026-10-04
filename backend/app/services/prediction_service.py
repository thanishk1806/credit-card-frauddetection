"""
Orchestrates a full prediction request:
Realistic Input -> Feature Engineering -> Preprocessing -> Model Inference
-> Risk Scoring -> SHAP Explainability -> DB Persistence.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.ml.explainer import explain_prediction
from app.ml.predictor import get_best_model, predict_transaction
from app.ml.transaction_risk_layer import assess_transaction
from app.models import Prediction  # ensures User + Prediction both registered in metadata


def run_prediction(db: Session, user_id: uuid.UUID, transaction: dict) -> Prediction:
    result = predict_transaction(transaction)
    best_name, pipeline = get_best_model()

    fraud_contributors, legit_contributors, explanation_sentences, explanation_title = explain_prediction(
        pipeline, result["scaled_input"], result["raw_input"], is_fraud=result["is_fraud"]
    )

    # TrustCheck Transaction Risk Decision Layer (separate from ML pipeline)
    risk_assessment = assess_transaction(transaction, result["fraud_probability"])

    # Derive human-friendly top factors
    top_factors = []
    for c in fraud_contributors[:3]:
        top_factors.append({
            "feature": c["feature"],
            "label": c.get("label", c["feature"]),
            "impact": round(abs(c["shap_value"]), 4),
            "direction": "fraud",
            "description": c.get("description", f"Factor {c['feature']} pushed toward fraud."),
        })
    for c in legit_contributors[:2]:
        top_factors.append({
            "feature": c["feature"],
            "label": c.get("label", c["feature"]),
            "impact": round(abs(c["shap_value"]), 4),
            "direction": "legitimate",
            "description": c.get("description", f"Factor {c['feature']} supported legitimacy."),
        })

    # Parse realistic metadata
    amt = float(transaction.get("amount") or transaction.get("Amount") or 0.0)
    tx_type = transaction.get("transaction_type", "online")
    merchant = transaction.get("merchant_category", "retail_shopping")
    loc = transaction.get("location", "Hyderabad, India")
    device = transaction.get("device_type", "mobile")
    card_pres = bool(transaction.get("card_present", False))
    intl = bool(transaction.get("international_transaction", False))

    raw_input_dict = result["raw_input"].iloc[0].to_dict()

    record = Prediction(
        id=uuid.uuid4(),
        user_id=user_id,
        amount=amt,
        transaction_type=tx_type,
        merchant_category=merchant,
        location=loc,
        device_type=device,
        card_present=card_pres,
        international_transaction=intl,
        transaction_timestamp=datetime.now(timezone.utc),
        input_features=raw_input_dict,
        derived_features={
            "transaction_hour": raw_input_dict.get("transaction_hour"),
            "log_amount": raw_input_dict.get("log_amount"),
            "tx_velocity_5m": raw_input_dict.get("tx_velocity_5m"),
            "time_since_prev": raw_input_dict.get("time_since_prev"),
            "risk_feature_sum": raw_input_dict.get("risk_feature_sum"),
            # Separate Genuine ML outputs
            "ml_prediction": "FRAUD" if result["is_fraud"] else "LEGITIMATE",
            "ml_is_fraud": result["is_fraud"],
            "ml_fraud_probability": result["fraud_probability"],
            "ml_risk_score": result["risk_score"],
            "ml_risk_level": result["risk_level"],
            # TrustCheck Final Assessment
            "trustcheck_assessment": risk_assessment["trustcheck_assessment"],
            "trustcheck_is_fraud": risk_assessment["trustcheck_is_fraud"],
            "trustcheck_risk_score": risk_assessment["trustcheck_risk_score"],
            "trustcheck_risk_level": risk_assessment["trustcheck_risk_level"],
            "transaction_risk_score": risk_assessment["transaction_risk_score"],
            "risk_factors": risk_assessment["risk_factors"],
        },
        model_name=result["model_name"],
        is_fraud=risk_assessment["trustcheck_is_fraud"],
        fraud_probability=result["fraud_probability"],
        risk_score=risk_assessment["trustcheck_risk_score"],
        risk_level=risk_assessment["trustcheck_risk_level"],
        shap_explanation={
            "explanation": explanation_sentences,
            "explanation_title": explanation_title,
            "model_predictions": result["model_predictions"],
            "top_factors": top_factors,
            "top_fraud_contributors": fraud_contributors,
            "top_legitimate_contributors": legit_contributors,
            "decision_layer": {
                "trustcheck_assessment": risk_assessment["trustcheck_assessment"],
                "trustcheck_is_fraud": risk_assessment["trustcheck_is_fraud"],
                "trustcheck_risk_score": risk_assessment["trustcheck_risk_score"],
                "trustcheck_risk_level": risk_assessment["trustcheck_risk_level"],
                "transaction_risk_score": risk_assessment["transaction_risk_score"],
                "risk_factors": risk_assessment["risk_factors"],
                "ml_prediction": "FRAUD" if result["is_fraud"] else "LEGITIMATE",
                "ml_fraud_probability": result["fraud_probability"],
                "ml_risk_score": result["risk_score"],
                "ml_risk_level": result["risk_level"],
            },
        },
        created_at=datetime.now(timezone.utc),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record
