"""
Transaction prediction + SHAP explanation endpoints.
"""
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.database.session import get_db
from app.ml.predictor import ModelNotTrainedError
from app.models.prediction import Prediction
from app.models.user import User
from app.schemas.predict import PredictionResponse, TransactionInput
from app.services.prediction_service import run_prediction

router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict(
    payload: TransactionInput,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        record = run_prediction(db, current_user.id, payload.model_dump())
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {str(e)}")

    shap_data = record.shap_explanation or {}
    decision_layer = shap_data.get("decision_layer") or {}
    model_predictions = shap_data.get("model_predictions", {})
    explanation = shap_data.get("explanation", [])
    explanation_title = shap_data.get("explanation_title")

    ml_prob = record.fraud_probability
    ml_is_fraud = decision_layer.get("ml_is_fraud", ml_prob >= 0.5)
    ml_pred = decision_layer.get("ml_prediction", "FRAUD" if ml_is_fraud else "LEGITIMATE")
    ml_score = decision_layer.get("ml_risk_score", round(ml_prob * 100, 2))
    ml_level = decision_layer.get("ml_risk_level", "LOW" if ml_score < 30 else ("MEDIUM" if ml_score < 70 else "HIGH"))

    tc_assessment = decision_layer.get("trustcheck_assessment", "FRAUD" if record.is_fraud else "LEGITIMATE")
    tc_score = decision_layer.get("trustcheck_risk_score", record.risk_score)
    tc_level = decision_layer.get("trustcheck_risk_level", record.risk_level)
    tx_score = decision_layer.get("transaction_risk_score", record.risk_score)
    risk_factors = decision_layer.get("risk_factors", [])

    return PredictionResponse(
        prediction_id=record.id,
        # TrustCheck Final Assessment
        prediction=tc_assessment,
        is_fraud=record.is_fraud,
        prediction_label="FRAUD" if record.is_fraud else "NOT FRAUD",
        risk_score=record.risk_score,
        risk_level=record.risk_level,

        # Separate Genuine ML Model Output
        ml_prediction=ml_pred,
        ml_is_fraud=ml_is_fraud,
        ml_fraud_probability=ml_prob,
        ml_legitimate_probability=round(1.0 - ml_prob, 6),
        ml_risk_score=ml_score,
        ml_risk_level=ml_level,
        fraud_probability=ml_prob,
        legitimate_probability=round(1.0 - ml_prob, 6),

        # TrustCheck Decision Layer Output
        trustcheck_assessment=tc_assessment,
        trustcheck_risk_score=tc_score,
        trustcheck_risk_level=tc_level,
        transaction_risk_score=tx_score,
        risk_factors=risk_factors,

        model_used=record.model_name,
        model_predictions=model_predictions,
        explanation=explanation,
        explanation_title=explanation_title,
        top_factors=shap_data.get("top_factors", []),
        top_fraud_contributors=shap_data.get("top_fraud_contributors", []),
        top_legitimate_contributors=shap_data.get("top_legitimate_contributors", []),
        transaction_summary={
            "amount": record.amount,
            "transaction_type": record.transaction_type,
            "merchant_category": record.merchant_category,
            "location": record.location,
            "tx_velocity_5m": record.derived_features.get("tx_velocity_5m") if record.derived_features else 0.0,
            "card_present": record.card_present,
            "international_transaction": record.international_transaction,
        },
        created_at=record.created_at,
    )


@router.post("/explain/{prediction_id}")
def explain(prediction_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Returns the stored SHAP explanation for a previously made prediction."""
    try:
        pred_uuid = uuid.UUID(str(prediction_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")

    record = db.query(Prediction).filter(
        Prediction.id == pred_uuid, Prediction.user_id == current_user.id
    ).first()
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")
    return record.shap_explanation
