"""
Transaction prediction + SHAP explanation endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.database.session import get_db
from app.ml.predictor import ModelNotTrainedError
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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction failed")

    shap_data = record.shap_explanation or {}
    return PredictionResponse(
        prediction_id=record.id,
        is_fraud=record.is_fraud,
        prediction_label="FRAUD" if record.is_fraud else "NOT FRAUD",
        fraud_probability=record.fraud_probability,
        legitimate_probability=round(1 - record.fraud_probability, 6),
        risk_score=record.risk_score,
        risk_level=record.risk_level,
        model_used=record.model_name,
        top_fraud_contributors=shap_data.get("top_fraud_contributors", []),
        top_legitimate_contributors=shap_data.get("top_legitimate_contributors", []),
        created_at=record.created_at,
    )


@router.post("/explain/{prediction_id}")
def explain(prediction_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Returns the stored SHAP explanation for a previously made prediction."""
    from app.models.prediction import Prediction
    import uuid

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
