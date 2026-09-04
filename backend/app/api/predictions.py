"""
Prediction history endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.database.session import get_db
from app.models.prediction import Prediction
from app.models.user import User
from app.schemas.predict import PredictionHistoryItem

router = APIRouter(tags=["Prediction History"])


@router.get("/predictions", response_model=list[PredictionHistoryItem])
def list_predictions(
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    records = (
        db.query(Prediction)
        .filter(Prediction.user_id == current_user.id)
        .order_by(Prediction.created_at.desc())
        .limit(limit)
        .all()
    )
    return records


@router.get("/predictions/{prediction_id}", response_model=PredictionHistoryItem)
def get_prediction(prediction_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        import uuid
        pred_uuid = uuid.UUID(str(prediction_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")

    record = db.query(Prediction).filter(
        Prediction.id == pred_uuid, Prediction.user_id == current_user.id
    ).first()
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")
    return record
