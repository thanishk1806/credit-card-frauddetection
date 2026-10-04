"""
Dashboard + model performance endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.database.session import get_db
from app.ml.predictor import ModelNotTrainedError
from app.models.user import User
from app.schemas.dashboard import DashboardSummary
from app.services.dashboard_service import get_dashboard_summary
from app.services.live_stats_service import get_live_stats

router = APIRouter(tags=["Dashboard"])


@router.get("/dashboard/summary", response_model=DashboardSummary)
def dashboard_summary(current_user=Depends(get_current_user)):
    try:
        return get_dashboard_summary()
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))


@router.get("/dashboard/live-stats")
def live_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Returns real-time statistics from actual TrustCheck prediction records.
    Scoped to the current authenticated user's own predictions.
    """
    return get_live_stats(db, user_id=current_user.id)


@router.get("/models/performance", response_model=DashboardSummary)
def models_performance(current_user=Depends(get_current_user)):
    try:
        return get_dashboard_summary()
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))


@router.get("/models/best")
def best_model(current_user=Depends(get_current_user)):
    try:
        summary = get_dashboard_summary()
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    return summary["best_model"]
