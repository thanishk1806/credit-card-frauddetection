"""
Dashboard + model performance endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status

from app.auth.dependencies import get_current_user
from app.ml.predictor import ModelNotTrainedError
from app.schemas.dashboard import DashboardSummary
from app.services.dashboard_service import get_dashboard_summary

router = APIRouter(tags=["Dashboard"])


@router.get("/dashboard/summary", response_model=DashboardSummary)
def dashboard_summary(current_user=Depends(get_current_user)):
    try:
        return get_dashboard_summary()
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))


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
