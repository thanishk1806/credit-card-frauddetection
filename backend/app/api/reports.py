"""
PDF report generation + download endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.database.session import get_db
from app.ml.predictor import load_evaluation_results
from app.models.prediction import Prediction
from app.models.user import User
from app.reports.pdf_generator import build_prediction_report

router = APIRouter(prefix="/reports", tags=["Reports"])


def _get_owned_prediction(prediction_id: str, db: Session, user: User) -> Prediction:
    try:
        import uuid
        pred_uuid = uuid.UUID(str(prediction_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")

    record = db.query(Prediction).filter(
        Prediction.id == pred_uuid, Prediction.user_id == user.id
    ).first()
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")
    return record


@router.post("/{prediction_id}")
def generate_report(prediction_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Generates (and returns as bytes) the PDF report; also usable to
    trigger generation before calling the /download endpoint."""
    record = _get_owned_prediction(prediction_id, db, current_user)
    try:
        model_metrics = None
        try:
            results = load_evaluation_results()
            model_metrics = next(
                (m for m in results["models"] if m["model_name"] == record.model_name), None
            )
        except Exception:
            model_metrics = None

        pdf_bytes = build_prediction_report(record, model_metrics)
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="PDF generation failed")

    return Response(content=pdf_bytes, media_type="application/pdf")


@router.get("/{prediction_id}/download")
def download_report(prediction_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    record = _get_owned_prediction(prediction_id, db, current_user)
    try:
        model_metrics = None
        try:
            results = load_evaluation_results()
            model_metrics = next(
                (m for m in results["models"] if m["model_name"] == record.model_name), None
            )
        except Exception:
            model_metrics = None

        pdf_bytes = build_prediction_report(record, model_metrics)
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="PDF generation failed")

    filename = f"fraud_report_{prediction_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
