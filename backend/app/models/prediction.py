"""
Prediction history ORM model.

Only the model input features + prediction outcome are stored.
No sensitive real-world card data (PAN, CVV, etc.) is ever persisted
because the dataset itself does not contain such fields.
"""
import uuid
from datetime import datetime

from sqlalchemy import Column, String, DateTime, Float, Boolean, ForeignKey, JSON, Uuid

from app.database.session import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(Uuid(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)

    # Raw model input features (Time, V1..V28, Amount) stored as JSON
    input_features = Column(JSON, nullable=False)

    model_name = Column(String(64), nullable=False)
    is_fraud = Column(Boolean, nullable=False)
    fraud_probability = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=False)
    risk_level = Column(String(16), nullable=False)

    # Top SHAP contributors, stored so PDF reports can be regenerated later
    shap_explanation = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
