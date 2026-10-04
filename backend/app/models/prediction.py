"""
Prediction history ORM model.

Stores realistic transaction metadata + model inputs + prediction outcomes.
No sensitive real-world card data (PAN, CVV, PIN) is ever persisted.
"""
import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, JSON, String, Uuid

from app.database.session import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(Uuid(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)

    # Realistic transaction metadata
    amount = Column(Float, nullable=True)
    transaction_type = Column(String(32), nullable=True)
    merchant_category = Column(String(32), nullable=True)
    location = Column(String(128), nullable=True)
    device_type = Column(String(32), nullable=True)
    card_present = Column(Boolean, nullable=True, default=False)
    international_transaction = Column(Boolean, nullable=True, default=False)
    transaction_timestamp = Column(DateTime, nullable=True)

    # Raw model input features & derived features stored as JSON
    input_features = Column(JSON, nullable=False)
    derived_features = Column(JSON, nullable=True)

    model_name = Column(String(64), nullable=False)
    is_fraud = Column(Boolean, nullable=False)
    fraud_probability = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=False)
    risk_level = Column(String(16), nullable=False)

    # Top SHAP contributors, stored so PDF reports can be regenerated later
    shap_explanation = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
