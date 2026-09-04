"""
Pydantic schemas for transaction prediction.

Fields correspond to the credit card fraud feature set:
Time, V1..V28, Amount. Default values are provided for secondary PCA components (0.0)
so users can predict transactions by submitting just the primary intuitive fields (Amount, Time, V4, V14, V17).
"""
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model

# Default V1..V28 to 0.0 (normalized PCA mean) so user can submit 4-5 core fields
_v_fields = {f"V{i}": (float, Field(default=0.0, description=f"PCA component V{i}")) for i in range(1, 29)}

TransactionInput = create_model(
    "TransactionInput",
    Amount=(float, Field(..., ge=0, description="Transaction amount in USD")),
    Time=(float, Field(default=0.0, ge=0, description="Seconds elapsed since the first transaction")),
    **_v_fields,
)


class ShapContribution(BaseModel):
    feature: str
    value: float
    shap_value: float
    direction: str  # "fraud" or "legitimate"


class PredictionResponse(BaseModel):
    prediction_id: uuid.UUID
    is_fraud: bool
    prediction_label: str
    fraud_probability: float
    legitimate_probability: float
    risk_score: float
    risk_level: str
    model_used: str
    top_fraud_contributors: List[ShapContribution]
    top_legitimate_contributors: List[ShapContribution]
    created_at: datetime


class PredictionHistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    is_fraud: bool
    fraud_probability: float
    risk_score: float
    risk_level: str
    model_name: str
    created_at: datetime
