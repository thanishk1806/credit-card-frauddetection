"""
Pydantic schemas for realistic transaction prediction.

Separated into 3 layers:
Layer 1: Raw realistic transaction fields (amount, timestamp, type, merchant, location, device, velocity)
Layer 2: Derived / engineered features (time of day, log amount, velocity, risk flags)
Layer 3: Model features & explainable prediction results (per-model predictions + SHAP)
"""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShapContribution(BaseModel):
    feature: str
    label: Optional[str] = None
    value: float
    shap_value: float
    direction: str  # "fraud" or "legitimate"
    description: Optional[str] = None


class TopFactor(BaseModel):
    feature: str
    label: str
    impact: float
    direction: str
    description: str


class SingleModelPrediction(BaseModel):
    model_key: str
    model_name: str
    prediction: str  # "FRAUD" or "LEGITIMATE"
    is_fraud: bool
    fraud_probability: float
    legitimate_probability: float
    risk_score: float
    risk_level: str  # "LOW", "MEDIUM", "HIGH"


class TransactionInput(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    # Realistic user-facing transaction fields
    amount: Optional[float] = Field(None, gt=0, description="Transaction amount in local currency/USD (must be > 0)")
    timestamp: Optional[Union[str, datetime]] = Field(None, description="ISO timestamp or transaction date-time")
    transaction_type: Optional[str] = Field("online", description="Transaction type: online, pos, atm, mobile_app, contactless, wire_transfer")
    merchant_category: Optional[str] = Field("retail_shopping", description="Merchant category: grocery, electronics, food_dining, fuel, travel_airlines, luxury_jewelry, etc.")
    location: Optional[str] = Field("Hyderabad, India", description="Transaction location (city, country)")
    tx_velocity_5m: Optional[float] = Field(0.0, ge=0, description="Number of transactions made in the previous 5 minutes")
    card_present: Optional[bool] = Field(False, description="Whether the physical card was present")
    international_transaction: Optional[bool] = Field(False, description="Whether transaction was cross-border")

    # Optional internal fallback fields for backward compatibility
    device_type: Optional[str] = Field("mobile", description="Device type fallback")
    time_since_prev: Optional[float] = Field(1800.0, ge=0, description="Elapsed seconds since prior transaction")

    # Legacy fields for backward compatibility with existing tests
    Amount: Optional[float] = Field(None, ge=0, description="Legacy Amount field")
    Time: Optional[float] = Field(0.0, ge=0, description="Legacy Time offset in seconds")

    @model_validator(mode="before")
    @classmethod
    def validate_amount_presence(cls, values: Any) -> Any:
        if isinstance(values, dict):
            amt = values.get("amount")
            legacy_amt = values.get("Amount")
            if amt is None and legacy_amt is None:
                raise ValueError("Transaction amount is required and must be greater than 0.")
            if amt is not None and amt <= 0:
                raise ValueError("Transaction amount must be strictly greater than 0.")
            if legacy_amt is not None and amt is None:
                values["amount"] = float(legacy_amt)
            elif amt is not None and legacy_amt is None:
                values["Amount"] = float(amt)
        return values


class RiskFactor(BaseModel):
    signal: str
    label: str
    input_value: str
    risk_score: float
    risk_percent: float
    direction: str  # "suspicious" or "normal"
    description: str


class PredictionResponse(BaseModel):
    prediction_id: uuid.UUID

    # TrustCheck Final Assessment
    prediction: str  # "FRAUD" or "LEGITIMATE"
    is_fraud: bool
    prediction_label: str  # "FRAUD" or "NOT FRAUD"
    risk_score: float
    risk_level: str  # "LOW", "MEDIUM", "HIGH"

    # Separate Genuine ML Model Output
    ml_prediction: Optional[str] = "LEGITIMATE"
    ml_is_fraud: Optional[bool] = False
    ml_fraud_probability: Optional[float] = None
    ml_legitimate_probability: Optional[float] = None
    ml_risk_score: Optional[float] = None
    ml_risk_level: Optional[str] = None
    fraud_probability: float  # Genuine ML probability kept for backward compatibility
    legitimate_probability: float

    # TrustCheck Decision Layer Specifics
    trustcheck_assessment: Optional[str] = None
    trustcheck_risk_score: Optional[float] = None
    trustcheck_risk_level: Optional[str] = None
    transaction_risk_score: Optional[float] = None
    risk_factors: List[RiskFactor] = Field(default_factory=list)

    # Models & SHAP
    model_used: str
    model_predictions: Dict[str, SingleModelPrediction] = Field(default_factory=dict)
    explanation: List[str] = Field(default_factory=list)
    explanation_title: Optional[str] = None
    top_factors: List[TopFactor] = Field(default_factory=list)
    top_fraud_contributors: List[ShapContribution] = Field(default_factory=list)
    top_legitimate_contributors: List[ShapContribution] = Field(default_factory=list)
    transaction_summary: Optional[Dict[str, Any]] = None
    created_at: datetime


class PredictionHistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    is_fraud: bool
    fraud_probability: float
    risk_score: float
    risk_level: str
    model_name: str
    amount: Optional[float] = None
    transaction_type: Optional[str] = None
    merchant_category: Optional[str] = None
    location: Optional[str] = None
    created_at: datetime
