"""
Pydantic schemas for dashboard / model-performance endpoints.
"""
from typing import Dict, List, Optional

from pydantic import BaseModel


class DatasetOverview(BaseModel):
    total_transactions: int
    fraud_transactions: int
    legitimate_transactions: int
    fraud_percentage: float
    class_distribution_before_smote: Dict[str, int]
    class_distribution_after_smote: Dict[str, int]


class ModelMetrics(BaseModel):
    model_name: str
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    mcc: float
    accuracy: float
    confusion_matrix: List[List[int]]
    roc_curve: Dict[str, List[float]]
    pr_curve: Dict[str, List[float]]
    feature_importance: Optional[Dict[str, float]] = None


class BestModelInfo(BaseModel):
    model_name: str
    selection_metric: str
    metrics: ModelMetrics
    reason: str


class DashboardSummary(BaseModel):
    dataset: DatasetOverview
    best_model: BestModelInfo
    all_models: List[ModelMetrics]
    trained_at: str
