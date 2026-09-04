"""
Centralized application configuration.

All secrets and environment-specific values are loaded from environment
variables (via a .env file in development). Nothing sensitive is
hard-coded here.
"""
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "Intelligent Credit Card Fraud Detection System"
    ENV: str = "development"

    # Database
    DATABASE_URL: str = "sqlite:///./fraud_detection.db"

    # JWT
    JWT_SECRET_KEY: str = "insecure-dev-secret-change-me"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60

    # CORS
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Dataset / models
    DATASET_PATH: str = "../data/creditcard.csv"
    MODEL_DIR: str = "../ml_models"

    # Risk thresholds
    RISK_LOW_MAX: int = 30
    RISK_MEDIUM_MAX: int = 70

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def dataset_path_resolved(self) -> Path:
        p = Path(self.DATASET_PATH)
        return p if p.is_absolute() else (BASE_DIR / p).resolve()

    @property
    def model_dir_resolved(self) -> Path:
        p = Path(self.MODEL_DIR)
        resolved = p if p.is_absolute() else (BASE_DIR / p).resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved


@lru_cache
def get_settings() -> Settings:
    return Settings()
