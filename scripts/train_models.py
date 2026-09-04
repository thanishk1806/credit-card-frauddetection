#!/usr/bin/env python3
"""
Model training script.

Usage:
    cd backend
    python ../scripts/train_models.py

Loads the dataset (path from DATASET_PATH env var / .env), runs the full
data-leakage-safe pipeline (clean -> split -> scale -> SMOTE-in-CV ->
hyperparameter search -> evaluate -> select best -> persist), and writes
trained models + evaluation_results.json to MODEL_DIR.
"""
import sys
from pathlib import Path

# Allow running from repo root or scripts/ directly
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.config import get_settings  # noqa: E402
from app.ml.train import run_training_pipeline  # noqa: E402


def main():
    settings = get_settings()
    dataset_path = settings.dataset_path_resolved
    model_dir = settings.model_dir_resolved

    print(f"Dataset path: {dataset_path}")
    print(f"Model output directory: {model_dir}\n")

    run_training_pipeline(dataset_path, model_dir)


if __name__ == "__main__":
    main()
