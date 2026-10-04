"""
Defines the five ensemble models and their hyperparameter search spaces.
"""
from __future__ import annotations

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42


def get_model_definitions() -> dict:
    """Returns {model_name: (estimator, param_distributions)}.

    Search spaces are balanced for fast and accurate hyperparameter tuning.
    """
    return {
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [6, 10, 16, None],
                "clf__min_samples_split": [2, 5],
                "clf__min_samples_leaf": [1, 2],
            },
        ),
        "AdaBoost": (
            AdaBoostClassifier(random_state=RANDOM_STATE),
            {
                "clf__n_estimators": [50, 100, 200],
                "clf__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=-1,
            ),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [3, 5, 7],
                "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "clf__subsample": [0.7, 0.85, 1.0],
                "clf__colsample_bytree": [0.7, 0.85, 1.0],
            },
        ),
        "LightGBM": (
            LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
            {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "clf__num_leaves": [15, 31],
                "clf__max_depth": [-1, 6],
                "clf__subsample": [0.7, 0.85, 1.0],
                "clf__colsample_bytree": [0.7, 0.85, 1.0],
            },
        ),
        "CatBoost": (
            CatBoostClassifier(
                random_state=RANDOM_STATE,
                verbose=False,
                allow_writing_files=False,
                thread_count=1,
            ),
            {
                "clf__iterations": [100, 200],
                "clf__depth": [4, 6, 8],
                "clf__learning_rate": [0.01, 0.05, 0.1],
                "clf__l2_leaf_reg": [1, 3, 5],
            },
        ),
    }
