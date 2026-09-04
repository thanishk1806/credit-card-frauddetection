"""
Defines the five ensemble models and their hyperparameter search spaces.

Kept in one place so the training script and any future retraining code
stay consistent.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

RANDOM_STATE = 42


def get_model_definitions() -> dict:
    """Returns {model_name: (estimator, param_distributions)}.

    Search spaces are intentionally small/reasonable for a mini-project
    (RandomizedSearchCV with a limited n_iter), not an exhaustive grid.
    """
    return {
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "clf__n_estimators": [100, 200, 300],
                "clf__max_depth": [6, 10, 16, None],
                "clf__min_samples_split": [2, 5, 10],
                "clf__min_samples_leaf": [1, 2, 4],
                "clf__max_features": ["sqrt", "log2"],
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
                use_label_encoder=False,
                n_jobs=-1,
            ),
            {
                "clf__n_estimators": [100, 200, 300],
                "clf__max_depth": [3, 5, 7, 9],
                "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "clf__subsample": [0.7, 0.85, 1.0],
                "clf__colsample_bytree": [0.7, 0.85, 1.0],
            },
        ),
        "LightGBM": (
            LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
            {
                "clf__n_estimators": [100, 200, 300],
                "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "clf__num_leaves": [15, 31, 63],
                "clf__max_depth": [-1, 6, 10],
                "clf__subsample": [0.7, 0.85, 1.0],
                "clf__colsample_bytree": [0.7, 0.85, 1.0],
            },
        ),
        "CatBoost": (
            CatBoostClassifier(random_state=RANDOM_STATE, verbose=False),
            {
                "clf__iterations": [200, 400, 600],
                "clf__depth": [4, 6, 8, 10],
                "clf__learning_rate": [0.01, 0.05, 0.1],
                "clf__l2_leaf_reg": [1, 3, 5, 7],
            },
        ),
    }
