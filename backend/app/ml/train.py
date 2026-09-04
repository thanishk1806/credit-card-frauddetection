"""
Core training pipeline logic.

Pipeline (data-leakage-safe):

    Raw dataset
      -> cleaning (app.ml.preprocessing.load_dataset)
      -> train/test split (BEFORE any resampling)
      -> StandardScaler fit on train only, applied to both splits
      -> imblearn Pipeline(SMOTE -> classifier), so SMOTE is re-applied
         inside every cross-validation fold during hyperparameter search
      -> RandomizedSearchCV per model (scoring="f1" on the minority class)
      -> final fit of the tuned pipeline on the full (scaled) training set
         (SMOTE runs once here, only on training data)
      -> evaluation on the ORIGINAL, untouched, non-SMOTE test set
      -> best-model selection primarily by F1-score (PR-AUC as tie-break)
      -> persistence of models, scaler, and evaluation results to MODEL_DIR
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold

from app.ml.model_registry import get_model_definitions, RANDOM_STATE
from app.ml.preprocessing import (
    load_dataset,
    split_features_target,
    fit_scaler,
    apply_scaler,
    save_preprocessing,
    FEATURE_COLUMNS,
)

EVAL_RESULTS_FILENAME = "evaluation_results.json"
MODEL_FILENAME_TEMPLATE = "model_{name}.joblib"
BEST_MODEL_META_FILENAME = "best_model.json"

# Primary criterion for best-model selection. F1 (not accuracy) is used
# because accuracy is dominated by the overwhelming legitimate-class
# majority in this highly imbalanced dataset.
SELECTION_METRIC = "f1_score"

N_SEARCH_ITER = 8  # kept small/reasonable for a mini project
CV_FOLDS = 3


def _safe_model_key(name: str) -> str:
    return name.lower().replace(" ", "_")


def _curve_subsample(x: np.ndarray, y: np.ndarray, max_points: int = 100) -> Tuple[list, list]:
    """Downsample curve points so evaluation_results.json stays small."""
    if len(x) <= max_points:
        return x.tolist(), y.tolist()
    idx = np.linspace(0, len(x) - 1, max_points).astype(int)
    return x[idx].tolist(), y[idx].tolist()


def evaluate_model(pipeline: ImbPipeline, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)

    metrics = {
        "model_name": model_name,
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "mcc": float(matthews_corrcoef(y_test, y_pred)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    roc_fpr, roc_tpr = _curve_subsample(fpr, tpr)
    pr_rec, pr_prec = _curve_subsample(rec_curve, prec_curve)
    metrics["roc_curve"] = {"fpr": roc_fpr, "tpr": roc_tpr}
    metrics["pr_curve"] = {"recall": pr_rec, "precision": pr_prec}

    # Feature importance where supported
    clf = pipeline.named_steps["clf"]
    importance = None
    if hasattr(clf, "feature_importances_"):
        importance = dict(zip(FEATURE_COLUMNS, [float(v) for v in clf.feature_importances_]))
    metrics["feature_importance"] = importance

    return metrics


def run_training_pipeline(dataset_path: Path, model_dir: Path) -> dict:
    print("=" * 50)
    print("STEP 1/8: Loading and cleaning dataset")
    print("=" * 50)
    df = load_dataset(dataset_path)
    X, y = split_features_target(df)

    total = len(df)
    fraud_count = int(y.sum())
    legit_count = total - fraud_count
    print(f"Total transactions: {total} | Fraud: {fraud_count} | Legitimate: {legit_count}")

    print("\nSTEP 2/8: Train/test split (BEFORE SMOTE)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    class_dist_before = {"legitimate": int((y_train == 0).sum()), "fraud": int((y_train == 1).sum())}
    print(f"Training class distribution BEFORE SMOTE: {class_dist_before}")

    print("\nSTEP 3/8: Fitting scaler on training data only (Time, Amount)")
    scaler = fit_scaler(X_train)
    X_train_scaled = apply_scaler(X_train, scaler)
    X_test_scaled = apply_scaler(X_test, scaler)
    save_preprocessing(scaler, model_dir)

    # Report post-SMOTE distribution for the dashboard (computed once, not
    # part of the actual CV pipeline, purely informational)
    smote_preview = SMOTE(random_state=RANDOM_STATE)
    _, y_res_preview = smote_preview.fit_resample(X_train_scaled, y_train)
    class_dist_after = {
        "legitimate": int((y_res_preview == 0).sum()),
        "fraud": int((y_res_preview == 1).sum()),
    }
    print(f"Training class distribution AFTER SMOTE (preview): {class_dist_after}")

    print("\nSTEP 4/8: Training + hyperparameter tuning for 5 models")
    model_defs = get_model_definitions()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    all_metrics = []
    fitted_pipelines: Dict[str, ImbPipeline] = {}

    for name, (estimator, param_dist) in model_defs.items():
        print(f"\n--- Training {name} ---")
        pipeline = ImbPipeline(steps=[
            ("smote", SMOTE(random_state=RANDOM_STATE)),  # only ever touches training folds
            ("clf", estimator),
        ])

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=N_SEARCH_ITER,
            scoring="f1",  # imbalance-aware metric, not accuracy
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        search.fit(X_train_scaled, y_train)
        best_pipeline = search.best_estimator_
        print(f"Best params for {name}: {search.best_params_}")

        metrics = evaluate_model(best_pipeline, X_test_scaled, y_test, name)
        print(
            f"{name} -> Precision: {metrics['precision']:.4f} Recall: {metrics['recall']:.4f} "
            f"F1: {metrics['f1_score']:.4f} ROC-AUC: {metrics['roc_auc']:.4f} "
            f"PR-AUC: {metrics['pr_auc']:.4f} MCC: {metrics['mcc']:.4f}"
        )

        all_metrics.append(metrics)
        fitted_pipelines[name] = best_pipeline

    print("\nSTEP 5/8: Selecting best model")
    best = max(all_metrics, key=lambda m: (m[SELECTION_METRIC], m["pr_auc"]))
    best_name = best["model_name"]
    print(f"BEST MODEL: {best_name} (selected by {SELECTION_METRIC} = {best[SELECTION_METRIC]:.4f})")

    print("\nSTEP 6/8: Persisting trained models")
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, pipeline in fitted_pipelines.items():
        joblib.dump(pipeline, model_dir / MODEL_FILENAME_TEMPLATE.format(name=_safe_model_key(name)))

    print("\nSTEP 7/8: Persisting evaluation results")
    results = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "total_transactions": total,
            "fraud_transactions": fraud_count,
            "legitimate_transactions": legit_count,
            "fraud_percentage": round(100 * fraud_count / total, 4),
            "class_distribution_before_smote": class_dist_before,
            "class_distribution_after_smote": class_dist_after,
        },
        "models": all_metrics,
        "best_model": {
            "model_name": best_name,
            "selection_metric": SELECTION_METRIC,
            "reason": (
                f"{best_name} achieved the highest {SELECTION_METRIC.replace('_', ' ')} "
                f"({best[SELECTION_METRIC]:.4f}) among all five models on the held-out "
                "test set, with PR-AUC used as a tie-breaker. F1/PR-AUC are preferred over "
                "accuracy because fraud is a small minority class."
            ),
        },
    }
    with open(model_dir / EVAL_RESULTS_FILENAME, "w") as f:
        json.dump(results, f, indent=2)

    with open(model_dir / BEST_MODEL_META_FILENAME, "w") as f:
        json.dump({"model_name": best_name, "model_key": _safe_model_key(best_name)}, f, indent=2)

    print("\nSTEP 8/8: Training complete")
    print("=" * 50)
    print("MODEL TRAINING RESULTS")
    print("=" * 50)
    for m in all_metrics:
        print(
            f"\n{m['model_name']}\n"
            f"  Precision: {m['precision']:.4f}\n"
            f"  Recall:    {m['recall']:.4f}\n"
            f"  F1:        {m['f1_score']:.4f}\n"
            f"  ROC-AUC:   {m['roc_auc']:.4f}\n"
            f"  PR-AUC:    {m['pr_auc']:.4f}\n"
            f"  MCC:       {m['mcc']:.4f}"
        )
    print(f"\nBEST MODEL: {best_name}")
    print("=" * 50)

    return results
