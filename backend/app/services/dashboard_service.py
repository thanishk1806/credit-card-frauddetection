"""
Builds dashboard response payloads from the persisted evaluation_results.json
produced by scripts/train_models.py. No metrics are ever hard-coded here.
"""
from app.ml.predictor import load_evaluation_results


def get_dashboard_summary() -> dict:
    results = load_evaluation_results()

    dataset = results["dataset"]
    all_models = results["models"]
    best_meta = results["best_model"]

    best_metrics = next(m for m in all_models if m["model_name"] == best_meta["model_name"])

    return {
        "dataset": {
            "total_transactions": dataset["total_transactions"],
            "fraud_transactions": dataset["fraud_transactions"],
            "legitimate_transactions": dataset["legitimate_transactions"],
            "fraud_percentage": dataset["fraud_percentage"],
            "class_distribution_before_smote": dataset["class_distribution_before_smote"],
            "class_distribution_after_smote": dataset["class_distribution_after_smote"],
        },
        "best_model": {
            "model_name": best_meta["model_name"],
            "selection_metric": best_meta["selection_metric"],
            "metrics": best_metrics,
            "reason": best_meta["reason"],
        },
        "all_models": all_models,
        "trained_at": results["trained_at"],
    }
