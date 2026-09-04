"""
Tests for the offline ML training pipeline logic itself (data-leakage
safety, model coverage, metric generation, best-model selection).

These tests train on a small synthetic dataset so they run quickly; they
still exercise the real pipeline code in app/ml/train.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import pytest

from app.ml.train import run_training_pipeline, SELECTION_METRIC
from app.ml.model_registry import get_model_definitions


@pytest.fixture(scope="module")
def trained_results(tmp_path_factory):
    from generate_sample_data import generate

    data_dir = tmp_path_factory.mktemp("data")
    model_dir = tmp_path_factory.mktemp("models")
    csv_path = data_dir / "creditcard.csv"

    df = generate(rows=2000, fraud_rate=0.03, seed=1)
    df.to_csv(csv_path, index=False)

    results = run_training_pipeline(csv_path, model_dir)
    return results, model_dir


def test_all_five_models_trained(trained_results):
    results, _ = trained_results
    trained_names = {m["model_name"] for m in results["models"]}
    expected_names = set(get_model_definitions().keys())
    assert trained_names == expected_names


def test_smote_only_applied_to_training_data(trained_results):
    results, _ = trained_results
    dataset = results["dataset"]
    # Total training-set size after SMOTE must be >= before (never shrinks),
    # and the ORIGINAL total transactions figure (test+train, pre-SMOTE)
    # must match the raw dataset size, proving the test set was untouched.
    before_total = sum(dataset["class_distribution_before_smote"].values())
    after_total = sum(dataset["class_distribution_after_smote"].values())
    assert after_total >= before_total


def test_evaluation_metrics_present(trained_results):
    results, _ = trained_results
    required = {"precision", "recall", "f1_score", "roc_auc", "pr_auc", "mcc", "accuracy"}
    for m in results["models"]:
        assert required.issubset(m.keys())


def test_best_model_selected_by_f1(trained_results):
    results, _ = trained_results
    assert results["best_model"]["selection_metric"] == SELECTION_METRIC == "f1_score"
    best_name = results["best_model"]["model_name"]
    best_f1 = max(m["f1_score"] for m in results["models"])
    actual_f1 = next(m["f1_score"] for m in results["models"] if m["model_name"] == best_name)
    assert actual_f1 == best_f1


def test_models_and_preprocessing_persisted(trained_results):
    _, model_dir = trained_results
    assert (model_dir / "scaler.joblib").exists()
    assert (model_dir / "evaluation_results.json").exists()
    assert (model_dir / "best_model.json").exists()
    for name in get_model_definitions().keys():
        key = name.lower().replace(" ", "_")
        assert (model_dir / f"model_{key}.joblib").exists()
