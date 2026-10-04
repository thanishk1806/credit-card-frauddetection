"""
Test script to verify that changing 5-Min Transaction Velocity:
1. Reaches the backend feature engineering pipeline
2. Directly changes the numerical feature `tx_velocity_5m` and composite `risk_feature_sum`
3. Scaled model input vector updates accordingly
4. All models (Random Forest, AdaBoost, XGBoost, LightGBM, CatBoost) run inference and compute probabilities
5. Generates valid risk scores and SHAP explanations
"""
import sys
from pathlib import Path

# Add project root and backend to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.ml.predictor import predict_transaction
from app.ml.explainer import explain_prediction, FEATURE_LABELS
from app.ml.predictor import get_best_model

def run_velocity_test(velocity_val: int):
    tx = {
        "amount": 150.0,
        "timestamp": "2026-10-03T14:30:00Z",
        "transaction_type": "online",
        "merchant_category": "retail_shopping",
        "location": "New York, USA",
        "tx_velocity_5m": velocity_val,
        "card_present": False,
        "international_transaction": False,
    }
    
    result = predict_transaction(tx)
    raw_df = result["raw_input"]
    scaled_df = result["scaled_input"]
    best_name, pipeline = get_best_model()
    
    fraud_contributors, legit_contributors, explanation_sentences, explanation_title = explain_prediction(
        pipeline, scaled_df, raw_df, is_fraud=result["is_fraud"]
    )
    
    print(f"\n=======================================================")
    print(f"TEST CASE: 5-Min Velocity = {velocity_val}")
    print(f"=======================================================")
    print(f"Raw Input Velocity:       {raw_df.iloc[0]['tx_velocity_5m']}")
    print(f"Risk Feature Sum:         {raw_df.iloc[0]['risk_feature_sum']}")
    print(f"Scaled Velocity in Model: {scaled_df.iloc[0]['tx_velocity_5m']:.6f}")
    print(f"Scaled Risk Feature Sum:  {scaled_df.iloc[0]['risk_feature_sum']:.6f}")
    print(f"Final Prediction:         {'FRAUD' if result['is_fraud'] else 'LEGITIMATE'}")
    print(f"Fraud Probability:        {result['fraud_probability']:.6f}")
    print(f"Legitimate Probability:   {result['legitimate_probability']:.6f}")
    print(f"Sum of Probabilities:     {result['fraud_probability'] + result['legitimate_probability']:.6f}")
    print(f"Risk Score:               {result['risk_score']}")
    print(f"Risk Level:               {result['risk_level']}")
    print(f"Model Used:               {result['model_name']}")
    print(f"Per-Model Predictions:")
    for key, model_res in result["model_predictions"].items():
        print(f"  - {model_res['model_name']}: {model_res['prediction']} (Fraud Prob: {model_res['fraud_probability'] * 100:.2f}%, Risk: {model_res['risk_score']})")
    print(f"SHAP Explanation Title:   {explanation_title}")
    print(f"SHAP Explanation Sentences:")
    for s in explanation_sentences:
        print(f"  * {s}")
    
    return {
        "velocity": velocity_val,
        "raw_velocity": raw_df.iloc[0]['tx_velocity_5m'],
        "scaled_velocity": scaled_df.iloc[0]['tx_velocity_5m'],
        "fraud_prob": result['fraud_probability'],
        "risk_score": result['risk_score'],
    }

if __name__ == "__main__":
    test_cases = [0, 1, 2, 5, 10]
    results = []
    for v in test_cases:
        res = run_velocity_test(v)
        results.append(res)
    
    print("\n=======================================================")
    print("SUMMARY COMPARISON ACROSS VELOCITIES:")
    print("=======================================================")
    print(f"{'Velocity':<10} {'Raw Vel':<10} {'Scaled Vel':<14} {'Fraud Prob':<14} {'Risk Score':<12}")
    for r in results:
        print(f"{r['velocity']:<10} {r['raw_velocity']:<10.1f} {r['scaled_velocity']:<14.6f} {r['fraud_prob']:<14.6f} {r['risk_score']:<12.1f}")
