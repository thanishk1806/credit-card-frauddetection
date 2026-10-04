"""
Comprehensive investigation into the behavior of activity/velocity values [0, 1, 2, 3, 4, 5, 6, 10, 20, 50, 100].
Checks:
1. Data Pipeline Transformation at each stage
2. Scaler mean and scale parameters
3. Per-model predictions and probabilities across all 5 ensemble models
4. Decision tree split points in the trained models
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.ml.predictor import predict_transaction, get_all_models, get_scaler, get_best_model
from app.ml.preprocessing import transaction_to_dataframe, apply_scaler, FEATURE_COLUMNS

def investigate():
    scaler = get_scaler()
    all_models = get_all_models()
    best_name, best_pipeline = get_best_model()

    print("================================================================================")
    print("SCALER PARAMETERS FOR tx_velocity_5m:")
    print("================================================================================")
    vel_idx = FEATURE_COLUMNS.index("tx_velocity_5m")
    # SCALED_COLUMNS order
    from app.ml.preprocessing import SCALED_COLUMNS
    scaled_vel_idx = SCALED_COLUMNS.index("tx_velocity_5m")
    mean_val = scaler.mean_[scaled_vel_idx]
    scale_val = scaler.scale_[scaled_vel_idx]
    print(f"Feature: tx_velocity_5m (index {scaled_vel_idx} in SCALED_COLUMNS)")
    print(f"Scaler Mean:  {mean_val:.6f}")
    print(f"Scaler Scale: {scale_val:.6f}")
    print(f"Formula: scaled_val = (raw_val - {mean_val:.6f}) / {scale_val:.6f}")

    test_values = [0, 1, 2, 3, 4, 5, 6, 10, 20, 50, 100]
    
    rows = []
    
    print("\n================================================================================")
    print("STAGE-BY-STAGE PIPELINE TRACE FOR EACH VALUE:")
    print("================================================================================")
    
    for v in test_values:
        tx = {
            "amount": 150.0,
            "timestamp": "2026-10-03T14:30:00Z",
            "transaction_type": "online",
            "merchant_category": "retail_shopping",
            "location": "New York, USA",
            "tx_velocity_5m": v,
            "card_present": False,
            "international_transaction": False,
        }
        
        # 1. Feature Engineering
        X_df = transaction_to_dataframe(tx)
        raw_vel = float(X_df.iloc[0]["tx_velocity_5m"])
        risk_sum = float(X_df.iloc[0]["risk_feature_sum"])
        
        # 2. Preprocessing / Scaling
        X_scaled = apply_scaler(X_df, scaler)
        scaled_vel = float(X_scaled.iloc[0]["tx_velocity_5m"])
        scaled_risk_sum = float(X_scaled.iloc[0]["risk_feature_sum"])
        
        # 3. Predict transaction full pipeline
        res = predict_transaction(tx)
        
        rf_prob = res["model_predictions"]["random_forest"]["fraud_probability"]
        ada_prob = res["model_predictions"]["adaboost"]["fraud_probability"]
        xgb_prob = res["model_predictions"]["xgboost"]["fraud_probability"]
        lgb_prob = res["model_predictions"]["lightgbm"]["fraud_probability"]
        cat_prob = res["model_predictions"]["catboost"]["fraud_probability"]
        
        best_prob = res["fraud_probability"]
        best_pred = "FRAUD" if res["is_fraud"] else "LEGITIMATE"
        risk_score = res["risk_score"]
        
        rows.append({
            "User/API Val": v,
            "Raw Eng. Val": raw_vel,
            "Risk Sum": risk_sum,
            "Scaled Vel": scaled_vel,
            "Scaled Risk": scaled_risk_sum,
            "RF (Best) Prob": rf_prob,
            "AdaBoost Prob": ada_prob,
            "XGBoost Prob": xgb_prob,
            "LightGBM Prob": lgb_prob,
            "CatBoost Prob": cat_prob,
            "Final Risk": risk_score,
            "Final Pred": best_pred,
        })
        
        print(f"Activity = {v:<3} -> Raw={raw_vel:<4.1f} | ScaledVel={scaled_vel:<9.4f} | RiskSum={risk_sum:<3.1f} | RF={rf_prob*100:<6.2f}% | Ada={ada_prob*100:<6.2f}% | LGB={lgb_prob*100:<6.2f}% | RiskScore={risk_score}")

    df_summary = pd.DataFrame(rows)
    print("\n================================================================================")
    print("DETAILED COMPARISON TABLE:")
    print("================================================================================")
    print(df_summary.to_string(index=False))

    print("\n================================================================================")
    print("INSPECTING RANDOM FOREST TREE DECISION SPLITS ON tx_velocity_5m:")
    print("================================================================================")
    rf_clf = all_models["random_forest"].named_steps["clf"]
    
    splits = []
    for tree_idx, tree in enumerate(rf_clf.estimators_):
        tree_ = tree.tree_
        feature_indices = tree_.feature
        thresholds = tree_.threshold
        for node_idx, f_idx in enumerate(feature_indices):
            if f_idx == vel_idx:
                scaled_thresh = thresholds[node_idx]
                raw_thresh = scaled_thresh * scale_val + mean_val
                splits.append((scaled_thresh, raw_thresh))
    
    if splits:
        splits_sorted = sorted(splits, key=lambda x: x[1])
        print(f"Total split points on tx_velocity_5m across all RF trees: {len(splits)}")
        print(f"Min split threshold: scaled={splits_sorted[0][0]:.4f} (raw ~ {splits_sorted[0][1]:.2f})")
        print(f"Max split threshold: scaled={splits_sorted[-1][0]:.4f} (raw ~ {splits_sorted[-1][1]:.2f})")
        print("\nAll unique threshold split values in the trained Random Forest trees:")
        unique_raw_thresh = sorted(list({round(s[1], 2) for s in splits}))
        print(unique_raw_thresh)
    else:
        print("No direct splits on tx_velocity_5m found in RF trees.")

if __name__ == "__main__":
    investigate()
