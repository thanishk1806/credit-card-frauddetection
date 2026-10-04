"""
Verification script for TrustCheck Prediction Behavior.

Tests realistic user transactions across:
- TEST GROUP 1: Normal transactions (Expected: LEGITIMATE)
- TEST GROUP 2: Suspicious transactions (Expected: FRAUD)
- TEST GROUP 3: Mixed transactions (Evaluated based on overall combination)

Verifies:
1. Genuine ML model probability remains unaltered.
2. Genuine SHAP explanation remains intact.
3. TrustCheck Final Assessment produces both LEGITIMATE and FRAUD.
4. Risk score and risk levels are properly assigned.
5. Transaction risk decision layer provides explainable risk factors.
"""
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.database.session import Base, SessionLocal, engine
from app.models.user import User
from app.services.prediction_service import run_prediction
from app.schemas.predict import PredictionResponse

# Ensure DB tables exist
Base.metadata.create_all(bind=engine)
db = SessionLocal()

# Ensure a test user exists
user = db.query(User).filter_by(username="test_audit_user").first()
if not user:
    user = User(
        id=uuid.uuid4(),
        username="test_audit_user",
        email="audit@trustcheck.internal",
        hashed_password="not_a_real_password_hash",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

test_cases = [
    # ── TEST GROUP 1: NORMAL TRANSACTIONS ──
    {
        "group": "GROUP 1: NORMAL TRANSACTIONS",
        "name": "Normal Case 1 — Routine Morning Grocery",
        "input": {
            "amount": 42.50,
            "timestamp": "2026-10-04T10:15:00",
            "transaction_type": "pos",
            "merchant_category": "grocery",
            "location": "New York, USA",
            "tx_velocity_5m": 0,
            "card_present": True,
            "international_transaction": False,
        },
    },
    {
        "group": "GROUP 1: NORMAL TRANSACTIONS",
        "name": "Normal Case 2 — Afternoon Dining & Fuel",
        "input": {
            "amount": 68.00,
            "timestamp": "2026-10-04T13:45:00",
            "transaction_type": "contactless",
            "merchant_category": "food_dining",
            "location": "Chicago, USA",
            "tx_velocity_5m": 0,
            "card_present": True,
            "international_transaction": False,
        },
    },
    {
        "group": "GROUP 1: NORMAL TRANSACTIONS",
        "name": "Normal Case 3 — Domestic Utility Bill Payment",
        "input": {
            "amount": 89.20,
            "timestamp": "2026-10-04T11:00:00",
            "transaction_type": "online",
            "merchant_category": "utilities",
            "location": "Seattle, USA",
            "tx_velocity_5m": 0,
            "card_present": False,
            "international_transaction": False,
        },
    },

    # ── TEST GROUP 2: SUSPICIOUS TRANSACTIONS ──
    {
        "group": "GROUP 2: SUSPICIOUS TRANSACTIONS",
        "name": "Suspicious Case 1 — High-Value Luxury Jewelry at 2:30 AM (Cross-Border CNP)",
        "input": {
            "amount": 9500.00,
            "timestamp": "2026-10-04T02:30:00",
            "transaction_type": "online",
            "merchant_category": "luxury_jewelry",
            "location": "Lagos, Nigeria",
            "tx_velocity_5m": 6,
            "card_present": False,
            "international_transaction": True,
        },
    },
    {
        "group": "GROUP 2: SUSPICIOUS TRANSACTIONS",
        "name": "Suspicious Case 2 — Rapid International Wire Transfer at 3:15 AM",
        "input": {
            "amount": 16000.00,
            "timestamp": "2026-10-04T03:15:00",
            "transaction_type": "wire_transfer",
            "merchant_category": "electronics",
            "location": "Unknown / International",
            "tx_velocity_5m": 8,
            "card_present": False,
            "international_transaction": True,
        },
    },
    {
        "group": "GROUP 2: SUSPICIOUS TRANSACTIONS",
        "name": "Suspicious Case 3 — High Velocity Electronics Purchase at 1:15 AM",
        "input": {
            "amount": 6800.00,
            "timestamp": "2026-10-04T01:15:00",
            "transaction_type": "online",
            "merchant_category": "electronics",
            "location": "Eastern Europe",
            "tx_velocity_5m": 5,
            "card_present": False,
            "international_transaction": True,
        },
    },

    # ── TEST GROUP 3: MIXED TRANSACTIONS ──
    {
        "group": "GROUP 3: MIXED TRANSACTIONS",
        "name": "Mixed Case 1 — Moderate Domestic Online Retail in Afternoon",
        "input": {
            "amount": 250.00,
            "timestamp": "2026-10-04T15:30:00",
            "transaction_type": "online",
            "merchant_category": "retail_shopping",
            "location": "Dallas, USA",
            "tx_velocity_5m": 1,
            "card_present": False,
            "international_transaction": False,
        },
    },
    {
        "group": "GROUP 3: MIXED TRANSACTIONS",
        "name": "Mixed Case 2 — High-Value In-Store Purchase (Card Present, Domestic, Daytime)",
        "input": {
            "amount": 4500.00,
            "timestamp": "2026-10-04T12:00:00",
            "transaction_type": "pos",
            "merchant_category": "electronics",
            "location": "San Francisco, USA",
            "tx_velocity_5m": 0,
            "card_present": True,
            "international_transaction": False,
        },
    },
    {
        "group": "GROUP 3: MIXED TRANSACTIONS",
        "name": "Mixed Case 3 — Domestic Airline Booking in Evening",
        "input": {
            "amount": 1150.00,
            "timestamp": "2026-10-04T20:45:00",
            "transaction_type": "online",
            "merchant_category": "travel_airlines",
            "location": "Atlanta, USA",
            "tx_velocity_5m": 1,
            "card_present": False,
            "international_transaction": False,
        },
    },
]

print("=" * 80)
print("TRUSTCHECK PREDICTION BEHAVIOR VERIFICATION REPORT")
print("=" * 80)

current_group = None
results_summary = []

for tc in test_cases:
    if tc["group"] != current_group:
        current_group = tc["group"]
        print(f"\n{current_group}")
        print("-" * len(current_group))

    rec = run_prediction(db, user.id, tc["input"])
    shap_data = rec.shap_explanation or {}
    decision_layer = shap_data.get("decision_layer") or {}

    ml_prob = rec.fraud_probability
    ml_pred = decision_layer.get("ml_prediction", "FRAUD" if ml_prob >= 0.5 else "LEGITIMATE")
    tc_assessment = decision_layer.get("trustcheck_assessment", "FRAUD" if rec.is_fraud else "LEGITIMATE")
    tc_score = rec.risk_score
    tc_level = rec.risk_level

    results_summary.append({
        "name": tc["name"],
        "amount": tc["input"]["amount"],
        "ml_prob": round(ml_prob * 100, 2),
        "ml_pred": ml_pred,
        "tc_assessment": tc_assessment,
        "tc_score": tc_score,
        "tc_level": tc_level,
    })

    print(f"\nTest: {tc['name']}")
    print(f"  Inputs: Amount=${tc['input']['amount']:,.2f} | Time={tc['input']['timestamp']} | Channel={tc['input']['transaction_type']} | CardPresent={tc['input']['card_present']} | Intl={tc['input']['international_transaction']} | Velocity={tc['input']['tx_velocity_5m']}")
    print(f"  A. Genuine ML Output:      Prediction={ml_pred:10} | Fraud Probability={ml_prob*100:6.2f}% | Model={rec.model_name}")
    print(f"  B. TrustCheck Final:       Assessment={tc_assessment:10} | Risk Score={tc_score:5.1f}/100 | Risk Level={tc_level}")

    # Top SHAP factors (genuine from ML pipeline)
    top_shap = shap_data.get("top_fraud_contributors", [])[:2]
    if top_shap:
        shap_str = ", ".join([f"{f.get('label', f['feature'])} (impact: {abs(f['shap_value']):.4f})" for f in top_shap])
        print(f"  Genuine ML SHAP Factors:   {shap_str}")

    # Decision layer signals
    factors = decision_layer.get("risk_factors", [])
    elevated_factors = [f for f in factors if f.get("direction") == "suspicious"]
    if elevated_factors:
        print(f"  Elevated Risk Signals:     {len(elevated_factors)} signals ({', '.join([f['label'] for f in elevated_factors])})")
    else:
        print(f"  Elevated Risk Signals:     None (All parameters within normal baseline)")

print("\n" + "=" * 80)
print("VERIFICATION SUMMARY TABLE")
print("=" * 80)
print(f"{'Test Case':55} | {'ML Prob':8} | {'ML Pred':10} | {'TrustCheck':10} | {'Score':8} | {'Level':6}")
print("-" * 105)
for r in results_summary:
    print(f"{r['name'][:55]:55} | {r['ml_prob']:6.2f}%  | {r['ml_pred']:10} | {r['tc_assessment']:10} | {r['tc_score']:5.1f}/100 | {r['tc_level']:6}")
print("=" * 80)
