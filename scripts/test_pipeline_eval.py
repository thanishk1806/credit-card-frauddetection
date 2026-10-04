import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.ml.predictor import predict_transaction, get_best_model
from app.ml.explainer import explain_prediction

tests = [
    (
        "TEST 1: Normal low-value POS",
        {
            "amount": 34.50,
            "transaction_type": "pos",
            "merchant_category": "grocery",
            "location": "Hyderabad, India",
            "device_type": "pos_terminal",
            "card_present": True,
            "international_transaction": False,
            "time_since_prev": 3600.0,
            "tx_velocity_5m": 1.0,
        },
    ),
    (
        "TEST 2: Very high-value ($50k)",
        {
            "amount": 50000.0,
            "transaction_type": "online",
            "merchant_category": "electronics",
            "location": "Hyderabad, India",
            "device_type": "desktop",
            "card_present": False,
            "international_transaction": False,
            "time_since_prev": 1800.0,
            "tx_velocity_5m": 1.0,
        },
    ),
    (
        "TEST 3: Unusual late-night luxury",
        {
            "amount": 420.0,
            "transaction_type": "online",
            "merchant_category": "luxury_jewelry",
            "location": "London, UK",
            "device_type": "desktop",
            "card_present": False,
            "international_transaction": True,
            "time_since_prev": 120.0,
            "tx_velocity_5m": 2.0,
        },
    ),
    (
        "TEST 4: International transaction",
        {
            "amount": 150.0,
            "transaction_type": "online",
            "merchant_category": "retail_shopping",
            "location": "Singapore",
            "device_type": "mobile",
            "card_present": False,
            "international_transaction": True,
            "time_since_prev": 1800.0,
            "tx_velocity_5m": 1.0,
        },
    ),
    (
        "TEST 5: Rapid repeated velocity burst",
        {
            "amount": 2.50,
            "transaction_type": "online",
            "merchant_category": "retail_shopping",
            "location": "Unknown IP",
            "device_type": "unknown_device",
            "card_present": False,
            "international_transaction": True,
            "time_since_prev": 5.0,
            "tx_velocity_5m": 6.0,
        },
    ),
    (
        "TEST 6a: Grocery category",
        {
            "amount": 250.0,
            "transaction_type": "online",
            "merchant_category": "grocery",
            "card_present": False,
            "international_transaction": False,
        },
    ),
    (
        "TEST 6b: Electronics category",
        {
            "amount": 250.0,
            "transaction_type": "online",
            "merchant_category": "electronics",
            "card_present": False,
            "international_transaction": False,
        },
    ),
    (
        "TEST 7a: POS transaction",
        {
            "amount": 200.0,
            "transaction_type": "pos",
            "merchant_category": "retail_shopping",
            "card_present": True,
            "international_transaction": False,
        },
    ),
    (
        "TEST 7b: Wire transfer transaction",
        {
            "amount": 200.0,
            "transaction_type": "wire_transfer",
            "merchant_category": "retail_shopping",
            "card_present": False,
            "international_transaction": False,
        },
    ),
]

for name, tx in tests:
    res = predict_transaction(tx)
    print("=" * 60)
    print(f"=== {name} ===")
    verdict = "FRAUD" if res["is_fraud"] else "LEGITIMATE"
    print(f"Final Model: {res['model_name']} -> {verdict} (Fraud: {res['fraud_probability']*100:.2f}%, Risk: {res['risk_score']:.0f}/100 - {res['risk_level']})")
    print("Per-Model Inference:")
    for mkey, mpred in res["model_predictions"].items():
        print(f"   {mpred['model_name']:15} => {mpred['prediction']:10} (Fraud: {mpred['fraud_probability']*100:6.2f}%, Legit: {mpred['legitimate_probability']*100:6.2f}%)")

    _, pipe = get_best_model()
    _, _, sentences, title = explain_prediction(pipe, res["scaled_input"], res["raw_input"], is_fraud=res["is_fraud"])
    print(f"Explanation ({title}):")
    for s in sentences:
        print(f"   • {s}")
    print()
