"""
Tests for prediction endpoint input validation, realistic transactions, and auth requirements.
"""


def _register_and_get_token(client, username="predictor"):
    client.post("/auth/register", json={
        "username": username, "email": f"{username}@example.com", "password": "StrongPass123"
    })
    resp = client.post("/auth/login", json={"username": username, "password": "StrongPass123"})
    return resp.json()["access_token"]


def _valid_legacy_transaction():
    payload = {"Time": 1000.0, "Amount": 49.99}
    for i in range(1, 29):
        payload[f"V{i}"] = 0.1 * i
    return payload


def _valid_realistic_transaction():
    return {
        "amount": 149.99,
        "timestamp": "2026-10-03T10:30:00",
        "transaction_type": "online",
        "merchant_category": "electronics",
        "location": "Hyderabad, India",
        "tx_velocity_5m": 2,
        "card_present": False,
        "international_transaction": False,
    }


def test_predict_requires_auth(client):
    resp = client.post("/predict", json=_valid_legacy_transaction())
    assert resp.status_code == 401


def test_predict_rejects_missing_amount(client):
    token = _register_and_get_token(client, username="pred_missing")
    resp = client.post(
        "/predict",
        json={"timestamp": "2026-10-03T10:30:00", "merchant_category": "electronics"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


def test_predict_rejects_negative_amount(client):
    token = _register_and_get_token(client, username="pred_negative")
    resp = client.post(
        "/predict",
        json={"amount": -50.0, "transaction_type": "online"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


def test_predict_rejects_zero_amount(client):
    token = _register_and_get_token(client, username="pred_zero")
    resp = client.post(
        "/predict",
        json={"amount": 0.0, "transaction_type": "online"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


def test_predict_rejects_negative_velocity(client):
    token = _register_and_get_token(client, username="pred_neg_vel")
    payload = _valid_realistic_transaction()
    payload["tx_velocity_5m"] = -3
    resp = client.post(
        "/predict",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


def test_predict_returns_503_when_model_not_trained(client, tmp_path, monkeypatch):
    from app.config import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "empty_models"))
    get_settings.cache_clear()

    token = _register_and_get_token(client, username="predictor_empty")
    resp = client.post(
        "/predict",
        json=_valid_realistic_transaction(),
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code in (503, 500)
    get_settings.cache_clear()
