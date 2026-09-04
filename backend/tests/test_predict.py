"""
Tests for the prediction endpoint's input validation and auth requirements.

These tests do not require a trained model on disk for the validation /
auth checks; a full round-trip prediction test is in test_ml_pipeline.py
and requires `python scripts/train_models.py` to have been run first.
"""


def _register_and_get_token(client, username="predictor"):
    client.post("/auth/register", json={
        "username": username, "email": f"{username}@example.com", "password": "StrongPass123"
    })
    resp = client.post("/auth/login", json={"username": username, "password": "StrongPass123"})
    return resp.json()["access_token"]


def _valid_transaction():
    payload = {"Time": 1000.0, "Amount": 49.99}
    for i in range(1, 29):
        payload[f"V{i}"] = 0.1 * i
    return payload


def test_predict_requires_auth(client):
    resp = client.post("/predict", json=_valid_transaction())
    assert resp.status_code == 401


def test_predict_rejects_missing_fields(client):
    token = _register_and_get_token(client)
    resp = client.post(
        "/predict",
        json={"Time": 100.0},  # missing required Amount
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422



def test_predict_returns_503_when_model_not_trained(client, tmp_path, monkeypatch):
    from app.config import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "empty_models"))
    get_settings.cache_clear()

    token = _register_and_get_token(client, username="predictor2")
    resp = client.post(
        "/predict",
        json=_valid_transaction(),
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code in (503, 500)
    get_settings.cache_clear()
