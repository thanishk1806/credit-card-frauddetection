"""Tests for registration / login / protected routes."""


def test_register_and_login(client):
    resp = client.post("/auth/register", json={
        "username": "testuser", "email": "test@example.com", "password": "StrongPass123"
    })
    assert resp.status_code == 201
    body = resp.json()
    assert "hashed_password" not in body
    assert body["username"] == "testuser"

    resp = client.post("/auth/login", json={"username": "testuser", "password": "StrongPass123"})
    assert resp.status_code == 200
    assert "access_token" in resp.json()


def test_login_invalid_credentials(client):
    resp = client.post("/auth/login", json={"username": "nouser", "password": "wrong"})
    assert resp.status_code == 401


def test_protected_route_requires_token(client):
    resp = client.get("/auth/me")
    assert resp.status_code == 401


def test_duplicate_registration_rejected(client):
    client.post("/auth/register", json={
        "username": "dupe", "email": "dupe@example.com", "password": "StrongPass123"
    })
    resp = client.post("/auth/register", json={
        "username": "dupe", "email": "dupe2@example.com", "password": "StrongPass123"
    })
    assert resp.status_code == 409
