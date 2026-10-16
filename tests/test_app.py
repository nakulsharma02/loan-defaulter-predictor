import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)
OK = dict(age=35, income=50000, loan_amount=20000, credit_score=700,
          employment_years=5, education_level="Bachelors", housing_status="Rent")


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_home_page():
    assert client.get("/").status_code == 200


def test_api_predict_shape():
    r = client.post("/api/predict", json=OK)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["default_probability"] <= 100
    assert body["risk_level"] in {"Low", "Medium", "High", "Very-High"}


def test_rule_override():
    risky = {**OK, "credit_score": 400, "loan_amount": 500000, "income": 40000, "employment_years": 0}
    body = client.post("/api/predict", json=risky).json()
    assert body["rule_override"] is True and body["is_defaulter"] is True
    assert body["default_probability"] >= 90


def test_rejects_bad_input():
    assert client.post("/api/predict", json={**OK, "credit_score": 5000}).status_code == 422
    assert client.post("/api/predict", json={**OK, "education_level": "Wizard"}).status_code == 422
    assert client.post("/api/predict", json={**OK, "housing_status": "Castle"}).status_code == 422


def test_form_roundtrip():
    form = {k: str(v) for k, v in OK.items()}
    r = client.post("/predict", data=form)
    assert r.status_code == 200 and "Risk Level" in r.text