"""
FastAPI Loan Defaulter Prediction App
Run with: uvicorn main:app --reload
"""
import logging
import os

import joblib
import pandas as pd
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from features import (EDU_MAP, FEATURE_COLS, HOUSING_MAP, RULE_MIN_DEFAULT_PROB,
                      rule_triggered)

logger = logging.getLogger("loan_app")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise RuntimeError("model.pkl / scaler.pkl not found. Run `python train_model.py` first.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

app = FastAPI(title="Loan Defaulter Predictor")

static_path = os.path.join(BASE_DIR, "static")
if os.path.isdir(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


def _risk_level(prob: float) -> str:
    if prob < 30:
        return "Low"
    if prob < 60:
        return "Medium"
    if prob < 80:
        return "High"
    return "Very-High"


def _predict(age, income, loan_amount, credit_score, employment_years,
             education_level, housing_status) -> dict:
    """Single prediction path used by both the HTML form and the JSON API."""
    if education_level not in EDU_MAP:
        raise HTTPException(422, f"education_level must be one of {list(EDU_MAP)}")
    if housing_status not in HOUSING_MAP:
        raise HTTPException(422, f"housing_status must be one of {list(HOUSING_MAP)}")

    own, rent = HOUSING_MAP[housing_status]
    features_df = pd.DataFrame(
        [[age, income, loan_amount, credit_score, employment_years,
          EDU_MAP[education_level], own, rent]],
        columns=FEATURE_COLS,
    )
    scaled = scaler.transform(features_df)
    prediction = int(model.predict(scaled)[0])
    proba = model.predict_proba(scaled)[0]
    default_prob = round(float(proba[1]) * 100, 2)

    rule_override = rule_triggered(credit_score, loan_amount, income, employment_years)
    if rule_override:
        logger.info("Business rule override applied")
        prediction = 1
        default_prob = max(default_prob, RULE_MIN_DEFAULT_PROB)

    return {
        "prediction": prediction,
        "is_defaulter": bool(prediction),
        "default_prob": default_prob,
        "safe_prob": round(100 - default_prob, 2),
        "risk_level": _risk_level(default_prob),
        "rule_override": rule_override,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    age: float = Form(..., ge=18, le=100),
    income: float = Form(..., ge=0),
    loan_amount: float = Form(..., ge=0),
    credit_score: float = Form(..., ge=300, le=850),
    employment_years: float = Form(..., ge=0, le=60),
    education_level: str = Form(...),
    housing_status: str = Form(...),
):
    out = _predict(age, income, loan_amount, credit_score, employment_years,
                   education_level, housing_status)
    result = {
        "is_defaulter": out["is_defaulter"],
        "default_prob": out["default_prob"],
        "safe_prob": out["safe_prob"],
        "label": "⚠️ LIKELY DEFAULTER" if out["is_defaulter"] else "✅ NOT A DEFAULTER",
        "risk_level": out["risk_level"],
    }
    form_data = {
        "age": age, "income": income, "loan_amount": loan_amount,
        "credit_score": credit_score, "employment_years": employment_years,
        "education_level": education_level, "housing_status": housing_status,
    }
    return templates.TemplateResponse(
        request, "index.html", {"result": result, "form_data": form_data}
    )


class CustomerData(BaseModel):
    age: float = Field(..., ge=18, le=100)
    income: float = Field(..., ge=0)
    loan_amount: float = Field(..., ge=0)
    credit_score: float = Field(..., ge=300, le=850)
    employment_years: float = Field(..., ge=0, le=60)
    education_level: str
    housing_status: str


@app.post("/api/predict")
async def api_predict(data: CustomerData):
    out = _predict(**data.model_dump())
    return {
        "prediction": out["prediction"],
        "is_defaulter": out["is_defaulter"],
        "default_probability": out["default_prob"],
        "safe_probability": out["safe_prob"],
        "risk_level": out["risk_level"],
        "rule_override": out["rule_override"],
    }
