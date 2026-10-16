# 🏦 Loan Defaulter Predictor — FastAPI Web App

A web application that predicts whether a bank customer is likely to default on a loan.
It uses a Logistic Regression model (scikit-learn) trained on `credit_risk_dataset.csv`
and is served with FastAPI (HTML form + JSON API).

> **Educational project.** Not for real lending decisions.

## 📁 Project structure

```
├── main.py              FastAPI app (routes + prediction logic)
├── features.py          Shared encodings, feature order and business-rule constants
├── train_model.py       Trains the model and writes model.pkl / scaler.pkl
├── model.pkl, scaler.pkl  Trained artifacts (created with scikit-learn 1.8.0)
├── templates/index.html Web UI
├── tests/test_app.py    Basic API tests
├── Loan Deafulter .ipynb  Exploration notebook
├── requirements.txt / requirements-dev.txt
├── Procfile, runtime.txt  Deployment config
└── static/              Reserved for custom assets
```

The dataset (`credit_risk_dataset.csv`) is **not** in the repo. It is only needed to retrain.

## 🚀 Run locally

```
pip install -r requirements.txt
uvicorn main:app --reload
```
Open http://127.0.0.1:8000

### Retrain (optional)
Put `credit_risk_dataset.csv` next to `train_model.py` (or set `LOAN_CSV_PATH`), then:
```
python train_model.py
```
This prints accuracy, ROC-AUC, confusion matrix and classification report, and overwrites
`model.pkl` and `scaler.pkl`. Keep the installed scikit-learn version the same as in
`requirements.txt`, because pickled models are version-sensitive.

### Tests
```
pip install -r requirements-dev.txt
pytest
```

## 🌐 Deploy
`Procfile` starts the app with `uvicorn main:app --host 0.0.0.0 --port $PORT`, and
`runtime.txt` pins Python (adjust to a version your host supports). Health check: `GET /health`.

## 🌐 API

| Method | URL            | Description                         |
| ------ | -------------- | ----------------------------------- |
| GET    | `/`            | Web UI                              |
| POST   | `/predict`     | Form submission → HTML result       |
| POST   | `/api/predict` | JSON API                            |
| GET    | `/health`      | Health check                        |

```
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"age":35,"income":50000,"loan_amount":120000,"credit_score":580,
       "employment_years":3,"education_level":"Bachelors","housing_status":"Rent"}'
```
Response fields: `prediction`, `is_defaulter`, `default_probability`, `safe_probability`,
`risk_level`, `rule_override`.

Valid ranges: age 18–100, credit_score 300–850, employment_years 0–60, income and
loan_amount ≥ 0. `education_level`: High School / Bachelors / Masters / PhD.
`housing_status`: Mortgage / Own / Rent. Invalid input returns HTTP 422.

## 📊 Model details

| Item         | Value                                                         |
| ------------ | ------------------------------------------------------------- |
| Algorithm    | Logistic Regression (`max_iter=1000`)                         |
| Class weight | `balanced` (handles imbalanced data)                          |
| Scaler       | StandardScaler, fitted on training data only                  |
| Split        | 80/20, stratified, `random_state=10`                          |
| Missing data | Income filled with the training-set mean                      |
| Encoding     | Education ordinal (0–3); Housing one-hot, Mortgage = baseline |

Risk levels from default probability: Low < 30, Medium < 60, High < 80, Very-High ≥ 80.

### Manual business rule
If `credit_score < 500` **and** `loan_amount > 5 × income` **and** `employment_years < 1`,
the applicant is marked a likely defaulter and the default probability is raised to at least
90%. This rule sits on top of the model (it is not part of the metrics printed by
`train_model.py`), and the API reports it via `rule_override`.
