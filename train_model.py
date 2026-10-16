"""
Run this script to train the model and save it.

Usage:
    python train_model.py
    LOAN_CSV_PATH=/path/to/credit_risk_dataset.csv python train_model.py

Looks for credit_risk_dataset.csv next to this script unless LOAN_CSV_PATH is set.
Creates model.pkl and scaler.pkl (both are loaded by main.py).
"""
import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from features import EDU_MAP, FEATURE_COLS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.environ.get("LOAN_CSV_PATH", os.path.join(BASE_DIR, "credit_risk_dataset.csv"))

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Dataset not found at {CSV_PATH}.\n"
        "Place 'credit_risk_dataset.csv' next to train_model.py or set LOAN_CSV_PATH."
    )

df = pd.read_csv(CSV_PATH)
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ── Encoding (no statistics are learned here, so it is safe before the split) ──
df["Education_Level"] = df["Education_Level"].map(EDU_MAP)
# drop_first removes the alphabetically first category, which is Mortgage.
df = pd.get_dummies(df, columns=["Housing_Status"], drop_first=True, dtype=int)

X = df[FEATURE_COLS].copy()
y = df["Default"]

# ── Split first, so test rows never influence imputation or scaling ───────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=10
)

# Fill missing Income using the TRAINING mean only.
income_mean = X_train["Income"].mean()
X_train["Income"] = X_train["Income"].fillna(income_mean)
X_test["Income"] = X_test["Income"].fillna(income_mean)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ── Train ─────────────────────────────────────────────────────────────────────
lr = LogisticRegression(max_iter=1000, class_weight="balanced")
lr.fit(X_train_s, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = lr.predict(X_test_s)
print(f"\nAccuracy : {lr.score(X_test_s, y_test):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, lr.predict_proba(X_test_s)[:, 1]):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump(lr, os.path.join(BASE_DIR, "model.pkl"))
joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))
print("\nSaved model.pkl and scaler.pkl")
