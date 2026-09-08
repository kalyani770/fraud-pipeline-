"""
serving/api.py — Session 4: FastAPI Fraud Detection Inference Endpoint
Run: uvicorn serving.api:app --reload --port 8000
Test: python serving/test_api.py
"""

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import shap

# ── Load Model Artifacts ──────────────────────────────────────────────────────
print("Loading model artifacts...")
model         = joblib.load("model/fraud_model.joblib")
feature_cols  = joblib.load("model/feature_names.joblib")
threshold     = joblib.load("model/threshold.joblib")
explainer     = shap.TreeExplainer(model)
print(f"Model loaded | threshold={threshold:.3f} | features={len(feature_cols)}")

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud scoring with SHAP explanations",
    version="1.0.0",
)

# ── Request Schema ────────────────────────────────────────────────────────────
class Transaction(BaseModel):
    TransactionAmt: float = Field(..., example=117.0)
    card1:          float = Field(-999, example=9500.0)
    card2:          float = Field(-999, example=111.0)
    C1:             float = Field(-999, example=1.0)
    C2:             float = Field(-999, example=1.0)
    C6:             float = Field(-999, example=1.0)
    C11:            float = Field(-999, example=1.0)
    V1:             float = Field(-999, example=1.0)
    V2:             float = Field(-999, example=1.0)
    V3:             float = Field(-999, example=1.0)
    addr1:          float = Field(-999, example=299.0)
    dist1:          float = Field(-999, example=0.0)
    amt_log:        Optional[float] = None
    amt_is_round:   Optional[int]   = None
    card1_txn_count:Optional[float] = Field(-999, example=10.0)
    product_encoded:Optional[int]   = Field(0,    example=0)
    email_match:    Optional[int]   = Field(0,    example=1)


# ── Response Schema ───────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    transaction_id:  Optional[str]
    fraud_prob:      float
    is_fraud:        bool
    threshold_used:  float
    risk_level:      str
    top_shap_factors: list


# ── Helper: Build feature vector ──────────────────────────────────────────────
def build_feature_vector(txn: Transaction) -> pd.DataFrame:
    """Convert incoming transaction to the feature vector the model expects."""
    data = txn.dict()

    # Compute engineered features if not provided
    amt = data["TransactionAmt"]
    if data.get("amt_log") is None:
        data["amt_log"] = float(np.log1p(amt))
    if data.get("amt_is_round") is None:
        data["amt_is_round"] = int(amt % 1 == 0)

    # Build a single-row dataframe with all expected features
    row = {}
    for col in feature_cols:
        row[col] = data.get(col, -999)

    df = pd.DataFrame([row])[feature_cols].astype(np.float32)
    return df


def get_risk_level(prob: float) -> str:
    if prob >= 0.8:   return "CRITICAL"
    if prob >= 0.6:   return "HIGH"
    if prob >= 0.4:   return "MEDIUM"
    if prob >= 0.2:   return "LOW"
    return "MINIMAL"


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Fraud Detection API",
        "status":  "running",
        "endpoints": ["/predict", "/health", "/docs"]
    }


@app.get("/health")
def health():
    return {
        "status":    "healthy",
        "threshold": float(threshold),
        "features":  int(len(feature_cols)),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(txn: Transaction, transaction_id: Optional[str] = None):
    try:
        # Build feature vector
        X = build_feature_vector(txn)

        # Get fraud probability
        fraud_prob = float(model.predict_proba(X)[0, 1])
        is_fraud   = fraud_prob >= threshold

        # SHAP explanation — top 5 factors
        shap_vals  = explainer.shap_values(X)[0]
        shap_pairs = sorted(
            zip(feature_cols, shap_vals),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

        top_factors = [
            {
                "feature":    feat,
                "shap_value": round(float(val), 4),
                "direction":  "increases_fraud_risk" if val > 0 else "decreases_fraud_risk"
            }
            for feat, val in shap_pairs
        ]

        return PredictionResponse(
            transaction_id  = transaction_id,
            fraud_prob      = round(fraud_prob, 4),
            is_fraud        = is_fraud,
            threshold_used  = round(threshold, 3),
            risk_level      = get_risk_level(fraud_prob),
            top_shap_factors= top_factors,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(transactions: list[Transaction]):
    """Score multiple transactions at once."""
    if len(transactions) > 100:
        raise HTTPException(status_code=400, detail="Max 100 transactions per batch")

    results = []
    for txn in transactions:
        X          = build_feature_vector(txn)
        fraud_prob = float(model.predict_proba(X)[0, 1])
        results.append({
            "fraud_prob": round(fraud_prob, 4),
            "is_fraud":   bool(fraud_prob >= threshold),
            "risk_level": get_risk_level(fraud_prob),
        })
    return {"results": results, "count": len(results)}
