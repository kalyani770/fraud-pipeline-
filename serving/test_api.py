"""
test_api.py — Test the fraud detection API with sample transactions
Run AFTER starting the API: uvicorn serving.api:app --reload --port 8000
Usage: python serving/test_api.py
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def pretty(label, response):
    print(f"\n{'='*50}")
    print(f"TEST: {label}")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

# ── Test 1: Health check ──────────────────────────────────────────────────────
r = requests.get(f"{BASE_URL}/health")
pretty("Health Check", r)

# ── Test 2: Typical legit transaction ────────────────────────────────────────
legit_txn = {
    "TransactionAmt": 49.99,
    "card1": 9500.0,
    "card2": 111.0,
    "C1": 1.0, "C2": 1.0, "C6": 1.0, "C11": 1.0,
    "V1": 1.0, "V2": 1.0, "V3": 1.0,
    "addr1": 299.0, "dist1": 0.0,
    "card1_txn_count": 15.0,
    "product_encoded": 0,
    "email_match": 1,
}
r = requests.post(f"{BASE_URL}/predict", json=legit_txn,
                  params={"transaction_id": "TXN-001"})
pretty("Typical Legit Transaction ($49.99, email match)", r)

# ── Test 3: Suspicious transaction ───────────────────────────────────────────
suspicious_txn = {
    "TransactionAmt": 500.0,   # round number
    "card1": 1234.0,
    "card2": -999,             # missing card info
    "C1": 1.0, "C2": 1.0, "C6": 1.0, "C11": 1.0,
    "V1": 0.0, "V2": 0.0, "V3": 0.0,
    "addr1": -999, "dist1": -999,
    "card1_txn_count": 1.0,    # card used only once ever
    "product_encoded": 2,      # Product C — highest fraud rate
    "email_match": 0,          # emails don't match
}
r = requests.post(f"{BASE_URL}/predict", json=suspicious_txn,
                  params={"transaction_id": "TXN-002"})
pretty("Suspicious Transaction ($500 round, no email match, Product C)", r)

# ── Test 4: Batch scoring ─────────────────────────────────────────────────────
batch = [legit_txn, suspicious_txn, legit_txn]
r = requests.post(f"{BASE_URL}/predict/batch", json=batch)
pretty("Batch Scoring (3 transactions)", r)

print("\n✅ All tests complete. Check http://localhost:8000/docs for interactive API.")
