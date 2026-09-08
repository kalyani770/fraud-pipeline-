"""
ingestion/consumer.py — Session 5: Real-time Kafka Consumer
Reads transactions from Kafka → computes real-time features → 
stores in Redis → calls FastAPI → logs prediction

Run AFTER: uvicorn serving.api:app --reload --port 8000
Usage: python ingestion/consumer.py
"""

import json
import time
from datetime import datetime
from collections import defaultdict

import redis
import requests
from kafka import KafkaConsumer

# ── Config ────────────────────────────────────────────────────────────────────
KAFKA_BROKER  = "localhost:9092"
TOPIC         = "fraud-transactions"
API_URL       = "http://localhost:8000/predict"
REDIS_HOST    = "localhost"
REDIS_PORT    = 6379

# Sliding window for velocity features (seconds)
WINDOW_5MIN   = 300
WINDOW_1HOUR  = 3600

print("=" * 60)
print("REAL-TIME FRAUD DETECTION CONSUMER")
print("=" * 60)


# ── Connect to Redis ──────────────────────────────────────────────────────────
print("\n[1/3] Connecting to Redis...")
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
r.ping()
print("✅ Redis connected")


# ── Connect to Kafka ──────────────────────────────────────────────────────────
print("[2/3] Connecting to Kafka...")
consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    auto_offset_reset="latest",        # only new messages from now
    value_deserializer=lambda b: json.loads(b.decode("utf-8")),
    consumer_timeout_ms=60_000,        # stop if no messages for 60s
)
print("✅ Kafka connected")


# ── Verify API is running ─────────────────────────────────────────────────────
print("[3/3] Checking API health...")
try:
    health = requests.get("http://localhost:8000/health", timeout=5)
    print(f"✅ API healthy | {health.json()}")
except Exception:
    print("❌ API not running. Start it first:")
    print("   uvicorn serving.api:app --reload --port 8000")
    exit(1)

print("\n🚀 Listening for transactions...\n")


# ── Helper: Redis Feature Store ───────────────────────────────────────────────
def update_card_velocity(card_id: str, amount: float, timestamp: float) -> dict:
    """
    Store transaction in Redis and compute velocity features.
    Uses a Redis List per card — each entry is 'timestamp:amount'
    
    Returns:
        txn_count_5min  — transactions in last 5 minutes
        txn_count_1hr   — transactions in last 1 hour  
        amt_sum_5min    — total amount in last 5 minutes
        is_first_txn    — 1 if this card has never been seen before
    """
    key = f"card:{card_id}:txns"

    # Push current transaction
    r.lpush(key, f"{timestamp}:{amount}")
    r.expire(key, WINDOW_1HOUR * 2)     # keep data for 2 hours

    # Read all recent transactions for this card
    entries     = r.lrange(key, 0, -1)  # get all entries
    now         = timestamp
    count_5min  = 0
    count_1hr   = 0
    amt_5min    = 0.0

    for entry in entries:
        ts, amt = entry.split(":", 1)
        ts, amt = float(ts), float(amt)
        age     = now - ts

        if age <= WINDOW_5MIN:
            count_5min += 1
            amt_5min   += amt
        if age <= WINDOW_1HOUR:
            count_1hr  += 1

    return {
        "txn_count_5min": count_5min,
        "txn_count_1hr":  count_1hr,
        "amt_sum_5min":   round(amt_5min, 2),
        "is_first_txn":   1 if count_1hr <= 1 else 0,
    }


def get_card_risk_profile(card_id: str) -> str:
    """Look up card's historical risk label stored in Redis."""
    return r.get(f"card:{card_id}:risk") or "unknown"


def store_prediction(txn_id: str, prediction: dict):
    """Cache the prediction result in Redis for 24 hours."""
    r.setex(
        f"pred:{txn_id}",
        86400,                          # 24 hours TTL
        json.dumps(prediction)
    )


# ── Stats tracker ─────────────────────────────────────────────────────────────
stats = defaultdict(int)
start_time = time.time()


# ── Main Consumer Loop ────────────────────────────────────────────────────────
for msg in consumer:
    event = msg.value

    try:
        # ── Extract fields from Kafka message ──
        card_id    = str(event.get("card1", "unknown"))
        amount     = float(event.get("amount", 0))
        txn_id     = event.get("event_id", "unknown")
        timestamp  = time.time()

        # ── Compute real-time features from Redis ──
        velocity   = update_card_velocity(card_id, amount, timestamp)

        # ── Build feature payload for API ──
        payload = {
            "TransactionAmt":  amount,
            "card1":           float(event.get("card1",  -999)),
            "card2":           float(event.get("card2",  -999)),
            "C1":              float(event.get("c1",     -999)),
            "C2":              float(event.get("c2",     -999)),
            "C6":              float(event.get("c1",     -999)),
            "C11":             float(event.get("c1",     -999)),
            "V1":              -999.0,
            "V2":              -999.0,
            "V3":              -999.0,
            "addr1":           float(event.get("addr1",  -999)),
            "dist1":           float(event.get("dist1",  -999)),
            "card1_txn_count": float(velocity["txn_count_1hr"]),
            "product_encoded": int(event.get("product_cd", "W") == "C") * 2,
            "email_match":     int(
                event.get("p_email_domain") == event.get("r_email_domain")
            ),
        }

        # ── Call FastAPI for prediction ──
        t0       = time.time()
        response = requests.post(
            API_URL,
            json=payload,
            params={"transaction_id": txn_id},
            timeout=2,
        )
        latency_ms = round((time.time() - t0) * 1000, 1)

        if response.status_code == 200:
            pred = response.json()

            # ── Store prediction in Redis ──
            store_prediction(txn_id, {
                **pred,
                "velocity": velocity,
                "scored_at": datetime.utcnow().isoformat(),
            })

            # ── Update stats ──
            stats["total"]    += 1
            stats["fraud"]    += int(pred["is_fraud"])
            stats["latency"]  += latency_ms

            # ── Log result ──
            label     = "🚨 FRAUD   " if pred["is_fraud"] else "✅ LEGIT   "
            risk      = pred["risk_level"]
            fraud_prob= pred["fraud_prob"]
            v5        = velocity["txn_count_5min"]

            print(
                f"{label} | prob={fraud_prob:.3f} | risk={risk:<8} | "
                f"amt=${amount:<8.2f} | vel_5m={v5} | "
                f"latency={latency_ms}ms"
            )

            # ── Alert on high velocity ──
            if velocity["txn_count_5min"] >= 5:
                print(f"  ⚠️  HIGH VELOCITY: card {card_id} made "
                      f"{v5} transactions in 5 minutes!")

            # ── Print stats every 50 transactions ──
            if stats["total"] % 50 == 0:
                avg_latency = stats["latency"] / stats["total"]
                fraud_rate  = stats["fraud"] / stats["total"] * 100
                elapsed     = round(time.time() - start_time, 0)
                print(f"\n  📊 [{elapsed}s] Processed {stats['total']} txns | "
                      f"fraud rate={fraud_rate:.1f}% | "
                      f"avg latency={avg_latency:.1f}ms\n")

        else:
            print(f"❌ API error {response.status_code}: {response.text[:100]}")
            stats["errors"] += 1

    except requests.Timeout:
        print(f"⏱️  Timeout on txn {txn_id} — API too slow")
        stats["timeouts"] += 1
    except Exception as e:
        print(f"❌ Error processing message: {e}")
        stats["errors"] += 1

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CONSUMER STOPPED")
print(f"Total processed : {stats['total']}")
print(f"Fraud detected  : {stats['fraud']} ({stats['fraud']/max(stats['total'],1)*100:.1f}%)")
print(f"Errors          : {stats['errors']}")
print(f"Avg latency     : {stats['latency']/max(stats['total'],1):.1f}ms")
print("=" * 60)
