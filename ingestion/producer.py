"""
producer.py — Reads the IEEE-CIS fraud dataset row by row and
publishes each transaction as a JSON event to a Kafka topic.

Run after: docker-compose up -d
Usage:     python ingestion/producer.py --file data/train_transaction.csv --speed 1
"""

import argparse
import json
import time
import uuid
from datetime import datetime

import pandas as pd
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

# ── Config ──────────────────────────────────────────────────────────────────
KAFKA_BROKER = "localhost:9092"
TOPIC        = "fraud-transactions"

# Columns we care about (subset of the full IEEE-CIS schema)
KEEP_COLS = [
    "TransactionID", "isFraud", "TransactionAmt", "ProductCD",
    "card1", "card2", "card4", "card6",
    "P_emaildomain", "R_emaildomain",
    "C1", "C2", "C6", "C11",
    "V1", "V2", "V3", "V4",
    "addr1", "dist1",
]


# ── Helpers ──────────────────────────────────────────────────────────────────
def make_producer(retries: int = 5) -> KafkaProducer:
    """Create a Kafka producer, retrying if the broker isn't ready yet."""
    for attempt in range(1, retries + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks="all",           # wait for leader + ISR acknowledgement
                retries=3,
            )
            print(f"✅ Connected to Kafka broker at {KAFKA_BROKER}")
            return producer
        except NoBrokersAvailable:
            print(f"⏳ Broker not ready (attempt {attempt}/{retries}). Retrying in 3s…")
            time.sleep(3)
    raise RuntimeError("Could not connect to Kafka. Is docker-compose up?")


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load the CSV, keep only relevant columns, fill nulls."""
    print(f"📂 Loading dataset from: {filepath}")
    available = pd.read_csv(filepath, nrows=0).columns.tolist()
    cols      = [c for c in KEEP_COLS if c in available]
    df        = pd.read_csv(filepath, usecols=cols)
    df        = df.fillna(-999)          # sentinel for missing numerics
    print(f"   Loaded {len(df):,} rows | {df['isFraud'].sum():,} fraud cases "
          f"({df['isFraud'].mean()*100:.1f}%)")
    return df


def row_to_event(row: dict) -> dict:
    """Enrich a raw CSV row with metadata before publishing."""
    return {
        "event_id":        str(uuid.uuid4()),          # unique message ID
        "event_timestamp": datetime.utcnow().isoformat() + "Z",
        "transaction_id":  int(row.get("TransactionID", -1)),
        "amount":          float(row.get("TransactionAmt", 0)),
        "product_cd":      str(row.get("ProductCD", "unknown")),
        "card1":           float(row.get("card1", -999)),
        "card2":           float(row.get("card2", -999)),
        "card4":           str(row.get("card4", "unknown")),
        "card6":           str(row.get("card6", "unknown")),
        "p_email_domain":  str(row.get("P_emaildomain", "unknown")),
        "r_email_domain":  str(row.get("R_emaildomain", "unknown")),
        "c1":              float(row.get("C1", -999)),
        "c2":              float(row.get("C2", -999)),
        "addr1":           float(row.get("addr1", -999)),
        "dist1":           float(row.get("dist1", -999)),
        # Ground-truth label — present in training data only
        # In production this would be absent; model predicts it
        "is_fraud_label":  int(row.get("isFraud", -1)),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fraud transaction Kafka producer")
    parser.add_argument("--file",  default="data/train_transaction.csv",
                        help="Path to IEEE-CIS train_transaction.csv")
    parser.add_argument("--speed", type=float, default=0.05,
                        help="Seconds between messages (0 = as fast as possible)")
    parser.add_argument("--limit", type=int,   default=5000,
                        help="Max rows to publish (default 5000 for testing)")
    args = parser.parse_args()

    df       = load_dataset(args.file)
    producer = make_producer()
    subset   = df.head(args.limit)

    print(f"\n🚀 Publishing {len(subset):,} transactions to topic '{TOPIC}'…")
    print(f"   Speed: {args.speed}s between messages\n")

    for i, (_, row) in enumerate(subset.iterrows(), start=1):
        event = row_to_event(row.to_dict())
        producer.send(TOPIC, value=event)

        # Progress log every 100 messages
        if i % 100 == 0:
            label = "🚨 FRAUD" if event["is_fraud_label"] == 1 else "✅ legit"
            print(f"   [{i:>5}/{len(subset)}] txn={event['transaction_id']} "
                  f"amt=${event['amount']:.2f} {label}")

        time.sleep(args.speed)

    producer.flush()
    producer.close()
    print(f"\n✅ Done. Published {len(subset):,} events to '{TOPIC}'.")


if __name__ == "__main__":
    main()
