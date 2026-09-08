"""
verify_consumer.py — Quick sanity check: reads messages from the
fraud-transactions topic and prints a summary of what it sees.

Run AFTER producer.py in a separate terminal.
Usage: python ingestion/verify_consumer.py --limit 10
"""

import argparse
import json

from kafka import KafkaConsumer

KAFKA_BROKER = "localhost:9092"
TOPIC        = "fraud-transactions"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10,
                        help="Number of messages to read before stopping")
    args = parser.parse_args()

    print(f"👂 Listening on '{TOPIC}' (will stop after {args.limit} messages)…\n")

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        auto_offset_reset="earliest",        # read from the very beginning
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        consumer_timeout_ms=10_000,          # stop if no message for 10s
    )

    count = fraud_count = 0
    for msg in consumer:
        event = msg.value
        count += 1
        is_fraud = event.get("is_fraud_label", -1)
        if is_fraud == 1:
            fraud_count += 1

        label = "🚨 FRAUD" if is_fraud == 1 else "✅ legit"
        print(
            f"[{count:>3}] {label} | "
            f"txn={event.get('transaction_id')} | "
            f"amt=${event.get('amount', 0):.2f} | "
            f"product={event.get('product_cd')} | "
            f"ts={event.get('event_timestamp')}"
        )

        if count >= args.limit:
            break

    consumer.close()
    print(f"\n📊 Summary: {count} messages read, {fraud_count} fraud "
          f"({fraud_count/max(count,1)*100:.1f}%)")


if __name__ == "__main__":
    main()
