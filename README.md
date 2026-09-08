# Real-Time Fraud Detection Pipeline

**Stack:** Kafka · XGBoost · FastAPI · Redis · Evidently · Streamlit · CausalML  
**Goal:** End-to-end ML pipeline that detects fraud in real time and measures causal impact of interventions.

---

## Project Structure

```
fraud-pipeline/
├── docker-compose.yml        ← Kafka + Redis infrastructure
├── requirements.txt          ← All Python dependencies
├── data/                     ← Put IEEE-CIS CSV files here
├── ingestion/
│   ├── producer.py           ← Publishes transactions to Kafka
│   └── verify_consumer.py   ← Sanity check: reads from Kafka
├── model/                    ← Feature engineering + XGBoost (Week 2)
├── serving/                  ← FastAPI inference endpoint (Week 3)
├── monitoring/               ← Evidently + Streamlit dashboard (Week 4)
└── causal/                   ← CausalML impact measurement (Week 5)
```

---

## Week 1 · Session 1: Setup

### Step 1 — Prerequisites

Install these before anything else:

| Tool | Download | Why |
|------|----------|-----|
| Docker Desktop | https://docs.docker.com/get-docker/ | Runs Kafka + Redis |
| Python 3.10+ | https://www.python.org/downloads/ | Project language |
| Git | https://git-scm.com/ | Version control |

Verify installs:
```bash
docker --version    # Docker version 24+
python --version    # Python 3.10+
```

---

### Step 2 — Get the Dataset

1. Go to https://www.kaggle.com/c/ieee-fraud-detection/data
2. Click **Download All** (you need a free Kaggle account)
3. Unzip and copy `train_transaction.csv` into the `data/` folder

```
fraud-pipeline/
└── data/
    └── train_transaction.csv   ← ~500 MB, 590k rows
```

---

### Step 3 — Python Environment

```bash
# From the fraud-pipeline/ directory:
python -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

---

### Step 4 — Start Infrastructure

```bash
# Start Kafka + Zookeeper + Redis in the background
docker-compose up -d

# Verify all 4 containers are running
docker ps
```

You should see these containers:
- `zookeeper` — Kafka's coordination service
- `kafka` — The message broker
- `kafka-ui` — Visual dashboard (optional but helpful)
- `redis` — Online feature store (used in Week 3)

Open http://localhost:8080 in your browser → you'll see the Kafka UI.

---

### Step 5 — Publish Transactions to Kafka

```bash
# Publish the first 5,000 rows (takes ~4 min at 0.05s/msg)
python ingestion/producer.py --file data/train_transaction.csv --limit 5000

# Or go faster (0s delay between messages)
python ingestion/producer.py --file data/train_transaction.csv --limit 5000 --speed 0
```

Watch the output — you'll see progress logs every 100 messages.

---

### Step 6 — Verify Messages Arrived

Open a **second terminal**, activate venv, then:

```bash
python ingestion/verify_consumer.py --limit 20
```

You should see 20 transactions printed with fraud labels. Fraud rate should be ~3.5%.

Also check http://localhost:8080 → Topics → `fraud-transactions` → Messages.

---

### Session 1 Done ✅

You now have:
- [x] Kafka running locally in Docker
- [x] Transactions streaming from the IEEE-CIS dataset
- [x] Messages confirmed arriving in the topic
- [x] Redis ready for Week 3's feature store

**Next session (Day 3–4):** EDA on the dataset + understanding class imbalance.
