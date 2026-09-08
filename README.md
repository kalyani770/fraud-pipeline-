# 🛡️ Real-Time Fraud Detection Pipeline

> End-to-end ML system detecting payment fraud in real time with causal impact measurement of interventions.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Kafka](https://img.shields.io/badge/Kafka-7.5.0-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.108-teal)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red)

---

## Project Overview

Most fraud models tell you **who** is fraudulent. This system goes further — it also tells you **for whom blocking actually works**, using causal inference to separate genuine intervention effects from selection bias.

**Built to answer three questions:**
1. Is this transaction fraudulent? *(XGBoost classifier)*
2. Why did the model flag it? *(SHAP explainability)*
3. Did blocking it actually cause fewer losses? *(T-Learner causal inference)*

---

## Architecture

```
Payment Stream (IEEE-CIS Dataset)
        |
   Kafka Topic [fraud-transactions]
        |
   Real-time Consumer
     |              |
  Redis           FastAPI /predict
(velocity         XGBoost Model
 features)        SHAP Explanation
        |
   Streamlit Dashboard
        |
   T-Learner Causal Inference
```

---

## Results

| Metric | Value |
|--------|-------|
| Dataset | IEEE-CIS (590,540 transactions) |
| Fraud rate | 3.5% (27:1 class imbalance) |
| Model | XGBoost (300 trees, scale_pos_weight=27.6) |
| Evaluation | PR-AUC (not accuracy) |
| Inference latency | ~23ms end-to-end |
| Threshold | Tuned for F1 (not default 0.5) |

### Causal Impact (T-Learner)

| Segment | CATE | Implication |
|---------|------|-------------|
| Q1 Low Risk | +82.15% | Selection bias dominates — allow through |
| Q2 Med Risk | +71.31% | Soft intervention (OTP) recommended |
| Q3 High Risk | +60.91% | Monitor closely |
| Q4 Very High | +57.18% | Hard block |

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Ingestion | Apache Kafka | Real-time transaction streaming |
| Online features | Redis | Sub-millisecond velocity features |
| Offline features | Parquet | Training feature store |
| Model | XGBoost + SHAP | Fraud classification + explainability |
| Serving | FastAPI | Low-latency inference endpoint |
| Monitoring | Streamlit + Plotly | Live dashboard |
| Causal inference | T-Learner (XGBoost) | Intervention impact measurement |
| Infrastructure | Docker Compose | Reproducible environment |

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/fraud-pipeline
cd fraud-pipeline
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
docker-compose up -d
python model/eda.py
python model/train.py
uvicorn serving.api:app --reload --port 8000
streamlit run monitoring/dashboard.py
```

---

## Key Engineering Decisions

**Why Kafka?** Decouples producers from consumers. If model crashes, messages queue and replay. Database would lose transactions.

**Why Redis?** Velocity features need <1ms reads. Postgres takes 10-50ms — too slow for 100ms fraud decision budget.

**Why PR-AUC?** 96.5% accuracy by predicting nothing. PR-AUC is the only meaningful metric for 27:1 imbalanced data.

**Why threshold tuning?** Default 0.5 is arbitrary. F1-optimal threshold surfaces the precision/recall trade-off to business stakeholders.

**Why CausalML?** Fraud model tells us who is high risk. T-Learner tells us for whom intervention actually works — a completely different question driving tiered intervention policy.

---

## Author

Built by **Kalyani** — Data Scientist portfolio project targeting FAANG-tier ML Engineer roles.
