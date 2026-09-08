"""
monitoring/dashboard.py — Session 6: Real-time Fraud Detection Dashboard
Run: streamlit run monitoring/dashboard.py
"""

import json
import time
import redis
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from collections import deque

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Config ────────────────────────────────────────────────────────────────────
API_URL    = "https://fraud-detection-api-4zha.onrender.com"
REDIS_HOST = "localhost"
REDIS_PORT = 6379

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid #7c3aed;
    }
    .fraud-alert {
        background: #2d1515;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #ef4444;
        margin: 4px 0;
    }
    .legit-txn {
        background: #152d1e;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #22c55e;
        margin: 4px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Connect to Redis ──────────────────────────────────────────────────────────
@st.cache_resource
def get_redis():
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        r.ping()
        return r
    except Exception:
        return None


# ── Check API health ──────────────────────────────────────────────────────────
@st.cache_data(ttl=10)
def get_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.json()
    except Exception:
        return None


# ── Score a transaction via API ───────────────────────────────────────────────
def score_transaction(txn: dict) -> dict:
    try:
        r = requests.post(
            f"{API_URL}/predict",
            json=txn,
            timeout=3,
        )
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


# ── Generate synthetic transaction ───────────────────────────────────────────
def generate_transaction(force_fraud: bool = False) -> dict:
    """Generate a random transaction for live demo."""
    if force_fraud:
        return {
            "TransactionAmt":  float(np.random.choice([100, 200, 500, 1000])),
            "card1":           float(np.random.randint(1000, 2000)),
            "card2":           -999.0,
            "C1": 1.0, "C2": 1.0, "C6": 1.0, "C11": 1.0,
            "V1": 0.0, "V2": 0.0, "V3": 0.0,
            "addr1":           -999.0,
            "dist1":           -999.0,
            "card1_txn_count": 1.0,
            "product_encoded": 2,
            "email_match":     0,
        }
    else:
        return {
            "TransactionAmt":  float(round(np.random.uniform(5, 300), 2)),
            "card1":           float(np.random.randint(5000, 15000)),
            "card2":           float(np.random.randint(100, 600)),
            "C1": 1.0, "C2": 1.0, "C6": 1.0, "C11": 1.0,
            "V1": 1.0, "V2": 1.0, "V3": 1.0,
            "addr1":           float(np.random.randint(100, 500)),
            "dist1":           float(np.random.randint(0, 100)),
            "card1_txn_count": float(np.random.randint(5, 50)),
            "product_encoded": int(np.random.choice([0, 1, 3, 4])),
            "email_match":     1,
        }


# ── Session state for live transactions ──────────────────────────────────────
if "transactions" not in st.session_state:
    st.session_state.transactions = deque(maxlen=100)
if "total"  not in st.session_state: st.session_state.total  = 0
if "fraud"  not in st.session_state: st.session_state.fraud  = 0
if "alerts" not in st.session_state: st.session_state.alerts = deque(maxlen=10)


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

st.title("🛡️ Real-Time Fraud Detection Dashboard")
st.caption("Live transaction scoring powered by XGBoost + SHAP + FastAPI")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")

    health = get_api_health()
    if health:
        st.success("✅ API Online")
        st.caption(f"Threshold: {health.get('threshold', 'N/A'):.3f}")
        st.caption(f"Features: {health.get('features', 'N/A')}")
    else:
        st.error("❌ API Offline — start uvicorn first")

    r = get_redis()
    if r:
        st.success("✅ Redis Online")
    else:
        st.error("❌ Redis Offline")

    st.divider()
    st.subheader("🎮 Live Demo")
    col1, col2 = st.columns(2)
    score_legit = col1.button("Score Legit", use_container_width=True)
    score_fraud = col2.button("Score Fraud", use_container_width=True)
    auto_run    = st.toggle("Auto-score every 2s")
    st.divider()
    st.caption("Built with Kafka · XGBoost · FastAPI · Redis · Streamlit")


# ── Score transaction on button click ────────────────────────────────────────
def process_and_store(force_fraud=False):
    txn  = generate_transaction(force_fraud=force_fraud)
    pred = score_transaction(txn)
    if pred:
        record = {
            "time":       datetime.now().strftime("%H:%M:%S"),
            "amount":     txn["TransactionAmt"],
            "fraud_prob": pred["fraud_prob"],
            "is_fraud":   pred["is_fraud"],
            "risk_level": pred["risk_level"],
            "top_factor": pred["top_shap_factors"][0]["feature"] if pred["top_shap_factors"] else "N/A",
        }
        st.session_state.transactions.appendleft(record)
        st.session_state.total += 1
        if pred["is_fraud"]:
            st.session_state.fraud += 1
            st.session_state.alerts.appendleft(record)


if score_legit: process_and_store(force_fraud=False)
if score_fraud: process_and_store(force_fraud=True)
if auto_run:
    process_and_store(force_fraud=np.random.random() < 0.1)
    time.sleep(2)
    st.rerun()


# ── KPI Metrics Row ───────────────────────────────────────────────────────────
st.subheader("📊 Live Metrics")
k1, k2, k3, k4 = st.columns(4)

total     = st.session_state.total
fraud     = st.session_state.fraud
fraud_rate= (fraud / total * 100) if total > 0 else 0.0
txns      = list(st.session_state.transactions)
avg_prob  = np.mean([t["fraud_prob"] for t in txns]) if txns else 0.0

k1.metric("Total Scored",   total)
k2.metric("Fraud Detected", fraud,  delta=f"{fraud_rate:.1f}% rate")
k3.metric("Avg Fraud Prob", f"{avg_prob:.3f}")
k4.metric("Model Threshold",f"{health.get('threshold', 0):.3f}" if health else "N/A")


# ── Charts Row ────────────────────────────────────────────────────────────────
if txns:
    df = pd.DataFrame(txns)

    st.subheader("📈 Transaction Analysis")
    c1, c2 = st.columns(2)

    # Fraud probability over time
    with c1:
        fig = px.line(
            df, y="fraud_prob", x=range(len(df)),
            title="Fraud Probability — Recent Transactions",
            labels={"x": "Transaction", "fraud_prob": "Fraud Probability"},
            color_discrete_sequence=["#7c3aed"],
        )
        fig.add_hline(
            y=health.get("threshold", 0.5) if health else 0.5,
            line_dash="dash", line_color="red",
            annotation_text="Threshold"
        )
        fig.update_layout(
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Risk level distribution
    with c2:
        risk_counts = df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        color_map = {
            "MINIMAL":  "#22c55e",
            "LOW":      "#84cc16",
            "MEDIUM":   "#f59e0b",
            "HIGH":     "#f97316",
            "CRITICAL": "#ef4444",
        }
        fig2 = px.bar(
            risk_counts, x="Risk Level", y="Count",
            title="Risk Level Distribution",
            color="Risk Level",
            color_discrete_map=color_map,
        )
        fig2.update_layout(
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="white",
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Amount vs fraud prob scatter
    st.subheader("💰 Amount vs Fraud Probability")
    fig3 = px.scatter(
        df, x="amount", y="fraud_prob",
        color="is_fraud",
        color_discrete_map={True: "#ef4444", False: "#22c55e"},
        labels={"amount": "Transaction Amount ($)", "fraud_prob": "Fraud Probability"},
        title="Transaction Amount vs Fraud Probability",
        hover_data=["risk_level", "top_factor"],
    )
    fig3.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
    )
    st.plotly_chart(fig3, use_container_width=True)


# ── Recent Alerts ─────────────────────────────────────────────────────────────
st.subheader("🚨 Recent Fraud Alerts")
alerts = list(st.session_state.alerts)
if alerts:
    for alert in alerts[:5]:
        st.markdown(f"""
        <div class="fraud-alert">
            🚨 <b>FRAUD DETECTED</b> — {alert['time']} |
            Amount: <b>${alert['amount']:.2f}</b> |
            Prob: <b>{alert['fraud_prob']:.3f}</b> |
            Risk: <b>{alert['risk_level']}</b> |
            Top factor: <b>{alert['top_factor']}</b>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("No fraud alerts yet — click 'Score Fraud' to simulate one.")


# ── Transaction Log ───────────────────────────────────────────────────────────
st.subheader("📋 Transaction Log")
if txns:
    log_df = pd.DataFrame(txns)[
        ["time", "amount", "fraud_prob", "risk_level", "is_fraud", "top_factor"]
    ]
    log_df.columns = ["Time", "Amount ($)", "Fraud Prob", "Risk Level", "Is Fraud", "Top SHAP Factor"]
    st.dataframe(
        log_df.style.apply(
            lambda row: ["background-color: #2d1515" if row["Is Fraud"]
                        else "background-color: #152d1e"] * len(row),
            axis=1
        ),
        use_container_width=True,
        height=300,
    )
else:
    st.info("No transactions scored yet. Click 'Score Legit' or 'Score Fraud' to start.")
