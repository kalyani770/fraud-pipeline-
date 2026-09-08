"""
eda.py — Session 2: Exploratory Data Analysis + Feature Engineering
Run: python model/eda.py
Outputs:
  - data/features_offline.parquet   (offline feature store)
  - model/eda_report.txt            (summary of findings)
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = "data/train_transaction.csv"
PARQUET_OUT  = "data/features_offline.parquet"
PLOTS_DIR    = "model/plots"
REPORT_OUT   = "model/eda_report.txt"

os.makedirs(PLOTS_DIR, exist_ok=True)

report_lines = []

def log(msg=""):
    print(msg)
    report_lines.append(msg)


# ── Step 1: Load Data ─────────────────────────────────────────────────────────
log("=" * 60)
log("STEP 1: LOADING DATA")
log("=" * 60)

df = pd.read_csv(DATA_PATH)
log(f"Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
log(f"Memory usage   : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
log(f"Fraud cases    : {df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.2f}%)")
log(f"Legit cases    : {(df['isFraud']==0).sum():,} ({(df['isFraud']==0).mean()*100:.2f}%)")


# ── Step 2: Class Imbalance Analysis ─────────────────────────────────────────
log()
log("=" * 60)
log("STEP 2: CLASS IMBALANCE")
log("=" * 60)

fraud_count = df['isFraud'].sum()
legit_count = (df['isFraud'] == 0).sum()
imbalance_ratio = legit_count / fraud_count

log(f"Imbalance ratio : {imbalance_ratio:.1f}:1 (legit:fraud)")
log(f"scale_pos_weight: {imbalance_ratio:.1f}  ← use this in XGBoost")
log()
log("WHY THIS MATTERS:")
log("  A naive model predicting 'not fraud' always gets 96.5% accuracy")
log("  but catches ZERO fraud. Accuracy is useless here.")
log("  We use Precision-Recall AUC instead.")

# Plot class distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Class Imbalance Analysis", fontsize=14, fontweight="bold")

# Bar chart
axes[0].bar(["Legit", "Fraud"], [legit_count, fraud_count],
            color=["#2ecc71", "#e74c3c"], edgecolor="black", linewidth=0.5)
axes[0].set_title("Transaction Count by Class")
axes[0].set_ylabel("Count")
for i, v in enumerate([legit_count, fraud_count]):
    axes[0].text(i, v + 1000, f"{v:,}", ha="center", fontweight="bold")

# Transaction amount by class
axes[1].hist(df[df['isFraud']==0]['TransactionAmt'].clip(0, 500),
             bins=50, alpha=0.6, label="Legit", color="#2ecc71")
axes[1].hist(df[df['isFraud']==1]['TransactionAmt'].clip(0, 500),
             bins=50, alpha=0.6, label="Fraud", color="#e74c3c")
axes[1].set_title("Transaction Amount Distribution")
axes[1].set_xlabel("Amount (clipped at $500)")
axes[1].set_ylabel("Count")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/01_class_imbalance.png", dpi=120, bbox_inches="tight")
plt.close()
log(f"\n  Plot saved: {PLOTS_DIR}/01_class_imbalance.png")


# ── Step 3: Missing Value Analysis ───────────────────────────────────────────
log()
log("=" * 60)
log("STEP 3: MISSING VALUES")
log("=" * 60)

missing = df.isnull().mean().sort_values(ascending=False)
high_missing = missing[missing > 0.5]
log(f"Columns with >50% missing: {len(high_missing)}")
log(f"We will DROP these columns — too sparse to be useful")
log()

# Show top 10 most missing
log("Top 10 columns by missing %:")
for col, pct in missing.head(10).items():
    log(f"  {col:<20} {pct*100:.1f}% missing")

# Plot missing values
fig, ax = plt.subplots(figsize=(12, 4))
missing_plot = missing[missing > 0].head(40)
colors = ["#e74c3c" if v > 0.5 else "#f39c12" if v > 0.2 else "#3498db"
          for v in missing_plot.values]
ax.bar(range(len(missing_plot)), missing_plot.values * 100, color=colors)
ax.axhline(50, color="red", linestyle="--", label=">50% → drop")
ax.axhline(20, color="orange", linestyle="--", label=">20% → impute carefully")
ax.set_title("Missing Value % by Column (top 40)")
ax.set_ylabel("Missing %")
ax.set_xticks(range(len(missing_plot)))
ax.set_xticklabels(missing_plot.index, rotation=90, fontsize=7)
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/02_missing_values.png", dpi=120, bbox_inches="tight")
plt.close()
log(f"\n  Plot saved: {PLOTS_DIR}/02_missing_values.png")


# ── Step 4: Key Feature Analysis ─────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 4: KEY FEATURE ANALYSIS")
log("=" * 60)

# Transaction amount stats by class
log("TransactionAmt stats:")
log(f"  Fraud  — mean: ${df[df['isFraud']==1]['TransactionAmt'].mean():.2f} "
    f"| median: ${df[df['isFraud']==1]['TransactionAmt'].median():.2f}")
log(f"  Legit  — mean: ${df[df['isFraud']==0]['TransactionAmt'].mean():.2f} "
    f"| median: ${df[df['isFraud']==0]['TransactionAmt'].median():.2f}")

# ProductCD fraud rate
log()
log("Fraud rate by ProductCD:")
prod_fraud = df.groupby("ProductCD")["isFraud"].agg(["mean", "count"])
prod_fraud.columns = ["fraud_rate", "count"]
prod_fraud = prod_fraud.sort_values("fraud_rate", ascending=False)
for prod, row in prod_fraud.iterrows():
    log(f"  {prod}: {row['fraud_rate']*100:.1f}% fraud rate ({row['count']:,} txns)")

# Card type fraud rate
if "card4" in df.columns:
    log()
    log("Fraud rate by card network (card4):")
    card_fraud = df.groupby("card4")["isFraud"].agg(["mean","count"])
    card_fraud.columns = ["fraud_rate","count"]
    card_fraud = card_fraud.sort_values("fraud_rate", ascending=False)
    for card, row in card_fraud.iterrows():
        log(f"  {card}: {row['fraud_rate']*100:.1f}% ({row['count']:,} txns)")

# Plot key features
fig = plt.figure(figsize=(14, 5))
gs  = gridspec.GridSpec(1, 3, figure=fig)
fig.suptitle("Key Feature Analysis", fontsize=14, fontweight="bold")

# ProductCD fraud rate
ax1 = fig.add_subplot(gs[0])
ax1.bar(prod_fraud.index, prod_fraud["fraud_rate"] * 100, color="#3498db", edgecolor="black", linewidth=0.5)
ax1.set_title("Fraud Rate by ProductCD")
ax1.set_ylabel("Fraud Rate %")
ax1.set_xlabel("Product Code")

# Amount boxplot
ax2 = fig.add_subplot(gs[1])
data_box = [
    df[df['isFraud']==0]['TransactionAmt'].clip(0, 500).dropna(),
    df[df['isFraud']==1]['TransactionAmt'].clip(0, 500).dropna()
]
bp = ax2.boxplot(data_box, tick_labels=["Legit", "Fraud"], patch_artist=True)
bp["boxes"][0].set_facecolor("#2ecc71")
bp["boxes"][1].set_facecolor("#e74c3c")
ax2.set_title("Amount Distribution (clipped $500)")
ax2.set_ylabel("Transaction Amount ($)")

# Card4 fraud rates
ax3 = fig.add_subplot(gs[2])
if "card4" in df.columns:
    ax3.barh(card_fraud.index, card_fraud["fraud_rate"] * 100, color="#9b59b6", edgecolor="black", linewidth=0.5)
    ax3.set_title("Fraud Rate by Card Network")
    ax3.set_xlabel("Fraud Rate %")

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/03_key_features.png", dpi=120, bbox_inches="tight")
plt.close()
log(f"\n  Plot saved: {PLOTS_DIR}/03_key_features.png")


# ── Step 5: Feature Engineering ──────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 5: FEATURE ENGINEERING")
log("=" * 60)
log("Building offline feature table...")

# Select core columns (drop >50% missing)
cols_to_drop = missing[missing > 0.5].index.tolist()
df_feat = df.drop(columns=cols_to_drop)
log(f"Dropped {len(cols_to_drop)} high-missing columns")
log(f"Remaining columns: {df_feat.shape[1]}")

# ── Feature 1: Log-transform amount (handles skew) ──
df_feat["amt_log"] = np.log1p(df["TransactionAmt"])
log("\n[+] amt_log: log(1 + TransactionAmt) — reduces right skew")

# ── Feature 2: Amount is round number (fraud signal) ──
df_feat["amt_is_round"] = (df["TransactionAmt"] % 1 == 0).astype(int)
log("[+] amt_is_round: 1 if amount has no cents — fraudsters often use round numbers")

# ── Feature 3: High-risk amount bands ──
df_feat["amt_band"] = pd.cut(
    df["TransactionAmt"],
    bins=[0, 50, 200, 500, 1000, np.inf],
    labels=["low", "medium", "high", "very_high", "extreme"]
)
df_feat["amt_band"] = df_feat["amt_band"].astype(str)
log("[+] amt_band: bucketed amount — captures non-linear fraud risk by amount range")

# ── Feature 4: Email domain match (same sender/receiver domain) ──
if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
    df_feat["email_match"] = (
        df["P_emaildomain"] == df["R_emaildomain"]
    ).astype(int)
    log("[+] email_match: 1 if purchaser and recipient email domains match")

# ── Feature 5: Card velocity proxy (card1 transaction count) ──
card_counts = df.groupby("card1")["TransactionID"].transform("count")
df_feat["card1_txn_count"] = card_counts
log("[+] card1_txn_count: how many transactions this card appears in the dataset")

# ── Feature 6: ProductCD encoded ──
df_feat["product_encoded"] = df["ProductCD"].map(
    {"W": 0, "H": 1, "C": 2, "S": 3, "R": 4}
).fillna(-1).astype(int)
log("[+] product_encoded: ordinal encoding of ProductCD")

# ── Fill remaining nulls ──
num_cols = df_feat.select_dtypes(include=[np.number]).columns
df_feat[num_cols] = df_feat[num_cols].fillna(-999)
cat_cols = df_feat.select_dtypes(include=["object"]).columns
df_feat[cat_cols] = df_feat[cat_cols].fillna("unknown")

# ── Step 6: Save Offline Feature Store ───────────────────────────────────────
log()
log("=" * 60)
log("STEP 6: SAVING OFFLINE FEATURE STORE")
log("=" * 60)

# Keep only numeric features + label for model training
final_cols = (
    ["TransactionID", "isFraud"] +
    [c for c in df_feat.select_dtypes(include=[np.number]).columns
     if c not in ["TransactionID", "isFraud"]]
)
df_final = df_feat[final_cols]
df_final.to_parquet(PARQUET_OUT, index=False)

log(f"Saved to        : {PARQUET_OUT}")
log(f"Final shape     : {df_final.shape[0]:,} rows × {df_final.shape[1]} columns")
log(f"Feature count   : {df_final.shape[1] - 2} features (excl. ID + label)")
log()

# Feature summary
log("Engineered features added:")
new_feats = ["amt_log", "amt_is_round", "card1_txn_count", "product_encoded"]
if "email_match" in df_final.columns:
    new_feats.append("email_match")
for f in new_feats:
    if f in df_final.columns:
        fraud_mean   = df_final[df_final["isFraud"]==1][f].mean()
        legit_mean   = df_final[df_final["isFraud"]==0][f].mean()
        log(f"  {f:<25} fraud_mean={fraud_mean:.3f}  legit_mean={legit_mean:.3f}")


# ── Step 7: Save Report ───────────────────────────────────────────────────────
log()
log("=" * 60)
log("SESSION 2 COMPLETE ✅")
log("=" * 60)
log()
log("Next steps (Session 3):")
log("  1. Train XGBoost on features_offline.parquet")
log("  2. Evaluate with Precision-Recall AUC (not accuracy)")
log("  3. Add SHAP explainability")
log("  4. Tune classification threshold for business trade-off")

with open(REPORT_OUT, "w",encoding="utf-8") as f:
    f.write("\n".join(report_lines))
log(f"\nReport saved: {REPORT_OUT}")
