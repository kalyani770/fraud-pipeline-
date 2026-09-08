"""
causal/causal_impact.py — Session 7: Causal Impact of Fraud Interventions
Measures whether blocking flagged transactions CAUSED fewer fraud losses,
or whether those transactions would have resolved without intervention.

Uses meta-learner approach manually (no causalml library needed — 
avoids Windows compilation issues) with XGBoost T-Learner.

Run: python causal/causal_impact.py
Outputs:
  - causal/causal_report.txt
  - causal/plots/06_causal_impact.png
  - causal/plots/07_cate_by_segment.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

os.makedirs("causal/plots", exist_ok=True)

report = []
def log(msg=""):
    print(msg)
    report.append(msg)


log("=" * 60)
log("SESSION 7: CAUSAL IMPACT ANALYSIS")
log("=" * 60)
log()
log("QUESTION WE ARE ANSWERING:")
log("  Did blocking flagged transactions CAUSE fewer fraud losses?")
log("  Or would those fraudsters have stopped anyway?")
log()


# ── Step 1: Load data + model ─────────────────────────────────────────────────
log("=" * 60)
log("STEP 1: LOAD DATA + MODEL PREDICTIONS")
log("=" * 60)

df           = pd.read_parquet("data/features_offline.parquet")
model        = joblib.load("model/fraud_model.joblib")
feature_cols = joblib.load("model/feature_names.joblib")
threshold    = joblib.load("model/threshold.joblib")

X = df[feature_cols].astype(np.float32)
y = df["isFraud"].astype(int)

# Get model fraud probabilities
fraud_prob = model.predict_proba(X)[:, 1]
df["fraud_prob"]   = fraud_prob
df["model_flagged"] = (fraud_prob >= threshold).astype(int)

log(f"Dataset     : {len(df):,} transactions")
log(f"Flagged     : {df['model_flagged'].sum():,} ({df['model_flagged'].mean()*100:.1f}%)")
log(f"Actual fraud: {y.sum():,} ({y.mean()*100:.1f}%)")


# ── Step 2: Simulate Treatment/Control ───────────────────────────────────────
log()
log("=" * 60)
log("STEP 2: SIMULATE TREATMENT & CONTROL GROUPS")
log("=" * 60)
log()
log("REAL WORLD:")
log("  Treatment = transaction was BLOCKED by fraud system")
log("  Control   = transaction was ALLOWED through")
log("  Outcome   = did fraud loss actually occur?")
log()
log("SIMULATION (since we only have historical data):")
log("  Treatment = model flagged the transaction (model_flagged=1)")
log("  Control   = model did not flag it (model_flagged=0)")
log("  Outcome   = isFraud (ground truth label)")

# Treatment assignment — transactions the model flagged
df["treatment"] = df["model_flagged"]
df["outcome"]   = df["isFraud"]

# Naive comparison (before causal adjustment)
treated_outcome = df[df["treatment"]==1]["outcome"].mean()
control_outcome = df[df["treatment"]==0]["outcome"].mean()
naive_effect    = treated_outcome - control_outcome

log()
log("NAIVE COMPARISON (biased — ignores confounders):")
log(f"  Fraud rate in FLAGGED group  : {treated_outcome*100:.1f}%")
log(f"  Fraud rate in ALLOWED group  : {control_outcome*100:.1f}%")
log(f"  Naive difference             : {naive_effect*100:.1f}%")
log()
log("WHY THIS IS BIASED:")
log("  The model flags HIGH-RISK transactions by design.")
log("  Of course flagged transactions have more fraud —")
log("  that's what the model is supposed to do.")
log("  We need causal inference to get the TRUE effect.")


# ── Step 3: T-Learner (Manual Meta-Learner) ───────────────────────────────────
log()
log("=" * 60)
log("STEP 3: T-LEARNER CAUSAL ESTIMATION")
log("=" * 60)
log()
log("T-Learner trains TWO separate models:")
log("  Model T1 — trained ONLY on treated units (flagged txns)")
log("  Model T0 — trained ONLY on control units (allowed txns)")
log("  CATE = T1.predict(X) - T0.predict(X) for each transaction")
log("  CATE = Conditional Average Treatment Effect")

# Use a subset for speed
sample_size = min(50000, len(df))
df_sample   = df.sample(sample_size, random_state=42)

X_sample    = df_sample[feature_cols].astype(np.float32)
t_sample    = df_sample["treatment"].values
y_sample    = df_sample["outcome"].values

# Split treated and control
X_treated   = X_sample[t_sample == 1]
y_treated   = y_sample[t_sample == 1]
X_control   = X_sample[t_sample == 0]
y_control   = y_sample[t_sample == 0]

log()
log(f"Sample size : {sample_size:,}")
log(f"Treated     : {len(X_treated):,} transactions")
log(f"Control     : {len(X_control):,} transactions")
log()
log("Training T1 (treated model)...")

# Train T1 — outcome model for treated group
t1 = XGBClassifier(
    n_estimators=100, max_depth=4,
    learning_rate=0.1, random_state=42,
    eval_metric="logloss", n_jobs=-1,
)
t1.fit(X_treated, y_treated, verbose=False)

log("Training T0 (control model)...")

# Train T0 — outcome model for control group
t0 = XGBClassifier(
    n_estimators=100, max_depth=4,
    learning_rate=0.1, random_state=42,
    eval_metric="logloss", n_jobs=-1,
)
t0.fit(X_control, y_control, verbose=False)

log("Computing CATE (individual treatment effects)...")

# CATE = predicted outcome under treatment - predicted outcome under control
mu1   = t1.predict_proba(X_sample)[:, 1]   # P(fraud | treated)
mu0   = t0.predict_proba(X_sample)[:, 1]   # P(fraud | control)
cate  = mu1 - mu0                           # individual causal effect

df_sample = df_sample.copy()
df_sample["mu1"]  = mu1
df_sample["mu0"]  = mu0
df_sample["cate"] = cate

ate = cate.mean()   # Average Treatment Effect across all units

log()
log("CAUSAL RESULTS:")
log(f"  ATE (Average Treatment Effect) : {ate*100:.2f}%")
log()
if ate > 0:
    log("  INTERPRETATION: Blocking transactions INCREASES fraud outcome")
    log("  by {:.1f}% on average — suggests model is flagging legitimate".format(abs(ate*100)))
    log("  transactions that then get disputed as fraud.")
elif ate < 0:
    log(f"  INTERPRETATION: Blocking transactions REDUCES fraud outcome")
    log(f"  by {abs(ate)*100:.1f}% on average — interventions are effective.")
else:
    log("  INTERPRETATION: Blocking has no measurable causal effect.")


# ── Step 4: CATE by Segment ───────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 4: CATE BY CUSTOMER SEGMENT")
log("=" * 60)
log()
log("KEY INSIGHT: Average effect hides segment differences.")
log("Some customers respond strongly to intervention,")
log("others would have resolved fraud without intervention.")

# Segment by fraud probability quartile
df_sample["risk_quartile"] = pd.qcut(
    df_sample["fraud_prob"],
    q=4,
    labels=["Q1-Low Risk", "Q2-Med Risk", "Q3-High Risk", "Q4-Very High"]
)

segment_cate = df_sample.groupby("risk_quartile", observed=True)["cate"].agg(
    ["mean", "std", "count"]
).reset_index()
segment_cate.columns = ["Segment", "Mean CATE", "Std CATE", "Count"]

log()
log("CATE by Risk Segment:")
for _, row in segment_cate.iterrows():
    direction = "↑ increases" if row["Mean CATE"] > 0 else "↓ reduces"
    log(f"  {row['Segment']:<20} CATE={row['Mean CATE']*100:+.2f}%  "
        f"{direction} fraud | n={row['Count']:,}")

log()
log("BUSINESS IMPLICATION:")
log("  Block aggressively for Q4 (Very High Risk) — highest causal impact")
log("  Be careful with Q1 (Low Risk) — blocking may not help")


# ── Step 5: Plot Results ──────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 5: VISUALIZATIONS")
log("=" * 60)

# Plot 1: CATE distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Causal Impact Analysis — T-Learner Results",
             fontsize=14, fontweight="bold")

axes[0].hist(cate, bins=50, color="#7c3aed", alpha=0.8, edgecolor="black", linewidth=0.3)
axes[0].axvline(ate, color="red", linestyle="--", linewidth=2,
                label=f"ATE = {ate*100:.2f}%")
axes[0].axvline(0, color="white", linestyle="-", linewidth=1, alpha=0.5)
axes[0].set_title("Distribution of Individual Treatment Effects (CATE)")
axes[0].set_xlabel("CATE (causal effect on fraud probability)")
axes[0].set_ylabel("Count")
axes[0].legend()
axes[0].set_facecolor("#1e1e2e")
fig.patch.set_facecolor("#0e1117")
axes[0].tick_params(colors="white")
axes[0].title.set_color("white")
axes[0].xaxis.label.set_color("white")
axes[0].yaxis.label.set_color("white")

# Plot 2: CATE by segment
colors = ["#22c55e", "#f59e0b", "#f97316", "#ef4444"]
bars   = axes[1].bar(
    segment_cate["Segment"],
    segment_cate["Mean CATE"] * 100,
    color=colors, edgecolor="black", linewidth=0.5,
    yerr=segment_cate["Std CATE"] * 100,
    capsize=5, error_kw={"color": "white"},
)
axes[1].axhline(0, color="white", linestyle="-", linewidth=0.5, alpha=0.5)
axes[1].set_title("Average CATE by Risk Segment")
axes[1].set_xlabel("Risk Segment")
axes[1].set_ylabel("Mean CATE (%)")
axes[1].set_facecolor("#1e1e2e")
axes[1].tick_params(colors="white")
axes[1].title.set_color("white")
axes[1].xaxis.label.set_color("white")
axes[1].yaxis.label.set_color("white")
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=15, ha="right")

plt.tight_layout()
plt.savefig("causal/plots/06_causal_impact.png", dpi=120,
            bbox_inches="tight", facecolor="#0e1117")
plt.close()
log(f"Plot saved: causal/plots/06_causal_impact.png")

# Plot 3: Mu1 vs Mu0 scatter
fig2, ax = plt.subplots(figsize=(8, 6))
scatter  = ax.scatter(
    mu0[:2000], mu1[:2000],
    c=cate[:2000], cmap="RdYlGn_r",
    alpha=0.5, s=10,
)
ax.plot([0, 1], [0, 1], "w--", linewidth=1, label="No effect line")
ax.set_xlabel("P(fraud | control) — without intervention")
ax.set_ylabel("P(fraud | treated) — with intervention")
ax.set_title("Counterfactual Outcomes: Treatment vs Control")
ax.legend()
plt.colorbar(scatter, ax=ax, label="CATE")
ax.set_facecolor("#1e1e2e")
fig2.patch.set_facecolor("#0e1117")
ax.tick_params(colors="white")
ax.title.set_color("white")
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
plt.tight_layout()
plt.savefig("causal/plots/07_cate_by_segment.png", dpi=120,
            bbox_inches="tight", facecolor="#0e1117")
plt.close()
log(f"Plot saved: causal/plots/07_cate_by_segment.png")


# ── Step 6: Business Recommendations ─────────────────────────────────────────
log()
log("=" * 60)
log("STEP 6: BUSINESS RECOMMENDATIONS")
log("=" * 60)
log()

high_impact = df_sample[df_sample["cate"] < -0.1]
low_impact  = df_sample[df_sample["cate"] > 0.1]

log(f"High-impact interventions (CATE < -10%) : {len(high_impact):,} transactions")
log(f"  → Block these aggressively")
log()
log(f"Low/negative impact (CATE > +10%)       : {len(low_impact):,} transactions")
log(f"  → Consider softer intervention (flag but don't block)")
log()
log("POLICY RECOMMENDATION:")
log("  Tier 1 (CATE < -0.1) : Hard block — intervention strongly reduces fraud")
log("  Tier 2 (CATE -0.1–0) : Soft flag — monitor, add friction (OTP)")
log("  Tier 3 (CATE > 0)    : Allow — blocking may increase disputes")


# ── Save report ───────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("SESSION 7 COMPLETE ✅")
log("=" * 60)
log()
log("Next: Session 8 — GitHub README + deployment + resume write-up")

with open("causal/causal_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report))
print(f"\nReport saved: causal/causal_report.txt")
