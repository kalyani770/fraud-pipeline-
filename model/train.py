"""
train.py — Session 3: Train XGBoost Fraud Detection Model
Run: python model/train.py
Outputs:
  - model/fraud_model.joblib       (trained model)
  - model/feature_names.joblib     (feature list)
  - model/threshold.joblib         (optimal threshold)
  - model/plots/04_pr_curve.png    (precision-recall curve)
  - model/plots/05_shap.png        (SHAP feature importance)
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
import shap

# ── Config ────────────────────────────────────────────────────────────────────
PARQUET_PATH  = "data/features_offline.parquet"
MODEL_OUT     = "model/fraud_model.joblib"
FEATURES_OUT  = "model/feature_names.joblib"
THRESHOLD_OUT = "model/threshold.joblib"
PLOTS_DIR     = "model/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Step 1: Load Features ─────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: LOADING FEATURES")
print("=" * 60)

df = pd.read_parquet(PARQUET_PATH)
print(f"Loaded : {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Fraud  : {df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.2f}%)")

# Drop ID column, keep everything else as features
DROP_COLS = ["TransactionID", "isFraud"]
FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS]

X = df[FEATURE_COLS].astype(np.float32)
y = df["isFraud"].astype(int)

print(f"Features: {len(FEATURE_COLS)}")


# ── Step 2: Train / Test Split ────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 2: TRAIN / TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y        # keeps fraud % same in both splits
)

print(f"Train : {len(X_train):,} rows | fraud: {y_train.sum():,}")
print(f"Test  : {len(X_test):,}  rows | fraud: {y_test.sum():,}")
print()
print("WHY stratify=y?")
print("  Without it, random split might put most fraud in train")
print("  and none in test — model would look perfect but never be")
print("  tested on real fraud cases.")


# ── Step 3: Train XGBoost ─────────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 3: TRAINING XGBOOST")
print("=" * 60)

# Calculate imbalance ratio for scale_pos_weight
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
spw = round(neg / pos, 1)
print(f"scale_pos_weight = {spw} (negatives/positives = {neg}/{pos})")
print()
print("Training... (takes 1-2 minutes)")

model = XGBClassifier(
    n_estimators=300,        # number of trees
    max_depth=6,             # depth of each tree
    learning_rate=0.05,      # step size — smaller = more careful
    scale_pos_weight=spw,    # handle class imbalance
    subsample=0.8,           # use 80% of rows per tree (prevents overfitting)
    colsample_bytree=0.8,    # use 80% of features per tree
    eval_metric="aucpr",     # optimize for PR-AUC directly
    random_state=42,
    n_jobs=-1,               # use all CPU cores
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,              # print progress every 50 trees
)

print("\nTraining complete!")


# ── Step 4: Evaluate with PR-AUC ─────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 4: EVALUATION — PR-AUC")
print("=" * 60)

y_prob = model.predict_proba(X_test)[:, 1]   # fraud probability
pr_auc = average_precision_score(y_test, y_prob)
print(f"PR-AUC Score: {pr_auc:.4f}")
print()
print("WHAT THIS MEANS:")
print("  PR-AUC = 1.0 → perfect model")
print("  PR-AUC = 0.5 → random guessing on balanced data")
print("  A good fraud model typically scores 0.7 - 0.85")


# ── Step 5: Threshold Tuning ──────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 5: THRESHOLD TUNING")
print("=" * 60)

precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_prob)

# Find threshold that maximises F1 score
f1_scores = (2 * precision_vals[:-1] * recall_vals[:-1] /
             (precision_vals[:-1] + recall_vals[:-1] + 1e-8))
best_idx       = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_precision = precision_vals[best_idx]
best_recall    = recall_vals[best_idx]
best_f1        = f1_scores[best_idx]

print(f"Best threshold : {best_threshold:.3f}")
print(f"At this point  :")
print(f"  Precision : {best_precision:.3f}  (of flagged txns, this % are real fraud)")
print(f"  Recall    : {best_recall:.3f}  (of all fraud, this % were caught)")
print(f"  F1 Score  : {best_f1:.3f}")
print()
print("WHY NOT 0.5?")
print("  Default threshold of 0.5 is arbitrary.")
print("  We tune it to find the precision/recall balance")
print("  that makes sense for the business.")

# Plot PR Curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Model Evaluation", fontsize=14, fontweight="bold")

# PR Curve
axes[0].plot(recall_vals, precision_vals, color="#3498db", linewidth=2,
             label=f"XGBoost (PR-AUC = {pr_auc:.3f})")
axes[0].axhline(y=y_test.mean(), color="red", linestyle="--",
                label=f"Random baseline ({y_test.mean():.3f})")
axes[0].scatter(best_recall, best_precision, color="red", s=100, zorder=5,
                label=f"Best threshold ({best_threshold:.2f})")
axes[0].set_xlabel("Recall (fraud cases caught)")
axes[0].set_ylabel("Precision (flagged = real fraud)")
axes[0].set_title("Precision-Recall Curve")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Confusion matrix at best threshold
y_pred = (y_prob >= best_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred)
im = axes[1].imshow(cm, interpolation="nearest", cmap="Blues")
axes[1].set_title(f"Confusion Matrix (threshold={best_threshold:.2f})")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(["Legit", "Fraud"])
axes[1].set_yticklabels(["Legit", "Fraud"])
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black",
                    fontsize=12, fontweight="bold")
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/04_pr_curve.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"\nPlot saved: {PLOTS_DIR}/04_pr_curve.png")

# Print classification report
print()
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))


# ── Step 6: SHAP Explainability ───────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 6: SHAP EXPLAINABILITY")
print("=" * 60)
print("Computing SHAP values (takes 1-2 minutes)...")

# Use a sample for speed
sample_size = min(2000, len(X_test))
X_sample    = X_test.iloc[:sample_size]

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# Global feature importance plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("SHAP Feature Importance", fontsize=14, fontweight="bold")

plt.sca(axes[0])
shap.summary_plot(
    shap_values, X_sample,
    plot_type="bar",
    max_display=15,
    show=False,
    plot_size=None,
)
axes[0].set_title("Top 15 Features by Mean |SHAP|")

plt.sca(axes[1])
shap.summary_plot(
    shap_values, X_sample,
    max_display=15,
    show=False,
    plot_size=None,
)
axes[1].set_title("SHAP Value Distribution")

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/05_shap.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"Plot saved: {PLOTS_DIR}/05_shap.png")

# Top 10 features by importance
mean_shap   = np.abs(shap_values).mean(axis=0)
top_indices = np.argsort(mean_shap)[::-1][:10]
print()
print("Top 10 most important features:")
for rank, idx in enumerate(top_indices, 1):
    print(f"  {rank:>2}. {FEATURE_COLS[idx]:<30} SHAP={mean_shap[idx]:.4f}")


# ── Step 7: Save Model ────────────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 7: SAVING MODEL")
print("=" * 60)

joblib.dump(model,         MODEL_OUT)
joblib.dump(FEATURE_COLS,  FEATURES_OUT)
joblib.dump(best_threshold, THRESHOLD_OUT)

print(f"Model saved     : {MODEL_OUT}")
print(f"Features saved  : {FEATURES_OUT}")
print(f"Threshold saved : {THRESHOLD_OUT}")
print()
print("=" * 60)
print("SESSION 3 COMPLETE ✅")
print("=" * 60)
print()
print("Next steps (Session 4):")
print("  1. Wrap model in FastAPI endpoint")
print("  2. POST /predict → returns fraud_prob + is_fraud + shap")
print("  3. Connect Kafka consumer → FastAPI → real-time scoring")
