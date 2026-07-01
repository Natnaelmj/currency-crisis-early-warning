"""
Currency Crisis Early-Warning Model
--------------------------------------
Trains classifiers to predict currency crisis risk from macroeconomic
early-warning indicators (EWIs), in the spirit of the Kaminsky-Reinhart
"signals approach" and later machine-learning EWI literature.

Compares a logistic regression (interpretable, coefficients map directly to
economic theory) against a random forest (usually higher accuracy, less
interpretable) -- a classic tradeoff in applied economics.

Run:
    python3 generate_sample_data.py   # only needed once
    python3 model.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay
)

FEATURES = [
    "reserves_to_gdp", "current_account_gdp", "external_debt_gdp",
    "real_interest_rate_diff", "credit_growth", "real_exchange_rate_gap",
    "fiscal_balance_gdp",
]

df = pd.read_csv("data/currency_crisis_panel.csv")
X = df[FEATURES]
y = df["crisis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---- Logistic regression (scaled features so coefficients are comparable) ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logit = LogisticRegression(max_iter=1000)
logit.fit(X_train_scaled, y_train)
logit_probs = logit.predict_proba(X_test_scaled)[:, 1]
logit_preds = logit.predict(X_test_scaled)

print("=== Logistic Regression ===")
print(classification_report(y_test, logit_preds, zero_division=0))
print(f"ROC-AUC: {roc_auc_score(y_test, logit_probs):.3f}\n")

coef_table = pd.DataFrame({
    "feature": FEATURES,
    "coefficient": logit.coef_[0].round(3),
}).sort_values("coefficient", key=abs, ascending=False)
print("Standardized coefficients (larger |value| = stronger predictor):")
print(coef_table.to_string(index=False))

# ---- Random forest ----
rf = RandomForestClassifier(n_estimators=300, max_depth=4, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:, 1]
rf_preds = rf.predict(X_test)

print("\n=== Random Forest ===")
print(classification_report(y_test, rf_preds, zero_division=0))
print(f"ROC-AUC: {roc_auc_score(y_test, rf_probs):.3f}\n")

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": rf.feature_importances_.round(3),
}).sort_values("importance", ascending=False)
print("Feature importances:")
print(importance.to_string(index=False))

# ---- Plots ----
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# ROC curves
for name, probs, color in [("Logistic Regression", logit_probs, "#2DD4BF"),
                            ("Random Forest", rf_probs, "#FFB020")]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})", color=color)
axes[0].plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
axes[0].set_title("ROC Curve")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()

# Confusion matrix (random forest)
cm = confusion_matrix(y_test, rf_preds)
ConfusionMatrixDisplay(cm, display_labels=["No crisis", "Crisis"]).plot(ax=axes[1], colorbar=False, cmap="Blues")
axes[1].set_title("Random Forest — Confusion Matrix")

# Feature importance bar chart
axes[2].barh(importance["feature"], importance["importance"], color="#8B96AB")
axes[2].set_title("Random Forest Feature Importance")
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig("model_results.png", dpi=150)
print("\nSaved model_results.png")
