"""
Sales Forecasting
----------------------
Forecasts daily sales for the next 30 days using two approaches, compared
on a true holdout period:

  1. Seasonal-naive baseline (this week's value = same weekday last week)
  2. Gradient Boosting Regressor on engineered lag/calendar features

Any real forecasting project needs a baseline to prove the fancier model is
actually earning its complexity -- that comparison is the point here.

Run:
    python3 generate_sample_data.py   # only needed once
    python3 model.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

df = pd.read_csv("data/daily_sales.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

HOLDOUT_DAYS = 30

# ---- Feature engineering ----
def add_features(data):
    data = data.copy()
    data["day_of_week"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month
    data["day_of_year"] = data["date"].dt.dayofyear
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["is_holiday_season"] = data["month"].isin([11, 12]).astype(int)

    for lag in [1, 7, 14, 28]:
        data[f"lag_{lag}"] = data["sales"].shift(lag)

    for window in [7, 28]:
        data[f"rolling_mean_{window}"] = data["sales"].shift(1).rolling(window).mean()

    return data

feat_df = add_features(df).dropna().reset_index(drop=True)

FEATURES = [
    "day_of_week", "month", "day_of_year", "is_weekend", "is_holiday_season",
    "lag_1", "lag_7", "lag_14", "lag_28", "rolling_mean_7", "rolling_mean_28",
]

train = feat_df.iloc[:-HOLDOUT_DAYS]
test = feat_df.iloc[-HOLDOUT_DAYS:]

X_train, y_train = train[FEATURES], train["sales"]
X_test, y_test = test[FEATURES], test["sales"]

# ---- Baseline: seasonal-naive (same weekday, 1 week ago) ----
baseline_preds = test["lag_7"].values

# ---- Gradient Boosting model ----
model = GradientBoostingRegressor(
    n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42
)
model.fit(X_train, y_train)
gb_preds = model.predict(X_test)

# ---- Evaluation ----
def report(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    print(f"{name:28s} MAE={mae:6.1f}  RMSE={rmse:6.1f}  MAPE={mape:5.1f}%")
    return mae, rmse, mape

print(f"=== Holdout evaluation (last {HOLDOUT_DAYS} days) ===")
report("Seasonal-naive baseline", y_test, baseline_preds)
report("Gradient Boosting", y_test, gb_preds)

# ---- Feature importance ----
importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_.round(3),
}).sort_values("importance", ascending=False)
print("\nFeature importances:")
print(importance.to_string(index=False))

# ---- Plots ----
fig, axes = plt.subplots(2, 1, figsize=(13, 9))

# full history
axes[0].plot(df["date"], df["sales"], color="#8B96AB", linewidth=0.8, label="Actual sales")
axes[0].axvspan(test["date"].min(), test["date"].max(), color="#FFB020", alpha=0.15, label="Holdout period")
axes[0].set_title("Full Sales History (2021-2023)")
axes[0].legend()

# holdout zoom
axes[1].plot(test["date"], y_test.values, marker="o", color="#8B96AB", label="Actual")
axes[1].plot(test["date"], baseline_preds, marker="o", color="#FFB020", linestyle="--", label="Seasonal-naive")
axes[1].plot(test["date"], gb_preds, marker="o", color="#2DD4BF", label="Gradient Boosting")
axes[1].set_title(f"Holdout Period Forecast (last {HOLDOUT_DAYS} days)")
axes[1].legend()
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig("forecast_results.png", dpi=150)
print("\nSaved forecast_results.png")
