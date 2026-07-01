"""
Generates a synthetic daily sales time series with trend, weekly seasonality,
yearly seasonality, and holiday spikes -- the standard structure of real
retail demand data.

NOTE: Synthetic, built to reflect realistic time series structure. Swap in a
real dataset (Kaggle "Store Sales - Time Series Forecasting", Walmart Sales
Forecasting) for a live version -- see README.
"""

import numpy as np
import pandas as pd

np.random.seed(0)

start = pd.Timestamp("2021-01-01")
n_days = 3 * 365
dates = pd.date_range(start, periods=n_days, freq="D")

t = np.arange(n_days)

# trend: gradual growth
trend = 200 + 0.15 * t

# weekly seasonality: weekends higher
day_of_week = dates.dayofweek
weekly = np.where(day_of_week >= 5, 60, 0) + np.where(day_of_week == 4, 25, 0)

# yearly seasonality: higher in Nov-Dec (holiday shopping), lower in Feb
day_of_year = dates.dayofyear
yearly = 80 * np.sin(2 * np.pi * (day_of_year - 80) / 365) * -1
yearly += np.where(dates.month.isin([11, 12]), 90, 0)
yearly += np.where(dates.month == 2, -30, 0)

# random noise
noise = np.random.normal(0, 25, n_days)

# occasional promo spikes (5% of days)
promo = np.random.binomial(1, 0.05, n_days) * np.random.uniform(50, 150, n_days)

sales = np.clip(np.asarray(trend + weekly + yearly + noise + promo), 0, None).round(1)

df = pd.DataFrame({"date": dates, "sales": sales})
df.to_csv("data/daily_sales.csv", index=False)
print(f"Wrote data/daily_sales.csv: {len(df)} rows, {dates.min().date()} to {dates.max().date()}")
print(df.head())
print(df["sales"].describe())
