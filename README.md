# Sales Forecasting

Forecasts daily sales using engineered lag/calendar features and a Gradient
Boosting model, benchmarked against a seasonal-naive baseline — the
comparison that actually proves a fancier model earns its complexity.

## Problem

Predict next month's daily sales for inventory planning, staffing, and
revenue forecasting — a bread-and-butter time series task in retail,
e-commerce, and operations analytics roles.

## Approach

1. **Synthetic but realistic data** — 3 years of daily sales with trend,
   weekly seasonality (weekend lift), yearly seasonality (holiday-season
   spike, February dip), and random promo spikes.
2. **Feature engineering** — calendar features (day of week, month, holiday
   season flag) plus lag features (1, 7, 14, 28 days back) and rolling
   means (7-day, 28-day) — the standard toolkit for turning a time series
   into a supervised learning problem.
3. **Baseline first** — a seasonal-naive forecast (today = same weekday one
   week ago) sets the bar any real model needs to clear.
4. **Gradient Boosting Regressor** — trained on the engineered features,
   evaluated against the same holdout period as the baseline.
5. **Evaluation** — MAE, RMSE, and MAPE on a true 30-day holdout (not just
   cross-validation, which can leak future information in time series).

## Files

- `generate_sample_data.py` — builds the synthetic daily sales dataset
- `model.py` — feature engineering, baseline, Gradient Boosting model,
  evaluation, saves `forecast_results.png`
- `data/daily_sales.csv` — dataset used

## Running it

```bash
pip install pandas matplotlib numpy scikit-learn
python3 generate_sample_data.py
python3 model.py
```

## Results

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Seasonal-naive baseline | 30.3 | 46.6 | 5.3% |
| Gradient Boosting | 25.4 | 34.2 | 4.4% |

The Gradient Boosting model beats the baseline by roughly 1 percentage
point of MAPE — a real, if modest, improvement. `rolling_mean_7` dominates
feature importance, which makes sense: recent momentum is the strongest
signal in most demand data. That the baseline is already fairly strong is
itself a useful finding — it's a reminder to always benchmark against a
simple method before reaching for something complex.

## Why no ARIMA/Prophet

This was built in a sandbox with no internet access, so `statsmodels` and
`prophet` weren't installable — the comparison here is baseline vs.
feature-based gradient boosting instead of classical time series models.
For a live version, adding a SARIMA or Prophet model as a third comparison
point would be the natural next step; both handle seasonality more
explicitly than the lag-feature approach used here.

## Using real data

Swap in a real dataset for a live version — Kaggle's "Store Sales - Time
Series Forecasting" (Corporación Favorita) or "Walmart Recruiting - Store
Sales Forecasting" both work with minimal changes to column names in
`model.py`.
