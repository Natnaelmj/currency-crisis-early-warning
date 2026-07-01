"""
Generates a synthetic country-year panel dataset for the currency crisis
early-warning model.

The features are the classic Kaminsky-Reinhart-style "early warning
indicators" (EWIs) used in the empirical currency-crisis literature:

  - reserves_to_gdp          : FX reserves / GDP (low reserves = vulnerable)
  - current_account_gdp      : current account balance / GDP (deficits = vulnerable)
  - external_debt_gdp        : external debt / GDP (high debt = vulnerable)
  - real_interest_rate_diff  : domestic - foreign real interest rate
                                (large diff can signal capital flight risk / defense of peg)
  - credit_growth            : domestic credit growth (credit booms precede crises --
                                classic "this time is different" pattern)
  - real_exchange_rate_gap   : % overvaluation vs. long-run equilibrium
                                (overvaluation is a first-generation-model trigger)
  - fiscal_balance_gdp       : government balance / GDP

A crisis is generated as a noisy logistic function of these indicators, so
the "ground truth" relationship is realistic but not perfectly learnable --
same as real-world data.

NOTE: This is illustrative synthetic data built to reflect the *shape* of
real early-warning-indicator relationships, not actual IMF/World Bank data.
Swap in the real thing (IMF IFS, World Bank WDI, Reinhart-Rogoff crisis
dates) for a genuinely live version -- see README for pointers.
"""

import numpy as np
import pandas as pd

np.random.seed(11)

countries = [f"Country {c}" for c in "ABCDEFGHIJKLMNOP"]
years = list(range(2000, 2023))

rows = []
for country in countries:
    # country-level "structural" tendency (some countries are just more fragile)
    fragility = np.random.normal(0, 1)

    reserves = np.random.uniform(15, 35)
    debt = np.random.uniform(20, 60)
    ca = np.random.uniform(-6, 2)
    fiscal = np.random.uniform(-5, 2)
    credit = np.random.uniform(5, 15)
    rer_gap = np.random.uniform(-5, 5)
    rate_diff = np.random.uniform(-1, 4)

    for year in years:
        # indicators evolve as a random walk with mean reversion
        reserves += np.random.normal(0, 2) - 0.05 * (reserves - 25)
        debt += np.random.normal(0, 3) - 0.03 * (debt - 40)
        ca += np.random.normal(0, 0.8) - 0.1 * (ca + 2)
        fiscal += np.random.normal(0, 0.7) - 0.1 * (fiscal + 1)
        credit += np.random.normal(0, 2.5) - 0.05 * (credit - 8)
        rer_gap += np.random.normal(0, 1.5) - 0.08 * rer_gap
        rate_diff += np.random.normal(0, 0.6) - 0.1 * (rate_diff - 1)

        reserves = max(reserves, 2)
        debt = max(debt, 5)

        # crisis probability: worse fundamentals -> higher probability
        # (weights loosely reflect literature emphasis on reserves, debt, overvaluation)
        z = (
            -0.12 * reserves
            + 0.05 * debt
            - 0.35 * ca
            + 0.30 * credit
            + 0.25 * rer_gap
            + 0.20 * rate_diff
            - 0.15 * fiscal
            + 0.8 * fragility
        )
        # scaled + shifted (calibrated against the z-distribution) so the
        # baseline crisis rate lands around 8-10%, consistent with real-world
        # crisis frequency in the EWI literature
        prob = 1 / (1 + np.exp(-(z - 8.0) / 2))
        crisis = np.random.binomial(1, min(prob, 0.9))

        rows.append({
            "country": country,
            "year": year,
            "reserves_to_gdp": round(reserves, 2),
            "current_account_gdp": round(ca, 2),
            "external_debt_gdp": round(debt, 2),
            "real_interest_rate_diff": round(rate_diff, 2),
            "credit_growth": round(credit, 2),
            "real_exchange_rate_gap": round(rer_gap, 2),
            "fiscal_balance_gdp": round(fiscal, 2),
            "crisis": crisis,
        })

df = pd.DataFrame(rows)
df.to_csv("data/currency_crisis_panel.csv", index=False)
print(f"Wrote data/currency_crisis_panel.csv: {len(df)} rows, "
      f"{df['crisis'].sum()} crisis-years ({df['crisis'].mean()*100:.1f}%)")
print(df.head())
