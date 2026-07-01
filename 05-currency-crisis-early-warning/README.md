# Currency Crisis Early-Warning Model

A machine learning early-warning system for currency crises, built around
the macroeconomic indicators from the empirical crisis literature — and
tied directly to the exchange rate crisis models (Krugman first-generation,
Obstfeld second-generation) and 2007–2011 financial crisis material from my
International Economics course.

## The economics behind it

Two broad families of theory motivate the indicators used here:

- **First-generation models (Krugman, 1979)** — a crisis is triggered when
  persistent fiscal deficits deplete FX reserves defending a peg, until
  reserves hit a critical floor and speculative attack forces devaluation.
  → captured by `reserves_to_gdp`, `fiscal_balance_gdp`, `current_account_gdp`.
- **Second-generation models (Obstfeld, 1994)** — crises can be
  self-fulfilling: if markets expect a government to abandon a peg (because
  defending it is costly), the expectation itself triggers the attack, even
  without unsustainable fundamentals. Overvaluation and interest rate
  defense are key signals here.
  → captured by `real_exchange_rate_gap`, `real_interest_rate_diff`.
- **Credit-boom / "this time is different" literature** — rapid credit
  growth is one of the most consistently reliable crisis predictors across
  empirical studies (Kaminsky & Reinhart, 1999; Reinhart & Rogoff, 2009).
  → captured by `credit_growth`.

## Approach

- Built a synthetic panel of 16 countries × 23 years (2000–2022) with the
  seven indicators above, and a binary crisis label generated from a noisy
  function of those indicators (~15% crisis rate — deliberately close to
  real-world crisis frequency, not artificially balanced).
- Trained two classifiers:
  - **Logistic regression** — coefficients map directly back to economic
    theory (which indicator matters, and in which direction).
  - **Random forest** — usually stronger predictive performance, at the
    cost of interpretability.
- Evaluated both with ROC-AUC, precision/recall, and confusion matrices.

## Files

- `generate_sample_data.py` — builds the synthetic panel dataset
- `model.py` — trains both models, prints metrics, saves `model_results.png`
- `data/currency_crisis_panel.csv` — the dataset used

## Running it

```bash
pip install pandas matplotlib numpy scikit-learn
python3 generate_sample_data.py
python3 model.py
```

## Results (on this synthetic data)

- Logistic regression: ROC-AUC ≈ 0.84. Credit growth and real exchange rate
  overvaluation are the strongest predictors — consistent with the
  literature above. At the default 0.5 probability threshold it fails to
  flag any crisis (recall = 0), because crises are rare and the model
  plays it safe — a textbook class-imbalance problem. Lowering the
  decision threshold (e.g. to 0.25–0.3) trades some false positives for
  much better crisis recall, which is usually the right tradeoff for an
  early-warning system: missing a real crisis is costlier than a false alarm.
- Random forest: ROC-AUC ≈ 0.78, weaker overall but catches slightly more
  crises at default threshold. Feature importances agree with the logistic
  regression's ranking — credit growth and overvaluation dominate.

## Limitations & next steps

- **Synthetic data.** Built to mirror realistic indicator relationships,
  not actual crisis dates. To make this genuinely live:
  - Crisis dates: Reinhart & Rogoff's "This Time Is Different" crisis
    chronology, or the IMF's official crisis database
  - Indicators: IMF International Financial Statistics (IFS), World Bank
    World Development Indicators (WDI)
- **Threshold tuning** and **precision-recall tradeoff analysis** would be
  the natural next addition — plot precision/recall vs. threshold and pick
  one aligned with how costly a missed crisis vs. a false alarm actually is.
- Panel structure (country + year) means observations aren't fully
  independent; a fixed-effects logistic model or panel-aware
  cross-validation would be a more rigorous next step.
