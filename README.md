# UK Electricity Demand Forecasting with Machine Learning

Forecasting half-hourly electricity demand using scikit-learn — with feature engineering, model comparison, and error analysis targeted at grid management applications.

## Background

Accurate demand forecasting is critical to grid balancing, renewable dispatch, and net zero planning. This project builds an end-to-end ML pipeline for half-hourly UK electricity demand, structured around the kind of problem National Grid ESO and energy companies solve daily.

Built by [Janusshan Sivanesakanthan](https://github.com/janusshan24) — a data scientist with domain experience at Shell and Ford Motors, trained in ML through the IBM Data Science Professional Certificate.

## What's in this project

- **Synthetic UK-style demand data** — half-hourly, 2 years, with realistic intraday/seasonal/temperature patterns
- **Feature engineering** — calendar features, cyclical encodings, lag features (24h, 1-week), rolling statistics
- **Three models compared** — Linear Regression (baseline), Random Forest, Gradient Boosting
- **Full evaluation** — RMSE, MAE, R², MAPE, actual vs predicted plots, residual analysis, feature importance
- **Error analysis by hour and month** — revealing when and why models struggle

## Results Summary

| Model | RMSE (MW) | R² | MAPE |
|-------|-----------|-----|------|
| Linear Regression | ~1,100 | ~0.93 | ~3.9% |
| Random Forest | ~720 | ~0.97 | ~2.4% |
| **Gradient Boosting** | **~620** | **~0.98** | **~2.1%** |

Gradient Boosting reduces RMSE by ~45% vs the linear baseline. Lag features and temperature are the most predictive inputs.

## Setup

```bash
git clone https://github.com/janusshan24/uk-energy-demand-forecasting
cd uk-energy-demand-forecasting
pip install -r requirements.txt
jupyter notebook energy_demand_forecasting.ipynb
```

## Requirements

See `requirements.txt`. Python 3.8+ recommended.

## Data

Synthetic data is generated in-notebook using realistic UK demand parameters. To extend with real data, the [National Grid ESO Data Portal](https://data.nationalgrideso.com/) provides free half-hourly demand via public API.

## Next steps

- Integrate live National Grid ESO API data
- Add weather forecast features (wind speed, cloud cover)
- Explore LSTM / Transformer architectures for longer-horizon forecasting
- Probabilistic forecasting (prediction intervals) for grid balancing
