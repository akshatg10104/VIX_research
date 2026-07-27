# VIX Regime Classification — Research Summary

**Akshat Gupta | TAMS | Advisor: Prof. Jianguo Liu (UNT)**  
**Date: May 2026**

---

## Overview

This project applies the selective predicting methodology from Liu & Jiang (2020) to VIX volatility regime classification. The goal is to predict whether the VIX index will be in a high-volatility regime (≥ 20) N trading days ahead, and to demonstrate that confidence-based selective predicting consistently improves accuracy over baseline models.

---

## Data

- **Source:** Yahoo Finance (daily, 1990–present), FRED (Fed Funds Rate, yield curve)
- **Series:** VIX, VIX3M, S&P 500, Gold, 10-Year Treasury Yield, DXY, SKEW Index
- **Features:** 36 engineered features across VIX technicals, cross-asset signals, and macro indicators
- **Label:** Binary — VIX ≥ 20 (high-vol = 1) or VIX < 20 (low-vol = 0) at day N ahead
- **Train/Test split:** Chronological 80/20 (no shuffling — respects time series structure)
- **Dataset:** ~5,000 trading days after feature engineering

---

## Models

| Model | Description |
|---|---|
| Random Forest | 300–500 trees, hyperparameter-tuned via RandomizedSearchCV + TimeSeriesSplit |
| XGBoost | HistGradientBoostingClassifier (sklearn), tuned analogously |

Both models use **isotonic calibration** (CalibratedClassifierCV with 5-fold TimeSeriesSplit CV) to produce well-calibrated probability estimates for the selective predicting filter.

---

## Selective Predicting (Liu & Jiang 2020 Adaptation)

A prediction is only made when the model's confidence exceeds a threshold:

> **Predict only if P(high-vol) > 0.5 + t  OR  P(high-vol) < 0.5 − t**

Days where the model is uncertain (near 0.5) are abstained from. This trades coverage for accuracy.

---

## Main Results (VIX ≥ 20, t = 0.25)

### Accuracy

| N (days ahead) | RF Baseline | RF Selective | Gain | Coverage |
|---|---|---|---|---|
| 5  | 83.8% | 91.1% | **+7.3%** | 78.4% of days |
| 10 | 78.6% | 88.8% | **+10.2%** | 70.5% of days |
| 15 | 74.9% | 90.0% | **+15.1%** | 58.1% of days |
| 20 | 72.6% | 90.9% | **+18.3%** | 50.6% of days |
| 25 | 69.7% | 91.1% | **+21.5%** | 32.9% of days |

| N (days ahead) | XGB Baseline | XGB Selective | Gain | Coverage |
|---|---|---|---|---|
| 5  | 82.9% | 90.7% | **+7.8%** | 79.4% of days |
| 10 | 79.0% | 91.0% | **+12.0%** | 61.1% of days |
| 15 | 73.5% | 87.7% | **+14.2%** | 54.6% of days |
| 20 | 74.4% | 92.1% | **+17.7%** | 35.4% of days |
| 25 | 66.5% | 92.5% | **+26.0%** | 24.1% of days |

### Key finding
Confidence-based selective predicting yields consistent, large accuracy gains across all prediction horizons. Gains increase with N — at 25 days ahead, the XGBoost model improves from 66.5% to 92.5%.

---

## Threshold Sensitivity

Accuracy monotonically increases with confidence threshold t. Higher t = fewer predictions but more accurate:

| t | RF N=5 SelAcc | Coverage | RF N=10 SelAcc | Coverage |
|---|---|---|---|---|
| 0.10 | 86.5% | 90.9% | 83.2% | 90.6% |
| 0.25 | 91.1% | 78.4% | 88.8% | 70.5% |
| 0.35 | 95.1% | 67.2% | 95.1% | 42.5% |
| 0.40 | 97.6% | 49.0% | 97.4% | 15.4% |

---

## Robustness Check — Multiple VIX Thresholds

Selective predicting gains are positive across VIX thresholds of 18, 20, and 22 (all N values):

**Random Forest — Accuracy Gain (Selective vs Baseline)**

|  | VIX ≥ 18 | VIX ≥ 20 | VIX ≥ 22 |
|---|---|---|---|
| N=5  | +7.6% | +9.0% | +8.9% |
| N=10 | +11.1% | +8.9% | +10.7% |
| N=15 | +16.6% | +10.8% | +11.4% |
| N=20 | +15.2% | +15.1% | +13.1% |
| N=25 | +25.9% | +8.9% | +9.1% |

---

## Sustained Regime Labels

A harder variant: predict whether VIX ≥ 20 for **3 consecutive days** starting at day N. Models perform *better* on this task — sustained regimes have stronger precursors.

| N | RF Sustained Selective | RF Instantaneous Selective |
|---|---|---|
| 5  | 94.0% | 91.1% |
| 10 | 93.8% | 88.8% |
| 15 | 93.8% | 90.0% |
| 20 | 94.3% | 90.9% |
| 25 | 89.9% | 91.1% |

---

## Walk-Forward Validation (5 Expanding Windows)

Confirms generalization across time periods:

| Model | N | Mean Base (±std) | Mean Selective (±std) | Δ |
|---|---|---|---|---|
| RF | 5 | 82.8% ± 9.7% | **92.2% ± 1.7%** | +9.4% |
| XGB | 5 | 84.0% ± 8.0% | 84.6% ± 9.7% | +0.6% |
| RF | 10 | 72.3% ± 22.7% | **85.9% ± 7.7%** | +13.7% |

RF selective predicting is notably more *consistent* across time periods (lower std) than the baseline.

---

## Feature Importance (Random Forest)

Top features by mean decrease in impurity, averaged across N = 5, 10, 20:

| Rank | Feature | Avg Importance | Category |
|---|---|---|---|
| 1 | VIX_SMA10 | 19.8% | VIX trend |
| 2 | VIX_SMA20 | 15.3% | VIX trend |
| 3 | VIX_BB_UPPER | 10.9% | VIX Bollinger |
| 4 | VIX_BB_LOWER | 10.8% | VIX Bollinger |
| 5 | SP500_DRAWDOWN | 6.8% | Equity risk |
| 6 | SP500_RVOL | 5.7% | Equity realized vol |
| 7 | FEDFUNDS_CHANGE3M | 2.2% | Macro |
| 8 | VIX_BB_STD | 2.2% | VIX vol-of-vol |

**Insight:** VIX's own trend/level (SMA10, SMA20, Bollinger bands) accounts for ~57% of predictive power. Equity drawdown and realized volatility add ~12%. Macro features grow more important at longer horizons (FEDFUNDS_CHANGE3M rises from rank 9 at N=5 to rank 7 at N=20).

---

## Files

| Script | Purpose |
|---|---|
| `get_data.py` | Download market data (Yahoo Finance) |
| `get_macro_data.py` | Download macro data (FRED API) |
| `build_features.py` | Engineer 36 features + 11 label columns |
| `tune_and_train.py` | Main results: calibrated RF + XGB, hyperparameter search |
| `robustness_check.py` | VIX threshold sensitivity (18, 20, 22) |
| `confusion_matrices.py` | Confusion matrix visualization |
| `optimize_threshold.py` | Confidence threshold sweep (t = 0.05 to 0.40) |
| `sustained_labels.py` | Sustained regime label evaluation |
| `walk_forward.py` | 5-fold expanding walk-forward validation |
| `feature_importance.py` | RF feature importance by prediction horizon |
