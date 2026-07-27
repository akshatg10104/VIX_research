# Session 1 Summary — VIX Regime Classification
**Date:** May 11, 2026  
**Researcher:** Akshat Gupta, TAMS — under Professor Jianguo Liu, UNT

---

## What We Built

### 1. Data Pipeline (`get_data.py`)
Downloaded 6 raw price series from Yahoo Finance covering **1990–present**, saved to `data/raw_data.csv`:

| Ticker | Series | Source |
|---|---|---|
| ^VIX | CBOE Volatility Index | Yahoo Finance |
| ^VIX3M | 3-Month VIX | Yahoo Finance |
| ^GSPC | S&P 500 | Yahoo Finance |
| GC=F | Gold Futures | Yahoo Finance |
| ^TNX | 10Y Treasury Yield | Yahoo Finance |
| DX-Y.NYB | US Dollar Index | Yahoo Finance |

**Data limitation:** VIX3M only goes back to ~2006, so the effective dataset starts July 2006. This means the dot-com crash (2000–2002) and 9/11 volatility spike are excluded — worth noting as a limitation in the paper.

---

### 2. Feature Engineering (`build_features.py`)
Built **32 features** across 4 categories, saved to `data/features.csv` (4,985 rows after warm-up period).

**Label:**  
`LABEL = 1` if VIX ≥ 20 (high volatility / panic regime)  
`LABEL = 0` if VIX < 20 (low volatility / calm regime)  
Class split: 3,219 low-vol days (65%) vs 1,766 high-vol days (35%)

**Feature categories:**

| Category | Features |
|---|---|
| VIX Derived | SMA10, SMA20, MOM10, MOM20, ROC10, ROC20, BB_MID, BB_STD, BB_UPPER, BB_LOWER, BB_POSITION, RSI14, MA_CROSS, STDDEV |
| Term Structure | VIX_TERM_SPREAD (VIX3M − VIX) |
| Equity Market | SP500 returns (1/5/20d), MOM10, MOM20, realized vol (20d), drawdown (252d rolling) |
| Cross-Asset | Gold returns (1/5/20d), Treasury yield changes (1/5/20d), DXY returns (1/5/20d) |

---

### 3. Model Training (`train_models.py`)
Chronological 80/20 train-test split (no shuffling, respects time order):
- Training: 3,988 days (July 2006 – ~mid 2022)
- Testing: 997 days (~mid 2022 – May 2026)

**Results:**

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| XGBoost (HistGBT) | **98.90%** | **98.10%** | **98.41%** |
| Logistic Regression | 97.49% | 95.58% | 96.50% |
| Random Forest | 97.09% | 96.12% | 94.59% |
| Decision Tree | 94.68% | 90.91% | 92.36% |
| SVM (poly, degree=4, C=0.001) | 70.31% | 95.00% | 6.05% |

Plots saved: `results/model_comparison.png`, `results/feature_importance.png`

**Top 10 Features (Random Forest importance):**

| Rank | Feature | Importance |
|---|---|---|
| 1 | VIX_SMA10 | 0.232 |
| 2 | VIX_BB_MID | 0.137 |
| 3 | VIX_BB_UPPER | 0.119 |
| 4 | VIX_SMA20 | 0.110 |
| 5 | VIX_BB_LOWER | 0.077 |
| 6 | SP500_RVOL | 0.034 |
| 7 | VIX_BB_POSITION | 0.032 |
| 8 | SP500_DRAWDOWN | 0.031 |
| 9 | VIX_STDDEV | 0.026 |
| 10 | VIX_MOM20 | 0.024 |

---

## Key Findings

1. **Strong baseline performance.** All models except SVM classify the current VIX regime with high accuracy. XGBoost leads at 98.9%.

2. **Logistic Regression is nearly as good as XGBoost.** LR at 97.5% suggests the regime signal is largely linear — VIX's own moving averages are so predictive of today's regime that even a simple model almost perfectly separates the classes.

3. **SVM is broken at C=0.001.** Recall of 6% means the model predicts "low volatility" almost every day. The hyperparameters from Liu & Jiang (2020) were tuned for NASDAQ direction prediction — they do not transfer directly to VIX regime classification. This needs to be addressed before the paper.

4. **Cross-asset features contribute minimally.** Gold, treasury, and DXY features rank far below VIX-derived features in importance. This is a meaningful research finding: VIX's own short-term memory dominates. Cross-asset features may become more important after the label shift (see below).

5. **The current setup has lookahead bias.** The label is based on today's VIX value, and the features are also from today — the model is classifying the present, not predicting the future. This needs to be corrected before results are meaningful for the paper.

---

## Critical Next Step: Fix Lookahead Bias

The current label `LABEL = (VIX >= 20)` classifies the regime *on the same day as the features*. This is not a prediction — it is classification of the present using present data, which inflates all accuracy scores.

**Fix:** Shift the label forward by N days.

```python
# Predict whether VIX will be >= 20, N days from now
N = 5  # or 10, 20 — to be experimented with
df['LABEL'] = (df['VIX'].shift(-N) >= 20).astype(int)
df = df.dropna(subset=['LABEL'])  # drops last N rows
```

This means: "given today's features, will the market be in a high-volatility regime N days from now?" This is the actual research question — and directly mirrors the N-days-ahead framing of Liu & Jiang (2020).

Expected impact: accuracy will drop significantly (especially for LR), which is the honest result. The selective predicting methodology then becomes meaningful — it identifies the subset of days where the model is confident enough to make a prediction.

---

## Remaining Steps (per research timeline)

- [ ] Fix lookahead bias — shift label N days forward, retrain all models
- [ ] Tune SVM C parameter appropriately for this problem
- [ ] Implement selective predicting (confidence threshold filtering)
- [ ] Compare selective vs. non-selective accuracy across all models
- [ ] Add market breadth features (put/call ratio) — FRED or CBOE
- [ ] Add macro features (fed funds rate, yield curve slope) — FRED
- [ ] Write up results and draft paper sections
