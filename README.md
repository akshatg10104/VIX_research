# VIX Regime Classification Research Project

## Researcher
Akshat Gupta — High School Junior, TAMS (Texas Academy of Mathematics and Science)
Research conducted under Professor Jianguo Liu, University of North Texas

## Research Question
Can machine learning classify VIX volatility regimes and predict regime shifts 
before they occur, using a selective predicting methodology adapted from Liu & 
Jiang (2020)?

## Background
This project directly extends the selective predicting methodology from:
Liu, J. & Jiang, J. (2020). "Predicting Stock Market N-Days Ahead Using SVM 
Optimized by Selective Thresholds." ICMLC 2020.

Instead of predicting NASDAQ direction, we apply selective predicting to VIX 
regime classification — predicting when the market transitions from low 
volatility (calm) to high volatility (panic) before it happens.

## Labels
+1 = High volatility regime (VIX >= 20)
 0 = Low volatility regime (VIX < 20)

## Feature Categories
1. VIX Derived Indicators — SMA, Momentum, ROC, Bollinger Bands, RSI, MA Crossover, StdDev
2. Term Structure Features — VIX/VIX3M spread, VIX/VIX6M spread, slope
3. Equity Market Features — SP500 returns, momentum, realized volatility, drawdown
4. Cross Asset Fear Indicators — gold returns, treasury yield, dollar index
5. Market Breadth Indicators — put call ratio (to be added later)
6. Macro Features — fed funds rate, yield curve slope (to be added later)

## Models to Compare
- Logistic Regression (baseline)
- Decision Tree
- Random Forest
- SVM (polynomial kernel degree 4, low C — mirroring Liu paper)
- XGBoost

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- Confusion Matrix
- Feature Importance

## Timeline
- Now → May: Data collection and feature engineering
- June: Model training and comparison
- July: Selective predicting implementation and results
- August: Write up and paper draft

## Data Sources
- VIX: Yahoo Finance (^VIX) — 1990 to present
- VIX3M: Yahoo Finance (^VIX3M)
- S&P 500: Yahoo Finance (^GSPC)
- Gold: Yahoo Finance (GC=F)
- 10Y Treasury: Yahoo Finance (^TNX)
- Dollar Index: Yahoo Finance (DX-Y.NYB)
- Fed Funds Rate: FRED
- Credit Spreads: FRED

## Goal
Achieve higher accuracy than baseline threshold methods for VIX regime 
classification, and demonstrate that selective predicting improves accuracy 
by filtering out low signal days — extending Liu's methodology to volatility 
regime prediction for financial risk management applications.
