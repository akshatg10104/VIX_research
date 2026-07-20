# Selective Prediction and the Persistence Illusion: A Diagnostic Decomposition of VIX Regime Classification

Code and results for the paper (under review at the *Journal of Risk and Financial
Management*, manuscript jrfm-4438773).

**Authors:** Akshat Gupta (Texas Academy of Mathematics and Science, University of North
Texas) and Jianguo Liu (Department of Mathematics, University of North Texas).

The paper develops a six-component diagnostic protocol for evaluating confidence-based
selective prediction on serially correlated labels, and demonstrates it on VIX regime
classification (5,000 trading days, July 2006–May 2026) with a replication on S&P 500
trend regimes.

## Requirements

Python 3.9+ with: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `torch` (LSTM only),
`shap`, `matplotlib`.

```
pip install pandas numpy scikit-learn xgboost torch shap matplotlib
```

## Reproducing the paper

Run from the repository root. Steps 1–2 build the dataset; the analysis scripts are then
independent of one another.

1. `get_data.py`, `get_macro_data.py` — download raw market data (Yahoo Finance) and macro
   series (FRED) → `data/raw_data.csv`
2. `build_features.py` — construct the 36 engineered features and all horizon labels →
   `data/features.csv`

| Script | Paper section / output |
|---|---|
| `confidence_selective_predicting.py` | Main selective accuracy results (§5.1) |
| `tune_and_train.py`, `train_models.py` | Tuned model comparison |
| `persistence_baseline.py` | Coverage-matched baselines, moving-block bootstrap CIs incl. block-length sensitivity, transition-conditional analysis (§5.2, §5.5, §5.9, Appendix D) |
| `har_baseline.py` | Extended baseline table (persistence, LR, HAR-LR, Markov) |
| `mcnemar_test.py` | McNemar test on jointly covered days (§5.9) |
| `revision_experiments.py` | XGBoost benchmark, HAR forecast-then-threshold, risk–coverage curves + AURC, joint covered-set composition, AUC/Brier pre/post calibration, calm→high episode clustering, per-horizon sample sizes (§5.3, §5.10, §5.11, Appendices C and E) |
| `persistence_quantify.py` | Empirical transition probabilities and same-regime decomposition (§5.4) |
| `transition_reweight.py` | Cost-weighting experiment (§5.5) |
| `abstention_analysis.py`, `abstention_confound.py` | Abstention leading-indicator analysis and proximity confound test (§5.5) |
| `walk_forward.py` | 5-fold expanding walk-forward validation (§5.6) |
| `lstm_model.py` | LSTM comparison (§5.7) |
| `shap_analysis.py` | SHAP feature attribution (§5.8) |
| `cost_utility.py` | Cost-weighted utility analysis (§5.12) |
| `exclusion_2022.py` | 2022 bear-market exclusion robustness (§5.9) |
| `optimize_threshold.py`, `robustness_check.py`, `sustained_labels.py` | Threshold sensitivity, VIX-threshold robustness (18/20/22), sustained labels (Appendix B) |
| `sp500_replication.py` | S&P 500 trend-regime replication (§6) |
| `event_case_studies.py` | Event case-study figure (§5.5) |

Numerical outputs are written to `results/` (CSV) and figures to `vix paper/` (PNG).

## Leakage-free baseline construction

All coverage-matched baseline abstention thresholds (the persistence cutoff `c`, the
probability-gap thresholds for the LR/HAR-LR/Markov baselines, and the HAR forecast band
`d`) are calibrated on the **training window only** — as the training-window quantile that
reproduces the Random Forest's training coverage rate at τ = 0.25 — and then frozen before
any test-set evaluation. No test-set information (labels or inputs) enters baseline
construction, model tuning, scaler fitting, or calibration.

## License

MIT — see `LICENSE`.
