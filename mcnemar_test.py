"""
McNemar's test: RF selective vs persistence on jointly-covered days.

On days where both RF and coverage-matched persistence make a prediction,
tests whether the fraction of days RF-correct/persistence-wrong equals
persistence-correct/RF-wrong (H0: no difference in error rates).

Outputs: results/mcnemar_test.csv
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import chi2

df = pd.read_csv('data/features.csv', index_col=0, parse_dates=True)
RAW_COLS   = ['VIX', 'VIX3M', 'SP500', 'GOLD', 'TNY', 'DXY',
              'FEDFUNDS', 'YIELD_CURVE', 'VIX9D', 'SKEW']
LABEL_COLS = (['LABEL'] +
              [f'LABEL_{n}' for n in [5, 10, 15, 20, 25]] +
              [f'LABEL_SUSTAINED_{n}' for n in [5, 10, 15, 20, 25]])
feature_cols = [c for c in df.columns if c not in RAW_COLS + LABEL_COLS]

TAU  = 0.25
tscv = TimeSeriesSplit(n_splits=5)
rows = []

print(f"{'='*65}")
print("  McNEMAR'S TEST: RF SELECTIVE vs PERSISTENCE (jointly covered)")
print(f"{'='*65}")

for n in [5, 10, 15, 20, 25]:
    label_col = f'LABEL_{n}'
    sub = df[feature_cols + [label_col, 'VIX']].dropna(subset=[label_col])
    X   = sub[feature_cols].values
    y   = sub[label_col].astype(int).values
    vix = sub['VIX'].values
    split_idx = int(len(sub) * 0.8)

    X_tr_raw, X_te_raw = X[:split_idx], X[split_idx:]
    y_tr, y_te         = y[:split_idx], y[split_idx:]
    vix_te             = vix[split_idx:]
    vix_tr             = vix[:split_idx]

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr_raw)
    X_te   = scaler.transform(X_te_raw)

    rf  = RandomForestClassifier(n_estimators=200, max_depth=10,
                                  min_samples_leaf=5, random_state=42)
    cal = CalibratedClassifierCV(rf, cv=tscv, method='isotonic')
    cal.fit(X_tr, y_tr)
    proba  = cal.predict_proba(X_te)[:, 1]
    y_rf   = (proba >= 0.5).astype(int)

    # RF selective mask
    rf_mask = (proba > 0.5 + TAU) | (proba < 0.5 - TAU)
    n_rf    = rf_mask.sum()

    # Persistence at matched coverage — c calibrated on TRAINING window to
    # match the RF's training coverage rate, then frozen and applied to test
    proba_tr    = cal.predict_proba(X_tr)[:, 1]
    cov_rf_tr   = ((proba_tr > 0.5 + TAU) | (proba_tr < 0.5 - TAU)).mean()
    c_thresh    = np.quantile(np.abs(vix_tr - 20), 1 - cov_rf_tr)
    dist        = np.abs(vix_te - 20)
    p_mask      = dist >= c_thresh
    y_pers   = (vix_te >= 20).astype(int)

    # Jointly covered days
    joint_mask = rf_mask & p_mask
    n_joint    = joint_mask.sum()

    if n_joint < 20:
        print(f"  N={n:2d}: only {n_joint} jointly covered days — skip")
        continue

    y_true_j  = y_te[joint_mask]
    y_rf_j    = y_rf[joint_mask]
    y_pers_j  = y_pers[joint_mask]

    rf_correct   = (y_rf_j   == y_true_j)
    pers_correct = (y_pers_j == y_true_j)

    # McNemar contingency: b = RF wrong, persist right; c = RF right, persist wrong
    b = (~rf_correct &  pers_correct).sum()
    c = ( rf_correct & ~pers_correct).sum()

    # McNemar statistic (continuity-corrected for small b+c)
    if (b + c) == 0:
        chi2_stat = 0.0
        p_val = 1.0
    else:
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) >= 25 else (b - c) ** 2 / (b + c)
        p_val = chi2.sf(chi2_stat, df=1)

    rf_acc_j   = rf_correct.mean()
    pers_acc_j = pers_correct.mean()
    gap        = (rf_acc_j - pers_acc_j) * 100

    sig = p_val < 0.05
    print(f"  N={n:2d}: n_joint={n_joint:4d}  RF={rf_acc_j*100:.2f}%  "
          f"Persist={pers_acc_j*100:.2f}%  gap={gap:+.2f}pp  "
          f"b={b}  c={c}  chi2={chi2_stat:.3f}  p={p_val:.3f}  "
          f"{'SIGNIFICANT' if sig else 'not significant'}")

    rows.append(dict(N=n, N_joint=n_joint,
                     RF_Acc=round(rf_acc_j*100, 2),
                     Persist_Acc=round(pers_acc_j*100, 2),
                     Gap_pp=round(gap, 2),
                     b=b, c=c,
                     Chi2=round(chi2_stat, 3),
                     P_value=round(p_val, 3),
                     Significant=sig))

results = pd.DataFrame(rows)
results.to_csv('results/mcnemar_test.csv', index=False)
print("\nSaved → results/mcnemar_test.csv")
print("\n=== Summary ===")
print(results.to_string(index=False))
