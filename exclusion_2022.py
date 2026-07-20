"""
2022 exclusion robustness check.

Reruns Table 6 (baseline comparison) and Table 7 (transition matrix)
on the test set excluding 2022 trading days, to verify that conclusions
do not depend on the 2022 bear-market high-VIX cluster.

Outputs: results/exclusion_2022_baselines.csv
         results/exclusion_2022_transitions.csv
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.base import clone

df = pd.read_csv('data/features.csv', index_col=0, parse_dates=True)
RAW_COLS   = ['VIX', 'VIX3M', 'SP500', 'GOLD', 'TNY', 'DXY',
              'FEDFUNDS', 'YIELD_CURVE', 'VIX9D', 'SKEW']
LABEL_COLS = (['LABEL'] +
              [f'LABEL_{n}' for n in [5, 10, 15, 20, 25]] +
              [f'LABEL_SUSTAINED_{n}' for n in [5, 10, 15, 20, 25]])
feature_cols = [c for c in df.columns if c not in RAW_COLS + LABEL_COLS]

TAU  = 0.25
tscv = TimeSeriesSplit(n_splits=5)
base_rows  = []
trans_rows = []

print(f"{'='*70}")
print("  2022 EXCLUSION ROBUSTNESS CHECK")
print(f"{'='*70}")

for n in [5, 10, 15, 20, 25]:
    label_col = f'LABEL_{n}'
    sub = df[feature_cols + [label_col, 'VIX']].dropna(subset=[label_col])

    split_idx = int(len(sub) * 0.8)
    sub_tr = sub.iloc[:split_idx]
    sub_te = sub.iloc[split_idx:]

    # Full test set and 2022-excluded test set
    sub_te_ex2022 = sub_te[sub_te.index.year != 2022]
    n_removed = len(sub_te) - len(sub_te_ex2022)
    print(f"\n  N={n}: test={len(sub_te)}, after 2022 exclusion={len(sub_te_ex2022)} "
          f"(removed {n_removed} days)")

    if len(sub_te_ex2022) < 50:
        print("    Too few days after exclusion — skip")
        continue

    X_tr_raw = sub_tr[feature_cols].values
    y_tr     = sub_tr[label_col].astype(int).values
    vix_tr   = sub_tr['VIX'].values

    X_te_raw = sub_te_ex2022[feature_cols].values
    y_te     = sub_te_ex2022[label_col].astype(int).values
    vix_te   = sub_te_ex2022['VIX'].values

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr_raw)
    X_te   = scaler.transform(X_te_raw)

    # ── RF ────────────────────────────────────────────────────────────────────
    rf  = RandomForestClassifier(n_estimators=200, max_depth=10,
                                  min_samples_leaf=5, random_state=42)
    cal = CalibratedClassifierCV(rf, cv=tscv, method='isotonic')
    cal.fit(X_tr, y_tr)
    proba  = cal.predict_proba(X_te)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    sel_mask = (proba > 0.5 + TAU) | (proba < 0.5 - TAU)
    n_sel    = sel_mask.sum()
    rf_sel   = accuracy_score(y_te[sel_mask], y_pred[sel_mask]) * 100 if n_sel >= 10 else np.nan

    # RF training coverage rate — baseline coverage matching is calibrated on
    # training data only, then frozen
    proba_tr_rf = cal.predict_proba(X_tr)[:, 1]
    cov_rf_tr   = ((proba_tr_rf > 0.5 + TAU) | (proba_tr_rf < 0.5 - TAU)).mean()

    # ── HAR-LR ────────────────────────────────────────────────────────────────
    vix_all  = np.concatenate([vix_tr, vix_te])
    vix_w    = pd.Series(vix_all).rolling(5,  min_periods=1).mean().values
    vix_m    = pd.Series(vix_all).rolling(20, min_periods=1).mean().values
    X_har_tr = np.column_stack([vix_tr, vix_w[:len(vix_tr)], vix_m[:len(vix_tr)]])
    X_har_te = np.column_stack([vix_te, vix_w[len(vix_tr):], vix_m[len(vix_tr):]])
    sc_har   = StandardScaler()
    X_har_tr_s = sc_har.fit_transform(X_har_tr)
    X_har_te_s = sc_har.transform(X_har_te)
    lr_har  = LogisticRegression(max_iter=1000, random_state=42)
    cal_har = CalibratedClassifierCV(clone(lr_har), cv=tscv, method='isotonic')
    cal_har.fit(X_har_tr_s, y_tr)
    proba_har  = cal_har.predict_proba(X_har_te_s)[:, 1]
    y_pred_har = (proba_har >= 0.5).astype(int)

    # Match HAR coverage to RF — threshold from TRAINING probabilities, frozen
    proba_har_tr = cal_har.predict_proba(X_har_tr_s)[:, 1]
    thresh_har   = np.quantile(np.abs(proba_har_tr - 0.5), 1 - cov_rf_tr)
    mask_har     = np.abs(proba_har - 0.5) >= thresh_har
    har_acc    = accuracy_score(y_te[mask_har], y_pred_har[mask_har]) * 100 if mask_har.sum() >= 10 else np.nan

    # ── Persistence (c calibrated on training window, frozen) ─────────────────
    y_pers = (vix_te >= 20).astype(int)
    dist   = np.abs(vix_te - 20)
    c_thresh = np.quantile(np.abs(vix_tr - 20), 1 - cov_rf_tr)
    p_mask   = dist >= c_thresh
    pers_acc = accuracy_score(y_te[p_mask], y_pers[p_mask]) * 100 if p_mask.sum() >= 10 else np.nan

    rf_minus_persist = (rf_sel - pers_acc) if not (np.isnan(rf_sel) or np.isnan(pers_acc)) else np.nan
    har_minus_rf     = (har_acc - rf_sel)   if not (np.isnan(har_acc) or np.isnan(rf_sel))  else np.nan

    print(f"    Persist={pers_acc:.1f}%  HAR={har_acc:.1f}%  RF={rf_sel:.1f}%  "
          f"RF-Persist={rf_minus_persist:+.2f}pp  HAR-RF={har_minus_rf:+.2f}pp")

    base_rows.append(dict(N=n, Coverage=round(sel_mask.mean()*100, 1),
                          Persistence=round(pers_acc, 1),
                          HAR_LR=round(har_acc, 1),
                          RF_Sel=round(rf_sel, 1),
                          RF_minus_Persist=round(rf_minus_persist, 2),
                          HAR_minus_RF=round(har_minus_rf, 2),
                          N_test_ex2022=len(y_te)))

    # ── Transition matrix ─────────────────────────────────────────────────────
    calm_mask = (vix_te < 20)
    high_mask = ~calm_mask

    for trans_name, t_mask, true_val in [
        ('calm_calm',  calm_mask & (y_te == 0), 0),
        ('calm_high',  calm_mask & (y_te == 1), 1),
        ('high_calm',  high_mask & (y_te == 0), 0),
        ('high_high',  high_mask & (y_te == 1), 1),
    ]:
        covered = t_mask & sel_mask
        cov_pct = covered.sum() / max(t_mask.sum(), 1) * 100
        acc     = accuracy_score(y_te[covered], y_pred[covered]) * 100 if covered.sum() >= 3 else np.nan
        trans_rows.append(dict(N=n, Transition=trans_name,
                               N_days=t_mask.sum(),
                               Coverage_pct=round(cov_pct, 1),
                               Accuracy=round(acc, 1) if not np.isnan(acc) else np.nan,
                               N_test_ex2022=len(y_te)))

pd.DataFrame(base_rows).to_csv('results/exclusion_2022_baselines.csv', index=False)
pd.DataFrame(trans_rows).to_csv('results/exclusion_2022_transitions.csv', index=False)
print("\nSaved → results/exclusion_2022_baselines.csv")
print("Saved → results/exclusion_2022_transitions.csv")

print("\n=== Baseline Summary (2022 excluded) ===")
print(pd.DataFrame(base_rows).to_string(index=False))
print("\n=== Calm→High Accuracy (2022 excluded) ===")
trans_df = pd.DataFrame(trans_rows)
ch = trans_df[trans_df['Transition'] == 'calm_high'][['N','N_days','Coverage_pct','Accuracy']]
print(ch.to_string(index=False))
