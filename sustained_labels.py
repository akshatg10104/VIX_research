import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, brier_score_loss
import matplotlib.pyplot as plt

df = pd.read_csv('data/features.csv', index_col=0, parse_dates=True)

RAW_COLS   = ['VIX', 'VIX3M', 'SP500', 'GOLD', 'TNY', 'DXY',
              'FEDFUNDS', 'YIELD_CURVE', 'VIX9D', 'SKEW']
LABEL_COLS = (
    ['LABEL'] +
    [f'LABEL_{n}' for n in [5, 10, 15, 20, 25]] +
    [f'LABEL_SUSTAINED_{n}' for n in [5, 10, 15, 20, 25]]
)
feature_cols = [c for c in df.columns if c not in RAW_COLS + LABEL_COLS]

N_VALUES      = [5, 10, 15, 20, 25]
CONF_THRESH   = 0.25

RF_GRID = {
    'n_estimators':    [100, 200, 300, 500],
    'max_depth':       [5, 10, 15, 20, None],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features':    ['sqrt', 'log2', 0.3, 0.5],
    'class_weight':    ['balanced', None],
}
XGB_GRID = {
    'max_iter':          [100, 200, 300],
    'max_depth':         [3, 5, 7, 10],
    'learning_rate':     [0.01, 0.05, 0.1, 0.2],
    'min_samples_leaf':  [10, 20, 50],
    'l2_regularization': [0.0, 0.1, 1.0],
    'class_weight':      ['balanced', None],
}

tscv      = TimeSeriesSplit(n_splits=5)
tscv_tune = TimeSeriesSplit(n_splits=3)

all_rows = []

print(f"{'═'*80}")
print(f"  SUSTAINED REGIME LABELS — VIX ≥ 20 for 3 consecutive days starting at N")
print(f"{'═'*80}")

for n in N_VALUES:
    label_col = f'LABEL_SUSTAINED_{n}'
    inst_col  = f'LABEL_{n}'

    sub = df[feature_cols + [label_col, inst_col]].dropna(subset=[label_col])

    X = sub[feature_cols].values
    y_sus = sub[label_col].astype(int).values
    y_ins = sub[inst_col].dropna().reindex(sub.index).astype(int).values

    split_idx   = int(len(sub) * 0.8)
    X_train_raw = X[:split_idx]
    X_tst_raw   = X[split_idx:]
    y_train     = y_sus[:split_idx]
    y_test      = y_sus[split_idx:]
    y_test_ins  = y_ins[split_idx:]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_tst_raw)

    n_high = int((y_test == 1).sum())
    n_low  = int((y_test == 0).sum())

    print(f"\n  N={n}  test={len(y_test)}  ({n_high} sustained-high / {n_low} low)")
    print(f"  {'Model':<18} {'Base':>7}  {'Cal':>7}  {'ΔBase':>7}  {'%Sel':>7}  {'SelAcc':>8}")
    print(f"  {'─'*58}")

    for model_name, BaseClass, param_grid in [
        ('Random Forest', RandomForestClassifier,         RF_GRID),
        ('XGBoost',       HistGradientBoostingClassifier, XGB_GRID),
    ]:
        search = RandomizedSearchCV(
            BaseClass(random_state=42), param_grid,
            n_iter=30, cv=tscv_tune,
            scoring='accuracy', n_jobs=-1, random_state=42
        )
        search.fit(X_train, y_train)
        tuned = search.best_estimator_

        cal = CalibratedClassifierCV(tuned, cv=tscv, method='isotonic')
        cal.fit(X_train, y_train)

        base = BaseClass(random_state=42)
        base.fit(X_train, y_train)

        proba_base = base.predict_proba(X_test)[:, 1]
        proba_cal  = cal.predict_proba(X_test)[:, 1]

        acc_base = accuracy_score(y_test, (proba_base >= 0.5).astype(int))
        acc_cal  = accuracy_score(y_test, (proba_cal  >= 0.5).astype(int))

        mask    = (proba_cal > 0.5 + CONF_THRESH) | (proba_cal < 0.5 - CONF_THRESH)
        n_sel   = int(mask.sum())
        pct_sel = n_sel / len(y_test) * 100
        acc_sel = accuracy_score(y_test[mask], (proba_cal[mask] >= 0.5).astype(int)) if n_sel >= 10 else float('nan')

        print(f"  {model_name:<18} {acc_base*100:>6.2f}%  {acc_cal*100:>6.2f}%  "
              f"{(acc_cal-acc_base)*100:>+6.2f}%  {pct_sel:>6.1f}%  {acc_sel*100:>7.2f}%")

        for label_type, proba, y_pr in [
            ('Baseline',   proba_base, (proba_base >= 0.5).astype(int)),
            ('Calibrated', proba_cal,  (proba_cal  >= 0.5).astype(int)),
        ]:
            all_rows.append(dict(
                N=n, Model=model_name, LabelType='Sustained', Type=label_type,
                Accuracy=round(accuracy_score(y_test, y_pr)*100, 2),
                Precision=round(precision_score(y_test, y_pr, zero_division=0)*100, 2),
                Recall=round(recall_score(y_test, y_pr, zero_division=0)*100, 2),
                Brier=round(brier_score_loss(y_test, proba), 4),
            ))

        if n_sel >= 10:
            all_rows.append(dict(
                N=n, Model=model_name, LabelType='Sustained', Type=f'Selective(t={CONF_THRESH})',
                Accuracy=round(acc_sel*100, 2),
                Precision=round(precision_score(y_test[mask], (proba_cal[mask]>=0.5).astype(int), zero_division=0)*100, 2),
                Recall=round(recall_score(y_test[mask], (proba_cal[mask]>=0.5).astype(int), zero_division=0)*100, 2),
                Brier=None,
            ))

results = pd.DataFrame(all_rows)
results.to_csv('results/sustained_labels.csv', index=False)
print(f"\nResults saved → results/sustained_labels.csv")

print(f"\n{'═'*80}")
print(f"  SUSTAINED vs INSTANTANEOUS — Calibrated Accuracy Comparison")
print(f"{'═'*80}")
inst_df = pd.read_csv('results/tuned_model_results.csv')
for model_name in ['Random Forest', 'XGBoost']:
    print(f"\n  {model_name}")
    print(f"  {'':8}{'N=5':>10}{'N=10':>10}{'N=15':>10}{'N=20':>10}{'N=25':>10}")
    for label_type in ['Calibrated', f'Selective(t={CONF_THRESH})']:
        vals_sus = []
        vals_ins = []
        for n in N_VALUES:
            r_sus = results[(results['Model']==model_name) & (results['N']==n) & (results['Type']==label_type)]
            r_ins = inst_df[(inst_df['Model']==model_name) & (inst_df['N']==n) & (inst_df['Type']==label_type)]
            vals_sus.append(f"{r_sus['Accuracy'].values[0]:.1f}%" if len(r_sus) and r_sus['Accuracy'].values[0] else 'N/A')
            vals_ins.append(f"{r_ins['Accuracy'].values[0]:.1f}%" if len(r_ins) and r_ins['Accuracy'].values[0] else 'N/A')
        short = label_type.replace(f'Selective(t={CONF_THRESH})', 'Selective').replace('Calibrated', 'Calibrated')
        print(f"  {'Sust '+short:<16}" + "".join(f"{v:>10}" for v in vals_sus))
        print(f"  {'Inst '+short:<16}" + "".join(f"{v:>10}" for v in vals_ins))
