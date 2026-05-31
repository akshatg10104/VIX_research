import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import warnings
import json
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
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

N_VALUES       = [5, 10, 15, 20, 25]
CONF_THRESHOLD = 0.25
N_ITER_SEARCH  = 50

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

all_rows     = []
best_params  = {}

for n in N_VALUES:
    label_col = f'LABEL_{n}'
    sub       = df[feature_cols + [label_col]].dropna(subset=[label_col])

    X = sub[feature_cols].values
    y = sub[label_col].astype(int).values

    split_idx = int(len(sub) * 0.8)

    X_train_raw = X[:split_idx]
    X_tst_raw   = X[split_idx:]
    y_train     = y[:split_idx]
    y_test      = y[split_idx:]

    scaler    = StandardScaler()
    X_train   = scaler.fit_transform(X_train_raw)
    X_test    = scaler.transform(X_tst_raw)

    n_test = len(y_test)
    print(f"\n{'═'*72}")
    print(f"  N = {n}  |  train={len(y_train)}  test={n_test}"
          f"  ({int((y_test==1).sum())} high / {int((y_test==0).sum())} low)")
    print(f"{'═'*72}")
    print(f"  {'Model':<22} {'Base':>7}  {'Tuned':>7}  {'Cal':>7}  {'ΔBase':>7}  {'ΔTune':>7}  {'%Sel':>6}  {'SelAcc':>7}")
    print(f"  {'─'*70}")

    best_params[n] = {}

    for model_name, BaseClass, param_grid in [
        ('Random Forest', RandomForestClassifier,           RF_GRID),
        ('XGBoost',       HistGradientBoostingClassifier,   XGB_GRID),
    ]:
        base_model = BaseClass(random_state=42)
        search     = RandomizedSearchCV(
            base_model, param_grid,
            n_iter=N_ITER_SEARCH, cv=tscv_tune,
            scoring='accuracy', n_jobs=-1, random_state=42
        )
        search.fit(X_train, y_train)
        tuned_model = search.best_estimator_

        best_params[n][model_name] = search.best_params_

        cal_model = CalibratedClassifierCV(tuned_model, cv=tscv, method='isotonic')
        cal_model.fit(X_train, y_train)

        base_model_full = BaseClass(random_state=42)
        base_model_full.fit(X_train, y_train)

        proba_base  = base_model_full.predict_proba(X_test)[:, 1]
        proba_tuned = tuned_model.predict_proba(X_test)[:, 1]
        proba_cal   = cal_model.predict_proba(X_test)[:, 1]

        pred_base  = (proba_base  >= 0.5).astype(int)
        pred_tuned = (proba_tuned >= 0.5).astype(int)
        pred_cal   = (proba_cal   >= 0.5).astype(int)

        acc_base  = accuracy_score(y_test, pred_base)
        acc_tuned = accuracy_score(y_test, pred_tuned)
        acc_cal   = accuracy_score(y_test, pred_cal)

        mask    = (proba_cal > 0.5 + CONF_THRESHOLD) | (proba_cal < 0.5 - CONF_THRESHOLD)
        n_sel   = int(mask.sum())
        pct_sel = n_sel / n_test * 100
        acc_sel = accuracy_score(y_test[mask], pred_cal[mask]) if n_sel >= 10 else float('nan')

        delta_base  = acc_cal - acc_base
        delta_tuned = acc_cal - acc_tuned

        print(f"  {model_name:<22} {acc_base*100:>6.2f}%  {acc_tuned*100:>6.2f}%  "
              f"{acc_cal*100:>6.2f}%  {delta_base*100:>+6.2f}%  {delta_tuned*100:>+6.2f}%  "
              f"{pct_sel:>5.1f}%  {acc_sel*100:>6.2f}%")

        for label_type, proba, y_pred, acc in [
            ('Baseline',   proba_base,  pred_base,  acc_base),
            ('Tuned',      proba_tuned, pred_tuned, acc_tuned),
            ('Calibrated', proba_cal,   pred_cal,   acc_cal),
        ]:
            all_rows.append(dict(
                N=n, Model=model_name, Type=label_type,
                Accuracy=round(acc*100, 2),
                Precision=round(precision_score(y_test, y_pred, zero_division=0)*100, 2),
                Recall=round(recall_score(y_test, y_pred, zero_division=0)*100, 2),
                Brier=round(brier_score_loss(y_test, proba), 4),
            ))

        sel_acc = acc_sel if not np.isnan(acc_sel) else None
        all_rows.append(dict(
            N=n, Model=model_name, Type=f'Selective(t={CONF_THRESHOLD})',
            Accuracy=round(acc_sel*100, 2) if sel_acc else None,
            Precision=round(precision_score(y_test[mask], pred_cal[mask], zero_division=0)*100, 2) if n_sel>=10 else None,
            Recall=round(recall_score(y_test[mask], pred_cal[mask], zero_division=0)*100, 2) if n_sel>=10 else None,
            Brier=None,
        ))

results_df = pd.DataFrame(all_rows)
results_df.to_csv('results/tuned_model_results.csv', index=False)

with open('results/best_params.json', 'w') as f:
    json.dump(best_params, f, indent=2, default=str)

print(f"\n\nResults saved → results/tuned_model_results.csv")
print(f"Best params  → results/best_params.json")

print(f"\n{'═'*72}")
print("  FINAL SUMMARY — Calibrated + Selective (t=0.25) vs Raw Baseline")
print(f"{'═'*72}")
print(f"  {'Model':<22}  " + "".join(f"{'N='+str(n):>10}" for n in N_VALUES))
print(f"  {'─'*70}")
for model_name in ['Random Forest', 'XGBoost']:
    base_accs = []
    sel_accs  = []
    for n in N_VALUES:
        b = results_df[(results_df['Model']==model_name) & (results_df['N']==n) & (results_df['Type']=='Baseline')]['Accuracy'].values
        s = results_df[(results_df['Model']==model_name) & (results_df['N']==n) & (results_df['Type']==f'Selective(t={CONF_THRESHOLD})')]['Accuracy'].values
        base_accs.append(f"{b[0]:.1f}%" if len(b) else 'N/A')
        sel_accs.append(f"{s[0]:.1f}%" if len(s) and s[0] else 'N/A')
    print(f"  {model_name+' (base)':<22}  " + "".join(f"{v:>10}" for v in base_accs))
    print(f"  {model_name+' (sel)':<22}  " + "".join(f"{v:>10}" for v in sel_accs))
    print()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Tuned + Calibrated Models: Accuracy vs Horizon', fontsize=12)

for ax, model_name in zip(axes, ['Random Forest', 'XGBoost']):
    for label_type, style, color in [
        ('Baseline',                  '--', '#aaaaaa'),
        ('Tuned',                     '-',  '#FF9800'),
        ('Calibrated',                '-',  '#2196F3'),
        (f'Selective(t={CONF_THRESHOLD})', '-o', '#4CAF50'),
    ]:
        sub = results_df[(results_df['Model']==model_name) & (results_df['Type']==label_type)]
        if sub.empty:
            continue
        ax.plot(sub['N'], sub['Accuracy'], style, label=label_type, color=color, linewidth=2)
    ax.set_title(model_name)
    ax.set_xlabel('N (days ahead)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(N_VALUES)
    ax.legend()
    ax.grid(True, alpha=0.25)
    ax.set_ylim(60, 100)

plt.tight_layout()
plt.savefig('results/tuned_model_comparison.png', dpi=150)
plt.close()
print("Plot saved → results/tuned_model_comparison.png")
