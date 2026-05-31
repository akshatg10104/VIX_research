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
from sklearn.metrics import accuracy_score
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

N_VALUES    = [5, 10, 15, 20, 25]
THRESHOLDS  = np.arange(0.05, 0.41, 0.05).round(2)
MIN_DAYS    = 50

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

rows = []

print(f"{'═'*80}")
print(f"  THRESHOLD OPTIMIZATION — Best t per (Model, N)")
print(f"{'═'*80}")

for n in N_VALUES:
    label_col = f'LABEL_{n}'
    sub       = df[feature_cols + [label_col]].dropna(subset=[label_col])

    X = sub[feature_cols].values
    y = sub[label_col].astype(int).values

    split_idx   = int(len(sub) * 0.8)
    X_train_raw = X[:split_idx]
    X_tst_raw   = X[split_idx:]
    y_train     = y[:split_idx]
    y_test      = y[split_idx:]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_tst_raw)

    for model_name, BaseClass, param_grid in [
        ('Random Forest', RandomForestClassifier,         RF_GRID),
        ('XGBoost',       HistGradientBoostingClassifier, XGB_GRID),
    ]:
        search = RandomizedSearchCV(
            BaseClass(random_state=42), param_grid,
            n_iter=50, cv=tscv_tune,
            scoring='accuracy', n_jobs=-1, random_state=42
        )
        search.fit(X_train, y_train)
        tuned = search.best_estimator_

        cal = CalibratedClassifierCV(tuned, cv=tscv, method='isotonic')
        cal.fit(X_train, y_train)

        proba  = cal.predict_proba(X_test)[:, 1]
        y_pred = (proba >= 0.5).astype(int)
        base_acc = accuracy_score(y_test, y_pred)

        print(f"\n  {model_name}  N={n}  (base={base_acc*100:.2f}%)")
        print(f"  {'t':>6}  {'SelAcc':>8}  {'ΔAcc':>7}  {'%Days':>7}")
        print(f"  {'─'*34}")

        for t in THRESHOLDS:
            mask  = (proba > 0.5 + t) | (proba < 0.5 - t)
            n_sel = int(mask.sum())
            pct   = n_sel / len(y_test) * 100
            if n_sel < MIN_DAYS:
                break
            sel_acc = accuracy_score(y_test[mask], y_pred[mask])
            delta   = sel_acc - base_acc
            print(f"  t={t:.2f}  {sel_acc*100:>7.2f}%  {delta*100:>+6.2f}%  {pct:>6.1f}%")
            rows.append(dict(
                Model=model_name, N=n, Threshold=t,
                Base_Acc=round(base_acc*100, 2),
                Sel_Acc=round(sel_acc*100, 2),
                Delta=round(delta*100, 2),
                Pct_Days=round(pct, 1),
                N_Days=n_sel,
            ))

results = pd.DataFrame(rows)
results.to_csv('results/threshold_optimization.csv', index=False)

print(f"\n\n{'═'*80}")
print(f"  BEST THRESHOLD PER MODEL & N  (max SelAcc subject to ≥{MIN_DAYS} days)")
print(f"{'═'*80}")
print(f"  {'Model':<18} {'N':>4}  {'Best_t':>6}  {'SelAcc':>8}  {'ΔAcc':>7}  {'%Days':>7}")
print(f"  {'─'*60}")
best_rows = []
for model_name in ['Random Forest', 'XGBoost']:
    for n in N_VALUES:
        sub = results[(results['Model']==model_name) & (results['N']==n)]
        if sub.empty:
            continue
        best = sub.loc[sub['Sel_Acc'].idxmax()]
        print(f"  {model_name:<18} {n:>4}  t={best['Threshold']:.2f}  "
              f"{best['Sel_Acc']:>7.2f}%  {best['Delta']:>+6.2f}%  {best['Pct_Days']:>6.1f}%")
        best_rows.append(best)

pd.DataFrame(best_rows).to_csv('results/best_thresholds.csv', index=False)
print(f"\nResults saved → results/threshold_optimization.csv")
print(f"Best per N    → results/best_thresholds.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Selective Accuracy vs Confidence Threshold by N', fontsize=12)
colors = {5:'#e74c3c', 10:'#e67e22', 15:'#2ecc71', 20:'#3498db', 25:'#9b59b6'}

for ax, model_name in zip(axes, ['Random Forest', 'XGBoost']):
    for n in N_VALUES:
        sub = results[(results['Model']==model_name) & (results['N']==n)]
        if sub.empty:
            continue
        ax.plot(sub['Threshold'], sub['Sel_Acc'], marker='o',
                color=colors[n], label=f'N={n}', linewidth=2)
    ax.set_title(model_name)
    ax.set_xlabel('Confidence Threshold (t)')
    ax.set_ylabel('Selective Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig('results/threshold_optimization.png', dpi=150)
plt.close()
print("Plot saved → results/threshold_optimization.png")
