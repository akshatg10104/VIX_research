import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

df = pd.read_csv('data/features.csv', index_col=0, parse_dates=True)

RAW_COLS     = ['VIX', 'VIX3M', 'SP500', 'GOLD', 'TNY', 'DXY', 'FEDFUNDS', 'YIELD_CURVE']
LABEL_COLS   = ['LABEL', 'LABEL_5', 'LABEL_10', 'LABEL_15', 'LABEL_20', 'LABEL_25']
feature_cols = [c for c in df.columns if c not in RAW_COLS + LABEL_COLS]

VIX_THRESHOLDS = [18, 20, 22]
N_VALUES       = [5, 10, 15, 20, 25]
CONF_THRESHOLD = 0.25

MODELS = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost':       HistGradientBoostingClassifier(max_iter=100, random_state=42),
}

rows = []

for vix_thresh in VIX_THRESHOLDS:
    print(f"\n{'═'*72}")
    print(f"  VIX THRESHOLD = {vix_thresh}  (high-vol regime defined as VIX ≥ {vix_thresh})")
    print(f"{'═'*72}")

    for n in N_VALUES:
        shifted = df['VIX'].shift(-n)
        label   = (shifted >= vix_thresh).astype(float).where(shifted.notna())

        sub = df[feature_cols].copy()
        sub['_LABEL'] = label
        sub = sub.dropna(subset=['_LABEL'])

        X = sub[feature_cols].values
        y = sub['_LABEL'].astype(int).values

        n_high = int((y == 1).sum())
        n_low  = int((y == 0).sum())

        split_idx               = int(len(sub) * 0.8)
        X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
        y_train, y_test         = y[:split_idx], y[split_idx:]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test  = scaler.transform(X_test_raw)

        print(f"\n  N={n}  |  total: {len(y)} rows  ({n_high} high-vol / {n_low} low-vol  {n_high/len(y)*100:.0f}%/{n_low/len(y)*100:.0f}%)")
        print(f"  {'Model':<18} {'Base Acc':>9} {'Sel Acc':>9} {'ΔAcc':>7} {'%Days':>7}")
        print(f"  {'─'*52}")

        for model_name, model in MODELS.items():
            model.fit(X_train, y_train)
            proba  = model.predict_proba(X_test)[:, 1]
            y_pred = (proba >= 0.5).astype(int)

            base_acc = accuracy_score(y_test, y_pred)

            mask      = (proba > 0.5 + CONF_THRESHOLD) | (proba < 0.5 - CONF_THRESHOLD)
            n_sel     = int(mask.sum())
            pct_sel   = n_sel / len(y_test) * 100
            sel_acc   = accuracy_score(y_test[mask], y_pred[mask]) if n_sel >= 10 else float('nan')
            delta     = sel_acc - base_acc if not np.isnan(sel_acc) else float('nan')

            print(f"  {model_name:<18} {base_acc*100:>8.2f}% {sel_acc*100:>8.2f}% {delta*100:>+6.2f}% {pct_sel:>6.1f}%")

            rows.append(dict(
                VIX_Threshold=vix_thresh, N=n, Model=model_name,
                Base_Accuracy=round(base_acc*100, 2),
                Sel_Accuracy=round(sel_acc*100, 2) if not np.isnan(sel_acc) else None,
                Delta_Acc=round(delta*100, 2) if not np.isnan(delta) else None,
                Days_Selected=n_sel,
                Pct_Selected=round(pct_sel, 1),
            ))

results_df = pd.DataFrame(rows)
results_df.to_csv('results/robustness_check.csv', index=False)
print(f"\n\nResults saved → results/robustness_check.csv")

print(f"\n{'═'*72}")
print("  ROBUSTNESS SUMMARY — Selective Predicting Gain (t=0.25) by VIX Threshold")
print(f"{'═'*72}")

for model_name in ['Random Forest', 'XGBoost']:
    print(f"\n  {model_name}")
    header = f"  {'':12}" + "".join(f"{'VIX≥'+str(t):>12}" for t in VIX_THRESHOLDS)
    print(header)
    print(f"  {'─'*48}")
    for n in N_VALUES:
        vals = []
        for vix_thresh in VIX_THRESHOLDS:
            r = results_df[
                (results_df['Model'] == model_name) &
                (results_df['N'] == n) &
                (results_df['VIX_Threshold'] == vix_thresh)
            ].iloc[0]
            vals.append(f"{r['Delta_Acc']:>+10.2f}%" if r['Delta_Acc'] is not None else f"{'N/A':>11}")
        print(f"  N={n:<9}" + "".join(vals))

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
fig.suptitle(f'Robustness Check — Selective Predicting Gain (t={CONF_THRESHOLD}) by VIX Threshold',
             fontsize=12)

colors = {18: '#e74c3c', 20: '#2196F3', 22: '#4CAF50'}
markers = {18: 's', 20: 'o', 22: '^'}

for ax, model_name in zip(axes, ['Random Forest', 'XGBoost']):
    for vix_thresh in VIX_THRESHOLDS:
        sub = results_df[
            (results_df['Model'] == model_name) &
            (results_df['VIX_Threshold'] == vix_thresh)
        ]
        ax.plot(sub['N'], sub['Delta_Acc'], marker=markers[vix_thresh],
                color=colors[vix_thresh], label=f'VIX ≥ {vix_thresh}', linewidth=2)

    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_title(model_name)
    ax.set_xlabel('N (days ahead)')
    ax.set_ylabel('Accuracy Gain vs Baseline (%)')
    ax.set_xticks(N_VALUES)
    ax.legend()
    ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig('results/robustness_check.png', dpi=150)
plt.close()
print("Plot saved → results/robustness_check.png")
