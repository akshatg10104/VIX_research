import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/features.csv', index_col=0, parse_dates=True)

RAW_COLS   = ['VIX', 'VIX3M', 'SP500', 'GOLD', 'TNY', 'DXY',
              'FEDFUNDS', 'YIELD_CURVE', 'VIX9D', 'SKEW']
LABEL_COLS = (
    ['LABEL'] +
    [f'LABEL_{n}' for n in [5, 10, 15, 20, 25]] +
    [f'LABEL_SUSTAINED_{n}' for n in [5, 10, 15, 20, 25]]
)
feature_cols = [c for c in df.columns if c not in RAW_COLS + LABEL_COLS]

MODELS = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost':       HistGradientBoostingClassifier(max_iter=100, random_state=42),
}

N_VALUES   = [5, 20]
THRESHOLD  = 0.25
LABELS_CM  = ['Low Vol\n(UNDER)', 'High Vol\n(OVER)']

fig, axes = plt.subplots(
    nrows=len(MODELS) * len(N_VALUES),
    ncols=2,
    figsize=(10, 5 * len(MODELS) * len(N_VALUES))
)
fig.suptitle('Confusion Matrices — Baseline vs Confidence Selective Predicting (t=0.25)',
             fontsize=13, y=1.01)

row = 0
summary_rows = []

for model_name, model in MODELS.items():
    for n in N_VALUES:
        label_col = f'LABEL_{n}'
        sub       = df[feature_cols + [label_col]].dropna(subset=[label_col])

        X = sub[feature_cols].values
        y = sub[label_col].astype(int).values

        split_idx               = int(len(sub) * 0.8)
        X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
        y_train, y_test         = y[:split_idx], y[split_idx:]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test  = scaler.transform(X_test_raw)

        model.fit(X_train, y_train)
        proba  = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= 0.5).astype(int)

        conf_mask  = (proba > 0.5 + THRESHOLD) | (proba < 0.5 - THRESHOLD)
        y_test_sel = y_test[conf_mask]
        y_pred_sel = y_pred[conf_mask]

        for col_idx, (y_true_plot, y_pred_plot, label) in enumerate([
            (y_test,     y_pred,     'Baseline'),
            (y_test_sel, y_pred_sel, f'Selective (t={THRESHOLD})'),
        ]):
            cm  = confusion_matrix(y_true_plot, y_pred_plot)
            acc = accuracy_score(y_true_plot, y_pred_plot)
            prc = precision_score(y_true_plot, y_pred_plot, zero_division=0)
            rec = recall_score(y_true_plot, y_pred_plot, zero_division=0)
            n_sel = len(y_true_plot)
            pct   = n_sel / len(y_test) * 100

            ax = axes[row][col_idx]
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABELS_CM, yticklabels=LABELS_CM,
                ax=ax, cbar=False, linewidths=0.5
            )
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(
                f'{model_name}  |  N={n}  |  {label}\n'
                f'Acc={acc*100:.1f}%  Prec={prc*100:.1f}%  Rec={rec*100:.1f}%'
                + (f'  ({pct:.0f}% of days)' if label != 'Baseline' else ''),
                fontsize=9
            )

            summary_rows.append(dict(
                Model=model_name, N=n, Condition=label,
                Accuracy=round(acc*100, 2),
                Precision=round(prc*100, 2),
                Recall=round(rec*100, 2),
                Days=n_sel,
                Pct_Days=round(pct, 1),
                TN=cm[0,0], FP=cm[0,1], FN=cm[1,0], TP=cm[1,1]
            ))

        row += 1

plt.tight_layout()
plt.savefig('results/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot saved → results/confusion_matrices.png")

summary = pd.DataFrame(summary_rows)
summary.to_csv('results/confusion_matrices.csv', index=False)
print("Data saved → results/confusion_matrices.csv\n")

print(f"{'='*72}")
print(f"  {'Model':<18} {'N':>3}  {'Condition':<28} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'Days':>6} {'%Days':>6}")
print(f"  {'─'*68}")
for _, r in summary.iterrows():
    pct_str = f"{r['Pct_Days']:.0f}%" if r['Condition'] != 'Baseline' else '100%'
    print(f"  {r['Model']:<18} {r['N']:>3}  {r['Condition']:<28} "
          f"{r['Accuracy']:>6.1f}% {r['Precision']:>6.1f}% {r['Recall']:>6.1f}% "
          f"{int(r['Days']):>6}  {pct_str:>5}")

print(f"\n{'='*72}")
print("  DELTA: Selective vs Baseline")
print(f"{'='*72}")
for model_name in MODELS:
    for n in N_VALUES:
        base = summary[(summary['Model']==model_name) & (summary['N']==n) & (summary['Condition']=='Baseline')].iloc[0]
        sel  = summary[(summary['Model']==model_name) & (summary['N']==n) & (summary['Condition']==f'Selective (t={THRESHOLD})')].iloc[0]
        print(f"  {model_name}  N={n}:  "
              f"Acc {base['Accuracy']:.1f}% → {sel['Accuracy']:.1f}% ({sel['Accuracy']-base['Accuracy']:+.1f}%)  |  "
              f"Prec {base['Precision']:.1f}% → {sel['Precision']:.1f}%  |  "
              f"Rec {base['Recall']:.1f}% → {sel['Recall']:.1f}%  |  "
              f"{sel['Pct_Days']:.0f}% of days selected")
