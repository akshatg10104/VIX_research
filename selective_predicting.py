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

RAW_COLS     = ['VIX', 'VIX3M', 'SP500', 'GOLD', 'TNY', 'DXY']
LABEL_COLS   = ['LABEL', 'LABEL_5', 'LABEL_10', 'LABEL_15', 'LABEL_20', 'LABEL_25']
feature_cols = [c for c in df.columns if c not in RAW_COLS + LABEL_COLS]

MOM20_IDX  = feature_cols.index('VIX_MOM20')
N_VALUES   = [5, 10, 15, 20, 25]
THRESHOLDS = [None, 0.1, 0.2, 0.3]

MODELS = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost':       HistGradientBoostingClassifier(max_iter=100, random_state=42),
}

rows = []

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

    mom20_test = X_test[:, MOM20_IDX]

    print(f"\n{'='*66}")
    print(f"  N = {n} | test={len(y_test)} ({int((y_test==1).sum())} high-vol / {int((y_test==0).sum())} low-vol)")
    print(f"{'='*66}")
    print(f"  {'Threshold':<12} {'Model':<18} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'Days':>6} {'%Sel':>6} {'ΔAcc':>7}")
    print(f"  {'-'*62}")

    baseline_acc = {}

    for thresh in THRESHOLDS:
        if thresh is None:
            mask         = np.ones(len(y_test), dtype=bool)
            thresh_label = 'Baseline'
        else:
            mask         = mom20_test > thresh
            thresh_label = str(thresh)

        n_sel   = int(mask.sum())
        pct_sel = n_sel / len(y_test) * 100

        if n_sel < 10:
            continue

        y_sel = y_test[mask]
        X_sel = X_test[mask]

        for model_name, model in MODELS.items():
            if thresh is None:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_sel)

            acc  = accuracy_score(y_sel, y_pred)
            prec = precision_score(y_sel, y_pred, zero_division=0)
            rec  = recall_score(y_sel, y_pred, zero_division=0)

            if thresh is None:
                baseline_acc[model_name] = acc
                delta_str = '—'
            else:
                delta     = (acc - baseline_acc.get(model_name, acc)) * 100
                delta_str = f'{delta:+.2f}%'

            print(f"  {thresh_label:<12} {model_name:<18} "
                  f"{acc*100:>6.2f}% {prec*100:>6.2f}% {rec*100:>6.2f}% "
                  f"{n_sel:>6} {pct_sel:>6.1f}% {delta_str:>7}")

            rows.append({
                'N': n, 'Threshold': thresh_label, 'Model': model_name,
                'Accuracy': round(acc * 100, 2), 'Precision': round(prec * 100, 2),
                'Recall': round(rec * 100, 2), 'Days_Selected': n_sel,
                'Pct_Selected': round(pct_sel, 1),
            })

results_df = pd.DataFrame(rows)
results_df.to_csv('results/selective_predicting_results.csv', index=False)

fig, axes = plt.subplots(1, len(N_VALUES), figsize=(18, 5), sharey=True)
thresh_labels = ['Baseline', '0.1', '0.2', '0.3']
x_pos = np.arange(len(thresh_labels))

for ax, n in zip(axes, N_VALUES):
    sub = results_df[results_df['N'] == n]
    for model_name in ['Random Forest', 'XGBoost']:
        msub = sub[sub['Model'] == model_name].set_index('Threshold')
        accs = [msub.loc[t, 'Accuracy'] if t in msub.index else np.nan for t in thresh_labels]
        ax.plot(x_pos, accs, marker='o', label=model_name)
    ax.set_title(f'N = {n}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(thresh_labels)
    ax.set_xlabel('Threshold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(60, 100)

axes[0].set_ylabel('Accuracy (%)')
axes[0].legend()
plt.tight_layout()
plt.savefig('results/selective_predicting_accuracy.png', dpi=150)
plt.close()
