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

N_VALUES   = [5, 10, 15, 20, 25]
THRESHOLDS = [0.10, 0.15, 0.20, 0.25]

MODELS = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost':       HistGradientBoostingClassifier(max_iter=100, random_state=42),
}

def evaluate(y_true, y_pred):
    return (
        round(accuracy_score(y_true, y_pred)                   * 100, 2),
        round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
        round(recall_score(y_true, y_pred, zero_division=0)    * 100, 2),
    )

def print_row(thresh_label, model_name, acc, prec, rec, n_sel, pct, delta_acc):
    delta_str = '—' if delta_acc is None else f'{delta_acc:+.2f}%'
    print(f"  {thresh_label:<12} {model_name:<18} "
          f"{acc:>6.2f}%  {prec:>6.2f}%  {rec:>6.2f}%  "
          f"{n_sel:>5}  {pct:>5.1f}%  {delta_str:>8}")

all_rows = []

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

    n_test     = len(y_test)
    n_high_vol = int((y_test == 1).sum())
    n_low_vol  = int((y_test == 0).sum())

    print(f"\n{'═'*72}")
    print(f"  N = {n} days ahead  │  test: {n_test} days ({n_high_vol} high-vol / {n_low_vol} low-vol)")
    print(f"{'═'*72}")
    print(f"  {'Threshold':<12} {'Model':<18} {'Acc':>7}  {'Prec':>7}  {'Recall':>7}  {'Days':>5}  {'%Sel':>6}  {'ΔAcc':>8}")
    print(f"  {'─'*68}")

    baseline_acc = {}

    for model_name, model in MODELS.items():
        model.fit(X_train, y_train)
        proba  = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= 0.5).astype(int)

        acc, prec, rec = evaluate(y_test, y_pred)
        baseline_acc[model_name] = acc
        print_row('Baseline', model_name, acc, prec, rec, n_test, 100.0, None)
        all_rows.append(dict(N=n, Threshold='Baseline', Model=model_name,
                             Accuracy=acc, Precision=prec, Recall=rec,
                             Days_Selected=n_test, Pct_Selected=100.0, Delta_Acc=0.0))

        for t in THRESHOLDS:
            mask  = (proba > 0.5 + t) | (proba < 0.5 - t)
            n_sel = int(mask.sum())
            pct   = n_sel / n_test * 100

            if n_sel < 10:
                continue

            y_sel = y_test[mask]
            p_sel = y_pred[mask]

            acc, prec, rec = evaluate(y_sel, p_sel)
            delta          = acc - baseline_acc[model_name]
            print_row(f't={t}', model_name, acc, prec, rec, n_sel, pct, delta)
            all_rows.append(dict(N=n, Threshold=f't={t}', Model=model_name,
                                 Accuracy=acc, Precision=prec, Recall=rec,
                                 Days_Selected=n_sel, Pct_Selected=round(pct, 1),
                                 Delta_Acc=round(delta, 2)))

        print(f"  {'·'*68}")

results_df = pd.DataFrame(all_rows)
results_df.to_csv('results/confidence_selective_predicting.csv', index=False)
print(f"\nResults saved → results/confidence_selective_predicting.csv")

for model_name in ['XGBoost', 'Random Forest']:
    sub           = results_df[results_df['Model'] == model_name]
    pivot         = sub.pivot(index='Threshold', columns='N', values='Accuracy')
    baseline_vals = pivot.loc['Baseline']
    delta         = pivot.drop('Baseline').subtract(baseline_vals)
    delta.columns = [f'N={c}' for c in delta.columns]
    delta.index   = [f't={t}' for t in THRESHOLDS[:len(delta)]]
    print(f"\n  {model_name}")
    print(f"  {'':14}" + "".join(f"{c:>9}" for c in delta.columns))
    print(f"  {'─'*62}")
    for idx, row in delta.iterrows():
        print(f"  {idx:<14}" + "".join(f"{v:>+8.2f}%" for v in row.values))

fig, axes = plt.subplots(2, len(N_VALUES), figsize=(20, 9))
fig.suptitle(
    'Confidence-Based Selective Predicting\n'
    'Top: Accuracy vs Threshold  |  Bottom: % Days Selected vs Threshold',
    fontsize=12
)

thresh_x      = [0] + THRESHOLDS
thresh_labels = ['Base'] + [f't={t}' for t in THRESHOLDS]
colors        = {'XGBoost': '#2196F3', 'Random Forest': '#4CAF50'}

for col, n in enumerate(N_VALUES):
    ax_acc = axes[0][col]
    ax_sel = axes[1][col]
    sub    = results_df[results_df['N'] == n]

    for model_name, color in colors.items():
        msub = sub[sub['Model'] == model_name].set_index('Threshold')
        accs = [msub.loc['Baseline', 'Accuracy']] + \
               [msub.loc[f't={t}', 'Accuracy'] if f't={t}' in msub.index else np.nan for t in THRESHOLDS]
        sels = [100.0] + \
               [msub.loc[f't={t}', 'Pct_Selected'] if f't={t}' in msub.index else np.nan for t in THRESHOLDS]

        ax_acc.plot(range(len(thresh_x)), accs, marker='o', label=model_name, color=color)
        ax_sel.plot(range(len(thresh_x)), sels, marker='s', linestyle='--', color=color, alpha=0.8)

    for ax in (ax_acc, ax_sel):
        ax.set_xticks(range(len(thresh_labels)))
        ax.set_xticklabels(thresh_labels, fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.set_title(f'N = {n}', fontsize=10)

    ax_acc.set_ylim(60, 100)
    ax_sel.set_ylim(0, 110)

axes[0][0].set_ylabel('Accuracy (%)')
axes[1][0].set_ylabel('% Days Selected')
axes[0][0].legend(fontsize=8)
plt.tight_layout()
plt.savefig('results/confidence_selective_predicting.png', dpi=150)
plt.close()
print("Plot saved → results/confidence_selective_predicting.png")
