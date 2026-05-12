import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

df = pd.read_csv('data/features.csv', index_col=0, parse_dates=True)

RAW_COLS     = ['VIX', 'VIX3M', 'SP500', 'GOLD', 'TNY', 'DXY']
LABEL_COLS   = ['LABEL', 'LABEL_5', 'LABEL_10', 'LABEL_15', 'LABEL_20', 'LABEL_25']
feature_cols = [c for c in df.columns if c not in RAW_COLS + LABEL_COLS]

N_VALUES = [5, 10, 15, 20, 25]

def make_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree':       DecisionTreeClassifier(max_depth=5),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM':                 SVC(kernel='poly', degree=4, C=0.001, probability=True),
        'XGBoost':             HistGradientBoostingClassifier(max_iter=100, random_state=42),
    }

all_results = {}

for n in N_VALUES:
    label_col = f'LABEL_{n}'
    sub       = df[feature_cols + [label_col]].dropna(subset=[label_col])

    X = sub[feature_cols]
    y = sub[label_col].astype(int)

    split_idx   = int(len(sub) * 0.8)
    X_train_raw = X.iloc[:split_idx]
    X_test_raw  = X.iloc[split_idx:]
    y_train     = y.iloc[:split_idx]
    y_test      = y.iloc[split_idx:]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    horizon_results = {}
    models          = make_models()

    print(f"\n{'='*58}")
    print(f"  N = {n} days ahead   |  train={len(y_train)}d  test={len(y_test)}d")
    print(f"  Test label split: {int((y_test==1).sum())} high-vol / {int((y_test==0).sum())} low-vol")
    print(f"{'='*58}")
    print(f"  {'Model':<22} {'Acc':>7} {'Prec':>7} {'Recall':>8}")
    print(f"  {'-'*48}")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)

        horizon_results[name] = {
            'Accuracy':  round(acc  * 100, 2),
            'Precision': round(prec * 100, 2),
            'Recall':    round(rec  * 100, 2),
        }
        print(f"  {name:<22} {acc*100:>6.2f}% {prec*100:>6.2f}% {rec*100:>7.2f}%")

    all_results[n] = horizon_results

rows = []
for n, horizon in all_results.items():
    for model_name, metrics in horizon.items():
        rows.append({'N': n, 'Model': model_name, **metrics})

results_df = pd.DataFrame(rows)
results_df.to_csv('results/lookahead_model_comparison.csv', index=False)

print(f"\n{'='*58}")
print("  ACCURACY SUMMARY (%) — all horizons")
print(f"{'='*58}")
acc_pivot         = results_df.pivot(index='Model', columns='N', values='Accuracy')
acc_pivot.columns = [f'N={n}' for n in acc_pivot.columns]
print(acc_pivot.to_string())

fig, ax = plt.subplots(figsize=(10, 6))
for model_name in results_df['Model'].unique():
    sub = results_df[results_df['Model'] == model_name]
    ax.plot(sub['N'], sub['Accuracy'], marker='o', label=model_name)

ax.set_title('Model Accuracy vs Prediction Horizon (N days ahead)')
ax.set_xlabel('N (days ahead)')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(N_VALUES)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/accuracy_vs_horizon.png', dpi=150)
plt.close()
