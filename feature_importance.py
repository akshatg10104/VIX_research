import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
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

N_VALUES = [5, 10, 20]

tscv_cal = TimeSeriesSplit(n_splits=5)

print(f"{'═'*72}")
print(f"  FEATURE IMPORTANCE — Random Forest (mean decrease in impurity)")
print(f"{'═'*72}")

all_importances = {}

for n in N_VALUES:
    label_col = f'LABEL_{n}'
    sub       = df[feature_cols + [label_col]].dropna(subset=[label_col])
    X = sub[feature_cols].values
    y = sub[label_col].astype(int).values

    split_idx   = int(len(sub) * 0.8)
    scaler      = StandardScaler()
    X_train     = scaler.fit_transform(X[:split_idx])
    y_train     = y[:split_idx]

    rf = RandomForestClassifier(n_estimators=300, max_depth=10,
                                min_samples_leaf=5, random_state=42)
    rf.fit(X_train, y_train)

    imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    all_importances[n] = imp

    print(f"\n  N={n} — Top 15 features:")
    print(f"  {'Feature':<25} {'Importance':>12}")
    print(f"  {'─'*40}")
    for feat, val in imp.head(15).items():
        print(f"  {feat:<25} {val:>11.4f}")

avg_imp = pd.concat(all_importances.values(), axis=1).mean(axis=1).sort_values(ascending=False)
avg_imp.to_csv('results/feature_importance.csv', header=['Avg_Importance'])

print(f"\n\n{'═'*72}")
print(f"  AVERAGE IMPORTANCE ACROSS N=5,10,20")
print(f"{'═'*72}")
print(f"  {'Feature':<25} {'Avg Importance':>14}")
print(f"  {'─'*42}")
for feat, val in avg_imp.head(20).items():
    print(f"  {feat:<25} {val:>13.4f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle('Random Forest Feature Importance by Prediction Horizon (N)', fontsize=12)

for ax, n in zip(axes, N_VALUES):
    imp = all_importances[n].head(15)
    bars = ax.barh(range(len(imp)), imp.values[::-1], color='#3498db', alpha=0.8)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp.index[::-1], fontsize=9)
    ax.set_title(f'N={n}')
    ax.set_xlabel('Importance')
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('results/feature_importance.png', dpi=150)
plt.close()

fig2, ax2 = plt.subplots(figsize=(10, 8))
top20 = avg_imp.head(20)
colors_imp = ['#e74c3c' if 'VIX' in f else
              '#3498db' if 'SP500' in f else
              '#2ecc71' if 'SKEW' in f or 'GOLD' in f else
              '#e67e22' for f in top20.index]
ax2.barh(range(len(top20)), top20.values[::-1], color=colors_imp[::-1], alpha=0.85)
ax2.set_yticks(range(len(top20)))
ax2.set_yticklabels(top20.index[::-1], fontsize=10)
ax2.set_title('Average Feature Importance Across All Horizons', fontsize=12)
ax2.set_xlabel('Mean Importance (N=5, 10, 20)')
ax2.grid(True, alpha=0.3, axis='x')

from matplotlib.patches import Patch
legend = [Patch(color='#e74c3c', label='VIX features'),
          Patch(color='#3498db', label='S&P 500 features'),
          Patch(color='#2ecc71', label='SKEW / Gold features'),
          Patch(color='#e67e22', label='Macro / Other')]
ax2.legend(handles=legend, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig('results/feature_importance_avg.png', dpi=150)
plt.close()

print(f"\nResults saved → results/feature_importance.csv")
print(f"Plots saved  → results/feature_importance.png, feature_importance_avg.png")
