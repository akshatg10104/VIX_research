"""
Permutation importance (unbiased alternative to MDI).
Averages across N=5, 10, 20 using fixed-spec calibrated RF.
Outputs:
  results/permutation_importance.csv
  vix paper/permutation_importance.png
"""
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'legend.fontsize': 10, 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
})

OUT = 'vix paper'
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv('data/features.csv', index_col=0, parse_dates=True)
RAW_COLS   = ['VIX', 'VIX3M', 'SP500', 'GOLD', 'TNY', 'DXY',
              'FEDFUNDS', 'YIELD_CURVE', 'VIX9D', 'SKEW']
LABEL_COLS = (['LABEL'] +
              [f'LABEL_{n}' for n in [5, 10, 15, 20, 25]] +
              [f'LABEL_SUSTAINED_{n}' for n in [5, 10, 15, 20, 25]])
feature_cols = [c for c in df.columns if c not in RAW_COLS + LABEL_COLS]

N_EVAL  = [5, 10, 20]
tscv    = TimeSeriesSplit(n_splits=5)
N_PERMS = 15   # permutation repeats per feature

importances = pd.DataFrame(index=feature_cols)

for n in N_EVAL:
    label_col = f'LABEL_{n}'
    sub = df[feature_cols + [label_col]].dropna(subset=[label_col])
    X   = sub[feature_cols].values
    y   = sub[label_col].astype(int).values
    split_idx = int(len(sub) * 0.8)

    X_tr_raw, X_te_raw = X[:split_idx], X[split_idx:]
    y_tr, y_te         = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr_raw)
    X_te   = scaler.transform(X_te_raw)

    rf     = RandomForestClassifier(n_estimators=200, max_depth=10,
                                    min_samples_leaf=5, random_state=42)
    cal_rf = CalibratedClassifierCV(rf, cv=tscv, method='isotonic')
    cal_rf.fit(X_tr, y_tr)

    print(f"  Computing permutation importance for N={n}...", flush=True)
    perm = permutation_importance(cal_rf, X_te, y_te,
                                  n_repeats=N_PERMS, random_state=42,
                                  scoring='accuracy', n_jobs=-1)
    importances[f'N={n}'] = perm.importances_mean

importances['Avg'] = importances.mean(axis=1)
importances = importances.sort_values('Avg', ascending=False)
importances.to_csv('results/permutation_importance.csv')
print(f"\nSaved → results/permutation_importance.csv")

# ── Figure ────────────────────────────────────────────────────────────────────
top20 = importances.head(20)

def feat_color(name):
    if 'VIX' in name:   return '#e74c3c'
    if 'SP500' in name: return '#3498db'
    if 'SKEW' in name or 'GOLD' in name: return '#2ecc71'
    return '#e67e22'

cols_fi = [feat_color(f) for f in top20.index[::-1]]

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(len(top20)), top20['Avg'].values[::-1],
        color=cols_fi, alpha=0.88, edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20.index[::-1], fontsize=10)
ax.set_title('Permutation Feature Importance — Avg across N=5, 10, 20\n'
             '(Mean accuracy drop when feature values are shuffled)', fontsize=12)
ax.set_xlabel('Mean Accuracy Decrease (permutation importance)')
ax.grid(True, alpha=0.2, axis='x')
ax.axvline(0, color='black', linewidth=0.8)

legend_handles = [
    mpatches.Patch(color='#e74c3c', label='VIX features'),
    mpatches.Patch(color='#3498db', label='S&P 500 features'),
    mpatches.Patch(color='#2ecc71', label='SKEW / Gold features'),
    mpatches.Patch(color='#e67e22', label='Macro / Other'),
]
ax.legend(handles=legend_handles, loc='lower right', fontsize=10, framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{OUT}/permutation_importance.png', bbox_inches='tight')
plt.close()
print(f"Figure saved → {OUT}/permutation_importance.png")
print(top20[['Avg']].head(10).to_string())
