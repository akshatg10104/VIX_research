"""
Cost-weighted utility analysis.
FN cost = C (missed spike), FP cost = 1 (false alarm), TN/TP cost = 0.
Utility = -(FP * 1 + FN * C) / N_covered

Compare: RF selective, baseline (predict all), persistence at matched coverage.
Evaluate for C in {1, 5, 10, 20} and N in {5, 10}.

Outputs:
  results/cost_utility.csv
  vix paper/cost_utility.png
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
import matplotlib.pyplot as plt
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

TAU      = 0.25
N_EVAL   = [5, 10]
C_VALUES = [1, 5, 10, 20]   # relative cost of FN vs FP
tscv     = TimeSeriesSplit(n_splits=5)

def utility_rate(y_true, y_pred, C):
    """Return utility per prediction: -(FP + C*FN) / n_predictions."""
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return -(fp + C * fn) / len(y_pred)

def find_persist_c(vix_vals, n_target):
    lo, hi = 0.0, 25.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if np.sum(np.abs(vix_vals - 20) >= mid) > n_target:
            lo = mid
        else:
            hi = mid
    return mid

rows = []

for n in N_EVAL:
    label_col = f'LABEL_{n}'
    sub = df[feature_cols + [label_col, 'VIX']].dropna(subset=[label_col])
    X   = sub[feature_cols].values
    y   = sub[label_col].astype(int).values
    vix = sub['VIX'].values
    split_idx = int(len(sub) * 0.8)

    X_tr_raw, X_te_raw = X[:split_idx], X[split_idx:]
    y_tr, y_te         = y[:split_idx], y[split_idx:]
    vix_te             = vix[split_idx:]

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr_raw)
    X_te   = scaler.transform(X_te_raw)

    # Baseline: uncalibrated RF
    from sklearn.base import clone
    rf_base = clone(RandomForestClassifier(n_estimators=200, max_depth=10,
                                           min_samples_leaf=5, random_state=42))
    rf_base.fit(X_tr, y_tr)
    y_pred_base = rf_base.predict(X_te)

    # Selective: calibrated RF
    rf     = RandomForestClassifier(n_estimators=200, max_depth=10,
                                    min_samples_leaf=5, random_state=42)
    cal_rf = CalibratedClassifierCV(rf, cv=tscv, method='isotonic')
    cal_rf.fit(X_tr, y_tr)
    proba   = cal_rf.predict_proba(X_te)[:, 1]
    mask_rf = (proba > 0.5 + TAU) | (proba < 0.5 - TAU)
    n_rf    = int(mask_rf.sum())
    y_pred_rf = (proba >= 0.5).astype(int)

    # Persistence at matched coverage
    c = find_persist_c(vix_te, n_rf)
    mask_per  = np.abs(vix_te - 20) >= c
    y_pred_per = (vix_te >= 20).astype(int)

    for C in C_VALUES:
        u_base = utility_rate(y_te, y_pred_base, C)
        u_sel  = utility_rate(y_te[mask_rf], y_pred_rf[mask_rf], C) if n_rf >= 10 else np.nan
        u_per  = utility_rate(y_te[mask_per], y_pred_per[mask_per], C) if mask_per.sum() >= 10 else np.nan
        # Majority class baseline (always predict calm = 0)
        u_maj  = utility_rate(y_te, np.zeros_like(y_te), C)

        print(f"  N={n}, C={C:>2}: Baseline={u_base:.3f}  Selective={u_sel:.3f}  "
              f"Persistence={u_per:.3f}  Always-calm={u_maj:.3f}")

        rows.append(dict(
            N=n, C=C,
            Utility_Baseline=round(u_base, 4),
            Utility_Selective=round(u_sel, 4) if not np.isnan(u_sel) else None,
            Utility_Persistence=round(u_per, 4) if not np.isnan(u_per) else None,
            Utility_AlwaysCalm=round(u_maj, 4),
            Coverage_RF=round(n_rf / len(y_te) * 100, 1),
        ))

results = pd.DataFrame(rows)
results.to_csv('results/cost_utility.csv', index=False)
print(f"\nSaved → results/cost_utility.csv")
print(results.to_string(index=False))

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Cost-Weighted Utility by FN/FP Cost Ratio (C)\n'
             'Utility = −(FP + C × FN) / n_predictions;  higher is better',
             fontsize=12)

strategies = {
    'Utility_Baseline':    ('RF Baseline (all days)',   '#888888', '--'),
    'Utility_Selective':   (f'RF Selective (τ={TAU})',  '#2e7d32', '-o'),
    'Utility_Persistence': ('Persistence (matched cov)','#e74c3c', '-s'),
    'Utility_AlwaysCalm':  ('Always calm',              '#3498db', ':'),
}

for ax, n in zip(axes, N_EVAL):
    sub = results[results['N'] == n]
    for col, (label, color, ls) in strategies.items():
        vals = sub[col].values.astype(float)
        ax.plot(sub['C'], vals, ls, color=color, linewidth=2, markersize=7,
                label=label, markerfacecolor='white')
    ax.set_title(f'N = {n}')
    ax.set_xlabel('FN/FP Cost Ratio (C)')
    ax.set_ylabel('Utility per prediction (higher = better)')
    ax.set_xticks(C_VALUES)
    ax.legend(framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.axhline(0, color='black', linestyle=':', linewidth=0.8, alpha=0.5)

plt.tight_layout()
plt.savefig(f'{OUT}/cost_utility.png', bbox_inches='tight')
plt.close()
print(f"Figure saved → {OUT}/cost_utility.png")
