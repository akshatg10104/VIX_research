"""
Two diagnostics:
1. Calibration reliability diagrams + Brier scores (N=5 and N=10)
2. Moving-block bootstrap 95% CI on (RF selective acc − Persistence acc)
   at matched coverage, for all N values
Outputs to paper draft 2/
"""
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, brier_score_loss
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'legend.fontsize': 10, 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
})

OUT = 'vix paper'

df     = pd.read_csv('data/features.csv', index_col=0, parse_dates=True)
th_csv = pd.read_csv('results/threshold_optimization.csv')

RAW_COLS   = ['VIX', 'VIX3M', 'SP500', 'GOLD', 'TNY', 'DXY',
              'FEDFUNDS', 'YIELD_CURVE', 'VIX9D', 'SKEW']
LABEL_COLS = (['LABEL'] +
              [f'LABEL_{n}' for n in [5, 10, 15, 20, 25]] +
              [f'LABEL_SUSTAINED_{n}' for n in [5, 10, 15, 20, 25]])
feature_cols = [c for c in df.columns if c not in RAW_COLS + LABEL_COLS]

N_VALUES   = [5, 10, 15, 20, 25]
TAU        = 0.25
VIX_THRESH = 20
tscv       = TimeSeriesSplit(n_splits=5)

# RF coverage targets from threshold_optimization.csv
rf_coverage = {}
for n in N_VALUES:
    row = th_csv[(th_csv['Model'] == 'Random Forest') & (th_csv['N'] == n) & (th_csv['Threshold'] == TAU)]
    rf_coverage[n] = row['Pct_Days'].values[0] / 100 if not row.empty else 0.5

# ── Fit RF + collect test probabilities for each N ────────────────────────────
print("Fitting calibrated RF models...")
proба_store  = {}   # proba_rf per N
y_test_store = {}
vix_test_store = {}

for n in N_VALUES:
    print(f"  N={n}...", end=' ', flush=True)
    label_col = f'LABEL_{n}'
    sub = df[feature_cols + [label_col]].dropna(subset=[label_col])

    X = sub[feature_cols].values
    y = sub[label_col].astype(int).values
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
    proba  = cal_rf.predict_proba(X_te)[:, 1]

    proба_store[n]    = proba
    y_test_store[n]   = y_te
    test_idx          = sub.index[split_idx:]
    vix_test_store[n] = df.loc[test_idx, 'VIX'].values
    print("done")

# ── PART 1: Calibration reliability diagrams ─────────────────────────────────
print("\nGenerating calibration diagrams...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    'Calibration Reliability Diagram: Random Forest (Test Set)\n'
    'Dashed line = perfect calibration; histogram = predicted probability density',
    fontsize=12
)

for ax, n in zip(axes, [5, 10]):
    proba  = proба_store[n]
    y_te   = y_test_store[n]
    brier  = brier_score_loss(y_te, proba)

    frac_pos, mean_pred = calibration_curve(y_te, proba, n_bins=10, strategy='quantile')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.2, alpha=0.5, label='Perfect calibration')
    ax.plot(mean_pred, frac_pos, 'o-', color='#2e7d32', linewidth=2.2, markersize=8,
            label=f'RF  (Brier = {brier:.3f})')

    ax2 = ax.twinx()
    ax2.hist(proba, bins=25, alpha=0.15, color='#2196F3', density=True)
    ax2.set_ylabel('Density', color='#5599cc', fontsize=9)
    ax2.tick_params(axis='y', labelcolor='#5599cc', labelsize=8)
    ax2.set_ylim(bottom=0)

    base_rate = y_te.mean()
    ax.axhline(base_rate, color='#e67e22', linestyle=':', linewidth=1.2, alpha=0.7,
               label=f'Test base rate ({base_rate:.1%})')

    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction Positive (Empirical)')
    ax.set_title(f'N = {n}  (test base rate: {base_rate:.1%})')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2)

plt.tight_layout()
out_cal = f'{OUT}/calibration_diagram.png'
plt.savefig(out_cal, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {out_cal}")
print("\nDone.")

