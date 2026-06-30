"""
Persistence quantification.

Shows empirically that VIX autocorrelation explains nearly all of the RF's
covered accuracy, and quantifies the structural persistence in the data.

Outputs:
  results/persistence_quantify.csv
  vix paper/persistence_quantify.png
"""
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'legend.fontsize': 10, 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
})

df = pd.read_csv('data/features.csv', index_col=0, parse_dates=True)
RAW_COLS   = ['VIX', 'VIX3M', 'SP500', 'GOLD', 'TNY', 'DXY',
              'FEDFUNDS', 'YIELD_CURVE', 'VIX9D', 'SKEW']
LABEL_COLS = (['LABEL'] +
              [f'LABEL_{n}' for n in [5, 10, 15, 20, 25]] +
              [f'LABEL_SUSTAINED_{n}' for n in [5, 10, 15, 20, 25]])
feature_cols = [c for c in df.columns if c not in RAW_COLS + LABEL_COLS]

TAU    = 0.25
tscv   = TimeSeriesSplit(n_splits=5)
HORIZONS = [5, 10, 15, 20, 25]

rows = []

print(f"{'='*70}")
print("  PERSISTENCE QUANTIFICATION")
print(f"{'='*70}")
print(f"\n{'N':>4}  {'P(H|H)':>8}  {'P(H|C)':>8}  {'Persist ratio':>14}  "
      f"{'RF cov acc':>10}  {'Same-regime %':>14}  {'Diff-regime %':>14}")
print(f"  {'-'*80}")

for n in HORIZONS:
    label_col = f'LABEL_{n}'
    sub = df[feature_cols + [label_col, 'VIX']].dropna(subset=[label_col])
    split_idx = int(len(sub) * 0.8)

    sub_tr = sub.iloc[:split_idx]
    sub_te = sub.iloc[split_idx:]

    X_tr = sub_tr[feature_cols].values
    y_tr = sub_tr[label_col].astype(int).values
    X_te = sub_te[feature_cols].values
    y_te = sub_te[label_col].astype(int).values
    vix_te = sub_te['VIX'].values

    # ── Empirical transition probabilities (from full dataset) ─────────────
    vix_all   = sub['VIX'].values
    label_all = sub[label_col].astype(int).values
    high_today = (vix_all >= 20)
    calm_today = ~high_today

    p_high_given_high = label_all[high_today].mean()
    p_high_given_calm = label_all[calm_today].mean()

    # ── RF selective predictions on test set ───────────────────────────────
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    rf  = RandomForestClassifier(n_estimators=200, max_depth=10,
                                  min_samples_leaf=5, random_state=42)
    cal = CalibratedClassifierCV(rf, cv=tscv, method='isotonic')
    cal.fit(X_tr_s, y_tr)
    proba  = cal.predict_proba(X_te_s)[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    sel    = (proba > 0.5 + TAU) | (proba < 0.5 - TAU)

    rf_acc = (y_pred[sel] == y_te[sel]).mean() * 100

    # ── Decompose RF correct predictions ──────────────────────────────────
    # "Same regime" = model predicts same as VIX_t regime (persistence call)
    persist_pred = (vix_te >= 20).astype(int)
    rf_correct   = sel & (y_pred == y_te)
    # Days where RF was right AND it predicted the same regime as VIX_t
    same_regime_right  = rf_correct & (y_pred == persist_pred)
    diff_regime_right  = rf_correct & (y_pred != persist_pred)

    n_rf_correct = rf_correct.sum()
    same_pct = same_regime_right.sum() / max(n_rf_correct, 1) * 100
    diff_pct = diff_regime_right.sum() / max(n_rf_correct, 1) * 100

    print(f"  N={n:2d}  {p_high_given_high*100:>7.1f}%  "
          f"{p_high_given_calm*100:>7.1f}%  "
          f"{p_high_given_high/max(p_high_given_calm,1e-9):>13.1f}x  "
          f"{rf_acc:>9.1f}%  "
          f"{same_pct:>13.1f}%  "
          f"{diff_pct:>13.1f}%")

    rows.append(dict(
        N=n,
        P_high_given_high=round(p_high_given_high * 100, 1),
        P_high_given_calm=round(p_high_given_calm * 100, 1),
        Persistence_ratio=round(p_high_given_high / max(p_high_given_calm, 1e-9), 2),
        RF_covered_acc=round(rf_acc, 1),
        Pct_correct_same_regime=round(same_pct, 1),
        Pct_correct_diff_regime=round(diff_pct, 1),
    ))

results = pd.DataFrame(rows)
results.to_csv('results/persistence_quantify.csv', index=False)
print("\nSaved → results/persistence_quantify.csv")

# ── Figure ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('VIX Persistence Quantification: Why the RF Mostly Learns Autocorrelation',
             fontsize=13, fontweight='bold')

ns = results['N'].values

# Panel 1: Transition probabilities
ax = axes[0]
ax.plot(ns, results['P_high_given_high'], 'o-', color='#e74c3c', linewidth=2,
        markersize=8, label='P(high | VIX≥20 today)')
ax.plot(ns, results['P_high_given_calm'], 's--', color='#3498db', linewidth=2,
        markersize=8, label='P(high | VIX<20 today)')
ax.set_xlabel('Prediction horizon N (days)')
ax.set_ylabel('Probability (%)')
ax.set_title('Empirical Transition Probabilities')
ax.set_xticks(ns)
ax.legend()
ax.grid(True, alpha=0.2)
ax.set_ylim(0, 100)

# Panel 2: Persistence ratio
ax = axes[1]
bars = ax.bar(ns, results['Persistence_ratio'], color='#9b59b6', alpha=0.8,
              edgecolor='white', linewidth=0.5)
for bar, v in zip(bars, results['Persistence_ratio']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{v:.1f}x', ha='center', va='bottom', fontsize=10)
ax.set_xlabel('Prediction horizon N (days)')
ax.set_ylabel('P(high|high) / P(high|calm)')
ax.set_title('Persistence Ratio\n(how much more likely high→high vs calm→high)')
ax.set_xticks(ns)
ax.grid(True, alpha=0.2, axis='y')

# Panel 3: Decomposition of RF correct predictions
ax = axes[2]
x = np.arange(len(ns))
w = 0.35
b1 = ax.bar(x - w/2, results['Pct_correct_same_regime'], w,
            label='Predicted same regime as VIX today\n(persistence-consistent)', color='#27ae60', alpha=0.85)
b2 = ax.bar(x + w/2, results['Pct_correct_diff_regime'], w,
            label='Predicted regime change correctly', color='#f39c12', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f'N={n}' for n in ns])
ax.set_ylabel('% of RF correct covered predictions')
ax.set_title('RF Correct Predictions:\nSame-regime vs regime-change calls')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, axis='y')
ax.set_ylim(0, 110)

plt.tight_layout()
plt.savefig('vix paper/persistence_quantify.png', bbox_inches='tight')
plt.close()
print("Saved → vix paper/persistence_quantify.png")
print("\n=== Summary ===")
print(results.to_string(index=False))
