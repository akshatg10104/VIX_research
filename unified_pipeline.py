"""
Unified results pipeline — one fixed spec for ALL tables.
RF:  n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42
HGB: max_iter=200, max_depth=5, learning_rate=0.1, random_state=42
Both calibrated with isotonic via TimeSeriesSplit(n_splits=5)

Outputs:
  results/unified_model_results.csv   — Table 2 with extended metrics
  vix paper/tuned_model_comparison.png
"""
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, balanced_accuracy_score)
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
N_VALUES = [5, 10, 15, 20, 25]
tscv     = TimeSeriesSplit(n_splits=5)

MODELS = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10,
                                            min_samples_leaf=5, random_state=42),
    'XGBoost':       HistGradientBoostingClassifier(max_iter=200, max_depth=5,
                                                    learning_rate=0.1, random_state=42),
}

rows = []
print("Fitting unified pipeline (fixed-spec calibrated models)...")

for n in N_VALUES:
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

    for model_name, model_spec in MODELS.items():
        print(f"  N={n}, {model_name}...", end=' ', flush=True)

        # Baseline: uncalibrated, predict on all days
        from sklearn.base import clone
        base_model = clone(model_spec)
        base_model.fit(X_tr, y_tr)
        y_pred_base = base_model.predict(X_te)
        base_acc = accuracy_score(y_te, y_pred_base) * 100

        # Selective: calibrated, abstain when |p - 0.5| <= tau
        cal_model = CalibratedClassifierCV(clone(model_spec), cv=tscv, method='isotonic')
        cal_model.fit(X_tr, y_tr)
        proba = cal_model.predict_proba(X_te)[:, 1]
        y_pred_cal = (proba >= 0.5).astype(int)

        mask    = (proba > 0.5 + TAU) | (proba < 0.5 - TAU)
        n_sel   = mask.sum()
        cov     = n_sel / len(y_te) * 100
        y_m     = y_te[mask]
        p_m     = y_pred_cal[mask]

        if n_sel >= 10:
            sel_acc  = accuracy_score(y_m, p_m) * 100
            prec     = precision_score(y_m, p_m, zero_division=0) * 100
            rec      = recall_score(y_m, p_m, zero_division=0) * 100
            f1       = f1_score(y_m, p_m, zero_division=0) * 100
            bal_acc  = balanced_accuracy_score(y_m, p_m) * 100
        else:
            sel_acc = prec = rec = f1 = bal_acc = float('nan')

        gain = sel_acc - base_acc if not np.isnan(sel_acc) else float('nan')
        print(f"base={base_acc:.1f}%, sel={sel_acc:.1f}%, cov={cov:.1f}%")

        rows.append(dict(
            N=n, Model=model_name,
            Base_Acc=round(base_acc, 2),
            Sel_Acc=round(sel_acc, 2) if not np.isnan(sel_acc) else None,
            Gain=round(gain, 2) if not np.isnan(gain) else None,
            Coverage=round(cov, 1),
            Precision=round(prec, 2) if not np.isnan(prec) else None,
            Recall=round(rec, 2) if not np.isnan(rec) else None,
            F1=round(f1, 2) if not np.isnan(f1) else None,
            Bal_Acc=round(bal_acc, 2) if not np.isnan(bal_acc) else None,
        ))

results = pd.DataFrame(rows)
results.to_csv('results/unified_model_results.csv', index=False)
print(f"\nSaved → results/unified_model_results.csv")
print(results.to_string(index=False))

# ── Figure: baseline vs selective accuracy ────────────────────────────────────
DISPLAY = {'Random Forest': 'Random Forest', 'XGBoost': 'Hist. Grad. Boosting (HGB)'}
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
fig.suptitle('Accuracy vs Prediction Horizon: Baseline and Selective Predicting '
             '(fixed-spec calibrated pipeline)', fontsize=13)

for ax, model_name in zip(axes, ['Random Forest', 'XGBoost']):
    sub = results[results['Model'] == model_name].sort_values('N')
    ax.plot(sub['N'], sub['Base_Acc'], '--', color='#aaaaaa', linewidth=1.8,
            label='Baseline (uncalibrated)')
    ax.plot(sub['N'], sub['Sel_Acc'], '-o', color='#2e7d32', linewidth=2.2,
            markersize=7, label=f'Selective (τ={TAU})')
    ax.plot(sub['N'], sub['Bal_Acc'], '-s', color='#e67e22', linewidth=1.8,
            markersize=6, linestyle='-.', label='Balanced acc (covered days)')
    ax.set_title(DISPLAY.get(model_name, model_name))
    ax.set_xlabel('N (trading days ahead)')
    ax.set_xticks(N_VALUES)
    ax.set_ylim(50, 100)
    ax.grid(True, alpha=0.2)
    ax.legend(framealpha=0.9)

axes[0].set_ylabel('Accuracy (%)')
plt.tight_layout()
plt.savefig(f'{OUT}/tuned_model_comparison.png', bbox_inches='tight')
plt.close()
print(f"Figure saved → {OUT}/tuned_model_comparison.png")
