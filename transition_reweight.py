"""
Transition-reweighted RF experiment.

Tests whether explicit cost-weighting (class_weight={0:1, 1:C}) rescues
calm→high recall, or whether the blindness is informational not
objective-function-driven.

Outputs:
  results/transition_reweight.csv   — transition matrix per weight
  results/reweight_utility.csv      — cost-weighted utility with new column
  vix paper/transition_reweight.png — transition recall vs cost weight figure
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
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'legend.fontsize': 10, 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
})

os.makedirs('vix paper', exist_ok=True)

df = pd.read_csv('data/features.csv', index_col=0, parse_dates=True)
RAW_COLS   = ['VIX', 'VIX3M', 'SP500', 'GOLD', 'TNY', 'DXY',
              'FEDFUNDS', 'YIELD_CURVE', 'VIX9D', 'SKEW']
LABEL_COLS = (['LABEL'] +
              [f'LABEL_{n}' for n in [5, 10, 15, 20, 25]] +
              [f'LABEL_SUSTAINED_{n}' for n in [5, 10, 15, 20, 25]])
feature_cols = [c for c in df.columns if c not in RAW_COLS + LABEL_COLS]

TAU      = 0.25
N_EVAL   = [5, 10]
tscv     = TimeSeriesSplit(n_splits=5)
# Cost weights to test: 1 = unweighted, then escalating FN costs
WEIGHTS  = [1, 5, 10, 20]

trans_rows   = []
utility_rows = []

print(f"{'='*70}")
print("  TRANSITION-REWEIGHTED RF EXPERIMENT")
print(f"{'='*70}")

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

    print(f"\n  N={n}  (test: {len(y_te)} days, {y_te.sum()} high-vol days)")
    print(f"  {'Weight C':>10}  {'CH_cov':>8}  {'CH_acc':>8}  "
          f"{'CC_cov':>8}  {'CC_acc':>8}  {'Sel_acc':>8}  {'Coverage':>9}")
    print(f"  {'-'*72}")

    # Identify regime transition types in test set
    calm_mask_te  = (vix_te < 20)
    high_mask_te  = (vix_te >= 20)

    for cw in WEIGHTS:
        class_weight = {0: 1, 1: cw}
        rf  = RandomForestClassifier(n_estimators=200, max_depth=10,
                                     min_samples_leaf=5, random_state=42,
                                     class_weight=class_weight)
        cal = CalibratedClassifierCV(rf, cv=tscv, method='isotonic')
        cal.fit(X_tr, y_tr)
        proba  = cal.predict_proba(X_te)[:, 1]
        y_pred = (proba >= 0.5).astype(int)

        # Selective predictions
        sel_mask = (proba > 0.5 + TAU) | (proba < 0.5 - TAU)
        sel_acc  = accuracy_score(y_te[sel_mask], y_pred[sel_mask]) if sel_mask.sum() >= 10 else np.nan
        coverage = sel_mask.mean() * 100

        # Transition breakdown on covered days
        # calm→high: VIX_t < 20 AND y_te == 1
        ch_mask = calm_mask_te & (y_te == 1)
        cc_mask = calm_mask_te & (y_te == 0)

        ch_covered = ch_mask & sel_mask
        cc_covered = cc_mask & sel_mask

        ch_cov  = ch_covered.sum() / max(ch_mask.sum(), 1) * 100
        ch_acc  = accuracy_score(y_te[ch_covered], y_pred[ch_covered]) * 100 if ch_covered.sum() >= 3 else np.nan
        cc_cov  = cc_covered.sum() / max(cc_mask.sum(), 1) * 100
        cc_acc  = accuracy_score(y_te[cc_covered], y_pred[cc_covered]) * 100 if cc_covered.sum() >= 3 else np.nan

        print(f"  C={cw:>4} (wt)  {ch_cov:>7.1f}%  "
              f"{'---' if np.isnan(ch_acc) else f'{ch_acc:>6.1f}%':>8}  "
              f"{cc_cov:>7.1f}%  "
              f"{'---' if np.isnan(cc_acc) else f'{cc_acc:>6.1f}%':>8}  "
              f"{'---' if np.isnan(sel_acc) else f'{sel_acc*100:>6.1f}%':>8}  "
              f"{coverage:>8.1f}%")

        # Cost-weighted utility for each C_cost ratio
        for c_cost in [1, 5, 10, 20]:
            if sel_mask.sum() < 10:
                utility = np.nan
            else:
                FP = ((y_pred[sel_mask] == 1) & (y_te[sel_mask] == 0)).sum()
                FN = ((y_pred[sel_mask] == 0) & (y_te[sel_mask] == 1)).sum()
                utility = -(FP + c_cost * FN) / sel_mask.sum()
            utility_rows.append(dict(N=n, ClassWeight=cw, CostRatio=c_cost,
                                     Utility=round(utility, 3) if not np.isnan(utility) else np.nan,
                                     Coverage=round(coverage, 1)))

        trans_rows.append(dict(
            N=n, ClassWeight=cw,
            CH_Coverage=round(ch_cov, 1),
            CH_Accuracy=round(ch_acc, 1) if not np.isnan(ch_acc) else np.nan,
            CC_Coverage=round(cc_cov, 1),
            CC_Accuracy=round(cc_acc, 1) if not np.isnan(cc_acc) else np.nan,
            Sel_Accuracy=round(sel_acc * 100, 1) if not np.isnan(sel_acc) else np.nan,
            Coverage=round(coverage, 1),
        ))

trans_df   = pd.DataFrame(trans_rows)
utility_df = pd.DataFrame(utility_rows)
trans_df.to_csv('results/transition_reweight.csv', index=False)
utility_df.to_csv('results/reweight_utility.csv', index=False)
print("\nSaved → results/transition_reweight.csv")
print("Saved → results/reweight_utility.csv")

# ── Figure: calm→high recall vs class weight ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Effect of Cost-Weighting on Calm→High Transition Recall', fontsize=13)

for ax, n in zip(axes, N_EVAL):
    sub = trans_df[trans_df['N'] == n]
    ax.plot(sub['ClassWeight'], sub['CH_Accuracy'].fillna(0),
            's-', color='#e74c3c', linewidth=2, markersize=8, label='Calm→High accuracy (covered)')
    ax.plot(sub['ClassWeight'], sub['CC_Accuracy'],
            'o--', color='#3498db', linewidth=2, markersize=8, label='Calm→Calm accuracy (covered)')
    ax.plot(sub['ClassWeight'], sub['Sel_Accuracy'],
            '^:', color='#2ecc71', linewidth=2, markersize=8, label='Overall selective accuracy')
    ax.axhline(0, color='red', linewidth=0.8, linestyle=':', alpha=0.5)
    ax.set_xlabel('Class weight C (on calm→high / high-vol days)')
    ax.set_ylabel('Accuracy on covered days (%)')
    ax.set_title(f'N = {n}')
    ax.set_xticks(WEIGHTS)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(-5, 105)

plt.tight_layout()
plt.savefig('vix paper/transition_reweight.png', bbox_inches='tight')
plt.close()
print("Saved → vix paper/transition_reweight.png")

# ── Print utility comparison table ───────────────────────────────────────────
print("\n=== Cost-Weighted Utility by Class Weight (N=10) ===")
n10 = utility_df[utility_df['N'] == 10]
pivot = n10.pivot_table(index='CostRatio', columns='ClassWeight', values='Utility')
print(pivot.to_string())
