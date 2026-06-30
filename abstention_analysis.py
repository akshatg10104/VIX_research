"""
Abstention rate analysis: does the confidence filter abstain MORE in the
days immediately preceding calm→high transitions?

If yes: abstention itself is a leading indicator of regime change.
If no: the filter provides no warning signal whatsoever.

Outputs:
  results/abstention_before_transition.csv
  vix paper/abstention_analysis.png
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
N_EVAL   = [5, 10]   # N=5 and N=10 have enough calm→high events
LOOKBACK = 20        # inspect abstention rate in days 1..20 before the transition
tscv     = TimeSeriesSplit(n_splits=5)

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
    vix_tr, vix_te     = vix[:split_idx], vix[split_idx:]

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr_raw)
    X_te   = scaler.transform(X_te_raw)

    rf     = RandomForestClassifier(n_estimators=200, max_depth=10,
                                    min_samples_leaf=5, random_state=42)
    cal_rf = CalibratedClassifierCV(rf, cv=tscv, method='isotonic')
    cal_rf.fit(X_tr, y_tr)
    proba  = cal_rf.predict_proba(X_te)[:, 1]

    # abstained = True when model did NOT make a prediction
    abstained = (np.abs(proba - 0.5) <= TAU)

    # Identify test indices that are calm→high transitions
    # calm at t (VIX_te < 20) and label=1 (VIX_{t+n} >= 20)
    calm_to_high = np.where((vix_te < 20) & (y_te == 1))[0]
    calm_stable  = np.where((vix_te < 20) & (y_te == 0))[0]

    N_te = len(y_te)
    overall_abstain_rate = abstained.mean()

    print(f"\n  N={n}: {len(calm_to_high)} calm→high events, "
          f"overall abstention rate={overall_abstain_rate:.1%}")

    # For each lookback lag k, compute abstention rate among:
    # (a) days that are k days BEFORE a calm→high event
    # (b) days that are k days BEFORE a calm→calm day (control)
    for k in range(1, LOOKBACK + 1):
        # days that are k steps before a calm→high event
        target_indices_ch = calm_to_high - k
        valid_ch = target_indices_ch[(target_indices_ch >= 0) & (target_indices_ch < N_te)]
        abstain_rate_ch = abstained[valid_ch].mean() if len(valid_ch) > 0 else np.nan

        # control: k steps before calm→calm
        target_indices_cc = calm_stable - k
        valid_cc = target_indices_cc[(target_indices_cc >= 0) & (target_indices_cc < N_te)]
        abstain_rate_cc = abstained[valid_cc].mean() if len(valid_cc) > 0 else np.nan

        rows.append(dict(N=n, Lag=k,
                         Abstain_before_CH=round(abstain_rate_ch * 100, 1) if not np.isnan(abstain_rate_ch) else None,
                         Abstain_before_CC=round(abstain_rate_cc * 100, 1) if not np.isnan(abstain_rate_cc) else None,
                         Overall_Abstain=round(overall_abstain_rate * 100, 1),
                         N_CH_events=len(valid_ch),
                         N_CC_events=len(valid_cc)))

results = pd.DataFrame(rows)
results.to_csv('results/abstention_before_transition.csv', index=False)
print(f"\nSaved → results/abstention_before_transition.csv")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Abstention Rate in Days Preceding Regime Transitions\n'
             '(Higher abstention = model is more uncertain in pre-transition days)',
             fontsize=12)

for ax, n in zip(axes, N_EVAL):
    sub = results[results['N'] == n]
    overall = sub['Overall_Abstain'].iloc[0]
    ax.axhline(overall, color='gray', linestyle='--', linewidth=1.2,
               label=f'Overall abstention rate ({overall:.1f}%)', alpha=0.7)
    ax.plot(sub['Lag'], sub['Abstain_before_CH'], 'o-', color='#e74c3c',
            linewidth=2, markersize=6, label='Before calm→high spike')
    ax.plot(sub['Lag'], sub['Abstain_before_CC'], 's-', color='#2e7d32',
            linewidth=2, markersize=6, label='Before calm→calm (control)')
    ax.set_title(f'N = {n}')
    ax.set_xlabel('Days before transition')
    ax.set_ylabel('Abstention Rate (%)')
    ax.set_xlim(1, LOOKBACK)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(f'{OUT}/abstention_analysis.png', bbox_inches='tight')
plt.close()
print(f"Figure saved → {OUT}/abstention_analysis.png")
