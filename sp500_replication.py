"""
Replication: S&P 500 trend-regime classification.

Tests whether the persistence illusion is specific to VIX or is a general
property of selective prediction on any autocorrelated regime label.

Label: SP500_{t+N} is below its 200-day moving average at t+N
       (i.e., the market is in a downtrend in N days).
       BEAR_N_t = 1  if SP500_{t+N} < MA200_{t+N}
                  0  if SP500_{t+N} >= MA200_{t+N}

Same features (36), same RF spec, same tau=0.25, same baselines.
HAR-analog: SP500 today, 5-day return, 20-day return.

Outputs:
  results/sp500_replication_baselines.csv
  results/sp500_replication_transitions.csv
  results/sp500_persist_quant.csv
  vix paper/sp500_replication.png
"""
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.base import clone

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'legend.fontsize': 10, 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
})

TAU  = 0.25
tscv = TimeSeriesSplit(n_splits=5)

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv('data/features.csv', index_col=0, parse_dates=True)
RAW_COLS   = ['VIX', 'VIX3M', 'SP500', 'GOLD', 'TNY', 'DXY',
              'FEDFUNDS', 'YIELD_CURVE', 'VIX9D', 'SKEW']
LABEL_COLS = (['LABEL'] +
              [f'LABEL_{n}' for n in [5, 10, 15, 20, 25]] +
              [f'LABEL_SUSTAINED_{n}' for n in [5, 10, 15, 20, 25]])
feature_cols = [c for c in df.columns if c not in RAW_COLS + LABEL_COLS]

print(f"{'='*70}")
print("  S&P 500 TREND-REGIME REPLICATION")
print(f"{'='*70}")
print("  Label: SP500_{t+N} below its 200-day moving average at t+N")
print(f"  Features: same 36 as VIX analysis  |  tau = {TAU}")

# ── Construct S&P 500 bear-market label ────────────────────────────────────
sp500       = df['SP500'].copy()
sp500_ma200 = sp500.rolling(200, min_periods=150).mean()
bear_today  = (sp500 < sp500_ma200).astype(int)   # regime at t

# Build BEAR_N labels: shift bear_today back N days
# bear_today.shift(-N) = whether bear at t+N
for n in [5, 10, 15, 20, 25]:
    df[f'BEAR_{n}'] = bear_today.shift(-n)

base_rows  = []
trans_rows = []
pq_rows    = []

for n in [5, 10, 15, 20, 25]:
    label_col = f'BEAR_{n}'
    sub = df[feature_cols + [label_col, 'SP500']].dropna(subset=[label_col])
    # Drop rows where SP500 features might be NaN
    sub = sub.dropna()
    split_idx = int(len(sub) * 0.8)

    sub_tr = sub.iloc[:split_idx]
    sub_te = sub.iloc[split_idx:]

    X_tr = sub_tr[feature_cols].values
    y_tr = sub_tr[label_col].astype(int).values
    X_te = sub_te[feature_cols].values
    y_te = sub_te[label_col].astype(int).values
    sp_te = sub_te['SP500'].values

    # Bear regime at time t (for persistence baseline)
    sp500_sub     = sp500.loc[sub.index]
    ma200_sub     = sp500_ma200.loc[sub.index]
    bear_regime_t = (sp500_sub < ma200_sub).astype(int).values
    bear_te       = bear_regime_t[split_idx:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    base_rate = y_te.mean() * 100
    print(f"\n  N={n}  test={len(y_te)} days  bear base rate={base_rate:.1f}%  "
          f"bear_today in test={bear_te.mean()*100:.1f}%")

    # ── RF ────────────────────────────────────────────────────────────────
    rf  = RandomForestClassifier(n_estimators=200, max_depth=10,
                                  min_samples_leaf=5, random_state=42)
    cal = CalibratedClassifierCV(rf, cv=tscv, method='isotonic')
    cal.fit(X_tr_s, y_tr)
    proba  = cal.predict_proba(X_te_s)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    sel_mask = (proba > 0.5 + TAU) | (proba < 0.5 - TAU)
    n_sel    = sel_mask.sum()
    rf_sel   = accuracy_score(y_te[sel_mask], y_pred[sel_mask]) * 100 if n_sel >= 10 else np.nan
    rf_bal   = balanced_accuracy_score(y_te[sel_mask], y_pred[sel_mask]) * 100 if n_sel >= 10 else np.nan
    rf_cov   = sel_mask.mean() * 100

    # RF training coverage rate — baseline coverage matching is calibrated on
    # training data only, then frozen
    proba_tr_rf = cal.predict_proba(X_tr_s)[:, 1]
    cov_rf_tr   = ((proba_tr_rf > 0.5 + TAU) | (proba_tr_rf < 0.5 - TAU)).mean()

    # ── Persistence baseline (c calibrated on training window, frozen) ─────
    # distance measured as a fraction of MA200 so the threshold is scale-free
    # across the 2006–2026 index-level range
    y_pers   = bear_te.copy()
    dist     = np.abs(sp500_sub.iloc[split_idx:].values / ma200_sub.iloc[split_idx:].values - 1)
    dist_tr  = np.abs(sp500_sub.iloc[:split_idx].values / ma200_sub.iloc[:split_idx].values - 1)
    c_thresh = np.nanquantile(dist_tr, 1 - cov_rf_tr)
    p_mask   = dist >= c_thresh
    pers_acc = accuracy_score(y_te[p_mask], y_pers[p_mask]) * 100 if p_mask.sum() >= 10 else np.nan

    # ── HAR-analog LR ─────────────────────────────────────────────────────
    # Features: SP500 level, 5d MA, 20d MA — compute from sub's SP500 column
    sp0_series = sub['SP500']
    sp5_series  = sp0_series.rolling(5,  min_periods=1).mean()
    sp20_series = sp0_series.rolling(20, min_periods=1).mean()
    sp0  = sp0_series.values
    sp5  = sp5_series.values
    sp20 = sp20_series.values

    X_har_tr = np.column_stack([sp0[:split_idx], sp5[:split_idx], sp20[:split_idx]])
    X_har_te = np.column_stack([sp0[split_idx:], sp5[split_idx:], sp20[split_idx:]])
    sc_har   = StandardScaler()
    X_har_tr_s = sc_har.fit_transform(X_har_tr)
    X_har_te_s = sc_har.transform(X_har_te)

    lr      = LogisticRegression(max_iter=1000, random_state=42)
    cal_har = CalibratedClassifierCV(clone(lr), cv=tscv, method='isotonic')
    cal_har.fit(X_har_tr_s, y_tr)
    proba_har  = cal_har.predict_proba(X_har_te_s)[:, 1]
    y_pred_har = (proba_har >= 0.5).astype(int)

    # Match HAR to RF coverage — threshold from TRAINING probabilities, frozen
    proba_har_tr = cal_har.predict_proba(X_har_tr_s)[:, 1]
    thresh_har   = np.quantile(np.abs(proba_har_tr - 0.5), 1 - cov_rf_tr)
    mask_har     = np.abs(proba_har - 0.5) >= thresh_har
    har_acc    = accuracy_score(y_te[mask_har], y_pred_har[mask_har]) * 100 if mask_har.sum() >= 10 else np.nan

    rf_minus_persist = (rf_sel - pers_acc) if not (np.isnan(rf_sel) or np.isnan(pers_acc)) else np.nan
    print(f"    Persist={pers_acc:.1f}%  HAR={har_acc:.1f}%  RF_sel={rf_sel:.1f}%  "
          f"BalAcc={rf_bal:.1f}%  Cov={rf_cov:.1f}%  RF-Persist={rf_minus_persist:+.2f}pp")

    base_rows.append(dict(N=n, BaseRate=round(base_rate, 1),
                          Coverage=round(rf_cov, 1),
                          Persistence=round(pers_acc, 1) if not np.isnan(pers_acc) else np.nan,
                          HAR_LR=round(har_acc, 1) if not np.isnan(har_acc) else np.nan,
                          RF_Sel=round(rf_sel, 1) if not np.isnan(rf_sel) else np.nan,
                          RF_BalAcc=round(rf_bal, 1) if not np.isnan(rf_bal) else np.nan,
                          RF_minus_Persist=round(rf_minus_persist, 2) if not np.isnan(rf_minus_persist) else np.nan))

    # ── Persistence quantification ─────────────────────────────────────────
    # P(bear in N days | bear today) and P(bear in N days | bull today)
    label_all = df[label_col].dropna().astype(int)
    bear_all  = (sp500 < sp500_ma200).astype(int)
    aligned   = pd.concat([label_all, bear_all.rename('bear_today')], axis=1).dropna()
    p_bear_given_bear = aligned.loc[aligned['bear_today'] == 1, label_col].mean()
    p_bear_given_bull = aligned.loc[aligned['bear_today'] == 0, label_col].mean()

    # Decompose RF correct predictions
    rf_correct       = sel_mask & (y_pred == y_te)
    same_regime_right = rf_correct & (y_pred == bear_te)
    n_rf_correct      = rf_correct.sum()
    same_pct          = same_regime_right.sum() / max(n_rf_correct, 1) * 100

    pq_rows.append(dict(N=n,
        P_bear_given_bear=round(p_bear_given_bear * 100, 1),
        P_bear_given_bull=round(p_bear_given_bull * 100, 1),
        Persist_ratio=round(p_bear_given_bear / max(p_bear_given_bull, 1e-9), 2),
        RF_covered_acc=round(rf_sel, 1) if not np.isnan(rf_sel) else np.nan,
        Same_regime_pct=round(same_pct, 1)))

    print(f"    P(bear|bear)={p_bear_given_bear*100:.1f}%  "
          f"P(bear|bull)={p_bear_given_bull*100:.1f}%  "
          f"ratio={p_bear_given_bear/max(p_bear_given_bull,1e-9):.1f}x  "
          f"same-regime={same_pct:.1f}%")

    # ── Transition analysis (N=5 and N=10 only) ───────────────────────────
    if n in [5, 10]:
        bull_mask = (bear_te == 0)
        bear_mask = (bear_te == 1)
        for trans_name, t_mask in [
            ('bull_bull', bull_mask & (y_te == 0)),
            ('bull_bear', bull_mask & (y_te == 1)),
            ('bear_bull', bear_mask & (y_te == 0)),
            ('bear_bear', bear_mask & (y_te == 1)),
        ]:
            covered = t_mask & sel_mask
            cov_pct = covered.sum() / max(t_mask.sum(), 1) * 100
            acc     = accuracy_score(y_te[covered], y_pred[covered]) * 100 \
                      if covered.sum() >= 3 else np.nan
            trans_rows.append(dict(N=n, Transition=trans_name,
                                   N_days=int(t_mask.sum()),
                                   Coverage_pct=round(cov_pct, 1),
                                   Accuracy=round(acc, 1) if not np.isnan(acc) else np.nan))
            print(f"      {trans_name:<12}  days={t_mask.sum():3d}  "
                  f"cov={cov_pct:5.1f}%  "
                  f"acc={'---' if np.isnan(acc) else f'{acc:.1f}%':>7}")

# ── Save ──────────────────────────────────────────────────────────────────
pd.DataFrame(base_rows).to_csv('results/sp500_replication_baselines.csv', index=False)
pd.DataFrame(trans_rows).to_csv('results/sp500_replication_transitions.csv', index=False)
pd.DataFrame(pq_rows).to_csv('results/sp500_persist_quant.csv', index=False)
print("\nSaved results")

base_df  = pd.DataFrame(base_rows)
trans_df = pd.DataFrame(trans_rows)
pq_df    = pd.DataFrame(pq_rows)

# ── Figure ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Replication: S\&P 500 Trend-Regime Classification\n'
             '(Label: SP500 below 200-day MA in N days; same 36 features, same $\\tau=0.25$)',
             fontsize=12, fontweight='bold')

ns = base_df['N'].values

# Panel 1: RF vs persistence vs HAR
ax = axes[0]
ax.plot(ns, base_df['RF_Sel'],     'o-',  color='#2c3e50', lw=2, ms=8, label='RF selective')
ax.plot(ns, base_df['Persistence'],'s--', color='#e74c3c', lw=2, ms=8, label='Persistence (matched cov)')
ax.plot(ns, base_df['HAR_LR'],     '^:',  color='#27ae60', lw=2, ms=8, label='HAR-analog LR')
ax.set_xlabel('Horizon N (days)'); ax.set_ylabel('Accuracy on covered days (%)')
ax.set_title('RF vs Persistence vs HAR-Analog\n(same pattern as VIX)')
ax.set_xticks(ns); ax.legend(); ax.grid(True, alpha=0.2); ax.set_ylim(50, 100)

# Panel 2: Persistence ratio
ax = axes[1]
bars = ax.bar(ns, pq_df['Persist_ratio'], color='#8e44ad', alpha=0.8, edgecolor='white')
for bar, v in zip(bars, pq_df['Persist_ratio']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{v:.1f}x', ha='center', va='bottom', fontsize=10)
ax.set_xlabel('Horizon N (days)'); ax.set_ylabel('P(bear|bear) / P(bear|bull)')
ax.set_title('Persistence Ratio\n(SP500 trend autocorrelation)')
ax.set_xticks(ns); ax.grid(True, alpha=0.2, axis='y')

# Panel 3: Transition accuracy at N=5 and N=10
ax = axes[2]
trans_labels = ['bull→bull', 'bull→bear', 'bear→bull', 'bear→bear']
x = np.arange(len(trans_labels))
w = 0.35

for i, n_val in enumerate([5, 10]):
    sub = trans_df[trans_df['N'] == n_val].set_index('Transition')
    accs = []
    for t in ['bull_bull', 'bull_bear', 'bear_bull', 'bear_bear']:
        a = sub.loc[t, 'Accuracy'] if t in sub.index else np.nan
        accs.append(a if not (isinstance(a, float) and np.isnan(a)) else 0)

    offset = -w/2 if i == 0 else w/2
    color  = '#3498db' if i == 0 else '#e74c3c'
    bars   = ax.bar(x + offset, accs, w, label=f'N={n_val}', color=color, alpha=0.85)

ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='Random chance (50%)')
ax.set_xticks(x); ax.set_xticklabels(trans_labels, fontsize=9)
ax.set_ylabel('Accuracy on covered days (%)')
ax.set_title('Transition Accuracy\n(bull→bear mirrors VIX calm→high: near-0%)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.2, axis='y'); ax.set_ylim(0, 105)

plt.tight_layout()
plt.savefig('vix paper/sp500_replication.png', bbox_inches='tight')
plt.close()
print("Saved → vix paper/sp500_replication.png")

print("\n=== Baseline Summary ===")
print(base_df.to_string(index=False))
print("\n=== Persistence Quantification ===")
print(pq_df.to_string(index=False))
print("\n=== Transition Matrix (N=5, N=10) ===")
print(trans_df.to_string(index=False))
