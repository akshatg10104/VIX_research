"""
Extended baseline comparison:
  - Persistence (coverage-matched to RF)
  - Logistic regression on VIX level only
  - HAR-style logistic regression: VIX_D (current), VIX_W (5-day mean), VIX_M (20-day mean)
  - Markov-chain transition probability baseline
  - Fixed-spec calibrated RF

All evaluated on the same test set with matched coverage.
Outputs:
  results/extended_baselines.csv
  vix paper/persistence_baseline.png  (updated with HAR and Markov columns)
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
from sklearn.linear_model import LogisticRegression
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

OUT = 'vix paper'
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv('data/features.csv', index_col=0, parse_dates=True)
RAW_COLS   = ['VIX', 'VIX3M', 'SP500', 'GOLD', 'TNY', 'DXY',
              'FEDFUNDS', 'YIELD_CURVE', 'VIX9D', 'SKEW']
LABEL_COLS = (['LABEL'] +
              [f'LABEL_{n}' for n in [5, 10, 15, 20, 25]] +
              [f'LABEL_SUSTAINED_{n}' for n in [5, 10, 15, 20, 25]])
feature_cols = [c for c in df.columns if c not in RAW_COLS + LABEL_COLS]

# Pre-compute HAR features on the full series
df['VIX_W']  = df['VIX'].rolling(5,  min_periods=5).mean()   # 5-day mean
df['VIX_M']  = df['VIX'].rolling(20, min_periods=20).mean()  # 20-day mean
har_cols = ['VIX', 'VIX_W', 'VIX_M']

TAU      = 0.25
N_VALUES = [5, 10, 15, 20, 25]
tscv     = TimeSeriesSplit(n_splits=5)

def find_persist_c(vix_vals, n_target):
    """Binary search for c such that |VIX - 20| >= c gives ~n_target covered days."""
    lo, hi = 0.0, 25.0
    for _ in range(60):
        mid = (lo + hi) / 2
        n_c = int(np.sum(np.abs(vix_vals - 20) >= mid))
        if n_c > n_target:
            lo = mid
        else:
            hi = mid
    return mid

def block_bootstrap_gap(y_true, acc_rf, acc_base, n_bootstrap=2000, block_len=30):
    """Bootstrap CI on (acc_rf - acc_base) using paired block resampling."""
    n = len(y_true)
    diffs = []
    for _ in range(n_bootstrap):
        starts = np.random.randint(0, max(1, n - block_len + 1),
                                   size=int(np.ceil(n / block_len)))
        idx = np.concatenate([np.arange(s, min(s + block_len, n)) for s in starts])[:n]
        diffs.append(acc_rf - acc_base)
    diffs = np.array(diffs)
    return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)

rows = []
print("Computing extended baselines...")

for n in N_VALUES:
    label_col = f'LABEL_{n}'
    sub = df[feature_cols + har_cols + [label_col]].dropna(subset=[label_col] + har_cols)
    X_full = sub[feature_cols].values
    X_har  = sub[har_cols].values
    y      = sub[label_col].astype(int).values
    vix_v  = sub['VIX'].values

    split_idx = int(len(sub) * 0.8)
    X_tr, X_te         = X_full[:split_idx], X_full[split_idx:]
    X_har_tr, X_har_te = X_har[:split_idx], X_har[split_idx:]
    y_tr, y_te         = y[:split_idx], y[split_idx:]
    vix_te             = vix_v[split_idx:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    scaler_har = StandardScaler()
    X_har_tr_s = scaler_har.fit_transform(X_har_tr)
    X_har_te_s = scaler_har.transform(X_har_te)

    # ── RF (fixed-spec calibrated) ────────────────────────────────────────────
    rf     = RandomForestClassifier(n_estimators=200, max_depth=10,
                                    min_samples_leaf=5, random_state=42)
    cal_rf = CalibratedClassifierCV(rf, cv=tscv, method='isotonic')
    cal_rf.fit(X_tr_s, y_tr)
    proba_rf = cal_rf.predict_proba(X_te_s)[:, 1]
    mask_rf  = (proba_rf > 0.5 + TAU) | (proba_rf < 0.5 - TAU)
    n_rf     = int(mask_rf.sum())
    cov_rf   = n_rf / len(y_te) * 100
    y_pred_rf = (proba_rf >= 0.5).astype(int)
    rf_acc   = accuracy_score(y_te[mask_rf], y_pred_rf[mask_rf]) * 100 if n_rf >= 10 else np.nan

    # ── Persistence ───────────────────────────────────────────────────────────
    c = find_persist_c(vix_te, n_rf)
    mask_per  = np.abs(vix_te - 20) >= c
    y_per     = (vix_te >= 20).astype(int)
    per_acc   = accuracy_score(y_te[mask_per], y_per[mask_per]) * 100 if mask_per.sum() >= 10 else np.nan

    # ── LR (VIX only) ─────────────────────────────────────────────────────────
    scaler_vix = StandardScaler()
    vix_tr_s   = scaler_vix.fit_transform(vix_v[:split_idx].reshape(-1,1))
    vix_te_s   = scaler_vix.transform(vix_te.reshape(-1,1))
    lr_vix     = LogisticRegression(max_iter=1000, random_state=42)
    cal_vix    = CalibratedClassifierCV(lr_vix, cv=tscv, method='isotonic')
    cal_vix.fit(vix_tr_s, y_tr)
    proba_vix  = cal_vix.predict_proba(vix_te_s)[:, 1]
    mask_vix   = (proba_vix > 0.5 + TAU) | (proba_vix < 0.5 - TAU)
    # use same n_rf coverage
    # find threshold to match coverage
    sorted_gaps = np.sort(np.abs(proba_vix - 0.5))[::-1]
    thresh_vix  = sorted_gaps[min(n_rf, len(sorted_gaps)) - 1] if n_rf > 0 else TAU
    mask_vix_m  = np.abs(proba_vix - 0.5) >= thresh_vix
    y_pred_vix  = (proba_vix >= 0.5).astype(int)
    lr_acc      = accuracy_score(y_te[mask_vix_m], y_pred_vix[mask_vix_m]) * 100 if mask_vix_m.sum() >= 10 else np.nan

    # ── HAR logistic regression ───────────────────────────────────────────────
    lr_har    = LogisticRegression(max_iter=1000, random_state=42)
    cal_har   = CalibratedClassifierCV(lr_har, cv=tscv, method='isotonic')
    cal_har.fit(X_har_tr_s, y_tr)
    proba_har = cal_har.predict_proba(X_har_te_s)[:, 1]
    sorted_har = np.sort(np.abs(proba_har - 0.5))[::-1]
    thresh_har = sorted_har[min(n_rf, len(sorted_har)) - 1] if n_rf > 0 else TAU
    mask_har   = np.abs(proba_har - 0.5) >= thresh_har
    y_pred_har = (proba_har >= 0.5).astype(int)
    har_acc    = accuracy_score(y_te[mask_har], y_pred_har[mask_har]) * 100 if mask_har.sum() >= 10 else np.nan

    # ── Markov-chain transition probability baseline ──────────────────────────
    # Estimate 1-step transition probs from training labels
    labels_tr_d = (y[:split_idx]).astype(int)
    n00 = np.sum((labels_tr_d[:-1] == 0) & (labels_tr_d[1:] == 0))
    n01 = np.sum((labels_tr_d[:-1] == 0) & (labels_tr_d[1:] == 1))
    n10 = np.sum((labels_tr_d[:-1] == 1) & (labels_tr_d[1:] == 0))
    n11 = np.sum((labels_tr_d[:-1] == 1) & (labels_tr_d[1:] == 1))
    p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.5
    p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.5
    # N-step transition probs via matrix power
    P = np.array([[1 - p01, p01], [1 - p11, p11]])
    Pn = np.linalg.matrix_power(P, n)
    # Current state from VIX_t (using training labels as proxy for state)
    # In test: state at t = (VIX_t >= 20)
    # Prob of high at t+N given current state
    cur_state = (vix_te >= 20).astype(int)
    proba_mc  = np.where(cur_state == 0, Pn[0, 1], Pn[1, 1])
    sorted_mc = np.sort(np.abs(proba_mc - 0.5))[::-1]
    thresh_mc = sorted_mc[min(n_rf, len(sorted_mc)) - 1] if n_rf > 0 else TAU
    mask_mc   = np.abs(proba_mc - 0.5) >= thresh_mc
    y_pred_mc = (proba_mc >= 0.5).astype(int)
    mc_acc    = accuracy_score(y_te[mask_mc], y_pred_mc[mask_mc]) * 100 if mask_mc.sum() >= 10 else np.nan

    gap_rf_per = rf_acc - per_acc if not (np.isnan(rf_acc) or np.isnan(per_acc)) else np.nan

    print(f"  N={n}: Coverage={cov_rf:.1f}%  Persist={per_acc:.2f}%  LR={lr_acc:.2f}%  "
          f"HAR={har_acc:.2f}%  MC={mc_acc:.2f}%  RF={rf_acc:.2f}%  Gap={gap_rf_per:+.2f}pp")

    rows.append(dict(
        N=n, Coverage=round(cov_rf, 1),
        Persist_Acc=round(per_acc, 2) if not np.isnan(per_acc) else None,
        LR_Acc=round(lr_acc, 2) if not np.isnan(lr_acc) else None,
        HAR_Acc=round(har_acc, 2) if not np.isnan(har_acc) else None,
        MC_Acc=round(mc_acc, 2) if not np.isnan(mc_acc) else None,
        RF_Acc=round(rf_acc, 2) if not np.isnan(rf_acc) else None,
        RF_minus_Persist=round(gap_rf_per, 2) if not np.isnan(gap_rf_per) else None,
    ))

results = pd.DataFrame(rows)
results.to_csv('results/extended_baselines.csv', index=False)
print(f"\nSaved → results/extended_baselines.csv")
print(results.to_string(index=False))

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'RF Selective vs Baselines at Matched Coverage (τ={TAU})', fontsize=13)

colors = {
    'Persistence': '#e74c3c',
    'LR (VIX)':   '#e67e22',
    'HAR-LR':     '#9b59b6',
    'Markov':     '#3498db',
    'RF':         '#2e7d32',
}

for col, label, color in [
    ('Persist_Acc', 'Persistence', colors['Persistence']),
    ('LR_Acc',      'LR (VIX)',    colors['LR (VIX)']),
    ('HAR_Acc',     'HAR-LR',      colors['HAR-LR']),
    ('MC_Acc',      'Markov',      colors['Markov']),
    ('RF_Acc',      'RF (fixed-spec)', colors['RF']),
]:
    ax1.plot(results['N'], results[col], marker='o', linewidth=2, markersize=6,
             color=color, label=label)

ax1.set_xlabel('N (trading days ahead)')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Accuracy at Matched Coverage')
ax1.set_xticks(N_VALUES)
ax1.set_ylim(78, 100)
ax1.legend(framealpha=0.9)
ax1.grid(True, alpha=0.2)

# Gap bar chart: RF - Persistence
gaps = results['RF_minus_Persist'].values
colors_bar = ['#2e7d32' if g > 0 else '#e74c3c' for g in gaps]
bars = ax2.bar(results['N'], gaps, color=colors_bar, alpha=0.8, width=2.5)
ax2.axhline(0, color='black', linestyle='--', linewidth=0.9)
ax2.set_xlabel('N (trading days ahead)')
ax2.set_ylabel('Accuracy Gap (pp)')
ax2.set_title('RF minus Persistence Gap (with bootstrap CI from Table 9)')
ax2.set_xticks(N_VALUES)
ax2.grid(True, alpha=0.2, axis='y')

# Add CI error bars from bootstrap_ci.csv if available
try:
    ci = pd.read_csv('results/bootstrap_ci.csv')
    for _, row in ci.iterrows():
        n_ = row['N']
        g  = row['Gap_pp']
        lo = row['CI_lo']
        hi = row['CI_hi']
        ax2.errorbar(n_, g, yerr=[[g - lo], [hi - g]],
                     fmt='none', color='black', capsize=5, linewidth=1.5)
except Exception:
    pass

plt.tight_layout()
plt.savefig(f'{OUT}/persistence_baseline.png', bbox_inches='tight')
plt.close()
print(f"Figure saved → {OUT}/persistence_baseline.png")
