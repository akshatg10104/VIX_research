"""
Yield-curve inversion regime replication of the diagnostic protocol.

Label: T10Y2Y < 0 (yield curve inverted) N days ahead.
This is structurally distinct from VIX (driven by monetary policy / rate
dynamics) and from S&P 500 trend regimes (equity price trend).

In the test period (2022-2026), the yield curve inverted deeply in 2022-2023
before uninverting, giving genuine class variation with a different persistence
mechanism than volatility or equity regimes.

Uses the same 36 features, same RF spec, and same τ=0.25 selective threshold.

Outputs:
  results/yc_replication.csv
  vix paper/yc_replication.png
"""
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'legend.fontsize': 9, 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ── Load features ─────────────────────────────────────────────────────────────
df_feat = pd.read_csv('data/features.csv', index_col=0, parse_dates=True)
df_raw  = pd.read_csv('data/raw_data.csv',  index_col=0, parse_dates=True)

RAW_COLS   = ['VIX','VIX3M','SP500','GOLD','TNY','DXY',
              'FEDFUNDS','YIELD_CURVE','VIX9D','SKEW']
LABEL_COLS = (['LABEL'] +
              [f'LABEL_{n}' for n in [5,10,15,20,25]] +
              [f'LABEL_SUSTAINED_{n}' for n in [5,10,15,20,25]])
feature_cols = [c for c in df_feat.columns if c not in RAW_COLS + LABEL_COLS]

# Merge feature set with raw YIELD_CURVE
df = df_feat[feature_cols].copy()
df['YC'] = df_raw['YIELD_CURVE'].reindex(df.index).ffill()

print(f"Feature set: {len(feature_cols)} features")
print(f"Sample: {df.index[0].date()} to {df.index[-1].date()}")
print(f"YIELD_CURVE range: {df['YC'].min():.2f} to {df['YC'].max():.2f}")

# ── Build forward-shifted labels ──────────────────────────────────────────────
# Label = 1 if yield curve is INVERTED (T10Y2Y < 0) N trading days ahead
HORIZONS = [5, 10, 15, 20, 25]
for n in HORIZONS:
    df[f'YC_LABEL_{n}'] = (df['YC'].shift(-n) < 0).astype(float)
    df.loc[df['YC'].shift(-n).isna(), f'YC_LABEL_{n}'] = np.nan

split_idx = int(len(df) * 0.8)
split_date = df.index[split_idx]
print(f"\n80/20 split at: {split_date.date()}")
for n in HORIZONS:
    col  = f'YC_LABEL_{n}'
    sub  = df[col].dropna()
    test = df[col].iloc[split_idx:].dropna()
    print(f"  N={n:2d}: {len(sub)} obs total; test base rate "
          f"= {test.mean()*100:.1f}% inverted ({test.sum():.0f} of {len(test)} days)")

# ── Fixed-spec RF + calibration pipeline ─────────────────────────────────────
TAU       = 0.25
RF_PARAMS = dict(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42)

results = []

print(f"\n{'='*65}")
print("  YIELD CURVE INVERSION REPLICATION RESULTS")
print(f"{'='*65}")

for n in HORIZONS:
    label_col = f'YC_LABEL_{n}'
    sub       = df[feature_cols + [label_col, 'YC']].dropna(subset=[label_col])
    y         = sub[label_col].astype(int)

    split_i    = int(len(sub) * 0.8)
    X_tr_raw, y_tr = sub[feature_cols].iloc[:split_i], y.iloc[:split_i]
    X_te_raw, y_te = sub[feature_cols].iloc[split_i:], y.iloc[split_i:]
    yc_today       = sub['YC'].iloc[split_i:].values  # current yield curve level

    scaler = StandardScaler().fit(X_tr_raw)
    X_tr   = scaler.transform(X_tr_raw)
    X_te   = scaler.transform(X_te_raw)

    tscv   = TimeSeriesSplit(n_splits=5)
    cal_rf = CalibratedClassifierCV(
        RandomForestClassifier(**RF_PARAMS), cv=tscv, method='isotonic')
    cal_rf.fit(X_tr, y_tr)

    p_hat   = cal_rf.predict_proba(X_te)[:, 1]
    y_pred  = (p_hat > 0.5).astype(int)

    covered  = np.abs(p_hat - 0.5) > TAU
    cov_frac = covered.mean()
    sel_acc  = (y_pred[covered] == y_te.values[covered]).mean() if covered.sum() > 0 else np.nan

    # Balanced accuracy
    if covered.sum() > 0:
        mask1, mask0 = covered & (y_te.values==1), covered & (y_te.values==0)
        tpr = (y_pred[mask1]==1).mean() if mask1.sum()>0 else 0.5
        tnr = (y_pred[mask0]==0).mean() if mask0.sum()>0 else 0.5
        bal_acc = (tpr + tnr) / 2
    else:
        bal_acc = np.nan

    # Persistence baseline: YC inverted today → predict inverted in N days
    # Coverage-matched: use |YC_today| as confidence proxy (far from 0 = confident)
    pers_pred = (yc_today < 0).astype(int)   # persistence rule
    dist      = np.abs(yc_today)
    th_p      = np.quantile(dist, 1 - cov_frac) if cov_frac < 1 else 0.0
    p_cov     = dist >= th_p
    pers_acc  = (pers_pred[p_cov] == y_te.values[p_cov]).mean() if p_cov.sum() > 0 else np.nan

    # Persistence quantification
    inv_today = (yc_today < 0)
    p_ii = y_te.values[inv_today].mean()  if inv_today.sum() > 0  else np.nan
    p_ni = y_te.values[~inv_today].mean() if (~inv_today).sum() > 0 else np.nan
    ratio = p_ii / p_ni if (p_ni and p_ni > 0) else np.nan

    # Same-regime decomposition
    correct  = y_pred[covered] == y_te.values[covered]
    same_reg = y_pred[covered] == inv_today[covered].astype(int)
    same_frac = same_reg[correct].mean() if correct.sum() > 0 else np.nan

    # McNemar on jointly covered days
    both_cov   = covered & p_cov
    rf_right   = y_pred[both_cov] == y_te.values[both_cov]
    pers_right = pers_pred[both_cov] == y_te.values[both_cov]
    b = (~rf_right & pers_right).sum()
    c = (rf_right & ~pers_right).sum()

    # Transition analysis
    for from_s, to_s, name in [
        (0, 0, 'normal→normal'), (0, 1, 'normal→invert'),
        (1, 1, 'invert→invert'), (1, 0, 'invert→normal'),
    ]:
        is_from = inv_today == from_s
        is_to   = y_te.values == to_s
        mask_t  = is_from & is_to
        mask_c  = mask_t & covered
        n_days  = int(mask_t.sum())
        n_cov   = int(mask_c.sum())
        cov_r   = n_cov / n_days if n_days > 0 else 0.0
        acc_t   = (y_pred[mask_c] == y_te.values[mask_c]).mean() if n_cov > 0 else np.nan
        results.append({
            'N': n, 'Transition': name, 'Days': n_days,
            'Coverage': cov_r, 'Acc_covered': acc_t,
            'SelAcc': sel_acc, 'PersAcc': pers_acc,
            'CovFrac': cov_frac, 'BalAcc': bal_acc,
            'PersRatio': ratio, 'SameFrac': same_frac,
            'McNemar_b': b, 'McNemar_c': c,
        })

    print(f"\n  N={n}:")
    print(f"    Coverage: {cov_frac*100:.1f}%  Sel acc: {sel_acc*100:.1f}%  "
          f"BalAcc: {bal_acc*100:.1f}%  Persist: {pers_acc*100:.1f}%")
    print(f"    RF−Persist: {(sel_acc-pers_acc)*100:+.1f} pp")
    print(f"    P(inv|inv): {p_ii*100:.1f}%  P(inv|normal): {p_ni*100:.1f}%  "
          f"Ratio: {ratio:.1f}x")
    print(f"    Same-regime %: {same_frac*100:.1f}%  McNemar b={b}, c={c}")
    trans_rows = [r for r in results if r['N']==n]
    for r in trans_rows:
        acc_s = f"{r['Acc_covered']*100:.1f}%" if r['Acc_covered']==r['Acc_covered'] else "N/A"
        print(f"      {r['Transition']}: {r['Days']} days, "
              f"{r['Coverage']*100:.0f}% cov, acc={acc_s}")

res_df = pd.DataFrame(results)
res_df.to_csv('results/yc_replication.csv', index=False)
print("\nSaved → results/yc_replication.csv")

# ── Figure ────────────────────────────────────────────────────────────────────
summary = res_df.drop_duplicates(subset=['N'])[['N','SelAcc','PersAcc','CovFrac',
                                                 'BalAcc','PersRatio','SameFrac']]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Yield Curve Inversion Regime: Diagnostic Protocol Replication',
             fontsize=12, fontweight='bold')

ns    = HORIZONS
sels  = summary['SelAcc'].values * 100
perss = summary['PersAcc'].values * 100
bals  = summary['BalAcc'].values * 100

# Panel 1: RF vs Persistence
ax = axes[0]
ax.plot(ns, sels,  'o-', color='#e74c3c', lw=2, label='RF Selective')
ax.plot(ns, perss, 's--', color='#3498db', lw=2, label='Persistence')
ax.plot(ns, bals,  '^:', color='#2ecc71', lw=1.5, label='Balanced Acc.')
ax.axhline(50, ls='--', color='gray', lw=1)
for i, (n, s, p) in enumerate(zip(ns, sels, perss)):
    ax.text(n, max(s,p)+1.5, f'{s-p:+.0f}pp', ha='center', fontsize=7.5, color='darkred')
ax.set_xlabel('N (days)'); ax.set_ylabel('Accuracy (%)')
ax.set_title('RF vs. Persistence'); ax.legend(fontsize=8)
ax.set_ylim(40, 110); ax.grid(True, alpha=0.3)

# Panel 2: Persistence ratio
ax = axes[1]
ratios = summary['PersRatio'].values
ax.bar(ns, ratios, color='#9b59b6', alpha=0.8, width=3)
ax.axhline(9.7,  ls='--', color='#e74c3c', lw=1.5, label='VIX N=5 (9.7×)')
ax.axhline(23.8, ls=':', color='#3498db', lw=1.5, label='SP500 N=5 (23.8×)')
ax.set_xlabel('N (days)'); ax.set_ylabel('P(inv|inv) / P(inv|normal)')
ax.set_title('Persistence Ratio'); ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# Panel 3: Transition accuracy heatmap
ax = axes[2]
trans_names = ['normal→normal','normal→invert','invert→invert','invert→normal']
heat = np.full((len(trans_names), 2), np.nan)
for j, n in enumerate([5, 10]):
    rows = res_df[res_df['N']==n]
    for i, tr in enumerate(trans_names):
        r = rows[rows['Transition']==tr]
        if len(r) > 0 and not np.isnan(r['Acc_covered'].iloc[0]):
            heat[i, j] = r['Acc_covered'].iloc[0] * 100

im = ax.imshow(heat, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
ax.set_xticks([0,1]); ax.set_xticklabels(['N=5','N=10'])
ax.set_yticks(range(4)); ax.set_yticklabels(trans_names, fontsize=8)
ax.set_title('Transition Accuracy (covered)')
for i in range(4):
    for j in range(2):
        v = heat[i,j]
        txt = f'{v:.0f}%' if not np.isnan(v) else 'N/A'
        col = 'white' if (not np.isnan(v) and (v<30 or v>80)) else 'black'
        ax.text(j, i, txt, ha='center', va='center', fontsize=10,
                fontweight='bold', color=col)
plt.colorbar(im, ax=ax, label='Accuracy (%)')

plt.tight_layout()
plt.savefig('vix paper/yc_replication.png', bbox_inches='tight')
plt.close()
print("Saved → vix paper/yc_replication.png")
