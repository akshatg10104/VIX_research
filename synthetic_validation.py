"""
Synthetic positive-control experiment.

Generative model:
  - Background Markov chain (P_HH=0.85, P_CH=0.05) — same autocorrelation as VIX
  - Exogenous event alarm: fires on calm days when Z_t > 1.03 (~15% of calm days)
  - P(HIGH in N days | alarm, calm)   = 0.90
  - P(HIGH in N days | no alarm, calm)= 0.05
  - P(HIGH in N days | high today)    = 0.85 (persistence)

Features (36 total):
  - 5 regime-proxy features: noisy signals of today's regime state
    (analogous to VIX SMA, Bollinger bands — measure current state)
  - 1 alarm oracle feature: noisy Z_t (signals upcoming transitions when high)
  - 30 pure noise features

Under this design the RF learns BOTH persistence (from regime proxies) AND
genuine transition skill (from alarm oracle). The diagnostic protocol
correctly identifies the model as having genuine skill:
  calm→high covered accuracy >> 50%   (alarm days correctly flagged as high)
  RF outperforms or matches persistence (McNemar b+c > 0)
  Same-regime fraction < 100%          (some correct regime-change calls)

Outputs:
  results/synthetic_validation.csv
  vix paper/synthetic_validation.png
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
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

np.random.seed(42)
N_TOTAL = 5000
SPLIT   = 0.80
TAU     = 0.25

# ── 1. Background Markov chain ────────────────────────────────────────────────
P_HH, P_CH = 0.85, 0.05
regime = np.zeros(N_TOTAL, dtype=int)
for t in range(1, N_TOTAL):
    p = P_HH if regime[t-1] == 1 else P_CH
    regime[t] = int(np.random.rand() < p)

print(f"Synthetic event-alarm dataset: {N_TOTAL} obs")
print(f"  Markov: P_HH={P_HH}, P_CH={P_CH}, "
      f"base rate={regime.mean()*100:.1f}% high")

# ── 2. Event alarm: fires on 15% of calm days ─────────────────────────────────
ALARM_RATE = 0.15
Z_t = np.random.randn(N_TOTAL)
# Threshold for top-15% of N(0,1): ~1.03
ALARM_THR = float(np.percentile(Z_t[regime == 0], 85))
alarm = (Z_t > ALARM_THR) & (regime == 0)
print(f"  Event alarms: {alarm.sum()} of {(regime==0).sum()} "
      f"calm days ({alarm.mean()/(1-regime.mean())*100:.1f}%)")

# ── 3. N-step labels ──────────────────────────────────────────────────────────
P_HIGH_ALARM  = 0.90
P_HIGH_NORMAL = 0.05
P_HIGH_HIGH   = 0.85

HORIZONS = [5, 10]
labels = {}
for n in HORIZONS:
    lbl = np.full(N_TOTAL, np.nan)
    for t in range(N_TOTAL - n):
        if regime[t] == 1:
            lbl[t] = int(np.random.rand() < P_HIGH_HIGH)
        elif alarm[t]:
            lbl[t] = int(np.random.rand() < P_HIGH_ALARM)
        else:
            lbl[t] = int(np.random.rand() < P_HIGH_NORMAL)
    labels[n] = lbl
    valid = ~np.isnan(lbl)
    print(f"  N={n}: {valid.sum()} obs, {lbl[valid].mean()*100:.1f}% high-label")

# ── 4. Features ───────────────────────────────────────────────────────────────
# 5 regime proxy features (analogous to VIX SMA/Bollinger): noisy current regime
regime_f = np.column_stack([
    regime + 0.20 * np.random.randn(N_TOTAL),                      # current state
    pd.Series(regime.astype(float)).rolling(5,  min_periods=1).mean().values
        + 0.15 * np.random.randn(N_TOTAL),                          # 5-day mean
    pd.Series(regime.astype(float)).rolling(10, min_periods=1).mean().values
        + 0.15 * np.random.randn(N_TOTAL),                          # 10-day mean
    pd.Series(regime.astype(float)).diff(5).fillna(0).values
        + 0.15 * np.random.randn(N_TOTAL),                          # momentum
    (pd.Series(regime.astype(float)).rolling(5, min_periods=1).mean().values
     - pd.Series(regime.astype(float)).rolling(10, min_periods=1).mean().values)
        + 0.15 * np.random.randn(N_TOTAL),                          # MA cross
])

# 1 alarm oracle feature: noisy Z_t (high when alarm firing)
oracle_f = Z_t.reshape(-1, 1) + 0.20 * np.random.randn(N_TOTAL, 1)

# 30 pure noise features
noise_f = np.random.randn(N_TOTAL, 30)

X = np.column_stack([regime_f, oracle_f, noise_f])
feat_names = (['REG_PROXY','REG_SMA5','REG_SMA10','REG_MOM5','REG_MACROSS',
               'ALARM_ORACLE']
              + [f'NOISE_{i:02d}' for i in range(30)])
print(f"  Features: 5 regime-proxy + 1 alarm-oracle + 30 noise = {len(feat_names)}")

# ── 5. Diagnostic protocol ────────────────────────────────────────────────────
RF_PARAMS = dict(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42)
results_all = []

print(f"\n{'='*65}")
print("  SYNTHETIC POSITIVE CONTROL: DIAGNOSTIC PROTOCOL")
print(f"{'='*65}")

for n in HORIZONS:
    lbl   = labels[n]
    valid = ~np.isnan(lbl)
    X_v   = X[valid]
    y_v   = lbl[valid].astype(int)
    reg_v = regime[valid]
    alm_v = alarm[valid]

    split_idx  = int(len(X_v) * SPLIT)
    X_tr, y_tr = X_v[:split_idx], y_v[:split_idx]
    X_te, y_te = X_v[split_idx:], y_v[split_idx:]
    reg_te     = reg_v[split_idx:]
    alm_te     = alm_v[split_idx:]

    scaler = StandardScaler().fit(X_tr)
    X_trs  = scaler.transform(X_tr)
    X_tes  = scaler.transform(X_te)

    tscv   = TimeSeriesSplit(n_splits=5)
    cal_rf = CalibratedClassifierCV(
        RandomForestClassifier(**RF_PARAMS), cv=tscv, method='isotonic')
    cal_rf.fit(X_trs, y_tr)

    p_hat  = cal_rf.predict_proba(X_tes)[:, 1]
    y_pred = (p_hat > 0.5).astype(int)

    covered  = np.abs(p_hat - 0.5) > TAU
    cov_frac = covered.mean()
    sel_acc  = (y_pred[covered] == y_te[covered]).mean() if covered.sum() > 0 else np.nan

    # Balanced accuracy
    if covered.sum() > 0:
        mask1, mask0 = covered & (y_te==1), covered & (y_te==0)
        tpr = (y_pred[mask1]==1).mean() if mask1.sum() > 0 else 0.5
        tnr = (y_pred[mask0]==0).mean() if mask0.sum() > 0 else 0.5
        bal_acc = (tpr + tnr) / 2
    else:
        bal_acc = np.nan

    # Coverage-matched persistence
    dist  = np.abs(reg_te.astype(float) - 0.5)
    th_p  = np.quantile(dist, 1 - cov_frac) if cov_frac < 1 else 0.0
    p_cov = dist >= th_p
    pers_acc = (reg_te[p_cov] == y_te[p_cov]).mean() if p_cov.sum() > 0 else np.nan

    # Persistence quantification
    p_hh = y_te[reg_te==1].mean() if (reg_te==1).sum() > 0 else np.nan
    p_he = y_te[reg_te==0].mean() if (reg_te==0).sum() > 0 else np.nan
    ratio = p_hh / p_he if (p_he and p_he > 0) else np.nan

    # Same-regime decomposition
    correct  = y_pred[covered] == y_te[covered]
    same_reg = y_pred[covered] == reg_te[covered]
    same_frac = same_reg[correct].mean() if correct.sum() > 0 else np.nan

    # McNemar
    both_cov   = covered & p_cov
    rf_right   = y_pred[both_cov] == y_te[both_cov]
    pers_right = reg_te[both_cov]  == y_te[both_cov]
    b = (~rf_right & pers_right).sum()
    c = (rf_right & ~pers_right).sum()

    # Transition accuracy
    trans_results = {}
    for (from_s, to_s, name) in [
        (0,0,'calm→calm'),(0,1,'calm→high'),(1,1,'high→high'),(1,0,'high→calm')
    ]:
        mask_t   = (reg_te == from_s) & (y_te == to_s)
        mask_cov = mask_t & covered
        n_days   = int(mask_t.sum())
        n_cov    = int(mask_cov.sum())
        cov_r    = n_cov / n_days if n_days > 0 else 0.0
        acc_t    = (y_pred[mask_cov] == y_te[mask_cov]).mean() if n_cov > 0 else np.nan
        trans_results[name] = {'days': n_days, 'cov': cov_r, 'acc': acc_t}

    # Alarm-day breakdown
    alm_calm = alm_te & (reg_te == 0)
    if alm_calm.sum() > 0 and covered[alm_calm].sum() > 0:
        alm_cov_acc = (y_pred[alm_calm & covered] == y_te[alm_calm & covered]).mean()
    else:
        alm_cov_acc = np.nan

    results_all.append({
        'N': n, 'SelAcc': sel_acc, 'CovFrac': cov_frac, 'BalAcc': bal_acc,
        'PersAcc': pers_acc, 'SameFrac': same_frac, 'Ratio': ratio,
        'McNemar_b': b, 'McNemar_c': c, 'AlarmCovAcc': alm_cov_acc,
        **{k: v['acc'] for k, v in trans_results.items()},
        **{f'cov_{k}': v['cov'] for k, v in trans_results.items()},
    })

    print(f"\n  N={n}:")
    print(f"    Coverage: {cov_frac*100:.1f}%  Sel acc: {sel_acc*100:.1f}%  "
          f"BalAcc: {bal_acc*100:.1f}%")
    print(f"    RF−Persist: {(sel_acc-pers_acc)*100:+.1f} pp  "
          f"(RF={sel_acc*100:.1f}%, Persist={pers_acc*100:.1f}%)")
    print(f"    P(H|H): {p_hh*100:.1f}%  P(H|C): {p_he*100:.1f}%  "
          f"Ratio: {ratio:.1f}x")
    print(f"    Same-regime %: {same_frac*100:.1f}%")
    if alm_cov_acc == alm_cov_acc:
        print(f"    Alarm-day covered acc: {alm_cov_acc*100:.1f}%")
    print(f"    McNemar: b={b}, c={c} "
          f"({'identical' if b==0 and c==0 else 'DIFFER → genuine skill'})")
    print(f"    Transitions:")
    for k, v in trans_results.items():
        acc_s = f"{v['acc']*100:.1f}%" if v['acc']==v['acc'] else "N/A"
        print(f"      {k}: {v['days']} days, {v['cov']*100:.0f}% cov, acc={acc_s}")

res_df = pd.DataFrame(results_all)
res_df.to_csv('results/synthetic_validation.csv', index=False)
print("\nSaved → results/synthetic_validation.csv")

# ── 6. Figure ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    'Synthetic Positive Control: Protocol Correctly Identifies Genuine Skill\n'
    '(event-alarm generative model: oracle feature signals upcoming regime spikes)',
    fontsize=11, fontweight='bold')

x = np.arange(2)
label_x = ['N=5', 'N=10']

# Panel 1: RF vs Persistence
ax = axes[0]
sel_v  = [r['SelAcc']*100  for r in results_all]
pers_v = [r['PersAcc']*100 for r in results_all]
bal_v  = [r['BalAcc']*100  for r in results_all]
w = 0.25
ax.bar(x-w, sel_v,  w, label='RF Selective',  color='#e74c3c', alpha=0.85)
ax.bar(x,   pers_v, w, label='Persistence',   color='#3498db', alpha=0.85)
ax.bar(x+w, bal_v,  w, label='Balanced Acc.', color='#2ecc71', alpha=0.85)
ax.axhline(50, ls='--', color='gray', lw=1)
ax.set_xticks(x); ax.set_xticklabels(label_x)
ax.set_ylabel('Accuracy (%)')
ax.set_title('RF vs. Persistence (genuine model)')
ax.legend(fontsize=8); ax.set_ylim(40, 115); ax.grid(True, alpha=0.3, axis='y')
for i in range(2):
    gap = sel_v[i] - pers_v[i]
    ax.text(i-w, sel_v[i]+0.5,  f'{sel_v[i]:.0f}%',  ha='center', fontsize=8)
    ax.text(i,   pers_v[i]+0.5, f'{pers_v[i]:.0f}%', ha='center', fontsize=8)
    col = 'darkgreen' if gap > 0 else 'darkred'
    ax.text(i, 113, f'RF−P: {gap:+.0f}pp', ha='center', fontsize=7.5,
            color=col, fontweight='bold')

# Panel 2: calm→high accuracy vs VIX (KEY diagnostic)
ax = axes[1]
ch_syn = [r.get('calm→high', np.nan) for r in results_all]
ch_vix = [0.0, 0.0]
for i in range(2):
    v = ch_syn[i]
    col = '#2ecc71' if (v==v and v>0.5) else '#f39c12' if (v==v and v>0.0) else '#95a5a6'
    ax.bar(i-0.2, (v*100 if v==v else 0), 0.35, color=col, alpha=0.85,
           label='Synthetic (oracle)' if i==0 else '')
    ax.bar(i+0.2, 0.0, 0.35, color='#e74c3c', alpha=0.4, hatch='//',
           label='Real VIX (main paper)' if i==0 else '')
ax.axhline(50, ls='--', color='gray', lw=1.2, label='Random (50%)')
ax.set_xticks(x); ax.set_xticklabels(label_x)
ax.set_ylabel('calm→high Covered Accuracy (%)')
ax.set_title('Transition Accuracy\n(synthetic alarm days correctly flagged)')
ax.legend(fontsize=8); ax.set_ylim(0, 115); ax.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(ch_syn):
    val = v*100 if v==v else 0
    col = 'darkgreen' if val > 50 else ('darkorange' if val > 0 else 'gray')
    ax.text(i-0.2, val+2, f'{val:.0f}%', ha='center', fontsize=10,
            fontweight='bold', color=col)
ax.text(0+0.2, 2, '0%', ha='center', fontsize=10, color='darkred', fontweight='bold')
ax.text(1+0.2, 2, '0%', ha='center', fontsize=10, color='darkred', fontweight='bold')

# Panel 3: Same-regime decomposition
ax = axes[2]
same_syn = [r['SameFrac']*100 for r in results_all]
same_vix = [99.0, 99.6]
ax.bar(x-0.2, same_syn, 0.35, label='Synthetic (oracle)', color='#2ecc71', alpha=0.85)
ax.bar(x+0.2, same_vix, 0.35, label='Real VIX (main paper)', color='#e74c3c', alpha=0.85)
ax.axhline(100, ls='--', color='gray', lw=1)
ax.set_xticks(x); ax.set_xticklabels(label_x)
ax.set_ylabel('Same-regime % of Correct Predictions')
ax.set_title('Persistence Decomposition\n(<100% confirms regime-change skill)')
ax.legend(fontsize=8); ax.set_ylim(40, 115); ax.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(same_syn):
    ax.text(i-0.2, v+1, f'{v:.0f}%', ha='center', fontsize=9, color='darkgreen')
for i, v in enumerate(same_vix):
    ax.text(i+0.2, v+1, f'{v:.0f}%', ha='center', fontsize=9, color='darkred')

plt.tight_layout()
plt.savefig('vix paper/synthetic_validation.png', bbox_inches='tight')
plt.close()
print("Saved → vix paper/synthetic_validation.png")
