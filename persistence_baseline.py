"""
Computes three things on the same fresh RF model (n_estimators=200, max_depth=10,
min_samples_leaf=5, random_state=42) to guarantee internal consistency:

1. Persistence baseline at matched coverage (c chosen to match actual RF coverage)
2. Single-feature LR baseline (VIX level only)
3. Block-bootstrap 95% CI on (RF - Persistence) accuracy gap
4. Transition-conditional analysis (calm→calm, calm→high, high→calm, high→high)

RF accuracy and bootstrap CIs all reference the same model instance per horizon.
Tuned pipeline RF accuracy (Table 2) is not used here — see paper footnote.
"""
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})

OUT = 'vix paper'

df     = pd.read_csv('data/features.csv', index_col=0, parse_dates=True)
raw_df = pd.read_csv('data/raw_data.csv',  index_col=0, parse_dates=True)

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

rows_persist = []
rows_logreg  = []
rows_trans   = []
rows_extended = []   # for extended baseline CSV (adds HAR + Markov)

# Storage for bootstrap
proba_store = {}
y_store     = {}
vix_store   = {}
c_store     = {}   # training-calibrated persistence cutoff per horizon

print(f"{'═'*88}")
print("  PERSISTENCE BASELINE  vs  RF SELECTIVE  (all using same fresh RF spec)")
print(f"{'═'*88}")
print(f"  {'N':>3}  {'Persist Acc':>12}  {'RF Cov':>8}  {'LR Sel Acc':>12}  "
      f"{'RF Sel Acc':>12}  {'Gap':>9}")
print(f"  {'─'*72}")

for n in N_VALUES:
    label_col = f'LABEL_{n}'
    sub = df[feature_cols + [label_col]].dropna(subset=[label_col])

    X = sub[feature_cols].values
    y = sub[label_col].astype(int).values

    split_idx = int(len(sub) * 0.8)
    X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
    y_train, y_test         = y[:split_idx], y[split_idx:]

    test_idx  = sub.index[split_idx:]
    vix_test  = df.loc[test_idx, 'VIX'].values
    vix_train_arr = df.loc[sub.index[:split_idx], 'VIX'].values

    # ── Fresh RF (used for ALL comparisons including bootstrap) ──────────────
    print(f"  N={n} fitting RF...", end=' ', flush=True)
    scaler_f = StandardScaler()
    X_tr = scaler_f.fit_transform(X_train_raw)
    X_te = scaler_f.transform(X_test_raw)

    rf     = RandomForestClassifier(n_estimators=200, max_depth=10,
                                    min_samples_leaf=5, random_state=42)
    cal_rf = CalibratedClassifierCV(rf, cv=tscv, method='isotonic')
    cal_rf.fit(X_tr, y_train)
    proba_rf  = cal_rf.predict_proba(X_te)[:, 1]
    mask_rf   = (proba_rf > 0.5 + TAU) | (proba_rf < 0.5 - TAU)
    y_pred_rf = (proba_rf >= 0.5).astype(int)
    acc_rf    = accuracy_score(y_test[mask_rf], y_pred_rf[mask_rf]) if mask_rf.sum() >= 10 else np.nan
    cov_rf    = mask_rf.mean()
    n_rf      = int(mask_rf.sum())

    # RF coverage rate on the TRAINING window — all baseline coverage matching
    # is calibrated against this rate on training data only, then frozen
    proba_rf_tr = cal_rf.predict_proba(X_tr)[:, 1]
    mask_rf_tr  = (proba_rf_tr > 0.5 + TAU) | (proba_rf_tr < 0.5 - TAU)
    cov_rf_tr   = mask_rf_tr.mean()

    # Store for bootstrap
    proba_store[n] = proba_rf
    y_store[n]     = y_test
    vix_store[n]   = vix_test

    # ── 1. Persistence baseline at matched coverage ──────────────────────────
    # c is calibrated on the TRAINING window to match the RF's training
    # coverage rate, then frozen and applied unchanged to the test window
    best_c, best_diff = 0.0, float('inf')
    for c in np.arange(0, 15, 0.1):
        m_tr = np.abs(vix_train_arr - VIX_THRESH) >= c
        d = abs(m_tr.mean() - cov_rf_tr)
        if d < best_diff:
            best_diff, best_c = d, c

    c_store[n] = best_c

    mask_p   = np.abs(vix_test - VIX_THRESH) >= best_c
    y_pred_p = (vix_test[mask_p] >= VIX_THRESH).astype(int)
    acc_p    = accuracy_score(y_test[mask_p], y_pred_p) if mask_p.sum() >= 10 else np.nan
    cov_p    = mask_p.mean()
    gap      = (acc_rf - acc_p) * 100 if not (np.isnan(acc_rf) or np.isnan(acc_p)) else np.nan

    rows_persist.append(dict(N=n, c=round(best_c, 1),
                             Persist_Acc=round(acc_p * 100, 2),
                             Persist_Cov=round(cov_p * 100, 1),
                             RF_Acc=round(acc_rf * 100, 2),
                             RF_Cov=round(cov_rf * 100, 1),
                             Gap_pp=round(gap, 2)))

    # ── 2. Single-feature LR (VIX level only) ────────────────────────────────
    vix_train = df.loc[sub.index[:split_idx], 'VIX'].values.reshape(-1, 1)
    vix_te    = vix_test.reshape(-1, 1)

    scaler_lr = StandardScaler()
    vix_tr_s  = scaler_lr.fit_transform(vix_train)
    vix_te_s  = scaler_lr.transform(vix_te)

    lr      = LogisticRegression(random_state=42)
    cal_lr  = CalibratedClassifierCV(lr, cv=tscv, method='isotonic')
    cal_lr.fit(vix_tr_s, y_train)

    proba_lr  = cal_lr.predict_proba(vix_te_s)[:, 1]
    mask_lr   = (proba_lr > 0.5 + TAU) | (proba_lr < 0.5 - TAU)
    y_pred_lr = (proba_lr >= 0.5).astype(int)
    acc_lr    = accuracy_score(y_test[mask_lr], y_pred_lr[mask_lr]) if mask_lr.sum() >= 10 else np.nan
    cov_lr    = mask_lr.mean()

    rows_logreg.append(dict(N=n,
                            LR_Sel_Acc=round(acc_lr * 100, 2),
                            LR_Cov=round(cov_lr * 100, 1),
                            RF_Acc=round(acc_rf * 100, 2)))

    print(f"done  → {acc_p*100:.2f}%  {cov_rf*100:.1f}%  {acc_lr*100:.2f}%  "
          f"{acc_rf*100:.2f}%  {gap:+.2f}pp")

    # ── 2b. HAR-style LR (VIX_D, VIX_W=5-day mean, VIX_M=20-day mean) ─────────
    vix_full_train = df.loc[sub.index[:split_idx], 'VIX'].values
    vix_full_test  = vix_test
    # Compute rolling means using ONLY in-sample info when computing test features:
    # for test, use the last known training values
    vix_all   = np.concatenate([vix_full_train, vix_full_test])
    vix_w_all = pd.Series(vix_all).rolling(5,  min_periods=1).mean().values
    vix_m_all = pd.Series(vix_all).rolling(20, min_periods=1).mean().values
    X_har_tr  = np.column_stack([vix_full_train,
                                  vix_w_all[:split_idx],
                                  vix_m_all[:split_idx]])
    X_har_te  = np.column_stack([vix_full_test,
                                  vix_w_all[split_idx:],
                                  vix_m_all[split_idx:]])
    scaler_har = StandardScaler()
    X_har_tr_s = scaler_har.fit_transform(X_har_tr)
    X_har_te_s = scaler_har.transform(X_har_te)
    lr_har     = LogisticRegression(max_iter=1000, random_state=42)
    cal_har    = CalibratedClassifierCV(clone(lr_har), cv=tscv, method='isotonic')
    cal_har.fit(X_har_tr_s, y_train)
    proba_har  = cal_har.predict_proba(X_har_te_s)[:, 1]
    # coverage-match threshold calibrated on TRAINING probabilities, frozen
    proba_har_tr = cal_har.predict_proba(X_har_tr_s)[:, 1]
    thresh_har = np.quantile(np.abs(proba_har_tr - 0.5), 1 - cov_rf_tr)
    mask_har   = np.abs(proba_har - 0.5) >= thresh_har
    y_pred_har = (proba_har >= 0.5).astype(int)
    acc_har = accuracy_score(y_test[mask_har], y_pred_har[mask_har]) if mask_har.sum() >= 10 else np.nan

    # ── 2c. Markov-chain baseline ─────────────────────────────────────────────
    labels_tr_1step = y_train
    n00 = np.sum((labels_tr_1step[:-1] == 0) & (labels_tr_1step[1:] == 0))
    n01 = np.sum((labels_tr_1step[:-1] == 0) & (labels_tr_1step[1:] == 1))
    n10 = np.sum((labels_tr_1step[:-1] == 1) & (labels_tr_1step[1:] == 0))
    n11 = np.sum((labels_tr_1step[:-1] == 1) & (labels_tr_1step[1:] == 1))
    p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.5
    p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.5
    P   = np.array([[1 - p01, p01], [1 - p11, p11]])
    Pn  = np.linalg.matrix_power(P, n)
    cur_state_te = (vix_test >= VIX_THRESH).astype(int)
    proba_mc     = np.where(cur_state_te == 0, Pn[0, 1], Pn[1, 1])
    # coverage-match threshold calibrated on TRAINING states, frozen
    cur_state_tr = (vix_train_arr >= VIX_THRESH).astype(int)
    proba_mc_tr  = np.where(cur_state_tr == 0, Pn[0, 1], Pn[1, 1])
    thresh_mc    = np.quantile(np.abs(proba_mc_tr - 0.5), 1 - cov_rf_tr)
    mask_mc      = np.abs(proba_mc - 0.5) >= thresh_mc
    y_pred_mc    = (proba_mc >= 0.5).astype(int)
    acc_mc       = accuracy_score(y_test[mask_mc], y_pred_mc[mask_mc]) if mask_mc.sum() >= 10 else np.nan

    rows_extended.append(dict(
        N=n, Coverage=round(cov_rf * 100, 1),
        Persist_Acc=round(acc_p * 100, 2) if not np.isnan(acc_p) else None,
        LR_Acc=round(acc_lr * 100, 2) if not np.isnan(acc_lr) else None,
        HAR_Acc=round(acc_har * 100, 2) if not np.isnan(acc_har) else None,
        MC_Acc=round(acc_mc * 100, 2) if not np.isnan(acc_mc) else None,
        RF_Acc=round(acc_rf * 100, 2) if not np.isnan(acc_rf) else None,
        RF_minus_Persist=round(gap, 2) if not np.isnan(gap) else None,
    ))

    # ── 3. Transition-conditional analysis ───────────────────────────────────
    regime_now  = (vix_test >= VIX_THRESH).astype(int)
    regime_next = y_test

    trans_types = {
        'calm→calm':   (regime_now == 0) & (regime_next == 0),
        'calm→high':   (regime_now == 0) & (regime_next == 1),
        'high→calm':   (regime_now == 1) & (regime_next == 0),
        'high→high':   (regime_now == 1) & (regime_next == 1),
    }

    for ttype, t_mask in trans_types.items():
        n_days    = t_mask.sum()
        n_covered = (t_mask & mask_rf).sum()
        cov_t     = n_covered / n_days if n_days > 0 else np.nan

        if n_covered >= 5:
            acc_t  = accuracy_score(y_test[t_mask & mask_rf], y_pred_rf[t_mask & mask_rf])
            prec_t = precision_score(y_test[t_mask & mask_rf], y_pred_rf[t_mask & mask_rf], zero_division=0)
            rec_t  = recall_score(y_test[t_mask & mask_rf], y_pred_rf[t_mask & mask_rf], zero_division=0)
        else:
            acc_t = prec_t = rec_t = np.nan

        rows_trans.append(dict(N=n, Transition=ttype,
                               N_Days=int(n_days), N_Covered=int(n_covered),
                               Coverage=round(cov_t * 100, 1) if not np.isnan(cov_t) else None,
                               Accuracy=round(acc_t * 100, 2) if not np.isnan(acc_t) else None,
                               Precision=round(prec_t * 100, 2) if not np.isnan(prec_t) else None,
                               Recall=round(rec_t * 100, 2) if not np.isnan(rec_t) else None))

# ── Save primary results ──────────────────────────────────────────────────────
persist_df   = pd.DataFrame(rows_persist)
logreg_df    = pd.DataFrame(rows_logreg)
trans_df     = pd.DataFrame(rows_trans)
extended_df  = pd.DataFrame(rows_extended)

persist_df.to_csv('results/persistence_baseline.csv', index=False)
logreg_df.to_csv('results/logreg_baseline.csv', index=False)
trans_df.to_csv('results/transition_analysis.csv', index=False)
extended_df.to_csv('results/extended_baselines.csv', index=False)
print(f"\nExtended baselines:")
print(extended_df.to_string(index=False))

# ── Block-bootstrap CI on (RF - Persistence) gap ─────────────────────────────
def block_bootstrap_gap(y_true, proba_rf, vix_vals, c_frozen,
                        n_bootstrap=2000, rng=None, block_len=None):
    """c_frozen is the training-calibrated persistence cutoff — no test-set
    information is used to define either prediction set."""
    if rng is None:
        rng = np.random.default_rng(42)
    N_test = len(y_true)
    if block_len is None:
        block_len = max(int(N_test**0.5), 20)

    mask_rf   = (proba_rf > 0.5 + TAU) | (proba_rf < 0.5 - TAU)
    y_pred_rf = (proba_rf >= 0.5).astype(int)

    mask_p    = np.abs(vix_vals - VIX_THRESH) >= c_frozen
    y_pred_p  = (vix_vals >= VIX_THRESH).astype(int)

    acc_rf_pt = accuracy_score(y_true[mask_rf], y_pred_rf[mask_rf])
    acc_p_pt  = accuracy_score(y_true[mask_p],  y_pred_p[mask_p])
    gap       = (acc_rf_pt - acc_p_pt) * 100

    diffs = []
    idx   = np.arange(N_test)
    n_blk = int(np.ceil(N_test / block_len))
    for _ in range(n_bootstrap):
        starts = rng.integers(0, N_test - block_len + 1, size=n_blk)
        boot   = np.concatenate([idx[s:s + block_len] for s in starts])[:N_test]
        bm_rf  = mask_rf[boot]
        bm_p   = mask_p[boot]
        if bm_rf.sum() < 10 or bm_p.sum() < 10:
            continue
        a_rf = accuracy_score(y_true[boot][bm_rf], y_pred_rf[boot][bm_rf])
        a_p  = accuracy_score(y_true[boot][bm_p],  y_pred_p[boot][bm_p])
        diffs.append((a_rf - a_p) * 100)

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return gap, lo, hi

print(f"\n{'═'*88}")
print("  BLOCK-BOOTSTRAP 95% CI  (2000 reps, block length ≈ 30 days)")
print(f"{'═'*88}")
rng      = np.random.default_rng(42)
boot_rows = []

for n in N_VALUES:
    print(f"  N={n} bootstrapping...", end=' ', flush=True)
    gap, lo, hi = block_bootstrap_gap(y_store[n], proba_store[n], vix_store[n],
                                      c_store[n], rng=rng)
    sig = '  ← borderline' if -0.5 < lo <= 0 else ('  * p<0.05' if lo > 0 else '')
    print(f"  gap={gap:+.2f}pp  95% CI [{lo:+.2f}, {hi:+.2f}]{sig}")
    boot_rows.append(dict(N=n, Gap_pp=round(gap, 2),
                          CI_lo=round(lo, 2), CI_hi=round(hi, 2),
                          Significant=(lo > 0)))

boot_df = pd.DataFrame(boot_rows)
boot_df.to_csv('results/bootstrap_ci.csv', index=False)
print("  Saved → results/bootstrap_ci.csv")

# ── Block-length sensitivity (editor Major 4): 32 / 60 / 90 days ─────────────
print(f"\n{'═'*88}")
print("  BLOCK-LENGTH SENSITIVITY  (Editor Major 4)")
print(f"{'═'*88}")
sens_rows = []
for blk in [None, 60, 90]:
    for n in N_VALUES:
        gap, lo, hi = block_bootstrap_gap(y_store[n], proba_store[n], vix_store[n],
                                          c_store[n], rng=np.random.default_rng(42),
                                          block_len=blk)
        blk_label = blk if blk is not None else max(int(len(y_store[n])**0.5), 20)
        sens_rows.append(dict(Block_len=blk_label, N=n, Gap_pp=round(gap, 2),
                              CI_lo=round(lo, 2), CI_hi=round(hi, 2),
                              Significant=(lo > 0)))
        print(f"  block={blk_label:>3}  N={n:2d}  gap={gap:+.2f}pp  "
              f"95% CI [{lo:+.2f}, {hi:+.2f}]")

pd.DataFrame(sens_rows).to_csv('results/bootstrap_blocklen_sensitivity.csv', index=False)
print("  Saved → results/bootstrap_blocklen_sensitivity.csv")

# ── Transition summary print ──────────────────────────────────────────────────
print(f"\n{'═'*88}")
print("  TRANSITION-CONDITIONAL ANALYSIS — RF Selective (τ=0.25)")
print(f"{'═'*88}")
print(f"  {'N':>3}  {'Transition':>14}  {'Days':>6}  {'Cov%':>6}  "
      f"{'Acc%':>8}  {'Prec%':>8}  {'Rec%':>7}")
print(f"  {'─'*68}")
for _, r in trans_df.iterrows():
    acc_s  = f"{r['Accuracy']:.2f}%" if r['Accuracy'] is not None else 'N/A'
    prec_s = f"{r['Precision']:.2f}%" if r['Precision'] is not None else 'N/A'
    rec_s  = f"{r['Recall']:.2f}%" if r['Recall'] is not None else 'N/A'
    cov_s  = f"{r['Coverage']:.1f}%" if r['Coverage'] is not None else 'N/A'
    print(f"  {r['N']:>3}  {r['Transition']:>14}  {r['N_Days']:>6}  "
          f"{cov_s:>6}  {acc_s:>8}  {prec_s:>8}  {rec_s:>7}")

# ── Figure 1: comparison plot ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Selective Accuracy: RF Model vs Naive Persistence Baseline\n'
             '(coverage-matched, τ=0.25)', fontsize=12)

ns        = N_VALUES
rf_accs   = [persist_df[persist_df['N'] == n]['RF_Acc'].values[0] for n in ns]
pers_accs = [persist_df[persist_df['N'] == n]['Persist_Acc'].values[0] for n in ns]
lr_accs   = [logreg_df[logreg_df['N'] == n]['LR_Sel_Acc'].values[0] for n in ns]

ax = axes[0]
ax.plot(ns, rf_accs,   'o-',  color='#2e7d32', linewidth=2.2, markersize=7, label='RF Selective (τ=0.25)')
ax.plot(ns, pers_accs, 's--', color='#e74c3c', linewidth=2.0, markersize=7, label='Persistence (matched cov.)')
ax.plot(ns, lr_accs,   '^--', color='#e67e22', linewidth=2.0, markersize=7, label='Logistic Reg. (VIX only)')
ax.set_title('Accuracy at Matched Coverage')
ax.set_xlabel('N (trading days ahead)')
ax.set_ylabel('Selective Accuracy (%)')
ax.set_xticks(ns)
ax.set_ylim(70, 100)
ax.legend(framealpha=0.9)
ax.grid(True, alpha=0.2)

# Gap plot
ax2  = axes[1]
gaps = [r - p for r, p in zip(rf_accs, pers_accs)]
los  = [g - r for g, r in zip(gaps, boot_df['CI_lo'].values)]
his  = [r - g for g, r in zip(gaps, boot_df['CI_hi'].values)]
cols = ['#2e7d32' if sig else '#e74c3c' for sig in boot_df['Significant'].values]
ax2.bar(ns, gaps, width=2.2, color=cols, alpha=0.78, zorder=3)
ax2.errorbar(ns, gaps, yerr=[los, his], fmt='none',
             color='black', capsize=5, linewidth=1.5, zorder=4)
ax2.axhline(0, color='black', linewidth=0.9, linestyle='--')
ax2.set_title('RF − Persistence Gap with 95% CI')
ax2.set_xlabel('N (trading days ahead)')
ax2.set_ylabel('Gain (percentage points)')
ax2.set_xticks(ns)
ax2.legend(handles=[
    plt.Rectangle((0,0),1,1, fc='#2e7d32', alpha=0.78, label='Significant (CI excl. 0)'),
    plt.Rectangle((0,0),1,1, fc='#e74c3c', alpha=0.78, label='Not significant'),
], framealpha=0.9, fontsize=9)
ax2.grid(True, alpha=0.2, axis='y', zorder=0)

plt.tight_layout()
plt.savefig(f'{OUT}/persistence_baseline.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'\nPlot saved → {OUT}/persistence_baseline.png')

# ── Figure 2: bootstrap CI bar chart (standalone) ────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
gaps_b  = boot_df['Gap_pp'].values
los_b   = gaps_b - boot_df['CI_lo'].values
his_b   = boot_df['CI_hi'].values - gaps_b
cols_b  = ['#2e7d32' if sig else '#e74c3c' for sig in boot_df['Significant'].values]

ax.bar(ns, gaps_b, width=2.5, color=cols_b, alpha=0.75, zorder=3)
ax.errorbar(ns, gaps_b, yerr=[los_b, his_b], fmt='none',
            color='black', capsize=6, linewidth=1.5, zorder=4)
ax.axhline(0, color='black', linewidth=1.0, linestyle='--', alpha=0.5)
for n, g, lo, hi in zip(ns, gaps_b, boot_df['CI_lo'], boot_df['CI_hi']):
    ax.text(n, max(g, hi) + 0.3, f'{g:+.1f}pp', ha='center', va='bottom', fontsize=9)
ax.set_xlabel('Prediction Horizon N (trading days)')
ax.set_ylabel('RF Selective − Persistence Accuracy (pp)')
ax.set_title(f'RF vs Persistence Accuracy Gap with 95% Moving-Block Bootstrap CI\n'
             f'(τ = {TAU}, persistence coverage-matched to RF; green = significant at 5%)')
ax.set_xticks(ns)
ax.set_xlim(2, 28)
ax.grid(True, alpha=0.2, axis='y', zorder=0)
ax.legend(handles=[
    plt.Rectangle((0,0),1,1, fc='#2e7d32', alpha=0.75, label='CI excludes 0'),
    plt.Rectangle((0,0),1,1, fc='#e74c3c', alpha=0.75, label='CI includes 0'),
], loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUT}/bootstrap_ci.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Plot saved → {OUT}/bootstrap_ci.png')

# ── Figure 3: transition heatmap ──────────────────────────────────────────────
pivot = trans_df.pivot(index='Transition', columns='N', values='Accuracy')
fig, ax = plt.subplots(figsize=(9, 4))
im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
ax.set_xticks(range(len(ns)))
ax.set_xticklabels([f'N={n}' for n in ns])
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
for i in range(len(pivot.index)):
    for j in range(len(ns)):
        v = pivot.values[i, j]
        txt = f'{v:.0f}%' if (v is not None and not np.isnan(v)) else 'no cov.'
        color = 'white' if (not np.isnan(v) and (v < 30 or v > 80)) else 'black'
        ax.text(j, i, txt, ha='center', va='center', fontsize=10, color=color)
plt.colorbar(im, ax=ax, label='Accuracy on covered days (%)')
ax.set_title('RF Selective Accuracy by Regime Transition Type\n(covered days only, τ=0.25)')
plt.tight_layout()
plt.savefig(f'{OUT}/transition_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Plot saved → {OUT}/transition_analysis.png')
print('\nDone.')
