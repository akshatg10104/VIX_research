"""
Tests whether the abstention-rate elevation before calm→high transitions
is simply a restatement of "VIX is close to the threshold."

Three analyses:
1. AUC comparison: |VIX_t - 20| vs abstention_t as predictors of calm→high on calm days
2. Moving-block bootstrap CI on (abstention_before_CH - abstention_overall)
3. HAR-LR vs RF accuracy gap bootstrap CI (symmetry with RF vs persistence)

Outputs:
  results/abstention_confound.csv   — AUC comparison table
  results/abstention_bootstrap.csv  — bootstrap CIs on abstention gap
  results/har_bootstrap.csv         — bootstrap CIs on HAR-LR vs RF gap
  vix paper/abstention_confound.png — ROC curves figure
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
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from scipy import stats as scipy_stats
from sklearn.base import clone
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
N_EVAL   = [5, 10]
tscv     = TimeSeriesSplit(n_splits=5)
N_BOOT   = 2000
rng      = np.random.default_rng(42)

auc_rows   = []
boot_rows  = []
har_rows   = []

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle('Abstention Confound Test: ROC Curves\n'
             '(predicting calm→high on calm days only)', fontsize=13)

for col_idx, n in enumerate(N_EVAL):
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

    # ── RF calibrated ─────────────────────────────────────────────────────────
    rf     = RandomForestClassifier(n_estimators=200, max_depth=10,
                                    min_samples_leaf=5, random_state=42)
    cal_rf = CalibratedClassifierCV(rf, cv=tscv, method='isotonic')
    cal_rf.fit(X_tr, y_tr)
    proba_rf  = cal_rf.predict_proba(X_te)[:, 1]
    abstained = (np.abs(proba_rf - 0.5) <= TAU).astype(int)

    # ── HAR-LR ────────────────────────────────────────────────────────────────
    vix_all   = np.concatenate([vix_tr, vix_te])
    vix_w_all = pd.Series(vix_all).rolling(5,  min_periods=1).mean().values
    vix_m_all = pd.Series(vix_all).rolling(20, min_periods=1).mean().values
    X_har_tr  = np.column_stack([vix_tr,
                                  vix_w_all[:split_idx],
                                  vix_m_all[:split_idx]])
    X_har_te  = np.column_stack([vix_te,
                                  vix_w_all[split_idx:],
                                  vix_m_all[split_idx:]])
    sc_har = StandardScaler()
    X_har_tr_s = sc_har.fit_transform(X_har_tr)
    X_har_te_s = sc_har.transform(X_har_te)
    lr_har    = LogisticRegression(max_iter=1000, random_state=42)
    cal_har   = CalibratedClassifierCV(clone(lr_har), cv=tscv, method='isotonic')
    cal_har.fit(X_har_tr_s, y_tr)
    proba_har = cal_har.predict_proba(X_har_te_s)[:, 1]

    # ── Restrict to CALM days (VIX_t < 20) for the confound test ─────────────
    calm_mask = (vix_te < 20)
    y_calm    = y_te[calm_mask]           # 0=calm→calm, 1=calm→high
    abs_calm  = abstained[calm_mask]      # abstention on calm days
    dist_calm = (20 - vix_te[calm_mask])  # distance to threshold (>0 for calm)
    # Normalize to [0,1] for AUC comparability
    dist_norm = dist_calm / dist_calm.max()
    # dist as "spike signal": higher proximity (smaller distance) → higher score
    prox_calm = 1 - dist_norm             # proximity score

    n_spikes = int(y_calm.sum())
    print(f"\nN={n}: {calm_mask.sum()} calm days, {n_spikes} calm→high spikes")

    # AUC of each signal as a binary calm→high classifier
    auc_abstain = roc_auc_score(y_calm, abs_calm)   if y_calm.sum() > 0 else np.nan
    auc_prox    = roc_auc_score(y_calm, prox_calm)  if y_calm.sum() > 0 else np.nan
    auc_har     = roc_auc_score(y_calm, proba_har[calm_mask]) if y_calm.sum() > 0 else np.nan
    auc_rf_proba = roc_auc_score(y_calm, proba_rf[calm_mask]) if y_calm.sum() > 0 else np.nan

    print(f"  AUC — Abstention: {auc_abstain:.3f} | VIX Proximity: {auc_prox:.3f} | "
          f"HAR prob: {auc_har:.3f} | RF prob: {auc_rf_proba:.3f}")

    # ── Incremental test: does abstention add info beyond proximity alone? ────────
    auc_combined = np.nan
    lr_pval = np.nan
    if y_calm.sum() >= 10 and (y_calm == 0).sum() >= 10:
        X_prox_only  = prox_calm.reshape(-1, 1)
        X_prox_abs   = np.column_stack([prox_calm, abs_calm])
        lr1 = LogisticRegression(max_iter=1000, random_state=42).fit(X_prox_only,  y_calm)
        lr2 = LogisticRegression(max_iter=1000, random_state=42).fit(X_prox_abs,   y_calm)
        auc_combined = roc_auc_score(y_calm, lr2.predict_proba(X_prox_abs)[:, 1])
        # Likelihood-ratio test (1 extra df)
        eps = 1e-10
        ll1 = np.sum(y_calm * np.log(lr1.predict_proba(X_prox_only)[:, 1] + eps) +
                     (1-y_calm) * np.log(lr1.predict_proba(X_prox_only)[:, 0] + eps))
        ll2 = np.sum(y_calm * np.log(lr2.predict_proba(X_prox_abs)[:, 1] + eps) +
                     (1-y_calm) * np.log(lr2.predict_proba(X_prox_abs)[:, 0] + eps))
        lr_stat = max(0.0, 2 * (ll2 - ll1))
        lr_pval = scipy_stats.chi2.sf(lr_stat, df=1)
        print(f"  Incremental: AUC(proximity) = {auc_prox:.3f} | "
              f"AUC(proximity+abstention) = {auc_combined:.3f} | "
              f"LR p = {lr_pval:.3f} ({'significant' if lr_pval < 0.05 else 'not significant'})")

    auc_rows.append(dict(N=n, AUC_Abstention=round(auc_abstain,3),
                         AUC_VIX_Proximity=round(auc_prox,3),
                         AUC_Combined=round(auc_combined,3) if not np.isnan(auc_combined) else np.nan,
                         LR_pval=round(lr_pval,3) if not np.isnan(lr_pval) else np.nan,
                         AUC_HAR=round(auc_har,3),
                         AUC_RF=round(auc_rf_proba,3),
                         N_calm_spikes=n_spikes))

    # ── ROC curves ────────────────────────────────────────────────────────────
    ax = axes[0, col_idx]
    for signal, label, color in [
        (abs_calm,   f'Abstention (AUC={auc_abstain:.3f})', '#e74c3c'),
        (prox_calm,  f'VIX proximity (AUC={auc_prox:.3f})', '#3498db'),
        (proba_har[calm_mask], f'HAR-LR prob (AUC={auc_har:.3f})', '#9b59b6'),
    ]:
        fpr, tpr, _ = roc_curve(y_calm, signal)
        ax.plot(fpr, tpr, linewidth=2, color=color, label=label)
    ax.plot([0,1],[0,1],'k--',linewidth=0.9,alpha=0.4,label='Random')
    ax.set_title(f'ROC: calm→high prediction, N={n} (n={n_spikes} spikes)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    # ── Bootstrap CI on abstention-rate gap (before_CH - overall) ─────────────
    N_te        = len(y_te)
    block_len   = max(int(N_te**0.5), 20)
    calm_to_high_idx = np.where((vix_te < 20) & (y_te == 1))[0]
    overall_abs_rate = abstained.mean()

    diffs = []
    for _ in range(N_BOOT):
        starts = rng.integers(0, max(1, N_te - block_len + 1),
                              size=int(np.ceil(N_te / block_len)))
        boot_idx = np.concatenate([np.arange(s, min(s + block_len, N_te))
                                   for s in starts])[:N_te]
        # Identify calm→high days in the bootstrap sample
        ch_in_boot = np.intersect1d(calm_to_high_idx, boot_idx)
        if len(ch_in_boot) < 5:
            continue
        abs_before_ch = abstained[ch_in_boot - 1]   # 1 day before each CH event
        abs_before_ch = abs_before_ch[(ch_in_boot - 1) >= 0]
        if len(abs_before_ch) < 3:
            continue
        overall_boot = abstained[boot_idx].mean()
        diffs.append(abs_before_ch.mean() - overall_boot)

    if len(diffs) >= 100:
        point_est = abstained[calm_to_high_idx - 1]
        point_est = point_est[(calm_to_high_idx - 1) >= 0].mean() - overall_abs_rate
        ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
        sig = ci_lo > 0
        print(f"  Abstention gap CI: {point_est:+.3f}  [{ci_lo:+.3f}, {ci_hi:+.3f}]  "
              f"{'SIGNIFICANT' if sig else 'not significant'}")
        boot_rows.append(dict(N=n, Point_Est=round(point_est,3),
                              CI_lo=round(ci_lo,3), CI_hi=round(ci_hi,3),
                              Significant=sig, N_boot=len(diffs)))
    else:
        print(f"  Warning: insufficient bootstrap samples ({len(diffs)})")

    # ── Bootstrap CI on HAR-LR vs RF accuracy gap ─────────────────────────────
    mask_rf  = (proba_rf > 0.5 + TAU) | (proba_rf < 0.5 - TAU)
    n_rf     = int(mask_rf.sum())
    y_pred_rf = (proba_rf >= 0.5).astype(int)
    acc_rf_pt = accuracy_score(y_te[mask_rf], y_pred_rf[mask_rf]) if n_rf >= 10 else np.nan

    gaps_har  = np.sort(np.abs(proba_har - 0.5))[::-1]
    thresh_har = gaps_har[min(n_rf, len(gaps_har)) - 1] if n_rf > 0 else TAU
    mask_har_m = np.abs(proba_har - 0.5) >= thresh_har
    y_pred_har = (proba_har >= 0.5).astype(int)
    acc_har_pt = accuracy_score(y_te[mask_har_m], y_pred_har[mask_har_m]) if mask_har_m.sum() >= 10 else np.nan
    gap_har_rf = (acc_har_pt - acc_rf_pt) * 100 if not (np.isnan(acc_har_pt) or np.isnan(acc_rf_pt)) else np.nan

    har_diffs = []
    for _ in range(N_BOOT):
        starts = rng.integers(0, max(1, N_te - block_len + 1),
                              size=int(np.ceil(N_te / block_len)))
        boot_idx = np.concatenate([np.arange(s, min(s + block_len, N_te))
                                   for s in starts])[:N_te]
        bm_rf  = mask_rf[boot_idx]
        bm_har = mask_har_m[boot_idx]
        if bm_rf.sum() < 10 or bm_har.sum() < 10:
            continue
        a_rf  = accuracy_score(y_te[boot_idx][bm_rf],  y_pred_rf[boot_idx][bm_rf])
        a_har = accuracy_score(y_te[boot_idx][bm_har], y_pred_har[boot_idx][bm_har])
        har_diffs.append((a_har - a_rf) * 100)

    if len(har_diffs) >= 100:
        ci_lo_h, ci_hi_h = np.percentile(har_diffs, [2.5, 97.5])
        sig_h = ci_lo_h > 0
        print(f"  HAR-LR vs RF gap: {gap_har_rf:+.2f}pp  [{ci_lo_h:+.2f}, {ci_hi_h:+.2f}]  "
              f"{'SIGNIFICANT' if sig_h else 'not significant'}")
        har_rows.append(dict(N=n, Gap_pp=round(gap_har_rf,2),
                             CI_lo=round(ci_lo_h,2), CI_hi=round(ci_hi_h,2),
                             Significant=sig_h))

    # ── Bootstrap distribution histogram ─────────────────────────────────────
    ax2 = axes[1, col_idx]
    ax2.hist(diffs, bins=40, color='#e74c3c', alpha=0.7, density=True, label='Bootstrap dist.')
    ax2.axvline(point_est, color='black', linewidth=1.8, label=f'Observed: {point_est:+.2f}')
    ax2.axvline(ci_lo, color='gray', linewidth=1.2, linestyle='--',
                label=f'95% CI [{ci_lo:+.2f}, {ci_hi:+.2f}]')
    ax2.axvline(ci_hi, color='gray', linewidth=1.2, linestyle='--')
    ax2.axvline(0, color='navy', linewidth=1.0, linestyle=':', alpha=0.5)
    ax2.set_title(f'Bootstrap: abstention gap before spike, N={n}')
    ax2.set_xlabel('Abstention rate before CH − overall abstention rate')
    ax2.set_ylabel('Density')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(f'{OUT}/abstention_confound.png', bbox_inches='tight')
plt.close()
print(f"\nSaved → {OUT}/abstention_confound.png")

auc_df  = pd.DataFrame(auc_rows)
boot_df = pd.DataFrame(boot_rows)
har_df  = pd.DataFrame(har_rows)

auc_df.to_csv('results/abstention_confound.csv', index=False)
boot_df.to_csv('results/abstention_bootstrap.csv', index=False)
har_df.to_csv('results/har_bootstrap.csv', index=False)
print(f"Saved → results/abstention_confound.csv")
print(f"Saved → results/abstention_bootstrap.csv")
print(f"Saved → results/har_bootstrap.csv")

print("\n=== AUC Summary ===")
print(auc_df.to_string(index=False))
print("\n=== Abstention Bootstrap CIs ===")
print(boot_df.to_string(index=False))
print("\n=== HAR-LR vs RF Bootstrap CIs ===")
print(har_df.to_string(index=False))
