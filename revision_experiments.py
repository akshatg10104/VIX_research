"""
JRFM revision (jrfm-4438773) — new experiments requested by editor and reviewers.

Outputs:
  results/xgb_comparison.csv            — XGBoost selective acc/coverage/BA + transition breakdown
                                          (Editor Theme 3, R2 Comment 3)
  results/har_threshold_baseline.csv    — HAR level forecast of VIX_{t+N}, thresholded at 20,
                                          abstention band calibrated on training (Editor Major 3)
  results/risk_coverage_curves.csv      — full tau sweep for RF/HGB/XGB/persistence (Editor Major 5)
  results/aurc_summary.csv              — AURC per model per horizon (Editor Major 5)
  results/joint_coverage_composition.csv— |RF ∩ persistence| size and transition mix (Editor Major 2)
  results/auc_brier.csv                 — AUC + Brier pre/post calibration (Editor Major 7)
  results/transition_episodes.csv       — distinct calm→high episodes among covered days (Editor Major 6)
  results/test_sample_sizes.csv         — test N and class frequencies per horizon (Editor Minor 3)
  vix paper/risk_coverage.png           — risk–coverage figure

All baseline coverage matching is calibrated on the training window only and frozen
before touching test data.
"""
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, brier_score_loss
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})

df = pd.read_csv('data/features.csv', index_col=0, parse_dates=True)

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

MODELS = {
    'RF':  lambda: RandomForestClassifier(n_estimators=200, max_depth=10,
                                          min_samples_leaf=5, random_state=42),
    'HGB': lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                                  random_state=42),
    'XGB': lambda: XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                 subsample=0.8, colsample_bytree=0.8,
                                 eval_metric='logloss', random_state=42),
}

xgb_rows, har_rows, rc_rows, aurc_rows = [], [], [], []
joint_rows, ab_rows, ep_rows, size_rows = [], [], [], []


def selective_metrics(proba, y, tau=TAU):
    mask = (proba > 0.5 + tau) | (proba < 0.5 - tau)
    pred = (proba >= 0.5).astype(int)
    if mask.sum() < 10:
        return np.nan, np.nan, mask, pred
    acc = accuracy_score(y[mask], pred[mask])
    bal = balanced_accuracy_score(y[mask], pred[mask])
    return acc, bal, mask, pred


def risk_coverage(conf, pred, y):
    """Sweep confidence threshold; return (coverage, risk) pairs and AURC."""
    order = np.argsort(-conf)          # most confident first
    correct = (pred == y).astype(float)[order]
    n = len(y)
    cum_err = np.cumsum(1 - correct)
    cov  = np.arange(1, n + 1) / n
    risk = cum_err / np.arange(1, n + 1)
    aurc = np.trapz(risk, cov)
    return cov, risk, aurc


for n in N_VALUES:
    label_col = f'LABEL_{n}'
    sub = df[feature_cols + [label_col]].dropna(subset=[label_col])
    X = sub[feature_cols].values
    y = sub[label_col].astype(int).values

    split_idx = int(len(sub) * 0.8)
    X_tr_raw, X_te_raw = X[:split_idx], X[split_idx:]
    y_tr, y_te         = y[:split_idx], y[split_idx:]
    vix_tr = df.loc[sub.index[:split_idx], 'VIX'].values
    vix_te = df.loc[sub.index[split_idx:], 'VIX'].values
    test_dates = sub.index[split_idx:]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_te = scaler.transform(X_te_raw)

    # ── Per-horizon sample sizes and class frequencies (Editor Minor 3) ──────
    size_rows.append(dict(N=n, Train_N=split_idx, Test_N=len(y_te),
                          Train_high_pct=round(y_tr.mean() * 100, 1),
                          Test_high_pct=round(y_te.mean() * 100, 1)))

    # Transition types on test window
    regime_now = (vix_te >= VIX_THRESH).astype(int)
    trans = {'calm→calm': (regime_now == 0) & (y_te == 0),
             'calm→high': (regime_now == 0) & (y_te == 1),
             'high→calm': (regime_now == 1) & (y_te == 0),
             'high→high': (regime_now == 1) & (y_te == 1)}

    print(f"\n═══ N={n} ═══")
    proba_cal, proba_raw, masks = {}, {}, {}

    for name, make in MODELS.items():
        print(f"  {name}: fitting raw + calibrated...", flush=True)
        raw = make()
        raw.fit(X_tr, y_tr)
        proba_raw[name] = raw.predict_proba(X_te)[:, 1]

        cal = CalibratedClassifierCV(make(), cv=tscv, method='isotonic')
        cal.fit(X_tr, y_tr)
        proba_cal[name] = cal.predict_proba(X_te)[:, 1]

        # AUC + Brier pre/post calibration (Editor Major 7)
        ab_rows.append(dict(N=n, Model=name,
                            AUC_pre=round(roc_auc_score(y_te, proba_raw[name]), 4),
                            AUC_post=round(roc_auc_score(y_te, proba_cal[name]), 4),
                            Brier_pre=round(brier_score_loss(y_te, proba_raw[name]), 4),
                            Brier_post=round(brier_score_loss(y_te, proba_cal[name]), 4)))

        acc, bal, mask, pred = selective_metrics(proba_cal[name], y_te)
        masks[name] = mask

        if name == 'XGB':
            row = dict(N=n, Sel_Acc=round(acc * 100, 2) if not np.isnan(acc) else np.nan,
                       Coverage=round(mask.mean() * 100, 1),
                       Bal_Acc=round(bal * 100, 2) if not np.isnan(bal) else np.nan)
            for tname, tmask in trans.items():
                cm = tmask & mask
                row[f'{tname}_cov'] = round(cm.sum() / max(tmask.sum(), 1) * 100, 1)
                row[f'{tname}_acc'] = (round(accuracy_score(y_te[cm], pred[cm]) * 100, 1)
                                       if cm.sum() > 0 else np.nan)
            xgb_rows.append(row)

    # RF training coverage rate for all frozen coverage matching
    cal_rf_tmp = CalibratedClassifierCV(MODELS['RF'](), cv=tscv, method='isotonic')
    cal_rf_tmp.fit(X_tr, y_tr)
    proba_rf_tr = cal_rf_tmp.predict_proba(X_tr)[:, 1]
    cov_rf_tr = ((proba_rf_tr > 0.5 + TAU) | (proba_rf_tr < 0.5 - TAU)).mean()

    # ── HAR level-forecast-then-threshold (Editor Major 3) ───────────────────
    vix_w_tr = pd.Series(vix_tr).rolling(5,  min_periods=1).mean().values
    vix_m_tr = pd.Series(vix_tr).rolling(20, min_periods=1).mean().values
    vix_all  = np.concatenate([vix_tr, vix_te])
    vix_w_all = pd.Series(vix_all).rolling(5,  min_periods=1).mean().values
    vix_m_all = pd.Series(vix_all).rolling(20, min_periods=1).mean().values

    # target: VIX level at t+N (train rows only; y label already encodes test truth)
    vix_series = df['VIX'].reindex(sub.index)
    target_lvl = vix_series.shift(-n).values
    X_har_tr = np.column_stack([vix_tr, vix_w_all[:split_idx], vix_m_all[:split_idx]])
    X_har_te = np.column_stack([vix_te, vix_w_all[split_idx:], vix_m_all[split_idx:]])
    tgt_tr   = target_lvl[:split_idx]
    ok       = ~np.isnan(tgt_tr)

    har_reg = LinearRegression().fit(X_har_tr[ok], tgt_tr[ok])
    fc_tr   = har_reg.predict(X_har_tr)
    fc_te   = har_reg.predict(X_har_te)

    # abstention band d calibrated on TRAINING forecasts, frozen
    d_band  = np.quantile(np.abs(fc_tr - VIX_THRESH), 1 - cov_rf_tr)
    mask_h  = np.abs(fc_te - VIX_THRESH) >= d_band
    pred_h  = (fc_te >= VIX_THRESH).astype(int)
    acc_h   = accuracy_score(y_te[mask_h], pred_h[mask_h]) if mask_h.sum() >= 10 else np.nan
    ch      = trans['calm→high']
    ch_cov  = (ch & mask_h)
    ch_acc  = accuracy_score(y_te[ch_cov], pred_h[ch_cov]) if ch_cov.sum() > 0 else np.nan
    acc_rf, _, mask_rf_sel, pred_rf = selective_metrics(proba_cal['RF'], y_te)
    har_rows.append(dict(N=n, d_band=round(d_band, 2),
                         HAR_thr_Acc=round(acc_h * 100, 2) if not np.isnan(acc_h) else np.nan,
                         HAR_thr_Cov=round(mask_h.mean() * 100, 1),
                         RF_Sel_Acc=round(acc_rf * 100, 2),
                         CalmHigh_days=int(ch.sum()),
                         CalmHigh_covered=int(ch_cov.sum()),
                         CalmHigh_Acc=round(ch_acc * 100, 1) if not np.isnan(ch_acc) else np.nan))

    # ── Persistence mask (training-calibrated c, frozen) ─────────────────────
    c_frozen = np.quantile(np.abs(vix_tr - VIX_THRESH), 1 - cov_rf_tr)
    mask_p   = np.abs(vix_te - VIX_THRESH) >= c_frozen
    pred_p   = (vix_te >= VIX_THRESH).astype(int)

    # ── Joint covered set composition (Editor Major 2) ───────────────────────
    joint = mask_rf_sel & mask_p
    jrow  = dict(N=n, Joint_N=int(joint.sum()),
                 Joint_pct_of_test=round(joint.mean() * 100, 1),
                 RF_cov_pct=round(mask_rf_sel.mean() * 100, 1),
                 Persist_cov_pct=round(mask_p.mean() * 100, 1))
    for tname, tmask in trans.items():
        jrow[f'{tname}_n'] = int((joint & tmask).sum())
        jrow[f'{tname}_pct_of_joint'] = round((joint & tmask).sum() / max(joint.sum(), 1) * 100, 1)
    joint_rows.append(jrow)

    # ── Risk–coverage curves + AURC (Editor Major 5) ─────────────────────────
    for name in MODELS:
        conf = np.abs(proba_cal[name] - 0.5)
        pred = (proba_cal[name] >= 0.5).astype(int)
        cov, risk, aurc = risk_coverage(conf, pred, y_te)
        aurc_rows.append(dict(N=n, Model=name, AURC=round(aurc, 4)))
        step = max(len(cov) // 100, 1)
        for i in range(0, len(cov), step):
            rc_rows.append(dict(N=n, Model=name, Coverage=round(cov[i], 3),
                                Risk=round(risk[i], 4)))
    conf_p = np.abs(vix_te - VIX_THRESH)
    cov, risk, aurc = risk_coverage(conf_p, pred_p, y_te)
    aurc_rows.append(dict(N=n, Model='Persistence', AURC=round(aurc, 4)))
    step = max(len(cov) // 100, 1)
    for i in range(0, len(cov), step):
        rc_rows.append(dict(N=n, Model='Persistence', Coverage=round(cov[i], 3),
                            Risk=round(risk[i], 4)))

    # ── Episode clustering of covered calm→high days (Editor Major 6) ────────
    ch_covered_pos = np.where(trans['calm→high'] & mask_rf_sel)[0]
    episodes = []
    if len(ch_covered_pos) > 0:
        start = ch_covered_pos[0]
        prev  = ch_covered_pos[0]
        for p in ch_covered_pos[1:]:
            if p - prev > 10:
                episodes.append((start, prev))
                start = p
            prev = p
        episodes.append((start, prev))
    for (s, e) in episodes:
        ep_rows.append(dict(N=n,
                            Episode_start=str(test_dates[s].date()),
                            Episode_end=str(test_dates[e].date()),
                            Covered_days=int(((ch_covered_pos >= s) & (ch_covered_pos <= e)).sum())))
    print(f"  calm→high covered days={len(ch_covered_pos)}  episodes={len(episodes)}")

# ── Save all outputs ─────────────────────────────────────────────────────────
pd.DataFrame(xgb_rows).to_csv('results/xgb_comparison.csv', index=False)
pd.DataFrame(har_rows).to_csv('results/har_threshold_baseline.csv', index=False)
pd.DataFrame(rc_rows).to_csv('results/risk_coverage_curves.csv', index=False)
pd.DataFrame(aurc_rows).to_csv('results/aurc_summary.csv', index=False)
pd.DataFrame(joint_rows).to_csv('results/joint_coverage_composition.csv', index=False)
pd.DataFrame(ab_rows).to_csv('results/auc_brier.csv', index=False)
pd.DataFrame(ep_rows).to_csv('results/transition_episodes.csv', index=False)
pd.DataFrame(size_rows).to_csv('results/test_sample_sizes.csv', index=False)

# ── Risk–coverage figure ─────────────────────────────────────────────────────
rc_df = pd.DataFrame(rc_rows)
fig, axes = plt.subplots(1, 5, figsize=(22, 4.2), sharey=True)
colors = {'RF': '#1f77b4', 'HGB': '#2ca02c', 'XGB': '#9467bd', 'Persistence': '#d62728'}
for ax, n in zip(axes, N_VALUES):
    d_n = rc_df[rc_df.N == n]
    for model, grp in d_n.groupby('Model'):
        g = grp.sort_values('Coverage')
        ax.plot(g.Coverage, g.Risk, label=model, color=colors[model],
                lw=1.8, ls='--' if model == 'Persistence' else '-')
    ax.set_title(f'N = {n}')
    ax.set_xlabel('Coverage')
axes[0].set_ylabel('Selective risk (error rate)')
axes[0].legend(frameon=False)
plt.tight_layout()
plt.savefig('vix paper/risk_coverage.png', dpi=200, bbox_inches='tight')
print("\nAll revision experiment outputs saved.")

print("\n=== XGB summary ===");           print(pd.DataFrame(xgb_rows).to_string(index=False))
print("\n=== HAR-threshold summary ==="); print(pd.DataFrame(har_rows).to_string(index=False))
print("\n=== AURC ===");                  print(pd.DataFrame(aurc_rows).pivot(index='N', columns='Model', values='AURC').to_string())
print("\n=== Joint coverage ===");        print(pd.DataFrame(joint_rows).to_string(index=False))
print("\n=== Episodes ===");              print(pd.DataFrame(ep_rows).to_string(index=False))
