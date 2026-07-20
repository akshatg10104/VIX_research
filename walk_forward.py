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
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df = pd.read_csv('data/features.csv', index_col=0, parse_dates=True)

RAW_COLS   = ['VIX', 'VIX3M', 'SP500', 'GOLD', 'TNY', 'DXY',
              'FEDFUNDS', 'YIELD_CURVE', 'VIX9D', 'SKEW']
LABEL_COLS = (
    ['LABEL'] +
    [f'LABEL_{n}' for n in [5, 10, 15, 20, 25]] +
    [f'LABEL_SUSTAINED_{n}' for n in [5, 10, 15, 20, 25]]
)
feature_cols = [c for c in df.columns if c not in RAW_COLS + LABEL_COLS]

N_VALUES       = [5, 10, 20]
CONF_THRESHOLD = 0.25
N_SPLITS       = 5

MODELS = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10,
                                            min_samples_leaf=5, random_state=42),
    'HGB':       HistGradientBoostingClassifier(max_iter=200, max_depth=5,
                                                    learning_rate=0.05, random_state=42),
}

tscv_cal = TimeSeriesSplit(n_splits=5)

rows = []

print(f"{'═'*80}")
print(f"  WALK-FORWARD VALIDATION — {N_SPLITS} expanding windows")
print(f"{'═'*80}")

for n in N_VALUES:
    label_col = f'LABEL_{n}'
    sub       = df[feature_cols + [label_col, 'VIX']].dropna(subset=[label_col])
    X   = sub[feature_cols].values
    y   = sub[label_col].astype(int).values
    vix = sub['VIX'].values

    tscv_wf = TimeSeriesSplit(n_splits=N_SPLITS, gap=n)

    print(f"\n  N={n}")
    print(f"  {'Model':<18}  {'Fold':>5}  {'Base':>8}  {'SelAcc':>8}  {'ΔAcc':>7}  {'%Days':>7}")
    print(f"  {'─'*62}")

    # Persistence tracking — filled alongside the RF fold loop
    fold_base_persist = []
    fold_sel_persist  = []
    fold_pct_persist  = []

    for model_name, model_template in MODELS.items():
        fold_base = []
        fold_sel  = []
        fold_pct  = []

        for fold, (train_idx, test_idx) in enumerate(tscv_wf.split(X), 1):
            X_train_raw, X_test_raw = X[train_idx], X[test_idx]
            y_train_all, y_test     = y[train_idx], y[test_idx]

            if len(np.unique(y_train_all)) < 2 or len(y_test) < 50:
                continue

            cal_cut = int(len(y_train_all) * 0.8)
            X_fit_raw = X_train_raw[:cal_cut]
            X_cal_raw = X_train_raw[cal_cut:]
            y_fit     = y_train_all[:cal_cut]
            y_cal     = y_train_all[cal_cut:]

            if len(np.unique(y_fit)) < 2 or len(np.unique(y_cal)) < 2:
                continue

            scaler  = StandardScaler()
            X_fit   = scaler.fit_transform(X_fit_raw)
            X_cal   = scaler.transform(X_cal_raw)
            X_test  = scaler.transform(X_test_raw)
            X_train = scaler.transform(X_train_raw)
            y_train = y_train_all

            ModelClass = type(model_template)
            params     = model_template.get_params()
            mdl        = ModelClass(**params)
            mdl.fit(X_fit, y_fit)

            cal = CalibratedClassifierCV(mdl, cv='prefit', method='isotonic')
            cal.fit(X_cal, y_cal)

            proba  = cal.predict_proba(X_test)[:, 1]
            y_pred = (proba >= 0.5).astype(int)

            base_acc = accuracy_score(y_test, y_pred)

            mask    = (proba > 0.5 + CONF_THRESHOLD) | (proba < 0.5 - CONF_THRESHOLD)
            n_sel   = int(mask.sum())
            pct_sel = n_sel / len(y_test) * 100

            if n_sel < 20:
                sel_acc = float('nan')
            else:
                sel_acc = accuracy_score(y_test[mask], y_pred[mask])

            fold_base.append(base_acc)
            if not np.isnan(sel_acc):
                fold_sel.append(sel_acc)
                fold_pct.append(pct_sel)

            # ── Persistence baseline at RF-matched coverage (only computed once per fold) ──
            # c calibrated on the fold's TRAINING window to match the RF's
            # training coverage rate, then frozen and applied to the fold test set
            if model_name == 'Random Forest' and n_sel >= 20:
                vix_test      = vix[test_idx]
                persist_pred  = (vix_test >= 20).astype(int)
                dist          = np.abs(vix_test - 20)
                proba_tr_wf   = cal.predict_proba(X_train)[:, 1]
                cov_tr_wf     = ((proba_tr_wf > 0.5 + CONF_THRESHOLD) |
                                 (proba_tr_wf < 0.5 - CONF_THRESHOLD)).mean()
                vix_train_wf  = vix[train_idx]
                c_thresh      = np.quantile(np.abs(vix_train_wf - 20), 1 - cov_tr_wf)
                p_mask        = dist >= c_thresh
                p_base_acc    = accuracy_score(y_test, persist_pred)
                p_sel_acc     = accuracy_score(y_test[p_mask], persist_pred[p_mask]) if p_mask.sum() >= 10 else float('nan')
                fold_base_persist.append(p_base_acc)
                if not np.isnan(p_sel_acc):
                    fold_sel_persist.append(p_sel_acc)
                    fold_pct_persist.append(p_mask.mean() * 100)

            sel_str = f"{sel_acc*100:>7.2f}%" if not np.isnan(sel_acc) else "     N/A"
            print(f"  {model_name:<18}  {fold:>5}  {base_acc*100:>7.2f}%  {sel_str}  "
                  f"{(sel_acc-base_acc)*100:>+6.2f}%  {pct_sel:>6.1f}%" if not np.isnan(sel_acc)
                  else f"  {model_name:<18}  {fold:>5}  {base_acc*100:>7.2f}%  {sel_str}")

        if fold_base:
            mean_base = np.mean(fold_base)
            std_base  = np.std(fold_base)
            mean_sel  = np.mean(fold_sel) if fold_sel else float('nan')
            std_sel   = np.std(fold_sel)  if fold_sel else float('nan')
            mean_pct  = np.mean(fold_pct) if fold_pct else float('nan')
            delta     = mean_sel - mean_base if not np.isnan(mean_sel) else float('nan')

            print(f"  {model_name:<18}  {'MEAN':>5}  "
                  f"{mean_base*100:>6.2f}±{std_base*100:.1f}%  "
                  f"{mean_sel*100:>6.2f}±{std_sel*100:.1f}%  "
                  f"{delta*100:>+6.2f}%  {mean_pct:>6.1f}%")

            rows.append(dict(
                N=n, Model=model_name,
                Mean_Base=round(mean_base*100, 2), Std_Base=round(std_base*100, 2),
                Mean_Sel=round(mean_sel*100, 2) if not np.isnan(mean_sel) else None,
                Std_Sel=round(std_sel*100, 2) if not np.isnan(std_sel) else None,
                Mean_Delta=round(delta*100, 2) if not np.isnan(delta) else None,
                Mean_Pct=round(mean_pct, 1) if not np.isnan(mean_pct) else None,
            ))

    # ── Persistence summary row ───────────────────────────────────────────────
    if fold_base_persist:
        mb = np.mean(fold_base_persist); sb = np.std(fold_base_persist)
        ms = np.mean(fold_sel_persist)  if fold_sel_persist  else float('nan')
        ss = np.std(fold_sel_persist)   if fold_sel_persist  else float('nan')
        mp = np.mean(fold_pct_persist)  if fold_pct_persist  else float('nan')
        dt = ms - mb if not np.isnan(ms) else float('nan')
        print(f"  {'Persistence (matched)':<18}  {'MEAN':>5}  "
              f"{mb*100:>6.2f}±{sb*100:.1f}%  "
              f"{ms*100:>6.2f}±{ss*100:.1f}%  "
              f"{dt*100:>+6.2f}%  {mp:>6.1f}%")
        rows.append(dict(
            N=n, Model='Persistence',
            Mean_Base=round(mb*100, 2), Std_Base=round(sb*100, 2),
            Mean_Sel=round(ms*100, 2) if not np.isnan(ms) else None,
            Std_Sel=round(ss*100, 2)  if not np.isnan(ss) else None,
            Mean_Delta=round(dt*100, 2) if not np.isnan(dt) else None,
            Mean_Pct=round(mp, 1)    if not np.isnan(mp) else None,
        ))

results = pd.DataFrame(rows)
results.to_csv('results/walk_forward.csv', index=False)

print(f"\n\n{'═'*80}")
print(f"  WALK-FORWARD SUMMARY")
print(f"{'═'*80}")
print(f"  {'Model':<18}  {'N':>4}  {'Base (mean±std)':>16}  {'Sel (mean±std)':>15}  {'Δ':>7}  {'%Days':>7}")
print(f"  {'─'*72}")
for _, r in results.iterrows():
    print(f"  {r['Model']:<18}  {int(r['N']):>4}  "
          f"{r['Mean_Base']:>6.2f}±{r['Std_Base']:.1f}%  "
          f"{r['Mean_Sel']:>6.2f}±{r['Std_Sel']:.1f}%  "
          f"{r['Mean_Delta']:>+6.2f}%  {r['Mean_Pct']:>6.1f}%")

print(f"\nResults saved → results/walk_forward.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Walk-Forward Validation ({N_SPLITS} folds) — Base vs Selective (t={CONF_THRESHOLD})',
             fontsize=12)
colors = {5: '#e74c3c', 10: '#3498db', 20: '#2ecc71'}

for ax, model_name in zip(axes, ['Random Forest', 'HGB']):
    sub  = results[results['Model'] == model_name]
    pers = results[results['Model'] == 'Persistence']
    x    = sub['N'].values

    ax.errorbar(x - 0.3, sub['Mean_Base'], yerr=sub['Std_Base'],
                fmt='o--', label='Baseline', color='#aaaaaa', capsize=4, linewidth=2)
    ax.errorbar(x + 0.3, sub['Mean_Sel'], yerr=sub['Std_Sel'],
                fmt='s-', label=f'Selective (t={CONF_THRESHOLD})', color='#3498db', capsize=4, linewidth=2)
    if len(pers) > 0:
        ax.errorbar(x, pers['Mean_Sel'].values, yerr=pers['Std_Sel'].values,
                    fmt='^:', label='Persistence (matched cov)', color='#e67e22', capsize=4, linewidth=1.5)

    ax.set_title(model_name)
    ax.set_xlabel('N (days ahead)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.25)

plt.tight_layout()
import os; os.makedirs('vix paper', exist_ok=True)
plt.savefig('vix paper/walk_forward.png', dpi=150)
plt.close()
print("Plot saved → vix paper/walk_forward.png")
