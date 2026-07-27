"""
Regenerate threshold_optimization.csv using unified fixed-spec pipeline.
RF: n_estimators=200, max_depth=10, min_samples_leaf=5
HGB: max_iter=200, max_depth=5, learning_rate=0.1
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/features.csv', index_col=0, parse_dates=True)
RAW_COLS   = ['VIX', 'VIX3M', 'SP500', 'GOLD', 'TNY', 'DXY',
              'FEDFUNDS', 'YIELD_CURVE', 'VIX9D', 'SKEW']
LABEL_COLS = (['LABEL'] +
              [f'LABEL_{n}' for n in [5, 10, 15, 20, 25]] +
              [f'LABEL_SUSTAINED_{n}' for n in [5, 10, 15, 20, 25]])
feature_cols = [c for c in df.columns if c not in RAW_COLS + LABEL_COLS]

N_VALUES   = [5, 10, 15, 20, 25]
THRESHOLDS = np.arange(0.05, 0.41, 0.05).round(2)
tscv       = TimeSeriesSplit(n_splits=5)

MODELS = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10,
                                            min_samples_leaf=5, random_state=42),
    'XGBoost':       HistGradientBoostingClassifier(max_iter=200, max_depth=5,
                                                    learning_rate=0.1, random_state=42),
}

rows = []
from sklearn.base import clone

for n in N_VALUES:
    label_col = f'LABEL_{n}'
    sub = df[feature_cols + [label_col]].dropna(subset=[label_col])
    X = sub[feature_cols].values
    y = sub[label_col].astype(int).values
    split_idx = int(len(sub) * 0.8)
    X_tr_raw, X_te_raw = X[:split_idx], X[split_idx:]
    y_tr, y_te         = y[:split_idx], y[split_idx:]
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr_raw)
    X_te   = scaler.transform(X_te_raw)

    for model_name, model_spec in MODELS.items():
        cal_mdl = CalibratedClassifierCV(clone(model_spec), cv=tscv, method='isotonic')
        cal_mdl.fit(X_tr, y_tr)
        proba = cal_mdl.predict_proba(X_te)[:, 1]
        y_pred = (proba >= 0.5).astype(int)

        for tau in THRESHOLDS:
            mask  = (proba > 0.5 + tau) | (proba < 0.5 - tau)
            n_sel = int(mask.sum())
            pct   = n_sel / len(y_te) * 100
            acc   = accuracy_score(y_te[mask], y_pred[mask]) * 100 if n_sel >= 50 else np.nan
            rows.append(dict(Model=model_name, N=n, Threshold=round(tau, 2),
                             Sel_Acc=round(acc, 2) if not np.isnan(acc) else None,
                             Pct_Days=round(pct, 1)))
        print(f"  N={n}, {model_name} done")

results = pd.DataFrame(rows)
results.to_csv('results/threshold_optimization.csv', index=False)
print(f"Saved → results/threshold_optimization.csv")
print(results[results['Threshold']==0.25].to_string(index=False))
