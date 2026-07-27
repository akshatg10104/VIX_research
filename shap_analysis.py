"""
SHAP analysis of Random Forest feature importance.

Shows that VIX autocorrelation features dominate all predictions,
supporting the thesis that the RF is essentially learning persistence.

Outputs:
  results/shap_mean_abs.csv
  vix paper/shap_analysis.png
"""
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'legend.fontsize': 10, 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
})

df = pd.read_csv('data/features.csv', index_col=0, parse_dates=True)
RAW_COLS   = ['VIX', 'VIX3M', 'SP500', 'GOLD', 'TNY', 'DXY',
              'FEDFUNDS', 'YIELD_CURVE', 'VIX9D', 'SKEW']
LABEL_COLS = (['LABEL'] +
              [f'LABEL_{n}' for n in [5, 10, 15, 20, 25]] +
              [f'LABEL_SUSTAINED_{n}' for n in [5, 10, 15, 20, 25]])
feature_cols = [c for c in df.columns if c not in RAW_COLS + LABEL_COLS]
tscv = TimeSeriesSplit(n_splits=5)

print(f"{'='*65}")
print("  SHAP ANALYSIS")
print(f"{'='*65}")

# Feature category labels for color coding
CATEGORY = {}
vix_tech = ['VIX_SMA10','VIX_SMA20','VIX_MOM10','VIX_MOM20','VIX_ROC10',
            'VIX_ROC20','VIX_BB_STD','VIX_BB_UPPER','VIX_BB_LOWER',
            'VIX_BB_POSITION','VIX_RSI14','VIX_MA_CROSS','VIX_TERM_SPREAD']
skew_f   = ['SKEW_LEVEL','SKEW_CHANGE5','SKEW_CHANGE20']
sp500_f  = ['SP500_RET1','SP500_RET5','SP500_RET20','SP500_MOM10',
            'SP500_MOM20','SP500_RVOL','SP500_DRAWDOWN']
cross_f  = ['GOLD_RET1','GOLD_RET5','GOLD_RET20',
            'TNY_CHANGE1','TNY_CHANGE5','TNY_CHANGE20',
            'DXY_RET1','DXY_RET5','DXY_RET20']
macro_f  = ['FEDFUNDS_CHANGE1M','FEDFUNDS_CHANGE3M',
            'YIELD_CURVE_CHANGE1M','YIELD_CURVE_CHANGE3M']
for f in vix_tech: CATEGORY[f] = 'VIX technical'
for f in skew_f:   CATEGORY[f] = 'SKEW'
for f in sp500_f:  CATEGORY[f] = 'S&P 500'
for f in cross_f:  CATEGORY[f] = 'Cross-asset'
for f in macro_f:  CATEGORY[f] = 'Macro'

CAT_COLORS = {
    'VIX technical': '#e74c3c',
    'SKEW':          '#9b59b6',
    'S&P 500':       '#3498db',
    'Cross-asset':   '#2ecc71',
    'Macro':         '#f39c12',
}

shap_dfs = {}

for n in [5, 10]:
    label_col = f'LABEL_{n}'
    sub = df[feature_cols + [label_col]].dropna(subset=[label_col])
    split_idx = int(len(sub) * 0.8)

    X_tr_raw = sub[feature_cols].iloc[:split_idx].values
    y_tr     = sub[label_col].iloc[:split_idx].astype(int).values
    X_te_raw = sub[feature_cols].iloc[split_idx:].values

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr_raw)
    X_te   = scaler.transform(X_te_raw)

    print(f"\n  Fitting RF for N={n}...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                 min_samples_leaf=5, random_state=42)
    rf.fit(X_tr, y_tr)

    print(f"  Computing SHAP values (TreeExplainer)...")
    explainer   = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_te)

    # shap_values shape: (n_samples, n_features) or (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]   # class-1 slice
    else:
        sv = shap_values

    mean_abs = np.abs(sv).mean(axis=0)
    shap_df  = pd.DataFrame({'Feature': feature_cols,
                              'MeanAbsSHAP': mean_abs,
                              'Category': [CATEGORY.get(f, 'Other') for f in feature_cols]})
    shap_df  = shap_df.sort_values('MeanAbsSHAP', ascending=False).reset_index(drop=True)
    shap_df['Rank'] = shap_df.index + 1
    shap_df['N']    = n
    shap_dfs[n]     = (shap_df, sv)

    print(f"  Top-10 features (N={n}):")
    for _, row in shap_df.head(10).iterrows():
        print(f"    {int(row['Rank']):>2}. {row['Feature']:<25} {row['MeanAbsSHAP']:.4f}  [{row['Category']}]")

# Save CSV
all_shap = pd.concat([shap_dfs[n][0] for n in [5, 10]], ignore_index=True)
all_shap.to_csv('results/shap_mean_abs.csv', index=False)
print("\nSaved → results/shap_mean_abs.csv")

# ── Figure ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle('SHAP Feature Importance: What the RF Actually Learns',
             fontsize=13, fontweight='bold')

for ax, n in zip(axes, [5, 10]):
    shap_df = shap_dfs[n][0].head(15)
    colors  = [CAT_COLORS.get(c, '#95a5a6') for c in shap_df['Category']]

    bars = ax.barh(range(len(shap_df)), shap_df['MeanAbsSHAP'],
                   color=colors, alpha=0.85, edgecolor='white')
    ax.set_yticks(range(len(shap_df)))
    ax.set_yticklabels(shap_df['Feature'], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title(f'N = {n}  (top 15 of 36 features)')
    ax.grid(True, alpha=0.2, axis='x')

    # Category legend
    legend_patches = [mpatches.Patch(color=v, label=k)
                      for k, v in CAT_COLORS.items()]
    # anchored above the summary badge below, so the two never overlap
    ax.legend(handles=legend_patches, loc='lower right',
              bbox_to_anchor=(1.0, 0.14), fontsize=8, framealpha=0.9)

    # Fraction of total SHAP mass in VIX technical features
    total_shap = shap_dfs[n][0]['MeanAbsSHAP'].sum()
    vix_shap   = shap_dfs[n][0][shap_dfs[n][0]['Category'] == 'VIX technical']['MeanAbsSHAP'].sum()
    ax.text(0.98, 0.02,
            f'VIX technical features:\n{vix_shap/total_shap*100:.0f}% of total SHAP mass',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='#ffe0e0', alpha=0.9))

plt.tight_layout()
plt.savefig('vix paper/shap_analysis.png', bbox_inches='tight')
plt.close()
print("Saved → vix paper/shap_analysis.png")

# Print category breakdown
print("\n=== SHAP mass by feature category ===")
for n in [5, 10]:
    shap_df    = shap_dfs[n][0]
    total_shap = shap_df['MeanAbsSHAP'].sum()
    cat_sum    = shap_df.groupby('Category')['MeanAbsSHAP'].sum().sort_values(ascending=False)
    print(f"\n  N={n}:")
    for cat, val in cat_sum.items():
        print(f"    {cat:<20} {val:.4f}  ({val/total_shap*100:.1f}%)")
